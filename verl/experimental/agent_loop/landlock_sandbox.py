"""
Stateful Python sandbox backed by an IPython kernel.

Variables, imports, and other state persist across execute() calls within
the same sandbox instance, enabling multi-turn code execution.

Isolation strategy (layered):
  1. bubblewrap (bwrap), when available -- the PRIMARY fs layer. The kernel runs
     in a mount namespace with DEFAULT-DENY reads: only an allowlist of system
     dirs plus the rollout's own paths (workspace, env_dir, tmpdir) are visible;
     /tmp is a fresh tmpfs so sibling rollouts' env_dirs (and their data.db /
     answers), /mnt model weights, /root, etc. are invisible. Writes stay
     confined to the workspace + private /tmp & /dev/shm + tmpdir. Network is
     NOT isolated. Disable with VERL_SANDBOX_USE_BWRAP=0.
  2. Linux Landlock (kernel >= 5.13, no root required) -- FALLBACK when bwrap is
     unavailable. Applied in the kernel child via preexec_fn. Grants
     read+execute everywhere, but confines ALL writes (including subprocess
     calls made by user code) to the workspace and a private tmpdir.
  3. cgroup v2 leaf per kernel (when writable) -- the memory cap: aggregate
     memory.max + pids.max over the whole {bwrap -> kernel -> children} subtree.
     A per-process RLIMIT_AS in preexec is only a fallback for when the cgroup is
     unavailable (never stacked: AS over-counts mmap and false-positives).
  4. Python-level open() guard -- defense-in-depth for write-mode built-in
     open() calls outside the workspace.
"""

import asyncio
import concurrent.futures
import contextlib
import ctypes
import gc
import logging
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Keys already logged by _log_once, so repeated identical messages (e.g. the
# per-sandbox confinement mode, which is the same for every kernel in a worker)
# are emitted only ONCE per process instead of spamming the log every start.
_LOGGED_ONCE: set[str] = set()
_LOG_ONCE_LOCK = threading.Lock()


def _log_once(key: str, level: int, msg: str) -> None:
    """Emit `msg` at `level` the first time `key` is seen in this process."""
    with _LOG_ONCE_LOCK:
        if key in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(key)
    logger.log(level, msg)


# Default per-kernel memory cap (bytes). Seeds the cgroup memory.max default
# (the primary cap, see cgroup_mem_max below) and the fallback RLIMIT_AS used
# only when the cgroup is unavailable. Override via the `mem_limit_bytes` kwarg
# to StatefulSandbox or the VERL_SANDBOX_MEM_LIMIT_BYTES env var. 0 disables it.
# 4GiB is generous for the lightweight tasksync tool sandboxes.
_DEFAULT_MEM_LIMIT_BYTES = int(os.getenv("VERL_SANDBOX_MEM_LIMIT_BYTES", str(4 * 1024**3)))

# --- bubblewrap (bwrap) filesystem confinement -----------------------------
# When bwrap is on the PATH it becomes the PRIMARY fs-confinement layer. Reads
# are default-DENY: only an allowlist of system dirs (read-only) plus this
# rollout's own paths are visible. Everything else is invisible -- /mnt model
# weights & checkpoints, /root, and crucially OTHER rollouts' env_dirs (each a
# tempfile.mkdtemp under /tmp holding that task's data.db / answers): /tmp is
# remounted as a fresh tmpfs and only the caller's workspace / env_dir / tmpdir
# are bound back over it. Writes stay confined to the workspace + a private
# /tmp & /dev/shm + the kernel tmpdir (the same set Landlock enforced). Network
# is intentionally NOT isolated (kernel<->client ZMQ uses 127.0.0.1 and rollouts
# may need outbound access).
#
# Set VERL_SANDBOX_USE_BWRAP=0 to fall back to the Landlock path. bwrap and
# Landlock are not stacked: bwrap's fresh /tmp & /dev/shm tmpfs aren't in a
# host-built Landlock allowlist, so a Landlock ruleset would wrongly block them.
_BWRAP_BIN = shutil.which("bwrap")
_USE_BWRAP = os.getenv("VERL_SANDBOX_USE_BWRAP", "1") != "0" and _BWRAP_BIN is not None
# Read-only system dirs the kernel needs to run Python + user code (imports,
# shared libs, certs, system tools). Colon-separated; missing paths are skipped
# (--ro-bind-try). Set to "/" to expose the whole root read-only (old behaviour).
_BWRAP_READ_PATHS = [
    p
    for p in os.getenv(
        "VERL_SANDBOX_BWRAP_READ_PATHS",
        "/usr:/lib:/lib64:/bin:/sbin:/etc:/opt:/run:/sys",
    ).split(":")
    if p
]

# --- cgroup v2 aggregate memory / pid caps ---------------------------------
# The cgroup-v2 leaf per kernel is the PRIMARY memory cap: it bounds real
# aggregate RSS over the whole {bwrap -> kernel -> children} subtree (memory.max)
# plus a pids cap against fork bombs (pids.max). Requires a writable cgroup-v2
# root (we run as root on jd). RLIMIT_AS (in _preexec) is only a FALLBACK, used
# when the cgroup cap isn't in effect (disabled, or leaf creation/join failed);
# it is never stacked on the cgroup, because AS counts virtual address space and
# over-counts mmap-heavy code (importing torch alone reserves tens of GB of VA),
# which would false-positive into MemoryError below the real RSS limit.
_CGROUP_ENABLE = os.getenv("VERL_SANDBOX_CGROUP_ENABLE", "1") != "0"
_CGROUP_ROOT = os.getenv("VERL_SANDBOX_CGROUP_ROOT", "/sys/fs/cgroup/verl_sandbox")
_CGROUP_PIDS_MAX = os.getenv("VERL_SANDBOX_CGROUP_PIDS_MAX", "256")
_CGROUP_BASE_READY = False
_CGROUP_BASE_LOCK = threading.Lock()

_KERNEL_START_LOCK = threading.Lock()

# Track all spawned kernel PIDs so we can force-kill stragglers between envs.
_SPAWNED_PIDS: set[int] = set()
_PID_LOCK = threading.Lock()

# prctl(PR_SET_PDEATHSIG, sig) -- kernel sends `sig` to this process when its
# parent dies. Linux-only; key=1 per <sys/prctl.h>. Preserved across execve()
# for the no-setuid case we use here.
_PR_SET_PDEATHSIG = 1


# Cap on concurrent sandbox bring-up (fork ipykernel + import env.py +
# Task_Env() construction). Each rollout brings up 1-2 sandboxes; under heavy
# concurrency this stage saturates CPU/IO/RAM on the node and makes setup_code
# run past the execute() budget -> StatefulSandbox.start() / setup time out ->
# the rollout is dropped to an empty trajectory.
#
# This gate is process-wide, i.e. PER AgentLoopWorker. Workers are round-robin
# scheduled one-per-node (see AgentLoopManager._init_agent_loop_workers), so the
# value is effectively the per-node concurrent-bring-up cap. With ptc_mode=ptc
# each admitted rollout starts 2 kernels, so the node sees up to 2x this many
# kernels initializing at once.
#
# Keep this small. The pre-0528 path ran bring-up synchronously on the event
# loop, which serialized it to ~1 rollout (2 kernels) at a time per node and had
# a <1% empty-trajectory rate. A value of 16 here meant ~32 kernels racing per
# node and pushed heavy envs past their setup budget (3%+ empty, up to ~15/16 on
# the heaviest tasks). 4 keeps enough overlap to hide per-bring-up latency
# without the contention. Tune up only if bring-up latency dominates and the
# node has spare CPU/RAM.
_DEFAULT_MAX_CONCURRENT_STARTUPS = int(
    os.getenv("VERL_SANDBOX_MAX_CONCURRENT_STARTUPS", "16")
)
_STARTUP_SEM: asyncio.Semaphore | None = None


def _get_startup_sem() -> asyncio.Semaphore:
    """Lazily build the worker-wide sandbox-startup semaphore.

    Bound to the running event loop on first call. Safe under single-threaded
    asyncio (AgentLoopWorker has one event loop), no extra locking needed.
    """
    global _STARTUP_SEM
    if _STARTUP_SEM is None:
        _STARTUP_SEM = asyncio.Semaphore(_DEFAULT_MAX_CONCURRENT_STARTUPS)
    return _STARTUP_SEM


@contextlib.asynccontextmanager
async def sandbox_startup_gate():
    """Async gate around the sandbox bring-up phase. See _get_startup_sem()."""
    sem = _get_startup_sem()
    async with sem:
        yield


# Cap on rollouts concurrently holding live episode resources (1-2 ipykernel
# sandboxes each, plus the env_dir/Task_Env the agent loop attaches).
# AgentLoopWorker.generate_sequences launches its whole chunk with
# asyncio.gather, so without this gate the steady-state kernel population on a
# node scales linearly with env.batch_size * env.group_size until the OOM
# killer takes out a Ray worker or the AgentLoopWorker exhausts file
# descriptors (~10 ZMQ fds per live rollout). The startup gate above only
# bounds concurrent *bring-up*; this one bounds the population.
#
# Process-wide, i.e. PER AgentLoopWorker, and shared by every agent loop in
# the worker -- the slot accounting below must be global so the idle sweep
# (kill_all_kernels when in-flight hits zero) cannot fire while another loop
_DEFAULT_MAX_CONCURRENT_ROLLOUTS = 64
_ROLLOUT_SEM: asyncio.Semaphore | None = None

# Episodes currently holding an admission slot. Maintained on the event-loop
# thread only (AgentLoopWorker has one loop), so plain int updates are safe.
_INFLIGHT_ROLLOUTS = 0


def _get_rollout_sem() -> asyncio.Semaphore:
    """Lazily build the worker-wide rollout admission semaphore."""
    global _ROLLOUT_SEM
    if _ROLLOUT_SEM is None:
        _ROLLOUT_SEM = asyncio.Semaphore(_DEFAULT_MAX_CONCURRENT_ROLLOUTS)
    return _ROLLOUT_SEM


async def acquire_rollout_slot() -> None:
    """Block until this episode may hold live sandboxes (see _ROLLOUT_SEM).

    Pair every successful acquire with exactly one release_rollout_slot() in a
    finally block that runs AFTER the episode's sandbox cleanup.
    """
    global _INFLIGHT_ROLLOUTS
    await _get_rollout_sem().acquire()
    _INFLIGHT_ROLLOUTS += 1


def release_rollout_slot() -> None:
    """Release an admission slot; sweep orphan kernels when the worker idles.

    Synchronous on purpose: it must run to completion even while the caller is
    unwinding a CancelledError (awaits in a cancelled task's finally are
    aborted, plain calls are not). When the in-flight count hits zero no
    episode is active, so any PID still tracked in _SPAWNED_PIDS belongs to a
    sandbox whose per-rollout cleanup failed or was cancelled mid-await; those
    orphans used to accumulate monotonically across training steps until the
    node OOMed. kill_all_kernels() is a no-op when nothing leaked.
    """
    global _INFLIGHT_ROLLOUTS
    _INFLIGHT_ROLLOUTS -= 1
    _get_rollout_sem().release()
    if _INFLIGHT_ROLLOUTS == 0:
        kill_all_kernels()
        # Same invariant for env_dirs: with no episode active, anything still
        # tracked is an orphan (episode cancelled before its cleanup ran, or a
        # removal that partially failed). rmtree on the tool executor so the
        # event loop is not blocked.
        with _ENV_DIR_LOCK:
            orphan_dirs = list(_TRACKED_ENV_DIRS)
        for d in orphan_dirs:
            _TOOL_EXECUTOR.submit(cleanup_env_dir, d)
        # Reclaim reference cycles now that no episode is active. Each rollout's
        # Task_Env is a self-referential cycle (Task_Env.tools holds bound
        # methods of the Task_Env, whose __globals__ pin the whole exec'd env
        # module namespace) so it -- and any large module-level data env.py
        # loaded -- is freed only by the cyclic collector, never by refcounting.
        # teardown_env() breaks the cycle per rollout, but a cancellation can
        # skip that; collecting at idle (once per step) keeps the cyclic-garbage
        # backlog from growing without bound across steps until the node OOMs.
        gc.collect()


# Env dirs of in-flight episodes. Tracked from mkdtemp onward so that a
# cancellation anywhere in the episode cannot leak the directory -- including
# while data.py is still populating it on an executor thread, where the
# rollout coroutine drops the return value and never learns the path. Anything
# still tracked when the worker goes idle is removed by the sweep above.
_TRACKED_ENV_DIRS: set[str] = set()
_ENV_DIR_LOCK = threading.Lock()


def track_env_dir(env_dir: str) -> None:
    """Register an episode env_dir for orphan sweeping. Thread-safe."""
    with _ENV_DIR_LOCK:
        _TRACKED_ENV_DIRS.add(env_dir)


def cleanup_env_dir(env_dir: str) -> None:
    """Blocking rmtree + untrack; safe to call from any thread.

    Untracks only once the directory is actually gone, so a partial removal
    (e.g. a dying kernel still writing into its workspace) is retried by the
    next idle sweep instead of silently leaking.
    """
    shutil.rmtree(env_dir, ignore_errors=True)
    if not os.path.exists(env_dir):
        with _ENV_DIR_LOCK:
            _TRACKED_ENV_DIRS.discard(env_dir)


def teardown_env(env) -> None:
    """Release a finished episode's Task_Env so its memory is reclaimed now.

    The agent loops construct a fresh user ``Task_Env`` per rollout (tens of
    thousands per run). That object is a reference cycle -- ``task_env.tools``
    maps each tool name to a *bound method* of ``task_env``, and every bound
    method keeps ``task_env`` alive (``__self__``) *and* pins the entire exec'd
    env-module namespace (``__func__.__globals__``), which may include large
    objects env.py loaded at import time (DataFrames, parsed datasets, ...).
    Because of the cycle none of this is freed by refcounting when the rollout
    returns; it survives until the cyclic collector happens to run. Under
    per-rollout churn plus the node's Megatron CPU-offload pressure that
    backlog grows until the OOM killer takes out a worker.

    Breaking the cycle here (clearing ``tools`` and dropping ``env``'s handles)
    makes the Task_Env, its class, and the module namespace collectable by
    refcounting immediately. Best-effort and exception-safe: it runs on the
    episode's cleanup path, which must not raise. Idempotent.
    """
    if env is None:
        return
    task_env = getattr(env, "task_env", None)
    if task_env is not None:
        # Defensive: let envs that hold a persistent handle (sqlite connection,
        # open file, ...) release it. Most envs open/close per tool call and
        # expose no closer; that is fine.
        for closer in ("close", "cleanup"):
            fn = getattr(task_env, closer, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break
        tools = getattr(task_env, "tools", None)
        if isinstance(tools, dict):
            tools.clear()
    # Drop every back-reference the episode held so the chain collapses under
    # refcounting. setattr is guarded because the env-state dataclasses differ
    # slightly between agent loops (e.g. only the PTC loop omits `sandbox`).
    for attr in ("task_env", "sandbox", "ptc_sandbox", "terminal_executor"):
        if hasattr(env, attr):
            try:
                setattr(env, attr, None)
            except Exception:
                pass


def _set_pdeathsig(sig: int) -> None:
    """Best-effort: ask the kernel to deliver `sig` when our parent dies.

    Called from preexec_fn (already in the forked child). Failure is silent --
    we still have the explicit cleanup() path and kill_all_kernels() as
    fallbacks if the syscall is unavailable.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(
            ctypes.c_int(_PR_SET_PDEATHSIG),
            ctypes.c_ulong(sig),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
    except Exception:
        pass


# Dedicated, process-lifetime thread used ONLY to fork (Popen) sandbox kernels.
# PR_SET_PDEATHSIG (set in _preexec) delivers the death signal when the *thread*
# that called fork() exits -- NOT when the process exits. A run_in_executor()
# pool thread is transient and gets recycled, which would SIGKILL live kernels
# mid-rollout. Forking from this single, never-shutdown thread makes the death
# signal fire only when the worker process actually dies. Popen returns once the
# child is exec'd (~ms), so serializing just the fork costs nothing -- the slow
# conn-file wait / channel setup still run on the caller's thread.
_FORK_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="sandbox-fork"
)


# Worker-wide pool for the *steady-state* blocking tool calls (sandbox.execute /
# terminal.execute) that run on every agent turn. These are subprocess-backed
# (jupyter ZMQ / Popen) and were previously invoked synchronously inside the
# agent-loop coroutine, which froze the AgentLoopWorker's single event loop --
# and therefore all other rollouts' LLM generation -- for the full duration of
# each Python/terminal call. Offloading them here lets the event loop keep
# driving generation while tool code runs.
#
# max_workers is the per-node cap on concurrent tool executions. Correctness
# note: a given sandbox is only ever executed by one rollout, and that rollout
# awaits its tool calls in order, so no sandbox is touched by two threads at
# once (sequential cross-thread use of a jupyter client is safe; concurrent is
# not). This pool must NOT be used for in-process Task_Env calls (env.step /
# env tools) -- those hold thread-affine state (e.g. a sqlite3 connection bound
# to the event-loop thread) and stay on the event loop.
_DEFAULT_MAX_CONCURRENT_TOOL_EXEC = int(
    os.getenv("VERL_SANDBOX_MAX_CONCURRENT_EXEC", "32")
)
_TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_DEFAULT_MAX_CONCURRENT_TOOL_EXEC, thread_name_prefix="sandbox-tool"
)


def get_tool_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Worker-wide executor for blocking sandbox/terminal tool calls.

    See _TOOL_EXECUTOR. Bounded by VERL_SANDBOX_MAX_CONCURRENT_EXEC.
    """
    return _TOOL_EXECUTOR


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def _find_child_pid(ppid: int) -> int | None:
    """Return the single child PID of `ppid`, or None. Used to locate the kernel
    when it runs under bwrap (bwrap is the immediate child / process-group
    leader; the kernel is bwrap's child). Tries the cheap children file, then
    falls back to scanning /proc."""
    try:
        with open(f"/proc/{ppid}/task/{ppid}/children") as f:
            kids = f.read().split()
        if kids:
            return int(kids[0])
    except (OSError, ValueError):
        pass
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/stat") as f:
                data = f.read()
            # Fields after the (possibly space/paren-containing) comm: state ppid
            after = data[data.rfind(")") + 2:].split()
            if int(after[1]) == ppid:
                return int(e)
        except (OSError, IndexError, ValueError):
            continue
    return None


def _force_kill_pid(pid: int):
    """Best-effort kill a process group, then the PID itself."""
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def kill_all_kernels():
    """Force-kill every kernel process we ever spawned. Call between envs."""
    with _PID_LOCK:
        pids = list(_SPAWNED_PIDS)
        _SPAWNED_PIDS.clear()
    for pid in pids:
        _force_kill_pid(pid)
    # Reap zombies
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
    # Reclaim cgroup leaves whose kernels just died (incl. cancelled rollouts
    # whose StatefulSandbox.cleanup() never ran). Done after the reap so the
    # leaves are empty and removable.
    _sweep_cgroup_leaves()


# -- Landlock isolation --------------------------------------------------------

def _landlock_abi_version() -> int:
    """Return the kernel's Landlock ABI version, or 0 if unavailable."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # SYS_landlock_create_ruleset = 444 (x86_64 and aarch64)
        LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
        ver = libc.syscall(
            ctypes.c_long(444),   # SYS_landlock_create_ruleset
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
        )
        return max(int(ver), 0)
    except Exception:
        return 0


def _apply_landlock(writable_dirs: list[str]) -> bool:
    """Apply Landlock FS restrictions to the calling process.

    Called inside the child process (preexec_fn) before exec.
    After this call the kernel process and any subprocess it spawns will:
      - be allowed to read / execute anywhere on the filesystem
      - be allowed to write ONLY inside the directories in *writable_dirs*

    Returns True on success, False if Landlock is unsupported or failed
    (the caller should then fall back gracefully).

    No special privileges are required (works without root / sudo).
    Requires Linux kernel >= 5.13 with CONFIG_SECURITY_LANDLOCK=y.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)

        # Syscall numbers -- identical on x86_64 and aarch64
        SYS_landlock_create_ruleset = 444
        SYS_landlock_add_rule       = 445
        SYS_landlock_restrict_self  = 446
        LANDLOCK_RULE_PATH_BENEATH  = 1

        # FS access-right bit positions (cumulative across Landlock ABI versions)
        _BIT = {
            "EXECUTE":     1 << 0,   # v1
            "WRITE_FILE":  1 << 1,   # v1
            "READ_FILE":   1 << 2,   # v1
            "READ_DIR":    1 << 3,   # v1
            "REMOVE_DIR":  1 << 4,   # v1
            "REMOVE_FILE": 1 << 5,   # v1
            "MAKE_CHAR":   1 << 6,   # v1
            "MAKE_DIR":    1 << 7,   # v1
            "MAKE_REG":    1 << 8,   # v1
            "MAKE_SOCK":   1 << 9,   # v1
            "MAKE_FIFO":   1 << 10,  # v1
            "MAKE_BLOCK":  1 << 11,  # v1
            "MAKE_SYM":    1 << 12,  # v1
            "REFER":       1 << 13,  # v2 (kernel >= 5.19)
            "TRUNCATE":    1 << 14,  # v3 (kernel >= 6.2)
            "IOCTL_DEV":   1 << 15,  # v4 (kernel >= 6.7)
        }

        # Query ABI version
        LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
        abi = libc.syscall(
            ctypes.c_long(SYS_landlock_create_ruleset),
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
        )
        if abi < 1:
            return False

        # Build flag sets according to the ABI version the kernel supports
        READ_FLAGS = _BIT["EXECUTE"] | _BIT["READ_FILE"] | _BIT["READ_DIR"]
        WRITE_FLAGS = (
            _BIT["WRITE_FILE"]  | _BIT["REMOVE_FILE"] | _BIT["REMOVE_DIR"] |
            _BIT["MAKE_CHAR"]   | _BIT["MAKE_DIR"]    | _BIT["MAKE_REG"]   |
            _BIT["MAKE_SOCK"]   | _BIT["MAKE_FIFO"]   | _BIT["MAKE_BLOCK"] |
            _BIT["MAKE_SYM"]
        )
        if abi >= 2:
            WRITE_FLAGS |= _BIT["REFER"]
        if abi >= 3:
            WRITE_FLAGS |= _BIT["TRUNCATE"]
        if abi >= 4:
            WRITE_FLAGS |= _BIT["IOCTL_DEV"]

        ALL_FLAGS = READ_FLAGS | WRITE_FLAGS

        # Struct layout used by landlock_create_ruleset (v1 layout: 8 bytes)
        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64)]

        # Struct layout used by landlock_add_rule with LANDLOCK_RULE_PATH_BENEATH
        class PathBeneathAttr(ctypes.Structure):
            _fields_ = [
                ("allowed_access", ctypes.c_uint64),
                ("parent_fd",      ctypes.c_int32),
                ("_pad",           ctypes.c_uint32),
            ]

        # Create ruleset -- we declare that we handle all known FS rights
        rs_attr = RulesetAttr(handled_access_fs=ALL_FLAGS)
        rfd = libc.syscall(
            ctypes.c_long(SYS_landlock_create_ruleset),
            ctypes.byref(rs_attr),
            ctypes.c_size_t(ctypes.sizeof(rs_attr)),
            ctypes.c_uint32(0),
        )
        if rfd < 0:
            return False

        try:
            # Rule 1: read + execute on the entire filesystem "/"
            root_fd = os.open("/", os.O_RDONLY | os.O_CLOEXEC)
            try:
                rule = PathBeneathAttr(allowed_access=READ_FLAGS, parent_fd=root_fd)
                ret = libc.syscall(
                    ctypes.c_long(SYS_landlock_add_rule),
                    ctypes.c_int(rfd),
                    ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint32(0),
                )
                if ret < 0:
                    return False
            finally:
                os.close(root_fd)

            # Rule 2: full access (read + write) on each explicitly allowed directory
            for d in writable_dirs:
                os.makedirs(d, exist_ok=True)
                dfd = os.open(d, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    rule = PathBeneathAttr(allowed_access=ALL_FLAGS, parent_fd=dfd)
                    ret = libc.syscall(
                        ctypes.c_long(SYS_landlock_add_rule),
                        ctypes.c_int(rfd),
                        ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                        ctypes.byref(rule),
                        ctypes.c_uint32(0),
                    )
                    if ret < 0:
                        return False
                finally:
                    os.close(dfd)

            # no_new_privs must be set before landlock_restrict_self
            PR_SET_NO_NEW_PRIVS = 38
            libc.prctl(
                ctypes.c_int(PR_SET_NO_NEW_PRIVS),
                ctypes.c_ulong(1),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            )

            # Apply the ruleset to the current process (and all future children)
            ret = libc.syscall(
                ctypes.c_long(SYS_landlock_restrict_self),
                ctypes.c_int(rfd),
                ctypes.c_uint32(0),
            )
            return ret == 0

        finally:
            os.close(rfd)

    except Exception:
        return False


# -- Sandbox class -------------------------------------------------------------

def _ensure_cgroup_base() -> bool:
    """Create the parent cgroup once and enable the memory+pids controllers in
    its subtree so per-kernel leaves can set memory.max / pids.max. Best effort;
    returns True if the base looks usable. Safe to call concurrently."""
    global _CGROUP_BASE_READY
    if _CGROUP_BASE_READY:
        return True
    with _CGROUP_BASE_LOCK:
        if _CGROUP_BASE_READY:
            return True
        try:
            os.makedirs(_CGROUP_ROOT, exist_ok=True)
            # Delegate memory+pids to children. May already be on, or the real
            # root may not have delegated them -- in that case the leaf write
            # below fails and we fall back to RLIMIT_AS only.
            try:
                with open(os.path.join(_CGROUP_ROOT, "cgroup.subtree_control"), "w") as f:
                    f.write("+memory +pids")
            except OSError:
                pass
            _CGROUP_BASE_READY = True
            return True
        except OSError as e:
            _log_once(
                "cgroup_base_unusable",
                logging.WARNING,
                f"[sandbox] cgroup base {_CGROUP_ROOT} unusable ({e}); "
                f"falling back to RLIMIT_AS only (logged once).",
            )
            return False


def _create_cgroup(name: str, mem_max: int, pids_max: str) -> str | None:
    """Create a per-kernel cgroup-v2 leaf with the given memory / pids caps.
    Returns the leaf path, or None on any failure (caller keeps RLIMIT_AS)."""
    if not _ensure_cgroup_base():
        return None
    leaf = os.path.join(_CGROUP_ROOT, name)
    try:
        os.makedirs(leaf, exist_ok=True)
        if mem_max and mem_max > 0:
            with open(os.path.join(leaf, "memory.max"), "w") as f:
                f.write(str(mem_max))
            # Forbid swap so memory.max is a hard ceiling, not soft.
            try:
                with open(os.path.join(leaf, "memory.swap.max"), "w") as f:
                    f.write("0")
            except OSError:
                pass
        if pids_max:
            try:
                with open(os.path.join(leaf, "pids.max"), "w") as f:
                    f.write(str(pids_max))
            except OSError:
                pass
        return leaf
    except OSError as e:
        _log_once(
            "cgroup_create_failed",
            logging.WARNING,
            f"[sandbox] could not create cgroup leaf ({e}); "
            f"falling back to RLIMIT_AS only (logged once).",
        )
        try:
            os.rmdir(leaf)
        except OSError:
            pass
        return None


def _sweep_cgroup_leaves() -> None:
    """Best-effort rmdir of empty cgroup leaves under _CGROUP_ROOT. A leaf is
    only removable once empty (all its procs are gone), so this reclaims the
    dirs of kernels that died without their StatefulSandbox.cleanup() running
    (e.g. a cancelled rollout). Leaves still holding live procs fail rmdir and
    are skipped."""
    if not _CGROUP_ENABLE:
        return
    try:
        names = os.listdir(_CGROUP_ROOT)
    except OSError:
        return
    for name in names:
        leaf = os.path.join(_CGROUP_ROOT, name)
        try:
            os.rmdir(leaf)  # fails (and is skipped) if non-empty or not a dir
        except OSError:
            pass


def _build_bwrap_argv(
    workspace: str,
    read_paths_extra: "list[str] | tuple" = (),
    write_paths_extra: "list[str] | tuple" = (),
) -> list[str]:
    """bwrap args implementing default-deny reads (see _BWRAP_READ_PATHS).

    Read-only: the system allowlist + each path in read_paths_extra (e.g. the
    rollout's env_dir, so the PTC kernel can import env.py / open data.db).
    Writable: workspace + each path in write_paths_extra (e.g. the kernel's
    connection-file tmpdir) + a private /tmp & /dev/shm. /tmp is a fresh tmpfs
    so sibling rollouts' env_dirs are hidden; the caller's own paths are bound
    back AFTER it. workspace is bound after the read-only extras so it overlays
    read-write even when it sits inside one (workspace == env_dir/workspace). No
    --unshare-* so the process keeps loopback (kernel ZMQ) and stays in our
    process group (killpg)."""
    argv = [_BWRAP_BIN]
    for p in _BWRAP_READ_PATHS:
        argv += ["--ro-bind-try", p, p]
    argv += [
        "--proc", "/proc",                # fresh /proc for the namespace
        "--dev", "/dev",                  # minimal writable device nodes
        "--tmpfs", "/tmp",                # private writable /tmp (hides siblings)
        "--tmpfs", "/dev/shm",            # private writable shm
    ]
    for p in read_paths_extra:            # e.g. env_dir: env.py / data.db, read-only
        argv += ["--ro-bind-try", p, p]
    argv += ["--bind", workspace, workspace]   # workspace: writable (overlays env_dir)
    for p in write_paths_extra:           # e.g. kernel.json dir: writable
        argv += ["--bind", p, p]
    argv += [
        "--setenv", "HOME", workspace,    # libs write ~/.cache into the workspace
        "--chdir", workspace,
        "--die-with-parent",
        "--",
    ]
    return argv


class StatefulSandbox:
    """A stateful Python sandbox backed by an IPython kernel.

    State (variables, imports, defined functions, etc.) persists across
    execute() calls. Use as a context manager for automatic cleanup.

    Isolation
    ---------
    The IPython kernel subprocess is started with a preexec_fn that applies
    Linux Landlock before exec().  This restricts the kernel process -- and any
    subprocess it spawns -- to:
      - read + execute access everywhere (needed for Python imports, /proc, ...)
      - write access only inside  workspace_path  and a private tmpdir

    If Landlock is unavailable a Python-level open() guard is installed inside
    the kernel as a fallback (only write-mode opens outside the workspace are
    blocked; reads are unrestricted).

    Example:
        with StatefulSandbox(workspace_path="/tmp/ws") as sb:
            sb.execute("x = 42")
            print(sb.execute("print(x)"))  # prints 42
    """

    def __init__(
        self,
        workspace_path: str,
        timeout: float = 10.0,
        mem_limit_bytes: int | None = None,
        interrupt_grace_seconds: float = 2.0,
        extra_read_paths: list[str] | None = None,
    ):
        self.workspace_path = workspace_path
        # Extra dirs to expose read-only under bwrap (besides the system
        # allowlist), e.g. the rollout's env_dir so the PTC kernel can import
        # env.py / open data.db. Leave empty for the main code-exec sandbox so
        # agent code cannot read the task's data.db / answers.
        self.extra_read_paths = list(extra_read_paths or [])
        self.timeout = timeout
        self.mem_limit_bytes = (
            mem_limit_bytes if mem_limit_bytes is not None else _DEFAULT_MEM_LIMIT_BYTES
        )
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self._tmpdir: str | None = None
        self._proc: subprocess.Popen | None = None
        # PID to target for SIGINT interrupts. Equals self._proc.pid normally;
        # under bwrap it is bwrap's child (the real kernel) so an interrupt does
        # not also hit the bwrap process-group leader (which would die on SIGINT
        # and take the kernel down via --die-with-parent).
        self._kernel_pid: int | None = None
        self._client = None
        self._conn_file: Path | None = None
        self._cleaned_up = False
        self._dead = False
        self._landlock_active = False
        self._cgroup_dir: str | None = None
        # cgroup memory.max for this kernel's subtree; defaults to the same
        # value as the RLIMIT_AS cap. 0 disables the cgroup memory cap.
        self.cgroup_mem_max = int(
            os.getenv("VERL_SANDBOX_CGROUP_MEM_MAX", str(self.mem_limit_bytes or 0))
        )

    def start(self):
        """Start the IPython kernel and initialize workspace state."""
        self._cleaned_up = False
        try:
            self._start_impl()
        except Exception:
            # A concurrent cleanup() (rollout cancelled while start() was on
            # an executor thread) may have set _cleaned_up mid-start, before
            # our kernel existed -- it then had nothing to kill and will never
            # be called again. Force this cleanup() to actually run so the
            # kernel we may have just forked is reaped.
            self._cleaned_up = False
            self.cleanup()
            raise

    def _start_impl(self):
        try:
            from jupyter_client import BlockingKernelClient
        except ImportError:
            raise RuntimeError(
                "jupyter_client is required for StatefulSandbox. "
                "Install it with: pip install jupyter_client ipykernel"
            )

        self._tmpdir = tempfile.mkdtemp(prefix="sandbox_")
        self._conn_file = Path(self._tmpdir) / "kernel.json"

        # Directories the kernel process is allowed to write to.
        # workspace_path : user code output
        # self._tmpdir   : kernel connection JSON file
        # /tmp           : Python's own tempfile operations inside the kernel
        os.makedirs(self.workspace_path, exist_ok=True)  # bwrap --bind needs it to exist
        writable = [self.workspace_path, self._tmpdir, "/tmp"]

        abi = _landlock_abi_version()
        # bwrap is the primary fs-confinement layer when present; Landlock is the
        # fallback. Don't run both (see _USE_BWRAP note): bwrap's fresh /tmp and
        # /dev/shm tmpfs aren't in a host-built Landlock allowlist.
        use_bwrap = _USE_BWRAP
        landlock_ok = (abi >= 1) and not use_bwrap
        mem_limit = self.mem_limit_bytes

        # Per-kernel cgroup-v2 leaf: aggregate memory + pids ceiling over the
        # whole {bwrap -> kernel -> children} subtree. Created here; the child
        # joins it in _preexec by writing its own pid to cgroup.procs (membership
        # is inherited across fork/exec, so every descendant is capped too).
        if _CGROUP_ENABLE:
            self._cgroup_dir = _create_cgroup(
                os.path.basename(self._tmpdir),
                mem_max=self.cgroup_mem_max,
                pids_max=_CGROUP_PIDS_MAX,
            )
        cgroup_procs = (
            os.path.join(self._cgroup_dir, "cgroup.procs") if self._cgroup_dir else None
        )

        def _preexec():
            # Die with the parent. Prevents orphan ipykernels from surviving an
            # AgentLoopWorker crash / SIGKILL / OOM-kill and slowly leaking
            # memory on the node across retries.
            _set_pdeathsig(signal.SIGKILL)
            # cgroup is the memory cap: join it so memory.max / pids.max cover
            # this process and every descendant (bwrap, the kernel, anything
            # user code spawns) by real aggregate RSS.
            joined_cgroup = False
            if cgroup_procs is not None:
                try:
                    with open(cgroup_procs, "w") as f:
                        f.write(str(os.getpid()))
                    joined_cgroup = True
                except OSError:
                    joined_cgroup = False
            # RLIMIT_AS is ONLY a fallback for when the cgroup cap isn't in
            # effect (disabled, or leaf creation / join failed). It is never
            # stacked on top of the cgroup: AS counts virtual address space and
            # over-counts mmap-heavy code (importing torch alone reserves tens of
            # GB of VA), which would false-positive into MemoryError well below
            # the real RSS limit.
            if not joined_cgroup and mem_limit and mem_limit > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                except (ValueError, OSError):
                    pass
            if landlock_ok:
                _apply_landlock(writable)

        kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", str(self._conn_file)]
        if use_bwrap:
            # bwrap runs in the same process group (no --unshare-pid), so the
            # existing os.killpg(proc.pid, ...) paths still reach the kernel.
            kernel_cmd = (
                _build_bwrap_argv(
                    self.workspace_path,
                    read_paths_extra=self.extra_read_paths,
                    write_paths_extra=[self._tmpdir],
                )
                + kernel_cmd
            )

        # Fork on the dedicated lifetime thread (see _FORK_EXECUTOR) so the
        # kernel's PR_SET_PDEATHSIG is bound to a thread that lives as long as
        # the worker process, not a recyclable run_in_executor() pool thread.
        self._proc = _FORK_EXECUTOR.submit(
            subprocess.Popen,
            kernel_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.workspace_path,
            start_new_session=True,
            preexec_fn=_preexec,
        ).result()

        with _PID_LOCK:
            _SPAWNED_PIDS.add(self._proc.pid)

        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self._conn_file.exists():
                break
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"IPython kernel exited early with code {self._proc.returncode}"
                )
            time.sleep(0.05)
        else:
            raise RuntimeError("Timeout waiting for IPython kernel to start")

        client = BlockingKernelClient(connection_file=str(self._conn_file))
        client.load_connection_file()

        acquired = _KERNEL_START_LOCK.acquire(timeout=30)
        if not acquired:
            raise RuntimeError("Timed out waiting for kernel start lock")
        try:
            # Only channel/connection bring-up needs serializing (jupyter_client
            # connection-file load + ZMQ port setup is not concurrency-safe).
            client.start_channels()
        finally:
            _KERNEL_START_LOCK.release()
        # wait_for_ready() only polls the kernel's status channel and can block
        # up to its full timeout. Keeping it inside the lock serializes every
        # kernel start behind a ~10s critical section, which produces an
        # acquire() convoy ("Timed out waiting for kernel start lock") once a
        # few sandboxes start concurrently. Wait outside the lock instead.
        client.wait_for_ready(timeout=10.0)

        self._client = client
        self._landlock_active = landlock_ok

        # Resolve the kernel PID for interrupts. Under bwrap the kernel is
        # bwrap's child; fall back to bwrap's own pid if it can't be found (then
        # interrupts degrade to the killpg path, same as no-bwrap).
        self._kernel_pid = self._proc.pid
        if use_bwrap:
            child = _find_child_pid(self._proc.pid)
            if child is not None:
                self._kernel_pid = child

        if landlock_ok:
            # Landlock is active at the OS level; install a lightweight Python
            # guard only as defense-in-depth for write-mode open() calls.
            init_code = (
                f"import os, json, csv, sys\n"
                f"workspace = {repr(self.workspace_path)}\n"
                f"os.chdir({repr(self.workspace_path)})\n"
                f"_open_orig = open\n"
                f"_WRITE_MODES = frozenset('wax+')\n"
                f"def open(file, mode='r', *args, **kwargs):\n"
                f"    if _WRITE_MODES & set(str(mode)):\n"
                f"        _abs = os.path.realpath(str(file))\n"
                f"        _ws  = os.path.realpath({repr(self.workspace_path)})\n"
                f"        if not (_abs.startswith(_ws + os.sep) or _abs == _ws):\n"
                f"            raise PermissionError(\n"
                f"                f'Write denied: {{file!r}} is outside workspace. '\n"
                f"                f'(Landlock OS-level restriction also active)'\n"
                f"            )\n"
                f"    return _open_orig(file, mode, *args, **kwargs)\n"
            )
        else:
            # Landlock unavailable: rely entirely on the Python-level guard.
            init_code = (
                f"import os, json, csv, sys\n"
                f"workspace = {repr(self.workspace_path)}\n"
                f"os.chdir({repr(self.workspace_path)})\n"
                f"_open_orig = open\n"
                f"_WRITE_MODES = frozenset('wax+')\n"
                f"def open(file, mode='r', *args, **kwargs):\n"
                f"    if _WRITE_MODES & set(str(mode)):\n"
                f"        _abs = os.path.realpath(str(file))\n"
                f"        _ws  = os.path.realpath({repr(self.workspace_path)})\n"
                f"        if not (_abs.startswith(_ws + os.sep) or _abs == _ws):\n"
                f"            raise PermissionError(\n"
                f"                f'Write denied: {{file!r}} is outside workspace. '\n"
                f"                f'(Landlock unavailable; Python-level guard only)'\n"
                f"            )\n"
                f"    return _open_orig(file, mode, *args, **kwargs)\n"
            )

        self._run(init_code)

        if self._cleaned_up:
            # cleanup() raced this start (rollout cancelled while start() was
            # on an executor thread) and saw nothing to kill; it will never be
            # called again for this sandbox, so reap the kernel we just
            # started ourselves -- otherwise it survives until an idle sweep.
            self._kill_kernel_now()
            try:
                client.stop_channels()
                if client.context is not None:
                    client.context.destroy(linger=0)
            except Exception:
                pass
            raise RuntimeError("Sandbox cleanup() ran concurrently with start(); kernel reaped")

        # The confinement mode is identical for every sandbox in this worker, so
        # log it ONCE per process (keyed by mode) instead of on every start.
        cg = " + cgroup memory/pids cap" if self._cgroup_dir else ""
        if use_bwrap:
            _log_once(
                "confine:bwrap",
                logging.INFO,
                f"[sandbox] bwrap fs confinement active -- default-deny reads "
                f"(system allowlist + own paths), writes restricted to workspace{cg}.",
            )
        elif landlock_ok:
            _log_once(
                "confine:landlock",
                logging.INFO,
                f"[sandbox] Landlock ABI v{abi} active -- writes restricted to workspace{cg}.",
            )
        else:
            _log_once(
                "confine:none",
                logging.WARNING,
                f"[sandbox] neither bwrap nor Landlock active -- Python open() guard "
                f"only (weak isolation){cg}. Check bwrap is on PATH / "
                f"VERL_SANDBOX_USE_BWRAP, and kernel Landlock support.",
            )

    def _try_interrupt_to_idle(self, msg_id: str, grace_seconds: float) -> bool:
        """Send SIGINT to the kernel and wait up to grace_seconds for the
        in-flight execution (msg_id) to reach an idle status. Returns True if
        the kernel recovered, False otherwise. State (variables/imports) is
        preserved on a successful interrupt."""
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            if self._kernel_pid is not None and self._kernel_pid != self._proc.pid:
                # Under bwrap: SIGINT only the kernel, NOT the process group --
                # the group leader is bwrap, which would die on SIGINT and take
                # the kernel with it (--die-with-parent), turning a state-
                # preserving interrupt into a hard kill.
                os.kill(self._kernel_pid, signal.SIGINT)
            else:
                os.killpg(self._proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        deadline = time.time() + grace_seconds
        client = self._client
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            try:
                msg = client.get_iopub_msg(timeout=remaining)
            except Exception:
                return False
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})
            if msg_type == "status" and content.get("execution_state") == "idle":
                return True

    def _kill_kernel_now(self):
        """Hard-kill the kernel process group and mark sandbox dead."""
        self._dead = True
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is not None:
            # The child is reaped, so the OS may recycle its PID. Stop
            # tracking it now: if this sandbox's cleanup() never runs (e.g.
            # the rollout task is cancelled), a later kill_all_kernels()
            # sweep would otherwise SIGKILL whatever unrelated process
            # inherited the recycled PID.
            with _PID_LOCK:
                _SPAWNED_PIDS.discard(proc.pid)

    def _run(self, code: str, timeout: float | None = None) -> tuple[str, str, bool]:
        """Execute code in the kernel and return (stdout, stderr, success)."""
        if self._dead:
            raise RuntimeError("Kernel was killed after a previous timeout")
        if self._proc is not None and self._proc.poll() is not None:
            self._dead = True
            # poll() reaped the child, so the OS may recycle its PID. Stop
            # tracking it now (same hazard as in _kill_kernel_now): if this
            # sandbox's cleanup() never runs, a later kill_all_kernels()
            # sweep must not SIGKILL whatever inherited the recycled PID.
            with _PID_LOCK:
                _SPAWNED_PIDS.discard(self._proc.pid)
            raise RuntimeError(
                f"Kernel process has exited unexpectedly (code {self._proc.returncode})"
            )
        client = self._client
        msg_id = client.execute(code, silent=False, store_history=True)
        effective_timeout = self.timeout if timeout is None else timeout
        deadline = time.time() + effective_timeout

        out_parts: list[str] = []
        err_parts: list[str] = []
        success = True

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                # Try to interrupt cleanly so kernel state survives; otherwise
                # SIGKILL the kernel so subsequent execute() calls don't block
                # for another full timeout each.
                recovered = self._try_interrupt_to_idle(msg_id, self.interrupt_grace_seconds)
                if not recovered:
                    self._kill_kernel_now()
                raise TimeoutError("Kernel execution timed out")
            try:
                msg = client.get_iopub_msg(timeout=remaining)
            except Exception:
                recovered = self._try_interrupt_to_idle(msg_id, self.interrupt_grace_seconds)
                if not recovered:
                    self._kill_kernel_now()
                raise TimeoutError("Kernel execution timed out")

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})

            if msg_type == "stream":
                target = out_parts if content.get("name") == "stdout" else err_parts
                target.append(content.get("text", ""))
            elif msg_type in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "text/plain" in data:
                    out_parts.append(f"{data['text/plain']}\n")
            elif msg_type == "error":
                success = False
                tb = content.get("traceback") or []
                if tb:
                    err_parts.append("\n".join(tb) + "\n")
                else:
                    err_parts.append(
                        f"{content.get('ename', 'Error')}: {content.get('evalue', '')}\n"
                    )
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        return strip_ansi("".join(out_parts)), strip_ansi("".join(err_parts)), success

    def execute(self, code: str, timeout: float | None = None) -> tuple[str, bool]:
        """Execute code and return (output, success).

        State from previous calls is preserved. Returns an error string on
        failure rather than raising. ``success`` is False whenever the kernel
        reported an exception, the sandbox was unavailable, or execution timed
        out -- i.e. any case where ``output`` carries an error message.

        ``timeout`` overrides the sandbox-level default for this single call.
        """
        if not isinstance(code, str):
            return (
                f"[Error] The 'code' argument must be a string, got "
                f"{type(code).__name__}: {code!r}.",
                False,
            )
        if self._client is None:
            return "[Error] Sandbox not started. Call start() first.", False
        if self._dead:
            return (
                "[Error] Sandbox is no longer available "
                "(killed after a previous timeout / crash). Subsequent code will not run."
            ), False
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            output, error, success = self._run(code, timeout=effective_timeout)
            result = output
            if error.strip():
                result += ("\n[stderr]:\n" if result else "") + error.strip()
            return (result.strip() if result.strip() else "(no output)"), success
        except TimeoutError:
            suffix = (
                " Sandbox killed; subsequent code will not run."
                if self._dead
                else " Sandbox interrupted; state preserved."
            )
            return (
                f"[Error] Code execution timed out ({effective_timeout}-second limit).{suffix}",
                False,
            )
        except Exception as e:
            return f"[Error] {e}", False

    def cleanup(self):
        """Stop the kernel and remove temporary files."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._dead = True

        proc = self._proc
        self._proc = None
        if proc is not None:
            pid = proc.pid
            if proc.poll() is None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
            with _PID_LOCK:
                _SPAWNED_PIDS.discard(pid)

        client = self._client
        self._client = None
        if client is not None:
            try:
                client.stop_channels()
            except Exception:
                pass
            # Force-destroy the underlying ZMQ context to release ports immediately
            try:
                if hasattr(client, 'context') and client.context is not None:
                    client.context.destroy(linger=0)
            except Exception:
                pass

        tmpdir = self._tmpdir
        self._tmpdir = None
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # rmdir the cgroup leaf -- only succeeds once empty (all procs reaped).
        # Retry briefly in case the kernel hasn't fully exited yet.
        cgroup_dir = self._cgroup_dir
        self._cgroup_dir = None
        if cgroup_dir is not None:
            for _ in range(20):
                try:
                    os.rmdir(cgroup_dir)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.05)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
