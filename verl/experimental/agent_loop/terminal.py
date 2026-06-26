"""Terminal executor: runs shell commands inside the workspace directory."""

import itertools
import os
import resource
import signal
import subprocess

from verl.experimental.agent_loop.landlock_sandbox import (
    _CGROUP_ENABLE,
    _CGROUP_PIDS_MAX,
    _USE_BWRAP,
    _build_bwrap_argv,
    _create_cgroup,
)

# Default per-command memory cap (bytes). Seeds the cgroup memory.max (the
# primary cap, applied over the whole {bash -> children} tree by real RSS) and
# the fallback RLIMIT_AS used only when the cgroup is unavailable. RLIMIT_AS is
# inherited by every child the shell spawns, so a runaway `python3 foo.py` is
# bounded either way. Override via the `mem_limit_bytes` kwarg or the
# VERL_TERMINAL_MEM_LIMIT_BYTES env var. 0 disables the cap.
_DEFAULT_MEM_LIMIT_BYTES = int(os.getenv("VERL_TERMINAL_MEM_LIMIT_BYTES", str(4 * 1024**3)))

# Per-command cgroup leaf names: term_<pid>_<n>, unique within a worker process.
_LEAF_COUNTER = itertools.count()


class TerminalError(Exception):
    pass


class TerminalExecutor:
    """Executes shell commands in the workspace via /bin/bash. No allowlist; pipes / redirects / chaining all permitted."""

    def __init__(self, workspace: str, timeout: int = 60, mem_limit_bytes: int | None = None):
        self.workspace = os.path.abspath(os.path.realpath(workspace))
        self.timeout = timeout
        self.mem_limit_bytes = mem_limit_bytes if mem_limit_bytes is not None else _DEFAULT_MEM_LIMIT_BYTES

    def execute(self, command_string: str) -> str:
        if not isinstance(command_string, str):
            raise TerminalError(
                f"The 'command' argument must be a string, got {type(command_string).__name__}: "
                f"{command_string!r}. Pass a shell command, e.g. {{\"command\": \"ls -la\"}}."
            )
        if not command_string or not command_string.strip():
            raise TerminalError(
                "The 'command' argument is empty. Pass a non-empty shell command, "
                "e.g. {\"command\": \"ls -la\"}."
            )

        mem_limit = self.mem_limit_bytes
        os.makedirs(self.workspace, exist_ok=True)  # bwrap --bind / cwd need it

        # Per-command cgroup leaf: real-RSS memory.max + pids cap over the whole
        # {bash -> children} tree. RLIMIT_AS is only a fallback for when the
        # cgroup isn't available (see landlock_sandbox for the rationale).
        cgroup_dir = None
        if _CGROUP_ENABLE:
            cgroup_dir = _create_cgroup(
                f"term_{os.getpid()}_{next(_LEAF_COUNTER)}",
                mem_max=mem_limit,
                pids_max=_CGROUP_PIDS_MAX,
            )
        cgroup_procs = os.path.join(cgroup_dir, "cgroup.procs") if cgroup_dir else None

        def _preexec():
            # Join the cgroup so its caps cover bash and every child it spawns.
            joined = False
            if cgroup_procs is not None:
                try:
                    with open(cgroup_procs, "w") as f:
                        f.write(str(os.getpid()))
                    joined = True
                except OSError:
                    joined = False
            # RLIMIT_AS only when the cgroup cap isn't in effect; never stacked
            # (AS over-counts mmap and false-positives, e.g. importing torch).
            if not joined and mem_limit and mem_limit > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                except (ValueError, OSError):
                    pass

        # Wrap in bwrap (same default-deny fs confinement as the kernel sandbox)
        # so terminal commands can't read the task's data.db / other rollouts'
        # env_dirs / model weights, nor write outside the workspace. No env_dir
        # is exposed (unlike the PTC sandbox), so `cat ../data.db` cannot reach
        # the answers. Falls back to a bare bash when bwrap is unavailable.
        argv = ["/bin/bash", "-c", command_string]
        if _USE_BWRAP:
            argv = _build_bwrap_argv(self.workspace) + argv

        # Use Popen + start_new_session so we can kill the *entire process
        # group* on timeout. subprocess.run(timeout=...) only SIGKILLs the
        # shell leader, leaving any children (e.g. a `python3 infinite.py` the
        # shell spawned) running as orphans and burning CPU/RAM until the host
        # OOMs. Killing the pgid takes them all down together.
        try:
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=self.workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    # Shell tools (file/grep on binaries, locale-mismatched output,
                    # etc.) routinely emit non-UTF-8 bytes. Without ``replace`` the
                    # default strict decoder raises UnicodeDecodeError inside
                    # ``_translate_newlines`` and crashes the rollout worker.
                    errors="replace",
                    start_new_session=True,
                    preexec_fn=_preexec,
                )
            except Exception as e:
                raise TerminalError(f"Execution failed: {e}")

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    stdout, stderr = "", ""
                raise TerminalError(f"Command timed out after {self.timeout}s")
        finally:
            # Reclaim the leaf once the process group is gone. rmdir fails if a
            # backgrounded child still lingers; the idle sweep
            # (kill_all_kernels -> _sweep_cgroup_leaves) reclaims those later.
            if cgroup_dir is not None:
                try:
                    os.rmdir(cgroup_dir)
                except OSError:
                    pass

        output = stdout
        if stderr:
            output += f"\n[stderr]: {stderr}"
        if not output.strip():
            output = f"(no output, exit code: {returncode})"
        return output
