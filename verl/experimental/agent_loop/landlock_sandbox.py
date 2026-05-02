"""
Stateful Python sandbox backed by an IPython kernel.

Variables, imports, and other state persist across execute() calls within
the same sandbox instance, enabling multi-turn code execution.

Isolation strategy (layered):
  1. Linux Landlock (kernel >= 5.13, no root required) -- applied in the kernel
     child process via preexec_fn.  Grants read+execute everywhere, but
     confines ALL writes (including from subprocess calls made by user code)
     to the workspace directory and a private tmpdir.
  2. Python-level open() guard -- defense-in-depth for write attempts that
     happen to go through the built-in open(); also covers the case where
     Landlock is unavailable.
"""

import ctypes
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

# Default per-kernel address-space cap (bytes). Override via the
# `mem_limit_bytes` kwarg to StatefulSandbox or the
# VERL_SANDBOX_MEM_LIMIT_BYTES env var. 0 disables the cap.
_DEFAULT_MEM_LIMIT_BYTES = int(os.getenv("VERL_SANDBOX_MEM_LIMIT_BYTES", str(8 * 1024**3)))

_KERNEL_START_LOCK = threading.Lock()

# Track all spawned kernel PIDs so we can force-kill stragglers between envs.
_SPAWNED_PIDS: set[int] = set()
_PID_LOCK = threading.Lock()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


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
    ):
        self.workspace_path = workspace_path
        self.timeout = timeout
        self.mem_limit_bytes = (
            mem_limit_bytes if mem_limit_bytes is not None else _DEFAULT_MEM_LIMIT_BYTES
        )
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self._tmpdir: str | None = None
        self._proc: subprocess.Popen | None = None
        self._client = None
        self._conn_file: Path | None = None
        self._cleaned_up = False
        self._dead = False
        self._landlock_active = False

    def start(self):
        """Start the IPython kernel and initialize workspace state."""
        self._cleaned_up = False
        try:
            self._start_impl()
        except Exception:
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
        writable = [self.workspace_path, self._tmpdir, "/tmp"]

        abi = _landlock_abi_version()
        landlock_ok = abi >= 1
        mem_limit = self.mem_limit_bytes

        def _preexec():
            # Cap address space so a single misbehaving kernel cannot OOM the
            # node. Soft+hard set to the same value; hitting it raises
            # MemoryError inside the kernel rather than killing the host.
            if mem_limit and mem_limit > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                except (ValueError, OSError):
                    pass
            if landlock_ok:
                _apply_landlock(writable)

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "ipykernel_launcher", "-f", str(self._conn_file)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.workspace_path,
            start_new_session=True,
            preexec_fn=_preexec,
        )

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
            client.start_channels()
            client.wait_for_ready(timeout=10.0)
        finally:
            _KERNEL_START_LOCK.release()

        self._client = client
        self._landlock_active = landlock_ok

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

        if landlock_ok:
            print(f"[sandbox] Landlock ABI v{abi} active -- writes restricted to workspace.")
        else:
            print("[sandbox] WARNING: Landlock unavailable -- Python open() guard only.")

    def _try_interrupt_to_idle(self, msg_id: str, grace_seconds: float) -> bool:
        """Send SIGINT to the kernel and wait up to grace_seconds for the
        in-flight execution (msg_id) to reach an idle status. Returns True if
        the kernel recovered, False otherwise. State (variables/imports) is
        preserved on a successful interrupt."""
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
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

    def _run(self, code: str) -> tuple[str, str, bool]:
        """Execute code in the kernel and return (stdout, stderr, success)."""
        if self._dead:
            raise RuntimeError("Kernel was killed after a previous timeout")
        if self._proc is not None and self._proc.poll() is not None:
            self._dead = True
            raise RuntimeError(
                f"Kernel process has exited unexpectedly (code {self._proc.returncode})"
            )
        client = self._client
        msg_id = client.execute(code, silent=False, store_history=True)
        deadline = time.time() + self.timeout

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

    def execute(self, code: str) -> tuple[str, bool]:
        """Execute code and return (output, success).

        State from previous calls is preserved. Returns an error string on
        failure rather than raising. ``success`` is False whenever the kernel
        reported an exception, the sandbox was unavailable, or execution timed
        out -- i.e. any case where ``output`` carries an error message.
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
        try:
            output, error, success = self._run(code)
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
                f"[Error] Code execution timed out ({self.timeout}-second limit).{suffix}",
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
