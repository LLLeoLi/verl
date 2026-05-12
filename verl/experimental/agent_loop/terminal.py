"""Terminal executor: runs shell commands inside the workspace directory."""

import os
import resource
import signal
import subprocess

# Default per-command address-space cap (bytes). Inherited by every child
# process the shell spawns, so a runaway `python3 foo.py` raises MemoryError
# instead of eating the whole node. Override via the `mem_limit_bytes` kwarg
# or the VERL_TERMINAL_MEM_LIMIT_BYTES env var. 0 disables the cap.
_DEFAULT_MEM_LIMIT_BYTES = int(os.getenv("VERL_TERMINAL_MEM_LIMIT_BYTES", str(8 * 1024**3)))


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

        def _preexec():
            if mem_limit and mem_limit > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                except (ValueError, OSError):
                    pass

        # Use Popen + start_new_session so we can kill the *entire process
        # group* on timeout. subprocess.run(timeout=...) only SIGKILLs the
        # shell leader, leaving any children (e.g. a `python3 infinite.py` the
        # shell spawned) running as orphans and burning CPU/RAM until the host
        # OOMs. Killing the pgid takes them all down together.
        try:
            proc = subprocess.Popen(
                command_string,
                shell=True,
                executable="/bin/bash",
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

        output = stdout
        if stderr:
            output += f"\n[stderr]: {stderr}"
        if not output.strip():
            output = f"(no output, exit code: {returncode})"
        return output
