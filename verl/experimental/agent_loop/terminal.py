"""Terminal executor: runs shell commands inside the workspace directory."""

import os
import subprocess


class TerminalError(Exception):
    pass


class TerminalExecutor:
    """Executes shell commands in the workspace via /bin/bash. No allowlist; pipes / redirects / chaining all permitted."""

    def __init__(self, workspace: str, timeout: int = 60):
        self.workspace = os.path.abspath(os.path.realpath(workspace))
        self.timeout = timeout

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
        try:
            result = subprocess.run(
                command_string,
                shell=True,
                executable="/bin/bash",
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise TerminalError(f"Command timed out after {self.timeout}s")
        except Exception as e:
            raise TerminalError(f"Execution failed: {e}")

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = f"(no output, exit code: {result.returncode})"
        return output
