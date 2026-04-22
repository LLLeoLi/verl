"""Secure terminal executor restricted to a small set of commands within a workspace."""

import os
import shlex
import subprocess


class TerminalError(Exception):
    pass


class TerminalExecutor:
    """Secure terminal executor restricted to ls, find, cat within workspace."""

    ALLOWED_COMMANDS = {"ls", "find", "cat"}

    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(os.path.realpath(workspace))

    def _normalize_path(self, path: str) -> str:
        if os.path.isabs(path):
            real = os.path.abspath(os.path.realpath(path))
        else:
            real = os.path.abspath(os.path.realpath(os.path.join(self.workspace, path)))
        if not real.startswith(self.workspace):
            raise TerminalError(f"Path '{path}' is outside workspace")
        return real

    def _build_args(self, raw_args: list) -> list:
        validated = []
        for arg in raw_args:
            if arg.startswith("-"):
                validated.append(arg)
            elif (arg.startswith(("./", "../", "/")) and not arg.startswith("//")) or arg == ".":
                validated.append(self._normalize_path(arg))
            elif "/" in arg and os.path.exists(os.path.join(self.workspace, arg)):
                validated.append(self._normalize_path(arg))
            else:
                validated.append(arg)  # pattern, text, etc.
        return validated

    def execute(self, command_string: str) -> str:
        if len(command_string) > 1024:
            raise TerminalError("Command too long (max 1024 chars)")
        try:
            parts = shlex.split(command_string)
        except ValueError as e:
            raise TerminalError(f"Invalid command format: {e}")
        if not parts:
            raise TerminalError("Empty command")

        cmd = parts[0]
        if cmd not in self.ALLOWED_COMMANDS:
            raise TerminalError(
                f"Command '{cmd}' not allowed. Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
            )

        args = self._build_args(parts[1:])
        try:
            result = subprocess.run(
                [cmd] + args,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if not output.strip():
                output = f"(no output, exit code: {result.returncode})"
            return output
        except subprocess.TimeoutExpired:
            raise TerminalError("Command timed out after 30s")
        except Exception as e:
            raise TerminalError(f"Execution failed: {e}")
