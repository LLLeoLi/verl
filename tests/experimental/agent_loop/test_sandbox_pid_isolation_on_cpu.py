# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandboxed rollout code must not be able to signal processes outside itself.

Regression test for a whole-node kill: without a PID namespace the sandbox
shares the pod's, so `--proc /proc` still enumerates every process on the node
and model-generated code (running as root on the training cluster) could
`pkill -f python` / `kill -TERM -1` the raylet, the Ray WorkerDicts and the vLLM
servers. Ray then reports the node as "terminated expectedly: received SIGTERM"
and the whole job cascades. Landlock confines the filesystem only -- it does not
scope signals -- so the PID namespace is what actually contains this.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import pytest

from verl.experimental.agent_loop.landlock_sandbox import (
    StatefulSandbox,
    bwrap_pid_ns_usable,
)

pytestmark = pytest.mark.skipif(
    not bwrap_pid_ns_usable(),
    reason="bwrap cannot create a PID namespace here (needs userns/CAP_SYS_ADMIN)",
)


@pytest.fixture
def victim():
    """A process outside the sandbox, standing in for the Ray raylet."""
    d = tempfile.mkdtemp(prefix="verl_victim_")
    script = os.path.join(d, "raylet_lookalike.sh")
    with open(script, "w") as f:
        f.write("sleep 120\n")
    proc = subprocess.Popen(
        ["/bin/bash", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    assert proc.poll() is None, "victim failed to start"
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture
def sandbox():
    ws = tempfile.mkdtemp(prefix="verl_sbx_ws_")
    sb = StatefulSandbox(workspace_path=ws, timeout=20.0)
    sb.start()
    try:
        yield sb
    finally:
        sb.cleanup()


def test_sandbox_sees_only_its_own_namespace(sandbox):
    out, ok = sandbox.execute(
        "import os\n"
        "print('N', len([p for p in os.listdir('/proc') if p.isdigit()]))"
    )
    assert ok, out
    n = int(out.split("N")[1].split()[0])
    assert 0 < n <= 5, f"sandbox can see {n} processes; PID namespace not applied"


def test_pkill_cannot_reach_outside_process(sandbox, victim):
    out, _ = sandbox.execute(
        "import subprocess\n"
        "subprocess.run(['pkill', '-TERM', '-f', 'raylet_lookalike'])"
    )
    time.sleep(0.5)
    assert victim.poll() is None, f"pkill inside the sandbox killed an outside process ({out})"


def test_kill_by_host_pid_cannot_reach_outside_process(sandbox, victim):
    out, _ = sandbox.execute(
        "import os, signal\n"
        "try:\n"
        f"    os.kill({victim.pid}, signal.SIGTERM); print('SIGNALLED')\n"
        "except ProcessLookupError:\n"
        "    print('BLOCKED')"
    )
    time.sleep(0.5)
    assert "BLOCKED" in out, out
    assert victim.poll() is None, "os.kill inside the sandbox reached an outside process"


def test_kernel_pid_is_the_interpreter_not_bwrap_init(sandbox):
    """--unshare-pid inserts bwrap's init between bwrap and the kernel.

    That init keeps bwrap's own argv, which also ends in "ipykernel_launcher",
    so a naive match picks it and interrupts silently degrade to a hard kill.
    """
    with open(f"/proc/{sandbox._kernel_pid}/cmdline", "rb") as f:
        argv = [a for a in f.read().split(b"\0") if a]
    assert os.path.basename(argv[0]) != b"bwrap", f"resolved bwrap init, not the kernel: {argv}"
    assert any(b"ipykernel" in a for a in argv), argv


def test_interrupt_preserves_kernel_state(sandbox):
    out, ok = sandbox.execute("x = 42\nprint(x)")
    assert ok and "42" in out, out

    out, ok = sandbox.execute("import time\ntime.sleep(30)", timeout=3.0)
    assert not ok and "timed out" in out.lower(), out

    out, ok = sandbox.execute("print('alive', x)")
    assert ok and "alive 42" in out, f"kernel did not survive the interrupt: {out}"
