# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local tools for the grader process that already lives inside Apptainer."""

import os
import signal
import subprocess
import tempfile
from pathlib import Path

from examples.gagar.app.code_generate.scoring import grade_to_weights
from examples.gagar.core.grs import GraderEnvironment


class SandboxBash:
    """Execute model commands inside the already isolated grader sandbox."""

    def __init__(self, manifest: dict) -> None:
        # A configuration mistake must never turn this into host bash execution.
        if not os.environ.get("APPTAINER_CONTAINER") or not Path("/candidates").is_dir():
            raise RuntimeError("sandbox grader must run inside its Apptainer instance")
        self.timeout = manifest.get("command_timeout", 120)
        Path("/workspace").mkdir(exist_ok=True)

    def execute(self, action: dict) -> dict:
        if set(action) != {"name", "command"} or not isinstance(action["command"], str):
            raise ValueError("bash accepts only name and command; use /candidates/<rollout_n>/solution.py")
        # Neither host credentials nor the inference socket configuration
        # are put in the shell environment. Network isolation is enforced
        # by the outer container, not by checking command strings.
        env = {
            "PATH": "/opt/sf/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/workspace",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "PYTHONPATH": "/opt/sandboxfusion",
            "SANDBOX_CONFIG": "gagar",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(
                ["/bin/bash", "-c", action["command"]],
                cwd="/workspace",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also remove ordinary background jobs after a command exits.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            output.seek(0)
            text = output.read(12000).decode(errors="replace")
        return {"output": text, "returncode": -1 if timed_out else proc.returncode, "timed_out": timed_out}

    def close(self) -> None:
        # Container ownership and command-process cleanup stay with the worker.
        pass


def create_environment(manifest: dict) -> GraderEnvironment:
    """Bind this task's tools and scoring to the shared grader environment."""
    return GraderEnvironment(
        manifest,
        backend=SandboxBash(manifest),
        grade_to_weights=grade_to_weights,
        sections={"code", "test_output"},
        evidence_access=(
            "Read code and test_output using read_evidence. Original thinking is not provided. "
            "Frozen Python files: /candidates/<rollout_n>/solution.py. "
            "bash(command) runs here in /workspace; use it for bounded experiments. "
            "Python and the SandboxFusion harness are installed in /opt/sf and /opt/sandboxfusion."
        ),
        environment_type="gagar_apptainer_sandbox",
    )
