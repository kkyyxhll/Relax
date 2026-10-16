# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local tools for the grader process that already lives inside Apptainer."""

import argparse
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import yaml

from examples.gagar.code_generate.scoring import grade_to_weights
from examples.gagar.grader_common import GraderEnvironment, run_grader


class SandboxBash:
    """Execute model commands inside the already isolated grader sandbox."""

    def __init__(self, manifest: dict) -> None:
        # A configuration mistake must never turn this into host bash execution.
        if not os.environ.get("APPTAINER_CONTAINER") or not Path("/candidates").is_dir():
            raise RuntimeError("sandbox grader must run inside its Apptainer instance")
        self.timeout = manifest.get("command_timeout", 120)
        Path("/workspace").mkdir(exist_ok=True)

    def __call__(self, action: dict) -> dict:
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


def create_environment(manifest: dict) -> GraderEnvironment:
    """Bind this task's tools and scoring to the shared grader environment."""
    return GraderEnvironment(
        manifest,
        bash=SandboxBash(manifest),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    config = yaml.safe_load(Path(__file__).with_name("grader.yaml").read_text())
    config["agent"].setdefault("instance_template", Path(__file__).with_name("instance.jinja").read_text())
    env = create_environment(manifest)
    run_grader(manifest, args.output, env, config)


if __name__ == "__main__":
    main()
