# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local tools for the grader process that already lives inside Apptainer."""

import argparse
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from examples.gagar.code_generate.scoring import grade_to_weights
from examples.gagar.grader_common import read_evidence, run_grader, submit_result


class SandboxEnvironment:
    def __init__(self, manifest: dict) -> None:
        # A configuration mistake must never turn this into host bash execution.
        if not os.environ.get("APPTAINER_CONTAINER") or not Path("/candidates").is_dir():
            raise RuntimeError("sandbox grader must run inside its Apptainer instance")
        self.manifest = manifest
        self.timeout = manifest.get("command_timeout", 120)
        Path("/workspace").mkdir(exist_ok=True)

    def execute(self, action: dict, cwd: str = "") -> dict:
        try:
            if action["name"] == "read_evidence":
                return read_evidence(self.manifest, action, {"code", "test_output"})
            if action["name"] == "submit_grades":
                submit_result(grade_to_weights([c["reward"] for c in self.manifest["candidates"]], action))
            if action["name"] != "bash":
                raise ValueError("unknown tool")
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
        except (ValueError, KeyError, TypeError) as exc:
            return {"output": f"Tool rejected: {exc}", "returncode": -1}

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {
            "evidence_access": (
                "Read code and test_output using read_evidence. Original thinking is not provided. "
                "Frozen Python files: /candidates/<rollout_n>/solution.py. "
                "bash(command) runs here in /workspace; use it for bounded experiments. "
                "Python and the SandboxFusion harness are installed in /opt/sf and /opt/sandboxfusion."
            )
        }

    def serialize(self) -> dict:
        return {"info": {"environment_type": "gagar_apptainer_sandbox"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    config = yaml.safe_load(Path(__file__).with_name("grader.yaml").read_text())
    config["agent"].setdefault("instance_template", Path(__file__).with_name("instance.jinja").read_text())
    run_grader(manifest, args.output, SandboxEnvironment(manifest), config)


if __name__ == "__main__":
    main()
