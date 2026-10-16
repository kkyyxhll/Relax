#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A mini-swe-agent process using Relax GenRM and sandbox-only tools."""

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml

from examples.gagar.grader_common import (
    GenRMModel as GenRMModel,
)
from examples.gagar.grader_common import (
    GraderAgent as GraderAgent,
)
from examples.gagar.grader_common import (
    parse_action as parse_action,
)
from examples.gagar.grader_common import (
    read_evidence,
    run_grader,
    submit_result,
)
from examples.gagar.scoring import grade_to_weights


class BridgeEnvironment:
    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        group_id = quote(manifest["group_id"], safe="")
        self.url = f"{manifest['bridge_url'].rstrip('/')}/gagar/groups/{group_id}/bash"
        self.client = httpx.Client(timeout=135, trust_env=False)

    def execute(self, action: dict, cwd: str = "") -> dict:
        try:
            name = action["name"]
            if name == "bash":
                # Validate the exact model argument set at the bridge, including
                # a missing rollout_n. Never accept an endpoint from the model.
                response = self.client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.manifest['token']}"},
                    json={key: value for key, value in action.items() if key != "name"},
                )
                response.raise_for_status()
                return response.json()
            if name == "read_evidence":
                return read_evidence(self.manifest, action, {"patch", "trajectory", "test_output", "submission"})
            if name == "submit_grades":
                rewards = [candidate["reward"] for candidate in self.manifest["candidates"]]
                submit_result(grade_to_weights(rewards, action))
            raise ValueError("unknown tool")
        except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
            return {"output": f"Tool rejected: {exc}", "returncode": -1}

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {
            "evidence_access": (
                "Inspect frozen patches, relevant original trajectories and test evidence. "
                "Use bash(rollout_n, command) to investigate the candidate repositories as needed."
            )
        }

    def serialize(self) -> dict:
        # Do not serialize the bridge capability or the complete frozen evidence
        # on every step; the server stores a sanitized manifest separately.
        return {"info": {"environment_type": "gagar_sandbox_bridge"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    config_path = Path(os.environ.get("GAGAR_CONFIG", Path(__file__).with_name("grader.yaml")))
    config = yaml.safe_load(config_path.read_text())
    config["agent"].setdefault("instance_template", Path(__file__).with_name("instance.jinja").read_text())
    env = BridgeEnvironment(manifest)
    try:
        run_grader(manifest, args.output, env, config)
    finally:
        env.client.close()


if __name__ == "__main__":
    main()
