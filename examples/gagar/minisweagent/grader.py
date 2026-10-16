#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A mini-swe-agent process using Relax GenRM and sandbox-only tools."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml

from examples.gagar.grader_common import GraderEnvironment, run_grader
from examples.gagar.minisweagent.scoring import grade_to_weights


class BridgeBash:
    """Route model commands to the selected candidate through the bridge."""

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        group_id = quote(manifest["group_id"], safe="")
        self.url = f"{manifest['bridge_url'].rstrip('/')}/gagar/groups/{group_id}/bash"
        self.client = httpx.Client(timeout=135, trust_env=False)

    def __call__(self, action: dict) -> dict:
        # Validate the exact model argument set at the bridge, including
        # a missing rollout_n. Never accept an endpoint from the model.
        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.manifest['token']}"},
            json={key: value for key, value in action.items() if key != "name"},
        )
        response.raise_for_status()
        return response.json()


def create_environment(manifest: dict) -> GraderEnvironment:
    """Bind this task's tools and scoring to the shared grader environment."""
    return GraderEnvironment(
        manifest,
        bash=BridgeBash(manifest),
        grade_to_weights=grade_to_weights,
        sections={"patch", "trajectory", "test_output", "submission"},
        evidence_access=(
            "Inspect frozen patches, relevant original trajectories and test evidence. "
            "Use bash(rollout_n, command) to investigate the candidate repositories as needed."
        ),
        environment_type="gagar_sandbox_bridge",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    config_path = Path(os.environ.get("GAGAR_CONFIG", Path(__file__).with_name("grader.yaml")))
    config = yaml.safe_load(config_path.read_text())
    config["agent"].setdefault("instance_template", Path(__file__).with_name("instance.jinja").read_text())
    env = create_environment(manifest)
    try:
        run_grader(manifest, args.output, env, config)
    finally:
        env.bash.client.close()


if __name__ == "__main__":
    main()
