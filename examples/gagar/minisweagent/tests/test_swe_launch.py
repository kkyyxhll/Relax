# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture the SWE submitter without running its server or Ray driver."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("eval_enabled", [False, True])
def test_swe_entry_owns_its_server_and_split_genrm(tmp_path, eval_enabled):
    root = Path(__file__).resolve().parents[4]
    recipe = root / "examples/gagar/minisweagent"
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "args.json"
    ray = binary / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
    )
    ray.chmod(0o755)
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    (venv / "activate").touch()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GAGAR_", "AGENT_SERVER_", "ENABLE_", "R2E_"))
    }
    env.update(
        PATH=f"{binary}:{os.environ['PATH']}",
        RELAX_ENTRYPOINT_MODE="ray-job",
        MODEL_CONFIG_DIR=str(root / "scripts/models"),
        MODEL_DIR=str(tmp_path / "models"),
        SAVE_DIR=str(tmp_path / "save"),
        R2E_DATA_PATH=str(tmp_path / "data"),
        R2E_SIF_DIR=str(tmp_path / "sifs"),
        EXP_ROOT=str(tmp_path / "experiment"),
        MINI_SWE_AGENT_VENV=str(venv.parent),
        RUNTIME_ENV_JSON='{"env_vars":{"EXISTING":"preserved"}}',
        CAPTURE=str(capture),
        ENABLE_EVAL=str(int(eval_enabled)),
    )
    subprocess.run(["bash", str(recipe / "run_gagar.sh")], env=env, cwd=tmp_path, check=True, capture_output=True)
    args = json.loads(capture.read_text())

    def value(flag):
        return args[args.index(flag) + 1]

    assert json.loads(value("--resource")) == {"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}
    assert args[args.index("--") + 1 : args.index("--") + 3] == ["bash", str(recipe / "driver.sh")]
    assert value("--agent-command") == f"bash {recipe}/agent_client.sh"
    assert value("--agent-cwd") == str(recipe)
    assert value("--custom-rm-path") == "examples.gagar.minisweagent.group_rm.reward_func"
    assert value("--genrm-num-gpus-per-engine") == "4"
    assert value("--context-parallel-size") == "4"
    assert "--disable-grpo-std-normalization" in args
    assert ("--eval-interval" in args) is eval_enabled
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["EXISTING"] == "preserved"
    assert runtime["env_vars"]["AGENT_SERVER_SIF_CACHE"] == "1"
    assert "examples.gagar.minisweagent.grader" in runtime["env_vars"]["GAGAR_GRADER_COMMAND"]
