# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture recipe arguments without starting Ray, models, or containers."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_code_generation_wrapper_preserves_baseline_and_selects_split_genrm(tmp_path, sandbox, direct):
    root = Path(__file__).resolve().parents[4]
    if direct:
        isolated = tmp_path / "recipe"
        for name in [
            "examples/gagar/code_generate/baseline.sh",
            "examples/gagar/code_generate/run_gagar.sh",
            "scripts/models/qwen35-4B.sh",
        ]:
            target = isolated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        entrypoint = isolated / "scripts/entrypoint/local.sh"
        entrypoint.parent.mkdir()
        # Match the real helper's optional variable reads, without starting anything.
        entrypoint.write_text('printf "%s" "$NCCL_NVLS_ENABLE" >/dev/null\nexport RELAX_ENTRYPOINT_MODE=local\n')
        root = isolated
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "ray-args.json"
    ray = binary / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
    )
    ray.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("GAGAR_")}
    env.pop("RESOURCE", None)
    env.update(
        PATH=f"{binary}:{os.environ['PATH']}",
        RELAX_ENTRYPOINT_MODE="ray-job",
        MODEL_CONFIG_DIR=str(root / "scripts/models"),
        MODEL_DIR=str(tmp_path / "models"),
        DATA_DIR=str(tmp_path / "data"),
        SAVE_DIR=str(tmp_path / "save"),
        EXP_NAME="argument-capture",
        SANDBOX_URL="http://binary-judge",
        HOST_IP="127.0.0.1",
        RUNTIME_ENV_JSON='{"env_vars":{"EXISTING":"preserved"}}',
        MINI_SWE_AGENT_VENV=str(tmp_path / "venv"),
        GAGAR_SANDBOX_IMAGE=str(tmp_path / "grader.sif"),
        CAPTURE=str(capture),
    )
    if direct:
        env.pop("RELAX_ENTRYPOINT_MODE")
        env.pop("NCCL_NVLS_ENABLE", None)
    script = "examples/gagar/code_generate/run_gagar.sh" if sandbox else "examples/gagar/code_generate/baseline.sh"
    subprocess.run(["bash", str(root / script)], env=env, check=True, capture_output=True, text=True)
    args = json.loads(capture.read_text())

    def value(flag):
        index = len(args) - 1 - args[::-1].index(flag)
        return args[index + 1]

    assert value("--rollout-max-response-len") == "16384"
    assert value("--global-batch-size") == "256"
    assert value("--empty-unused-memory-level") == "1"
    resource = json.loads(value("--resource"))
    assert resource["actor"] == [1, 8]
    assert ("--group-rm" in args) is sandbox
    assert ("--disable-grpo-std-normalization" in args) is sandbox
    assert value("--context-parallel-size") == "1"
    if sandbox:
        assert resource == {"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}
        assert value("--custom-rm-path") == "examples.gagar.code_generate.sandbox.reward_group"
        assert value("--reward-key") == "score"
        assert value("--eval-reward-key") == "test_reward"
        runtime = json.loads(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--runtime-env-json=")))
        assert runtime["env_vars"]["EXISTING"] == "preserved"
        assert runtime["env_vars"]["GAGAR_GRADER_MODE"] == "sandbox"
        assert runtime["env_vars"]["GAGAR_SANDBOX_IMAGE"] == env["GAGAR_SANDBOX_IMAGE"]
    else:
        assert resource == {"actor": [1, 8], "rollout": [1, 8]}
        assert value("--custom-rm-path") == "examples.gagar.code_generate.app.reward.reward"
