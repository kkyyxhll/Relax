# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Each task imports its own environment without another example recipe."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "task, modules",
    [
        ("code_generate", ["apptainer", "sandbox", "sandbox_env", "app.reward", "app.sandbox_fusion"]),
        ("minisweagent", ["agent_server", "bridge", "grader", "group_rm", "sif_cache"]),
    ],
)
def test_task_does_not_import_other_recipes(task, modules):
    pytest.importorskip("minisweagent")
    root = Path(__file__).resolve().parents[3]
    source = """
import importlib
import importlib.abc
import json
import sys

banned = ["examples.mini_swe_agent", "examples.code_generate"]
if sys.argv[1] == "code_generate":
    banned.append("examples.gagar.minisweagent")
else:
    banned.append("examples.gagar.code_generate")

class RejectOtherRecipe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in banned):
            raise RuntimeError("Cross-recipe import: " + fullname)

sys.meta_path.insert(0, RejectOtherRecipe())
for name in json.loads(sys.argv[2]):
    importlib.import_module("examples.gagar." + sys.argv[1] + "." + name)
"""
    subprocess.run(
        [sys.executable, "-c", source, task, json.dumps(modules)], cwd=root, check=True, capture_output=True
    )
