# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import json

import pytest


pytest.importorskip("minisweagent")
from examples.gagar.grader_common import read_evidence
from examples.gagar.minisweagent.grader import BridgeEnvironment


def test_trajectory_pages_remove_only_mirrored_payloads_without_mutating_audit():
    assistant = {
        "role": "assistant",
        "content": "Reasoning: investigate external solution access. " * 400,
        "tool_calls": [{"id": "call-a", "function": {"name": "bash", "arguments": '{"command":"curl example"}'}}],
        "reasoning_content": "Original reasoning must remain.",
        "extension": {"unknown": "original extension"},
    }
    response = {
        "id": "response-a",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": copy.deepcopy(assistant)}],
        "usage": {"completion_tokens": 99},
    }
    assistant["extra"] = {
        "response": response,
        "actions": [{"command": "curl example", "tool_call_id": "call-a"}],
        "timestamp": 123,
    }
    trajectory = [
        assistant,
        {
            "role": "tool",
            "content": "<output>retrieved code</output>",
            "tool_call_id": "call-a",
            "extra": {"raw_output": "retrieved code", "returncode": 0, "exception_info": "original diagnostic"},
        },
        {"role": "exit", "content": "done", "extra": {"exit_status": "Submitted", "submission": "patch"}},
    ]
    manifest = {"candidates": [{"trajectory": trajectory}]}
    original = copy.deepcopy(manifest)
    action = {"rollout_n": 0, "section": "trajectory"}
    text = ""
    offsets = []
    while True:
        page = read_evidence(manifest, action, {"trajectory"})
        text += page["output"]
        assert page["returncode"] == 0
        if page["next_offset"] is None:
            break
        offsets.append(page["next_offset"])
        action["offset"] = page["next_offset"]
    assert offsets
    view = json.loads(text)
    expected = copy.deepcopy(trajectory)
    del expected[0]["extra"]["response"]["choices"][0]["message"]
    del expected[1]["extra"]["raw_output"]
    assert view == expected
    assert manifest == original
    assert len(text) < len(json.dumps(trajectory, ensure_ascii=False))


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"message": {"role": "assistant", "content": "unique provider reasoning"}}]},
        {"choices": [{"message": {"role": "assistant", "content": "short"}}, {"message": {"content": "choice2"}}]},
        {"choices": [{"message": {"content": "short", "missing_field": None}}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": "provider-specific format"}]},
        {"choices": ["provider-specific choice"]},
        {"choices": None},
        "provider-specific response",
    ],
)
def test_trajectory_preserves_nonmirrored_responses_and_unshown_output(response):
    trajectory = [
        {
            "role": "assistant",
            "content": "short",
            "extra": {"response": response, "exception_str": "failure", "traceback": "original traceback"},
        },
        {"role": "tool", "content": "truncated...", "extra": {"raw_output": "complete original output"}},
    ]
    result = read_evidence(
        {"candidates": [{"trajectory": trajectory}]}, {"rollout_n": 0, "section": "trajectory"}, {"trajectory"}
    )
    assert json.loads(result["output"]) == trajectory


def test_bridge_trajectory_view_preserves_original_serialized_evidence_and_other_sections():
    manifest = {
        "group_id": "g",
        "bridge_url": "http://bridge",
        "candidates": [{"trajectory": "unstructured original trajectory", "patch": {"output": "original patch"}}],
    }
    env = BridgeEnvironment(manifest)
    try:
        result = env.execute({"name": "read_evidence", "rollout_n": 0, "section": "trajectory"})
        assert result == {"output": "unstructured original trajectory", "next_offset": None, "returncode": 0}
        patch = env.execute({"name": "read_evidence", "rollout_n": 0, "section": "patch"})
        assert json.loads(patch["output"]) == {"output": "original patch"}
    finally:
        env.client.close()
