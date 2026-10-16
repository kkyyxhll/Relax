# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest


genrm = pytest.importorskip("relax.components.genrm")


@pytest.mark.parametrize("as_json", [False, True])
def test_generate_request_preserves_assistant_reasoning(as_json):
    messages = [
        {"role": "system", "content": "grade"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "final action", "reasoning_content": "check assumptions"},
        {"role": "user", "content": "tool result"},
    ]
    payload = {"messages": messages}
    request = (
        genrm.GenerateRequest.model_validate_json(json.dumps(payload))
        if as_json
        else genrm.GenerateRequest.model_validate(payload)
    )
    assert request.model_dump(exclude_none=True)["messages"] == messages


def test_generate_request_still_accepts_messages_without_reasoning():
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}]
    request = genrm.GenerateRequest.model_validate({"messages": messages})
    assert request.model_dump(exclude_none=True)["messages"] == messages
