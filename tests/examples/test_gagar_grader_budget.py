# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import json
from types import SimpleNamespace

import httpx
import pytest


pytest.importorskip("minisweagent")
from minisweagent.exceptions import TimeExceeded

from examples.gagar import grader_common
from examples.gagar.grader import BridgeEnvironment, GenRMModel, GraderAgent
from examples.gagar.scoring import CRITERIA


@pytest.mark.parametrize(
    "elapsed, calls, step_limit, time_limit, reminder",
    [
        (1199, 20, 40, 1500, False),
        (1200, 20, 40, 1500, True),
        (1450, 20, 40, 1500, True),
        (10000, 20, 40, 0, False),
        (10, 37, 40, 0, True),
        (1250, 20, 0, 1500, True),
        (1500, 39, 40, 1500, False),
    ],
)
def test_grader_budget_reminder_respects_existing_limits(
    monkeypatch, elapsed, calls, step_limit, time_limit, reminder
):
    histories = []

    def query(messages):
        histories.append(copy.deepcopy(messages))
        return {"role": "assistant", "content": "response", "extra": {"cost": 0}}

    model = SimpleNamespace(query=query, format_message=lambda **kwargs: kwargs)
    agent = GraderAgent(
        model,
        object(),
        system_template="grade",
        instance_template="{{task}}",
        step_limit=step_limit,
        wall_time_limit_seconds=time_limit,
    )
    agent.messages = [{"role": "system", "content": "grade"}]
    agent.n_calls = calls
    agent._start_time = 0
    monkeypatch.setattr(grader_common.time, "time", lambda: elapsed)
    if time_limit and elapsed >= time_limit:
        with pytest.raises(TimeExceeded):
            agent.query()
        assert not histories
        assert agent.messages == [{"role": "system", "content": "grade"}]
        return
    agent.query()
    warnings = [m["content"] for m in histories[0] if m["role"] == "user"]
    assert bool(warnings) is reminder
    if reminder:
        assert len(warnings) == 1 and "submit_grades" in warnings[0]
        if time_limit:
            assert f"{time_limit - elapsed} seconds remain" in warnings[0]
        if not step_limit:
            assert "unlimited" in warnings[0] and "-20" not in warnings[0]
    assert agent.n_calls == calls + 1
    assert agent.config.step_limit == step_limit
    assert agent.config.wall_time_limit_seconds == time_limit


def test_wall_budget_leaves_time_for_real_submission_validation_and_repair(monkeypatch, tmp_path):
    clock = [0]
    monkeypatch.setattr(grader_common.time, "time", lambda: clock[0])
    manifest = {
        "group_id": "g",
        "bridge_url": "http://bridge",
        "token": "secret",
        "candidates": [{"rollout_n": 0, "reward": 1, "patch": "fix"}],
    }
    env = BridgeEnvironment(manifest)
    execute = env.execute
    investigations = []

    def investigate(action, **kwargs):
        if action["name"] == "read_evidence":
            investigations.append(action)
            clock[0] = 1250 if len(investigations) == 1 else 1501
        return execute(action, **kwargs)

    env.execute = investigate
    histories = []

    def generate(request):
        messages = json.loads(request.content)["messages"]
        histories.append(messages)
        if "seconds remain" not in messages[-1]["content"]:
            action = {"name": "read_evidence", "rollout_n": 0, "section": "patch"}
        elif len(histories) == 2:
            action = {"name": "submit_grades", "evaluations": [0], "ranking": [[0]]}
        else:
            action = {
                "name": "submit_grades",
                "evaluations": [
                    {
                        "rollout_n": 0,
                        "scores": {c: 5 for c in CRITERIA},
                        "issues": [],
                        "evidence": "frozen patch inspected",
                    }
                ],
                "ranking": [[0]],
            }
        return httpx.Response(200, json={"response": json.dumps(action)})

    model = GenRMModel("http://genrm")
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(generate))
    try:
        agent = GraderAgent(
            model,
            env,
            system_template="grade",
            instance_template="{{task}}",
            step_limit=40,
            wall_time_limit_seconds=1500,
            output_path=tmp_path / "trajectory.json",
        )
        result = agent.run("task")
        assert result["exit_status"] == "Submitted"
        assert json.loads(result["submission"])["weights"] == [1]
        assert len(investigations) == 1 and agent.n_calls == 3
        assert "Tool rejected" in histories[2][-2]["content"]
        assert "250 seconds remain" in histories[1][-1]["content"]
    finally:
        env.client.close()
        model.client.close()
