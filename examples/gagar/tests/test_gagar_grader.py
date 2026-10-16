# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import httpx
import pytest


pytest.importorskip("minisweagent")
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError

from examples.gagar.grader_common import GenRMModel, GraderAgent, parse_action
from examples.gagar.minisweagent.grader import BridgeEnvironment
from examples.gagar.minisweagent.scoring import CRITERIA


def test_parse_action_never_executes_thinking_block():
    text = '<think>{"name":"bash","rollout_n":99,"command":"evil"}</think>\n{"name":"read_evidence","rollout_n":0,"section":"patch"}'
    assert parse_action(text)["name"] == "read_evidence"
    with pytest.raises(ValueError):
        parse_action('<think>{"name":"bash","rollout_n":0,"command":"evil"}')


@pytest.mark.parametrize("closing_marker", ["", "\n</tool_call>"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize(
    "action",
    [
        {"name": "bash", "command": "printf '%s\\n' '<tool_call>' '</tool_call>' '\"quoted\"'"},
        {"name": "read_evidence", "rollout_n": 3, "section": "code", "offset": 12000},
        {"name": "submit_grades", "evaluations": [{"rollout_n": 3, "weight": 0.5, "evidence": "measured"}]},
    ],
)
def test_tool_call_marker_preserves_complete_action(action, thinking, closing_marker):
    answer = "<tool_call>\n" + json.dumps(action) + closing_marker
    response = "reasoning</think>\n" + answer if thinking else answer
    assert parse_action(response, thinking=thinking) == action


@pytest.mark.parametrize(
    "answer",
    [
        '<tool_call>{"name":"bash","command":"pwd"',
        '<tool_call>{"name":"bash","command":"pwd"} {"name":"bash","command":"ls"}',
        '<tool_call>{"name":"bash","command":"pwd"}</tool_call><tool_call>{"name":"bash","command":"ls"}',
        'prose <tool_call>{"name":"bash","command":"pwd"}</tool_call>',
        '<tool_call>{"name":"bash","command":"pwd"}</tool_call> prose',
        '<tool_call><tool_call>{"name":"bash","command":"pwd"}</tool_call></tool_call>',
        '<tool_call id="1">{"name":"bash","command":"pwd"}</tool_call>',
        '<tool_call></think>{"name":"bash","command":"pwd"}',
        '<tool_call>{"name":"bash","command":"print(1)\nprint(2)"}',
        '<tool_call>{"name":"bash","command":"echo "hello""}',
        '<tool_call>{"name":"unknown"}',
    ],
)
def test_tool_call_marker_does_not_repair_or_extract_actions(answer):
    with pytest.raises(ValueError):
        parse_action("reasoning</think>\n" + answer, thinking=True)


def test_tool_call_marker_cannot_end_unfinished_thinking():
    with pytest.raises(ValueError, match="unfinished thinking block"):
        parse_action('<think><tool_call>{"name":"bash","command":"pwd"}</tool_call>')


@pytest.mark.parametrize("prefix", ["", "<think>"])
@pytest.mark.parametrize("fenced", [False, True])
def test_prefilled_thinking_parses_only_final_action(prefix, fenced):
    action = {"name": "bash", "command": "grep '</think>' template.py"}
    final = json.dumps(action)
    if fenced:
        final = "```json\n" + final + "\n```"
    response = prefix + '{"name":"bash","command":"not an action"}</think>\n' + final
    assert parse_action(response, thinking=True) == action


@pytest.mark.parametrize(
    "response",
    [
        '{"name":"bash","command":"unfinished reasoning"}',
        '<think>{"name":"bash","command":"unfinished reasoning"}',
        "reasoning</think>",
        'reasoning</think>prose {"name":"bash","command":"pwd"}',
        'reasoning</think>{"name":"bash","command":"pwd"} {"name":"bash","command":"pwd"}',
    ],
)
def test_thinking_rejects_incomplete_or_ambiguous_action(response):
    with pytest.raises(ValueError):
        parse_action(response, thinking=True)


@pytest.mark.parametrize("prefix", ["", "<think>"])
@pytest.mark.parametrize("marker", ["", "<tool_call>\n"])
def test_genrm_thinking_retains_raw_history_and_extracts_only_final_action(prefix, marker):
    action = {"name": "read_evidence", "rollout_n": 0, "section": "code"}
    response = prefix + "Inspect the candidate before grading.</think>\n" + marker + json.dumps(action)
    histories = []

    def generate(request):
        histories.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"response": response})

    model = GenRMModel("http://genrm", thinking=True)
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(generate))
    try:
        message = model.query([{"role": "system", "content": "grade"}])
        assert message["content"] == response
        assert message["extra"]["actions"] == [action]
        model.query([message, {"role": "user", "content": "tool result"}])
        assert histories[-1][0] == {
            "role": "assistant",
            "content": "<think>\nInspect the candidate before grading.\n</think>\n\n" + marker + json.dumps(action),
        }
        assert message["content"] == response
    finally:
        model.client.close()


def test_unfinished_thinking_stays_reasoning_on_next_query():
    response = '{"name":"bash","command":"do not execute"}'
    histories = []

    def generate(request):
        histories.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"response": response})

    model = GenRMModel("http://genrm", thinking=True)
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(generate))
    try:
        with pytest.raises(FormatError) as error:
            model.query([{"role": "user", "content": "grade"}])
        assert error.value.messages[0]["extra"]["actions"] == []
        with pytest.raises(FormatError):
            model.query(error.value.messages)
        assert histories[-1][0] == {"role": "assistant", "content": "<think>\n" + response + "\n</think>\n\n"}
        assert error.value.messages[0]["content"] == response
        assert "Finish thinking and close it with </think>" in error.value.messages[1]["content"]
        assert "Shell" not in error.value.messages[1]["content"]
    finally:
        model.client.close()


@pytest.mark.parametrize(
    "response",
    [
        "bad response",
        'Inspecting the code.\n{"name":"read_evidence","rollout_n":0,"section":"patch"}',
        '{"name":"bash","rollout_n":0,"command":"print(1)\nprint(2)"}',
        r'{"name":"bash","rollout_n":0,"command":"grep a\|b file.py"}',
    ],
)
def test_genrm_format_errors_preserve_original_assistant_message(response):
    model = GenRMModel("http://genrm")
    model.client.close()
    model.client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"response": response}))
    )
    try:
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "system", "content": "prompt"}])
        assert exc.value.messages[0]["content"] == response
        feedback = exc.value.messages[1]["content"]
        assert "no prose or Markdown" in feedback
        assert exc.value.messages[0]["extra"]["actions"] == []
    finally:
        model.client.close()


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize(
    "answer,expected",
    [
        ('Inspecting.\n{"name":"bash","command":"pwd"}', "starts with text or XML"),
        ("<tool_call><function=bash>pwd</function></tool_call>", "starts with text or XML"),
        ("", "final answer is empty"),
        ('{"name":"bash","command":"pwd"} More text', "Extra data"),
        ('{"name":"bash","command":"pwd"} {"name":"bash","command":"ls"}', "Extra data"),
        ('{"name":"not_a_tool"}', "named bash, read_evidence or submit_grades"),
    ],
)
def test_format_feedback_identifies_envelope_error_without_rewriting_command(thinking, answer, expected):
    response = "Keep this private reasoning out of the error.</think>\n" + answer if thinking else answer
    model = GenRMModel("http://genrm", thinking=thinking)
    model.client.close()
    model.client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"response": response}))
    )
    try:
        with pytest.raises(FormatError) as error:
            model.query([{"role": "user", "content": "grade"}])
        original, feedback = error.value.messages
        assert original["content"] == response
        assert original["extra"]["actions"] == []
        assert expected in feedback["content"]
        assert "encode EVERY" not in feedback["content"]
        assert "private reasoning" not in feedback["content"]
    finally:
        model.client.close()


@pytest.mark.parametrize("thinking", [False, True])
def test_format_feedback_points_to_json_error_not_shell_quoting(thinking, tmp_path):
    answer = '{"name":"bash","command":"' + "a" * 2000 + """; python3 -c 'print("hello")'"}"""
    response = "reasoning</think>\n" + answer if thinking else answer
    model = GenRMModel("http://genrm", thinking=thinking)
    model.client.close()
    model.client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"response": response}))
    )
    model.audit_path = tmp_path / "events.jsonl"
    try:
        with pytest.raises(FormatError) as error:
            model.query([{"role": "user", "content": "grade"}])
        feedback = error.value.messages[1]["content"]
        assert "Invalid JSON at character" in feedback
        assert "Nearby text, shown as an escaped string:" in feedback
        assert "Shell single quotes do not escape JSON double quotes" in feedback
        assert r"line breaks as \n" in feedback
        assert r"backslashes as \\" in feedback
        assert "a" * 100 not in feedback
        assert "hello" in feedback
        assert "hello" not in model.audit_path.read_text()
        assert error.value.messages[0]["extra"]["actions"] == []
    finally:
        model.client.close()


@pytest.mark.parametrize("marker", ["", "<tool_call>\n"])
def test_mini_agent_recovers_invalid_tool_then_submits_weights(tmp_path, marker):
    manifest = {
        "group_id": "g",
        "bridge_url": "http://bridge",
        "token": "capability",
        "candidates": [{"rollout_n": 0, "reward": 1, "patch": "original patch"}, {"rollout_n": 1, "reward": 0}],
    }
    env = BridgeEnvironment(manifest)
    env.client.close()
    commands = []

    def bridge_response(request):
        action = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer capability"
        commands.append(action)
        if "rollout_n" not in action:
            return httpx.Response(400, json={"error": "missing rollout_n"})
        return httpx.Response(200, json={"output": "repo evidence", "returncode": 0})

    env.client = httpx.Client(transport=httpx.MockTransport(bridge_response))
    actions = [
        {"name": "read_evidence", "rollout_n": 0, "section": "patch"},
        {"name": "bash", "command": "pwd"},
        {"name": "bash", "rollout_n": 0, "command": "git diff"},
        {"name": "submit_grades", "evaluations": [0], "ranking": [[0]]},
        {
            "name": "submit_grades",
            "evaluations": [
                {
                    "rollout_n": 0,
                    "scores": {c: 5 for c in CRITERIA},
                    "issues": [],
                    "evidence": "inspected patch and tests",
                }
            ],
            "ranking": [[0]],
        },
    ]
    histories = []

    def generate(request):
        histories.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"response": marker + json.dumps(actions[len(histories) - 1])})

    model = GenRMModel("http://genrm")
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(generate))
    try:
        agent = DefaultAgent(
            model,
            env,
            system_template="grade",
            instance_template="{{ task }}",
            output_path=tmp_path / "trajectory.json",
            step_limit=6,
        )
        info = agent.run("task")
        assert info["exit_status"] == "Submitted"
        assert json.loads(info["submission"])["weights"] == [1, 0]
        assert commands == [{"command": "pwd"}, {"rollout_n": 0, "command": "git diff"}]
        assert [len(history) for history in histories] == [2, 4, 6, 8, 10]
        saved = (tmp_path / "trajectory.json").read_text()
        assert "original patch" in saved and "capability" not in saved
    finally:
        env.client.close()
        model.client.close()


def test_tool_argument_can_contain_literal_thinking_delimiter():
    action = {"name": "bash", "rollout_n": 0, "command": "grep '</think>' template.py"}
    assert parse_action(json.dumps(action)) == action
    with pytest.raises(ValueError):
        parse_action('{"name": []}')


@pytest.mark.parametrize("submitted", [True, False])
def test_frozen_evidence_preserves_original_outcome_submission_gate(monkeypatch, submitted):
    from examples.gagar.minisweagent import git_evidence
    from examples.gagar.minisweagent.agent_server import MyAgent

    calls = []

    class Environment:
        def execute(self, action, **kwargs):
            command = action["command"]
            calls.append(command)
            output = (
                "short test summary info\nPASSED tests/test_fix.py::test_fix"
                if command == "reward"
                else "original diff"
            )
            return {"returncode": 0, "output": output}

    baseline_tree = "a" * 40

    def snapshot(env, evidence):
        calls.append("baseline")
        evidence["baseline"] = {"tree": baseline_tree}

    def run(agent, task, **kwargs):
        assert calls == ["setup", "baseline"]
        agent.messages = [{"role": "assistant", "content": "original trajectory"}]
        return {"exit_status": "Submitted" if submitted else "LimitsExceeded", "submission": "patch"}

    monkeypatch.setattr(DefaultAgent, "run", run)
    monkeypatch.setattr(git_evidence, "capture_baseline", snapshot)
    agent = MyAgent(
        object(),
        Environment(),
        rollout_mode="train",
        row={"expected_output_json": '{"test_fix":"PASSED"}'},
        setup_command="setup",
        reward_command="reward",
        capture_evidence=True,
        system_template="do task",
        instance_template="{{ task }}",
    )
    result = agent.run("task")
    assert result["reward"] == int(submitted)
    assert calls == ["setup", "baseline", "git diff --binary HEAD", f"git diff --binary {baseline_tree}"] + (
        ["reward"] if submitted else []
    )
    assert "capture_error" not in agent.grading_evidence
    agent.messages[0]["content"] = "changed later"
    assert agent.grading_evidence["trajectory"][0]["content"] == "original trajectory"
    assert agent.grading_evidence["patch"]["output"] == "original diff"


def test_grader_reserves_final_calls_for_submission_and_repair(tmp_path):
    manifest = {
        "group_id": "g",
        "bridge_url": "http://bridge",
        "token": "secret",
        "candidates": [{"rollout_n": 0, "reward": 1, "patch": "fix"}],
    }
    env = BridgeEnvironment(manifest)
    model = GenRMModel("http://genrm")
    model.client.close()
    histories = []

    def generate(request):
        messages = json.loads(request.content)["messages"]
        histories.append(messages)
        if len(histories) == 1:
            content = "invalid JSON"
        elif len(histories) == 2:
            content = json.dumps({"name": "read_evidence", "rollout_n": 0, "section": "patch"})
        elif "model calls remain" not in messages[-1]["content"]:
            content = json.dumps({"name": "read_evidence", "rollout_n": 0, "section": "patch"})
        elif len(histories) == 3:
            content = json.dumps({"name": "submit_grades", "evaluations": [0], "ranking": [[0]]})
        else:
            content = json.dumps(
                {
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
            )
        return httpx.Response(200, json={"response": content})

    model.client = httpx.Client(transport=httpx.MockTransport(generate))
    try:
        agent = GraderAgent(
            model,
            env,
            system_template="grade",
            instance_template="{{ task }}",
            step_limit=5,
            output_path=tmp_path / "trajectory.json",
        )
        result = agent.run("task")
        assert result["exit_status"] == "Submitted"
        assert json.loads(result["submission"])["weights"] == [1]
        assert agent.n_calls == 4
        assert histories[2][-1]["content"].startswith("3 model calls remain")
        assert histories[3][-1]["content"].startswith("2 model calls remain")
        assert "Tool rejected" in histories[3][-2]["content"]
        assert "3 model calls remain" in (tmp_path / "trajectory.json").read_text()
    finally:
        model.client.close()
        env.client.close()


@pytest.mark.parametrize("status, exit_status, calls", [(400, "HTTPStatusError", 1), (200, "RepeatedFormatError", 3)])
def test_grader_failure_diagnostics_preserve_reason_without_evidence(
    monkeypatch, tmp_path, status, exit_status, calls
):
    from examples.gagar import grader_common

    model = GenRMModel("http://genrm")
    model.client.close()
    model.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json={"response": "bad response"}))
    )
    monkeypatch.setattr(grader_common, "GenRMModel", lambda *args, **kwargs: model)
    manifest = {
        "group_id": "group-a",
        "genrm_url": "http://genrm",
        "bridge_url": "http://bridge",
        "token": "secret-capability",
        "task": "private-task",
        "candidates": [{"rollout_n": 0, "reward": 1, "patch": "private-patch"}],
    }
    env = BridgeEnvironment(manifest)
    try:
        with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
            grader_common.run_grader(
                manifest,
                tmp_path / "result.json",
                env,
                {"agent": {"system_template": "grade", "instance_template": "{{ task }}", "step_limit": 4}},
            )
    finally:
        env.client.close()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["exit_status"] == exit_status
    assert summary["api_calls"] == summary["http_attempts"] == calls
    assert summary["passing_ids"] == [0]
    text = (tmp_path / "events.jsonl").read_text()
    events = [json.loads(line) for line in text.splitlines()]
    assert events[0]["event"] == "start" and events[-1]["event"] == "finished"
    assert all(event["group_id"] == "group-a" for event in events)
    assert all(secret not in text for secret in ["secret-capability", "private-task", "private-patch", "bad response"])


@pytest.mark.parametrize("evaluations", [None, 123])
def test_diagnostics_leave_invalid_submission_for_tool_to_reject(tmp_path, evaluations):
    model = GenRMModel("http://genrm")
    model.client.close()
    model.audit_path = tmp_path / "events.jsonl"
    action = {"name": "submit_grades", "evaluations": evaluations, "ranking": [["private-code"]]}
    model.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"response": json.dumps(action)}))
    )
    try:
        message = model.query([{"role": "user", "content": "task"}])
        assert message["extra"]["actions"] == [action]
        assert "private-code" not in model.audit_path.read_text()
        # A full/unwritable diagnostics location must not change the response.
        model.audit_path = tmp_path
        assert model.query([{"role": "user", "content": "task"}])["extra"]["actions"] == [action]
    finally:
        model.client.close()


@pytest.mark.parametrize("command", ["print(1)\nprint(2)", r"grep 'a\|b' file.py"])
def test_escaped_tool_command_is_preserved(command):
    action = {"name": "bash", "rollout_n": 0, "command": command}
    assert parse_action(json.dumps(action)) == action


def test_inline_history_survives_original_genrm_message_schema():
    genrm = pytest.importorskip("relax.components.genrm")
    action = '{"name":"bash","command":"python benchmark.py"}'
    messages = [
        {"role": "user", "content": "Grade candidates"},
        {"role": "assistant", "content": "<think>\nCheck complexity\n</think>\n\n" + action},
        {"role": "user", "content": "Tool result: 7.6 seconds"},
    ]
    request = genrm.GenerateRequest.model_validate_json(json.dumps({"messages": messages}))
    assert request.model_dump(exclude_none=True)["messages"] == messages
