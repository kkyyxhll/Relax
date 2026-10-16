# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from examples.gagar.code_generate.scoring import grade_to_weights
from examples.gagar.gagar import compute_group_rewards


def grading(weights):
    return {"evaluations": [{"rollout_n": i, "weight": w, "reason": "Inspected code"} for i, w in enumerate(weights)]}


def test_code_weights_are_independent_and_reach_shared_cap_formula():
    rewards = [1, 1, 1, 0]
    result = grade_to_weights(rewards, grading([0.3, 0.3, 0.4]))
    assert result["rewards"] == rewards
    assert result["weights"] == [0.3, 0.3, 0.4, 0]
    scores = compute_group_rewards(rewards, result["weights"])
    assert scores == pytest.approx([0.8625, 0.8625, 0.9, 0])
    mean = sum(scores) / 4
    assert [s - mean for s in scores] == pytest.approx([0.20625, 0.20625, 0.24375, -0.65625])
    assert grade_to_weights([1, 1, 0], grading([0.9, 0.9]))["weights"] == [0.9, 0.9, 0]
    assert grade_to_weights([1, 0], grading([0]))["weights"] == [0, 0]
    assert grade_to_weights([0, 0], grading([]))["weights"] == [0, 0]


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, "0.8"])
def test_model_weight_validation(bad):
    with pytest.raises(ValueError, match="weight"):
        grade_to_weights([1, 0], grading([bad]))


def test_model_cannot_omit_duplicate_or_regrade_failed_candidates():
    for value in [grading([]), grading([0.8, 0.7]), {"evaluations": grading([0.8])["evaluations"] * 2}]:
        with pytest.raises(ValueError):
            grade_to_weights([1, 0], value)


def test_proxy_only_forwards_messages_to_fixed_genrm(tmp_path):
    from examples.gagar.code_generate.sandbox import GenRMProxy

    proxy = GenRMProxy(tmp_path / "genrm.sock", "http://fixed-genrm")
    proxy.client.close()
    requests = []

    def generate(request):
        requests.append(request)
        return httpx.Response(200, json={"response": "answer"})

    proxy.client = httpx.Client(transport=httpx.MockTransport(generate))
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(tmp_path / "genrm.sock"))) as client:
            assert client.post("http://local/arbitrary", json={"messages": []}).status_code == 400
            assert client.get("http://local/generate").status_code == 501
            messages = [
                {"role": "user", "content": "grade"},
                {"role": "assistant", "content": "<think>check evidence</think>action"},
            ]
            result = client.post(
                "http://local/generate", json={"messages": messages, "url": "http://other", "command": "bad"}
            )
            assert result.json() == {"response": "answer"}
            assert len(requests) == 1
            assert str(requests[0].url) == "http://fixed-genrm/generate"
            assert json.loads(requests[0].content) == {"messages": messages}
    finally:
        proxy.close()


def test_no_host_environment_or_bind_injection(monkeypatch, tmp_path):
    from examples.gagar.code_generate.sandbox import process_environment

    monkeypatch.setenv("APPTAINER_BIND", "/:/host")
    monkeypatch.setenv("APPTAINERENV_TOKEN", "secret")
    monkeypatch.setenv("https_proxy", "http://proxy")
    env = process_environment(tmp_path)
    assert set(env) == {"PATH", "HOME", "TMPDIR", "LANG"}


@pytest.fixture
def host_runtime(monkeypatch):
    pytest.importorskip("minisweagent")
    from examples.gagar.code_generate import apptainer as agent_server
    from examples.gagar.code_generate import sandbox

    monkeypatch.setattr(agent_server, "_install_child_subreaper_and_reaper", lambda: None)
    monkeypatch.setattr(agent_server, "_drain_orphans", lambda: None)
    return sandbox, agent_server


def test_cleanup_failure_retains_lease_for_parent_retry(host_runtime, monkeypatch, tmp_path):
    sandbox, server = host_runtime
    names = iter(["mswe-123-abc"])
    monkeypatch.setattr(server, "_new_instance_name", lambda: next(names))

    def fail():
        raise RuntimeError("transient stop error")

    monkeypatch.setattr(server, "ApptainerInstanceEnvironment", lambda **kw: SimpleNamespace(cleanup=fail))
    leases = tmp_path / "leases"
    with pytest.raises(sandbox.SandboxCleanupError):
        with sandbox.scratch_directory(str(tmp_path)) as scratch:
            with sandbox.instance("image", scratch, {}, leases):
                pass
    assert scratch.exists()
    assert (leases / "mswe-123-abc").exists()
    stops = []
    monkeypatch.setattr(server, "_stop_apptainer_instance", lambda *args: stops.append(args[-1]) or "")
    sandbox.cleanup_leases(leases, 999)
    assert not stops and scratch.exists()
    sandbox.cleanup_leases(leases, 123)
    assert stops == ["mswe-123-abc"]
    assert not scratch.exists() and not list(leases.iterdir())


def test_constructor_failure_stops_partial_instance(host_runtime, monkeypatch, tmp_path):
    sandbox, server = host_runtime
    monkeypatch.setattr(server, "_new_instance_name", lambda: "mswe-123-partial")

    def fail(**kwargs):
        raise RuntimeError("start failed")

    monkeypatch.setattr(server, "ApptainerInstanceEnvironment", fail)
    stops = []
    monkeypatch.setattr(server, "_stop_apptainer_instance", lambda *args: stops.append(args[-1]) or "")
    with pytest.raises(RuntimeError, match="start failed"):
        with sandbox.instance("image", tmp_path, {}, tmp_path / "leases"):
            pytest.fail("must not run commands")
    assert stops == ["mswe-123-partial"]
    assert not list((tmp_path / "leases").iterdir())


def test_official_judge_error_propagates_and_cleans_scratch(host_runtime, monkeypatch, tmp_path):
    sandbox, _ = host_runtime
    monkeypatch.setenv("GAGAR_SANDBOX_IMAGE", "image")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("GAGAR_SANDBOX_TMPDIR", str(scratch))
    closed = []

    @contextmanager
    def fake_instance(*args):
        try:
            yield object()
        finally:
            closed.append(True)

    monkeypatch.setattr(sandbox, "instance", fake_instance)
    monkeypatch.setattr(
        sandbox,
        "execute",
        lambda *args: json.dumps({"accepted": False, "tests": [{"exec_info": {"status": "SandboxError"}}]}),
    )
    with pytest.raises(RuntimeError, match="infrastructure"):
        sandbox.run_group(
            {"task_id": "t", "label": "{}", "responses": ["</think>```python\nprint(1)\n```"]},
            tmp_path / "result.json",
        )
    assert closed == [True]
    assert not (tmp_path / "result.json").exists()
    assert not list(scratch.iterdir())


def test_grader_cannot_execute_bash_when_started_on_host(monkeypatch):
    pytest.importorskip("minisweagent")
    from examples.gagar.code_generate.sandbox_env import SandboxEnvironment

    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    with pytest.raises(RuntimeError, match="inside"):
        SandboxEnvironment({"candidates": []})


def test_sandbox_tool_rejects_bridge_arguments_without_executing(monkeypatch):
    pytest.importorskip("minisweagent")
    from examples.gagar.code_generate import sandbox_env

    env = sandbox_env.SandboxEnvironment.__new__(sandbox_env.SandboxEnvironment)
    monkeypatch.setattr(sandbox_env.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not execute"))
    assert env.execute({"name": "bash", "rollout_n": 0, "command": "pwd"})["returncode"] == -1
    assert env.execute({"name": "bash", "endpoint": "http://other", "command": "pwd"})["returncode"] == -1


def test_parent_retries_cleanup_worker_without_touching_other_owners(monkeypatch, tmp_path):
    from examples.gagar.code_generate import sandbox

    leases = tmp_path / "instances"
    leases.mkdir()
    (leases / "mswe-123-abc").touch()
    calls = []

    class Process:
        def __init__(self, code):
            self.returncode = code

        async def wait(self):
            return self.returncode

    async def spawn(*args, **kwargs):
        calls.append(args)
        return Process(1 if len(calls) == 1 else 0)

    monkeypatch.setattr(sandbox.asyncio, "create_subprocess_exec", spawn)
    asyncio.run(sandbox.finish_worker(SimpleNamespace(returncode=1, pid=123), "python", tmp_path, None))
    assert len(calls) == 2
    assert all(args[-2:] == ("--cleanup-owner", "123") for args in calls)


def test_parent_cleans_scratch_after_instance_lease_is_already_gone(tmp_path):
    from examples.gagar.code_generate import sandbox

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "artifact").write_text("left while model request was closing")
    (tmp_path / "scratch.json").write_text(json.dumps({"owner": 123, "path": str(scratch)}))
    sandbox.cleanup_scratch(tmp_path, 999)
    assert scratch.exists()
    asyncio.run(sandbox.finish_worker(SimpleNamespace(returncode=-9, pid=123), "unused", tmp_path, None))
    assert not scratch.exists()
    assert not (tmp_path / "scratch.json").exists()


def test_grader_manifest_excludes_thinking_but_host_audit_preserves_it(host_runtime, monkeypatch, tmp_path):
    sandbox, _ = host_runtime
    monkeypatch.setenv("GAGAR_SANDBOX_IMAGE", "image")
    monkeypatch.setenv("GAGAR_SANDBOX_TMPDIR", str(tmp_path))
    manifests = []

    @contextmanager
    def fake_instance(image, root, mounts, leases):
        input_dir = next(path for path, destination in mounts.items() if destination == "/input")
        if (input_dir / "manifest.json").exists():
            manifests.append(json.loads((input_dir / "manifest.json").read_text()))
        yield input_dir

    def execute(input_dir, argv, timeout):
        if argv[0] == "/opt/sf/bin/python":
            return json.dumps({"accepted": True, "tests": []})
        if argv == ["/bin/cat", "/workspace/result.json"]:
            return json.dumps(grading([0.9]))
        return "{}"

    monkeypatch.setattr(sandbox, "instance", fake_instance)
    monkeypatch.setattr(sandbox, "execute", execute)
    monkeypatch.setattr(sandbox, "GenRMProxy", lambda *a: SimpleNamespace(calls=1, close=lambda: None))
    sandbox.run_group(
        {
            "task_id": "t",
            "task": "Print 1.",
            "label": "{}",
            "responses": ["<think>PRIVATE_REASONING</think>```python\nprint(1)\n```"],
            "genrm_url": "http://genrm",
        },
        tmp_path / "result.json",
    )
    candidate = manifests[0]["candidates"][0]
    assert candidate["code"] == "print(1)" and candidate["reward"] == 1
    assert candidate["test_output"]["accepted"]
    assert "reason" not in candidate and "submission" not in candidate
    assert "PRIVATE_REASONING" not in json.dumps(manifests)
    assert "PRIVATE_REASONING" in (tmp_path / "outcomes.json").read_text()


def test_code_grader_evidence_tool_never_returns_original_thinking():
    pytest.importorskip("minisweagent")
    from examples.gagar.code_generate.sandbox_env import SandboxEnvironment

    env = SandboxEnvironment.__new__(SandboxEnvironment)
    env.manifest = {"candidates": [{"rollout_n": 0, "code": "print(1)", "reason": "PRIVATE_REASONING"}]}
    for section in ("reason", "submission"):
        result = env.execute({"name": "read_evidence", "rollout_n": 0, "section": section})
        assert result["returncode"] == -1
        assert "PRIVATE_REASONING" not in result["output"]
    assert "print(1)" in env.execute({"name": "read_evidence", "rollout_n": 0, "section": "code"})["output"]


@pytest.mark.parametrize("unavailable", [None, "missing", "disk_full"])
@pytest.mark.parametrize("grader_fails", [False, True])
def test_grader_diagnostics_survive_sandbox_cleanup(host_runtime, monkeypatch, tmp_path, unavailable, grader_fails):
    from pathlib import Path

    sandbox, _ = host_runtime
    monkeypatch.setenv("GAGAR_SANDBOX_IMAGE", "image")
    monkeypatch.setenv("GAGAR_SANDBOX_TMPDIR", str(tmp_path))
    closed = []
    artifacts = {
        "trajectory.json": '{"messages": []}',
        "events.jsonl": '{"event": "query"}\n',
        "summary.json": '{"api_calls": 1}',
    }

    @contextmanager
    def fake_instance(*args):
        try:
            yield object()
        finally:
            closed.append(True)

    def execute(env, argv, timeout):
        if argv[0] == "/opt/sf/bin/python":
            return json.dumps({"accepted": True, "tests": []})
        if argv[0] == "/opt/grader/bin/python":
            if grader_fails:
                raise RuntimeError("grader failed")
            return ""
        name = Path(argv[1]).name
        if name == "result.json":
            return json.dumps(grading([0.9]))
        if name == "events.jsonl" and unavailable == "missing":
            raise RuntimeError("file not found")
        return artifacts[name]

    original_write = Path.write_text

    def write(path, text, *args, **kwargs):
        if path == tmp_path / "events.jsonl" and unavailable == "disk_full":
            raise OSError("audit disk full")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(sandbox, "instance", fake_instance)
    monkeypatch.setattr(sandbox, "execute", execute)
    monkeypatch.setattr(sandbox, "GenRMProxy", lambda *a: SimpleNamespace(calls=1, close=lambda: None))
    sandbox.run_group(
        {
            "task_id": "t",
            "task": "Print 1.",
            "label": "{}",
            "responses": ["</think>```python\nprint(1)\n```"],
            "genrm_url": "http://genrm",
        },
        tmp_path / "result.json",
    )
    assert closed == [True, True]
    for name, text in artifacts.items():
        if name == "events.jsonl" and unavailable:
            assert not (tmp_path / name).exists()
        else:
            assert (tmp_path / name).read_text() == text
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["fallback"] is grader_fails
    assert result["error"] == ("grader failed" if grader_fails else None)
    assert result["grading"] == (None if grader_fails else grading([0.9]))


@pytest.mark.parametrize("subset", ["primeintellect", "taco", "lcbv5"])
def test_prepared_task_preserves_problem_and_starter(subset):
    from examples.gagar.code_generate.prepare_data import convert_row

    row = {
        "problem": "Print the literal text <think>.",
        "tests": json.dumps({"inputs": ["1"], "outputs": ["2"]})
        if subset == "taco"
        else json.dumps([{"input": "1", "output": "2"}]),
        "starter_code": "class Solution:\n    def solve(self, n): ...",
        "metadata": {"func_name": "solve"},
    }
    converted = convert_row(row, subset, 0)
    task = converted["metadata"]["task"]
    assert task == converted["input"][0]["content"]
    assert row["problem"] in task
    if subset == "lcbv5":
        assert row["starter_code"] in task


@pytest.mark.parametrize("has_raw_task", [False, True])
def test_grader_receives_raw_task_metadata(monkeypatch, tmp_path, has_raw_task):
    from pathlib import Path

    import relax.utils.utils as utils
    from examples.gagar.code_generate import sandbox

    raw_task = "Print the literal text <think>."
    actor_prompt = f"<|im_start|>user\n{raw_task}<|im_end|>\n<|im_start|>assistant\n<think>"
    metadata = {"task_id": "t"}
    if has_raw_task:
        metadata["task"] = raw_task
    samples = [
        SimpleNamespace(metadata=dict(metadata), prompt=actor_prompt, label="{}", response="answer") for _ in range(2)
    ]
    payloads = []

    class Process:
        returncode = 0

        async def wait(self):
            return self.returncode

    async def spawn(*args, **kwargs):
        payloads.append(json.loads(Path(args[args.index("--input") + 1]).read_text()))
        Path(args[args.index("--output") + 1]).write_text(
            json.dumps({"original": [1, 0], "grading": None, "fallback": True})
        )
        return Process()

    async def finish(*args):
        pass

    monkeypatch.setenv("GAGAR_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("MINI_SWE_AGENT_VENV", str(tmp_path / "venv"))
    monkeypatch.setattr(utils, "get_serve_url", lambda _: "http://genrm")
    monkeypatch.setattr(sandbox.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(sandbox, "finish_worker", finish)
    records = asyncio.run(sandbox.reward_group(SimpleNamespace(), samples))
    assert payloads[0]["task"] == (raw_task if has_raw_task else actor_prompt)
    assert [record["score"] for record in records] == [1, 0]
