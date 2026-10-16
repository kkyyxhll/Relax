# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from types import SimpleNamespace

import pytest


pytest.importorskip("minisweagent")
from examples.gagar.minisweagent import agent_server as server
from examples.gagar.minisweagent.bridge import SandboxBridge


@pytest.mark.parametrize(
    "retain, reward, failure", [(False, 1, False), (True, 1, False), (True, 0, False), (True, 0, True)]
)
def test_session_transfers_ownership_only_after_success(monkeypatch, tmp_path, retain, reward, failure):
    created = []

    class Environment:
        def __init__(self, **kwargs):
            self.cleaned = 0
            created.append(self)

        def cleanup(self):
            self.cleaned += 1

    class Agent:
        def __init__(self, *a, **kw):
            self.grading_evidence = {"patch": "original"}

        def run(self, task):
            if failure:
                raise RuntimeError("test infrastructure broke")
            return {"reward": reward, "n_calls": 3, "exit_status": "Submitted" if reward else "LimitsExceeded"}

    monkeypatch.setattr(server, "ApptainerInstanceEnvironment", Environment)
    monkeypatch.setattr(server, "MyLitellmModel", lambda **kw: object())
    monkeypatch.setattr(server, "MyAgent", Agent)
    instance = server.AgentServer.__new__(server.AgentServer)
    instance.lock = threading.Lock()
    record = server.SessionRecord("client", "g", 0, "train", "http://model", "key")
    instance.records = {"s": record}
    instance.modes = {"train": SimpleNamespace(get=lambda _: {"docker_image": "image", "problem_statement": "task"})}
    instance.tmp_base = tmp_path / "tmp"
    instance.work_dir = tmp_path
    instance.base_config = {"model": {}, "environment": {}, "agent": {}}
    instance.r2e_config = {
        "wheel_mount": "/wheels",
        "setup_commands": {"train": "setup"},
        "reward_commands": {"train": "reward"},
    }
    instance.sifs = {"image": tmp_path / "image.sif"}
    instance.sif_cache = None
    instance._sif_leases = {}
    instance._pending_sif_cleanup = set()
    instance.sif_dir = instance.wheel_dir = tmp_path
    instance.grading_bridge = SandboxBridge(tmp_path / "grading", [], lambda *a, **kw: None) if retain else None
    instance._run_session("s", 0)
    assert record.status == ("failed" if failure else "completed")
    assert record.env is None
    assert created[0].cleaned == int(not retain or failure)
    if retain:
        if not failure:
            assert instance.grading_bridge.candidates["s"].evidence["reward"] == reward
            assert instance._public_record("s", record)["gagar"]["session_id"] == "s"
        instance.grading_bridge.shutdown()
        assert created[0].cleaned == 1


def test_grader_process_cancellation_uses_tracked_child_cleanup():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        server._tracked_run(["sleep", "5"], timeout=10, cancel_event=cancel)
    assert not server._owned_child_pids


def test_cancel_running_sibling_releases_retained_group_and_rejects_late_retention(tmp_path):
    bridge = SandboxBridge(tmp_path / "grading", [], lambda *a, **kw: None)
    env = SimpleNamespace(cleanup=lambda: None)
    scratch = tmp_path / "s0"
    scratch.mkdir()
    bridge.retain("s0", group_id="g", task="task", env=env, tmp_dir=scratch, evidence={})
    instance = server.AgentServer.__new__(server.AgentServer)
    instance.lock = threading.Lock()
    record = server.SessionRecord("client", "g", 0, "train", "http://model", "key", status="running")
    instance.records = {"s1": record}
    instance.grading_bridge = bridge
    instance.cancel_session("s1")
    assert not bridge.candidates and record.cancel_event.is_set()
    assert not bridge.retain("s1", group_id="g", task="task", env=env, tmp_dir=tmp_path, evidence={})
    bridge.shutdown()


@pytest.mark.parametrize("capture_stage", ["baseline", "patch"])
def test_evidence_failure_does_not_skip_original_reward(monkeypatch, capture_stage):
    calls = []

    class Environment:
        def execute(self, action, **kwargs):
            command = action["command"]
            calls.append(command)
            if command in {"setup", "reward"}:
                return {"returncode": 0, "output": "tests passed"}
            if capture_stage == "baseline" or command == "git diff --binary HEAD":
                raise RuntimeError("capture failed")
            return {"returncode": 0, "output": "a" * 40 + "\n" + "b" * 40 + "\n"}

    agent = server.MyAgent(
        object(),
        Environment(),
        rollout_mode="train",
        row={},
        setup_command="setup",
        reward_command="reward",
        capture_evidence=True,
        system_template="system",
        instance_template="{{task}}",
    )
    monkeypatch.setattr(server.DefaultAgent, "run", lambda *a, **kw: {"exit_status": "Submitted"})
    monkeypatch.setattr(server, "_r2e_reward", lambda row, output: 1.0)
    result = agent.run("task")
    assert result["reward"] == 1.0
    assert "reward" in calls
    assert "capture_error" in agent.grading_evidence
    assert "patch" not in agent.grading_evidence
    assert agent.grading_evidence["test_output"] == "tests passed"
