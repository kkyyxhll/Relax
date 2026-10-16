# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from examples.gagar.minisweagent.bridge import GradingJob, SandboxBridge
from examples.gagar.minisweagent.group_rm import reward_func


class FakeEnvironment:
    def __init__(self):
        self.calls = []
        self.cleanups = 0

    def execute(self, action, **kwargs):
        self.calls.append((action, kwargs))
        return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nold submission", "returncode": 0}

    def cleanup(self):
        self.cleanups += 1


@pytest.fixture
def bridge(tmp_path):
    bridge = SandboxBridge(tmp_path / "grading", [], lambda *a, **k: None)
    yield bridge
    bridge.shutdown()


def retain(bridge, tmp_path, sid="s0", group="g", task="fix bug", reward=1):
    env = FakeEnvironment()
    scratch = tmp_path / sid
    scratch.mkdir(exist_ok=True)
    assert bridge.retain(sid, group_id=group, task=task, env=env, tmp_dir=scratch, evidence={"reward": reward})
    return env


def lease(bridge, session_ids=None):
    job = GradingJob(session_ids=session_ids or ["s0"], expires_at=time.monotonic() + 100)
    bridge.jobs["g"] = job
    return job


def test_routes_only_to_selected_candidate_and_disables_doer_submission_check(bridge, tmp_path):
    env0 = retain(bridge, tmp_path)
    env1 = retain(bridge, tmp_path, "s1")
    job = lease(bridge, ["s0", "s1"])
    result = bridge.execute("g", job.token, {"rollout_n": 1, "command": "git diff"})
    assert "COMPLETE_TASK" in result["output"]
    assert env0.calls == []
    assert env1.calls == [({"command": "git diff"}, {"timeout": 120, "check_finished": False})]


@pytest.mark.parametrize(
    "action",
    [
        {"command": "pwd"},
        {"rollout_n": -1, "command": "pwd"},
        {"rollout_n": 1, "command": "pwd"},
        {"rollout_n": True, "command": "pwd"},
        {"rollout_n": 0, "command": "pwd", "endpoint": "evil"},
    ],
)
def test_bad_tool_arguments_never_execute(bridge, tmp_path, action):
    env = retain(bridge, tmp_path)
    job = lease(bridge)
    with pytest.raises(ValueError):
        bridge.execute("g", job.token, action)
    assert not env.calls


def test_wrong_capability_and_released_lease_never_execute(bridge, tmp_path):
    env = retain(bridge, tmp_path)
    job = lease(bridge)
    with pytest.raises(ValueError):
        bridge.execute("g", "wrong", {"rollout_n": 0, "command": "pwd"})
    bridge.release("g")
    bridge.release("g")
    with pytest.raises(ValueError):
        bridge.execute("g", job.token, {"rollout_n": 0, "command": "pwd"})
    assert env.cleanups == 1 and not env.calls


def test_group_roster_rejects_cross_task_candidates(bridge, tmp_path):
    retain(bridge, tmp_path)
    retain(bridge, tmp_path, "other", group="another")
    with pytest.raises(ValueError, match="outside"):
        bridge.start({"group_id": "g", "session_ids": ["s0", "other"]})
    assert not bridge.jobs


def test_lease_extends_past_terminal_record_ttl_and_reclaims_dropped_groups(bridge, tmp_path):
    active = retain(bridge, tmp_path)
    orphan = retain(bridge, tmp_path, "orphan", group="dropped")
    bridge.candidates["s0"].expires_at = 0
    bridge.candidates["orphan"].expires_at = 0
    job = lease(bridge)
    bridge.reap()
    assert active.cleanups == 0 and orphan.cleanups == 1
    job.expires_at = 0
    bridge.reap()
    assert active.cleanups == 1


def test_cleanup_failure_keeps_ownership_and_scratch_for_retry(bridge, tmp_path):
    env = retain(bridge, tmp_path)
    lease(bridge)

    def fail():
        raise RuntimeError("stop failed")

    env.cleanup = fail
    with pytest.raises(RuntimeError):
        bridge.release("g")
    assert "s0" in bridge.candidates and (tmp_path / "s0").exists()
    env.cleanup = lambda: None
    bridge.reap()
    assert not bridge.candidates and not (tmp_path / "s0").exists()


def test_release_waits_for_inflight_command(bridge, tmp_path):
    env = retain(bridge, tmp_path)
    job = lease(bridge)
    entered, finish = threading.Event(), threading.Event()

    def execute(*a, **kw):
        entered.set()
        assert finish.wait(3)
        assert env.cleanups == 0
        return {"output": "ok"}

    env.execute = execute
    worker = threading.Thread(target=bridge.execute, args=("g", job.token, {"rollout_n": 0, "command": "test"}))
    worker.start()
    assert entered.wait(3)
    cleanup = threading.Thread(target=bridge.release, args=("g",))
    cleanup.start()
    finish.set()
    worker.join(3)
    cleanup.join(3)
    assert not worker.is_alive() and not cleanup.is_alive() and env.cleanups == 1


def test_capacity_does_not_block_siblings(bridge, tmp_path):
    bridge.max_retained_groups = 1
    retain(bridge, tmp_path)
    assert not bridge.retain(
        "other", group_id="other", task="task", env=FakeEnvironment(), tmp_dir=tmp_path, evidence={}
    )
    retain(bridge, tmp_path, "s1")


def test_group_rm_preserves_order_and_releases_on_success_or_failure(monkeypatch):
    import relax.utils.utils as utils

    monkeypatch.setattr(utils, "get_serve_url", lambda _: "http://genrm")
    original_client = httpx.AsyncClient
    for failed in (False, True):
        requests = []

        def respond(request):
            requests.append(request)
            if request.url.path.endswith("release"):
                return httpx.Response(204)
            if failed:
                return httpx.Response(500)
            return httpx.Response(
                200, json={"status": "completed", "result": {"weights": [0.6, 1, 0], "rewards": [1, 1, 0]}}
            )

        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw)
        )
        samples = [
            SimpleNamespace(
                reward=r, metadata={"gagar": {"group_id": "g", "session_id": sid, "server_url": "http://server"}}
            )
            for r, sid in [(1, "s2"), (1, "s0"), (0, "s1")]
        ]
        result = asyncio.run(reward_func(SimpleNamespace(), samples))
        assert result[0]["test_reward"] == 1
        if failed:
            assert [r["score"] for r in result] == [1, 1, 0]
            assert all(r["grader_fallback"] for r in result)
        else:
            assert result[1]["score"] > result[0]["score"]
            import json

            assert json.loads(requests[0].content)["session_ids"] == ["s2", "s0", "s1"]
        assert requests[-1].url.path.endswith("/release")


def test_eval_keeps_binary_rewards_without_http():
    samples = [SimpleNamespace(reward=r, metadata={}) for r in [1, 0]]
    assert [r["score"] for r in asyncio.run(reward_func(SimpleNamespace(), samples))] == [1, 0]


def test_one_cleanup_failure_does_not_skip_other_candidates(bridge, tmp_path):
    bad = retain(bridge, tmp_path)
    good = retain(bridge, tmp_path, "s1")

    def fail():
        raise RuntimeError("stop failed")

    bad.cleanup = fail
    with pytest.raises(RuntimeError):
        bridge.release("g")
    assert good.cleanups == 1
    assert set(bridge.candidates) == {"s0"}
    bad.cleanup = lambda: None
    bridge.reap()
    assert not bridge.candidates


def test_manifest_creation_failure_finishes_job_and_releases(bridge, tmp_path):
    env = retain(bridge, tmp_path)
    bridge.work_dir.write_text("not a directory")
    job = lease(bridge)
    bridge._run("g", job, {"token": "secret"})
    assert job.status == "failed"
    assert env.cleanups == 1 and not bridge.candidates


def test_incomplete_passing_evidence_skips_grader_and_releases_group(bridge, tmp_path):
    import json

    env = retain(bridge, tmp_path)
    job = lease(bridge)
    calls = []
    bridge.run_process = lambda *a, **kw: calls.append(a)
    manifest = {
        "token": job.token,
        "candidates": [{"rollout_n": 0, "reward": 1, "capture_error": "baseline: failed"}],
    }
    bridge._run("g", job, manifest)
    assert not calls
    assert job.status == "failed" and "incomplete evidence" in job.error
    assert env.cleanups == 1 and not bridge.candidates
    saved = json.loads(next(bridge.work_dir.glob("*/input.json")).read_text())
    assert "token" not in saved
    assert saved["candidates"][0]["capture_error"] == "baseline: failed"
