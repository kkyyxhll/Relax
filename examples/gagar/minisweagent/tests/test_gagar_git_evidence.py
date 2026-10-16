# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import subprocess
from types import SimpleNamespace

import pytest

from examples.gagar.minisweagent.git_evidence import capture_baseline, capture_patch


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "base.py").write_text("original\n")
    (tmp_path / "same.py").write_text("original\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "base")
    (tmp_path / "base.py").write_text("image-only change\n")
    (tmp_path / "same.py").write_text("image version\n")
    (tmp_path / "pre_staged.py").write_text("pre-staged image file\n")
    git(tmp_path, "add", "pre_staged.py")
    return tmp_path


class Environment:
    def __init__(self, repo):
        self.repo = repo

    def execute(self, action, **kwargs):
        result = subprocess.run(
            ["bash", "-c", action["command"]], cwd=self.repo, capture_output=True, text=True, timeout=10
        )
        return {"output": result.stdout + result.stderr, "returncode": result.returncode}


def test_dirty_image_is_not_an_actor_change_and_index_is_preserved(repo):
    env, evidence = Environment(repo), {}
    index = (repo / ".git/index").read_bytes()
    head = git(repo, "rev-parse", "HEAD")
    capture_baseline(env, evidence)
    assert (repo / ".git/index").read_bytes() == index
    capture_patch(env, evidence)
    assert "capture_error" not in evidence
    assert evidence["patch"]["output"] == ""
    assert "image-only change" in evidence["baseline"]["patch"]["output"]
    assert "pre-staged image file" in evidence["baseline"]["patch"]["output"]
    assert "image-only change" in evidence["head_patch"]["output"]
    assert git(repo, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("operation", ["edit", "revert", "delete", "rename", "staged_and_unstaged", "new"])
def test_patch_tracks_changes_relative_to_post_setup_tree(repo, operation):
    env, evidence = Environment(repo), {}
    capture_baseline(env, evidence)
    if operation == "edit":
        (repo / "same.py").write_text("actor edit\n")
    elif operation == "revert":
        (repo / "same.py").write_text("original\n")
    elif operation == "delete":
        (repo / "same.py").unlink()
    elif operation == "rename":
        git(repo, "mv", "same.py", "renamed.py")
    elif operation == "staged_and_unstaged":
        (repo / "same.py").write_text("staged\n")
        git(repo, "add", "same.py")
        (repo / "same.py").write_text("staged\nunstaged\n")
    else:
        (repo / "new.py").write_text("staged actor file\n")
        git(repo, "add", "new.py")
        (repo / "untracked.py").write_text("not submitted\n")
    capture_patch(env, evidence)
    assert "capture_error" not in evidence
    patch = evidence["patch"]["output"]
    assert "base.py" not in patch and "pre_staged.py" not in patch
    assert "untracked.py" not in patch
    # Apply the frozen patch to the independent baseline tree and compare file contents.
    reference = repo / "reference"
    reference.mkdir()
    archive = subprocess.run(
        ["git", "archive", evidence["baseline"]["tree"]], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["tar", "xf", "-", "-C", str(reference)], input=archive.stdout, check=True)
    subprocess.run(["git", "init"], cwd=reference, check=True, capture_output=True)
    subprocess.run(["git", "apply", "-"], cwd=reference, input=patch, text=True, check=True, capture_output=True)
    for name in ("same.py", "renamed.py", "new.py", "base.py", "pre_staged.py"):
        actual, expected = reference / name, repo / name
        assert actual.exists() == expected.exists()
        if actual.exists():
            assert actual.read_bytes() == expected.read_bytes()


def test_nonzero_capture_is_reported_without_using_head_patch():
    env = SimpleNamespace(execute=lambda *a, **kw: {"returncode": 128, "output": "not a repository"})
    evidence = {}
    capture_baseline(env, evidence)
    capture_patch(env, evidence)
    assert "capture_error" in evidence and "patch" not in evidence


def test_capture_does_not_swallow_keyboard_interrupt():
    def execute(*a, **kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        capture_baseline(SimpleNamespace(execute=execute), {})


def test_agent_freezes_post_setup_to_pre_test_changes(monkeypatch, repo):
    pytest.importorskip("minisweagent")
    from examples.gagar.minisweagent import agent_server as server

    class AgentEnvironment(Environment):
        def execute(self, action, **kwargs):
            if action["command"] == "setup":
                (repo / "same.py").write_text("setup state\n")
                return {"returncode": 0}
            if action["command"] == "reward":
                (repo / "same.py").write_text("test aftermath\n")
                return {"returncode": 0, "output": "tests passed"}
            return super().execute(action, **kwargs)

    def rollout(*args, **kwargs):
        (repo / "same.py").write_text("actor solution\n")
        return {"exit_status": "Submitted"}

    monkeypatch.setattr(server.DefaultAgent, "run", rollout)
    monkeypatch.setattr(server, "_r2e_reward", lambda row, output: 1.0)
    agent = server.MyAgent(
        object(),
        AgentEnvironment(repo),
        rollout_mode="train",
        row={},
        setup_command="setup",
        reward_command="reward",
        capture_evidence=True,
        system_template="system",
        instance_template="{{task}}",
    )
    result = agent.run("task")
    assert result["reward"] == 1.0
    patch = agent.grading_evidence["patch"]["output"]
    assert "-setup state" in patch and "+actor solution" in patch
    assert "base.py" not in patch and "test aftermath" not in patch
    assert (repo / "same.py").read_text() == "test aftermath\n"


def test_baseline_preserves_racy_index_timestamp(repo):
    import os

    # SIF images can preserve a same-size dirty file with the index's stat data.
    # Git hashes it only while the index timestamp still marks it as racy.
    git(repo, "config", "core.trustctime", "false")
    path = repo / "same.py"
    path.write_text("before\n")
    # Use an older whole second: some Git builds compare index mtimes in seconds.
    earlier = path.stat().st_mtime_ns - 10_000_000_000
    os.utime(path, ns=(earlier, earlier))
    git(repo, "add", "same.py")
    stamp = path.stat()
    path.write_text("after!\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    os.utime(repo / ".git/index", ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    evidence = {}
    capture_baseline(Environment(repo), evidence)
    assert "capture_error" not in evidence
    assert git(repo, "show", evidence["baseline"]["tree"] + ":same.py") == "after!\n"
