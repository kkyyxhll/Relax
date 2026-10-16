# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("minisweagent")
from examples.gagar.bridge import SandboxBridge
from examples.mini_swe_agent import agent_server as server
from examples.mini_swe_agent.sif_cache import SifCache


@pytest.fixture
def session(monkeypatch, tmp_path):
    created, stopped = [], []
    state = SimpleNamespace(constructor_error=False, stop_error="")

    def build(env):
        created.append(env)
        if state.constructor_error:
            raise RuntimeError("constructor failed after start")
        return f"instance://{env.instance_name}"

    def stop(*args):
        stopped.append(args[-1])
        return state.stop_error

    class Agent:
        grading_evidence = {"patch": "original"}

        def __init__(self, *args, **kwargs):
            pass

        def run(self, task):
            return {"reward": 1, "n_calls": 2}

    monkeypatch.setattr(server.ApptainerInstanceEnvironment, "_build_sandbox", build)
    monkeypatch.setattr(server, "_stop_apptainer_instance", stop)
    monkeypatch.setattr(server, "MyLitellmModel", lambda **kw: object())
    monkeypatch.setattr(server, "MyAgent", Agent)
    instance = server.AgentServer.__new__(server.AgentServer)
    instance.lock = threading.Lock()
    record = server.SessionRecord("client", "g", 0, "train", "http://model", "key")
    instance.records = {"s": record}
    instance.modes = {"train": SimpleNamespace(get=lambda _: {"docker_image": "image", "problem_statement": "task"})}
    instance.tmp_base = tmp_path / "tmp"
    instance.work_dir = tmp_path
    instance.base_config = {
        "model": {},
        "environment": {"image": "image", "executable": "fake-apptainer"},
        "agent": {},
    }
    instance.r2e_config = {
        "wheel_mount": "/wheels",
        "setup_commands": {"train": "setup"},
        "reward_commands": {"train": "reward"},
    }
    instance.apptainer_args = ("fake-apptainer", [])
    instance.sif_cache = SifCache(sif_dir=tmp_path / "sifs", build_tmp_root=tmp_path / "build")
    (instance.sif_cache.sif_dir / "image.sif").write_bytes(b"sif")
    instance._sif_leases = {}
    instance._pending_sif_cleanup = set()
    instance.wheel_dir = tmp_path
    instance.grading_bridge = None
    yield instance, record, state, created, stopped
    # Never let a deferred destructor reach a real apptainer after mocks revert.
    for env in created:
        env.instance_name = ""
        env._release_sif = None
    if instance.grading_bridge:
        instance.grading_bridge.shutdown()
    for name in list(instance._sif_leases):
        instance._release_sif(name)


def test_grading_retains_sif_until_sandbox_stop(session, tmp_path):
    instance, record, state, created, stopped = session
    instance.grading_bridge = SandboxBridge(tmp_path / "grading", [], lambda *a, **kw: None)
    instance._run_session("s", 0)
    assert record.status == "completed" and not stopped
    assert instance.sif_cache.stats()["pinned_count"] == 1
    instance.grading_bridge.release("g")
    assert len(stopped) == 1
    assert not instance._sif_leases and instance.sif_cache.stats()["pinned_count"] == 0
    created[0].cleanup()
    assert len(stopped) == 1


@pytest.mark.parametrize("constructor_error", [False, True])
def test_failed_cleanup_keeps_lease_until_reaper_confirms_stop(session, constructor_error):
    instance, record, state, created, stopped = session
    state.constructor_error = constructor_error
    state.stop_error = "still mounted"
    instance._run_session("s", 0)
    assert record.status == "failed" and instance._pending_sif_cleanup
    assert instance.sif_cache.stats()["pinned_count"] == 1
    assert (instance.tmp_base / "s-tmp").exists()
    state.stop_error = ""
    instance._reap_sif_cleanup()
    assert not instance._pending_sif_cleanup and not instance._sif_leases
    assert instance.sif_cache.stats()["pinned_count"] == 0
    assert not (instance.tmp_base / "s-tmp").exists()


def test_parallel_environment_cleanup_releases_once(session, tmp_path):
    instance, record, state, created, stopped = session
    instance.grading_bridge = SandboxBridge(tmp_path / "grading", [], lambda *a, **kw: None)
    instance._run_session("s", 0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: created[0].cleanup(), range(4)))
    assert len(stopped) == 1 and not instance._sif_leases
    assert instance.sif_cache.stats()["pinned_count"] == 0


def test_cancel_during_acquire_never_starts_instance(session, monkeypatch):
    instance, record, state, created, stopped = session
    acquire = instance.sif_cache.acquire

    def cancel_after_acquire(*args, **kwargs):
        path = acquire(*args, **kwargs)
        record.cancel_event.set()
        return path

    monkeypatch.setattr(instance.sif_cache, "acquire", cancel_after_acquire)
    instance._run_session("s", 0)
    assert record.status == "cancelled" and not created
    assert not instance._sif_leases and instance.sif_cache.stats()["pinned_count"] == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_build_abort_kills_child_process_group(tmp_path, cancel):
    executable = tmp_path / "fake-apptainer"
    pid_file = tmp_path / "child.pid"
    executable.write_text(
        f"#!{sys.executable}\nimport subprocess,time,pathlib\np=subprocess.Popen(['sleep','60'])\npathlib.Path({str(pid_file)!r}).write_text(str(p.pid))\ntime.sleep(60)\n"
    )
    executable.chmod(0o755)
    instance = server.AgentServer.__new__(server.AgentServer)
    instance._sif_stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(instance._run_sif_command, [str(executable), "build"], timeout=1)
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        if cancel:
            instance._sif_stop_event.set()
        with pytest.raises(RuntimeError if cancel else subprocess.TimeoutExpired):
            future.result(timeout=5)
    pid = int(pid_file.read_text())
    status = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 3
    while status.exists() and status.read_text().split()[2] != "Z" and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert not status.exists() or status.read_text().split()[2] == "Z"
        assert not server._owned_child_pids
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("on_demand, expected", [(False, 1), (True, 2)])
def test_on_demand_keeps_tasks_without_prebuilt_sif(monkeypatch, tmp_path, on_demand, expected):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"docker_image": ["cached", "uncached"], "problem_statement": ["a", "b"]}), tmp_path / "data.parquet"
    )
    sif_dir = tmp_path / "sifs"
    sif_dir.mkdir()
    (sif_dir / "cached.sif").write_bytes(b"sif")
    config = {
        "environment": {},
        "r2e": {"datasets": {"train": {"directory": ".", "pattern": "*.parquet", "samples": 1}}},
    }
    monkeypatch.setattr(server, "get_config_from_spec", lambda *args: config)
    monkeypatch.setattr(server, "_stop_mswe_instances", lambda *args, **kwargs: None)
    monkeypatch.setattr(server.AgentServer, "_record_sweep_loop", lambda self: None)
    monkeypatch.setenv("AGENT_SERVER_TMP_DIR", str(tmp_path / "scratch"))
    monkeypatch.setenv("AGENT_SERVER_SIF_CACHE", str(int(on_demand)))
    monkeypatch.setenv("ENABLE_GAGAR", "0")
    instance = server.AgentServer(
        data_path=str(tmp_path),
        sif_dir=str(sif_dir),
        work_dir=str(tmp_path / "work"),
        train_concurrency=1,
        eval_concurrency=1,
    )
    try:
        assert instance.modes["train"].count == expected
    finally:
        for executor in instance.executors.values():
            executor.shutdown()
