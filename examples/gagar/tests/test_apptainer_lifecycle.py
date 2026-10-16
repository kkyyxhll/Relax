# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import os
import subprocess
import threading
import time

import pytest


@pytest.fixture(params=["examples.gagar.minisweagent.agent_server", "examples.gagar.code_generate.apptainer"])
def runtime(request):
    pytest.importorskip("minisweagent")
    return importlib.import_module(request.param)


def test_tracked_run_preserves_nonzero_exit(runtime) -> None:
    result = runtime._tracked_run(["bash", "-c", "exit 7"], text=True)
    assert result.returncode == 7


def test_tracked_run_check_raises_on_nonzero(runtime) -> None:
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        runtime._tracked_run(
            ["bash", "-c", "echo boom; exit 3"],
            text=True,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert excinfo.value.returncode == 3


def test_tracked_run_timeout_reports_partial_output(runtime) -> None:
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        runtime._tracked_run(
            ["bash", "-c", "echo partial; sleep 5"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=0.3,
        )
    assert b"partial" in (excinfo.value.output or b"")


def test_tracked_run_exit_codes_not_swallowed_by_reaper(runtime) -> None:
    """The core regression: an aggressive orphan reaper running concurrently
    must never reap a pid a ``_tracked_run`` is waiting on, which would make
    the command look like it exited 0."""
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            runtime._reap_orphan_children()
            time.sleep(0.001)

    reaper = threading.Thread(target=hammer, daemon=True)
    reaper.start()

    results: list[tuple[int, int]] = []
    errors: list[Exception] = []

    def worker(code: int) -> None:
        try:
            result = runtime._tracked_run(["bash", "-c", f"sleep 0.05; exit {code}"], text=True)
            results.append((code, result.returncode))
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i % 20 + 1,)) for i in range(60)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        stop.set()
        reaper.join(timeout=2)

    assert not errors
    assert len(results) == 60
    swallowed = [(expected, actual) for expected, actual in results if expected != actual]
    assert not swallowed, f"reaper swallowed exit codes: {swallowed}"


def test_instance_owner_pid_roundtrip(runtime) -> None:
    name = runtime._new_instance_name()
    assert name.startswith(runtime.INSTANCE_PREFIX)
    assert runtime._instance_owner_pid(name) == os.getpid()


def test_instance_owner_pid_legacy_and_malformed_return_none(runtime) -> None:
    # Legacy `mswe-<uuid>` names (no owner segment) and non-mswe names.
    assert runtime._instance_owner_pid("mswe-deadbeefcafe") is None
    assert runtime._instance_owner_pid("mswe-") is None
    assert runtime._instance_owner_pid("other-123-x") is None
