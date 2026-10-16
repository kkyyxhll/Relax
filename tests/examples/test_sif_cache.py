# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import multiprocessing
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from examples.mini_swe_agent.sif_cache import SifCache, image_key


def _no_instances(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 0, "INSTANCE NAME PID IP IMAGE\n", "")


def _cache(tmp_path, **kwargs):
    return SifCache(sif_dir=tmp_path / "sifs", build_tmp_root=tmp_path / "build", run=_no_instances, **kwargs)


def _acquire_process(sif_dir, build_dir, ready, release, builds):
    def run(cmd, **kwargs):
        if cmd[1] == "build":
            with builds.get_lock():
                builds.value += 1
            Path(cmd[2]).write_bytes(b"image")
        return _no_instances(cmd)

    cache = SifCache(sif_dir=sif_dir, build_tmp_root=build_dir, run=run)
    path = cache.acquire("docker.io/test/image:tag")
    ready.put(str(path))
    release.wait(10)
    cache.release("test/image:tag")


def test_processes_share_one_build_and_hold_independent_leases(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, release, builds = ctx.Queue(), ctx.Event(), ctx.Value("i", 0)
    cache = _cache(tmp_path, max_entries=1)
    processes = [
        ctx.Process(target=_acquire_process, args=(cache.sif_dir, tmp_path / "build", ready, release, builds))
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        paths = [ready.get(timeout=15) for _ in processes]
        assert len(set(paths)) == 1 and builds.value == 1
        path = Path(paths[0])
        (cache.sif_dir / "other.sif").write_bytes(b"other")
        cache._evict_to_fit(exclude_key="other")
        assert path.read_bytes() == b"image"
    finally:
        release.set()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    cache._evict_to_fit(exclude_key="other")
    assert not path.exists()


def test_lru_respects_live_instances_and_reuses_existing_images(tmp_path):
    cache = _cache(tmp_path, max_entries=1)
    old = cache.sif_dir / "old.sif"
    hot = cache.sif_dir / "hot.sif"
    active = cache.sif_dir / "active.sif"
    for i, path in enumerate([old, hot, active]):
        path.write_bytes(b"sif")
        os.utime(path, (i + 1, i + 1))
    cache._run = lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, f"HEADER\ninstance 1 x {active}\n", "")
    assert cache.acquire("hot") == hot
    cache._evict_to_fit()
    assert not old.exists() and hot.exists() and active.exists()
    cache.release("hot")
    cache._evict_to_fit()
    assert not hot.exists() and active.exists()


def test_build_slots_bound_different_images_across_cache_objects(tmp_path):
    lock, first_two, proceed = threading.Lock(), threading.Event(), threading.Event()
    active = peak = total = 0

    def run(cmd, **kwargs):
        nonlocal active, peak, total
        if cmd[1] != "build":
            return _no_instances(cmd)
        with lock:
            active += 1
            total += 1
            peak = max(peak, active)
            if active == 2:
                first_two.set()
        assert proceed.wait(5)
        assert kwargs["env"]["APPTAINER_DISABLE_CACHE"] == "true"
        assert kwargs["timeout"] == 1800
        Path(cmd[2]).write_bytes(b"image")
        with lock:
            active -= 1
        return _no_instances(cmd)

    caches = [_cache(tmp_path, max_builds=2) for _ in range(8)]
    for cache in caches:
        cache._run = run
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(cache.acquire, f"image-{i}") for i, cache in enumerate(caches)]
        try:
            assert first_two.wait(5)
        finally:
            proceed.set()
        assert all(future.result(timeout=10).exists() for future in futures)
    assert total == 8 and peak == 2
    for i, cache in enumerate(caches):
        cache.release(f"image-{i}")


@pytest.mark.parametrize(
    "failure", [subprocess.CalledProcessError(7, "build", output="failed"), subprocess.TimeoutExpired("build", 1)]
)
def test_failed_build_never_publishes_partial_sif(tmp_path, failure):
    cache = _cache(tmp_path, build_retries=1)

    def run(cmd, **kwargs):
        Path(cmd[2]).write_bytes(b"partial")
        raise failure

    cache._run = run
    with pytest.raises(RuntimeError, match="apptainer build failed"):
        cache.acquire("image")
    assert not list(cache.sif_dir.glob("*.sif"))
    assert not list(cache.sif_dir.glob("*.tmp"))
    assert not list((tmp_path / "build").iterdir())
    cache._run = lambda cmd, **kwargs: (Path(cmd[2]).write_bytes(b"complete"), _no_instances(cmd))[1]
    assert cache.acquire("image").read_bytes() == b"complete"
    cache.release("image")


def test_cancelled_waiter_does_not_interrupt_shared_build(tmp_path):
    cache = _cache(tmp_path)
    started, finish, cancel = threading.Event(), threading.Event(), threading.Event()

    def run(cmd, **kwargs):
        started.set()
        assert finish.wait(5)
        Path(cmd[2]).write_bytes(b"image")
        return _no_instances(cmd)

    cache._run = run
    with ThreadPoolExecutor(max_workers=2) as pool:
        builder = pool.submit(cache.acquire, "image")
        assert started.wait(5)
        waiter = pool.submit(cache.acquire, "image", cancel_event=cancel)
        cancel.set()
        try:
            with pytest.raises(RuntimeError, match="cancelled"):
                waiter.result(timeout=3)
        finally:
            finish.set()
        assert builder.result(timeout=5).read_bytes() == b"image"
    assert cache.stats()["pinned_count"] == 1
    cache.release("image")


def test_image_normalization_matches_prebuilt_filenames():
    assert image_key("docker://docker.io/test/image:tag") == "test__image__tag"


@pytest.mark.parametrize(
    "failure",
    [OSError("cannot list"), subprocess.CalledProcessError(1, "list"), subprocess.TimeoutExpired("list", 10)],
)
def test_failed_instance_query_skips_eviction(tmp_path, failure):
    cache = _cache(tmp_path, max_bytes=1)
    path = cache.sif_dir / "active.sif"
    path.write_bytes(b"mounted")

    def run(cmd, **kwargs):
        assert kwargs["timeout"] == 10
        raise failure

    cache._run = run
    cache._evict_to_fit()
    assert path.exists()


def test_eviction_does_not_wait_for_another_cleaner(tmp_path):
    import fcntl

    cache = _cache(tmp_path, max_bytes=1)
    with (cache.sif_dir / ".eviction.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(cache._evict_to_fit).result(timeout=1)
