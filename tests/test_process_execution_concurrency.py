"""A callback error must not invalidate other fits using the same pool."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from whydra.algorithms.graph_core.process_execution import process_map


def _hold_until_released(directory):
    directory = Path(directory)
    (directory / "started").touch()
    deadline = time.monotonic() + 15
    while not (directory / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("test did not release worker")
        time.sleep(0.01)
    return 42


def _fail():
    raise ValueError("independent callback failed")


def test_callback_failure_preserves_concurrent_pool_lease(tmp_path):
    with ThreadPoolExecutor(max_workers=1) as clients:
        pending = clients.submit(
            process_map, _hold_until_released, [(str(tmp_path),)], n_jobs=2
        )
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / "started").exists():
                assert time.monotonic() < deadline, "process worker did not start"
                time.sleep(0.01)
            with pytest.raises(ValueError, match="independent callback failed"):
                process_map(_fail, [()], n_jobs=2)
        finally:
            (tmp_path / "release").touch()
        assert pending.result(timeout=10) == [42]
