"""Small correctness smoke for installed/source builds; never a speed benchmark.

Run with ``python -I ... --installed`` to exclude the checkout from imports.
The numerical workload is explicitly capped at two conditioning variables.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _worker_probe(array):
    return os.getpid(), float(array.sum()), bool(array.flags.writeable)


def main():
    """Check imports and execution with the requested native backend setting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-native", choices=("required", "disabled"), required=True
    )
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expect_native == "disabled":
        os.environ["WHYDRA_DISABLE_NATIVE"] = "1"
    if not args.installed:
        sys.path.insert(0, str(ROOT))
    import numpy as np

    import whydra
    from whydra.algorithms import wrappers
    from whydra.algorithms.graph_core import native_fisher, worker_data
    from whydra.algorithms.graph_core.process_execution import (
        process_map,
        shutdown_process_pools,
    )

    if args.installed:
        assert not Path(whydra.__file__).resolve().is_relative_to(ROOT / "whydra"), (
            whydra.__file__
        )
    assert (args.expect_native == "required") == native_fisher.NATIVE_AVAILABLE, (
        native_fisher.NATIVE_UNAVAILABLE_REASON
    )
    rng = np.random.default_rng(916)
    data = rng.normal(size=(180, 12))
    for column in range(1, data.shape[1]):
        data[:, column] += 0.75 * data[:, column - 1]
    cases = {
        "pc": (wrappers.StandalonePCStable, ("skeleton_backend",)),
        "fci": (
            wrappers.StandaloneFCIStableParallelCached,
            ("skeleton_backend", "pdsep_backend"),
        ),
        "rfci": (wrappers.StandaloneRFCIStable, ("skeleton_backend", "orient_backend")),
        "fci-plus": (
            wrappers.StandaloneFCIPlusStableParallelCached,
            ("skeleton_backend", "augment_backend", "dsep_backend"),
        ),
        "cfci": (
            wrappers.StandaloneCFCIStableParallelCached,
            ("skeleton_backend", "pdsep_backend"),
        ),
    }
    hashes = {}
    try:
        array = np.eye(4)
        probes = process_map(
            _worker_probe, [(array,)] * 4, n_jobs=2, shared_arg_indices=(0,)
        )
        assert all(pid != os.getpid() and value == 4.0 for pid, value, _ in probes)
        if worker_data._SHARING_SUPPORTED:
            assert not any(writable for _, _, writable in probes)
        for name, (cls, backends) in cases.items():
            print(f"Checking {name}: 1/2/4 workers", file=sys.stderr, flush=True)
            reference = None
            for jobs in (1, 2, 4):
                options = {
                    "alpha": 0.01,
                    "n_jobs": jobs,
                    "indep_test": "fisherz",
                    "max_cond_set_size": 2,
                    "ci_cache_size": 128,
                    "show_progress": False,
                    "verbose": False,
                }
                options.update(dict.fromkeys(backends, "processes"))
                graph = cls(**options).run(data)
                if reference is None:
                    reference = graph.graph.copy()
                np.testing.assert_array_equal(
                    graph.graph, reference, err_msg=f"{name}, jobs={jobs}"
                )
                actual = graph.metadata.get("execution", {}).get("actual_backends", [])
                assert "threads" not in actual, (name, jobs, actual)
            hashes[name] = hashlib.sha256(
                reference.astype(np.int64).tobytes()
            ).hexdigest()
    finally:
        shutdown_process_pools(wait=True)
    report = {
        "status": "ok",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "module": whydra.__file__,
        "installed": args.installed,
        "native": native_fisher.NATIVE_AVAILABLE,
        "native_math_compatible": native_fisher.NATIVE_MATH_COMPATIBLE,
        "shared_memory": worker_data._SHARING_SUPPORTED,
        "jobs": [1, 2, 4],
        "rows": 180,
        "variables": 12,
        "max_cond_set_size": 2,
        "graph_hashes": hashes,
        "worker_pids": sorted({pid for pid, _, _ in probes}),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
