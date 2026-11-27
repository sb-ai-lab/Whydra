#!/usr/bin/env python3
"""Python line coverage for the CUDA modules, from CUDA-marked pytest tests.

Runs ``pytest -m cuda`` under ``coverage.py`` and reports executable-statement
(line) coverage only for the Python modules that drive the CUDA backends. CUDA
kernels in ``.cu``/``.cpp`` are outside ``coverage.py``; they need compiler
instrumentation instead.

Usage:
    python scripts/check_cuda_coverage.py [--min 90]

Exits non-zero when the reported total is below ``--min`` (default: no gate).
Requires a CUDA-capable device and the project's built CUDA extensions.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUDA_MODULES = (
    "whydra/algorithms/graph_core/cuda_chisq.py",
    "whydra/algorithms/graph_core/cuda_fisher_z.py",
    "whydra/algorithms/graph_core/cuda_graph_batch.py",
    "whydra/algorithms/graph_core/cuda_pdsep.py",
    "whydra/algorithms/graph_core/cuda_rai.py",
    "whydra/algorithms/graph_core/cuda_udag2pag.py",
    "whydra/algorithms/graph_core/_cuda.py",
    "whydra/algorithms/graph_core/cuda_jit.py",
    "whydra/algorithms/graph_core/cl_convert.py",
    "whydra/algorithms/graph_core/meek_torch.py",
)

_TOTAL = re.compile(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", re.MULTILINE)


def main() -> int:
    """Run the CUDA tests and compare Python line coverage with the requested minimum."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--min",
        type=float,
        default=0.0,
        help="fail when the total line coverage is below this percentage",
    )
    args = parser.parse_args()

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--source=whydra.algorithms.graph_core",
            "-m",
            "pytest",
            "-q",
            "--require-cuda",
            "-m",
            "cuda",
        ],
        cwd=ROOT,
        check=False,
    )
    if run.returncode != 0:
        return run.returncode

    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--include=" + ",".join(CUDA_MODULES),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(report.stdout)
    sys.stderr.write(report.stderr)
    if report.returncode != 0:
        return report.returncode

    match = _TOTAL.search(report.stdout)
    if match is None:
        print("could not parse coverage total", file=sys.stderr)
        return 2
    total = float(match.group(1))
    if total < args.min:
        print(
            f"CUDA Python line coverage {total:.0f}% is below --min {args.min:.0f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
