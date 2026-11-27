"""Verify an installed wheel independently of imports from the source checkout.

Install with --no-deps into the supplied directory first. This check reuses
the interpreter's scientific stack and is not a clean dependency resolution.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    """Check the wheel's code and license notices, then exercise its installed package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("installation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.wheel) as wheel:
        files = wheel.namelist()
        assert not any(
            name.startswith(("whydra_benchmarks/", "playground/", "tests/"))
            for name in files
        )
        assert "whydra/evaluation/benchmark.py" not in files
        notices = []
        for source in sorted((root / "THIRD_PARTY_LICENSES").iterdir()):
            if source.suffix != ".txt" and source.name != "README.md":
                continue
            suffix = "/licenses/THIRD_PARTY_LICENSES/" + source.name
            matches = [name for name in files if name.endswith(suffix)]
            assert len(matches) == 1, (source.name, matches)
            assert wheel.read(matches[0]) == source.read_bytes(), source.name
            notices.extend(matches)
        mismatches = [
            name
            for name in files
            if name.startswith("whydra/")
            and name.endswith(".py")
            and (
                not (root / name).exists()
                or wheel.read(name) != (root / name).read_bytes()
            )
        ]
        assert not mismatches, mismatches
    installation = args.installation.resolve()
    program = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import whydra
import numpy as np
from whydra.algorithms.wrappers import StandalonePCStable, StandaloneFCIStable
assert pathlib.Path(whydra.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))
data = np.random.default_rng(0).normal(size=(100, 3))
pc = StandalonePCStable().run(data)
fci = StandaloneFCIStable().run(data)
assert pc.graph_kind == 'cpdag' and fci.graph_kind == 'pag'
repo = whydra.BenchmarkCaseRepository(pathlib.Path(sys.argv[1]) / 'external-cases')
assert repo.list_available_cases() == []
print(json.dumps({'module': whydra.__file__, 'version': whydra.__version__,
                  'cases_root': str(repo.cases_root), 'case_count': len(repo.list_available_cases()),
                  'graph_kinds': [pc.graph_kind, fci.graph_kind]}))
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", program, str(installation)],
        cwd=installation,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    report = {
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "source_python_files_match": not mismatches,
        "license_paths": notices,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "qualification": "Fresh --no-deps target; uses existing scientific dependencies, not a clean full-profile install",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
