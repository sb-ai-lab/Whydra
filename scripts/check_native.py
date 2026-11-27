"""Run clang-tidy with the compiler flags for Whydra's native extensions."""

import argparse
import os
import subprocess
import sys
import sysconfig
from importlib.util import find_spec
from pathlib import Path

from clang_tidy import get_executable

CORE = Path("whydra/algorithms/graph_core")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}


def sources_for(paths):
    """Select sources for clang-tidy.

    A shared header can affect any translation unit, even indirectly.
    """
    selected = set(paths)
    if CORE / "_pdsep_native.hpp" in selected:
        # This fragment is included inside _fisher_native.cpp's anonymous namespace.
        selected.remove(CORE / "_pdsep_native.hpp")
        selected.add(CORE / "_fisher_native.cpp")
    if selected and all(path.suffix.lower() in SOURCE_SUFFIXES for path in selected):
        return sorted(selected)
    names = subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0")
    return sorted(
        path
        for name in names
        if (path := Path(os.fsdecode(name))).suffix.lower() in SOURCE_SUFFIXES
        and path.is_file()
    )


def compiler_flags(path):
    """Yield each compiler configuration used to analyze the source file."""
    flags = ["-I", str(CORE)]
    for key in ("include", "platinclude"):
        flags.extend(["-isystem", sysconfig.get_path(key)])
    torch_include = os.environ.get("TORCH_INCLUDE_DIR")
    if not torch_include and (torch := find_spec("torch")) and torch.origin:
        torch_include = Path(torch.origin).parent / "include"
    if torch_include:
        flags.extend(
            [
                "-isystem",
                str(torch_include),
                "-isystem",
                str(Path(torch_include) / "torch/csrc/api/include"),
            ]
        )
    flags.append(f"-DTORCH_EXTENSION_NAME=_whydra_{path.stem}")

    if path.suffix.lower() == ".cu":
        cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda:
            flags.append(f"--cuda-path={cuda}")
        flags.extend(["-x", "cuda", "-std=c++17", "--cuda-gpu-arch=sm_80"])
        for side in ("--cuda-host-only", "--cuda-device-only"):
            yield [*flags, side]
            yield [*flags, side, "-DWHYDRA_ENABLE_FLOAT32=1"]
    elif path.suffix == ".c":
        yield [*flags, "-x", "c", "-std=c17"]
    else:
        if path == CORE / "_fisher_native.cpp":
            flags.append(
                "-ffp-model=strict" if os.name == "nt" else "-ffp-contract=off"
            )
        yield [*flags, "-x", "c++", "-std=c++17"]


def main():
    """Check the requested native sources and return the first failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="check every tracked source")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    if args.all == bool(args.paths):
        parser.error("specify --all or one or more native files")
    executable = get_executable("clang-tidy")
    for path in sources_for(args.paths):
        print("clang-tidy", path, flush=True)
        for flags in compiler_flags(path):
            result = subprocess.run(
                [executable, "--quiet", path, "--", *flags], check=False
            )
            if result.returncode:
                print("Compiler flags:", *flags, file=sys.stderr)
                return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
