"""Serialize dataset tables as parquet and the metadata and model as JSON (O-3).

A dataset directory (0.7.0) contains CURRENT, a text file with the active version
name, written atomically via a temporary file and os.replace. Each publication creates a
v-<utc>-<rand>/ directory. Earlier versions remain until prune_versions is called
explicitly. Each version contains:

* manifest.json (manifest_schema 2): package version, scenario, config_sha256, three
seeds, n_calib, n_requested, n_observed, split, dataset version, recipe, files with
SHA-256 hashes, byte sizes and row counts, columns with types and purposes, targets,
design, and environment including BLAS, CPU features, and libc.

* observed.parquet (O-1): observed X columns, optional z, t, y, and optional period and
cluster columns.

* ground_truth.parquet: per-row y0, y1, ite, tau, mu0, mu1, e, t_true, segments,
tau_cate, cvte, vte_w, and row_id. Additional fields depend on the axes: z, itt,
tau_itt, tau_itt_cate, compliance, e_assign; e_source and tau_cate_selected, possibly
tau_itt_selected and e_assign_selected; and period, cluster, dup_index, or miss_<col>.

* meta.json (meta_schema 4): version, configuration, seeds, n_requested, n, ate,
optional ate_source and ate_selected, late_realized, targets, achieved, design, recipe,
graph (adjacency, roles, dag_post), discrete nodes, and pathology_log.

* scm.json: the SCM.to_json() model artifact containing config, model_seed, n_calib,
graph, params, and calib.

Publication is atomic for readers. save_dataset writes a temporary directory inside
<out>, flushes files, renames the directory to v-…, then atomically switches CURRENT.
Readers that pinned the previous version can finish reading it after the switch. A
failed writer leaves the active version intact. load_dataset reads CURRENT once. It also
accepts a version directory or, with a warning, the flat 0.6.0 layout. replay(dir) is
the documented reproduction procedure: load_scm, sample(n_requested, sample_seed),
apply_pathologies(seed, plan=recipe), then restore split.
"""

import datetime as _dt
import hashlib
import json
import os
import pathlib
import platform
import re
import secrets
import shutil
import warnings
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from .config import PathologyConfig
from .generate import Dataset
from .graph import column_node
from .scm import SCM, config_sha256
from .truth import late_realized, truth_targets

META_SCHEMA = 4  # Version 0.7.0 added the recipe, selected fields, tau_itt_cate, and design.cate_dgp / test_dgp.
MANIFEST_SCHEMA = 2  # 0.7.0: version, recipe, environment.blas / cpu_features / threads
_SUPPORTED_MANIFEST = (1, 2)
# Reference environment for strict comparisons against golden results (also recorded upstream in
# tests/golden/environment.txt). It includes versions, BLAS build, NumPy-dispatched CPU features,
# thread settings, and NPY_DISABLE_CPU_FEATURES (0.7.0).
REFERENCE_ENVIRONMENT = {
    "python": "3.12.8",
    "numpy": "2.5.3",
    "pandas": "3.0.5",
    "blas": "scipy-openblas 0.3.34.106.0",
    "cpu_features": "X86_V3",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NPY_DISABLE_CPU_FEATURES": "",
}
# On X86_V4 (AVX512) hosts, disable feature groups above X86_V3 before importing NumPy
# to select its X86_V3 float64 exp/log kernels. This setting reproduced the reference
# output bit for bit (0.7.1). An empty NPY_DISABLE_CPU_FEATURES value also passes the
# reference check. NumPy 2.5 ignores individual names such as AVX2 and FMA3 here.
REFERENCE_EQUIVALENT_DISABLE = "X86_V4 AVX512_ICL AVX512_SPR"
_REFERENCE_ALTERNATIVES = {
    "NPY_DISABLE_CPU_FEATURES": ("", REFERENCE_EQUIVALENT_DISABLE)
}
_FILES = ("observed.parquet", "ground_truth.parquet", "meta.json", "scm.json")
CURRENT = "CURRENT"


def ground_truth_frame(ds: Dataset) -> pd.DataFrame:
    """Build the per-row ground-truth table with the same columns as ground_truth.parquet."""
    gt = pd.DataFrame(
        {
            "y0": ds.y0,
            "y1": ds.y1,
            "ite": ds.ite,
            "tau": ds.tau,
            "mu0": ds.mu0,
            "mu1": ds.mu1,
            "e": ds.e,
            "t_true": ds.t_true,
            "segments": ds.segments,
            "tau_cate": ds.tau_cate,
            "cvte": ds.cvte,
            "vte_w": ds.vte_w,
            "row_id": ds.row_id,
        }
    )
    for name in (
        "z",
        "itt",
        "tau_itt",
        "tau_itt_cate",
        "compliance",
        "e_assign",
        "e_source",
        "tau_cate_selected",
        "tau_itt_selected",
        "e_assign_selected",
        "period",
        "cluster",
        "dup_index",
        "tau_cate_x",
    ):
        v = getattr(ds, name)  # Only present when the axis was applied.
        if v is not None:
            gt[name] = v
    if (
        ds.y_scale is not None
    ):  # Natural outcome scale (0.9.0), only when scale_sigma > 0.
        gt["y_scale"] = ds.y_scale
    mt = ds.mediator_truth()
    if (
        mt is not None
    ):  # Binary mediators (0.10.0): export the n × m fields one column at a time.
        assert ds.mediators is not None
        for k in range(
            mt["p_m0"].shape[1]
        ):  # Gaussian mediators are omitted from ground_truth.
            gt[f"m{k}"] = ds.mediators[:, k]
            gt[f"m{k}_0"], gt[f"m{k}_1"] = mt["m0"][:, k], mt["m1"][:, k]
            gt[f"p_m{k}_0"], gt[f"p_m{k}_1"] = mt["p_m0"][:, k], mt["p_m1"][:, k]
            gt[f"tau_m{k}"], gt[f"theta_m{k}"] = mt["tau_m"][:, k], mt["theta"][:, k]
    if ds.x_missing is not None:
        for c in ds.x_missing.columns:
            gt[f"miss_{c}"] = ds.x_missing[c].to_numpy(dtype=bool)
    return gt


def _blas_build() -> str | None:
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{blas.get('name')} {blas.get('version')}"
    except (AttributeError, KeyError, TypeError):  # pragma: no cover
        return None


def _cpu_features() -> str | None:
    """Return NumPy's dispatched CPU features (SIMD Extensions.found), separated by commas."""
    try:
        simd = np.show_config(mode="dicts")["SIMD Extensions"]
        return ",".join(simd.get("found", []))
    except (AttributeError, KeyError, TypeError):  # pragma: no cover
        return None


def _libc() -> str | None:
    """Identify the process libc with platform.libc_ver().

    NumPy's baseline exp/log/tanh kernels call libm (0.7.1).
    """
    name, ver = platform.libc_ver()
    return f"{name} {ver}".strip() or None


def reference_allowed(key: str) -> tuple:
    """Return allowed values for a REFERENCE_ENVIRONMENT key.

    NPY_DISABLE_CPU_FEATURES may be empty or reference-equivalent.
    """
    return _REFERENCE_ALTERNATIVES.get(key, (REFERENCE_ENVIRONMENT[key],))


def environment() -> dict:
    """Collect environment information for the manifest.

    Record technical environment settings without secrets. reference is True when every
    REFERENCE_ENVIRONMENT key matches (0.7.0). For NPY_DISABLE_CPU_FEATURES, accept
    an empty value or REFERENCE_EQUIVALENT_DISABLE (0.7.1). Include libc in the
    fingerprint but exclude it from the reference match (0.7.1).
    """
    try:
        pa = version("pyarrow")
    except PackageNotFoundError:  # pragma: no cover
        pa = None
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libc": _libc(),
        "blas": _blas_build(),
        "cpu_features": _cpu_features(),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
    }
    env["threads"] = {
        "OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
        "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
    }
    env["reference"] = all(
        env.get(k) in reference_allowed(k) for k in REFERENCE_ENVIRONMENT
    )
    return env


def _column_kind(name: str, ds: Dataset) -> str:
    if name == "t":
        return "treatment"
    if name == "z":
        return "assignment"
    if name == "y":
        return "outcome"
    if name == "period":
        return "period"
    if name == "cluster":
        return "cluster"
    if name.startswith("y_pre") and name[5:].isdigit():
        return "pre_outcome"  # Pretreatment outcome lags (0.11.0).
    if column_node(name) >= 0:
        return "covariate"
    return "exposed"  # Mediators or descendants of Y exposed by bad_controls.


def dataset_meta(ds: Dataset) -> dict:
    """Collect the dataset metadata written alongside the generated tables."""
    from . import __version__

    assert ds.n_requested is not None
    return {
        "scmgen_version": __version__,
        "meta_schema": META_SCHEMA,
        "scenario": ds.config.name,
        "config": ds.config.to_dict(),
        "seed": ds.seed,
        "model_seed": ds.model_seed,
        "sample_seed": ds.sample_seed,
        "pathology_seed": ds.pathology_seed,
        "n_calib": ds.scm.n_calib if ds.scm is not None else None,
        "split": ds.split,
        "n_requested": int(ds.n_requested),
        "n": int(ds.n),
        "n_source": ds.n_source,
        "ate": ds.ate,
        "ate_source": ds.ate_source,
        "ate_selected": ds.ate_selected,
        "late_realized": late_realized(ds),
        "targets": truth_targets(ds),
        "achieved": ds.achieved(),
        "design": ds.design(),
        "recipe": ds.recipe(),
        "adjacency": ds.graph.adjacency.tolist(),
        "roles": list(ds.graph.roles),
        "roles_effective": list(
            ds.graph.roles
            if ds.graph.roles_effective is None
            else ds.graph.roles_effective
        ),
        "node_names": list(ds.graph.node_names),
        "dag_post": ds.dag_post(),
        "discrete": ds.discrete_truth(),  # None when there are no discrete nodes (since 0.4.0).
        "pathology_log": ds.pathology_log,
    }


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(
    ds: Dataset,
    tmp: pathlib.Path,
    observed: pd.DataFrame,
    gt: pd.DataFrame,
    version: str,
) -> dict:
    from . import __version__

    assert ds.n_requested is not None
    files = {}
    rows = {
        "observed.parquet": len(observed),
        "ground_truth.parquet": len(gt),
        "meta.json": None,
        "scm.json": None,
    }
    for name in _FILES:
        f = tmp / name
        if f.is_file():
            files[name] = {
                "sha256": _sha256_file(f),
                "bytes": f.stat().st_size,
                "rows": rows[name],
            }
    return {
        "scmgen_version": __version__,
        "manifest_schema": MANIFEST_SCHEMA,
        "meta_schema": META_SCHEMA,
        "version": version,
        "scenario": ds.config.name,
        "config_sha256": config_sha256(
            ds.scm.config if ds.scm is not None else ds.config
        ),
        "model_seed": ds.model_seed,
        "sample_seed": ds.sample_seed,
        "pathology_seed": ds.pathology_seed,
        "n_calib": ds.scm.n_calib if ds.scm is not None else None,
        "n_requested": int(ds.n_requested),
        "n_observed": int(ds.n),
        "split": ds.split,
        "recipe": ds.recipe(),
        "files": files,
        "columns": {
            "observed": [
                {
                    "name": str(c),
                    "dtype": str(observed[c].dtype),
                    "node": (column_node(c) if column_node(c) >= 0 else None),
                    "kind": _column_kind(str(c), ds),
                }
                for c in observed.columns
            ],
            "ground_truth": [
                {"name": str(c), "dtype": str(gt[c].dtype)} for c in gt.columns
            ],
        },
        "targets": truth_targets(ds),
        "design": ds.design(),
        "environment": environment(),
    }


# Write datasets.


def _fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover (Windows)
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def _write_bytes_sync(path: pathlib.Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _write_json(path: pathlib.Path, obj, indent: int) -> None:
    _write_bytes_sync(
        path,
        json.dumps(
            obj, ensure_ascii=False, indent=indent, default=_json_default
        ).encode("utf-8"),
    )


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    msg = f"cannot serialize {type(o).__name__} to JSON"
    raise TypeError(msg)


def _new_version_name() -> str:
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"v-{utc}-{secrets.token_hex(3)}"


def _write_pointer(out: pathlib.Path, version: str) -> None:
    """Sync a temporary file, replace CURRENT atomically, and try to sync the directory."""
    tmp = out / f".{CURRENT}.tmp-{os.getpid()}-{secrets.token_hex(3)}"
    _write_bytes_sync(tmp, (version + "\n").encode("utf-8"))
    os.replace(tmp, out / CURRENT)
    _fsync_dir(out)


def save_dataset(
    ds: Dataset, out_dir: str | pathlib.Path, overwrite: bool = False
) -> pathlib.Path:
    """Publish a new version in out_dir and return out_dir.

    With overwrite=False, os.mkdir(out_dir) claims the directory atomically. An existing
    directory raises FileExistsError before any file is written, so the kernel resolves
    concurrent claims. With overwrite=True, write the new version alongside earlier ones
    and atomically switch CURRENT. Readers may still be using earlier versions. Call
    prune_versions(out_dir, keep=1) separately to remove them.

    Write the version into a temporary directory under out_dir, fsync its files, rename
    it to v-…, then publish CURRENT. A failure before the CURRENT update leaves the
    previous version active. A failed first publication may leave a directory with no
    active version. Use current_version(out_dir) to find the active path.
    """
    out = pathlib.Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        out.mkdir(exist_ok=True)
    else:
        try:
            os.mkdir(out)
        except FileExistsError as exc:
            msg = f"{out}: directory already exists (use save_dataset(..., overwrite=True) to publish a new version)"
            raise FileExistsError(msg) from exc
    version = _new_version_name()
    tmp = out / f".tmp-{os.getpid()}-{secrets.token_hex(3)}"
    tmp.mkdir()
    try:
        observed, gt = ds.to_frame(), ground_truth_frame(ds)
        observed.to_parquet(tmp / "observed.parquet", index=False)
        gt.to_parquet(tmp / "ground_truth.parquet", index=False)
        _write_json(tmp / "meta.json", dataset_meta(ds), indent=1)
        if ds.scm is not None:
            _write_bytes_sync(tmp / "scm.json", ds.scm.to_json().encode("utf-8"))
        _write_json(
            tmp / "manifest.json", _manifest(ds, tmp, observed, gt, version), indent=1
        )  # Write this last.
        for name in ("observed.parquet", "ground_truth.parquet"):
            with open(tmp / name, "r+b") as f:
                os.fsync(f.fileno())
        _fsync_dir(tmp)
        os.rename(tmp, out / version)
        _fsync_dir(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        _write_pointer(out, version)
    except BaseException:  # The version was never published. Clean up the orphan.
        shutil.rmtree(out / version, ignore_errors=True)
        raise
    return out


def current_version(out_dir: str | pathlib.Path) -> pathlib.Path | None:
    """Return the active version directory named by CURRENT, or None if the pointer is absent."""
    out = pathlib.Path(out_dir)
    ptr = _bundle_file(out, CURRENT)
    if not ptr.is_file():
        return None
    name = ptr.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"v-[A-Za-z0-9_-]+", name):
        msg = f"{ptr}: invalid version name {name!r}"
        raise ValueError(msg)
    return _bundle_child(out, name)


def list_versions(out_dir: str | pathlib.Path) -> list[pathlib.Path]:
    """List v-… versions in the dataset directory by increasing name (UTC publication time)."""
    out = pathlib.Path(out_dir)
    return sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("v-"))


def prune_versions(out_dir: str | pathlib.Path, keep: int = 1) -> list[pathlib.Path]:
    """Remove old versions, keeping the newest keep versions and the active one.

    Return the paths selected for removal. Deletion errors are ignored, so
    a listed directory may still exist.
    """
    if keep < 1:
        msg = "prune_versions: keep must be >= 1"
        raise ValueError(msg)
    out = pathlib.Path(out_dir)
    active = current_version(out)
    versions = list_versions(out)
    doomed = [
        v for v in versions[:-keep] if active is None or v.resolve() != active.resolve()
    ]
    for v in doomed:
        shutil.rmtree(v, ignore_errors=True)
    return doomed


# Read datasets.


def _bundle_child(directory: pathlib.Path, name: str) -> pathlib.Path:
    child = directory / name
    if child.resolve().parent != directory.resolve():
        msg = f"{child}: path escapes the dataset directory"
        raise ValueError(msg)
    return child


def _bundle_file(directory: pathlib.Path, name: str) -> pathlib.Path:
    child = _bundle_child(directory, name)
    if child.exists() and not child.is_file():
        msg = f"{child}: expected a regular file"
        raise ValueError(msg)
    return child


def resolve_dataset_dir(path: str | pathlib.Path) -> pathlib.Path:
    """Locate the directory containing the requested dataset version.

    For <out> with CURRENT, read the pointer once and use the active version. Accept a
    direct v-… path. For the flat 0.6.0 layout with files in path and no CURRENT, issue
    a warning.
    """
    p = pathlib.Path(path)
    v = current_version(p)
    if v is not None:
        if not _bundle_file(v, "observed.parquet").is_file():
            msg = f"{p}: CURRENT points to {v.name}, but that version is missing or incomplete"
            raise FileNotFoundError(msg)
        return v
    if (
        _bundle_file(p, "observed.parquet").is_file()
        or _bundle_file(p, "scm.json").is_file()
    ):
        if not (p.name.startswith("v-") and (p.parent / CURRENT).is_file()):
            warnings.warn(
                f"{p}: reading flat dataset layout as stored (before 0.7.0, no CURRENT)",
                stacklevel=3,
            )
        return p
    versions = list_versions(p) if p.is_dir() else []
    if versions:
        msg = f"{p}: no active version (CURRENT); versions on disk: {[v.name for v in versions]}"
        raise FileNotFoundError(msg)
    msg = f"{p}: dataset directory not found"
    raise FileNotFoundError(msg)


def load_dataset(
    in_dir: str | pathlib.Path, verify: str | bool = "rows"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load (observed, ground_truth, meta) from a dataset.

    in_dir may contain CURRENT, point to a version directly, or use the flat 0.6.0
    layout. verify="rows" (default) checks manifest_schema, compares file row counts
    with the manifest, and checks table lengths against each other. "hashes" also
    verifies file SHA-256 hashes. False checks only length consistency, as in 0.5.0. A
    dataset without a manifest (0.5.0) is read with a warning.
    """
    p = resolve_dataset_dir(in_dir)
    manifest = None
    manifest_file = _bundle_file(p, "manifest.json")
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("manifest_schema") not in _SUPPORTED_MANIFEST:
            msg = (
                f"{p}: manifest_schema = {manifest.get('manifest_schema')!r}, expected one of "
                f"{_SUPPORTED_MANIFEST}"
            )
            raise ValueError(msg)
    elif verify:
        warnings.warn(
            f"{p}: manifest.json is missing (dataset written before 0.6.0); verification with verify={verify!r} "
            f"is unavailable",
            stacklevel=2,
        )
    if manifest is not None and verify == "hashes":
        for name, info in manifest["files"].items():
            if name not in _FILES:
                msg = f"{p}: unexpected file in manifest: {name!r}"
                raise ValueError(msg)
            actual = _sha256_file(_bundle_file(p, name))
            if actual != info["sha256"]:
                msg = f"{p / name}: sha256 {actual[:12]}... differs from manifest {info['sha256'][:12]}..."
                raise ValueError(msg)
    observed = pd.read_parquet(_bundle_file(p, "observed.parquet"))
    gt = pd.read_parquet(_bundle_file(p, "ground_truth.parquet"))
    meta = json.loads(_bundle_file(p, "meta.json").read_text(encoding="utf-8"))
    if len(observed) != len(gt):
        msg = f"{p}: observed ({len(observed)}) and ground_truth ({len(gt)}) have different lengths"
        raise ValueError(msg)
    if "n" in meta and meta["n"] != len(gt):
        msg = f"{p}: meta.n = {meta['n']}, ground_truth has {len(gt)} rows"
        raise ValueError(msg)
    if manifest is not None and verify:
        for name, df in (("observed.parquet", observed), ("ground_truth.parquet", gt)):
            want = manifest["files"][name]["rows"]
            if want != len(df):
                msg = f"{p / name}: found {len(df)} rows, manifest specifies {want}"
                raise ValueError(msg)
        if manifest["n_observed"] != len(gt):
            msg = f"{p}: manifest.n_observed = {manifest['n_observed']}, found {len(gt)} rows"
            raise ValueError(msg)
    return observed, gt, meta


def load_scm(path: str | pathlib.Path) -> SCM:
    """Load the SCM from a dataset directory (CURRENT, version, or flat layout) or a direct scm.json path."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = _bundle_file(resolve_dataset_dir(p), "scm.json")
    if not p.is_file():
        msg = f"{p}: scm.json not found"
        raise FileNotFoundError(msg)
    return SCM.from_json(p.read_text(encoding="utf-8"))


def load_recipe(path: str | pathlib.Path) -> dict:
    """Read the dataset recipe from the directory's manifest.json or meta.json."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = resolve_dataset_dir(p)
        for name in ("manifest.json", "meta.json"):
            f = _bundle_file(p, name)
            if f.is_file():
                doc = json.loads(f.read_text(encoding="utf-8"))
                if "recipe" in doc:
                    return doc["recipe"]
        msg = f"{p}: recipe not found (dataset written before 0.7.0)"
        raise ValueError(msg)
    return json.loads(p.read_text(encoding="utf-8"))["recipe"]


def replay(path: str | pathlib.Path) -> Dataset:
    """Reproduce a dataset from its saved recipe.

    Load the saved SCM with load_scm, call sample(n_requested, sample_seed),
    apply_pathologies(pathology_seed, plan=applied), then restore split and test_dgp. In
    the reference environment this gives bitwise reproduction.
    """
    from .pathologies import apply_pathologies

    scm = load_scm(path)
    recipe = load_recipe(path)
    if recipe.get("recipe_schema") != 1:
        msg = f"recipe_schema = {recipe.get('recipe_schema')!r}, expected 1"
        raise ValueError(msg)
    model, sample, patho = recipe["model"], recipe["sample"], recipe["pathologies"]
    if (
        model.get("model_seed") != scm.model_seed
        or int(model.get("n_calib")) != scm.n_calib
    ):
        msg = "recipe and scm.json disagree on model_seed or n_calib"
        raise ValueError(msg)
    if model.get("config_sha256") != config_sha256(scm.config):
        msg = "recipe and scm.json disagree on config_sha256"
        raise ValueError(msg)
    ds = scm.sample(int(sample["n_requested"]), int(sample["sample_seed"]))
    ds.split = sample.get("split", "train")
    applied = patho.get("applied") or []
    if applied:
        plan = [
            (
                int(a["index"]),
                PathologyConfig(
                    kind=a["kind"],
                    strength=a["strength"],
                    target=a["target"],
                    segments=a.get("segments"),
                    segment_share=a.get("segment_share", 0.25),
                    params=dict(a.get("params") or {}),
                ),
            )
            for a in applied
        ]
        ds = apply_pathologies(ds, int(patho["seed"]), plan=plan)
    elif patho.get("seed") is not None:
        ds = apply_pathologies(ds, int(patho["seed"]), plan=[])
    ds.test_dgp = recipe.get("test_dgp")
    ds.test_target_match = recipe.get("test_target_match")
    return ds
