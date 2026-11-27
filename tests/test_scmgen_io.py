import json

import pytest

from whydra.scmgen import generate, load_dataset, save_dataset, scenarios
from whydra.scmgen.io import current_version


@pytest.mark.parametrize(
    "name", ["../data", "/etc", r"C:\data", r"..\data", r"v-ok\..\data"]
)
def test_current_requires_a_local_version_name(tmp_path, name):
    (tmp_path / "CURRENT").write_text(name, encoding="utf-8")
    with pytest.raises(ValueError, match="CURRENT"):
        current_version(tmp_path)


def test_manifest_cannot_hash_a_parent_file(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside the dataset")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "observed.parquet").touch()
    (bundle / "manifest.json").write_text(
        json.dumps({"manifest_schema": 2, "files": {"../outside": {"sha256": "0"}}}),
        encoding="utf-8",
    )
    with (
        pytest.warns(UserWarning, match="CURRENT"),
        pytest.raises(ValueError, match="unexpected file"),
    ):
        load_dataset(bundle, verify="hashes")


@pytest.mark.parametrize(
    "name", ["CURRENT", "observed.parquet", "manifest.json", "v-version"]
)
def test_external_bundle_symlink_is_rejected(tmp_path, name):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    target = tmp_path / "outside"
    if name == "v-version":
        target.mkdir()
        (bundle / "CURRENT").write_text(name, encoding="utf-8")
    else:
        target.touch()
        if name == "manifest.json":
            (bundle / "observed.parquet").touch()
    try:
        (bundle / name).symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        pytest.skip(f"creating symlinks requires filesystem support: {error}")
    with pytest.raises(ValueError, match="escapes"):
        load_dataset(bundle)


def test_scenario_lookup_accepts_only_packaged_names(tmp_path):
    (tmp_path / "foreign.yaml").touch()
    with pytest.raises(KeyError):
        scenarios.load(str(tmp_path / "foreign"))


def test_versioned_dataset_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    dataset = generate(scenarios.load("rct-noise-y-seg"), n=32, seed=42)
    root = save_dataset(dataset, tmp_path / "dataset")
    observed, truth, meta = load_dataset(root, verify="hashes")
    assert observed.equals(dataset.to_frame())
    assert len(truth) == meta["n"] == 32
