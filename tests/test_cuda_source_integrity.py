import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (Path(__file__).parent / "fixtures" / "cuda_source_hashes.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("source", MANIFEST["files"])
def test_cuda_source_matches_pinned_hash(source):
    expected = MANIFEST["local_source_edits"].get(source, MANIFEST["files"][source])
    content = (ROOT / source).read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected, (
        f"{source} differs from the pinned snapshot from "
        f"muxas/cuda_dev@{MANIFEST['source_commit']}"
    )
