import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._pag_cases import check_cuda_rules


@pytest.mark.cuda
def test_r10_changes_exactly_the_expected_endpoints():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    check_cuda_rules(10)


def main() -> int:
    check_cuda_rules(7, 8, 9, 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
