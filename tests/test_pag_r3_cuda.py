import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._pag_cases import check_cuda_rules


def main() -> int:
    check_cuda_rules(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
