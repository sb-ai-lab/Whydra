import subprocess
import sys
from textwrap import dedent


def test_run_annotations_resolve_without_eagerly_loading_pandas():
    script = dedent(
        """
        import sys
        from typing import get_type_hints

        from whydra.algorithms.base import CausalAlgorithm
        from whydra.algorithms.wrappers import StandalonePCStable, run_standalone_algorithm

        assert "pandas" not in sys.modules
        methods = (CausalAlgorithm.run, StandalonePCStable.run, run_standalone_algorithm)
        inputs = [get_type_hints(method) for method in methods]

        import numpy as np
        from pandas import DataFrame
        from pathlib import Path

        assert inputs[0]["data"] == np.ndarray | DataFrame
        assert inputs[1]["data"] == np.ndarray | DataFrame
        assert inputs[2]["data_or_path"] == np.ndarray | DataFrame | str | Path
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
