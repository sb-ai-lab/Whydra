<h1>
  <img src="docs/whydra-logo_black.svg#gh-dark-mode-only" width="640" alt="Whydra — Hybrid Discovery Toolkit">
  <img src="docs/whydra-logo_white.svg#gh-light-mode-only" width="640" alt="Whydra — Hybrid Discovery Toolkit">
</h1>

[![Public main CI](https://github.com/sb-ai-lab/Whydra/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/sb-ai-lab/Whydra/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![Python line coverage from CPU tests: 63.83%](https://img.shields.io/badge/coverage-63.83%25-blue)](#test-coverage)
[![PyPI version](https://img.shields.io/pypi/v/whydra)](https://pypi.org/project/whydra/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/whydra)](https://pypi.org/project/whydra/)
[![Documentation](https://img.shields.io/badge/docs-guides%20%26%20API-blue)](https://sb-ai-lab.github.io/Whydra/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Website: whydra.ai](https://img.shields.io/badge/whydra.ai-006145)](https://whydra.ai/)

Whydra learns causal graphs from tabular data and time series. Selected algorithms
support parallel execution and background knowledge.

## Installation

Whydra requires Python 3.10 or later. Install it from PyPI:

```sh
pip install whydra
```

### Install from source

After cloning the repository, run the following command from its root directory:

```sh
python -m pip install .
```

With uv, run `uv sync`. To build a wheel and source distribution, run
`uv build` or install the development dependencies and run `python -m build`.

To save graph images with `whydra.evaluation.graph_utils.draw_graph`, install
the Python dependencies with `python -m pip install '.[viz]'`. You also need
Graphviz on your system.

- On Debian or Ubuntu, run `sudo apt-get install graphviz`.
- On Windows, run the [Graphviz installer](https://graphviz.org/download/)
  and add its `bin` directory to `PATH`.

## Quickstart

Basic usage example, see more in the documentation:

```python
import numpy as np
from whydra import StandalonePCStable

rng = np.random.default_rng(0)
x1, x2 = rng.normal(size=(2, 1000))
x3 = x1 + x2 + rng.normal(scale=0.5, size=1000)
data = np.column_stack([x1, x2, x3])  # (n_samples, n_features)

graph = StandalonePCStable(alpha=0.05, indep_test="fisherz", n_jobs=4).run(data)
print([str(edge) for edge in graph.get_graph_edges()])  # ['X1 --> X3', 'X2 --> X3']
```

## Algorithms

Pass one of these names to `whydra.create_standalone_algorithm()`, for example
`whydra.create_standalone_algorithm("rfci_stable", alpha=0.01)`.

| Algorithm | Factory names |
| --- | --- |
| PC-stable | `pc_stable`, `pc_stable_parallel` |
| FCI-stable | `fci_stable`, `fci_stable_cached`, `fci_stable_parallel_cached` |
| Conservative FCI | `cfci_stable`, `cfci_stable_parallel_cached` |
| RFCI | `rfci_stable` |
| GFCI | `gfci_stable` |
| FCI+ | `fci_plus_stable`, `fci_plus_stable_parallel_cached` |
| RAI (experimental) | `rai_stable` |
| DAG-GNN (requires PyTorch) | `dag_gnn` |
| PCMCI | `pcmci` |
| PCMCI+ | `pcmci_plus` |
| J-PCMCI+ | `j_pcmci_plus` |

`whydra.available_standalone_algorithms()` lists the corresponding class names.
For PC and the FCI family, use `fisherz` for continuous data and `chisq` for discrete data.

For Greedy Equivalence Search (GES), import `ges_search` from
`whydra.algorithms.score_based`.

## CUDA

PC-stable, FCI-stable, and RAI-stable support `device="cuda"`. With CUDA-enabled
PyTorch and the CUDA toolkit installed, build the extensions from the repository root.

```sh
python setup.py build_ext --inplace
```

See the [CUDA guide](docs/cuda.md) for supported tests, device selection, and limitations.


## Development

```sh
python -m pip install -e '.[dev,viz,comparison,torch]'
pre-commit install
pre-commit run --all-files
python -m pytest
```

On Windows, the `torch` extra needs
[long path support](https://pip.pypa.io/warnings/enable-long-paths)
enabled, otherwise pip stops while unpacking PyTorch.

With uv, run

```sh
uv sync --extra dev --extra viz --extra comparison --extra torch
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run --all-files
uv run --no-sync pytest
```

Install `.[torch]` for DAG-GNN and `.[docs]` to build the documentation.

### Test coverage

On 2026-11-26, the CPU test suite (`pytest -m "not cuda"`) exercised 14,923 of
23,379 executable Python lines in `whydra` (63.83%). The measurement includes
worker processes and excludes native C++ and CUDA code. CI has no minimum
coverage gate.

## Contributors

<p>
<a href="https://github.com/dwemer8"><img src="docs/contributors/dwemer8.png" alt="dwemer8" width="48" height="48"></a>
<a href="https://github.com/Eighonet"><img src="docs/contributors/Eighonet.png" alt="Eighonet" width="48" height="48"></a>
<a href="https://github.com/lange-am"><img src="docs/contributors/lange-am.png" alt="lange-am" width="48" height="48"></a>
<a href="https://github.com/Muxas"><img src="docs/contributors/Muxas.png" alt="Muxas" width="48" height="48"></a>
<a href="https://github.com/MVVladimir"><img src="docs/contributors/MVVladimir.png" alt="MVVladimir" width="48" height="48"></a>
<a href="https://github.com/promsoft"><img src="docs/contributors/promsoft.png" alt="promsoft" width="48" height="48"></a>
<a href="https://github.com/sergeyklitnyy2005"><img src="docs/contributors/sergeyklitnyy2005.png" alt="sergeyklitnyy2005" width="48" height="48"></a>
<a href="https://github.com/Widlar"><img src="docs/contributors/Widlar.png" alt="Widlar" width="48" height="48"></a>
</p>

Early contributors also include alaptev, Anna Raginova, and
Vladimir Kocheshkov. See the [full contributor list](CONTRIBUTORS.md).

## License and code provenance

Whydra's original code is licensed under the [Apache License, Version 2.0](LICENSE).
See the [provenance inventory](docs/CODE_PROVENANCE.md) for third-party licenses and attributions.
