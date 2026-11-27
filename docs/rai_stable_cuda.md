# RAI-stable on CUDA

Use `StandaloneRAIStable(device="cuda")` to run search on the GPU. The algorithm returns a CPDAG. [Set up the CUDA environment](cuda.md) before running it.

```python
import numpy as np

from whydra import StandaloneRAIStable

rng = np.random.default_rng(12)
x = rng.normal(size=1000)
z = rng.normal(size=1000)
y = x + z + rng.normal(size=1000)
data = np.column_stack([x, y, z])

graph = StandaloneRAIStable(
    alpha=0.05,
    indep_test="fisherz",
    n_jobs=1,
    device="cuda",
    max_cond_set_size=2,
    certify_skeleton=False,
).run(data)

assert graph.graph_kind == "cpdag"
print([str(edge) for edge in graph.get_graph_edges()])
```

Passing `device="cuda"` sets `n_jobs=1` in the public wrapper. The Hydra configuration is `algorithm=rai_stable_cuda`.

## Execution model

`RAIStableCudaLearner` inherits from the NumPy reference class and preserves the CPU order for recursion, graph updates, and edge orientation.

- The graph and endpoint marks stay on CPU. The CPU also handles the loop over conditioning depths, graph partitioning, and edge orientation. The implementation uses the reference `_FastPDAG` class.
- Before each refinement phase, the algorithm forms batches and runs the tests with `CudaFisherZ` for Fisher-Z or `CudaChisqHost` for $\chi^2$.
- The CPU applies results in serial search order. It runs any missing test and recomputes results near `alpha` using the configured CPU test. Before enumerating conditioning sets, the adapter estimates the work. If the estimate exceeds the budget, RAI runs all tests in that phase on CPU.
- Tests that exceed the GPU limit on conditioning-set size use the fallback path. RAI applies their results in the same order.
- `max_cond_set_size` caps the conditioning-set size. RAI selects variables from the node's possible parents. These include parents and neighbors connected to the node by an undirected edge.
- The low-level entry point `rai_stable(..., ci_test=...)` accepts the built-in `CIT` with `method="fisherz"` or `method="chisq"` and unmodified methods. Fisher-Z uses the same correlation matrix for GPU precomputation and CPU checks. Custom functions, subclasses, and overridden test methods require CPU execution.
- `certify_skeleton=True` runs an extra pass through the CPU implementation of `SkeletonDiscovery`. This adds computation and may change the statistical result. Use the same setting when comparing CPU and CUDA.
- The search uses the current GPU and one CUDA stream. Select the device before the call and pass `device="cuda"`.
- Any `background_knowledge` value other than `None`, including an empty object, raises `NotImplementedError` before initialization. This applies to CPU and CUDA.

## Comparing results with CPU

Use serial RAI-stable (`device="cpu", n_jobs=1`) as the reference. Compare the following on the same data and with the same settings.

- The skeleton.
- The final CPDAG endpoint matrix.
- The members of the separating sets.

Compare separating sets without regard to element order.

## Source files

| File | Role |
| --- | --- |
| [`rai_algo.py`](../whydra/algorithms/standalone/rai_algo.py) | `rai_stable` entry point and `device="cuda"` dispatch |
| [`cpu_cuda_rai_bridge.py`](../whydra/algorithms/graph_core/cpu_cuda_rai_bridge.py) | Configured conditional independence test and estimate of how many tests to run in advance |
| [`cuda_rai.py`](../whydra/algorithms/graph_core/cuda_rai.py) | CUDA class, masks, loop over conditioning depths, and application of results |
| [`rai_host.cu`](../whydra/algorithms/graph_core/rai_host.cu) | CPU-side graph operations implemented in C++ |
| [`wrappers.py`](../whydra/algorithms/wrappers.py) | `StandaloneRAIStable` wrapper |
| [`rai_stable_cuda.yaml`](../configs/algorithm/rai_stable_cuda.yaml) | Hydra configuration |
