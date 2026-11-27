# CUDA algorithms

PC-stable, FCI-stable, and RAI-stable support CUDA with the Fisher-Z and $\chi^2$ conditional independence tests.

| Algorithm | Hydra configuration | Example |
| --- | --- | --- |
| PC-stable-cuda | `algorithm=pc_stable_cuda` | [PC-stable on CUDA](pc_stable_cuda.md) |
| FCI-stable-cuda | `algorithm=fci_stable_cuda` | [FCI-stable on CUDA](fci_stable_cuda.md) |
| RAI-stable-cuda | `algorithm=rai_stable_cuda` | [RAI-stable on CUDA](rai_stable_cuda.md) |

In Python, pass `device="cuda"`. The `create_standalone_algorithm` factory accepts `pc_stable`, `fci_stable`, and `rai_stable`. The names ending in `_cuda` in the table are Hydra configurations.

[Third-party component sources](CODE_PROVENANCE.md).

## CUDA setup

Install a CUDA-enabled build of PyTorch and `setuptools>=77.0.3` in the same environment. The build requires the CUDA toolkit with `nvcc` and a compatible C++ compiler. Run the command from the repository root. The `--no-build-isolation` flag lets the build use the installed PyTorch.

```bash
WHYDRA_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation
```

```powershell
$env:WHYDRA_BUILD_CUDA = "1"
python -m pip install -e . --no-build-isolation
```

Set `CUDA_HOME` if the toolkit is installed in a nonstandard directory. Use `TORCH_CUDA_ARCH_LIST` to specify target GPU architectures. `WHYDRA_ENABLE_FLOAT32=1` adds `float32` Fisher-Z kernels. [Build settings](../setup.py).

## CPU integration

Fisher-Z runs the test on CPU if the conditioning-set size exceeds the GPU limit. The unregularized test uses the configured `CIT` when it falls back to CPU.

RAI runs tests on CPU if the estimated work exceeds the GPU budget. The extra RAI pass with `certify_skeleton=True` also runs on CPU.

CUDA supports the built-in `fisherz` and `chisq` tests. Use CPU mode with an algorithm that supports custom functions or overridden tests.

## Graph encoding

The resulting `graph.graph` matrix stores endpoint marks.

| Value | Whydra endpoint mark |
| ---: | --- |
| `0` | No edge |
| `-1` | Tail |
| `1` | Arrowhead |
| `2` | Circle |

For example, for an edge $i \to j$, `graph.graph[i, j]` is `1` and `graph.graph[j, i]` is `-1`.

## Limitations

CUDA mode uses one CPU worker. `device="cpu"` and `device=None` select CPU execution. `max_cond_set_size` caps the conditioning-set size. The memory required for the dense graph grows in proportion to $p^2$, where $p$ is the number of variables.

**Stable variants.** CUDA PC supports `stable=True`. `stable=False` raises `ValueError`. Use serial CPU PC to run with `stable=False`.

**Memory for separating sets.** Working buffers grow during search. If allocation fails, the CUDA implementation raises `RuntimeError` and reports the buffer name, required memory, and available GPU memory in MiB.

**Single CUDA device.** The algorithms use the current GPU and one CUDA stream. Select the GPU first with `torch.cuda.set_device(N)` or `CUDA_VISIBLE_DEVICES`, then pass `device="cuda"`.

**Background knowledge.** Use CPU mode to pass background knowledge to PC or FCI. CUDA modes accept `background_knowledge=None`. RAI accepts only `None` on both platforms.

[PC-stable limitations](pc_stable_cuda.md#limitations) &middot; [FCI-stable limitations](fci_stable_cuda.md#limitations) &middot; [RAI-stable execution mode](rai_stable_cuda.md).

## Comparing CPU and CUDA results

Run both implementations on the same data with the same settings. Set `device="cpu"` and `n_jobs=1` for serial CPU execution.

Compare the final CPDAG for PC and RAI or PAG for FCI. Compare the skeletons and the members of separating sets separately. Use a numerical tolerance when comparing p-values and `pMax` in FCI.
