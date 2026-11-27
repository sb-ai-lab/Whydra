# Code provenance and licensing

Whydra's original code is distributed under the [Apache License 2.0](../LICENSE).
The source distribution and wheel include the project license and
third-party license notices. `THIRD_PARTY_LICENSES/README.md` lists the notices.

## Code in the library

### RAI and Intel Labs causality-lab

The RAI implementation is based on `LearnStructRAI` and the
`MixedGraph` and `PDAG` graph primitives from Intel Labs causality-lab.
Whydra adapted them to its graph representation and conditional
independence tests with parallel execution. `THIRD_PARTY_LICENSES/Apache-2.0.txt`
contains the Intel Labs causality-lab license.

[Whydra implementation](../whydra/algorithms/standalone/rai_algo.py) &middot;
[Intel Labs reference source and license](https://github.com/IntelLabs/causality-lab/tree/36625da6eeef059e36dab2b4467235a036136b76)

### DAG-GNN

Whydra adapted several helper functions from DAG-GNN, including
`kl_gaussian_sem`, `nll_gaussian`, and `preprocess_adj_new1`. The adaptations
change device handling and add support for discrete data. The Apache-2.0
license text for DAG-GNN matches `THIRD_PARTY_LICENSES/Apache-2.0.txt`.

The `my_softmax` implementation follows NRI &rarr; DAG-GNN &rarr; Whydra.
The function matches in the pinned NRI and DAG-GNN sources, and DAG-GNN
credits NRI as one of its code sources. Whydra moves the requested axis
to the last position, applies `F.softmax(..., dim=-1)`, and restores the
original axis order. The default is the last axis. NRI distributes its
code under the MIT license with a copyright notice for Ethan Fetaya
and Thomas Kipf.

[Whydra helpers](../whydra/algorithms/standalone/dag_gnn/utils.py) &middot;
[DAG-GNN source helpers](https://github.com/fishmoon1234/DAG-GNN/blob/2f8688908d9ad3d85879feb97ff59ff1f73ed151/src/utils.py) &middot;
[DAG-GNN acknowledgments and license](https://github.com/fishmoon1234/DAG-GNN/tree/2f8688908d9ad3d85879feb97ff59ff1f73ed151) &middot;
[NRI attribution in DAG-GNN](https://github.com/fishmoon1234/DAG-GNN/blob/2f8688908d9ad3d85879feb97ff59ff1f73ed151/README.md#L42-L46) &middot;
[`my_softmax` in NRI](https://github.com/ethanfetaya/NRI/blob/ee546d931b68382f415633434e04ea06f1201f14/utils.py#L9-L12) &middot;
[`my_softmax` in DAG-GNN](https://github.com/fishmoon1234/DAG-GNN/blob/2f8688908d9ad3d85879feb97ff59ff1f73ed151/src/utils.py#L200-L203) &middot;
[Full NRI license](https://github.com/ethanfetaya/NRI/blob/ee546d931b68382f415633434e04ea06f1201f14/LICENSE)

### FCI v2, PC, and causal-learn

FCI v2 uses adapted versions of `existsSemiDirectedPath` and `getPossibleDsep`
from causal-learn. The `to_causallearn` adapter converts Whydra graphs to the
causal-learn format. This lets you compare the results of PC and orientation
with Meek's rules against the results from causal-learn.
`THIRD_PARTY_LICENSES/causal-learn-MIT.txt` contains the MIT license and
the causal-learn copyright notice (copyright 2022 cmu-phil).

[FCI source in causal-learn](https://github.com/py-why/causal-learn/blob/0.1.4.5/causallearn/search/ConstraintBased/FCI.py#L153-L244) &middot;
[MIT notice](https://github.com/py-why/causal-learn/blob/0dacacf36390e3084704636bc3c36e82e35ee0cc/LICENSE)

### SCMGen numerical functions

`truth.py` contains the coefficients and implementation of Wichura's
AS241 inverse normal cumulative distribution function. `betafn.py`
contains Lanczos coefficients for the log-gamma function.

[AS241 in Whydra](../whydra/scmgen/truth.py) &middot;
[Lanczos in Whydra](../whydra/scmgen/betafn.py) &middot;
[Wichura's paper](https://doi.org/10.2307/2347330)

### CUDA

The original set of CUDA files comes from revision
`muxas/cuda_dev@99012c91d0c9aa0d4b33add32df0cccd2b26d5a8`.
The manifest records SHA-256 checksums for all 24 upstream files and pins
the local copies with Whydra integration, Python lint and typing, or Windows
compiler compatibility changes. The source integrity test checks each local
file byte-for-byte against its expected snapshot.

[CUDA guide](cuda.md) &middot;
[Manifest](../tests/fixtures/cuda_source_hashes.json) &middot;
[Source integrity test](../tests/test_cuda_source_integrity.py)

## Illustrations and materials

### Documentation illustrations

The PC diagram reproduces Figure 11 from
*A Survey on Causal Discovery Methods for I.I.D. and Time Series Data*
by Uzma Hasan, Emam Hossain, and Md Osman Gani. Whydra resized the
figure and adapted its caption. The paper is licensed under CC BY 4.0.

[Caption in the documentation](./source/theory/constraint_based.rst) &middot;
[Original figure](https://openreview.net/pdf?id=YdMrdhGx9y#page=11) &middot;
[Paper and license information](https://arxiv.org/abs/2303.15027) &middot;
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

The README shows the Whydra logo and avatars from the contributors' GitHub profiles.

[Contributor instructions](../CONTRIBUTORS.md)
