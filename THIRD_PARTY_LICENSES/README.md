# Third-party notices for Whydra

Whydra's original code is licensed under the [Apache License, Version 2.0](../LICENSE).
The source and wheel distributions also include the following notices for
upstream works. Each notice applies to the corresponding component.

- **Intel Labs causality-lab.** Whydra's RAI implementation identifies itself as a port of `LearnStructRAI`, `MixedGraph` and `PDAG`. [Upstream RAI source](https://github.com/IntelLabs/causality-lab/blob/36625da6eeef059e36dab2b4467235a036136b76/causal_discovery_algs/rai.py), [upstream license](https://github.com/IntelLabs/causality-lab/blob/36625da6eeef059e36dab2b4467235a036136b76/LICENSE).
- **DAG-GNN.** Whydra's DAG-GNN includes adapted utility functions corresponding to the implementation by the authors of DAG-GNN. Intel Labs and DAG-GNN publish byte-identical Apache License 2.0 texts. Both are represented by the bundled `Apache-2.0.txt`. [DAG-GNN utility functions](https://github.com/fishmoon1234/DAG-GNN/blob/2f8688908d9ad3d85879feb97ff59ff1f73ed151/src/utils.py), [upstream license](https://github.com/fishmoon1234/DAG-GNN/blob/2f8688908d9ad3d85879feb97ff59ff1f73ed151/LICENSE).
- **causal-learn.** Whydra's FCI path-search code includes ports of `existsSemiDirectedPath` and `getPossibleDsep`. The full MIT notice, copyright 2022 cmu-phil, is in `causal-learn-MIT.txt`. [Upstream license](https://github.com/py-why/causal-learn/blob/0dacacf36390e3084704636bc3c36e82e35ee0cc/LICENSE).

The [provenance inventory](../docs/CODE_PROVENANCE.md) records the open FCI
provenance questions and the history of removed components.
