import numpy as np

from whydra.algorithms.wrappers import StandaloneCFCIStable


def test_cfci_pag_endpoint_codes_are_mapped_to_causallearn_endpoints():
    # Кодировка causal-learn: NULL=0, TAIL=-1, ARROW=1, CIRCLE=2.
    # X1 --> X2: стрелка у X2, хвост у X1.
    pag = np.array(
        [
            [0, 1],
            [-1, 0],
        ],
        dtype=int,
    )

    graph = StandaloneCFCIStable(n_jobs=1)._matrix_to_graph(pag, 2)

    assert [str(edge) for edge in graph.get_graph_edges()] == ["X1 --> X2"]
