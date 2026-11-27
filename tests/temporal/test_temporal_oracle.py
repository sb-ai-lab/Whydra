from itertools import permutations, product

import numpy as np

from whydra.algorithms import StandaloneJPCMCIPlus
from whydra.evaluation.metrics import dag_adj_to_cpdag_endpoint_matrix


def _labeled_dags():
    upper_triangle = np.triu_indices(4, k=1)
    seen = set()
    for edge_flags in product((False, True), repeat=6):
        ordered = np.zeros((4, 4), dtype=bool)
        ordered[upper_triangle] = edge_flags
        for labels in permutations(range(4)):
            dag = ordered[np.ix_(labels, labels)]
            key = dag.tobytes()
            if key not in seen:
                seen.add(key)
                yield dag


def test_joint_search_recovers_equivalence_from_exact_linear_model_moments():
    # Centered, orthogonal disturbances give the model covariance without sampling error.
    noise = np.vstack((np.eye(4), -np.eye(4)))
    weights = np.random.default_rng(6501).uniform(0.4, 1.4, size=(4, 4))
    # With exact moments the CI threshold should distinguish zero from even weak effects.
    model = StandaloneJPCMCIPlus(tau_max=0, alpha=1 - 1e-8)
    tested = 0
    for dag in _labeled_dags():
        observations = np.linalg.solve(np.eye(4) - (weights * dag).T, noise.T).T
        actual = model.run(observations).graph
        expected = dag_adj_to_cpdag_endpoint_matrix(dag)
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"DAG:\n{dag.astype(int)}"
        )
        tested += 1
    assert tested == 543
