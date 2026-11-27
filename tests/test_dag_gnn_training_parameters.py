import numpy as np
import pytest


@pytest.mark.torch
@pytest.mark.parametrize("penalty", [1e20, np.nan])
def test_initial_penalty_must_allow_a_training_step(penalty):
    pytest.importorskip("torch")
    from whydra.algorithms.standalone.dag_gnn.runner import train_dag_gnn

    with pytest.raises(ValueError, match="Initial penalty c_A"):
        train_dag_gnn(np.ones((4, 2)), c_A=penalty, k_max_iter=1, device="cpu")
