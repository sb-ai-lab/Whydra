# Training loop adapted from DAG-GNN by Yue Yu et al. (Apache-2.0).
# Whydra exposes a callable trainer with device selection and discrete-data support.
# Upstream: https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py
import sys
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import torch

_TRAINING_EXPORTS = {
    "optim": ("torch.optim", None),
    "trange": ("tqdm", "trange"),
    **{
        name: (".modules", name)
        for name in (
            "MLPEncoder",
            "MLPDecoder",
            "MLPDiscreteEncoder",
            "MLPDiscreteDecoder",
        )
    },
    **{
        name: (".utils", name)
        for name in (
            "nll_gaussian",
            "nll_categorical",
            "kl_gaussian_sem",
            "matrix_poly",
        )
    },
}
if TYPE_CHECKING:
    from torch import optim
    from tqdm import trange

    from .modules import MLPDecoder, MLPDiscreteDecoder, MLPDiscreteEncoder, MLPEncoder
    from .utils import kl_gaussian_sem, matrix_poly, nll_categorical, nll_gaussian

__all__ = [
    "MLPDecoder",
    "MLPDiscreteDecoder",
    "MLPDiscreteEncoder",
    "MLPEncoder",
    "kl_gaussian_sem",
    "matrix_poly",
    "nll_categorical",
    "nll_gaussian",
    "np",
    "optim",
    "stau",
    "torch",
    "train_dag_gnn",
    "trange",
]


def __getattr__(name):
    target = _TRAINING_EXPORTS.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = import_module(target[0], __package__)
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_TRAINING_EXPORTS))


def stau(w, tau):
    """Soft-threshold each weight at tau while preserving its sign."""
    prox_plus = torch.nn.Threshold(0.0, 0.0)
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def train_dag_gnn(
    data: np.ndarray,
    data_type: str = "continuous",
    lambda_A=0.0,
    c_A=1.0,
    h_tol=1e-8,
    epochs=300,
    batch_size=100,
    lr=3e-3,
    graph_threshold=0.3,
    tau_A=0.0,
    k_max_iter=100,
    l2_reg_diag=100.0,
    encoder_hidden=64,
    decoder_hidden=64,
    device=None,
):
    """Train DAG-GNN on the supplied data and return the learned adjacency matrix."""
    # The public constructor checks the torch dependency, but models, optimizer
    # and progress support are needed only when training actually starts.
    # Read through the module so caller-supplied training implementations win.
    runtime = sys.modules[__name__]
    for name in _TRAINING_EXPORTS:
        if name not in globals():
            __getattr__(name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples, n_nodes = data.shape

    # Draw from the shared NumPy stream so callers can reproduce the initial weights.
    init_rng = np.random.RandomState(np.random.get_bit_generator())
    adj_A_init = init_rng.uniform(low=-0.1, high=0.1, size=(n_nodes, n_nodes))
    np.fill_diagonal(adj_A_init, 0)

    z_dims = 1

    if data_type == "continuous":
        x_dims = 1
        encoder = (
            runtime.MLPEncoder(
                n_nodes, x_dims, encoder_hidden, z_dims, adj_A_init, batch_size
            )
            .double()
            .to(device)
        )
        decoder = runtime.MLPDecoder(z_dims, decoder_hidden, x_dims).double().to(device)
    elif data_type == "discrete":
        n_categories = np.max(data).astype(int) + 1
        encoder = (
            runtime.MLPDiscreteEncoder(
                n_nodes, n_categories, encoder_hidden, z_dims, adj_A_init
            )
            .double()
            .to(device)
        )
        decoder = (
            runtime.MLPDiscreteDecoder(z_dims, decoder_hidden, n_categories, n_nodes)
            .double()
            .to(device)
        )
    else:
        msg = "data_type must be 'continuous' or 'discrete'"
        raise ValueError(msg)

    optimizer = runtime.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    data_torch = torch.from_numpy(data).to(device)

    if k_max_iter > 0 and (np.isnan(c_A) or c_A >= 1e20):
        msg = "Initial penalty c_A must be below 1e20 to start DAG-GNN training"
        raise ValueError(msg)

    h_A_old = np.inf

    for _ in runtime.trange(k_max_iter, desc="Augmented Lagrangian"):
        if c_A >= 1e20:
            break
        while True:
            for _epoch in range(epochs):
                for j in range(0, n_samples, batch_size):
                    batch_raw = data_torch[j : j + batch_size]
                    if batch_raw.size(0) <= 1:
                        continue

                    if data_type == "continuous":
                        batch_input = batch_raw.double().unsqueeze(-1)
                        batch_target = batch_input
                    else:
                        batch_input = batch_raw.long()  # The encoder expects indices.
                        batch_target = batch_raw.long()

                    optimizer.zero_grad()

                    _, logits, origin_A, myA, wa = encoder(batch_input)
                    _, preds = decoder(logits, origin_A, wa)

                    if data_type == "continuous":
                        loss_nll = runtime.nll_gaussian(preds, batch_target, 0.0)
                    else:
                        loss_nll = runtime.nll_categorical(preds, batch_target)

                    loss_kl = runtime.kl_gaussian_sem(logits)
                    h_A = (
                        torch.trace(runtime.matrix_poly(origin_A * origin_A, n_nodes))
                        - n_nodes
                    )
                    sparse_loss = tau_A * torch.sum(torch.abs(origin_A))
                    diag_reg = l2_reg_diag * torch.trace(origin_A * origin_A)

                    loss = (
                        loss_nll
                        + loss_kl
                        + lambda_A * h_A
                        + 0.5 * c_A * h_A * h_A
                        + sparse_loss
                        + diag_reg
                    )

                    loss.backward()
                    optimizer.step()

                    if tau_A > 0:
                        myA.data = stau(myA.data, tau_A * lr)

            with torch.no_grad():
                h_A_new = (
                    torch.trace(
                        runtime.matrix_poly(
                            encoder.adj_A.data * encoder.adj_A.data, n_nodes
                        )
                    )
                    - n_nodes
                )

            if h_A_new.item() > 0.25 * h_A_old:
                c_A *= 10
                if c_A >= 1e20:
                    break
            else:
                break

        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    final_adj_matrix = encoder.adj_A.data.cpu().numpy()
    final_adj_matrix[np.abs(final_adj_matrix) < graph_threshold] = 0
    return final_adj_matrix
