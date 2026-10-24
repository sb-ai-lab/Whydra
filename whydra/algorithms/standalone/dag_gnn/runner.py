import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from .modules import MLPEncoder, MLPDecoder, MLPDiscreteEncoder, MLPDiscreteDecoder
from .utils import nll_gaussian, nll_categorical, kl_gaussian_sem, matrix_poly


def stau(w, tau):
    prox_plus = torch.nn.Threshold(0., 0.)
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def train_dag_gnn(
        data: np.ndarray,
        data_type: str = 'continuous',
        lambda_A=0.0, c_A=1.0, h_tol=1e-8,
        epochs=300, batch_size=100, lr=3e-3,
        graph_threshold=0.3, tau_A=0.0,
        k_max_iter=100, l2_reg_diag=100.0,
        encoder_hidden=64, decoder_hidden=64,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples, n_nodes = data.shape

    adj_A_init = np.random.uniform(low=-0.1, high=0.1, size=(n_nodes, n_nodes))
    np.fill_diagonal(adj_A_init, 0)

    z_dims = 1

    if data_type == 'continuous':
        x_dims = 1
        encoder = MLPEncoder(n_nodes, x_dims, encoder_hidden, z_dims, adj_A_init, batch_size).double().to(device)
        decoder = MLPDecoder(z_dims, decoder_hidden, x_dims).double().to(device)
        reconstruction_loss_fn = nll_gaussian
    elif data_type == 'discrete':
        n_categories = np.max(data).astype(int) + 1
        encoder = MLPDiscreteEncoder(n_nodes, n_categories, encoder_hidden, z_dims, adj_A_init).double().to(device)
        decoder = MLPDiscreteDecoder(z_dims, decoder_hidden, n_categories, n_nodes).double().to(device)
        reconstruction_loss_fn = nll_categorical
    else:
        raise ValueError("data_type must be 'continuous' or 'discrete'")

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    data_torch = torch.from_numpy(data).to(device)

    h_A_old = np.inf

    for _ in trange(k_max_iter, desc="Augmented Lagrangian"):
        while c_A < 1e+20:
            for epoch in range(epochs):
                for j in range(0, n_samples, batch_size):
                    batch_raw = data_torch[j:j + batch_size]
                    if batch_raw.size(0) <= 1: continue

                    if data_type == 'continuous':
                        batch_input = batch_raw.double().unsqueeze(-1)
                        batch_target = batch_input
                    else:
                        batch_input = batch_raw.long()  # Encoder ожидает индексы
                        batch_target = batch_raw.long()

                    optimizer.zero_grad()

                    _, logits, origin_A, myA, wa = encoder(batch_input)
                    _, preds = decoder(logits, origin_A, wa)

                    if data_type == 'continuous':
                        loss_nll = reconstruction_loss_fn(preds, batch_target, 0.0)
                    else:
                        loss_nll = reconstruction_loss_fn(preds, batch_target)

                    loss_kl = kl_gaussian_sem(logits)
                    h_A = torch.trace(matrix_poly(origin_A * origin_A, n_nodes)) - n_nodes
                    sparse_loss = tau_A * torch.sum(torch.abs(origin_A))
                    diag_reg = l2_reg_diag * torch.trace(origin_A * origin_A)

                    loss = loss_nll + loss_kl + lambda_A * h_A + 0.5 * c_A * h_A * h_A + sparse_loss + diag_reg

                    loss.backward()
                    optimizer.step()

                    if tau_A > 0:
                        myA.data = stau(myA.data, tau_A * lr)

            with torch.no_grad():
                h_A_new = torch.trace(matrix_poly(encoder.adj_A.data * encoder.adj_A.data, n_nodes)) - n_nodes

            if h_A_new.item() > 0.25 * h_A_old:
                c_A *= 10
            else:
                break

        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    final_adj_matrix = encoder.adj_A.data.cpu().numpy()
    final_adj_matrix[np.abs(final_adj_matrix) < graph_threshold] = 0
    return final_adj_matrix