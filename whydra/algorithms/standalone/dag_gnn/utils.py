# Helpers adapted from DAG-GNN by Yue Yu et al. (Apache-2.0).
# Whydra builds tensors on the input device and adds a categorical loss.
# Upstream: https://github.com/fishmoon1234/DAG-GNN/blob/master/src/utils.py
import numpy as np
import torch
import torch.nn.functional as F


def matrix_poly(matrix, d):
    """Raise the sum of the identity and the scaled matrix to the power d."""
    # Build the identity on the same device as the matrix.
    x = torch.eye(d).double().to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def nll_gaussian(preds, target, variance, add_const=False):
    """Return the summed Gaussian reconstruction loss divided by the batch size."""
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(
        torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance)
    )
    if add_const:
        # Move the constant to the predictions' device before adding it to the loss.
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi).to(preds.device) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds):
    """Return half the squared norm of the predicted means per sample."""
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5


def preprocess_adj_new(adj):
    """Return the identity matrix minus the transpose of the adjacency matrix."""
    # The identity and the adjacency matrix must share a device.
    adj_normalized = torch.eye(adj.shape[0]).double().to(adj.device) - (
        adj.transpose(0, 1)
    )
    return adj_normalized


def preprocess_adj_new1(adj):
    """Invert the identity matrix minus the transpose of the adjacency matrix."""
    # The identity and the adjacency matrix must share a device.
    adj_normalized = torch.inverse(
        torch.eye(adj.shape[0]).double().to(adj.device) - adj.transpose(0, 1)
    )
    return adj_normalized


def my_softmax(input_tensor, axis=-1):
    """Apply softmax along the specified axis."""
    #
    # Move the specified axis to the end, apply softmax, then move it back.
    #
    # Moving the requested axis to the last dimension gives the same softmax
    # as F.softmax(input_tensor, dim=axis).
    transposed_input = input_tensor.transpose(axis, -1)
    softmaxed_output = F.softmax(transposed_input, dim=-1)
    return softmaxed_output.transpose(axis, -1)


def nll_categorical(preds_probs, target_long):
    """Compute the negative log-likelihood for categorical data.

    The probabilities have shape [batch, n_vars, n_classes]. The long-integer tensor of
    true class indices has shape [batch, n_vars].
    """
    # We need to gather the probabilities of the true classes
    # Unsqueeze target to add a dimension for gathering: [batch, n_vars, 1]
    target_indices = target_long.unsqueeze(-1)

    # Gather probabilities of the correct classes
    true_class_probs = torch.gather(preds_probs, -1, target_indices).squeeze(-1)

    # To avoid log(0), add a small epsilon
    log_probs = torch.log(true_class_probs + 1e-9)

    # Compute the negative log-likelihood and average over batch and variables
    return -log_probs.mean()
