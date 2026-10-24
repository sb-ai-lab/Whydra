import torch
import numpy as np
import torch.nn.functional as F

def matrix_poly(matrix, d):
    # Исправлено: .to(matrix.device)
    x = torch.eye(d).double().to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)

def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        # Исправлено: .to(preds.device)
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi).to(preds.device) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def preprocess_adj_new(adj):
    # Исправлено: .to(adj.device)
    adj_normalized = (torch.eye(adj.shape[0]).double().to(adj.device) - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    # Исправлено: .to(adj.device)
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double().to(adj.device) - adj.transpose(0,1))
    return adj_normalized


def my_softmax(input_tensor, axis=-1):
    """
    Applies a softmax function along an arbitrary axis.
    """
    #
    # Move the specified axis to the end, apply softmax, then move it back.
    #
    # This is equivalent to F.softmax(input_tensor, dim=axis) in modern PyTorch,
    # but we use the original implementation for consistency.
    transposed_input = input_tensor.transpose(axis, -1)
    softmaxed_output = F.softmax(transposed_input, dim=-1)
    return softmaxed_output.transpose(axis, -1)


def nll_categorical(preds_probs, target_long):
    """
    Negative log-likelihood for categorical data.
    preds_probs: [batch, n_vars, n_classes] tensor of probabilities
    target_long: [batch, n_vars] tensor of true class indices (long type)
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