"""Apply Meek rules R1–R3 using the prebuilt CUDA extension compiled from ``meek.cu``."""

import numpy as np
import torch

_MEEK_CUDA = None


def _meek_cuda():
    global _MEEK_CUDA
    if _MEEK_CUDA is None:
        from ._cuda import meek

        _MEEK_CUDA = meek
    return _MEEK_CUDA


def orient_colliders_tensor(
    G: torch.Tensor,
    S: torch.Tensor,
    ranges: torch.Tensor,
) -> None:
    """In-place unshielded colliders on a resident CUDA endpoint matrix.

    ``G`` is ``int8`` (``-1`` tail / ``+1`` arrow). CSR ``S`` / ``ranges`` stay
    on device. Per center ``j`` the CTA scans unshielded sepset candidates,
    then one thread applies ``uc_priority=2`` (same order as
    ``find_unshielded_triples``). No host round-trip.
    """
    if not torch.is_tensor(G):
        msg = "G must be a CUDA int8 tensor"
        raise TypeError(msg)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        msg = "G must be square"
        raise ValueError(msg)
    if G.dtype != torch.int8 or not G.is_cuda:
        msg = "G must be CUDA int8"
        raise ValueError(msg)
    t = G if G.is_contiguous() else G.contiguous()
    S_t = S if S.is_contiguous() else S.contiguous()
    R_t = ranges if ranges.is_contiguous() else ranges.contiguous()
    if int(S_t.numel()) == 0:
        S_t = torch.zeros(1, dtype=torch.int64, device=G.device)
    _meek_cuda().orient_colliders(t, S_t, R_t)
    if t.data_ptr() != G.data_ptr():
        G.copy_(t)
    torch.cuda.synchronize(G.device)


def meek_orient_tensor(G: torch.Tensor, *, rules: int = 7) -> dict:
    """In-place Meek R1–R3 on a resident CUDA ``int8`` endpoint matrix.

    Same ``meek.cu`` kernel as ``meek_orient``: compact neighbors (multi-CTA),
    then one CTA of 256 threads, Gauss–Seidel R1–R3. R1 packs unshielded
    V-shapes ``(i,k)`` at center ``j`` once (multi-CTA count, scan, fill,
    CSR ``v_off[j]``), then applies in parallel inside each ``j``. R2 packs
    3-cliques ``i<j<k`` once (multi-CTA count, scan, fill into
    ``t_i``/``t_j``/``t_k``), then a 256-slot SMEM window over that
    list. Each slot holds a packed-list index; a hit ``atomicMin``s that
    index (not thread id). The thread whose ``g`` equals the min hit paints
    from registers. Slots at or before the hit (the whole window on a miss)
    pull the next triangles so every occupied slot re-tests.
    R3 packs kites ``(i,j,k,l)`` once in ``find_kites`` order, then the
    same packed-list window: the thread whose ``g`` equals the min hit
    paints ``i→l`` from registers. The suffix re-tests. Sequential
    gold is the original Python ``find_kites`` loop. No host round-trip.

    ``rules`` is a bitmask: 1=R1, 2=R2, 4=R3 (default all three).
    """
    if not torch.is_tensor(G):
        msg = "G must be a CUDA int8 tensor"
        raise TypeError(msg)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        msg = "G must be square"
        raise ValueError(msg)
    if G.dtype != torch.int8 or not G.is_cuda:
        msg = "G must be CUDA int8"
        raise ValueError(msg)
    t = G if G.is_contiguous() else G.contiguous()
    n_iters, n_orients = _meek_cuda().orient(t, int(rules))
    if t.data_ptr() != G.data_ptr():
        G.copy_(t)
    torch.cuda.synchronize(G.device)
    return {"meek_iters": int(n_iters), "meek_orients": int(n_orients)}


def meek_orient(G: np.ndarray, *, device: str | torch.device = "cuda") -> dict:
    """In-place Meek R1–R3 on ``G``. Computes on CUDA, writes back to ``G``.

    Uploads ``G`` as contiguous ``int8``. ``meek.cu`` first compacts
    neighbors (multi-CTA) then launches **one CTA of 256 threads**:
    Gauss–Seidel R1/R2/R3 so the CPDAG matches the Python Meek loop.
    R1 packs unshielded V-shapes once, CSR by center ``j``. R2 packs triangles
    once, then a 256-slot SMEM window keyed by packed-list index. R3 packs
    kites once and applies them in ``find_kites`` order. Integer compares / ``int8``
    stores only; adjacency (nonzero) is frozen.
    """
    G_np = np.asarray(G)
    if G_np.ndim != 2 or G_np.shape[0] != G_np.shape[1]:
        msg = "G must be square"
        raise ValueError(msg)
    t = torch.from_numpy(np.ascontiguousarray(G_np, dtype=np.int8)).to(
        torch.device(device)
    )
    stats = meek_orient_tensor(t)
    np.copyto(G_np, t.detach().cpu().numpy())
    return stats


__all__ = [
    "meek_orient",
    "meek_orient_tensor",
    "orient_colliders_tensor",
]
