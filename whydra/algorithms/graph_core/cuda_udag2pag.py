"""CUDA udag2pag local rules. PAG marks: 0 none, 1 circle, 2 arrow, 3 tail."""

import numpy as np
import torch

_UDAG2PAG_CUDA = None


def _udag2pag_cuda():
    """Prebuilt ``udag2pag.cu`` (PAG R1–R10, one CTA of 256 threads per rule)."""
    global _UDAG2PAG_CUDA
    if _UDAG2PAG_CUDA is None:
        from ._cuda import udag2pag

        _UDAG2PAG_CUDA = udag2pag
    return _UDAG2PAG_CUDA


def pack_r1(pag: np.ndarray, *, device: str | torch.device = "cuda"):
    """Packed R1 freeze pairs ``(a,b)`` in sequential ``(b, a)`` order. Host numpy ``(T, 2)``."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    a, b = ext.pack_r1(t)
    if int(a.numel()) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return torch.stack([a, b], dim=1).detach().cpu().numpy()


def pack_r2(pag: np.ndarray, *, device: str | torch.device = "cuda"):
    """Packed R2 freeze pairs ``(a,c)`` in sequential ``(c, a)`` order. Host numpy ``(T, 2)``."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    a, c = ext.pack_r2(t)
    if int(a.numel()) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return torch.stack([a, c], dim=1).detach().cpu().numpy()


def pack_r3(pag: np.ndarray, *, device: str | torch.device = "cuda"):
    """Packed R3 freeze pairs ``(b,d)`` in sequential ``(d, b)`` order. Host numpy ``(T, 2)``."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    b, d = ext.pack_r3(t)
    if int(b.numel()) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return torch.stack([b, d], dim=1).detach().cpu().numpy()


def apply_r1(
    pag: np.ndarray,
    unfVect=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R1 on a dense PAG. Matches the sequential `ind` freeze + (b, a) order.

    CUDA: packs freeze ``(a,b)`` with ``pag[a,b]==2`` and reverse nonzero
    in ``(b, a)`` order, then **one CTA, 256 threads** grid-strides ``c``.
    Empty ``unfVect`` orients unconditionally (v2); a non-empty list skips
    unfaithful triples.
    """
    from .cuda_pdsep import _unf_tensor

    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)
    ext.r1(t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r2(
    pag: np.ndarray,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R2. Packs freeze ``(a,c)`` in ``(c, a)`` order, then the 256-slot window."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    ext.r2(t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r3(
    pag: np.ndarray,
    unfVect=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R3. Packs freeze ``(b,d)`` in ``(d, b)`` order, then the 256-slot window."""
    from .cuda_pdsep import _unf_tensor

    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)
    ext.r3(t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def pack_r5(pag: np.ndarray, *, device: str | torch.device = "cuda"):
    """Packed R5 freeze pairs ``(a,b)`` in sequential ``(b, a)`` order. Host numpy ``(T, 2)``."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    a, b = ext.pack_r5(t)
    if int(a.numel()) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return torch.stack([a, b], dim=1).detach().cpu().numpy()


def _sep_tensors(sepset, p: int, device: torch.device):
    if sepset is None:
        return (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.full((p, p, 2), -1, dtype=torch.int64, device=device),
        )
    S, ranges = sepset.packed()
    if torch.is_tensor(S):
        S_t = S.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
    else:
        S_t = torch.as_tensor(np.ascontiguousarray(S, dtype=np.int64), device=device)
    if torch.is_tensor(ranges):
        ranges_t = ranges.to(device=device, dtype=torch.int64).contiguous()
    else:
        ranges_t = torch.as_tensor(
            np.ascontiguousarray(ranges, dtype=np.int64), device=device
        )
    if int(S_t.numel()) == 0:
        S_t = torch.zeros(1, dtype=torch.int64, device=device)
    return S_t, ranges_t


def apply_r4(
    pag: np.ndarray,
    sepset=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R4. Packs the R3 pair list, then sequential ``minDiscPath`` apply."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    S_t, ranges_t = _sep_tensors(sepset, p, dev)
    ext.r4(t, S_t, ranges_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r5(
    pag: np.ndarray,
    unfVect=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R5. Packs o–o freeze pairs in ``(b, a)`` order, then sequential apply."""
    from .cuda_pdsep import _unf_tensor

    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)
    ext.r5(t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r6(
    pag: np.ndarray,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R6. Packs the R3 pair list; parallel witness OR, sequential apply."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    ext.r6(t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def pack_r8(pag: np.ndarray, *, device: str | torch.device = "cuda"):
    """Packed R8/R9/R10 freeze pairs ``(a,c)`` in sequential ``(c, a)`` order."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    a, c = ext.pack_r8(t)
    if int(a.numel()) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return torch.stack([a, c], dim=1).detach().cpu().numpy()


def apply_r7(
    pag: np.ndarray,
    unfVect=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R7. Packs the R3 pair list; parallel witness OR, sequential apply."""
    from .cuda_pdsep import _unf_tensor

    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)
    ext.r7(t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r8(
    pag: np.ndarray,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R8. Packs ``a *→ c`` with circle at ``a``; parallel OR, sequential apply."""
    ext = _udag2pag_cuda()
    dev = torch.device(device)
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    ext.r8(t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r9(
    pag: np.ndarray,
    unfVect=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R9. Packs ``a *→ c`` with circle at ``a``, then sequential ``minUncovPdPath``."""
    from .cuda_pdsep import _unf_tensor

    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)
    ext.r9(t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def apply_r10(
    pag: np.ndarray,
    unfVect=None,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place FCI R10. Same packed list as R8/R9; sequential apply (new parent of ``c``)."""
    from .cuda_pdsep import _unf_tensor

    ext = _udag2pag_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    t = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)
    ext.r10(t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, t.detach().cpu().numpy())


def _pcalg_to_whydra_(t: torch.Tensor) -> None:
    """Convert pcalg endpoint codes to Whydra codes in place.

    Circle, arrow, and tail map from (1, 2, 3) to (2, 1, -1).
    Zero denotes an absent edge in both encodings.
    """
    t.copy_(torch.where(t == 3, -1, torch.where(t == 2, 1, torch.where(t == 1, 2, t))))


def _whydra_to_pcalg_(t: torch.Tensor) -> None:
    """Convert Whydra endpoint codes to pcalg codes in place.

    Circle, arrow, and tail map from (2, 1, -1) to (1, 2, 3).
    Zero denotes an absent edge in both encodings.
    """
    t.copy_(torch.where(t == -1, 3, torch.where(t == 1, 2, torch.where(t == 2, 1, t))))


def apply_udag2pag(
    pag: np.ndarray,
    sepset,
    unfVect=None,
    rules=None,
    *,
    orientCollider: bool = True,
    device: str | torch.device = "cuda",
    r1r2_host=None,
    whydra_marks: bool = False,
) -> dict:
    """Full CUDA udag2pag: paint, colliders, R1–R10 on one resident int8 PAG.

    Paint uses PyTorch ``masked_fill_``. Colliders reuse ``pdsep.cu``
    (one thread per vertex ``x``). R1 packs freeze ``(a,b)`` and grid-strides
    ``c``. R2–R3 pack freeze lists and apply a 256-slot window. R4–R10 pack
    freeze lists; R4–R8 and R10 sequential-apply. R9 packed sequential ``g``
    (writes do not change other pairs' PD-paths).

    ``r1r2_host``: callback ``fn(pag_numpy)`` that applies R1+R2 in the
    causal-learn ``rulesR1R2cycle`` order on the host (whydra marks). While
    set, the device R1/R2 kernels are disabled and the callback runs once per
    rule-iteration between R0 and R3. ``whydra_marks``: return marks in the
    whydra encoding (-1/1/2) instead of pcalg (0/1/2/3).
    """
    from .cuda_pdsep import _pdsep_cuda

    if rules is None:
        rules = [True] * 10
    t = torch.as_tensor(
        np.ascontiguousarray(pag, dtype=np.int8), device=torch.device(device)
    )
    stats = _udag2pag_on_tensor(
        t,
        sepset,
        unfVect,
        rules,
        orientCollider=orientCollider,
        ext=_udag2pag_cuda(),
        pdsep=_pdsep_cuda(),
        r1r2_host=r1r2_host,
        whydra_marks=whydra_marks,
    )
    np.copyto(pag, t.detach().cpu().numpy())
    return stats


def apply_udag2pag_tensor(
    pag: torch.Tensor,
    sepset,
    unfVect=None,
    rules=None,
    *,
    orientCollider: bool = True,
    r1r2_host=None,
    whydra_marks: bool = False,
) -> dict:
    """Apply ``apply_udag2pag`` semantics to a resident CUDA ``int8`` PAG without copying it to the host."""
    if rules is None:
        rules = [True] * 10
    from .cuda_pdsep import _pdsep_cuda

    t = pag if pag.is_contiguous() else pag.contiguous()
    stats = _udag2pag_on_tensor(
        t,
        sepset,
        unfVect,
        rules,
        orientCollider=orientCollider,
        ext=_udag2pag_cuda(),
        pdsep=_pdsep_cuda(),
        r1r2_host=r1r2_host,
        whydra_marks=whydra_marks,
    )
    if t.data_ptr() != pag.data_ptr():
        pag.copy_(t)
    return stats


def _udag2pag_on_tensor(
    t: torch.Tensor,
    sepset,
    unfVect,
    rules,
    *,
    orientCollider: bool,
    ext,
    pdsep,
    r1r2_host=None,
    whydra_marks: bool = False,
) -> dict:
    import time

    from .cuda_pdsep import _unf_tensor

    dev = t.device
    p = int(t.size(0))
    t0 = time.perf_counter()
    t.masked_fill_(t != 0, 1)
    torch.cuda.synchronize(dev)
    t_init = time.perf_counter() - t0

    S, ranges = sepset.packed()
    if torch.is_tensor(S):
        S_t = S.to(device=dev, dtype=torch.int64).reshape(-1).contiguous()
    else:
        S_t = torch.as_tensor(
            np.ascontiguousarray(np.asarray(S), dtype=np.int64), device=dev
        )
    if torch.is_tensor(ranges):
        ranges_t = ranges.to(device=dev, dtype=torch.int64).contiguous()
    else:
        ranges_t = torch.as_tensor(
            np.ascontiguousarray(ranges, dtype=np.int64), device=dev
        )
    if int(S_t.numel()) == 0:
        S_t = torch.zeros(1, dtype=torch.int64, device=dev)
    unf_t = _unf_tensor(unfVect or (), p, dev)

    t_col = 0.0
    if orientCollider:
        t0 = time.perf_counter()
        pdsep.mark_colliders(t, S_t, ranges_t, unf_t)
        torch.cuda.synchronize(dev)
        t_col = time.perf_counter() - t0

    rules_list = [bool(r) for r in list(rules)[:10]]
    while len(rules_list) < 10:
        rules_list.append(True)
    rules_t = torch.tensor(
        [1 if r else 0 for r in rules_list], dtype=torch.uint8, device=dev
    )
    if r1r2_host is not None:
        # Host applies R1/R2 (rulesR1R2cycle, whydra marks); the device only
        # runs R3..R10 and the tensor is converted around each orient call.
        rules_t[0] = 0
        rules_t[1] = 0
        _pcalg_to_whydra_(t)
        torch.cuda.synchronize(dev)
    tail_rules = any(rules_list[2:])

    old = torch.empty_like(t)
    n_iters = 0
    acc_rt = [0.0] * 10
    while True:
        n_iters += 1
        old.copy_(t)
        if r1r2_host is not None:
            t0 = time.perf_counter()
            host = t.detach().cpu().numpy().astype(int)
            r1r2_host(host)
            t.copy_(torch.as_tensor(host, dtype=t.dtype, device=t.device))
            torch.cuda.synchronize(dev)
            acc_rt[0] += time.perf_counter() - t0
        if tail_rules or r1r2_host is None:
            if r1r2_host is not None:
                _whydra_to_pcalg_(t)
            rule_times = ext.orient(t, unf_t, S_t, ranges_t, rules_t)
            rt = rule_times.flatten().tolist()
            for i in range(min(10, len(rt))):
                acc_rt[i] += float(rt[i])
            if r1r2_host is not None:
                _pcalg_to_whydra_(t)
                torch.cuda.synchronize(dev)
        if torch.equal(old, t):
            break
        if n_iters >= 10_000:
            break

    if r1r2_host is None and whydra_marks:
        _pcalg_to_whydra_(t)
        torch.cuda.synchronize(dev)

    out = {
        "init": t_init,
        "colliders": t_col,
        "copy": 0.0,
        "n_iters": float(n_iters),
    }
    for i in range(1, 11):
        out[f"R{i}"] = acc_rt[i - 1]
    return out


__all__ = [
    "apply_r1",
    "apply_r2",
    "apply_r3",
    "apply_r4",
    "apply_r5",
    "apply_r6",
    "apply_r7",
    "apply_r8",
    "apply_r9",
    "apply_r10",
    "apply_udag2pag",
    "apply_udag2pag_tensor",
    "pack_r1",
    "pack_r2",
    "pack_r3",
    "pack_r5",
    "pack_r8",
]
