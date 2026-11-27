"""Shared conditioning-budget contract for CI searches and orientation checks."""

from itertools import combinations
from numbers import Integral


def validate_max_cond_set_size(value):
    """Return a nonnegative integer limit, or None for an unlimited search."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        msg = "max_cond_set_size must be None or a nonnegative integer"
        raise ValueError(msg)
    return int(value)


def configure_conditioning(cg, max_cond_set_size):
    """Apply the conditioning limit to the graph, its metadata and compatible tests."""
    limit = validate_max_cond_set_size(max_cond_set_size)
    cg.max_cond_set_size = limit
    if hasattr(cg.test, "max_cond_set_size"):
        cg.test.max_cond_set_size = limit
    cg.G.metadata.update(
        max_cond_set_size=limit,
        effective_max_cond_set_size=limit,
        depth_limited=limit is not None,
    )
    return limit


def validate_cuda_background_knowledge(background_knowledge, algorithm: str) -> None:
    """Reject background knowledge for a CUDA algorithm that cannot use it."""
    if background_knowledge is not None:
        msg = (
            f"{algorithm}: CUDA does not support background_knowledge; "
            "use a CPU algorithm that supports the supplied constraints"
        )
        raise NotImplementedError(msg)


def bounded_size(owner, size):
    """Largest permitted cardinality, retaining an existing phase-specific cap."""
    limit = getattr(owner, "max_cond_set_size", None)
    return size if limit is None else min(size, limit)


def conditioning_allowed(owner, conditions):
    """Check whether the conditioning set fits the owner's size limit."""
    limit = getattr(owner, "max_cond_set_size", None)
    return limit is None or len(conditions) <= limit


def bounded_powerset(owner, nodes):
    """Yield node combinations in increasing size up to the conditioning limit."""
    nodes = tuple(nodes)
    for size in range(bounded_size(owner, len(nodes)) + 1):
        yield from combinations(nodes, size)
