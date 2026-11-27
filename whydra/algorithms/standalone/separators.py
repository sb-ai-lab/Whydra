"""Compatibility boundary for historical CausalGraph separator tables."""

import numpy as np


def sepset_to_pair_dict(sepset, p):
    """Return one established separator per pair; omit unknown entries.

    None is unknown, []/[()] establishes marginal independence. A legacy flat
    sequence is one separator, never a sequence of singleton separators.
    Multiple recorded separators are alternatives, not a union separator.
    """
    out = {}
    for i in range(p):
        for j in range(p):
            entry = sepset[i][j]
            if entry is None:
                continue
            if len(entry) == 0 or all(np.isscalar(v) for v in entry):
                candidates = [tuple(sorted(map(int, entry)))]
            else:
                candidates = [tuple(sorted(map(int, s))) for s in entry]
            out[(i, j)] = set(min(candidates, key=lambda s: (len(s), s)))
    return out
