import itertools
from itertools import combinations

import numpy as np

from ..graph_core import CIT
from ..graph_core import conditioning as _conditioning_exports
from ..graph_core.conditioning import (
    bounded_size,
    validate_max_cond_set_size,
)

# Import the shared test class from the neighboring module.
from .ci_adapters import LegacyCITest
from .fci_search import refine_rows
from .pag_rules import qreach, udag2pag

bounded_powerset = _conditioning_exports.bounded_powerset

# --- Helper functions ---


def skeleton(
    suffStat,
    indepTest,
    alpha,
    labels,
    method,
    fixedGaps,
    fixedEdges,
    NAdelete,
    m_max,
    numCores,
    verbose,
):
    """PC-stable neighborhoods at each conditioning depth (Colombo & Maathuis, 2014)."""
    size = len(labels)
    nodes = range(size)
    edges = np.ones((size, size), dtype=int)
    np.fill_diagonal(edges, 0)
    peak = np.full((size, size), -np.inf)
    np.fill_diagonal(peak, 1)
    evidence: dict[tuple[int, int], set | list] = {
        pair: set() for pair in itertools.permutations(nodes, 2)
    }
    limit = bounded_size(indepTest, m_max)
    depth = 0
    while edges.any() and depth <= limit:
        neighborhoods = [np.flatnonzero(row) for row in edges]
        deeper = False
        for x, adjacent in enumerate(neighborhoods):
            if len(adjacent) <= depth:
                continue
            for y in map(int, adjacent):
                if not edges[x, y]:
                    continue
                deeper |= len(adjacent) > depth + 1
                pool = adjacent[adjacent != y]
                # This entry point exposes the historical hash-set ordering of ties.
                for witness in set(combinations(pool, depth)):
                    condition = list(witness)
                    value = indepTest(suffStat, x, y, condition)
                    peak[x, y] = max(peak[x, y], value)
                    if value >= alpha:
                        edges[x, y] = edges[y, x] = 0
                        evidence[x, y], evidence[y, x] = condition, condition.copy()
                        break
        depth += 1
        if not deeper:
            break
    return {
        "sk": edges,
        "pMax": np.maximum(peak, peak.T),
        "sepset": evidence,
        "unfTriples": set(),
        "max_ord": depth - 1,
    }


def pdsep(
    skel,
    suffStat,
    indepTest,
    p,
    sepSet,
    alpha,
    pMax,
    m_max=float("inf"),
    pdsep_max=float("inf"),
    unfVect=None,
):
    """Refine the skeleton with Possible-D-SEP tests and return the separation evidence."""
    return refine_rows(
        skel["sk"],
        indepTest,
        suffStat,
        sepSet,
        alpha,
        pMax,
        m_max,
        unfVect,
        qreach,
        plain=True,
    )


def fci_stable(
    data, alpha=0.05, indep_test="fisherz", verbose=False, max_cond_set_size=None
):
    """Run stable FCI and return the PAG endpoint matrix."""
    max_cond_set_size = validate_max_cond_set_size(max_cond_set_size)
    _n, p = data.shape

    # Initialize the shared CIT implementation.
    cit = CIT(data, method=indep_test, max_cond_set_size=max_cond_set_size)

    # Adapt the test to the (suffStat, x, y, S) signature that FCI expects.
    test_func_wrapper = LegacyCITest(cit, max_cond_set_size)

    # CIT holds the data. The FCI test adapter ignores suffStat.
    suffStat = None

    graphDict = skeleton(
        suffStat,
        test_func_wrapper,
        alpha,
        labels=[str(i) for i in range(p)],
        method="stable",
        fixedGaps=None,
        fixedEdges=None,
        NAdelete=True,
        m_max=float("inf"),
        numCores=1,
        verbose=verbose,
    )

    if verbose:
        print(f"graphDict skeleton {graphDict}")

    if verbose:
        print(f"graphDict before pdsep {graphDict}")
        print(f"graphDict before suffStat {suffStat}")
        print(f"graphDict before alpha {alpha}")
        print(f"graphDict before test_func_wrapper {test_func_wrapper}")

    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict["sepset"],
        alpha,
        graphDict["pMax"],
        unfVect=graphDict["unfTriples"],
    )
    if verbose:
        print(f"graphDict after pdsep {graphDict}")
        print(f"graphDict after suffStat {suffStat}")
        print(f"graphDict after alpha {alpha}")
        print(f"graphDict after test_func_wrapper {test_func_wrapper}")

    pag = udag2pag(
        pdsep_res["G"], pdsep_res["sepset"], p, unfVect=graphDict["unfTriples"]
    )

    return pag
