"""Certification of a provisional RAI skeleton.

The legacy recursion can leave false edges and orient true parents backwards
even with perfect CI answers. Finalization therefore discards provisional
marks, refines the remaining skeleton with stable adjacency search, and builds
the CPDAG again. This is an explicit algorithmic safeguard, not an equivalent
optimization of the published RAI procedure. It never adds an adjacency.
"""

import numpy as np

from .skeleton import SkeletonDiscovery
from .standalone_pc_stable import apply_meek_rules, orient_colliders


def certify_rai_skeleton(learner, n_jobs=1):
    """Refine the provisional RAI skeleton and orient the remaining edges as a CPDAG."""
    cg = learner.cg
    before = int(np.count_nonzero(cg.G.graph) // 2)
    cg.G.graph[cg.G.graph != 0] = -1
    cg.set_ind_test(learner.ci_test)
    SkeletonDiscovery(cg, alpha=learner.alpha, n_jobs=n_jobs, show_progress=False).run()
    # SkeletonDiscovery preserves the PC membership-summary policy. RAI's
    # established separators for earlier deletions remain in their own cells.
    orient_colliders(cg)
    apply_meek_rules(cg)
    cg.G.graph_kind = "cpdag"
    cg.G.metadata["rai_certification"] = {
        "policy": "stable_adjacency_refinement",
        "edges_removed": before - int(np.count_nonzero(cg.G.graph) // 2),
        "changes_statistical_behavior": True,
    }
    return cg.G
