"""Sample a DAG with node roles (G-1 through G-4).

Assign covariates roles such as confounder, effect modifier, predictor, noise,
instrument or collider. Sample the edges between covariates (X–X) at the requested
density subject to those roles. Return the adjacency matrix over [X..., T, M..., Y] and
the role labels (G-3).

Since 0.5.0, the graph tracks the equations used to generate data. The
direct T -> Y edge is present by default and has been optional since 0.10.0. In a
randomized trial there are no X -> T edges: in `roles_effective`, confounders become
predictors and instruments become noise. A collider has no incoming X edges because only
T and Y enter its equation. Instruments, like noise nodes, are outside the X–X graph,
ensuring a valid instrument in observational mode by construction. After building the
segments, the generator calls `add_edges_to_y` to connect the features defining
segment-specific tau to Y.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .config import ROLE_COUNTERS, GraphConfig, check_roles

ROLE_CONFOUNDER = "confounder"
ROLE_MODIFIER = "modifier"
ROLE_PREDICTOR = "predictor"
ROLE_NOISE = "noise"
ROLE_INSTRUMENT = "instrument"
ROLE_COLLIDER = "collider"
# 0.10.0 (G-2): Roles involving mediators and nodes with no direct edges to T, M or Y.
ROLE_MED_CONFOUNDER = "med_confounder"  # X -> every M, and X -> Y.
ROLE_MED_PREDICTOR = "med_predictor"  # X -> every M.
ROLE_UPSTREAM = (
    "upstream"  # X–X edges only; outgoing edges may target any role that accepts them.
)
ROLE_SIDE = "side"  # X–X edges only, with outgoing edges only to other side nodes; never an ancestor of T, M or Y.
# All roles in block order. See config.ROLE_COUNTERS for the table and G-6 for the role contract.
ROLES = tuple(ROLE_COUNTERS)
# Topological bands in shuffled order: upstream nodes first, side nodes last. X–X edges point
# forward.
_TOPO_BAND = {ROLE_UPSTREAM: 0, ROLE_SIDE: 2}


@dataclass
class SampledGraph:
    """Store the DAG on [X_0..X_{d-1}, T, M_0..M_{m-1}, Y] and the covariate roles.

    Inside an `SCM`, `scm.freeze_graph` freezes the graph (0.7.1). The `_frozen` flag
    lives in `__dict__` outside the dataclass fields. Setting or deleting attributes and
    calling `add_edges_to_y` raise AttributeError. The matrices are read-only and
    `discrete` uses read-only MappingProxyType mappings and tuples. `Dataset.graph` is a
    mutable copy made by `scm.thaw_graph`.
    """

    roles: list[str]  # Role of each covariate X_i, in a list of length d.
    adjacency: np.ndarray  # (d+1+m+1) by (d+1+m+1); adjacency[i, j] = 1 means i -> j.
    node_names: list[str]  # Node names in matrix index order.
    xx_edges: np.ndarray  # Convenient d by d slice containing the X–X edges.
    structure: str = "er"  # X–X graph structure type (G-1a).
    blocks: list | None = None  # blocks: block index of each covariate.
    layers: list | None = None  # layered: layer index of each covariate.
    prototype: list | None = (
        None  # copying: index of the node each clone copies; -1 for all other nodes.
    )
    # Discrete nodes (since 0.4.0), keyed by node index: kind is binary or categorical; levels is
    # k; shares gives target proportions; level_effects gives standardized values for child nodes
    # and other mechanisms; thresholds gives inverse-normal (Φ⁻¹) cutoffs for the standardized
    # latent variable.
    discrete: dict | None = None
    # Roles used by the equations (0.5.0): in randomized trials, confounder becomes predictor and
    # instrument becomes noise. None means the roles are unchanged.
    roles_effective: list | None = None
    xx_density_requested: float = (
        0.0  # Requested X–X density for the density_* fields in graph_properties.
    )

    def __post_init__(self):
        if self.roles_effective is None:
            self.roles_effective = list(self.roles)

    # --- Freeze the entire graph inside an SCM (0.7.1) ---

    @property
    def frozen(self) -> bool:
        """Return True for a graph frozen by scm.freeze_graph, including its attributes and nested structures."""
        return self.__dict__.get("_frozen", False)

    def __setattr__(self, name, value):
        if self.frozen:
            msg = (
                f"SampledGraph is frozen (SCM.graph): attribute {name!r} is read-only; "
                f"use Dataset.graph or scmgen.scm.thaw_graph(g) for a mutable copy"
            )
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if self.frozen:
            msg = f"SampledGraph is frozen (SCM.graph): attribute {name!r} is read-only"
            raise AttributeError(msg)
        object.__delattr__(self, name)

    @property
    def d(self) -> int:
        """Return the number of covariate nodes."""
        return len(self.roles)

    def add_edges_to_y(self, nodes) -> None:
        """Add X_i -> Y edges for the features that the outcome equation reads through segment-specific tau."""
        if self.frozen:
            msg = "SampledGraph is frozen (SCM.graph): edges are read-only"
            raise AttributeError(msg)
        iy = self.idx_y()
        for i in nodes:
            self.adjacency[int(i), iy] = 1

    def parents(self, node: int) -> list[int]:
        """Return the indices of nodes with an edge into the given node."""
        return [int(i) for i in np.flatnonzero(self.adjacency[:, node])]

    def discrete_nodes(self, kind: str | None = None) -> list[int]:
        """Return discrete node indices in ascending order, optionally filtered by kind."""
        if not self.discrete:
            return []
        return sorted(
            i
            for i, spec in self.discrete.items()
            if kind is None or spec["kind"] == kind
        )

    def child_view(self, x: np.ndarray, idx) -> np.ndarray:
        """Return covariate values for a child node's equation.

        For the nodes in idx, replace each discrete code with the effect for that level
        and leave continuous values unchanged. With no discrete nodes, the result
        matches x[:, idx] bit for bit.
        """
        idx = np.asarray(idx, dtype=int)
        return self.level_view(
            x[:, idx], idx, inplace=True
        )  # x[:, idx] is a fresh copy.

    def level_view(self, x: np.ndarray, nodes, inplace: bool = False) -> np.ndarray:
        """Convert selected columns as for a child node's equation.

        x has shape (n, len(nodes)); nodes gives the node index of each column. NaNs
        stay NaN. If none of the selected nodes is discrete, return x unchanged.
        """
        disc = self.discrete or {}
        hit = [(k, disc[int(i)]) for k, i in enumerate(nodes) if int(i) in disc]
        if not hit:
            return x
        out = x if inplace and x.dtype == float else x.astype(float, copy=True)
        for k, spec in hit:
            col = out[:, k]
            ok = ~np.isnan(col)
            col[ok] = np.asarray(spec["level_effects"], dtype=float)[
                col[ok].astype(int)
            ]
        return out

    def idx_t(self) -> int:
        """Return the index of the treatment node."""
        return self.d

    def idx_y(self) -> int:
        """Return the index of the outcome node."""
        return self.adjacency.shape[0] - 1

    @property
    def n_mediators(self) -> int:
        """Return the number of mediator nodes in the graph."""
        return self.adjacency.shape[0] - self.d - 2

    def idx_m(self) -> list[int]:
        """Return the mediator node indices in graph order."""
        return list(range(self.d + 1, self.d + 1 + self.n_mediators))

    def ancestors(self, node: int) -> set[int]:
        """Return the node's ancestors according to adjacency, excluding the node itself."""
        out: set[int] = set()
        stack = [int(node)]
        while stack:
            for p in self.parents(stack.pop()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def covariate_indices(self, role: str) -> np.ndarray:
        """Return the indices of covariates with the given role."""
        return np.array([i for i, r in enumerate(self.roles) if r == role], dtype=int)


def sample_graph(
    cfg: GraphConfig, rng: np.random.Generator, rct: bool = False
) -> SampledGraph:
    """Build a DAG from the configuration, adding required role edges before sampling X–X edges at the requested density.

    Covariates follow index order, then treatment T, mediators M and outcome Y. Collider
    covariates are observed traps: they belong to X but follow their parents T and Y in
    topological order. With rct=True, e(x) is constant and X -> T edges are absent. The
    declared roles remain in `roles`; `roles_effective` records those used by the
    equations. The sequence of RNG calls is independent of rct.

    By default, `role_order="shuffled"` assigns roles to random positions within
    topological bands. X–X edges point only to later indices. With
    `role_order="blocks"`, confounders come first, so predictor -> confounder
    edges are impossible. A manual adjacency matrix (G-4) uses the specified
    role positions without shuffling.
    """
    if cfg.role_order not in ("shuffled", "blocks"):
        msg = f"role_order must be shuffled or blocks, got {cfg.role_order!r}"
        raise ValueError(msg)

    roles = [
        role
        for role, attr in ROLE_COUNTERS.items()
        for _ in range(int(getattr(cfg, attr)))
    ]
    d = len(roles)
    if d == 0:
        msg = "At least one covariate is required"
        raise ValueError(msg)
    if cfg.roles is not None:  # Manual layout (0.10.0, G-4).
        check_roles(cfg)
        roles = [str(r) for r in cfg.roles]
    elif cfg.role_order == "shuffled" and cfg.adjacency is None:
        roles = [roles[i] for i in rng.permutation(d)]
        roles.sort(
            key=lambda r: _TOPO_BAND.get(r, 1)
        )  # Stable sort; without upstream or side nodes it preserves the shuffled order.
    discrete = _sample_discrete(
        cfg, roles, rng
    )  # Since 0.4.0; zero discrete-node counts consume no random draws.
    m = cfg.n_mediators
    n_nodes = d + 1 + m + 1  # X..., T, M..., Y
    idx_t = d
    idx_m = list(range(d + 1, d + 1 + m))
    idx_y = n_nodes - 1

    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    # Add role-specific edges. In an RCT, assignment ignores X, so there are no X -> T edges.
    roles_eff = list(roles)
    for i, role in enumerate(roles):
        if role == ROLE_CONFOUNDER:
            adj[i, idx_y] = 1
            if rct:
                roles_eff[i] = ROLE_PREDICTOR
            else:
                adj[i, idx_t] = 1
        elif role == ROLE_MODIFIER:
            adj[i, idx_y] = 1  # The modifier affects Y through tau(x) times T.
        elif role == ROLE_PREDICTOR:
            adj[i, idx_y] = 1
        elif role == ROLE_INSTRUMENT:
            if rct:
                roles_eff[i] = ROLE_NOISE
            else:
                adj[i, idx_t] = 1
        elif role == ROLE_COLLIDER:
            adj[idx_t, i] = 1
            adj[idx_y, i] = 1
        elif role == ROLE_MED_CONFOUNDER:  # 0.10.0: X -> every M_k, and X -> Y.
            adj[i, idx_y] = 1
            for j in idx_m:
                adj[i, j] = 1
        elif role == ROLE_MED_PREDICTOR:  # 0.10.0: X -> every M_k.
            for j in idx_m:
                adj[i, j] = 1
        # ROLE_NOISE, ROLE_UPSTREAM and ROLE_SIDE add no role-specific edges.
    if cfg.direct_effect:
        adj[idx_t, idx_y] = (
            1  # The equation includes T -> Y even when tau = 0, unless direct_effect is disabled (optional since 0.10.0).
        )

    # Mediators: T -> M -> Y.
    for j in idx_m:
        adj[idx_t, j] = 1
        adj[j, idx_y] = 1

    # Use a manual X–X graph (G-4) or sample one (G-1, type from G-1a).
    if cfg.adjacency is not None:
        structure = "manual"
        blocks = layers = prototype = None
        xx = np.asarray(cfg.adjacency, dtype=np.int8).copy()
        if xx.shape != (d, d):
            msg = f"Manual adjacency must have shape {d}x{d}, got {xx.shape}"
            raise ValueError(msg)
        if np.any(np.tril(xx)):
            msg = "Manual adjacency must be strictly upper triangular (a DAG in node index order)"
            raise ValueError(msg)
        _isolate_inert_roles(xx, roles)
    else:
        structures = _structure_list(cfg)
        structure = "+".join(structures)
        upper = np.triu(np.ones((d, d), dtype=np.int8), k=1)
        prototype = None
        if structures == ["copying"]:
            xx, prototype = _copying_edges(cfg, d, rng, roles)
            blocks = layers = None
        else:
            prob, blocks, layers = _edge_probabilities(cfg, d, rng, roles)
            mask = rng.random((d, d)) < prob
            xx = (upper & mask).astype(np.int8)
        _isolate_inert_roles(xx, roles)
        _cap_in_degree(xx, cfg.max_in_degree, rng)
    adj[:d, :d] = xx

    names = [f"x{i}" for i in range(d)] + ["t"] + [f"m{k}" for k in range(m)] + ["y"]
    return SampledGraph(
        roles=roles,
        adjacency=adj,
        node_names=names,
        xx_edges=xx,
        structure=structure,
        blocks=blocks,
        layers=layers,
        prototype=prototype,
        discrete=discrete,
        roles_effective=roles_eff,
        xx_density_requested=float(cfg.xx_density),
    )


def _isolate_inert_roles(xx: np.ndarray, roles: list[str]) -> None:
    """Keep the covariate edges allowed by the node roles.

    X–X edges must match the equations. Noise and instrument nodes have no X–X edges in
    either direction, ensuring the instrument is valid by construction. A collider's
    equation reads only T and Y, and it generates no X, so it has neither incoming nor
    outgoing X–X edges. Since 0.10.0, a side node may point only to another side node
    and cannot be an ancestor of T, M or Y. Apply these rules to manual X–X matrices
    (G-4) as well.
    """
    not_side = [j for j, r in enumerate(roles) if r != ROLE_SIDE]
    for i, role in enumerate(roles):
        if role in (ROLE_NOISE, ROLE_INSTRUMENT, ROLE_COLLIDER):
            xx[i, :] = 0
            xx[:, i] = 0
        elif role == ROLE_SIDE:
            xx[i, not_side] = 0


# --- Discrete nodes (0.4.0) ---

_NORM = NormalDist()


def _level_shares(k: int, skew: float) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1, dtype=float) ** float(skew)
    return w / w.sum()


def _standardized_effects(values: np.ndarray, shares: np.ndarray) -> list[float]:
    """Standardize level effects to mean zero and variance one under the target proportions, independently of the sample."""
    mean = float(np.einsum("k,k->", shares, values))  # Avoid BLAS (0.7.0).
    var = float(np.einsum("k,k->", shares, values**2)) - mean**2
    sd = np.sqrt(var) if var > 1e-12 else 1.0
    return [float(v) for v in (values - mean) / sd]


def _discrete_spec(kind: str, shares: np.ndarray, values: np.ndarray) -> dict:
    cum = np.cumsum(shares)[:-1]
    return {
        "kind": kind,
        "levels": int(shares.size),
        "shares": [float(v) for v in shares],
        "level_effects": _standardized_effects(values, shares),
        "thresholds": [float(_NORM.inv_cdf(float(c))) for c in cum],
    }


def _sample_discrete(
    cfg: GraphConfig, roles: list[str], rng: np.random.Generator
) -> dict | None:
    """Sample discrete node positions and parameters together.

    Draw all positions with one rng.choice, assigning binary nodes first and categorical
    nodes next. Then draw the binary probabilities p if binary_rate is a range. Finally
    draw the categorical level effects in ascending node index order. A binary node has
    levels {0, 1} with proportions {1-p, p} and uses the standardized indicator as its
    effect.
    """
    nb, nc = int(cfg.n_binary), int(cfg.n_categorical)
    if nb < 0 or nc < 0:
        msg = "n_binary and n_categorical must be >= 0"
        raise ValueError(msg)
    if nb + nc == 0:
        return None
    allowed = set(cfg.discrete_roles) if cfg.discrete_roles is not None else None
    if allowed is not None:
        bad = allowed - (set(ROLES) - {ROLE_COLLIDER})
        if bad:
            msg = f"discrete_roles: invalid roles {sorted(bad)}"
            raise ValueError(msg)
    cand = [
        i
        for i, r in enumerate(roles)
        if r != ROLE_COLLIDER and (allowed is None or r in allowed)
    ]
    if len(cand) < nb + nc:
        msg = (
            f"Requested {nb + nc} discrete nodes, but there are {len(cand)} candidates with roles "
            f"{sorted(allowed) if allowed is not None else 'other than collider'}"
        )
        raise ValueError(msg)
    levels = cfg.categorical_levels
    levels = (
        [int(levels)] * nc
        if isinstance(levels, (int, float))
        else [int(v) for v in levels]
    )
    if len(levels) != nc:
        msg = f"categorical_levels: expected {nc} values, got {len(levels)}"
        raise ValueError(msg)
    if any(k < 2 for k in levels):
        msg = "categorical_levels: a categorical node must have at least two levels"
        raise ValueError(msg)

    pos = rng.choice(cand, size=nb + nc, replace=False)
    binary = sorted(int(i) for i in pos[:nb])
    categorical = sorted(int(i) for i in pos[nb:])
    rate = cfg.binary_rate
    if isinstance(rate, (list, tuple)):
        rates = rng.uniform(
            float(rate[0]), float(rate[1]), size=nb
        )  # size=0 does not advance the RNG.
    else:
        rates = np.full(nb, float(rate))
    if np.any((rates <= 0) | (rates >= 1)):
        msg = "binary_rate must be strictly between 0 and 1"
        raise ValueError(msg)

    out: dict[int, dict] = {}
    for i, p in zip(binary, rates, strict=False):
        shares = np.array([1.0 - p, p])
        out[i] = _discrete_spec("binary", shares, np.array([0.0, 1.0]))
        out[i]["rate"] = float(p)
    for i, k in zip(categorical, levels, strict=False):
        shares = _level_shares(k, cfg.categorical_skew)
        out[i] = _discrete_spec("categorical", shares, rng.normal(size=k))
    return dict(sorted(out.items()))


def column_node(col) -> int:
    """Extract an X node index from a column name: 'x7' and the one-hot column 'x7=2' both give 7; unrecognized names give -1."""
    name = str(col).split("=")[0]
    if name.startswith("x") and name[1:].isdigit():
        return int(name[1:])
    return -1


def standardize(v: np.ndarray) -> np.ndarray:
    """Compute (v - mean) / sd, or only center v when sd is near zero, for use by mechanisms and pathologies."""
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else v - v.mean()


def level_codes(z: np.ndarray, spec: dict) -> np.ndarray:
    """Return the level code from 0 to k-1 as a float by counting the inverse-normal (Φ⁻¹) thresholds at or below z."""
    return np.searchsorted(
        np.asarray(spec["thresholds"], dtype=float), z, side="right"
    ).astype(float)


def discretize(v: np.ndarray, spec: dict) -> np.ndarray:
    """Convert a continuous latent variable to discrete levels.

    Hybrid discretization standardizes the variable using sample statistics and
    applies inverse-normal (Φ⁻¹) cutoffs outside the SCM. Within the SCM,
    mechanisms.apply_covariates takes the standardization constants from calibration.
    """
    return level_codes(standardize(v), spec)


_STRUCTURES = ("er", "scale_free", "blocks", "layered", "copying")
_COMPOSE_ORDER = (
    "er",
    "blocks",
    "scale_free",
    "layered",
)  # Order of factors and RNG calls.


def _structure_list(cfg: GraphConfig) -> list[str]:
    """Normalize the structure list.

    Turn a string into a one-item list, or arrange a list in _COMPOSE_ORDER.
    Combinations of two or more items support only matrix-based families. Use copying on
    its own.
    """
    st = cfg.xx_structure
    items = [st] if isinstance(st, str) else list(st)
    if not items:
        msg = "xx_structure cannot be an empty list"
        raise ValueError(msg)
    for it in items:
        if it not in _STRUCTURES:
            msg = f"xx_structure must be er, scale_free, blocks, layered or copying, got {it!r}"
            raise ValueError(msg)
    if len(set(items)) != len(items):
        msg = f"xx_structure contains a repeated structure: {items}"
        raise ValueError(msg)
    if len(items) > 1:
        if "copying" in items:
            msg = "copying cannot be combined with other structures because it is not a matrix model"
            raise ValueError(msg)
        items = [k for k in _COMPOSE_ORDER if k in items]
    return items


def _cap_in_degree(
    xx: np.ndarray, max_in_degree: int | None, rng: np.random.Generator
) -> None:
    """Cap the number of parents of each node.

    A.4: For each column with too many parents, randomly retain max_in_degree of them.
    This uses one RNG call per affected node and none if the limit is None.
    """
    if max_in_degree is None:
        return
    k = int(max_in_degree)
    if k < 1:
        msg = f"max_in_degree must be >= 1 or None, got {max_in_degree!r}"
        raise ValueError(msg)
    for j in range(xx.shape[1]):
        parents = np.flatnonzero(xx[:, j])
        if parents.size > k:
            keep = rng.choice(parents, size=k, replace=False)
            xx[:, j] = 0
            xx[keep, j] = 1


def _copying_edges(
    cfg: GraphConfig, d: int, rng: np.random.Generator, roles: list[str]
):
    """Sample the copying model for cloned features.

    Process nodes in index order. With probability copy_share, node j becomes a clone:
    it chooses a prototype among earlier active nodes with parents, then inherits each
    of its parents with probability copy_fidelity. No edge joins the clone and
    prototype; they represent two aggregates of the same raw events. Otherwise, sample
    ordinary ER edges from earlier nodes with density xx_density. This construction is
    acyclic.
    """
    xx = np.zeros((d, d), dtype=np.int8)
    prototype = np.full(d, -1, dtype=int)
    active = _active_mask(roles)
    for j in range(1, d):
        candidates = [i for i in range(j) if active[i] and xx[:, i].any()]
        if candidates and rng.random() < cfg.copy_share:
            p_idx = int(rng.choice(candidates))
            parents = np.flatnonzero(xx[:, p_idx])
            keep = rng.random(parents.size) < cfg.copy_fidelity
            xx[parents[keep], j] = 1
            prototype[j] = p_idx
        else:
            xx[:j, j] = (rng.random(j) < cfg.xx_density).astype(np.int8)
    return xx, prototype.tolist()


def _edge_probabilities(
    cfg: GraphConfig, d: int, rng: np.random.Generator, roles: list[str]
):
    """Build the X–X edge probability matrix (generalized binomial random graph).

    er uses the scalar xx_density. The deterministic probability matrix consumes no
    random draws, preserving the existing RNG order.

    scale_free uses the Chung–Lu model: p_ij is proportional to w_i times w_j. Expected
    degrees w have a power-law tail with exponent sf_gamma. A random permutation assigns
    hub weights among active nodes; noise nodes and colliders receive no weights.
    Calibrate the mean density to xx_density. Since 0.3.0, the model operates in the
    admissible regime: iteratively cap the tail weights until w_max <= mean_w /
    sqrt(xx_density). Then no probability exceeds one before clipping. Otherwise,
    clipping the highest hub-to-hub probabilities at large d would give a density below
    target and negative assortativity instead of roughly zero.

    blocks uses a stochastic block model: p_in within a block, p_out between blocks.
    Assign nodes to blocks at random.

    layered assigns layers by node index. Adjacent layers connect with density
    xx_density; edges that skip a layer use xx_density * layer_skip. There are no
    within-layer edges.

    For a list of two or more families, use _compose_probabilities (since 0.3.0). The
    caller orients edges in ascending index order to produce a DAG for every family.
    """
    structures = _structure_list(cfg)
    if len(structures) > 1:
        return _compose_probabilities(cfg, structures, d, rng, roles)
    st = structures[0]
    if st == "er":
        return cfg.xx_density, None, None
    if st == "scale_free":
        active = _active_mask(roles)
        w = _sf_weights(cfg, active, rng)
        n_act = int(active.sum())
        if n_act == 0 or cfg.xx_density <= 0:
            return np.zeros((d, d)), None, None
        w = _sf_cap(w, active, cfg.xx_density)
        mean_w = w[active].mean()
        prob = cfg.xx_density * np.outer(w, w) / (mean_w**2)
        return np.clip(prob, 0.0, 1.0), None, None  # Clip as a safeguard.
    if st == "blocks":
        prob, blocks = _blocks_multiplier(cfg, d, rng)
        return prob, blocks, None
    if st == "layered":
        mask, layers = _layered_multiplier(cfg, d)
        return cfg.xx_density * mask, None, layers
    msg = f"_structure_list accepted {st!r}, but no implementation handled it"
    raise AssertionError(msg)


def _active_mask(roles: list[str]) -> np.ndarray:
    """Return the nodes in the X–X structure, excluding noise nodes, instruments and colliders."""
    return np.array(
        [r not in (ROLE_NOISE, ROLE_COLLIDER, ROLE_INSTRUMENT) for r in roles],
        dtype=bool,
    )


def _sf_weights(
    cfg: GraphConfig, active: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomly assign Zipf-like expected degrees to active nodes with one RNG call."""
    if cfg.sf_gamma <= 2.0:
        msg = "sf_gamma must be > 2 for a finite mean degree"
        raise ValueError(msg)
    w = np.zeros(active.size)
    n_act = int(active.sum())
    rank = rng.permutation(n_act) + 1  # Choose the hubs at random among active nodes.
    w[active] = rank ** (-1.0 / (cfg.sf_gamma - 1.0))
    return w


def _sf_cap(w: np.ndarray, active: np.ndarray, density: float) -> np.ndarray:
    """A.2: Cap Chung–Lu weights so the largest edge probability is at most one before clipping."""
    w = w.copy()
    for _ in range(100):
        mean_w = w[active].mean()
        cap = mean_w / np.sqrt(density)
        if w.max() <= cap * (1 + 1e-9):
            break
        w = np.minimum(w, cap * 0.999)
    return w


def _blocks_multiplier(cfg: GraphConfig, d: int, rng: np.random.Generator):
    k = max(1, int(cfg.blocks_k))
    blocks = rng.permutation(np.arange(d) % k)
    same = blocks[:, None] == blocks[None, :]
    return np.where(same, cfg.p_in, cfg.p_out), blocks.tolist()


def _layered_multiplier(cfg: GraphConfig, d: int):
    L = max(1, int(cfg.layers_l))
    layers = (np.arange(d) * L) // d  # Assign layers in index order.
    gap = layers[None, :] - layers[:, None]  # Layer of j minus layer of i.
    mask = np.where(gap == 1, 1.0, np.where(gap > 1, cfg.layer_skip, 0.0))
    return mask, layers.tolist()


def _compose_probabilities(
    cfg: GraphConfig,
    structures: list[str],
    d: int,
    rng: np.random.Generator,
    roles: list[str],
):
    """Combine the blocks, scale_free and layered factors; er is neutral (A.3).

    Normalize their product so that the mean probability over active pairs in the upper
    triangle equals xx_density. In a combination, p_in/p_out and layer_skip determine
    only the shape. Repeatedly clip and renormalize until the maximum is at most one and
    the mean matches the target. Draw for blocks first, then scale_free. Layered
    consumes no random draws.
    """
    active = _active_mask(roles)
    prob = np.ones((d, d))
    blocks = layers = None
    if "blocks" in structures:
        mult, blocks = _blocks_multiplier(cfg, d, rng)
        prob = prob * mult
    if "scale_free" in structures:
        w = _sf_weights(cfg, active, rng)
        prob = prob * np.outer(w, w)
    if "layered" in structures:
        mult, layers = _layered_multiplier(cfg, d)
        prob = prob * mult
    pairs = (
        np.triu(np.ones((d, d), dtype=bool), k=1) & active[:, None] & active[None, :]
    )
    prob = np.where(active[:, None] & active[None, :], prob, 0.0)
    if cfg.xx_density <= 0 or not pairs.any() or prob[pairs].sum() <= 0:
        return np.zeros((d, d)), blocks, layers
    for _ in range(100):
        prob = prob * (cfg.xx_density / prob[pairs].mean())
        if prob.max() <= 1 + 1e-9:
            break
        prob = np.minimum(prob, 1.0)
    return np.clip(prob, 0.0, 1.0), blocks, layers


def graph_properties(g: SampledGraph) -> dict:
    """Measure X–X DAG properties used to select presets for coverage."""
    xx = g.xx_edges.astype(int)
    d = xx.shape[0]
    n_pairs = d * (d - 1) / 2
    deg = xx.sum(axis=0) + xx.sum(axis=1)
    # Depth is the longest path length; the DAG follows index order.
    depth = np.zeros(d, dtype=int)
    for j in range(d):
        parents = np.flatnonzero(xx[:, j])
        depth[j] = depth[parents].max() + 1 if parents.size else 0
    active = _active_mask(g.roles)
    n_active_pairs = int(active.sum()) * (int(active.sum()) - 1) / 2
    props = {
        "structure": g.structure,
        # density counts edges among all X pairs, as before. *_requested, *_achieved and
        # *_feasible_max use active pairs only: noise nodes, instruments and colliders get no
        # edges. These values can be compared with xx_density.
        "density": float(xx.sum() / n_pairs) if n_pairs else 0.0,
        "density_requested": float(g.xx_density_requested),
        "density_achieved": float(xx.sum() / n_active_pairs) if n_active_pairs else 0.0,
        "density_feasible_max": float(n_active_pairs / n_pairs) if n_pairs else 0.0,
        "max_degree": int(deg.max()) if d else 0,
        "degree_gini": _gini(deg),
        "depth": int(depth.max()) if d else 0,
    }
    if g.blocks is not None:
        b = np.asarray(g.blocks)
        within = xx[b[:, None] == b[None, :]].sum()
        props["within_block_share"] = float(within / xx.sum()) if xx.sum() else 0.0
    # Share of pairs with identical parent sets among nodes with at least one parent. Group
    # identical columns instead of checking every pair: the O(d²) pairwise check took 0.2 s at d =
    # 400.
    with_parents = np.flatnonzero(xx.any(axis=0))
    n_wp = with_parents.size
    n_pairs_wp = n_wp * (n_wp - 1) // 2
    if n_pairs_wp:
        _, counts = np.unique(xx[:, with_parents], axis=1, return_counts=True)
        dup = int((counts * (counts - 1) // 2).sum())
        props["duplicate_pairs_share"] = float(dup / n_pairs_wp)
    else:
        props["duplicate_pairs_share"] = 0.0
    if g.prototype is not None:
        props["clone_share"] = float(np.mean(np.asarray(g.prototype) >= 0))
    props.update(_dag_shape(xx, active))  # A.5
    props.update(_hub_neighbourhood(xx))  # B.1
    props["triad_profile"] = _triad_profile(xx)  # B.2
    return props


def _dag_shape(xx: np.ndarray, active: np.ndarray) -> dict:
    """Summarize the DAG shape among active nodes (A.5), excluding noise nodes, instruments and colliders.

    An isolated active node counts as a root, a sink and a connected component of its
    own.
    """
    in_deg, out_deg = xx.sum(axis=0), xx.sum(axis=1)
    n_active = int(active.sum())
    idx = np.flatnonzero(active)
    sub = xx[np.ix_(idx, idx)]
    und = (sub + sub.T) > 0
    seen = np.zeros(len(idx), dtype=bool)
    n_comp = 0
    for start in range(len(idx)):
        if seen[start]:
            continue
        n_comp += 1
        stack = [start]
        seen[start] = True
        while stack:
            v = stack.pop()
            for u in np.flatnonzero(und[v] & ~seen):
                seen[u] = True
                stack.append(int(u))
    deg_active = und.sum(axis=1)
    return {
        "n_active": n_active,
        "n_roots": int((active & (in_deg == 0)).sum()),
        "n_sinks": int((active & (out_deg == 0)).sum()),
        "n_components": n_comp,
        "isolated_active_share": float((deg_active == 0).mean()) if n_active else 0.0,
        "max_in_degree": int(in_deg.max()) if in_deg.size else 0,
    }


def _hub_neighbourhood(xx: np.ndarray) -> dict:
    """Measure the association between a node's degree and the degrees of its neighbors.

    B.1 (arXiv:2607.26624): Compute Newman's assortativity and the slope of k_nn(k)
    against log k on the undirected X–X skeleton. The semilog regression of k_nn on log
    k is named `knn_semilog_slope`; the `knn_log_slope` alias was removed in 0.6.0.
    These measures are informative around d = 100 and above. At d = 18 they vary
    across seeds by about ±0.1 to ±0.3. For rank-one families (ER and Chung–Lu in the
    admissible regime), both are approximately zero. Return NaN when fewer than four
    nodes have edges or there are fewer than two edges.
    """
    a = ((xx + xx.T) > 0).astype(int)
    deg = a.sum(axis=1)
    has = deg > 0
    out = {"assortativity": float("nan"), "knn_semilog_slope": float("nan")}
    if has.sum() < 4:
        return out
    knn_v = np.einsum("ij,j->i", a, deg) / np.maximum(deg, 1)
    x = np.log(deg[has].astype(float))
    y = knn_v[has]
    x0 = x - x.mean()
    denom = (x0**2).sum()
    if denom > 1e-12:
        out["knn_semilog_slope"] = float((x0 * (y - y.mean())).sum() / denom)
    i, j = np.where(np.triu(a) > 0)
    if len(i) >= 2:
        di, dj = deg[i].astype(float), deg[j].astype(float)
        both = np.r_[di, dj]
        if both.std() > 1e-12:
            out["assortativity"] = float(np.corrcoef(both, np.r_[dj, di])[0, 1])
    return out


def _triad_profile(xx: np.ndarray) -> dict:
    """Compute the fractions of connected triples by type.

    B.2: Count connected triples with at least two DAG edges using closed-form degree
    formulas. A chain is i->j->k without i->k. A fork is i->j and i->k without a j–k
    edge. A collider is i->k and j->k without an i–j edge. A transitive triple contains
    all three edges. A DAG has no reciprocal edges or cycles. Return NaN when there are
    no triples.
    """
    a = xx.astype(np.int64)
    out_deg, in_deg = a.sum(axis=1), a.sum(axis=0)
    af = a.astype(float)  # einsum avoids BLAS; the integer counts are exact.
    t = round((np.einsum("ij,jk->ik", af, af, optimize=False) * af).sum())
    fork = int((out_deg * (out_deg - 1) // 2).sum()) - t
    coll = int((in_deg * (in_deg - 1) // 2).sum()) - t
    chain = int((in_deg * out_deg).sum()) - t
    total = fork + coll + chain + t
    if total == 0:
        return {k: float("nan") for k in ("chain", "fork", "collider", "transitive")}
    return {
        "chain": chain / total,
        "fork": fork / total,
        "collider": coll / total,
        "transitive": t / total,
    }


def _gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    return float(np.einsum("k,k->", 2 * np.arange(1, n + 1) - n - 1, v) / (n * v.sum()))
