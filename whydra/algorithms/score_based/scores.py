"""Local scores for score-based search, plus the cache GES relies on.

GES and GFCI share ``local_parent_score`` here to avoid a circular import.
``gfci_algo`` also exposes its private alias for callers using that name.
"""

import math

import numpy as np

SCORE_FUNCTIONS = frozenset({"local_score_BIC", "local_score_BDeu"})


def validate_score_func(score_func):
    """Return the score name, raising ValueError for an unregistered name."""
    if score_func not in SCORE_FUNCTIONS:
        msg = f"Unknown score_func {score_func!r}; available: {sorted(SCORE_FUNCTIONS)}"
        raise ValueError(msg)
    return score_func


def _encode_categories(data):
    from ..graph_core.validation import validate_data

    data = validate_data(data)
    encoded = np.empty(data.shape, dtype=np.int64)
    cardinalities = []
    for j in range(data.shape[1]):
        levels, encoded[:, j] = np.unique(data[:, j], return_inverse=True)
        cardinalities.append(len(levels))
    return encoded, cardinalities


def local_parent_score(data, child, parents, score_func) -> float:
    """Local Gaussian-BIC or discrete-BDeu score of ``child`` given ``parents``.

    Both branches are invariant to the order of ``parents``, which is what
    lets :class:`LocalScoreCache` key on a frozenset.
    """
    validate_score_func(score_func)
    if score_func == "local_score_BDeu":
        data, cardinalities = _encode_categories(data)
        return _local_score(data, child, parents, score_func, cardinalities)
    from ..graph_core.validation import validate_data

    return _local_score(validate_data(data, numeric=True), child, parents, score_func)


def _local_score(data, child, parents, score_func, cardinalities=None):
    n = data.shape[0]
    parents = sorted(set(parents))
    if child in parents:
        msg = "A node cannot be its own parent"
        raise ValueError(msg)
    if score_func == "local_score_BDeu":
        # Uniform BDeu prior (Cooper-Herskovits / Heckerman et al.), split
        # evenly across parent configurations and then across child states:
        # alpha_j = ess / q, alpha_jk = alpha_j / r. ess=1 matches the
        # intended default equivalent sample size.
        ess = 1.0
        y = np.asarray(data[:, child], dtype=int)
        assert cardinalities is not None
        r = cardinalities[child]
        if not parents:
            q = 1
            configs = np.zeros(n, dtype=int)
        else:
            x = np.asarray(data[:, parents], dtype=int)
            _, configs = np.unique(x, axis=0, return_inverse=True)
            q = math.prod(cardinalities[p] for p in parents)
        alpha_j = ess / q
        alpha_jk = alpha_j / r
        if alpha_jk == 0:
            msg = "BDeu prior underflows for this parent cardinality product"
            raise ValueError(msg)
        score = 0.0
        for config in np.unique(configs):
            counts = np.bincount(y[configs == config], minlength=r)
            n_j = int(counts.sum())
            score += math.lgamma(alpha_j) - math.lgamma(n_j + alpha_j)
            for n_jk in counts:
                if n_jk:
                    score += math.lgamma(int(n_jk) + alpha_jk) - math.lgamma(alpha_jk)
        return float(score)

    y = np.asarray(data[:, child], dtype=float)
    design = (
        np.ones((n, 1))
        if not parents
        else np.column_stack((np.ones(n), np.asarray(data[:, parents], dtype=float)))
    )
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    rss = max(float(np.dot(residual, residual)), np.finfo(float).eps)
    return float(-n * np.log(rss / n) - design.shape[1] * np.log(n))


_BUILTIN_LOCAL_SCORE = _local_score


class LocalScoreCache:
    """Memoises ``local_parent_score`` over ``(child, frozenset(parents))``.

    GES asks for the same local scores over and over: every ``Insert(x, y, T)``
    candidate needs ``score(y, Pa(y) u NA_yx u T)``, which is shared across all
    candidate ``x``, and the whole table is queried again on the next sweep.
    """

    __slots__ = (
        "_backend",
        "_cache",
        "_cardinalities",
        "_covariance",
        "_data",
        "_design_conditions",
        "_errors",
        "_execution_stack",
        "_executor",
        "_gram",
        "_log_n",
        "_n_jobs",
        "_parallel_backend",
        "_precise",
        "_revisions",
        "_safe_columns",
        "_score_func",
        "fallback_scores",
        "fast_scores",
        "hits",
        "misses",
        "parallel_batches",
        "process_batches",
        "process_rounds",
        "refinements",
    )

    def __init__(self, data, score_func, *, bic_backend="reference"):
        from ..graph_core.validation import validate_data

        self._score_func = validate_score_func(score_func)
        if bic_backend not in ("reference", "covariance", "gram"):
            msg = "bic_backend must be 'reference', 'covariance' or 'gram'"
            raise ValueError(msg)
        if bic_backend in ("covariance", "gram") and score_func != "local_score_BIC":
            msg = f"{bic_backend} backend requires local_score_BIC"
            raise ValueError(msg)
        self._backend = bic_backend
        self._data: np.ndarray
        self._cardinalities = None
        if score_func == "local_score_BDeu":
            self._data, self._cardinalities = _encode_categories(data)
        else:
            self._data = validate_data(data, numeric=True)
        self._reset_scores()
        if bic_backend == "covariance":
            from ._covariance_bic import CovarianceBIC

            self._covariance = CovarianceBIC(self._data)
        if bic_backend == "gram" and _local_score is _BUILTIN_LOCAL_SCORE:
            # With an intercept, Gaussian residual sums of squares depend only
            # on centred cross-products. Build them once instead of fitting an
            # n-row design matrix for every candidate parent set.
            means = self._data.mean(axis=0)
            centred = self._data - means
            with np.errstate(over="ignore", invalid="ignore"):
                self._gram = centred.T @ centred
                scales = np.sqrt(np.diag(self._gram) / len(self._data))
                # The intercept design may be rank-deficient numerically even
                # if the centred parent block is perfectly conditioned: a
                # large mean and scale combine quadratically in its condition.
                design_scale = np.maximum(1.0, np.abs(means) + scales)
                self._design_conditions = design_scale**2 / np.maximum(
                    scales, np.finfo(float).tiny
                )
            # Extreme offsets/scales can change the numerical rank chosen by
            # lstsq on the original intercept design. Retain that exact path.
            self._safe_columns = (
                np.isfinite(scales)
                & (scales >= 1e-6)
                & (scales <= 1e6)
                & (np.abs(means) <= 1e4 * scales)
                & (
                    self._design_conditions
                    <= min(1e8, 1 / (128 * np.finfo(float).eps * len(self._data)))
                )
            )

    def _reset_scores(self):
        self._cache = {}
        self._covariance = None
        self._revisions = [0] * self._data.shape[1]
        self._executor = self._execution_stack = None
        self._n_jobs = 1
        self._parallel_backend = "auto"
        self.hits = self.misses = 0
        self._gram = self._safe_columns = self._design_conditions = None
        self._errors = {}
        self._precise = set()
        self._log_n = math.log(len(self._data))
        self.fast_scores = self.fallback_scores = self.refinements = (
            self.parallel_batches
        ) = 0
        self.process_rounds = self.process_batches = 0

    @classmethod
    def _from_gaussian_statistics(cls, data, gram, safe_columns, design_conditions):
        """Create a worker-local cache over validated read-only fit inputs."""
        cache = cls.__new__(cls)
        cache._data = data
        cache._score_func = "local_score_BIC"
        cache._backend = "gram"
        cache._cardinalities = None
        cache._reset_scores()
        cache._gram = gram
        cache._safe_columns = safe_columns
        cache._design_conditions = design_conditions
        return cache

    def score(self, child, parents) -> float:
        """Return the local score of a child and its parents, reusing cached results."""
        key = (int(child), frozenset(int(p) for p in parents))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        if self._backend != "gram":
            self.misses += 1
            ordered = tuple(sorted(key[1]))
            value = (
                self._covariance.score(key[0], ordered)
                if self._covariance is not None
                else None
            )
            if value is None:
                value = _local_score(
                    self._data, key[0], ordered, self._score_func, self._cardinalities
                )
            self._cache[key] = value
            return value
        self.prefetch([key])
        return self._cache[key]

    def configure_execution(self, n_jobs, backend, stack):
        """Set the worker count, parallel backend and execution stack for scoring."""
        self._n_jobs, self._parallel_backend, self._execution_stack = (
            n_jobs,
            backend,
            stack,
        )

    def _score_batch(self, keys):
        """Return scores and conservative roundoff estimates without mutation."""
        if self._gram is None or _local_score is not _BUILTIN_LOCAL_SCORE:
            return [
                (
                    key,
                    _local_score(
                        self._data,
                        key[0],
                        sorted(key[1]),
                        self._score_func,
                        self._cardinalities,
                    ),
                    0.0,
                    True,
                )
                for key in keys
            ]
        n = len(self._data)
        k = len(keys[0][1])
        children = np.asarray([key[0] for key in keys], dtype=int)
        parents = np.asarray([sorted(key[1]) for key in keys], dtype=int).reshape(
            len(keys), k
        )
        assert self._safe_columns is not None
        assert self._design_conditions is not None
        eligible = self._safe_columns[children].copy()
        if k:
            eligible &= self._safe_columns[parents].all(axis=1)
        if n <= k + 1:
            eligible[:] = False
        child_ss = self._gram[children, children]
        rss = child_ss.copy()
        conditions = self._design_conditions[children].copy()
        if k:
            conditions = np.maximum(
                conditions, self._design_conditions[parents].max(axis=1)
            )
        explained = np.zeros(len(keys))
        if k:
            blocks = self._gram[parents[:, :, None], parents[:, None, :]]
            rhs = self._gram[parents, children[:, None]]
            selected = np.flatnonzero(eligible)
            if len(selected):
                try:
                    eigenvalues = np.linalg.eigvalsh(blocks[selected])
                    well_conditioned = eigenvalues[:, 0] > eigenvalues[:, -1] * 1e-8
                    eligible[selected[~well_conditioned]] = False
                    selected = selected[well_conditioned]
                    if len(selected):
                        conditions[selected] = np.maximum(
                            conditions[selected],
                            eigenvalues[well_conditioned, -1]
                            / eigenvalues[well_conditioned, 0],
                        )
                        coefficients = np.linalg.solve(
                            blocks[selected], rhs[selected, :, None]
                        )[:, :, 0]
                        explained[selected] = np.einsum(
                            "ij,ij->i", rhs[selected], coefficients
                        )
                        rss[selected] -= explained[selected]
                except np.linalg.LinAlgError:
                    # A failing member must not make a different score fail.
                    eligible[selected] = False
        # The reference clamps RSS to eps. Near that floor, covariance
        # cancellation and the clamp must be resolved by the original lstsq.
        eligible &= (
            np.isfinite(rss) & (rss > child_ss * 1e-8) & (rss > 8 * np.finfo(float).eps)
        )
        values = np.empty(len(keys))
        errors = np.zeros(len(keys))
        good = np.flatnonzero(eligible)
        if len(good):
            values[good] = -n * np.log(rss[good] / n) - (k + 1) * self._log_n
            # Used only to decide when to recompute competitive operators with
            # the original lstsq score. This never changes a tie/min_delta.
            rss_error = (
                64
                * np.finfo(float).eps
                * (n + k + 1)
                * conditions[good]
                * (child_ss[good] + np.abs(explained[good]))
            )
            errors[good] = n * rss_error / rss[good] + 32 * np.finfo(float).eps * (
                1 + np.abs(values[good])
            )
        for index in np.flatnonzero(~eligible):
            key = keys[index]
            values[index] = _local_score(
                self._data,
                key[0],
                sorted(key[1]),
                self._score_func,
                self._cardinalities,
            )
        return [
            (key, float(values[index]), float(errors[index]), not bool(eligible[index]))
            for index, key in enumerate(keys)
        ]

    def prefetch(self, keys, *, executor=None, n_jobs=1):
        """Compute unique missing local scores in batches of equal parent size.

        Worker threads only read sufficient statistics and return values. The
        parent owns all cache mutation, so no growing Python cache is copied
        to workers and the result is independent of scheduling.
        """
        missing = list(dict.fromkeys(key for key in keys if key not in self._cache))
        if self._backend != "gram":
            # Prefetch must retain the caller's explicit numerical backend.
            for child, parents in missing:
                self.score(child, parents)
            return
        n_jobs = max(n_jobs, self._n_jobs)
        groups = {}
        for key in missing:
            if key[0] in key[1]:
                msg = "A node cannot be its own parent"
                raise ValueError(msg)
            groups.setdefault(len(key[1]), []).append(key)
        batches = []
        for group in groups.values():
            # Small vectorized scores are faster without executor dispatch.
            # Larger linear-algebra batches release the GIL inside NumPy.
            chunk_size = max(512, (len(group) + n_jobs - 1) // n_jobs)
            batches.extend(
                group[start : start + chunk_size]
                for start in range(0, len(group), chunk_size)
            )
        compatible = (
            self._gram is not None
            and _local_score is _BUILTIN_LOCAL_SCORE
            and type(self) is LocalScoreCache
            and type(self)._score_batch is _BUILTIN_BATCH_SCORER
            and type(self)._from_gaussian_statistics.__func__ is _BUILTIN_CACHE_FACTORY
        )
        # Explicit backends permit measurement. Measured sparse searches are
        # faster with serial vectorization, so auto does not start a pool.
        parallel = (
            compatible
            and n_jobs > 1
            and len(batches) > 1
            and self._parallel_backend != "auto"
        )
        if parallel and self._parallel_backend == "processes":
            from ..graph_core.process_execution import process_map
            from ._gram_worker import score_batch

            arguments = (
                (
                    self._data,
                    self._gram,
                    self._safe_columns,
                    self._design_conditions,
                    batch,
                )
                for batch in batches
            )
            results = process_map(
                score_batch,
                arguments,
                n_jobs=min(n_jobs, len(batches)),
                shared_arg_indices=(0, 1, 2, 3),
            )
            self.process_rounds += 1
            self.process_batches += len(batches)
        else:
            if parallel and executor is None and self._execution_stack is not None:
                if self._executor is None:
                    from concurrent.futures import ThreadPoolExecutor

                    self._executor = self._execution_stack.enter_context(
                        ThreadPoolExecutor(max_workers=n_jobs)
                    )
                executor = self._executor
            if parallel and executor is not None:
                results = executor.map(self._score_batch, batches)
                self.parallel_batches += len(batches)
            else:
                results = map(self._score_batch, batches)
        for batch in results:
            for key, value, error, precise in batch:
                self._cache[key] = value
                self._errors[key] = error
                if precise:
                    self._precise.add(key)
                    self.fallback_scores += 1
                else:
                    self.fast_scores += 1
                self.misses += 1

    def refine(self, keys):
        """Recompute uncertain competitive scores with the original scorer."""
        changed = False
        for key in dict.fromkeys(keys):
            if key in self._precise:
                continue
            self._cache[key] = _local_score(
                self._data,
                key[0],
                sorted(key[1]),
                self._score_func,
                self._cardinalities,
            )
            self._errors[key] = 0.0
            self._precise.add(key)
            self.refinements += 1
            self._revisions[key[0]] += 1
            changed = True
        return changed


_BUILTIN_BATCH_SCORER = LocalScoreCache._score_batch
_BUILTIN_CACHE_FACTORY = LocalScoreCache._from_gaussian_statistics.__func__
