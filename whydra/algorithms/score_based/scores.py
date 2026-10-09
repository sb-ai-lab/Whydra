"""Local scores for score-based search, plus the cache GES relies on.

``local_parent_score`` used to live in ``standalone/gfci_algo.py``; it moved
here so that ``ges.py`` can use it without importing ``gfci_algo`` (which
imports GES in turn). ``gfci_algo`` re-exports it under its old private name
for backwards compatibility.
"""

from __future__ import annotations

import math

import numpy as np


def local_parent_score(data, child, parents, score_func) -> float:
    """Local Gaussian-BIC or discrete-BDeu score of ``child`` given ``parents``.

    Both branches are invariant to the order of ``parents``, which is what
    lets :class:`LocalScoreCache` key on a frozenset.
    """
    data = np.asarray(data)
    n = data.shape[0]
    parents = list(parents)
    if str(score_func).lower().endswith("bdeu"):
        # Uniform BDeu prior (Cooper-Herskovits / Heckerman et al.), split
        # evenly across parent configurations and then across child states:
        # alpha_j = ess / q, alpha_jk = alpha_j / r. ess=1 matches the
        # conventional default equivalent sample size used by bnlearn/pcalg.
        ess = 1.0
        y = np.asarray(data[:, child], dtype=int)
        r = int(y.max()) + 1 if y.size else 1
        if not parents:
            q = 1
            configs = np.zeros(n, dtype=int)
        else:
            x = np.asarray(data[:, parents], dtype=int)
            cardinalities = [int(np.max(data[:, p])) + 1 for p in parents]
            configs = np.ravel_multi_index(x.T, cardinalities)
            q = int(np.prod(cardinalities))
        alpha_j = ess / q
        alpha_jk = alpha_j / r
        score = 0.0
        for config in np.unique(configs):
            counts = np.bincount(y[configs == config], minlength=r)
            n_j = int(counts.sum())
            score += math.lgamma(alpha_j) - math.lgamma(n_j + alpha_j)
            for n_jk in counts:
                if n_jk:
                    score += math.lgamma(int(n_jk) + alpha_jk) - math.lgamma(alpha_jk)
        return float(score)

    # TODO(ges-tetrad-sembic): эта формула совпадает с ``local_score_BIC`` из
    # causal-learn, но не с ``sem-bic`` из Tetrad, из-за чего сверка GFCI с
    # эталонным GFCI (local_checks/gfci_vs_tetrad.py) меряет сумму двух разниц —
    # алгоритма и score-функции — и точного совпадения дать не может.
    #
    # Замер: на одном и том же графе эта формула даёт -821.18, а Tetrad
    # сообщает -972.30 (penaltyDiscount=1.0, p=7, n=3000). Перебор 24 комбинаций
    # не воспроизводит их число: количество параметров |Pa|, |Pa|+1, |Pa|+2;
    # оценка дисперсии rss/n, rss/(n-1), rss/(n-k), rss/(n-k-1); множитель n или
    # n-1. Ближайшая -n*log(rss/(n-k-1)) - (|Pa|+2)*log(n) промахивается на 69.
    # Значит формулы расходятся структурно, а не на константу penaltyDiscount.
    #
    # Проверено и отброшено как причина: локаль JVM (влияет только на печать),
    # формат чисел в CSV, maxDegree, completeRuleSetUsed, перебор
    # penaltyDiscount 0.5-2.0 (лучшее совпадение 6/10, не сводит).
    #
    # Просто перейти на формулу Tetrad НЕЛЬЗЯ вслепую: по SHD до истинного CPDAG
    # (10 графов, n=3000) текущая формула даёт 26, Tetrad FGES — 44, то есть
    # их score на этой синтетике хуже. Нужно поднять точную формулу из
    # исходников Tetrad (SemBicScore) и завести её отдельным вариантом
    # ``score_func`` для сверки, а не менять умолчание.
    y = np.asarray(data[:, child], dtype=float)
    design = np.ones((n, 1)) if not parents else np.column_stack(
        (np.ones(n), np.asarray(data[:, parents], dtype=float))
    )
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    rss = max(float(np.dot(residual, residual)), np.finfo(float).eps)
    return float(-n * np.log(rss / n) - design.shape[1] * np.log(n))


class LocalScoreCache:
    """Memoises ``local_parent_score`` over ``(child, frozenset(parents))``.

    GES asks for the same local scores over and over: every ``Insert(x, y, T)``
    candidate needs ``score(y, Pa(y) u NA_yx u T)``, which is shared across all
    candidate ``x``, and the whole table is queried again on the next sweep.
    """

    __slots__ = ("_data", "_score_func", "_cache", "hits", "misses")

    def __init__(self, data, score_func):
        self._data = np.asarray(data)
        self._score_func = score_func
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def score(self, child, parents) -> float:
        key = (int(child), frozenset(int(p) for p in parents))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = local_parent_score(self._data, key[0], sorted(key[1]), self._score_func)
        self._cache[key] = value
        return value
