import copy
from itertools import combinations
from collections import deque
import numpy as np
import math
import hashlib
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from joblib import Parallel, delayed

# Импортируем универсальный класс тестов из соседнего файла
from ..graph_core import CIT, CausalGraph
from ..graph_core.endpoints import Endpoint
from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)

# --- Вспомогательные функции ---
from .skeleton import SkeletonDiscovery
from .pag_init import to_circle_adjacency
from .pag_rules import faith_check, minUncovCircPath, minUncovPdPath


# ============================================================
#  Локальные копии вспомогательных функций FCI.
#  Перенесены из прежнего fci_algo.py.
#  Держим их здесь, чтобы файл не зависел от fci_algo.py:
#  там лежит эталонная реализация FCI из ветки dev, без
#  background_knowledge / protected_edges / rule_stats.
# ============================================================

class _SharedRFCICache:
    """Shared-memory hash table cache for CI p-values."""

    def __init__(self, capacity=1_000_003, n_locks=64, *, create=True, names=None, locks=None, owner=False):
        self.capacity = int(capacity)
        self.n_locks = int(n_locks)
        self.owner = bool(owner)
        self._names = names
        self._locks = locks
        self._manager = None

        self._shm_hi = None
        self._shm_lo = None
        self._shm_val = None
        self._shm_state = None
        self._keys_hi = None
        self._keys_lo = None
        self._values = None
        self._state = None

        if create:
            self._shm_hi = shared_memory.SharedMemory(create=True, size=self.capacity * 8)
            self._shm_lo = shared_memory.SharedMemory(create=True, size=self.capacity * 8)
            self._shm_val = shared_memory.SharedMemory(create=True, size=self.capacity * 8)
            self._shm_state = shared_memory.SharedMemory(create=True, size=self.capacity)
            self._names = (
                self._shm_hi.name,
                self._shm_lo.name,
                self._shm_val.name,
                self._shm_state.name,
            )
            if locks is None:
                # Picklable lock proxies for loky workers.
                self._manager = mp.Manager()
                self._locks = [self._manager.Lock() for _ in range(self.n_locks)]
            else:
                self._locks = locks
            self.owner = True
            self._attach_arrays()
            self._state[:] = 0

    def __getstate__(self):
        return {
            "capacity": self.capacity,
            "n_locks": self.n_locks,
            "owner": False,
            "names": self._names,
            "locks": self._locks,
        }

    def __setstate__(self, state):
        self.capacity = state["capacity"]
        self.n_locks = state["n_locks"]
        self.owner = state["owner"]
        self._names = state["names"]
        self._locks = state["locks"]
        self._manager = None

        self._shm_hi = None
        self._shm_lo = None
        self._shm_val = None
        self._shm_state = None
        self._keys_hi = None
        self._keys_lo = None
        self._values = None
        self._state = None

    def _ensure_attached(self):
        if self._keys_hi is None:
            n_hi, n_lo, n_val, n_state = self._names
            self._shm_hi = shared_memory.SharedMemory(name=n_hi)
            self._shm_lo = shared_memory.SharedMemory(name=n_lo)
            self._shm_val = shared_memory.SharedMemory(name=n_val)
            self._shm_state = shared_memory.SharedMemory(name=n_state)
            self._attach_arrays()

    def _attach_arrays(self):
        self._keys_hi = np.ndarray((self.capacity,), dtype=np.uint64, buffer=self._shm_hi.buf)
        self._keys_lo = np.ndarray((self.capacity,), dtype=np.uint64, buffer=self._shm_lo.buf)
        self._values = np.ndarray((self.capacity,), dtype=np.float64, buffer=self._shm_val.buf)
        self._state = np.ndarray((self.capacity,), dtype=np.uint8, buffer=self._shm_state.buf)

    @staticmethod
    def _fingerprint(key_obj):
        a, b, cond = key_obj
        h = hashlib.blake2b(digest_size=16)
        h.update(int(a).to_bytes(4, "little", signed=False))
        h.update(int(b).to_bytes(4, "little", signed=False))
        h.update(len(cond).to_bytes(4, "little", signed=False))
        for x in cond:
            h.update(int(x).to_bytes(4, "little", signed=False))
        digest = h.digest()
        hi = np.uint64(int.from_bytes(digest[:8], "little", signed=False))
        lo = np.uint64(int.from_bytes(digest[8:], "little", signed=False))
        return hi, lo

    def _start_idx(self, hi, lo):
        return int((int(hi) ^ int(lo)) % self.capacity)

    def get(self, key_obj):
        self._ensure_attached()
        hi, lo = self._fingerprint(key_obj)
        idx = self._start_idx(hi, lo)
        for _ in range(self.capacity):
            st = self._state[idx]
            if st == 0:
                return None
            if st == 1 and self._keys_hi[idx] == hi and self._keys_lo[idx] == lo:
                return float(self._values[idx])
            idx = (idx + 1) % self.capacity
        return None

    def set(self, key_obj, val):
        self._ensure_attached()
        hi, lo = self._fingerprint(key_obj)
        if self._locks is None:
            return
        # Лок берём от того же значения, от которого считается слот:
        # ключи с разным hi, но одинаковым hi^lo делят слот, и по int(hi)
        # они брали разные локи — запись из четырёх массивов рвалась.
        lock = self._locks[self._start_idx(hi, lo) % self.n_locks]
        with lock:
            idx = self._start_idx(hi, lo)
            for _ in range(self.capacity):
                st = self._state[idx]
                if st == 0 or (self._keys_hi[idx] == hi and self._keys_lo[idx] == lo):
                    self._keys_hi[idx] = hi
                    self._keys_lo[idx] = lo
                    self._values[idx] = float(val)
                    self._state[idx] = 1
                    return
                idx = (idx + 1) % self.capacity

    def close(self):
        for shm in (self._shm_hi, self._shm_lo, self._shm_val, self._shm_state):
            if shm is not None:
                shm.close()
        if self.owner and self._manager is not None:
            self._manager.shutdown()

    def unlink(self):
        if not self.owner:
            return
        for shm in (self._shm_hi, self._shm_lo, self._shm_val, self._shm_state):
            if shm is not None:
                shm.unlink()

def _ci_cache_key(i: int, j: int, S):
    """Canonical cache key for CI test symmetry in (i, j, S)."""
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset)):
        cond = tuple(sorted(int(x) for x in S))
    elif isinstance(S, (list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def _process_ci_eval(cit, shared_cache, x, y, S):
    key = _ci_cache_key(x, y, S)
    if shared_cache is not None:
        cached = shared_cache.get(key)
        if cached is not None:
            return cached, True
        pval = cit(x, y, list(key[2]))
        shared_cache.set(key, pval)
        return pval, False
    return cit(x, y, list(key[2])), None


def _process_ci_chunk(cit, shared_cache, x, y, cond_sets):
    return [_process_ci_eval(cit, shared_cache, x, y, S) for S in cond_sets]


def _record_process_ci_stats(cg, total, hits=0, misses=0):
    ci_stats = getattr(cg, "_parallel_ci_stats", None)
    phase = getattr(cg, "_parallel_phase", None)
    if ci_stats is None or phase is None:
        return
    ci_stats["total"] += int(total)
    ci_stats[phase["name"]] += int(total)
    if "ci_cache_hits" in ci_stats:
        ci_stats["ci_cache_hits"] += int(hits)
    if "ci_cache_misses" in ci_stats:
        ci_stats["ci_cache_misses"] += int(misses)


def _ci_batch(cg: CausalGraph, x: int, y: int, cond_sets, n_jobs: int = 1):
    """Evaluate CI tests for fixed (x, y) over many conditioning sets."""
    cond_sets = list(cond_sets)
    if not cond_sets:
        return []
    if n_jobs in (None, 0, 1):
        return [cg.ci_test(x, y, list(S)) for S in cond_sets]

    parallel_backend = getattr(cg, "_parallel_backend", "threads")
    parallel_executor = getattr(cg, "_parallel_executor", None)
    n_jobs_eff = int(n_jobs)
    if parallel_backend != "processes":
        owns_executor = parallel_executor is None
        executor = parallel_executor or ThreadPoolExecutor(max_workers=n_jobs_eff)
        try:
            futures = [
                executor.submit(cg.ci_test, x, y, list(S))
                for S in cond_sets
            ]
            return [future.result() for future in futures]
        finally:
            if owns_executor:
                executor.shutdown(wait=True)

    # rfci_stable выставляет _parallel_backend="processes" и _parallel_cit под
    # одним условием use_process_orient, поэтому здесь cit всегда есть.
    parallel_cit = cg._parallel_cit

    shared_cache = getattr(cg, "_parallel_shared_cache", None)
    chunk_size = max(1, math.ceil(len(cond_sets) / n_jobs_eff))
    chunks = [cond_sets[i:i + chunk_size] for i in range(0, len(cond_sets), chunk_size)]
    executor = parallel_executor or Parallel(n_jobs=n_jobs_eff, backend="loky")
    chunk_results = executor(
        delayed(_process_ci_chunk)(parallel_cit, shared_cache, x, y, chunk)
        for chunk in chunks
    )
    results = [item for chunk in chunk_results for item in chunk]
    pvals = [pval for pval, _hit in results]
    if shared_cache is None:
        _record_process_ci_stats(cg, len(results), misses=len(results))
    else:
        hits = sum(1 for _pval, hit in results if hit)
        misses = len(results) - hits
        _record_process_ci_stats(cg, len(results), hits=hits, misses=misses)
    return pvals


def _resolve_rfci_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        effective = "processes" if int(n_vars) > int(process_min_vars) else "threads"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        raise ValueError("RFCI backend must be 'auto', 'threads', or 'processes'")
    return requested, effective


def _sepset_to_set(seps):
    """Normalize sepset payload into set[int]."""
    if seps is None:
        return set()
    if isinstance(seps, (set, frozenset)):
        return {int(x) for x in seps}
    if isinstance(seps, (np.integer, int)):
        return {int(seps)}
    if isinstance(seps, (list, tuple, np.ndarray)):
        if len(seps) == 0:
            return set()
        if all(isinstance(x, (int, np.integer)) for x in seps):
            return {int(x) for x in seps}
        candidates = []
        for S in seps:
            if S is None:
                continue
            if isinstance(S, (set, frozenset)):
                candidates.append({int(x) for x in S})
            elif isinstance(S, (list, tuple, np.ndarray)):
                candidates.append({int(x) for x in S})
            else:
                candidates.append({int(S)})
        # TODO(rfci-sepset-union): при нескольких записанных разделяющих
        # множествах берётся минимальное по мощности, а не объединение. Из-за
        # этого sep_ik в _sep_union занижен, `j not in sep_ik` срабатывает чаще
        # и ориентируется больше коллайдеров, чем следует.
        # Проверено вызовом: _sepset_to_set([(1,4),(5,)]) -> {5}, а не {1,4,5}.
        # Объединение, которое скелет уже посчитал, лежит в graphDict['sepset'],
        # но результат skeleton.run_fci() не забирается.
        #
        # Механическая замена на объединение НЕ помогает: замер на 100
        # конфигурациях дал 6/100 расхождений с causal-learn против 5/100 и
        # 23 ячейки против 21 — то есть чуть хуже. Вдобавок соседний
        # _sepset_to_pair_dict в fci_plus_algo прямо документирует обратное:
        # объединение там ломало ориентацию коллайдеров, и минимум выбран
        # намеренно.
        #
        # Нужно решение на уровне соглашения: чем должен быть sepset для R0 —
        # одним корректным множеством или объединением всех найденных, — и
        # привести к нему обе реализации разом.
        return min(candidates, key=len) if candidates else set()
    return {int(seps)}



def _sep_union(cg: CausalGraph, i: int, k: int):
    return _sepset_to_set(cg.sepset[i][k]) | _sepset_to_set(cg.sepset[k][i])

def _unshielded_triples(cg: CausalGraph):
    """
    Неэкранированные тройки (k, j, m): j смежен и с k, и с m, а k и m
    между собой не смежны. Возвращаются с k < m.

    Двух соседей обязан иметь СРЕДНИЙ узел тройки. Раньше это условие
    проверялось на конце k (`if len(cg.neighbors(k)) < 2: continue`),
    из-за чего терялись все тройки, у которых конец имеет ровно одного
    соседа, — то есть простейшие коллайдеры вида X1 *-> X3 <-* X2.
    Перебор по среднему узлу заодно убирает лишний проход по всем парам.
    """
    triples = []
    adj = cg.G.graph
    for j in range(len(cg.nodes)):
        neighbors_j = sorted(int(x) for x in cg.neighbors(j))
        if len(neighbors_j) < 2:
            continue
        for k, m in combinations(neighbors_j, 2):
            if not adj[k, m]:
                triples.append((k, j, m))
    return triples

# def _orient_collider(cg: CausalGraph):
#     triples = _unshielded_triples(cg)
#     for (i, j, k) in triples:
#         # edge_ij = cg.G.get_edge(cg.nodes[i], cg.nodes[j])
#         # edge_kj = cg.G.get_edge(cg.nodes[k], cg.nodes[j])
#         # print(edge_ij, edge_kj)
#         if not any(j in S for S in (cg.sepset[i][k] or [])) and (cg.is_undirected(i, j) and cg.is_undirected(k, j)):
#             cg.add_directed_edge(cg.nodes[i], cg.nodes[j])
#             cg.add_directed_edge(cg.nodes[k], cg.nodes[j])


def _find_minimal_set_from_candidates(cg: CausalGraph, x: int, y: int, candidates_set, alpha: float):
    cand = list(candidates_set)
    for size in range(0, len(cand) + 1):
        for subset in combinations(cand, size):
            if cg.ci_test(x, y, list(subset)) >= alpha:
                S = list(subset)
                cg.sepset[x][y] = S
                cg.sepset[y][x] = S
                return S
    return None


def _triple_contains_pair(triple, r, j):
    pair = {r, j}
    return ({triple[0], triple[1]} == pair) or ({triple[1], triple[2]} == pair)

def _as_pairs(protected_edges):
    """Набор ``frozenset({x, y})`` из любых пар индексов."""

    return {frozenset((int(x), int(y))) for x, y in protected_edges}


def _empty_guard() -> BKOrientationGuard:
    """Прозрачный guard: без BK ничего не запрещает, обвязка правил — no-op."""

    return BKOrientationGuard(None, MatrixEncoding.STANDARD)


def _orient_v_structures_rfci(cg: CausalGraph, alpha, M, unfVect=None, protected_edges=(),
                              guard=None):
    if unfVect is None:
        unfVect = set()
    if guard is None:
        guard = _empty_guard()
    protected_edges = {
        frozenset((int(x), int(y))) for x, y in protected_edges
    }

    L = []
    triples = deque(M)

    while triples:
        i, j, k = triples.popleft()
        sep_ik = _sep_union(cg, i, k)

        # C = sepset(Xi,Xk)\{Xj}  (S считаем пустым)
        C = sep_ik - {j}

        p_value_ij = cg.ci_test(i, j, list(C))
        p_value_jk = cg.ci_test(j, k, list(C))

        if p_value_ij < alpha and p_value_jk < alpha:
            if (i, j, k) not in unfVect and (k, j, i) not in unfVect:
                L.append((i, j, k))
        else:
            for r, other, pval in ((i, k, p_value_ij), (k, i, p_value_jk)):
                if pval >= alpha:
                    if frozenset((int(r), int(j))) in protected_edges:
                        continue
                    candidates = set(sep_ik)
                    candidates.discard(other)

                    _find_minimal_set_from_candidates(cg, r, j, candidates, alpha)

                    neighbors_r = set(cg.neighbors(r))
                    neighbors_j = set(cg.neighbors(j))
                    triangle_nodes = neighbors_r & neighbors_j

                    a = min(r, j)
                    c = max(r, j)
                    for t in triangle_nodes:
                        if t == a or t == c:
                            continue
                        triples.append((a, t, c))

                    L = [tr for tr in L if not _triple_contains_pair(tr, r, j)]
                    triples = deque([tr for tr in triples if not _triple_contains_pair(tr, r, j)])

                    cg.remove_edge(cg.nodes[r], cg.nodes[j])

    pag = cg.G.graph
    for i, j, k in L:
        sep_ik = _sep_union(cg, i, k)
        if (j not in sep_ik
            and (pag[i, j] != NULL and pag[j, i] != NULL)
            and (pag[k, j] != NULL and pag[j, k] != NULL)):
            guard.set_mark(pag, i, j, ARROW)
            guard.set_mark(pag, k, j, ARROW)

    cg.G.graph = pag


# preserving standard Endpoint values in the returned graph.
# Кодировка STANDARD. Имена называют ту метку, которую действительно хранят:
# раньше константа TAIL держала значение CIRCLE, а CIRCLE — значение TAIL, и
# вызов вида `circle_value=TAIL` читался как ошибка, хотя был верным.
CIRCLE_MARK = Endpoint.CIRCLE.value   # 2: открытый конец (кружок) в предикатах правил
ARROW = Endpoint.ARROW.value          # 1
TAIL_MARK = Endpoint.TAIL.value       # -1: хвост в предикатах правил
NULL = Endpoint.NULL.value            # 0


def _apply_r1_r3(pag, unfVect=None, guard=None):
    if unfVect is None: unfVect = {}
    if guard is None: guard = _empty_guard()

    #R1
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] == ARROW and pag[j][i] != NULL:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for a, b in ind:
        indC = [i for i in range(len(pag)) if
                pag[b][i] != NULL and pag[i][b] == CIRCLE_MARK and
                pag[a][i] == NULL and pag[i][a] == NULL and i != a]
        if len(indC) != 0:
            if len(unfVect) == 0:
                for c in indC:
                    guard.set_mark(pag, b, c, ARROW)
                    guard.set_mark(pag, c, b, TAIL_MARK)
            else:
                for c in indC:
                    if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                        guard.set_mark(pag, b, c, ARROW)
                        guard.set_mark(pag, c, b, TAIL_MARK)

    #R2
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] == CIRCLE_MARK and pag[j][i] != NULL:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for a, c in ind:
        indB = [i for i in range(len(pag)) if
                (pag[a][i] == ARROW and pag[i][a] == TAIL_MARK and pag[c][i] != NULL and pag[i][c] == ARROW) or
                (pag[a][i] == ARROW and pag[i][a] != NULL and pag[c][i] == TAIL_MARK and pag[i][c] == ARROW)]
        if len(indB) > 0:
            guard.set_mark(pag, a, c, ARROW)

    #R3
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] != NULL and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for b, d in ind:
        indAC = [i for i in range(len(pag)) if
                 pag[b][i] != NULL and pag[i][b] == ARROW and
                 pag[i][d] == CIRCLE_MARK and pag[d][i] != NULL]
        if len(indAC) >= 2:
            if len(unfVect) == 0:
                counter = -1
                while counter < len(indAC) - 1 and pag[d][b] != ARROW:
                    counter += 1
                    ii = counter
                    while ii < len(indAC) - 1 and pag[d][b] != ARROW:
                        ii += 1
                        if pag[indAC[counter]][indAC[ii]] == NULL and pag[indAC[ii]][indAC[counter]] == NULL:
                            guard.set_mark(pag, d, b, ARROW)
            else:
                for a, c in combinations(indAC, 2):
                    if pag[a][c] == NULL and pag[c][a] == NULL and c != a:
                        if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                            guard.set_mark(pag, d, b, ARROW)

    return pag


def _apply_r5_r10(pag, p, unfVect=None, guard=None):
    if unfVect is None: unfVect = {}
    if guard is None: guard = _empty_guard()

    # R5
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] == CIRCLE_MARK and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    while len(ind) > 0:
        a, b = ind[0]
        ind = ind[1:]
        indC = [i for i in range(len(pag)) if
                pag[a][i] == CIRCLE_MARK and pag[i][a] == CIRCLE_MARK and pag[b][i] == NULL and pag[i][b] == NULL and i != b]
        indD = [i for i in range(len(pag)) if
                pag[b][i] == CIRCLE_MARK and pag[i][b] == CIRCLE_MARK and pag[a][i] == NULL and pag[i][a] == NULL and i != a]
        if len(indD) > 0 and len(indC) > 0:
            counterC = -1
            while counterC < len(indC) - 1 and pag[a][b] == CIRCLE_MARK:
                counterC += 1
                c = indC[counterC]
                counterD = -1
                while counterD < len(indD) - 1 and pag[a][b] == CIRCLE_MARK:
                    counterD += 1
                    d = indD[counterD]
                    if pag[c][d] == CIRCLE_MARK and pag[d][c] == CIRCLE_MARK:
                        if len(unfVect) == 0:
                            guard.set_mark(pag, a, b, TAIL_MARK)
                            guard.set_mark(pag, b, a, TAIL_MARK)
                            guard.set_mark(pag, a, c, TAIL_MARK)
                            guard.set_mark(pag, c, a, TAIL_MARK)
                            guard.set_mark(pag, c, d, TAIL_MARK)
                            guard.set_mark(pag, d, c, TAIL_MARK)
                            guard.set_mark(pag, d, b, TAIL_MARK)
                            guard.set_mark(pag, b, d, TAIL_MARK)
                        else:
                            path2check = [a, c, d, b]
                            if faith_check(path2check, unfVect, p):
                                guard.set_mark(pag, a, b, TAIL_MARK)
                                guard.set_mark(pag, b, a, TAIL_MARK)
                                guard.set_mark(pag, a, c, TAIL_MARK)
                                guard.set_mark(pag, c, a, TAIL_MARK)
                                guard.set_mark(pag, c, d, TAIL_MARK)
                                guard.set_mark(pag, d, c, TAIL_MARK)
                                guard.set_mark(pag, d, b, TAIL_MARK)
                                guard.set_mark(pag, b, d, TAIL_MARK)
                    else:
                        ucp = minUncovCircPath(p, pag=pag, path=(a, c, d, b), unfVect=unfVect)
                        if len(ucp) > 1:
                            guard.set_mark(pag, ucp[0], ucp[-1], TAIL_MARK)
                            guard.set_mark(pag, ucp[-1], ucp[0], TAIL_MARK)
                            for j in range(len(ucp) - 1):
                                guard.set_mark(pag, ucp[j], ucp[j + 1], TAIL_MARK)
                                guard.set_mark(pag, ucp[j + 1], ucp[j], TAIL_MARK)

    # R6
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] != NULL and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for b, c in ind:
        if len([i for i in range(len(pag)) if pag[b][i] == TAIL_MARK and pag[i][b] == TAIL_MARK]) > 0:
            guard.set_mark(pag, c, b, TAIL_MARK)

    # R7
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] != NULL and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for b, c in ind:
        indA = [i for i in range(len(pag)) if
                pag[b][i] == TAIL_MARK and pag[i][b] == CIRCLE_MARK and pag[c][i] == NULL and pag[i][c] == NULL and i != c]
        if len(indA) > 0:
            if len(unfVect) == 0:
                guard.set_mark(pag, c, b, TAIL_MARK)
            else:
                for a in indA:
                    if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                        guard.set_mark(pag, c, b, TAIL_MARK)

    # R8
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] == ARROW and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for a, c in ind:
        indB = [i for i in range(len(pag)) if
                pag[i][a] == TAIL_MARK and (pag[a][i] == ARROW or pag[a][i] == CIRCLE_MARK) and pag[c][i] == TAIL_MARK and pag[i][c] == ARROW]
        if len(indB) > 0:
            guard.set_mark(pag, c, a, TAIL_MARK)

    # R9
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] == ARROW and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    while len(ind) > 0:
        a, c = ind[0]
        ind = ind[1:]
        indB = [i for i in range(len(pag)) if
                (pag[a][i] == ARROW or pag[a][i] == CIRCLE_MARK) and
                (pag[i][a] == CIRCLE_MARK or pag[i][a] == TAIL_MARK) and
                (pag[c][i] == NULL and pag[i][c] == NULL) and
                i != c]
        while len(indB) > 0 and pag[c][a] == CIRCLE_MARK:
            b = indB[0]
            indB = indB[1:]
            upd = minUncovPdPath(p, pag, a, b, c, unfVect=unfVect)
            if len(upd) > 1:
                guard.set_mark(pag, c, a, TAIL_MARK)

    # R10
    ind = []
    for i in range(len(pag)):
        for j in range(len(pag[i])):
            if pag[i][j] == ARROW and pag[j][i] == CIRCLE_MARK:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    while len(ind) > 0:
        a, c = ind[0]
        ind = ind[1:]
        indB = [i for i in range(p) if pag[c][i] == TAIL_MARK and pag[i][c] == ARROW]
        if len(indB) >= 2:
            counterB = -1
            while counterB < len(indB) - 1 and pag[c][a] == CIRCLE_MARK:
                counterB += 1
                b = indB[counterB]
                indD = [i for i in indB if i != b]
                counterD = -1
                while counterD < len(indD) - 1 and pag[c][a] == CIRCLE_MARK:
                    counterD += 1
                    d = indD[counterD]
                    if (
                            (pag[a][b] == CIRCLE_MARK or pag[a][b] == ARROW) and
                            (pag[b][a] == CIRCLE_MARK or pag[b][a] == TAIL_MARK) and
                            (pag[a][d] == CIRCLE_MARK or pag[a][d] == ARROW) and
                            (pag[d][a] == CIRCLE_MARK or pag[d][a] == TAIL_MARK) and
                            pag[d][b] == NULL and pag[b][d] == NULL
                    ):
                        if len(unfVect) == 0:
                            guard.set_mark(pag, c, a, TAIL_MARK)
                        else:
                            if (b, a, d) not in unfVect and (d, a, b) not in unfVect:
                                guard.set_mark(pag, c, a, TAIL_MARK)
                    else:
                        indX = [i for i in range(p) if
                                (pag[a][i] == CIRCLE_MARK or pag[a][i] == ARROW) and
                                (pag[i][a] == CIRCLE_MARK or pag[i][a] == TAIL_MARK) and
                                i != c]
                        if len(indX) >= 2:
                            counterX1 = -1
                            while counterX1 < len(indX) - 1 and pag[c][a] == CIRCLE_MARK:
                                counterX1 += 1
                                first_pos = indX[counterX1]
                                indX2 = [i for i in indX if i != first_pos]
                                counterX2 = -1
                                while counterX2 < len(indX2) - 1 and pag[c][a] == CIRCLE_MARK:
                                    counterX2 += 1
                                    sec_pos = indX2[counterX2]
                                    t1 = minUncovPdPath(p, pag, a, first_pos, b, unfVect=unfVect)
                                    if len(t1) > 1:
                                        t2 = minUncovPdPath(p, pag, a, sec_pos, d, unfVect=unfVect)
                                        if len(t2) > 1 and first_pos != sec_pos and pag[first_pos][sec_pos] == NULL:
                                            if len(unfVect) == 0:
                                                guard.set_mark(pag, c, a, TAIL_MARK)
                                            elif (first_pos, a, sec_pos) not in unfVect and (sec_pos, a, first_pos) not in unfVect:
                                                guard.set_mark(pag, c, a, TAIL_MARK)
    return pag


def _is_adj(pag, u, v):
    return not (pag[u, v] == NULL and pag[v, u] == NULL)


def _min_discriminating_path(pag, a, b, c):
    """Find minimal discriminating path [Xi, ..., a, b, c] with Xi non-adjacent to c."""
    p = pag.shape[0]
    visited = np.zeros(p, dtype=bool)
    visited[[a, b, c]] = True

    indD = [
        d for d in range(p)
        if (not visited[d]) and _is_adj(pag, a, d) and pag[d, a] == ARROW
    ]
    queue = deque([ [a, d] for d in indD ])

    while queue:
        path = queue.popleft()
        d = path[-1]

        if not _is_adj(pag, c, d):
            return list(reversed(path)) + [b, c]

        if not visited[d]:
            visited[d] = True

        pred = path[-2]

        if pag[d, c] == ARROW and pag[c, d] == TAIL_MARK and pag[pred, d] == ARROW:
            for r in range(p):
                if visited[r] or r in path:
                    continue
                if _is_adj(pag, d, r) and pag[r, d] == ARROW:
                    queue.append(path + [r])
    return None

def _check_edges_on_disc_path(cg: CausalGraph, alpha, path, unfVect=None, max_k=None, n_jobs=1,
                              protected_edges=()):
    if unfVect is None:
        unfVect = set()

    p = cg.G.graph.shape[0]

    Xi = path[0]
    Xk = path[-1]
    sep_tot = _sep_union(cg, Xi, Xk)

    for t in range(len(path) - 1):
        Xr, Xq = path[t], path[t + 1]

        base = set(sep_tot)
        base.discard(Xr)
        base.discard(Xq)
        base = list(base)

        ell = -1
        max_ell = len(base) if max_k is None else min(len(base), int(max_k))
        while True:
            ell += 1
            if ell > max_ell:
                break

            cond_sets = list(combinations(base, ell))
            pvals = _ci_batch(cg, Xr, Xq, cond_sets, n_jobs=n_jobs)
            for Y, pval in zip(cond_sets, pvals):
                if pval >= alpha:
                    pag_before = cg.G.graph.copy()
                    # Инвариант цикла: от m не зависит, считаем один раз.
                    tri_nodes = [
                        m for m in range(p)
                        if m != Xr and m != Xq
                        and _is_adj(pag_before, Xr, m)
                        and _is_adj(pag_before, Xq, m)
                    ] if _is_adj(pag_before, Xr, Xq) else []

                    # Обязательное ребро должно пережить весь поиск, а не
                    # только скелет: та же проверка стоит в
                    # _orient_v_structures_rfci.
                    if frozenset((int(Xr), int(Xq))) in _as_pairs(protected_edges):
                        continue

                    cg.sepset[Xr][Xq] = list(Y)
                    cg.sepset[Xq][Xr] = list(Y)

                    cg.remove_edge(cg.nodes[Xr], cg.nodes[Xq])

                    a = min(Xr, Xq)
                    c = max(Xr, Xq)
                    M = [(a, m, c) for m in tri_nodes if m != a and m != c]

                    _orient_v_structures_rfci(cg, alpha, M, unfVect,
                                              protected_edges=protected_edges)

                    return True

            pag_chk = cg.G.graph
            if not _is_adj(pag_chk, Xr, Xq):
                break

    return False


def _apply_discriminating_path_rules(cg, pag, p, alpha, unfVect=None, n_jobs=1,
                                     max_k=None, guard=None, protected_edges=()):
    if unfVect is None:
        unfVect = set()
    if guard is None:
        guard = _empty_guard()

    changed = True
    while changed:
        changed = False
        pag = cg.G.graph

        candidates = [
            (b, c) for b in range(p) for c in range(p)
            if b != c and _is_adj(pag, b, c) and pag[c, b] == CIRCLE_MARK
        ]
        for b, c in candidates:
            if not _is_adj(pag, b, c) or pag[c, b] != CIRCLE_MARK:
                continue

            indA = [
                a for a in range(p)
                if a != b and a != c
                and _is_adj(pag, a, c) and pag[a, c] == ARROW and pag[c, a] == TAIL_MARK
                and _is_adj(pag, a, b) and pag[b, a] == ARROW and pag[a, b] != NULL
            ]
            for a in indA:
                if pag[c, b] != CIRCLE_MARK:
                    continue

                md_path = _min_discriminating_path(pag, a, b, c)
                if md_path is None:
                    continue

                deleted = _check_edges_on_disc_path(
                    cg, alpha, md_path, unfVect=unfVect, max_k=max_k, n_jobs=n_jobs,
                    protected_edges=protected_edges
                )
                pag = cg.G.graph
                if deleted:
                    changed = True
                    break

                Xi, Xk = md_path[0], md_path[-1]
                sep = _sep_union(cg, Xi, Xk)
                if b in sep:
                    guard.set_mark(pag, b, c, ARROW)
                    guard.set_mark(pag, c, b, TAIL_MARK)
                else:
                    guard.set_mark(pag, a, b, ARROW)
                    guard.set_mark(pag, b, a, ARROW)
                    guard.set_mark(pag, b, c, ARROW)
                    guard.set_mark(pag, c, b, ARROW)
                cg.G.graph = pag
                changed = True
                break
            if changed:
                break
    return cg.G.graph


def _orient_edges_rfci(cg, p, alpha, unfVect=None, n_jobs=1, max_k=None,
                       background_knowledge=None, node_names=None, guard=None,
                       protected_edges=()):
    if guard is None:
        guard = _empty_guard()
    pag = copy.deepcopy(cg.G.graph)
    prev_pag = np.zeros_like(pag)
    while not np.array_equal(pag, prev_pag):
        prev_pag = copy.deepcopy(pag)
        pag = _apply_r1_r3(pag, unfVect, guard=guard)
        cg.G.graph = pag

        # pag = np.array([
        #     [ 0, -1,  1,  0,  0,  2,  0,  0,  0,  0],
        #     [ 1,  0,  2,  2,  0,  0,  0,  0,  0,  0],
        #     [-1,  1,  0,  2,  0,  0,  0,  0,  0,  0],
        #     [ 0,  1,  1,  0,  2,  0,  0,  0,  0,  0],
        #     [ 0,  0,  0,  1,  0,  2,  0,  0,  0,  0],
        #     [ 2,  0,  0,  0,  1,  0,  2,  0,  0,  0],
        #     [ 0,  0,  0,  0,  0,  1,  0,  2,  0,  0],
        #     [ 0,  0,  0,  0,  0,  0,  1,  0,  2,  0],
        #     [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  2],
        #     [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
        # ], dtype=int)
        # cg.G.graph = pag

        pag = _apply_discriminating_path_rules(cg, pag, p, alpha, unfVect=unfVect, n_jobs=n_jobs,
                                               max_k=max_k, guard=guard,
                                               protected_edges=protected_edges)
        pag = _apply_r5_r10(pag, p, unfVect, guard=guard)
        cg.G.graph = pag
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names if node_names is not None else [node.name for node in cg.nodes],
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_orientation_rule_pass",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
            previous=prev_pag,
        )
        pag = cg.G.graph

    cg.G.graph = pag
    # print("After R5-R10:")
    # print(cg.G.graph)


def rfci_stable(cg, data, alpha=0.05, indep_test='fisherz', max_k=None, n_jobs=1, verbose=False,
                show_progress=False, use_ci_cache=True, use_shared_ci_cache=False,
                shared_ci_cache_capacity=1_000_003, skeleton_backend=None,
                orient_backend=None, ci_backend=None, process_min_vars=150,
                background_knowledge=None) -> np.ndarray:
    """test"""
    total_t0 = time.perf_counter()
    n, p = data.shape
    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    skeleton_backend_requested, skeleton_backend_eff = _resolve_rfci_backend(
        skeleton_backend,
        p,
        process_min_vars,
    )
    orient_backend_requested, orient_backend_eff = _resolve_rfci_backend(
        orient_backend if orient_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    use_process_skeleton = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff == "processes"
    )
    use_process_orient = (
        n_jobs_eff != 1
        and orient_backend_eff == "processes"
    )
    cit = CIT(data, method=indep_test)
    ci_stats = {
        "total": 0,
        "skeleton": 0,
        "rfci_orient": 0,
        "skeleton_backend": "processes" if use_process_skeleton else "threads",
        "orient_backend": "processes" if use_process_orient else "threads",
        "skeleton_backend_requested": skeleton_backend_requested,
        "orient_backend_requested": orient_backend_requested,
        "process_min_vars": int(process_min_vars),
        "n_vars": int(p),
        "n_jobs": int(n_jobs_eff),
    }
    use_shared_cache = bool(
        use_ci_cache and use_shared_ci_cache and use_process_orient
    )
    use_local_cache = bool(use_ci_cache and not use_shared_cache)
    if use_local_cache or use_shared_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = {} if use_local_cache else None
    ci_inflight = {}
    stats_lock = threading.Lock()
    shared_ci_cache = (
        _SharedRFCICache(capacity=shared_ci_cache_capacity, create=True)
        if use_shared_cache else None
    )

    def counted_cit(i, j, S):
        key = _ci_cache_key(i, j, S)
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1

        if shared_ci_cache is not None:
            cached = shared_ci_cache.get(key)
            if cached is not None:
                with stats_lock:
                    ci_stats["ci_cache_hits"] += 1
                return cached

            with stats_lock:
                ci_stats["ci_cache_misses"] += 1
            pval = cit(i, j, list(key[2]))
            shared_ci_cache.set(key, pval)
            return pval

        if ci_cache is None:
            return cit(i, j, S)

        while True:
            with stats_lock:
                cached = ci_cache.get(key)
                if cached is not None:
                    ci_stats["ci_cache_hits"] += 1
                    return cached
                event = ci_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    ci_inflight[key] = event
                    ci_stats["ci_cache_misses"] += 1
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            pval = cit(i, j, list(key[2]))
            with stats_lock:
                ci_cache[key] = pval
            return pval
        finally:
            with stats_lock:
                ci_inflight.pop(key, None)
                event.set()

    try:
        """"""
        node_names = [node.name for node in cg.nodes]
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.PRE_SEARCH,
            checkpoint="before_skeleton",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )
        protected_edges = ()
        if background_knowledge is not None:
            protected_edges = background_knowledge.required_pairs(
                node_names, phase=BKPhase.PRE_SEARCH
            )
        cg.set_ind_test(cit if use_process_skeleton else counted_cit)
        skeleton_t0 = time.perf_counter()
        skeleton = SkeletonDiscovery(
            cg,
            alpha,
            n_jobs=n_jobs_eff,
            verbose=verbose,
            show_progress=show_progress,
            parallel_backend="processes" if use_process_skeleton else "threads",
            process_ci_info=(
                {
                    "method": indep_test,
                    "corr_matrix": cit.corr_matrix,
                    "n_samples": cit.n_samples,
                }
                if use_process_skeleton
                else None
            ),
            protected_edges=protected_edges,
        )
        skeleton.run_fci()
        ci_stats["time_skeleton_sec"] = time.perf_counter() - skeleton_t0
        if use_process_skeleton:
            ci_stats["total"] += int(skeleton.ci_tests_performed)
            ci_stats["skeleton"] += int(skeleton.ci_tests_performed)
            if "ci_cache_misses" in ci_stats:
                ci_stats["ci_cache_misses"] += int(skeleton.ci_tests_performed)
            cg.set_ind_test(counted_cit)

        cg._parallel_backend = "processes" if use_process_orient else "threads"
        if n_jobs_eff != 1:
            if use_process_orient:
                cg._parallel_executor = Parallel(
                    n_jobs=n_jobs_eff,
                    backend="loky",
                    max_nbytes="10K",
                )
                cg._parallel_executor.__enter__()
            else:
                cg._parallel_executor = ThreadPoolExecutor(max_workers=n_jobs_eff)
        if use_process_orient:
            cg._parallel_cit = cit
            cg._parallel_shared_cache = shared_ci_cache
            cg._parallel_ci_stats = ci_stats
            cg._parallel_phase = phase
        # Explicit o-o initialization right after skeleton.
        cg.G.graph = to_circle_adjacency(cg.G.graph, MatrixEncoding.STANDARD)
        # print(cg.G.graph)

        # print("Starting RFCI orientation rules...")
        phase["name"] = "rfci_orient"
        orient_t0 = time.perf_counter()
        # Правила пишут метки только через guard, поэтому BK остаётся
        # инвариантом всего прохода ориентации, а не применяется однократно в
        # чекпоинтах: R0–R10 успевают переписать конец required-ребра, и если
        # восстановление направления замыкает цикл, чекпоинт откатывает
        # транзакцию целиком и ограничение остаётся нарушенным.
        orientation_guard = (
            background_knowledge.orientation_guard(
                node_names, phase=BKPhase.BETWEEN_ORIENTATION,
                encoding=MatrixEncoding.STANDARD,
            )
            if background_knowledge is not None
            else _empty_guard()
        )

        # BK применяется до R0: правила должны стартовать с графа, который уже
        # уважает ограничения.
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="before_orientation",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )

        M = _unshielded_triples(cg)
        _orient_v_structures_rfci(cg, alpha=alpha, M=M, protected_edges=protected_edges,
                                  guard=orientation_guard)
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_colliders",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )
        # print(cg.G.graph)
        _orient_edges_rfci(
            cg,
            p,
            alpha,
            n_jobs=n_jobs_eff,
            max_k=max_k,
            background_knowledge=background_knowledge,
            node_names=node_names,
            guard=orientation_guard,
            protected_edges=protected_edges,
        )
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.POST_ORIENTATION,
            checkpoint="after_orientation",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )
        ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
        if ci_cache is not None:
            ci_stats["ci_cache_size"] = int(len(ci_cache))

        executor = getattr(cg, "_parallel_executor", None)
        if executor is not None:
            if isinstance(executor, ThreadPoolExecutor):
                executor.shutdown(wait=True)
            else:
                executor.__exit__(None, None, None)
            delattr(cg, "_parallel_executor")

        ci_stats["time_total_sec"] = time.perf_counter() - total_t0
        cg.ci_stats = ci_stats

        return cg.G.graph
    finally:
        executor = getattr(cg, "_parallel_executor", None)
        if executor is not None:
            if isinstance(executor, ThreadPoolExecutor):
                executor.shutdown(wait=True)
            else:
                executor.__exit__(None, None, None)
        for attr in (
            "_parallel_backend",
            "_parallel_executor",
            "_parallel_cit",
            "_parallel_shared_cache",
            "_parallel_ci_stats",
            "_parallel_phase",
        ):
            if hasattr(cg, attr):
                delattr(cg, attr)
        if shared_ci_cache is not None:
            shared_ci_cache.close()
            shared_ci_cache.unlink()


#для тестов локально (numpy и time уже импортированы в шапке модуля)

def make_collider_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)

    L1 = rng.normal(size=n)
    L2 = rng.normal(size=n)

    X = L1 + rng.normal(scale=0.2, size=n)
    Y = L1 + rng.normal(scale=0.2, size=n)
    Z = X + rng.normal(scale=0.2, size=n)
    W = Y + rng.normal(scale=0.2, size=n)

    T = Z + W + rng.normal(scale=0.2, size=n)
    U = T + L2 + rng.normal(scale=0.2, size=n)
    V = U + L2+ rng.normal(scale=0.2, size=n)

    return np.column_stack([X, Y, Z, W, T, U, V])




if __name__ == "__main__":
    cg = CausalGraph(no_of_var=7, node_names=['X', 'Y', 'Z', 'W', 'T', 'U', 'V'])
    rfci_stable(cg, make_collider_data(), 0.05, indep_test='fisherz', n_jobs=1, verbose=True)


    # X = make_small_scale_data(p_prime=30, n=1000, seed=42)['X_obs']
    # node_names = [f"X{i}" for i in range(X.shape[1])]
    # cg = CausalGraph(no_of_var=X.shape[1], node_names=node_names)

    # rfci_stable(cg, X, 0.01, indep_test='fisherz', n_jobs=4, verbose=True)

    # python -m whydra.algorithms.standalone.fci_plus_algo
