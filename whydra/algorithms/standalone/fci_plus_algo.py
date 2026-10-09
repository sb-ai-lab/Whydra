import copy
from itertools import combinations
import numpy as np
from collections import deque
import threading
from joblib import Parallel, delayed

from ..graph_core import CIT, CausalGraph
from ..graph_core.endpoints import Endpoint
from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint

from .skeleton import SkeletonDiscovery

#импорт функции для тестовой проверки работы кода
from .pag_rules import PAG_ARROW, PAG_CIRCLE, PAG_TAIL, udag2pag


# ============================================================
#  Локальные копии вспомогательных функций FCI.
#  Перенесены из прежнего fci_algo.py.
#  Держим их здесь, чтобы файл не зависел от fci_algo.py:
#  там лежит эталонная реализация FCI из ветки dev, без
#  background_knowledge / protected_edges / rule_stats.
# ============================================================

TAIL = Endpoint.TAIL.value     # -1
ARROW = Endpoint.ARROW.value   # 1
CIRCLE = Endpoint.CIRCLE.value # 2
NULL = Endpoint.NULL.value    # 0


def _ci_cache_key(i: int, j: int, S):
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


def _sepset_vars(seps):
    vars_set = set()
    for x in seps or []:
        if isinstance(x, tuple):
            vars_set.update(x)
        elif isinstance(x, (list, set, np.ndarray)):
            vars_set.update(x)
        else:
            vars_set.add(x)
    return vars_set


def _has_sepset(seps):
    return seps is not None


def _sepset_to_pair_dict(sepset, p: int):
    """
    Приводит cg.sepset к виду, который ждёт udag2pag: {(x, z): множество узлов}.

    ``cg.sepset[i][j]`` хранит СПИСОК всех разделяющих множеств, найденных для
    пары — [(1,2,3), (4,5), ...]. Раньше здесь бралось их объединение, и это
    ломало ориентацию коллайдеров: правило R0 проверяет ``y not in sepset[(x,z)]``,
    а в объединение y попадал, если встретился хотя бы в одном множестве.
    В результате коллайдеры не ориентировались и на их месте оставались кружки.

    Берём одно множество — наименьшее по размеру. Для правила R0 годится любое
    корректное разделяющее множество, а наименьшее совпадает с тем, что нашла
    фаза скелета первой (поиск идёт по возрастанию глубины), и не зависит от
    порядка вставки.
    """
    out = {}
    for i in range(p):
        for j in range(p):
            s = sepset[i][j]
            if not s:
                out[(i, j)] = set()
                continue
            candidates = [
                set(S) if hasattr(S, "__iter__") else {int(S)}
                for S in s
            ]
            out[(i, j)] = min(candidates, key=len) if candidates else set()
    return out


def _edge_exists(mat, i, j):
    return mat[i][j] != NULL or mat[j][i] != NULL


def _is_bidirected(mat, i, j):
    return mat[i][j] == ARROW and mat[j][i] == ARROW


def _naa_ancestors(x, mat):
    p = mat.shape[0]
    queue = deque([int(x)])
    seen = {int(x)}
    out = []

    while queue:
        t = queue.popleft()
        for j in range(p):
            if j in seen:
                continue
            if t == x:
                legal = mat[j][t] == ARROW and mat[t][j] not in (NULL, ARROW)
            else:
                legal = mat[t][j] not in (NULL, ARROW)
            if legal:
                seen.add(j)
                queue.append(j)
                out.append(j)

    return set(out)


def _get_pdseps(mat):
    p = mat.shape[0]
    naaa = [None for _ in range(p)]
    links = []

    for i in range(1, p):
        naaa_i = _naa_ancestors(i, mat)
        naaa[i] = naaa_i
        for j in range(i):
            if not _is_bidirected(mat, i, j):
                continue

            if naaa[j] is None:
                naaa[j] = _naa_ancestors(j, mat)

            u_nodes = {
                k for k in range(p)
                if k not in (i, j) and _is_bidirected(mat, i, k)
            }
            v_nodes = {
                k for k in range(p)
                if k not in (i, j) and _is_bidirected(mat, j, k)
            }
            # TODO(fci-plus-pdsep): фильтр Lemma 4 обнуляет всех кандидатов, и
            # фаза _dsep не выполняется ни разу — Possible-D-SEP в FCI+ фактически
            # выключен, лишние рёбра остаются в PAG.
            #
            # Замер (10 графов со скрытым конфаундером, p 8-13, n=600, fisherz,
            # alpha=0.05): 94 двунаправленных ребра против 24 направленных,
            # _naa_ancestors непуст у 20 узлов из 113, кандидатов БЕЗ фильтра 65,
            # С фильтром 0. Очередь _dsep_search пуста всегда.
            #
            # Причина в первом шаге _naa_ancestors: он требует не-стрелку на
            # дальнем конце, а после AugmentGraph почти все рёбра <->. Условие
            # `mat[t][j] not in (NULL, ARROW)` уже допускает `o->`; отсекаются
            # именно двунаправленные звенья.
            #
            # Просто снять фильтр НЕЛЬЗЯ: проверено на эталоне, согласие с
            # causal-learn не растёт, а на части выборок падает (сид 5: 3/25
            # против 3/25; сид 9: 2/25 против 1/25, то есть хуже). Это заменило
            # бы «PDSEP молча не работает» на «работает с неверным фильтром» —
            # вторую поломку снаружи не видно.
            #
            # Нужна дефиниция NAA-предка из Claassen et al. для аугментированного
            # графа. До этого оставляем как есть, поведение задокументировано.
            u_nodes &= naaa[j]   # U (bidirected with i) must have NAA path to j (Lemma 4)
            v_nodes &= naaa_i    # V (bidirected with j) must have NAA path to i (Lemma 4)

            found = any(
                u != v and not _edge_exists(mat, u, v)
                for u in u_nodes
                for v in v_nodes
            )
            if found:
                links.append((i, j))

    return deque(links)


def _ci_batch(cg, tasks, n_jobs):
    tasks = list(tasks)
    if n_jobs in (None, 0, 1):
        return [(x, y, S, cg.ci_test(x, y, S)) for x, y, S in tasks]
    pvals = Parallel(n_jobs=int(n_jobs), prefer="threads", require="sharedmem")(
        delayed(cg.ci_test)(x, y, S) for x, y, S in tasks
    )
    return [(x, y, S, pval) for (x, y, S), pval in zip(tasks, pvals)]


# FIX: iterate over NON-ADJACENT pairs with stored sepsets (not adjacent pairs/edges).
# For each such (i,j) with Sep(i,j)=Z, test i _|/|_ j | Z ∪ {W} for W adjacent to {i,j}∪Z.
# If dependent (p<alpha) → W ∉ An({i,j}∪Z) → add arrowhead AT W from every node in {i,j}∪Z
# adjacent to W: pag[node][W] = ARROW (convention: pag[a][b] = mark at b's end of edge a-b).
def AugmentGraph(cg: CausalGraph, p: int, alpha: float, n_jobs: int = 1):
    pag = copy.deepcopy(cg.G.graph)
    sepset = cg.sepset

    for i in range(p):
        for j in range(i + 1, p):
            sep_entry = sepset[i][j]
            # Only process non-adjacent pairs that have a recorded separating set
            if not _has_sepset(sep_entry):
                continue
            sep = _sepset_vars(sep_entry)

            # Candidate W nodes: adjacent to {i,j} ∪ sep, excluding {i,j} ∪ sep themselves
            adjacent = set(np.where(pag[i, :] != 0)[0]) | set(np.where(pag[j, :] != 0)[0])
            for sep_node in sep:
                adjacent |= set(np.where(pag[sep_node, :] != 0)[0])

            del_nodes = {i, j} | sep
            adjacent -= del_nodes

            tasks = []
            task_meta = []
            for adj in sorted(adjacent):
                S = tuple(sorted(sep | {adj}))
                tasks.append((i, j, S))
                task_meta.append((tuple(sorted(del_nodes)), adj))

            pvals = _ci_batch(cg, tasks, n_jobs)
            for (_, _, _, p_value), (del_nodes_t, adj) in zip(pvals, task_meta):
                if p_value < alpha:
                    for node in del_nodes_t:
                        # pag[node][adj] = ARROW: arrowhead AT adj on edge node-adj
                        if pag[node][adj] != NULL and pag[node][adj] != ARROW:
                            pag[node][adj] = ARROW

    cg.G.graph = pag
    return pag


# FIX: use _get_pdseps (Lemma 4 bidirected pattern) instead of returning all edges.
def _dsep_search(cg: CausalGraph, p: int) -> deque:
    return _get_pdseps(cg.G.graph.copy())


def _get_hie(cg, S):
    H = set(S)
    old_H = None

    while old_H != H:
        old_H = H.copy()
        H_list = list(H)
        for a in H_list:
            for b in H_list:
                if a == b:
                    continue
                sep = cg.sepset[a][b]
                sep_vars = _sepset_vars(sep)
                if sep_vars:
                    H |= set(sep_vars)

    return H


# FIX: find smallest valid subset (minimum cardinality), same as parallel version.
def _min_dsep(cg: CausalGraph, S, i, j, alpha=0.05):
    # cg.sepset хранит СПИСОК разделяющих множеств (кортежей) — соглашение
    # SkeletonDiscovery._append_sepset, на него рассчитан _sepset_to_pair_dict.
    # Плоский список [3, 5] он разбирал как два множества и брал {3}, теряя узел.
    def _store(nodes):
        cg.sepset[i][j] = cg.sepset[j][i] = [tuple(nodes)]
        return list(nodes)

    S = tuple(sorted(set(S)))
    if not S:
        return _store(())

    for size in range(1, len(S) + 1):
        for subset in combinations(S, size):
            if cg.ci_test(i, j, list(subset)) >= alpha:
                return _store(subset)

    return _store(S)


# FIX: loop starts from n=1, m=1 per Algorithm 2 lines 8-9 (not n=0, m=0).
def _dsep(cg, alpha, x, y, base_x, base_y, k, p, n_jobs=1, protected_edges=()):
    protected_pairs = {frozenset((int(a), int(b))) for a, b in protected_edges}
    if frozenset((int(x), int(y))) in protected_pairs:
        return False, deque()
    max_n = min(k, len(base_x))
    max_m = min(k, len(base_y))
    for n in range(1, max_n + 1):
        for m in range(1, max_m + 1):
            for z_x in combinations(base_x, n):
                for z_y in combinations(base_y, m):
                    Z = _get_hie(cg, {x, y} | set(z_x) | set(z_y)) - {x, y}
                    p_value = cg.ci_test(x, y, list(Z))
                    if p_value >= alpha:
                        Z = _min_dsep(cg, Z, x, y, alpha=alpha)
                        # _min_dsep уже записал множество в нужном формате.
                        # Found a valid D-SEP set: remove the edge X-Y to ensure progress.
                        cg.G.graph[x][y] = NULL
                        cg.G.graph[y][x] = NULL
                        AugmentGraph(cg, p, alpha, n_jobs=n_jobs)
                        return True, _dsep_search(cg, p)

    return False, deque()


def fci_plus_stable(cg: CausalGraph, data: np.ndarray, alpha: float, indep_test = 'fisherz',
                     k=None, n_jobs: int = 1, verbose: bool = False, use_ci_cache=True,
                     background_knowledge=None, show_progress: bool = False) -> np.ndarray:

    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    cit = CIT(data, method=indep_test)
    ci_stats = {"total": 0, "skeleton": 0, "augment": 0, "dsep": 0}
    use_cache = bool(use_ci_cache)
    if use_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = {} if use_cache else None
    stats_lock = threading.Lock()

    def counted_cit(i, j, S):
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1
        if ci_cache is None:
            return cit(i, j, S)

        key = _ci_cache_key(i, j, S)
        with stats_lock:
            cached = ci_cache.get(key)
            if cached is not None:
                ci_stats["ci_cache_hits"] += 1
                return cached
            ci_stats["ci_cache_misses"] += 1

        pval = cit(i, j, list(key[2]))
        with stats_lock:
            ci_cache[key] = pval
        return pval

    cg.set_ind_test(counted_cit)
    n, p = data.shape
    if not k:
        k = p

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

    graphDict = SkeletonDiscovery(
        cg, alpha, n_jobs=n_jobs_eff, verbose=verbose, show_progress=show_progress,
        protected_edges=protected_edges,
    ).run_fci()

    phase["name"] = "augment"
    AugmentGraph(cg, p, alpha, n_jobs=n_jobs_eff)
    dsep_pos = _dsep_search(cg, p)

    phase["name"] = "dsep"
    while len(dsep_pos) > 0:
        x, y = dsep_pos.popleft()
        base_x = set(cg.neighbors(x)) - {y}
        base_y = set(cg.neighbors(y)) - {x}
        found_dsep, updated_dsep_pos = _dsep(
            cg, alpha, x, y, base_x, base_y, k, p, n_jobs=n_jobs_eff,
            protected_edges=protected_edges,
        )
        if found_dsep:
            dsep_pos = updated_dsep_pos

    pag = cg.G.graph.copy()

    # Перекодировки нет: TAIL/ARROW/CIRCLE здесь и PAG_* в pag_rules — одни и те
    # же Endpoint.*.value, прежний двойной цикл отображал значения сами в себя.

    sepset_pair = _sepset_to_pair_dict(cg.sepset, p)
    pag = udag2pag(
        pag,
        sepset_pair,
        p,
        orientCollider=True,
        background_knowledge=background_knowledge,
        node_names=[node.name for node in cg.nodes],
        # CFCI/FCI+ не ориентируют коллайдер, если для пары не записано
        # ни одного разделяющего множества — сохраняем прежнее поведение.
        require_recorded_sepset=True,
    )

    # Перекодировка не нужна: udag2pag и FCI+ работают в одной
    # кодировке causal-learn.
    cg.G.graph = pag
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = int(len(ci_cache))
    cg.ci_stats = ci_stats
    return pag


if __name__ == "__main__":
    # Импорты только для демо-скрипта: на уровне модуля они тянули
    # rfci_algo и fci_algo_v2 в каждый импорт библиотеки.
    from .rfci_algo import make_collider_data, rfci_stable
    from .fci_algo_v2 import fci_stable_v2
    df = make_collider_data()
    cg = CausalGraph(no_of_var=7, node_names=['X', 'Y', 'Z', 'W', 'T', 'U', 'V'])
    pag1 = fci_plus_stable(cg, df, 0.05, indep_test='fisherz', n_jobs=1, verbose=True)

    cg_rfci = CausalGraph(no_of_var=7, node_names=['X', 'Y', 'Z', 'W', 'T', 'U', 'V'])
    pag2 = rfci_stable(cg_rfci, df, 0.05, indep_test='fisherz', n_jobs=1, verbose=True)

    cg = CausalGraph(no_of_var=7, node_names=['X','Y','Z','W','T','U','V'])
    pag3 = fci_stable_v2(cg, df, alpha=0.05, indep_test='fisherz', n_jobs=1, verbose=True)
