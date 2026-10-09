from itertools import combinations

import threading

import numpy as np

from ..graph_core import CIT
from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from .skeleton import SkeletonDiscovery
from .pag_init import to_circle_adjacency
from .pag_rules import PAG_ARROW, PAG_CIRCLE, powerset, qreach, udag2pag


# ============================================================
#  Локальные копии вспомогательных функций FCI.
#  Перенесены из прежнего fci_algo.py.
#  Держим их здесь, чтобы файл не зависел от fci_algo.py:
#  там лежит эталонная реализация FCI из ветки dev, без
#  background_knowledge / protected_edges / rule_stats.
# ============================================================

def checkTriple(a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, suffStat, alpha, indepTest, maj_rule=True):
    nr_indep = 0
    temp = []

    if len(nbrsA) > 0:
        for s in powerset(nbrsA):
            pval = indepTest(suffStat, a, c, list(s))
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    if len(nbrsC) > 0:
        for s in powerset(nbrsC):
            pval = indepTest(suffStat, a, c, list(s))
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    newsepsetA = set(sepsetA) if sepsetA is not None else set()
    newsepsetC = set(sepsetC) if sepsetC is not None else set()

    if len(temp) == 0:
        temp.append(False)

    res = 3
    if maj_rule:
        if sum(temp) / len(temp) < .5:
            res = 1
            if b in newsepsetA: newsepsetA.remove(b)
            if b in newsepsetC: newsepsetC.remove(b)
        elif sum(temp) / len(temp) > .5:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)
        else:
            pass
    else:
        if sum(temp) / len(temp) == 0:
            res = 1
            if b in newsepsetA: newsepsetA.remove(b)
            if b in newsepsetC: newsepsetC.remove(b)
        elif sum(temp) / len(temp) == 1:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)
        else:
            pass

    return res, {'sepsetA': newsepsetA, 'sepsetC': newsepsetC}


def pc_cons_intern(graphDict, suffstat, alpha, indepTest, version_unf=(None, None), maj_rule=True,
                   verbose=False):
    sk = graphDict['sk']
    p = sk.shape[0]

    if np.any(sk):
        # Сортировка индексов как в pcalg для воспроизводимости
        ind = np.transpose(np.where(sk == PAG_CIRCLE))
        ind = ind[ind[:, 1].argsort()]

        tripleMatrix = []
        for a, b in ind:
            for c in range(p):
                if a < c and sk[a, c] == 0 and sk[b, c] == PAG_CIRCLE:
                    tripleMatrix.append((a, b, c))

        for a, b, c in tripleMatrix:
            nbrsA = np.where(sk[:, a] == PAG_CIRCLE)[0]
            nbrsC = np.where(sk[:, c] == PAG_CIRCLE)[0]

            res, r_abc = checkTriple(a, b, c, nbrsA, nbrsC, graphDict['sepset'][(a, c)],
                                     graphDict['sepset'][(c, a)],
                                     suffstat, alpha, indepTest, maj_rule=maj_rule)
            if res == 3:
                if 'unfTriples' in graphDict.keys():
                    graphDict['unfTriples'].add((a, b, c))
                else:
                    graphDict['unfTriples'] = {(a, b, c)}

            graphDict['sepset'][(a, c)] = r_abc['sepsetA']
            graphDict['sepset'][(c, a)] = r_abc['sepsetC']

    return graphDict


def pdsep(skel, suffStat, indepTest, p, sepSet, alpha, pMax, m_max=float('inf'), pdsep_max=float('inf'), unfVect=None,
          protected_edges=()):
    G = skel['sk'].astype(int)
    protected_edges = {
        frozenset((int(x), int(y))) for x, y in protected_edges
    }
    n_edgetest = [0 for i in range(1000)]
    ord = 0
    allPdsep_tmp = [set() for i in range(p)]
    amat = np.copy(G)
    pdsep_pairs_scanned = 0
    pdsep_pairs_with_diffset = 0
    pdsep_ci_tests = 0

    ind = []
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == PAG_CIRCLE:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))

    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepSet[(x, z)] or y in sepSet[(z, x)]):
                if len(unfVect) == 0:
                    amat[x][y] = amat[z][y] = PAG_ARROW
                else:
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = PAG_ARROW

    allPdsep = [qreach(x, amat) for x in range(p)]
    allPdsep_tmp = [[] for i in range(p)]

    for x in range(p):
        an0 = [True if amat[x][i] != 0 else False for i in range(len(amat))]
        tf1 = [i for i in allPdsep[x] if i != x]
        adj_x = [i for i in range(len(an0)) if an0[i] == True]

        for y in adj_x:
            if frozenset((int(x), int(y))) in protected_edges:
                continue
            pdsep_pairs_scanned += 1
            tf = [i for i in tf1 if i != y]
            diff_set = [i for i in tf if i not in adj_x]
            allPdsep_tmp[x] = tf + [y]

            if len(diff_set) > 0:
                pdsep_pairs_with_diffset += 1
                done = False
                ord = 0
                while not done and ord < min(len(tf), m_max):
                    ord += 1
                    # Синглтоны тестируются, как в pdsep из fci_algo_paralel_cash
                    # (его берёт параллельный CFCI) и в pdsep_v2 у FCI. Прежняя
                    # нижняя граница |S|>=2 делала ветку ord == 1 недостижимой и
                    # разводила две обёртки CFCI на одних данных.
                    if ord == 1:
                        for S in diff_set:
                            pdsep_ci_tests += 1
                            pval = indepTest(suffStat, x, y, [S])
                            n_edgetest[ord + 1] += 1
                            if pval > pMax[x, y]: pMax[x, y] = pval
                            if pval >= alpha:
                                amat[x][y] = amat[y][x] = 0
                                sepSet[(x, y)] = sepSet[(y, x)] = {S}
                                done = True
                                break
                    else:
                        for S in combinations(tf, ord):
                            if not set(S).issubset(adj_x):
                                pdsep_ci_tests += 1
                                pval = indepTest(suffStat, x, y, list(S))
                                n_edgetest[ord + 1] += 1
                                if pval > pMax[x, y]: pMax[x, y] = pval
                                if pval >= alpha:
                                    amat[x][y] = amat[y][x] = 0
                                    sepSet[(x, y)] = sepSet[(y, x)] = set(S)
                                    done = True
                                    break

    # Обновляем G на основе amat (так как pdsep удаляет ребра)
    for i in range(len(amat)):
        for j in range(len(amat[i])):
            if amat[i][j] == 0:
                G[i][j] = False
            else:
                G[i][j] = PAG_CIRCLE

    return {
        'G': G,
        "sepset": sepSet,
        "pMax": pMax,
        "allPdsep": allPdsep_tmp,
        "max_ord": ord,
        "pdsep_stats": {
            "pairs_scanned": int(pdsep_pairs_scanned),
            "pairs_with_diffset": int(pdsep_pairs_with_diffset),
            "ci_tests": int(pdsep_ci_tests),
        },
    }


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


def cfci_stable(cg, data, alpha=0.05, indep_test='fisherz', maj_rule=False, n_jobs=1, verbose=False,
                use_ci_cache=True, background_knowledge=None, show_progress: bool = False):
    cit = CIT(data, method=indep_test)
    ci_stats = {"total": 0, "skeleton": 0, "triple_check": 0, "pdsep": 0}
    # Кэш живёт и при n_jobs > 1 — синхронизируется локом, как в
    # cfci_algo_paralel_cash. Прежде он отключался целиком, а ключи
    # ci_cache_* всё равно публиковались и оставались нулевыми, что
    # читалось в статистике как идеально отработавший кэш.
    ci_cache = {} if use_ci_cache else None
    if ci_cache is not None:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    cache_lock = threading.Lock()
    phase = {"name": "skeleton"}

    def counted_cit(i, j, S):
        with cache_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1
        if ci_cache is None:
            return cit(i, j, S)

        key = _ci_cache_key(i, j, S)
        with cache_lock:
            if key in ci_cache:
                ci_stats["ci_cache_hits"] += 1
                return ci_cache[key]
            ci_stats["ci_cache_misses"] += 1

        pval = cit(i, j, list(key[2]))
        with cache_lock:
            ci_cache[key] = pval
        return pval

    cg.set_ind_test(counted_cit)

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

    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    graphDict = SkeletonDiscovery(
        cg, alpha, n_jobs=n_jobs_eff, verbose=verbose, show_progress=show_progress,
        protected_edges=protected_edges,
    ).run_fci()
    graphDict['sk'] = to_circle_adjacency(graphDict['sk'], MatrixEncoding.STANDARD)
    _, p = data.shape

    def test_func_wrapper(suffStat, x, y, S):
        return counted_cit(x, y, S)

    suffStat = None

    phase["name"] = "pdsep"
    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict['sepset'],
        alpha,
        graphDict['pMax'],
        unfVect=set(),
        protected_edges=protected_edges,
    )

    # Conservative FCI classifies triples on the final skeleton and sepsets.
    phase["name"] = "triple_check"
    post_pdsep_graph = {
        "sk": to_circle_adjacency(pdsep_res["G"], MatrixEncoding.STANDARD),
        "sepset": pdsep_res["sepset"],
        "pMax": pdsep_res["pMax"],
        "unfTriples": set(),
    }

    pc_cons_intern(
        post_pdsep_graph,
        suffStat,
        alpha,
        test_func_wrapper,
        version_unf=(1, 1),
        maj_rule=maj_rule,
        verbose=verbose,
    )

    rule_stats = {}
    pag = udag2pag(
        pdsep_res["G"],
        post_pdsep_graph["sepset"],
        p,
        unfVect=post_pdsep_graph["unfTriples"],
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=[node.name for node in cg.nodes],
        # CFCI/FCI+ не ориентируют коллайдер, если для пары не записано
        # ни одного разделяющего множества — сохраняем прежнее поведение.
        require_recorded_sepset=True,
    )
    pdsep_stats = pdsep_res.get("pdsep_stats", {})
    ci_stats["pdsep_pairs_scanned"] = int(pdsep_stats.get("pairs_scanned", 0))
    ci_stats["pdsep_pairs_with_diffset"] = int(pdsep_stats.get("pairs_with_diffset", 0))
    ci_stats["pdsep_ci_tests"] = int(pdsep_stats.get("ci_tests", 0))
    ci_stats["ambiguous_triples"] = int(len(post_pdsep_graph["unfTriples"]))
    for rule, changes in rule_stats.items():
        ci_stats[f"orient_{rule.lower()}_changes"] = int(changes)
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = int(len(ci_cache))
    cg.ci_stats = ci_stats
    return np.array(pag, dtype=int)
