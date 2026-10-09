from itertools import combinations
import numpy as np

# Импортируем универсальный класс тестов из соседнего файла
from ..graph_core import CIT
from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint

# --- Вспомогательные функции ---
from .skeleton import SkeletonDiscovery
from .pag_init import to_circle_adjacency
from .pag_rules import PAG_ARROW, PAG_CIRCLE, powerset, qreach, udag2pag


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


# def skeleton(suffStat, indepTest, alpha, labels, method,
#              fixedGaps, fixedEdges,
#              NAdelete, m_max, numCores, verbose):
#     sepset = {}
#     for i in itertools.permutations([i for i in range(len(labels))], 2):
#         sepset[i] = set()
#
#     G = np.ones((len(labels), len(labels)), dtype=int)
#     pMax = np.full((len(labels), len(labels)), -np.inf)
#
#     for i in range(len(labels)):
#         pMax[i, i] = 1
#         G[i, i] = 0
#
#     done = False
#     ord = 0
#     n_edgetests = {}
#
#     while done != True and np.any(G) and ord <= m_max:
#         ord1 = ord + 1
#         n_edgetests[ord1] = 0
#         done = True
#
#         ind = np.transpose(np.where(G == 1))
#
#         # STABLE SKELETON: Snapshot
#         G1 = np.copy(G)
#
#         for x, y in ind:
#             if G[y, x] == 1:
#                 nbrs = np.where(G1[:, x] == 1)[0]
#                 nbrs = nbrs[nbrs != y]
#
#                 if len(nbrs) >= ord:
#                     if len(nbrs) > ord:
#                         done = False
#
#                     for nbrs_S in set(itertools.combinations(nbrs, ord)):
#                         n_edgetests[ord1] = n_edgetests[ord1] + 1
#                         pval = indepTest(suffStat, x, y, list(nbrs_S))
#                         if pMax[x, y] < pval:
#                             pMax[x, y] = pval
#                         if pval >= alpha:
#                             G[x, y] = G[y, x] = 0
#                             sepset[(x, y)] = list(nbrs_S)
#                             sepset[(y, x)] = list(nbrs_S)
#                             break
#         ord += 1
#
#     for i in range(0, len(labels) - 1):
#         for j in range(1, len(labels)):
#             pMax[i, j] = pMax[j, i] = max(pMax[i, j], pMax[j, i])
#
#     return {'sk': G, 'pMax': pMax, 'sepset': sepset, "unfTriples": set(), "max_ord": ord - 1}


# --- Функции для поиска путей (из pcalg) ---

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


def fci_stable(cg, data, alpha=0.05, indep_test='fisherz', n_jobs=1, verbose=False, use_ci_cache=True,
               background_knowledge=None, show_progress: bool = False):
    # SkeletonDiscovery uses cg.ci_test(...), so CI test must be set before run_fci().
    cit = CIT(data, method=indep_test)
    ci_stats = {"total": 0, "skeleton": 0, "pdsep": 0}
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = {} if use_ci_cache else None

    def counted_cit(i, j, S):
        ci_stats["total"] += 1
        ci_stats[phase["name"]] += 1
        if ci_cache is None:
            return cit(i, j, S)

        key = _ci_cache_key(i, j, S)
        if key in ci_cache:
            ci_stats["ci_cache_hits"] += 1
            return ci_cache[key]

        ci_stats["ci_cache_misses"] += 1
        pval = cit(i, j, list(key[2]))
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

    # Step 1: Skeleton discovery
    # print(f'going skeleton')
    # print(f'cg.G.graph 0 {cg.G.graph}')

    # ini_file: str = 'results/cg_matrix_ini.csv'
    # np.savetxt(ini_file, cg.G.graph, delimiter=',')

    graphDict = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs,
        verbose=verbose,
        show_progress=show_progress,
        protected_edges=protected_edges,
    ).run_fci()
    # Explicitly represent post-skeleton graph as o-o for FCI phases.
    graphDict['sk'] = to_circle_adjacency(graphDict['sk'], MatrixEncoding.STANDARD)

    # skeleton_file: str = 'results/cg_matrix_skeleton.csv'
    # np.savetxt(skeleton_file, cg.G.graph, delimiter=',')


    n, p = data.shape

    # Обертка для совместимости с функциями FCI, которые ожидают (suffStat, x, y, S)
    def test_func_wrapper(suffStat, x, y, S):
        return counted_cit(x, y, S)

    # suffStat больше не нужен, так как CIT хранит данные внутри себя
    suffStat = None

    # graphDict = skeleton(suffStat, test_func_wrapper, alpha, labels=[str(i) for i in range(p)], method="stable",
    #                      fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float('inf'),
    #                      numCores=1, verbose=verbose)

    # print(f'graphDict skeleton pdsep {graphDict}')
    # print(f'graphDict skeleton pMax {graphDict["pMax"]}')
    # print(f'graphDict skeleton alpha {alpha}')
    # print(f'graphDict skeleton test_func_wrapper {test_func_wrapper}')

    # Classical FCI: do not apply conservative triple classification here.
    # print(f'graphDict before pdsep {graphDict}')
    # print(f'graphDict before suffStat {suffStat}')
    # print(f'graphDict before alpha {alpha}')
    # print(f'graphDict before test_func_wrapper {test_func_wrapper}')

    phase["name"] = "pdsep"
    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict['sepset'],
        alpha,
        graphDict['pMax'],
        unfVect=graphDict['unfTriples'],
        protected_edges=protected_edges,
    )
    # print(f'graphDict after pdsep {graphDict}')
    # print(f'graphDict after suffStat {suffStat}')
    # print(f'graphDict after alpha {alpha}')
    # print(f'graphDict after test_func_wrapper {test_func_wrapper}')

    rule_stats = {}
    pag = udag2pag(
        pdsep_res['G'],
        pdsep_res['sepset'],
        p,
        unfVect=graphDict['unfTriples'],
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=node_names,
    )
    pdsep_stats = pdsep_res.get("pdsep_stats", {})
    ci_stats["pdsep_pairs_scanned"] = int(pdsep_stats.get("pairs_scanned", 0))
    ci_stats["pdsep_pairs_with_diffset"] = int(pdsep_stats.get("pairs_with_diffset", 0))
    ci_stats["pdsep_ci_tests"] = int(pdsep_stats.get("ci_tests", 0))
    for k, v in rule_stats.items():
        ci_stats[f"orient_{k.lower()}_changes"] = int(v)
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = int(len(ci_cache))
    cg.ci_stats = ci_stats

    return pag
