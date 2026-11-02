import itertools
from itertools import combinations
import numpy as np

# Импортируем универсальный класс тестов из соседнего файла
from ..graph_core import CIT
from .pag_rules import PAG_ARROW, PAG_CIRCLE, qreach, udag2pag

# --- Вспомогательные функции ---

def skeleton(suffStat, indepTest, alpha, labels, method,
             fixedGaps, fixedEdges,
             NAdelete, m_max, numCores, verbose):
    sepset = {}
    for i in itertools.permutations([i for i in range(len(labels))], 2):
        sepset[i] = set()

    G = np.ones((len(labels), len(labels)), dtype=int)
    pMax = np.full((len(labels), len(labels)), -np.inf)

    for i in range(len(labels)):
        pMax[i, i] = 1
        G[i, i] = 0

    done = False
    ord = 0
    n_edgetests = {}

    while done != True and np.any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True

        ind = np.transpose(np.where(G != 0))

        # STABLE SKELETON: Snapshot
        G1 = np.copy(G)

        for x, y in ind:
            if G[y, x] != 0:
                nbrs = np.where(G1[:, x] != 0)[0]
                nbrs = nbrs[nbrs != y]

                if len(nbrs) >= ord:
                    if len(nbrs) > ord:
                        done = False

                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        pval = indepTest(suffStat, x, y, list(nbrs_S))
                        if pMax[x, y] < pval:
                            pMax[x, y] = pval
                        if pval >= alpha:
                            G[x, y] = G[y, x] = 0
                            sepset[(x, y)] = list(nbrs_S)
                            sepset[(y, x)] = list(nbrs_S)
                            break
        ord += 1

    for i in range(0, len(labels) - 1):
        for j in range(1, len(labels)):
            pMax[i, j] = pMax[j, i] = max(pMax[i, j], pMax[j, i])

    return {'sk': G, 'pMax': pMax, 'sepset': sepset, "unfTriples": set(), "max_ord": ord - 1}


# --- Функции для поиска путей (из pcalg) ---

def pdsep(skel, suffStat, indepTest, p, sepSet, alpha, pMax, m_max=float('inf'), pdsep_max=float('inf'), unfVect=None):
    G = skel['sk'].astype(int)
    n_edgetest = [0 for i in range(1000)]
    ord = 0
    allPdsep_tmp = [set() for i in range(p)]
    # G — булева смежность; метки строим явно: раньше это работало лишь
    # потому, что в кодировке pcalg смежность 1 совпадала с кружком 1.
    amat = np.where(np.asarray(G) != 0, PAG_CIRCLE, 0)

    ind = []
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] != 0:
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
            tf = [i for i in tf1 if i != y]
            diff_set = [i for i in tf if i not in adj_x]
            allPdsep_tmp[x] = tf + [y]

            if len(diff_set) > 0:
                done = False
                ord = 0
                while not done and ord < min(len(tf), m_max):
                    ord += 1
                    if ord == 1:
                        for S in diff_set:
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
                G[i][j] = True

    return {'G': G, "sepset": sepSet, "pMax": pMax, "allPdsep": allPdsep_tmp, "max_ord": ord}


def fci_stable(
    data,
    alpha=0.05,
    indep_test='fisherz',
    verbose=False,
):
    """Строит PAG классическим алгоритмом FCI-Stable.

    Args:
        data: матрица наблюдений, строки — объекты, столбцы — переменные.
        alpha: порог независимости.
        indep_test: название CI-теста.
        verbose: печатать промежуточные результаты.

    Returns:
        Матрица PAG с метками концов рёбер.
    """
    n, p = data.shape

    # Инициализируем универсальный CIT
    cit = CIT(data, method=indep_test)

    # Обертка для совместимости с функциями FCI, которые ожидают (suffStat, x, y, S)
    def test_func_wrapper(suffStat, x, y, S):
        return cit(x, y, S)

    # suffStat больше не нужен, так как CIT хранит данные внутри себя
    suffStat = None

    graphDict = skeleton(suffStat, test_func_wrapper, alpha, labels=[str(i) for i in range(p)], method="stable",
                         fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float('inf'),
                         numCores=1, verbose=verbose)

    if verbose:
        print(f'graphDict skeleton {graphDict}')

    # Классический FCI-Stable не содержит консервативного шага: causal-learn fci
    # и pcalg fci() по умолчанию его не вызывают. С проверкой != 0 прежний
    # вызов превращал v1 в консервативный FCI.
    if verbose:
        print(f'graphDict before pdsep {graphDict}')
        print(f'graphDict before suffStat {suffStat}')
        print(f'graphDict before alpha {alpha}')
        print(f'graphDict before test_func_wrapper {test_func_wrapper}')

    pdsep_res = pdsep(graphDict, suffStat, test_func_wrapper, p, graphDict['sepset'], alpha, graphDict['pMax'],
                      unfVect=graphDict['unfTriples'])
    if verbose:
        print(f'graphDict after pdsep {graphDict}')
        print(f'graphDict after suffStat {suffStat}')
        print(f'graphDict after alpha {alpha}')
        print(f'graphDict after test_func_wrapper {test_func_wrapper}')

    pag = udag2pag(pdsep_res['G'], pdsep_res['sepset'], p, unfVect=graphDict['unfTriples'])

    return pag