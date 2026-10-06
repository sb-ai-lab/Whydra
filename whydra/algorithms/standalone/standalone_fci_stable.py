import copy
import itertools
from itertools import combinations, chain
import numpy as np

# Импортируем универсальный класс тестов из соседнего файла
from ..graph_core import CIT

# --- Вспомогательные функции ---

def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


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
        ind = np.transpose(np.where(sk == 1))
        ind = ind[ind[:, 1].argsort()]

        tripleMatrix = []
        for a, b in ind:
            for c in range(p):
                if a < c and sk[a, c] == 0 and sk[b, c] == 1:
                    tripleMatrix.append((a, b, c))

        for a, b, c in tripleMatrix:
            nbrsA = np.where(sk[:, a] == 1)[0]
            nbrsC = np.where(sk[:, c] == 1)[0]

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

        ind = np.transpose(np.where(G == 1))

        # STABLE SKELETON: Snapshot
        G1 = np.copy(G)

        for x, y in ind:
            if G[y, x] == 1:
                nbrs = np.where(G1[:, x] == 1)[0]
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

def updateList(path, set_nodes, old_list):
    temp = []
    if len(old_list) > 0:
        temp = old_list
    temp.extend([path + [s] for s in set_nodes])
    return temp


def faith_check(cp, unfVect, p, boolean=True):
    if (boolean == False): return 0
    n = len(cp)
    i1 = [i for i in range(n)]
    i2 = [(i + 1) % n for i in i1]
    i3 = [(i + 2) % n for i in i1]

    for i in range(len(i1)):
        if unfVect is not None:
            if (cp[i1[i]], cp[i2[i]], cp[i3[i]]) in unfVect or (cp[i3[i]], cp[i2[i]], cp[i1[i]]) in unfVect:
                if boolean: return False
    if boolean: return True


def minDiscPath(pag, a, b, c):
    p = pag.shape[0]
    visited = [False] * p
    visited[a] = visited[b] = visited[c] = True

    # Ищем соседей a, которые соединены с a ребром типа 2 (стрелка в a)
    indD = [i for i in range(p) if pag[a, i] != 0 and pag[i, a] == 2 and not visited[i]]
    if len(indD) > 0:
        path_list = updateList([a], indD, [])
        while len(path_list) > 0:
            mpath = path_list[0]
            d = mpath[-1]
            if pag[c, d] == 0 and pag[d, c] == 0:
                mpath.reverse()
                return mpath + [b, c]
            else:
                if len(mpath) < 2:
                    pred = a
                else:
                    pred = mpath[-2]

                path_list = path_list[1:]
                visited[d] = True

                if pag[d, c] == 2 and pag[c, d] == 3 and pag[pred, d] == 2:
                    indR = [i for i in range(p) if pag[d, i] != 0 and pag[i, d] == 2 and not visited[i]]
                    if len(indR) > 0:
                        path_list = updateList(mpath, indR, path_list)
    return []


def minUncovCircPath(p, pag, path, unfVect):
    visited = [False] * p
    for node in path: visited[node] = True
    a, b, c, d = path
    min_ucp_path = []

    indX = [i for i in range(p) if pag[c, i] == 1 and pag[i, c] == 1 and not visited[i]]
    if len(indX) > 0:
        path_list = updateList([c], indX, [])
        done = False
        while not done and len(path_list) > 0:
            mpath = path_list[0]
            path_list = path_list[1:]
            x = mpath[-1]
            visited[x] = True
            if pag[x, d] == 1 and pag[d, x] == 1:
                full_path = [a] + mpath + [d, b]
                uncov = True
                for i in range(len(full_path) - 2):
                    if pag[full_path[i], full_path[i + 2]] != 0 or pag[full_path[i + 2], full_path[i]] != 0:
                        uncov = False
                        break
                if uncov:
                    if len(unfVect) == 0 or faith_check(full_path, unfVect, p):
                        min_ucp_path = full_path
                        done = True
            else:
                indR = [i for i in range(p) if pag[x, i] == 1 and pag[i, x] == 1 and not visited[i]]
                if len(indR) > 0:
                    path_list = updateList(mpath, indR, path_list)
    return min_ucp_path


def minUncovPdPath(p, pag, a, b, c, unfVect):
    visited = [False] * p
    visited[a] = True;
    visited[b] = True;
    visited[c] = True
    min_upd_path = []

    indD = [i for i in range(p) if
            (pag[b, i] == 1 or pag[b, i] == 2) and
            (pag[i, b] == 1 or pag[i, b] == 3) and
            pag[i, a] == 0 and not visited[i]]

    if len(indD) > 0:
        path_list = updateList([b], indD, [])
        done = False
        while len(path_list) > 0 and not done:
            mpath = path_list[0]
            path_list = path_list[1:]
            d = mpath[-1]
            visited[d] = True

            if (pag[d, c] == 1 or pag[d, c] == 2) and (pag[c, d] == 1 or pag[c, d] == 3):
                full_path = [a] + mpath + [c]
                uncov = True
                for i in range(len(full_path) - 2):
                    if pag[full_path[i], full_path[i + 2]] != 0 or pag[full_path[i + 2], full_path[i]] != 0:
                        uncov = False
                        break
                if uncov:
                    if len(unfVect) == 0 or faith_check(full_path, unfVect, p):
                        min_upd_path = full_path
                        done = True
            else:
                indR = [i for i in range(p) if
                        (pag[d, i] == 1 or pag[d, i] == 2) and
                        (pag[i, d] == 1 or pag[i, d] == 3) and
                        not visited[i]]
                if len(indR) > 0:
                    path_list = updateList(mpath, indR, path_list)
    return min_upd_path


def legal_path(a, b, c, amat):
    a_b = amat[a][b]
    if a == c or a_b == 0 or amat[b][c] == 0:
        return False
    return amat[a][c] != 0 or (a_b == 2 and amat[c][b] == 2)


def qreach(x, amat):
    p = amat.shape[0]
    A = (amat != 0).astype(int)
    PSEP = list(np.where(A[x, :] != 0)[0])
    Q = copy.deepcopy(PSEP)
    P = [x] * len(Q)
    A_search = np.copy(A)
    A_search[x, :] = 0

    idx = 0
    while idx < len(Q):
        a = Q[idx]
        pred = P[idx]
        idx += 1
        nbrs = np.where(A_search[a, :] != 0)[0]

        for b in nbrs:
            if b == x: continue
            if legal_path(pred, a, b, amat):
                A_search[a, b] = 0
                Q.append(b)
                P.append(a)
                PSEP.append(b)
    while x in PSEP:
        PSEP.remove(x)
    return sorted(list(set(PSEP)))


def pdsep(skel, suffStat, indepTest, p, sepSet, alpha, pMax, m_max=float('inf'), pdsep_max=float('inf'), unfVect=None):
    G = skel['sk'].astype(int)
    n_edgetest = [0 for i in range(1000)]
    ord = 0
    allPdsep_tmp = [set() for i in range(p)]
    amat = np.copy(G)

    ind = []
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == 1:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))

    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepSet[(x, z)] or y in sepSet[(z, x)]):
                if len(unfVect) == 0:
                    amat[x][y] = amat[z][y] = 2
                else:
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = 2

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


def udag2pag(pag, sepset, p, unfVect=None, rules=None, orientCollider=True):
    if rules is None: rules = [True] * 10
    if unfVect is None: unfVect = set()

    pag = np.array(pag, dtype=int)
    skel_mask = (pag != 0)
    pag[skel_mask] = 1

    if orientCollider:
        ind = []
        for i in range(len(pag)):
            for j in range(len(pag[i])):
                if pag[i][j] == 1:
                    ind.append((i, j))
        ind = sorted(ind, key=lambda x: (x[1], x[0]))
        for x, y in ind:
            allZ = [i for i in range(len(pag[y])) if pag[y][i] != 0 and i != x]
            for z in allZ:
                if pag[x][z] == 0 and not (y in sepset.get((x, z), []) or y in sepset.get((z, x), [])):
                    if len(unfVect) == 0:
                        pag[x][y] = pag[z][y] = 2
                    else:
                        if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                            pag[x][y] = pag[z][y] = 2

    old_pag1 = None
    while not np.array_equal(old_pag1, pag):
        old_pag1 = copy.deepcopy(pag)

        # R1
        if rules[0]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, b in ind:
                indC = [i for i in range(len(pag)) if
                        pag[b][i] != 0 and pag[i][b] == 1 and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indC) != 0:
                    if len(unfVect) != 0:
                        for c in indC:
                            pag[b][c] = 2
                            pag[c][b] = 3
                    else:
                        for c in indC:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                pag[b][c] = 2
                                pag[c][b] = 3
        # R2
        if rules[1]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 1 and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == 2 and pag[i][a] == 3 and pag[c][i] != 0 and pag[i][c] == 2) or
                        (pag[a][i] == 2 and pag[i][1] != 0 and pag[c][i] == 3 and pag[i][c] == 2)]
                if len(indB) > 0:
                    pag[a][c] = 2
        # R3
        if rules[2]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, d in ind:
                indAC = [i for i in range(len(pag)) if
                         pag[b][i] != 0 and pag[i][b] == 2 and pag[i][d] == 1 and pag[d][i] != 0]
                if len(indAC) >= 2:
                    if len(unfVect) == 0:
                        counter = -1
                        while counter < len(indAC) - 1 and pag[d][b] != 2:
                            counter += 1
                            ii = counter
                            while ii < len(indAC) - 1 and pag[d][b] != 2:
                                ii += 1
                                if pag[indAC[counter]][indAC[ii]] == 0 and pag[indAC[ii]][indAC[counter]] == 0:
                                    pag[d][b] = 2
                    else:
                        for a, c in combinations(indAC, 2):
                            if pag[a][c] == 0 and pag[c][a] == 0 and c != a:
                                if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                    pag[d][b] = 2
        # R4
        if rules[3]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                b, c = ind[0]
                ind = ind[1:]
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == 2 and pag[i][b] != 0 and pag[c][i] == 3 and pag[i][c] == 2]

                while len(indA) > 0 and pag[c][b] == 1:
                    a = indA[0]
                    indA = indA[1:]
                    done = False
                    while done == False and pag[a][b] != 0 and pag[a][c] != 0 and pag[b][c] != 0:
                        md_path = minDiscPath(pag, a, b, c)
                        if len(md_path) == 1:
                            done = True
                        else:
                            if b in sepset.get((md_path[0], md_path[-1]), []) or b in sepset.get(
                                    (md_path[-1], md_path[0]), []):
                                pag[b][c] = 2
                                pag[c][b] = 3
                            else:
                                pag[a][b] = pag[b][c] = pag[c][b] = 2
                            done = True

        # R5
        if rules[4]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 1 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind[0]
                ind = ind[1:]
                indC = [i for i in range(len(pag)) if
                        pag[a][i] == 1 and pag[i][a] == 1 and pag[b][i] == 0 and pag[i][b] == 0 and i != b]
                indD = [i for i in range(len(pag)) if
                        pag[b][i] == 1 and pag[i][b] == 1 and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indD) > 0 and len(indC) > 0:
                    counterC = -1
                    while counterC < len(indC) - 1 and pag[a][b] == 1:
                        counterC += 1
                        c = indC[counterC]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[a][b] == 1:
                            counterD += 1
                            d = indD[counterD]
                            if pag[c][d] == 1 and pag[d][c] == 1:
                                if len(unfVect) == 0:
                                    pag[a][b] = pag[b][a] = 3
                                    pag[a][c] = pag[c][a] = 3
                                    pag[c][d] = pag[c][d] = 3
                                    pag[d][b] = pag[b][d] = 3
                                else:
                                    path2check = [a, c, d, b]
                                    if faith_check(path2check, unfVect, p):
                                        pag[a][b] = pag[b][a] = 3
                                        pag[a][c] = pag[c][a] = 3
                                        pag[c][d] = pag[c][d] = 3
                                        pag[d][b] = pag[b][d] = 3
                            else:
                                ucp = minUncovCircPath(p, pag=pag, path=(a, c, d, b), unfVect=unfVect)
                                if len(ucp) > 1:
                                    pag[ucp[0]][ucp[-1]] = pag[ucp[-1]][ucp[0]] = 3
                                    for j in range(len(ucp) - 1):
                                        pag[ucp[j]][ucp[j + 1]] = pag[ucp[j + 1]][ucp[j]] = 3
        # R6
        if rules[5]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            for b, c in ind:
                if len([i for i in range(len(pag)) if pag[b][i] == 3 and pag[i][b] == 3]) > 0:
                    pag[c][b] = 3
        # R7
        if rules[6]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, c in ind:
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == 3 and pag[i][b] == 1 and pag[c][i] == 0 and pag[i][c] == 0 and i != c]
                if len(indA) > 0:
                    if len(unfVect) == 0:
                        pag[c][b] = 3
                    else:
                        for a in indA:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                pag[c][b] = 3
        # R8
        if rules[7]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        pag[i][a] == 3 and (pag[a][i] == 2 or pag[a][i] == 1) and pag[c][i] == 3 and pag[i][c] == 2]
                if len(indB) > 0:
                    pag[c][a] = 3
        # R9
        if rules[8]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                a, c = ind[0]
                ind = ind[1:]
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == 2 or pag[a][i] == 1) and
                        (pag[i][a] == 1 or pag[i][a] == 3) and
                        (pag[c][i] == 0 and pag[i][c] == 0) and
                        i != c]
                while len(indB) > 0 and pag[c][a] == 1:
                    b = indB[0]
                    indB = indB[1:]
                    upd = minUncovPdPath(p, pag, a, b, c, unfVect=unfVect)
                    if len(upd) > 1:
                        pag[c][a] = 3
        # R10
        if rules[9]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind[0]
                ind = ind[1:]
                indB = [i for i in range(p) if pag[c][i] == 3 and pag[i][c] == 2]
                if len(indB) >= 2:
                    counterB = -1
                    while counterB < len(indB) - 1 and pag[c][a] == 1:
                        counterB += 1
                        b = indB[counterB]
                        indD = [i for i in indB if i != b]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[c][a] == 1:
                            counterD += 1
                            d = indD[counterD]
                            if (
                                    (pag[a][b] == 1 or pag[a][b] == 2) and
                                    (pag[b][a] == 1 or pag[b][a] == 3) and
                                    (pag[a][d] == 1 or pag[a][d] == 2) and
                                    (pag[d][a] == 1 or pag[d][a] == 3) and
                                    pag[d][b] == 0 and pag[b][d] == 0
                            ):
                                if len(unfVect) == 0:
                                    pag[c][a] = 3
                                else:
                                    if (b, a, d) not in unfVect and (d, a, b) not in unfVect:
                                        pag[c][a] = 3
                            else:
                                indX = [i for i in range(p) if
                                        (pag[a][i] == 1 or pag[a][i] == 2) and
                                        (pag[i][a] == 1 or pag[i][a] == 3) and
                                        i != c]
                                if len(indX) >= 2:
                                    counterX1 = -1
                                    while counterX1 < len(indX) - 1 and pag[c][a] == 1:
                                        counterX1 += 1
                                        # first_pos = indA[counterX1]  # indA not defined in original code properly?
                                        # В оригинале indA определен как [] в начале функции и не заполняется.
                                        # Это баг в pcalg порте?
                                        # Предположим, что indX используется.

                                        first_pos = indX[counterX1]

                                        indX2 = [i for i in indX if i != first_pos]
                                        counterX2 = -1
                                        while counterX2 < len(indX2) - 1 and pag[c][a] == 1:
                                            counterX2 += 1
                                            sec_pos = indX2[counterX2]
                                            t1 = minUncovPdPath(p, pag, a, first_pos, b, unfVect=unfVect)
                                            if len(t1) > 1:
                                                t2 = minUncovPdPath(p, pag, a, sec_pos, d, unfVect=unfVect)
                                                if len(t2) > 1 and first_pos != sec_pos and pag[first_pos][
                                                    sec_pos] == 0:
                                                    if len(unfVect) == 0:
                                                        pag[c][a] = 3
                                                    elif (first_pos, a, sec_pos) not in unfVect and (
                                                            sec_pos, a, first_pos) not in unfVect:
                                                        pag[c][a] = 3
        return pag


def fci_stable(data, alpha=0.05, indep_test='fisherz', verbose=False):
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

    print(f'graphDict skeleton {graphDict}')

    pc_cons_intern(graphDict, suffStat, alpha, test_func_wrapper, maj_rule=False)
    print(f'graphDict before pdsep {graphDict}')
    print(f'graphDict before suffStat {suffStat}')
    print(f'graphDict before alpha {alpha}')
    print(f'graphDict before test_func_wrapper {test_func_wrapper}')

    pdsep_res = pdsep(graphDict, suffStat, test_func_wrapper, p, graphDict['sepset'], alpha, graphDict['pMax'],
                      unfVect=graphDict['unfTriples'])
    print(f'graphDict after pdsep {graphDict}')
    print(f'graphDict after suffStat {suffStat}')
    print(f'graphDict after alpha {alpha}')
    print(f'graphDict after test_func_wrapper {test_func_wrapper}')

    pag = udag2pag(pdsep_res['G'], pdsep_res['sepset'], p, unfVect=graphDict['unfTriples'])

    return pag