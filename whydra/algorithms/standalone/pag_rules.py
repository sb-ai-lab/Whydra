"""Общие правила и вспомогательные обходы для семейства FCI.

Эти функции были побайтово продублированы в шести-девяти модулях каждая.
Копии разошлись бы при первой же правке, поэтому живут здесь в одном
экземпляре; модули семейства импортируют их отсюда.
"""

import copy
from itertools import chain, combinations

import numpy as np

from ..graph_core.endpoints import Endpoint

from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)


#: Метки концов рёбер в PAG-матрицах семейства FCI.
#:
#: Кодировка causal-learn — та же, что у Endpoint, CausalGraph, PC, RAI, RFCI,
#: GFCI и background knowledge. Раньше семейство FCI/CFCI/FCI+ считало в
#: кодировке pcalg (circle=1, arrow=2, tail=3), из-за чего в проекте
#: сосуществовали две кодировки, а 1 и 2 означали в них разное. Теперь
#: кодировка в библиотеке одна.
PAG_NULL = Endpoint.NULL.value      # 0
PAG_TAIL = Endpoint.TAIL.value      # -1
PAG_ARROW = Endpoint.ARROW.value    # 1
PAG_CIRCLE = Endpoint.CIRCLE.value  # 2


def updateList(path, set_nodes, old_list):
    temp = []
    if len(old_list) > 0:
        temp = old_list
    temp.extend([path + [s] for s in set_nodes])
    return temp


def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def legal_path(a, b, c, amat):
    a_b = amat[a][b]
    if a == c or a_b == 0 or amat[b][c] == 0:
        return False
    return amat[a][c] != 0 or (a_b == PAG_ARROW and amat[c][b] == PAG_ARROW)


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
    indD = [i for i in range(p) if pag[a, i] != 0 and pag[i, a] == PAG_ARROW and not visited[i]]
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

                if pag[d, c] == PAG_ARROW and pag[c, d] == PAG_TAIL and pag[pred, d] == PAG_ARROW:
                    indR = [i for i in range(p) if pag[d, i] != 0 and pag[i, d] == PAG_ARROW and not visited[i]]
                    if len(indR) > 0:
                        path_list = updateList(mpath, indR, path_list)
    return []

def minUncovCircPath(p, pag, path, unfVect):
    visited = [False] * p
    for node in path: visited[node] = True
    a, b, c, d = path
    min_ucp_path = []

    indX = [i for i in range(p) if pag[c, i] == PAG_CIRCLE and pag[i, c] == PAG_CIRCLE and not visited[i]]
    if len(indX) > 0:
        path_list = updateList([c], indX, [])
        done = False
        while not done and len(path_list) > 0:
            mpath = path_list[0]
            path_list = path_list[1:]
            x = mpath[-1]
            visited[x] = True
            if pag[x, d] == PAG_CIRCLE and pag[d, x] == PAG_CIRCLE:
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
                indR = [i for i in range(p) if pag[x, i] == PAG_CIRCLE and pag[i, x] == PAG_CIRCLE and not visited[i]]
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
            (pag[b, i] == PAG_CIRCLE or pag[b, i] == PAG_ARROW) and
            (pag[i, b] == PAG_CIRCLE or pag[i, b] == PAG_TAIL) and
            pag[i, a] == 0 and not visited[i]]

    if len(indD) > 0:
        path_list = updateList([b], indD, [])
        done = False
        while len(path_list) > 0 and not done:
            mpath = path_list[0]
            path_list = path_list[1:]
            d = mpath[-1]
            visited[d] = True

            if (pag[d, c] == PAG_CIRCLE or pag[d, c] == PAG_ARROW) and (pag[c, d] == PAG_CIRCLE or pag[c, d] == PAG_TAIL):
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
                        (pag[d, i] == PAG_CIRCLE or pag[d, i] == PAG_ARROW) and
                        (pag[i, d] == PAG_CIRCLE or pag[i, d] == PAG_TAIL) and
                        not visited[i]]
                if len(indR) > 0:
                    path_list = updateList(mpath, indR, path_list)
    return min_upd_path


def udag2pag(pag, sepset, p, unfVect=None, rules=None, orientCollider=True, rule_stats=None,
             background_knowledge=None, node_names=None, require_recorded_sepset=False):
    """Правила ориентации R0–R10, общие для всего семейства FCI.

    Раньше эта функция жила в семи модулях тремя расходящимися версиями.
    Единственное содержательное различие между ними вынесено в параметр:

    ``require_recorded_sepset``
        Как трактовать пару, для которой разделяющее множество вообще не
        записано. ``False`` (FCI, FCI-cash, GFCI) — ориентировать коллайдер:
        «y не входит в пустое множество» истинно. ``True`` (CFCI, FCI+) —
        не ориентировать: без записанного множества утверждать нечего.
        Значение по умолчанию сохраняет поведение основной ветки FCI.
    """
    if rules is None: rules = [True] * 10
    if unfVect is None: unfVect = set()
    if rule_stats is None:
        rule_stats = {}
    for k in ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]:
        rule_stats.setdefault(k, 0)

    pag = np.array(pag, dtype=int)
    if node_names is None:
        node_names = list(range(len(pag)))

    # Правила ниже пишут метки только через guard, поэтому background knowledge
    # остаётся инвариантом всего прохода, а не применяется однократно перед
    # правилами. Без BK guard ничего не запрещает и обвязка прозрачна.
    _bk = (
        background_knowledge.orientation_guard(
            node_names, phase=BKPhase.BETWEEN_ORIENTATION, encoding=MatrixEncoding.STANDARD
        )
        if background_knowledge is not None
        else BKOrientationGuard(None, MatrixEncoding.STANDARD)
    )
    skel_mask = (pag != 0)
    pag[skel_mask] = PAG_CIRCLE

    # BK применяется после скелета/PDSEP, но ДО R0 и R1–R10: правила должны
    # стартовать с графа, который уже уважает ограничения.
    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="before_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )

    if orientCollider:
        before_r0 = pag.copy()
        ind = []
        for i in range(len(pag)):
            for j in range(len(pag[i])):
                if pag[i][j] == PAG_CIRCLE:
                    ind.append((i, j))
        ind = sorted(ind, key=lambda x: (x[1], x[0]))
        for x, y in ind:
            allZ = [i for i in range(len(pag[y])) if pag[y][i] != 0 and i != x]
            for z in allZ:
                if require_recorded_sepset:
                    sep_xz = sepset.get((x, z))
                    if sep_xz is None:
                        sep_xz = sepset.get((z, x))
                    is_collider = sep_xz is not None and pag[x][z] == 0 and y not in sep_xz
                else:
                    is_collider = pag[x][z] == 0 and not (
                        y in sepset.get((x, z), []) or y in sepset.get((z, x), []))
                if is_collider:
                    if len(unfVect) == 0:
                        _bk.set_mark(pag, x, y, PAG_ARROW)
                        _bk.set_mark(pag, z, y, PAG_ARROW)
                    else:
                        if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                            _bk.set_mark(pag, x, y, PAG_ARROW)
                            _bk.set_mark(pag, z, y, PAG_ARROW)
        rule_stats["R0"] += int(np.count_nonzero(pag != before_r0))

    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="after_colliders",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
        previous=before_r0 if orientCollider else None,
    )

    old_pag1 = None
    while not np.array_equal(old_pag1, pag):
        old_pag1 = copy.deepcopy(pag)

        # R1
        if rules[0]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, b in ind:
                indC = [i for i in range(len(pag)) if
                        pag[b][i] != 0 and pag[i][b] == PAG_CIRCLE and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indC) != 0:
                    if len(unfVect) == 0:
                        for c in indC:
                            _bk.set_mark(pag, b, c, PAG_ARROW)
                            _bk.set_mark(pag, c, b, PAG_TAIL)
                    else:
                        for c in indC:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                _bk.set_mark(pag, b, c, PAG_ARROW)
                                _bk.set_mark(pag, c, b, PAG_TAIL)
            rule_stats["R1"] += int(np.count_nonzero(pag != before_r))
        # R2
        if rules[1]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_CIRCLE and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == PAG_ARROW and pag[i][a] == PAG_TAIL and pag[c][i] != 0 and pag[i][c] == PAG_ARROW) or
                        (pag[a][i] == PAG_ARROW and pag[i][a] != 0 and pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW)]
                if len(indB) > 0:
                    _bk.set_mark(pag, a, c, PAG_ARROW)
            rule_stats["R2"] += int(np.count_nonzero(pag != before_r))
        # R3
        if rules[2]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, d in ind:
                indAC = [i for i in range(len(pag)) if
                         pag[b][i] != 0 and pag[i][b] == PAG_ARROW and pag[i][d] == PAG_CIRCLE and pag[d][i] != 0]
                if len(indAC) >= 2:
                    if len(unfVect) == 0:
                        counter = -1
                        while counter < len(indAC) - 1 and pag[d][b] != PAG_ARROW:
                            counter += 1
                            ii = counter
                            while ii < len(indAC) - 1 and pag[d][b] != PAG_ARROW:
                                ii += 1
                                if pag[indAC[counter]][indAC[ii]] == 0 and pag[indAC[ii]][indAC[counter]] == 0:
                                    _bk.set_mark(pag, d, b, PAG_ARROW)
                    else:
                        for a, c in combinations(indAC, 2):
                            if pag[a][c] == 0 and pag[c][a] == 0 and c != a:
                                if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                    _bk.set_mark(pag, d, b, PAG_ARROW)
            rule_stats["R3"] += int(np.count_nonzero(pag != before_r))
        # R4
        if rules[3]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                b, c = ind[0]
                ind = ind[1:]
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == PAG_ARROW and pag[i][b] != 0 and pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW]

                while len(indA) > 0 and pag[c][b] == PAG_CIRCLE:
                    a = indA[0]
                    indA = indA[1:]
                    done = False
                    while done == False and pag[a][b] != 0 and pag[a][c] != 0 and pag[b][c] != 0:
                        md_path = minDiscPath(pag, a, b, c)
                        if len(md_path) == 0:
                            done = True
                        else:
                            if b in sepset.get((md_path[0], md_path[-1]), []) or b in sepset.get(
                                    (md_path[-1], md_path[0]), []):
                                _bk.set_mark(pag, b, c, PAG_ARROW)
                                _bk.set_mark(pag, c, b, PAG_TAIL)
                            else:
                                _bk.set_mark(pag, a, b, PAG_ARROW)
                                _bk.set_mark(pag, b, c, PAG_ARROW)
                                _bk.set_mark(pag, c, b, PAG_ARROW)
                            done = True
            rule_stats["R4"] += int(np.count_nonzero(pag != before_r))

        # R5
        if rules[4]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_CIRCLE and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind[0]
                ind = ind[1:]
                indC = [i for i in range(len(pag)) if
                        pag[a][i] == PAG_CIRCLE and pag[i][a] == PAG_CIRCLE and pag[b][i] == 0 and pag[i][b] == 0 and i != b]
                indD = [i for i in range(len(pag)) if
                        pag[b][i] == PAG_CIRCLE and pag[i][b] == PAG_CIRCLE and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indD) > 0 and len(indC) > 0:
                    counterC = -1
                    while counterC < len(indC) - 1 and pag[a][b] == PAG_CIRCLE:
                        counterC += 1
                        c = indC[counterC]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[a][b] == PAG_CIRCLE:
                            counterD += 1
                            d = indD[counterD]
                            if pag[c][d] == PAG_CIRCLE and pag[d][c] == PAG_CIRCLE:
                                if len(unfVect) == 0:
                                    _bk.set_mark(pag, a, b, PAG_TAIL)
                                    _bk.set_mark(pag, b, a, PAG_TAIL)
                                    _bk.set_mark(pag, a, c, PAG_TAIL)
                                    _bk.set_mark(pag, c, a, PAG_TAIL)
                                    _bk.set_mark(pag, c, d, PAG_TAIL)
                                    _bk.set_mark(pag, d, c, PAG_TAIL)
                                    _bk.set_mark(pag, d, b, PAG_TAIL)
                                    _bk.set_mark(pag, b, d, PAG_TAIL)
                                else:
                                    path2check = [a, c, d, b]
                                    if faith_check(path2check, unfVect, p):
                                        _bk.set_mark(pag, a, b, PAG_TAIL)
                                        _bk.set_mark(pag, b, a, PAG_TAIL)
                                        _bk.set_mark(pag, a, c, PAG_TAIL)
                                        _bk.set_mark(pag, c, a, PAG_TAIL)
                                        _bk.set_mark(pag, c, d, PAG_TAIL)
                                        _bk.set_mark(pag, d, c, PAG_TAIL)
                                        _bk.set_mark(pag, d, b, PAG_TAIL)
                                        _bk.set_mark(pag, b, d, PAG_TAIL)
                            else:
                                ucp = minUncovCircPath(p, pag=pag, path=(a, c, d, b), unfVect=unfVect)
                                if len(ucp) > 1:
                                    _bk.set_mark(pag, ucp[0], ucp[-1], PAG_TAIL)
                                    _bk.set_mark(pag, ucp[-1], ucp[0], PAG_TAIL)
                                    for j in range(len(ucp) - 1):
                                        _bk.set_mark(pag, ucp[j], ucp[j + 1], PAG_TAIL)
                                        _bk.set_mark(pag, ucp[j + 1], ucp[j], PAG_TAIL)
            rule_stats["R5"] += int(np.count_nonzero(pag != before_r))
        # R6
        if rules[5]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            for b, c in ind:
                if len([i for i in range(len(pag)) if pag[b][i] == PAG_TAIL and pag[i][b] == PAG_TAIL]) > 0:
                    _bk.set_mark(pag, c, b, PAG_TAIL)
            rule_stats["R6"] += int(np.count_nonzero(pag != before_r))
        # R7
        if rules[6]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, c in ind:
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == PAG_TAIL and pag[i][b] == PAG_CIRCLE and pag[c][i] == 0 and pag[i][c] == 0 and i != c]
                if len(indA) > 0:
                    if len(unfVect) == 0:
                        _bk.set_mark(pag, c, b, PAG_TAIL)
                    else:
                        for a in indA:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                _bk.set_mark(pag, c, b, PAG_TAIL)
            rule_stats["R7"] += int(np.count_nonzero(pag != before_r))
        # R8
        if rules[7]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        pag[i][a] == PAG_TAIL and (pag[a][i] == PAG_ARROW or pag[a][i] == PAG_CIRCLE) and pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW]
                if len(indB) > 0:
                    _bk.set_mark(pag, c, a, PAG_TAIL)
            rule_stats["R8"] += int(np.count_nonzero(pag != before_r))
        # R9
        if rules[8]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                a, c = ind[0]
                ind = ind[1:]
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == PAG_ARROW or pag[a][i] == PAG_CIRCLE) and
                        (pag[i][a] == PAG_CIRCLE or pag[i][a] == PAG_TAIL) and
                        (pag[c][i] == 0 and pag[i][c] == 0) and
                        i != c]
                while len(indB) > 0 and pag[c][a] == PAG_CIRCLE:
                    b = indB[0]
                    indB = indB[1:]
                    upd = minUncovPdPath(p, pag, a, b, c, unfVect=unfVect)
                    if len(upd) > 1:
                        _bk.set_mark(pag, c, a, PAG_TAIL)
            rule_stats["R9"] += int(np.count_nonzero(pag != before_r))
        # R10
        if rules[9]:
            before_r = pag.copy()
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, c = ind[0]
                ind = ind[1:]
                indB = [i for i in range(p) if pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW]
                if len(indB) >= 2:
                    counterB = -1
                    while counterB < len(indB) - 1 and pag[c][a] == PAG_CIRCLE:
                        counterB += 1
                        b = indB[counterB]
                        indD = [i for i in indB if i != b]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[c][a] == PAG_CIRCLE:
                            counterD += 1
                            d = indD[counterD]
                            if (
                                    (pag[a][b] == PAG_CIRCLE or pag[a][b] == PAG_ARROW) and
                                    (pag[b][a] == PAG_CIRCLE or pag[b][a] == PAG_TAIL) and
                                    (pag[a][d] == PAG_CIRCLE or pag[a][d] == PAG_ARROW) and
                                    (pag[d][a] == PAG_CIRCLE or pag[d][a] == PAG_TAIL) and
                                    pag[d][b] == 0 and pag[b][d] == 0
                            ):
                                if len(unfVect) == 0:
                                    _bk.set_mark(pag, c, a, PAG_TAIL)
                                else:
                                    if (b, a, d) not in unfVect and (d, a, b) not in unfVect:
                                        _bk.set_mark(pag, c, a, PAG_TAIL)
                            else:
                                indX = [i for i in range(p) if
                                        (pag[a][i] == PAG_CIRCLE or pag[a][i] == PAG_ARROW) and
                                        (pag[i][a] == PAG_CIRCLE or pag[i][a] == PAG_TAIL) and
                                        i != c]
                                if len(indX) >= 2:
                                    counterX1 = -1
                                    while counterX1 < len(indX) - 1 and pag[c][a] == PAG_CIRCLE:
                                        counterX1 += 1
                                        # first_pos = indA[counterX1]  # indA not defined in original code properly?
                                        # В оригинале indA определен как [] в начале функции и не заполняется.
                                        # Это баг в pcalg порте?
                                        # Предположим, что indX используется.

                                        first_pos = indX[counterX1]

                                        indX2 = [i for i in indX if i != first_pos]
                                        counterX2 = -1
                                        while counterX2 < len(indX2) - 1 and pag[c][a] == PAG_CIRCLE:
                                            counterX2 += 1
                                            sec_pos = indX2[counterX2]
                                            t1 = minUncovPdPath(p, pag, a, first_pos, b, unfVect=unfVect)
                                            if len(t1) > 1:
                                                t2 = minUncovPdPath(p, pag, a, sec_pos, d, unfVect=unfVect)
                                                if len(t2) > 1 and first_pos != sec_pos and pag[first_pos][
                                                    sec_pos] == 0:
                                                    if len(unfVect) == 0:
                                                        _bk.set_mark(pag, c, a, PAG_TAIL)
                                                    elif (first_pos, a, sec_pos) not in unfVect and (
                                                            sec_pos, a, first_pos) not in unfVect:
                                                        _bk.set_mark(pag, c, a, PAG_TAIL)
            rule_stats["R10"] += int(np.count_nonzero(pag != before_r))
        apply_bk_checkpoint(
            background_knowledge,
            pag,
            node_names,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_orientation_rule_pass",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
            previous=old_pag1,
        )

    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="after_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )
    return pag
