import logging
import os
from itertools import chain, combinations
from math import comb
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from joblib import Parallel, delayed

from ..graph_core import CausalGraph, CIT, Endpoint
from ..graph_core.independence_tests import chi_square_from_encoded, fisher_z_from_corr
from .profiler import profiler

logger = logging.getLogger(__name__)

UNDIRECTED = "---"  # метка ребра: ненаправленная «голова»
DIRECTED = "<--"    # метка ребра: направленная «голова» (стрелка в узел)
EDGE_MARKS = (UNDIRECTED, DIRECTED)  # фиксированный порядок обхода (в референсе set; порядок проверенно не важен)

# Параллельный слой: фазу отдаём воркерам, только если она достаточно велика.
PARALLEL_MIN_PAIRS = 2              # связанных пар в фазе (меньше — параллелить нечего)
PARALLEL_MIN_COST_PER_JOB = 32      # оценка стоимости фазы >= PARALLEL_MIN_COST_PER_JOB * n_jobs,
#                                     стоимость = число condition set'ов x (1 + order): тест высокого
#                                     порядка дороже теста порядка 0 примерно во столько раз
PARALLEL_CHUNKS_PER_JOB = 4         # чанков на фазу = PARALLEL_CHUNKS_PER_JOB * n_jobs
JOBLIB_MAX_NBYTES = "100K"          # массивы больше этого joblib отдаёт воркерам через memmap


class _PDAG:
    """Частично ориентированный граф (порт MixedGraph + PDAG из референса IntelLabs).

    Внутреннее представление: ``_graph[node][DIRECTED]`` — множество родителей узла,
    ``_graph[node][UNDIRECTED]`` — множество ненаправленных соседей. Все операции
    повторяют референс над теми же ``set``: от порядка итерации зависит результат.
    """

    def __init__(self, nodes_set: Set[int]):
        """Создать пустой PDAG на заданном множестве узлов.

        :param nodes_set: множество узлов графа.
        """
        self._graph: Dict[int, Dict[str, Set[int]]] = dict()
        self.nodes_set = nodes_set
        self.create_empty_graph(self.nodes_set)
        self.orientation_rules = {
            1: self.orient_by_rule_1,
            2: self.orient_by_rule_2,
            3: self.orient_by_rule_3,
            4: self.orient_by_rule_4,
        }

    def create_empty_graph(self, nodes_set: Optional[Set[int]] = None) -> None:
        """Инициализировать пустые множества рёбер для каждого узла.

        :param nodes_set: множество узлов (по умолчанию — все узлы графа).
        """
        if nodes_set is None:
            nodes_set = self.nodes_set
        for node in nodes_set:
            self._graph[node] = dict()
            for head_type in EDGE_MARKS:
                self._graph[node][head_type] = set()

    def create_complete_graph(self, nodes_set: Optional[Set[int]] = None) -> None:
        """Построить полный ненаправленный граф на заданном множестве узлов.

        :param nodes_set: множество узлов (по умолчанию — все узлы графа).
        """
        if nodes_set is None:
            nodes_set = self.nodes_set
        self.create_empty_graph(nodes_set)
        for node in nodes_set:
            self._graph[node][UNDIRECTED] = nodes_set - {node}

    def parents(self, target_node: int) -> Set[int]:
        """Вернуть множество родителей узла.

        :param target_node: целевой узел.
        :return: множество родителей.
        """
        return self._graph[target_node][DIRECTED]

    def undirected_neighbors(self, node: int) -> Set[int]:
        """Вернуть множество ненаправленных соседей узла.

        :param node: узел.
        :return: множество соседей.
        """
        return self._graph[node][UNDIRECTED]

    def fan_in(self, target_node: int) -> int:
        """Вернуть число входящих связей узла (родители + ненаправленные соседи).

        :param target_node: целевой узел.
        :return: число входящих связей.
        """
        return len(self.parents(target_node)) + len(self.undirected_neighbors(target_node))

    def is_connected(self, node_i: int, node_j: int) -> bool:
        """Проверить наличие ребра между двумя узлами (любого типа).

        :param node_i: первый узел.
        :param node_j: второй узел.
        :return: True, если узлы смежны.
        """
        assert node_i != node_j
        for (node_p, node_c) in [(node_i, node_j), (node_j, node_i)]:  # смена ролей «родитель»-«потомок»
            for edge_mark in EDGE_MARKS:
                if node_p in self._graph[node_c][edge_mark]:
                    return True
        return False

    def find_adjacent_nodes(
        self,
        node_i: int,
        pool_nodes: Optional[Set[int]] = None,
        edge_type: Optional[Tuple[str, str]] = None,
    ) -> Set[int]:
        """Найти смежные узлы, опционально ограничив пул и тип ребра.

        :param node_i: узел, для которого ищутся соседи.
        :param pool_nodes: ограничение на пул узлов (None — без ограничения).
        :param edge_type: пара меток (метка у node_i, метка у соседа) или None — любые рёбра.
        :return: множество смежных узлов.
        """
        if edge_type is None:
            connected_nodes = set()
            for edge_mark in EDGE_MARKS:
                connected_nodes.update(self._graph[node_i][edge_mark])
        else:
            mark_origin = edge_type[0]
            mark_neighbor = edge_type[1]
            connected_nodes = set(filter(
                lambda neighbor: node_i in self._graph[neighbor][mark_neighbor],
                self._graph[node_i][mark_origin]
            ))
        if pool_nodes is not None:
            connected_nodes = connected_nodes & pool_nodes
        return connected_nodes

    def find_reachable_set(
        self,
        anchor_node: int,
        nodes_pool: Set[int],
        edge_type_list: Optional[List[Tuple[str, str]]],
    ) -> Set[int]:
        """Итеративный BFS: найти множество узлов, достижимых из якоря.

        :param anchor_node: якорный узел.
        :param nodes_pool: пул допустимых узлов.
        :param edge_type_list: список допустимых типов рёбер (None — любые).
        :return: множество достижимых узлов (без якоря).
        """
        if not nodes_pool:
            return set()
        reachable_set = set()
        allowed_pool = set(nodes_pool)
        frontier = {anchor_node}
        while frontier:
            next_frontier = set()
            for node in frontier:
                if edge_type_list is None:
                    next_frontier.update(
                        self.find_adjacent_nodes(node, allowed_pool, None))
                else:
                    for edge_type in edge_type_list:
                        next_frontier.update(
                            self.find_adjacent_nodes(node, allowed_pool, edge_type))
            reachable_set.update(next_frontier)
            allowed_pool.difference_update(next_frontier)
            frontier = next_frontier
        return reachable_set

    def find_unconnected_subgraphs(
        self,
        en_nodes: Optional[Set[int]] = None,
        sym_edge_mark: Optional[str] = None,
    ) -> List[Set[int]]:
        """Разбить узлы на компоненты связности.

        :param en_nodes: множество узлов (по умолчанию — все узлы графа).
        :param sym_edge_mark: симметричная метка ребра для связности (None — любые рёбра).
        :return: список множеств узлов (компонент связности).
        """
        if en_nodes is None:
            en_nodes = self.nodes_set
        connected_sets = []
        nodes = en_nodes.copy()
        edge_type_list = None
        if sym_edge_mark in EDGE_MARKS:
            edge_type_list = [(sym_edge_mark, sym_edge_mark)]
        while len(nodes) > 0:
            node_i = nodes.pop()
            reachable_set = self.find_reachable_set(node_i, nodes, edge_type_list)
            nodes.difference_update(reachable_set)
            reachable_set.add(node_i)
            connected_sets.append(reachable_set)
        return connected_sets

    def find_partial_topological_order(self, en_nodes: Optional[Set[int]] = None) -> List[Set[int]]:
        """Рекурсивно найти частичный топологический порядок (список множеств).

        :param en_nodes: множество узлов (по умолчанию — все узлы графа).
        :return: список множеств узлов, от листьев к корням.
        """
        if en_nodes is None:
            en_nodes = self.nodes_set
        if len(en_nodes) == 0:
            return []
        parents_set = set()
        for node in en_nodes:
            parents_set.update(self.parents(node) & en_nodes)  # узлы, которые являются чьими-то родителями
        leaves_set = en_nodes - parents_set  # узлы, не являющиеся родителями эндогенных узлов
        if len(leaves_set) == 0:
            return [parents_set]  # различить топологические порядки не удалось
        else:
            high_topological_ordering = self.find_partial_topological_order(parents_set)  # рекурсивный вызов
            return [leaves_set] + high_topological_ordering

    def orient_edge(self, source_node: int, target_node: int) -> None:
        """Ориентировать ребро source_node --> target_node.

        :param source_node: источник.
        :param target_node: цель.
        """
        self._graph[target_node][DIRECTED].add(source_node)        # добавить направленную «голову»
        self._graph[target_node][UNDIRECTED].discard(source_node)  # удалить ненаправленную «голову»
        self._graph[source_node][UNDIRECTED].discard(target_node)  # удалить ненаправленную «голову»

    def delete_directed_edge(self, source_node: int, target_node: int) -> None:
        """Удалить направленное ребро source_node --> target_node.

        :param source_node: источник.
        :param target_node: цель.
        """
        self._graph[target_node][DIRECTED].discard(source_node)

    def delete_undirected_edge(self, node_i: int, node_j: int) -> None:
        """Удалить ненаправленное ребро между двумя узлами.

        :param node_i: первый узел.
        :param node_j: второй узел.
        """
        self._graph[node_i][UNDIRECTED].discard(node_j)
        self._graph[node_j][UNDIRECTED].discard(node_i)

    def delete_edge(self, node_i: int, node_j: int) -> None:
        """Удалить ребро между двумя узлами (направленное в обе стороны + ненаправленное).

        :param node_i: первый узел.
        :param node_j: второй узел.
        """
        self.delete_directed_edge(node_i, node_j)  # удалить направленную «голову» в node_j
        self.delete_directed_edge(node_j, node_i)  # удалить направленную «голову» в node_i
        self.delete_undirected_edge(node_i, node_j)  # удалить ненаправленные «головы»

    def add_edges(
        self,
        parents_set: Set[int],
        target_node: int,
        arrowhead_type: str = UNDIRECTED,
    ) -> None:
        """Добавить рёбра от множества узлов к целевому узлу с заданным типом «головы».

        :param parents_set: множество узлов-источников.
        :param target_node: целевой узел.
        :param arrowhead_type: тип «головы» (UNDIRECTED или DIRECTED).
        """
        if arrowhead_type != UNDIRECTED and arrowhead_type != DIRECTED:
            raise ValueError
        self._graph[target_node][arrowhead_type] |= parents_set
        if arrowhead_type == UNDIRECTED:
            for parent_node in parents_set:
                self._graph[parent_node][UNDIRECTED].add(target_node)  # обратное ребро

    def convert_bidirected_to_undirected(self, nodes: Optional[Set[int]] = None) -> None:
        """Заменить двунаправленные (конфликтные) рёбра на ненаправленные.

        :param nodes: множество узлов (по умолчанию — все узлы графа).
        """
        if nodes is None:
            nodes = self.nodes_set
        for node_x, node_y in combinations(nodes, 2):
            if node_y in self._graph[node_x][DIRECTED] and node_x in self._graph[node_y][DIRECTED]:
                self._graph[node_x][DIRECTED].discard(node_y)  # удалить направленные «головы»
                self._graph[node_y][DIRECTED].discard(node_x)
                self._graph[node_x][UNDIRECTED].add(node_y)    # добавить ненаправленные «головы»
                self._graph[node_y][UNDIRECTED].add(node_x)

    def maximally_orient_pattern(self, rules_set, en_nodes: Optional[Set[int]] = None) -> None:
        """Применять правила ориентации, пока граф меняется.

        :param rules_set: номера правил (1..4) в порядке применения.
        :param en_nodes: множество узлов (по умолчанию — все узлы графа).
        """
        if en_nodes is None:
            en_nodes = self.nodes_set
        graph_modified = True
        while graph_modified:
            graph_modified = False
            for rule_idx in rules_set:
                rule = self.orientation_rules[rule_idx]
                graph_modified |= rule(en_nodes)

    def orient_by_rule_1(self, en_nodes: Set[int]) -> bool:
        """Правило R1: Z --> X --- Y, Z не смежен с Y => X --> Y.

        :param en_nodes: множество узлов.
        :return: True, если граф был изменён.
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()  # соседи текущего Y
            for node_x in x_nodes:  # проверить все ненаправленные рёбра «в» Y
                for node_z in self.parents(node_x):
                    if not self.is_connected(node_z, node_y):
                        self.orient_edge(source_node=node_x, target_node=node_y)  # ориентировать X --> Y
                        graph_modified = True
                        break  # X --> Y ориентировано — перейти к следующему X --- Y
        return graph_modified

    def orient_by_rule_2(self, en_nodes: Set[int]) -> bool:
        """Правило R2: X --> Z --> Y и X --- Y => X --> Y.

        :param en_nodes: множество узлов.
        :return: True, если граф был изменён.
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            for node_x in x_nodes:
                z_nodes = self.parents(node_y)  # направленные родители
                for node_z in z_nodes:
                    if node_x in self.parents(node_z):
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_modified = True
                        break
        return graph_modified

    def orient_by_rule_3(self, en_nodes: Set[int]) -> bool:
        """Правило R3: X --- W --> Y, X --- Z --> Y, W и Z не смежны, X --- Y => X --> Y.

        :param en_nodes: множество узлов.
        :return: True, если граф был изменён.
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            wz_nodes = self.parents(node_y)
            for node_x in x_nodes:
                wz_nodes_of_x = self.undirected_neighbors(node_x).intersection(wz_nodes)  # W, Z — соседи X
                for node_w, node_z in combinations(wz_nodes_of_x, 2):
                    if self.is_connected(node_w, node_z):
                        continue  # пропустить: W и Z смежны
                    self.orient_edge(source_node=node_x, target_node=node_y)
                    graph_modified = True
                    break
        return graph_modified

    def orient_by_rule_4(self, en_nodes: Set[int]) -> bool:
        """Правило R4: W --> Z --> Y, X --- Z, X --- W, W не смежен с Y, X --- Y => X --> Y.

        :param en_nodes: множество узлов.
        :return: True, если граф был изменён.
        """
        graph_updated = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            z_nodes = self.parents(node_y)
            for node_x in x_nodes:
                for node_z in z_nodes:
                    if not self.is_connected(node_z, node_x):  # Z и X должны быть смежны
                        continue
                    w_nodes = self.parents(node_z).intersection(self.undirected_neighbors(node_x))
                    if len(w_nodes) > 0:
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_updated = True
                        break
        return graph_updated


class _SeparationSet:
    """Хранилище сепарирующих множеств для неупорядоченных пар узлов (порт SeparationSet)."""

    def __init__(self, nodes_set: Set[int]):
        """Создать пустые сепсеты для всех неупорядоченных пар узлов.

        :param nodes_set: множество узлов.
        """
        self._sepset = dict()
        self.nodes_set = nodes_set
        for (i, j) in combinations(nodes_set, 2):
            hkey = self.get_hash_key(i, j)
            self._sepset[hkey] = set()

    @staticmethod
    def get_hash_key(node_1: int, node_2: int) -> Tuple[int, int]:
        """Вернуть канонический ключ пары узлов (меньший узел первым).

        :param node_1: первый узел.
        :param node_2: второй узел.
        :return: ключ пары.
        """
        hkey = (node_1, node_2) if node_1 < node_2 else (node_2, node_1)
        return hkey

    def set_sepset(self, node_1: int, node_2: int, sepset: Sequence[int]) -> None:
        """Перезаписать сепсет пары узлов.

        :param node_1: первый узел.
        :param node_2: второй узел.
        :param sepset: сепарирующее множество.
        """
        hkey = self.get_hash_key(node_1, node_2)
        self._sepset[hkey] = set(sepset)

    def get_sepset(self, node_1: int, node_2: int) -> Set[int]:
        """Вернуть сепсет пары узлов.

        :param node_1: первый узел.
        :param node_2: второй узел.
        :return: сепарирующее множество.
        """
        hkey = self.get_hash_key(node_1, node_2)
        return self._sepset[hkey]


def _unique_element_iterator(iterable: Iterable[Sequence[int]]) -> Iterable[Tuple[int, ...]]:
    """Итератор, отдающий каждое condition set один раз (ключ — отсортированный кортеж).

    :param iterable: входная последовательность кортежей.
    :return: генератор уникальных кортежей в исходном порядке.
    """
    seen: Set[Tuple[int, ...]] = set()
    for item in iterable:
        key = tuple(sorted(item))
        if key in seen:
            continue
        seen.add(key)
        yield tuple(item)


class _CIOracle:
    """Обёртка над CI-тестом: отвечает на вопрос об условной независимости пары узлов."""

    def __init__(self, tester: CIT, alpha: float):
        """Сохранить тестер независимости и уровень значимости.

        :param tester: CI-тест, вызываемый как tester(x, y, cond_set) -> p-value.
        :param alpha: уровень значимости.
        """
        self.tester = tester
        self.alpha = alpha

    def cond_indep(self, x: int, y: int, cond_set: Sequence[int]) -> bool:
        """Проверить условную независимость x и y при данном множестве.

        Множество S передаётся в тест в отсортированном виде: так p-value не зависит
        от порядка, в котором S был перечислен.

        :param x: первый узел.
        :param y: второй узел.
        :param cond_set: кондиционирующее множество.
        :return: True, если p-value превышает alpha (независимы).
        """
        p_val = self.tester(x, y, tuple(sorted(cond_set)))
        return p_val > self.alpha


def _pot_parents(graph: _PDAG, node: int, other: int) -> Set[int]:
    """Потенциальные родители узла (родители | ненаправленные соседи) без другого узла.

    :param graph: PDAG.
    :param node: целевой узел.
    :param other: исключаемый узел (второй конец проверяемого ребра).
    :return: множество потенциальных родителей.
    """
    return (graph.parents(node) | graph.undirected_neighbors(node)) - {other}


def _cond_sets(
    pot_parents_a: Set[int],
    pot_parents_b: Set[int],
    order: int,
) -> Iterable[Tuple[int, ...]]:
    """Итератор condition set'ов порядка order из двух пулов (без повторов).

    :param pot_parents_a: первый пул потенциальных родителей.
    :param pot_parents_b: второй пул потенциальных родителей.
    :param order: размер кондиционирующего множества.
    :return: генератор уникальных кортежей.
    """
    return _unique_element_iterator(chain(combinations(pot_parents_a, order), combinations(pot_parents_b, order)))


def _make_tester(
    method: str,
    arrays: Tuple[np.ndarray, ...],
    n_samples: int,
) -> Callable[[int, int, Sequence[int]], float]:
    """Построить функцию p-value из закодированных массивов CI-теста.

    :param method: ``chisq`` или ``fisherz``.
    :param arrays: chisq — ``(data_int, cardinalities)``; fisherz — ``(corr_matrix,)``.
    :param n_samples: число наблюдений.
    :return: функция ``(x, y, S) -> p-value``.
    """
    if method == "chisq":
        data_int, cardinalities = arrays
        return lambda x, y, S: chi_square_from_encoded(data_int, cardinalities, n_samples, x, y, S)
    if method == "fisherz":
        (corr_matrix,) = arrays
        return lambda x, y, S: fisher_z_from_corr(corr_matrix, n_samples, x, y, S)
    raise ValueError(f"Неизвестный метод CI-теста: {method}")


def _ci_worker(
    method: str,
    arrays: Tuple[np.ndarray, ...],
    n_samples: int,
    alpha: float,
    order: int,
    jobs: List[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]],
) -> List[Tuple[int, int, Tuple[int, ...], float]]:
    """Спекулятивные CI-тесты одного чанка пар: каждая пара до первого p > alpha.

    Условные множества берутся из снимка графа на начало фазы, поэтому это
    только предвычисление p-value; решения принимает главный процесс. Тестер
    собирается на каждый чанк заново (это замыкание над memmap-массивами, цена
    нулевая); держать его в кэше воркера нельзя — тогда memmap не освобождается
    и joblib секунду ждёт очистки временной папки на выходе из ``Parallel``.

    :param method: ``chisq`` или ``fisherz``.
    :param arrays: массивы CI-теста.
    :param n_samples: число наблюдений.
    :param alpha: уровень значимости (только для ранней остановки).
    :param order: размер кондиционирующего множества.
    :param jobs: список ``(x, y, pot_parents_x, pot_parents_y)``.
    :return: плоский список ``(x, y, sorted_cset, p_value)``.
    """
    tester = _make_tester(method, arrays, n_samples)
    out: List[Tuple[int, int, Tuple[int, ...], float]] = []
    for x, y, pot_x, pot_y in jobs:
        for cset in _cond_sets(pot_x, pot_y, order):
            key = tuple(sorted(cset))
            p = tester(x, y, key)
            out.append((x, y, key, p))
            if p > alpha:
                break
    return out


class _SpeculativeMemo:
    """Лениво читает генератор результатов joblib (чанки в порядке подачи) и отдаёт p-value."""

    def __init__(
        self,
        results: Iterator[List[Tuple[int, int, Tuple[int, ...], float]]],
        chunk_of_pair: List[int],
    ):
        """Создать memo над генератором результатов.

        :param results: генератор joblib, по одному списку на чанк.
        :param chunk_of_pair: ``chunk_of_pair[i]`` — индекс чанка с парой номер i (-1: пара не была связана).
        """
        self._results = results
        self._chunk_of_pair = chunk_of_pair
        self._loaded = 0
        self._memo: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
        self.n_worker_tests = 0

    def _load_chunk(self, chunk: List[Tuple[int, int, Tuple[int, ...], float]]) -> None:
        """Положить результаты одного чанка в memo.

        :param chunk: список ``(x, y, sorted_cset, p_value)``.
        """
        for x, y, key, p in chunk:
            self._memo[(x, y, key)] = p
            self.n_worker_tests += 1
        self._loaded += 1

    def ensure_pair(self, pair_idx: int) -> None:
        """Дочитать чанки, пока чанк пары ``pair_idx`` не окажется в memo.

        :param pair_idx: номер пары в порядке фазы.
        """
        target = self._chunk_of_pair[pair_idx]
        if target == -1:
            return
        while self._loaded <= target:
            self._load_chunk(next(self._results))

    def get(self, x: int, y: int, sorted_cset: Tuple[int, ...]) -> Optional[float]:
        """Вернуть p-value или None (промах).

        :param x: первая переменная (в порядке вызова теста).
        :param y: вторая переменная.
        :param sorted_cset: отсортированное кондиционирующее множество.
        :return: p-value или None.
        """
        return self._memo.get((x, y, sorted_cset))

    def drain(self) -> None:
        """Дочитать все оставшиеся чанки (обязательно до следующего вызова Parallel)."""
        for chunk in self._results:
            self._load_chunk(chunk)


class RAIStableLearner:
    """Порт LearnStructRAI (IntelLabs causality-lab) на примитивы whydra.

    Последовательный режим повторяет референс побитово. Параллельный слой
    (``n_jobs > 1``) ускоряет только CI-тесты и не меняет результат.
    """

    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        node_names: Sequence[str],
        verbose: bool = False,
    ):
        """Инициализировать RAI.

        :param data: матрица данных (n_samples x n_vars).
        :param alpha: уровень значимости CI-теста.
        :param indep_test: название CI-теста (``fisherz`` / ``chisq``).
        :param node_names: имена переменных.
        :param verbose: логировать статистику фаз.
        """
        self.n_vars = data.shape[1]
        self.node_names = list(node_names)
        self.ci_test = CIT(data, method=indep_test)
        self.oracle = _CIOracle(self.ci_test, alpha)
        self.nodes_set = set(range(self.n_vars))
        self.graph = _PDAG(self.nodes_set)
        self.graph.create_complete_graph(self.nodes_set)
        self.sepset = _SeparationSet(self.nodes_set)
        self.verbose = verbose
        self.n_ci_tests = 0
        self.n_jobs = 1
        self._parallel = None
        #: Список решений ``(x, y, sorted_cset, independent)``, если вызывающий подставил list (для тестов).
        self.trace: Optional[List[Tuple[int, int, Tuple[int, ...], bool]]] = None
        self.stats = {"phases": 0, "parallel_phases": 0, "ci_total": 0, "ci_worker": 0, "ci_sync": 0}

    def learn_structure(self, n_jobs: int = 1):
        """Запустить обучение структуры RAI и вернуть граф.

        При ``n_jobs > 1`` CI-тесты фаз считаются воркерами joblib наперёд
        («спекулятивно»), а решения принимает последовательный проход, поэтому
        результат от ``n_jobs`` не зависит. При ``n_jobs <= 1`` joblib не создаётся.

        :param n_jobs: число процессов для CI-тестов.
        :return: GeneralGraph whydra (CPDAG).
        """
        self.n_jobs = n_jobs
        if n_jobs > 1:
            with Parallel(
                n_jobs=n_jobs,
                backend="loky",
                max_nbytes=JOBLIB_MAX_NBYTES,
                mmap_mode="r",
                return_as="generator",
                pre_dispatch="all",
            ) as parallel:
                self._parallel = parallel
                try:
                    self._learn()
                finally:
                    self._parallel = None
        else:
            self._parallel = None
            self._learn()
        return self._to_general_graph()

    def _learn(self) -> None:
        """Тело алгоритма: рекурсия, финальные R1–R3, пере-ориентация скелета."""
        en_nodes = self.nodes_set
        ex_nodes = set()
        with profiler.time_block("standalone_rai_total", tags={"algo": "rai"}):
            self._learn_recursively(en_nodes=en_nodes, ex_nodes=ex_nodes, order=0)
            self.graph.maximally_orient_pattern((1, 2, 3))
            self._re_orient_skeleton()

    def _worker_arrays(self) -> Tuple[str, Tuple[np.ndarray, ...], int]:
        """Массивы CI-теста для воркеров: всегда одни и те же объекты ndarray.

        joblib дампит массив в memmap один раз на контекст ``Parallel``; новая копия
        на каждый вызов дампилась бы заново.

        :return: ``(method, arrays, n_samples)``.
        """
        method = self.ci_test.method
        if method == "chisq":
            return method, (self.ci_test.data_int, self.ci_test.cardinalities), self.ci_test.n_samples
        if method == "fisherz":
            return method, (self.ci_test.corr_matrix,), self.ci_test.n_samples
        raise ValueError(f"Неизвестный метод CI-теста: {method}")

    def _re_orient_skeleton(self) -> None:
        """Сбросить направления и заново ориентировать скелет: v-структуры по итоговым родителям + R1–R3."""
        cpdag_final = _PDAG(self.graph.nodes_set)
        for node in self.graph.nodes_set:
            connected_nodes_set = self.graph.parents(node) | self.graph.undirected_neighbors(node)
            cpdag_final.add_edges(parents_set=connected_nodes_set, target_node=node, arrowhead_type=UNDIRECTED)

        for node in self.graph.nodes_set:
            parents_set = self.graph.parents(node)
            for (parent_i, parent_j) in combinations(parents_set, 2):
                if not self.graph.is_connected(parent_i, parent_j):
                    cpdag_final.orient_edge(source_node=parent_i, target_node=node)  # ориентировать v-структуру
                    cpdag_final.orient_edge(source_node=parent_j, target_node=node)

        cpdag_final.maximally_orient_pattern((1, 2, 3))  # правила ориентации R1, R2, R3
        self.graph = cpdag_final

    def _exit_cond(self, en_nodes: Set[int], order: int) -> bool:
        """Условие выхода из рекурсии: ни у одного узла fan-in не больше order.

        :param en_nodes: множество эндогенных узлов.
        :param order: текущий порядок CI-тестов.
        :return: True, если рекурсию можно остановить.
        """
        for node in en_nodes:
            if self.graph.fan_in(node) > order:
                return False
        else:
            return True

    def _learn_recursively(self, en_nodes: Set[int], ex_nodes: Set[int], order: int) -> None:
        """Рекурсивный шаг RAI: уточнить и ориентировать, разбить на предков/потомков, рекурсия.

        :param en_nodes: эндогенные узлы.
        :param ex_nodes: экзогенные узлы.
        :param order: текущий порядок CI-тестов.
        """
        if self._exit_cond(en_nodes, order):
            return
        self._refine_and_orient(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order)
        d_nodes, list_of_ancestors_sets, a_nodes = self._split_ancestors_descendant(en_nodes=en_nodes)
        for ancestor_set in list_of_ancestors_sets:
            self._learn_recursively(en_nodes=ancestor_set, ex_nodes=ex_nodes, order=order + 1)
        self._learn_recursively(en_nodes=d_nodes, ex_nodes=a_nodes | ex_nodes, order=order + 1)

    def _refine_and_orient(self, en_nodes: Set[int], ex_nodes: Set[int], order: int) -> None:
        """Экзогенная фаза → ориентация → эндогенная фаза → ориентация.

        :param en_nodes: эндогенные узлы.
        :param ex_nodes: экзогенные узлы.
        :param order: текущий порядок CI-тестов.
        """
        self._refine_exogenous_effect(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order)
        self.maximally_orient_edges(en_nodes=en_nodes)
        self._refine_endogenous(en_nodes=en_nodes, order=order)
        self.maximally_orient_edges(en_nodes=en_nodes)

    def _split_ancestors_descendant(self, en_nodes: Set[int]) -> Tuple[Set[int], List[Set[int]], Set[int]]:
        """Разбить узлы на потомков и список несвязных множеств предков.

        :param en_nodes: эндогенные узлы.
        :return: (d_nodes, list_of_ancestors_sets, a_nodes).
        """
        d_nodes = self._get_lowest_topological_set(en_nodes)
        a_nodes = en_nodes - d_nodes  # множества узлов, которые нужно разделить
        list_of_ancestors_sets = self.graph.find_unconnected_subgraphs(a_nodes)  # несвязные множества предков
        return d_nodes, list_of_ancestors_sets, a_nodes

    def _refine_exogenous_effect(self, en_nodes: Set[int], ex_nodes: Set[int], order: int) -> None:
        """Проверить рёбра «экзогенный — эндогенный» CI-тестами порядка order и удалить лишние.

        Пары перебираются как в референсе: ``for node in en_nodes: for ex in ex_nodes``,
        тест вызывается для ``(ex, node)``.

        :param en_nodes: эндогенные узлы.
        :param ex_nodes: экзогенные узлы.
        :param order: текущий порядок CI-тестов.
        """
        pairs = [(ex, node) for node in en_nodes for ex in ex_nodes]
        self._refine_pairs(pairs, order, phase="exogenous", n_en=len(en_nodes), n_ex=len(ex_nodes))

    def _refine_endogenous(self, en_nodes: Set[int], order: int) -> None:
        """Проверить рёбра между эндогенными узлами CI-тестами порядка order и удалить лишние.

        :param en_nodes: эндогенные узлы.
        :param order: текущий порядок CI-тестов.
        """
        pairs = list(combinations(en_nodes, 2))
        self._refine_pairs(pairs, order, phase="endogenous", n_en=len(en_nodes), n_ex=0)

    def _refine_pairs(
        self,
        pairs: List[Tuple[int, int]],
        order: int,
        phase: str,
        n_en: int,
        n_ex: int,
    ) -> None:
        """Одна фаза уточнения — последовательный цикл референса над живым графом.

        В параллельном режиме перед циклом воркеры считают p-value по снимку
        графа (``_speculate``); цикл берёт их из memo, а промахи считает сам.
        Все решения об удалении рёбер принимаются здесь, в порядке референса.

        :param pairs: пары в порядке фазы (аргументы теста в этом же порядке).
        :param order: размер кондиционирующего множества.
        :param phase: имя фазы для логов.
        :param n_en: число эндогенных узлов (для логов).
        :param n_ex: число экзогенных узлов (для логов).
        """
        memo = self._speculate(pairs, order) if self._parallel is not None else None
        n_start = self.n_ci_tests
        n_sync_start = self.stats["ci_sync"]
        for pair_idx, (a, b) in enumerate(pairs):
            if not self.graph.is_connected(a, b):
                continue
            pot_a = _pot_parents(self.graph, a, b)
            pot_b = _pot_parents(self.graph, b, a)
            if memo is not None:
                memo.ensure_pair(pair_idx)
            for cset in _cond_sets(pot_a, pot_b, order):
                if self._ci_memo(a, b, cset, memo):
                    self.graph.delete_edge(a, b)
                    self.sepset.set_sepset(a, b, cset)
                    break
        if memo is not None:
            memo.drain()
            self.stats["ci_worker"] += memo.n_worker_tests
            self.stats["parallel_phases"] += 1
        self.stats["phases"] += 1
        if self.verbose:
            logger.info(
                f"RAI phase={phase} order={order} en_nodes={n_en} ex_nodes={n_ex} "
                f"pairs={len(pairs)} ci_tests={self.n_ci_tests - n_start} "
                f"ci_sync={self.stats['ci_sync'] - n_sync_start} parallel={memo is not None}"
            )

    def _ci_memo(
        self,
        a: int,
        b: int,
        cset: Sequence[int],
        memo: Optional[_SpeculativeMemo],
    ) -> bool:
        """Решение об условной независимости: p-value из memo или синхронный тест.

        Выражение решения одно для обоих путей: ``ci_test(a, b, sorted(S)) > alpha``.

        :param a: первая переменная (в порядке вызова теста).
        :param b: вторая переменная.
        :param cset: кондиционирующее множество (исходный кортеж).
        :param memo: memo спекулятивных p-value или None.
        :return: True, если переменные условно независимы.
        """
        key = tuple(sorted(cset))
        self.n_ci_tests += 1
        self.stats["ci_total"] += 1
        p = memo.get(a, b, key) if memo is not None else None
        if p is None:
            p = self.ci_test(a, b, key)
            if memo is not None:
                self.stats["ci_sync"] += 1
        independent = p > self.oracle.alpha
        if self.trace is not None:
            self.trace.append((a, b, key, independent))
        return independent

    def _speculate(self, pairs: List[Tuple[int, int]], order: int) -> Optional[_SpeculativeMemo]:
        """Отдать воркерам спекулятивные CI-тесты фазы; None, если фаза слишком мала.

        Снимок ``pot_parents`` каждой связанной пары берётся в живом порядке
        итерации множеств, чанки режутся в порядке пар (нужно пайплайну replay).

        :param pairs: пары в порядке фазы.
        :param order: размер кондиционирующего множества.
        :return: ``_SpeculativeMemo`` или None.
        """
        graph = self.graph
        n_jobs = self.n_jobs
        jobs = []
        chunk_of_pair = [-1] * len(pairs)
        costs = []
        for idx, (a, b) in enumerate(pairs):
            if not graph.is_connected(a, b):
                continue
            if order == 0:
                pot_a = pot_b = ()  # единственный condition set — пустой; соседей воркеру не шлём
            else:
                pot_a = tuple(_pot_parents(graph, a, b))
                pot_b = tuple(_pot_parents(graph, b, a))
            jobs.append((idx, a, b, pot_a, pot_b))
            costs.append((comb(len(pot_a), order) + comb(len(pot_b), order)) * (1 + order))
        if len(jobs) < PARALLEL_MIN_PAIRS or sum(costs) < PARALLEL_MIN_COST_PER_JOB * n_jobs:
            return None
        n_chunks = min(len(jobs), PARALLEL_CHUNKS_PER_JOB * n_jobs)
        threshold = sum(costs) / n_chunks
        buckets: List[List[Tuple[int, int, int, Tuple[int, ...], Tuple[int, ...]]]] = [[] for _ in range(n_chunks)]
        chunk_idx = 0
        acc = 0
        for job, cost in zip(jobs, costs):
            buckets[chunk_idx].append(job)
            chunk_of_pair[job[0]] = chunk_idx
            acc += cost
            if chunk_idx < n_chunks - 1 and acc >= threshold:
                chunk_idx += 1
                acc = 0
        chunks = [bucket for bucket in buckets if bucket]  # пустыми бывают только хвостовые чанки
        method, arrays, n_samples = self._worker_arrays()
        results = self._parallel(
            delayed(_ci_worker)(
                method,
                arrays,
                n_samples,
                self.oracle.alpha,
                order,
                [(a, b, pot_a, pot_b) for (_, a, b, pot_a, pot_b) in chunk],
            )
            for chunk in chunks
        )
        return _SpeculativeMemo(results, chunk_of_pair)

    def _get_lowest_topological_set(self, en_nodes: Set[int]) -> Set[int]:
        """Множество узлов низшего топологического порядка (потомки).

        :param en_nodes: эндогенные узлы.
        :return: множество узлов-потомков.
        """
        ordered_sets = self.graph.find_partial_topological_order(en_nodes)
        if len(ordered_sets) > 1:  # два или более слоя
            return en_nodes - ordered_sets[-1]  # все узлы, не входящие в верхний слой
        else:
            return ordered_sets[0]  # один слой (разделения не найдено)

    def maximally_orient_edges(self, en_nodes: Set[int]) -> None:
        """Ориентировать v-структуры, применить R1–R4, конфликтные рёбра сделать ненаправленными.

        :param en_nodes: эндогенные узлы.
        """
        self.orient_v_structures(en_nodes)  # [R0]: ориентировать v-структуры
        self.graph.maximally_orient_pattern([1, 2, 3, 4])
        self.graph.convert_bidirected_to_undirected(en_nodes)  # двунаправленные (ложные) — как ненаправленные

    def orient_v_structures(self, en_nodes: Set[int]) -> None:
        """Ориентировать v-структуры по сепсетам (два прохода, как в референсе).

        :param en_nodes: эндогенные узлы.
        """
        pre_neighbors = dict()
        pre_parents = dict()
        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(node).copy()  # соседи до изменения графа

        for node_z in en_nodes:
            xy_nodes = pre_neighbors[node_z]  # ненаправленные соседи
            for node_x, node_y in combinations(xy_nodes, 2):
                if self.graph.is_connected(node_x, node_y):
                    continue  # пропустить пару: узлы смежны
                if node_z not in self.sepset.get_sepset(node_x, node_y):
                    self.graph.orient_edge(source_node=node_x, target_node=node_z)  # ориентировать X --> Z
                    self.graph.orient_edge(source_node=node_y, target_node=node_z)  # ориентировать Y --> Z

        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(node).copy()
            pre_parents[node] = self.graph.parents(node).copy()

        for node_z in en_nodes:
            parents_z = pre_parents[node_z]  # родители до модификации графа
            neighbors_z = pre_neighbors[node_z]  # оставшиеся ненаправленные рёбра X --- Z
            for node_x in neighbors_z:
                for node_y in parents_z:
                    if self.graph.is_connected(node_x, node_y):
                        continue
                    if node_z not in self.sepset.get_sepset(node_x, node_y):
                        self.graph.orient_edge(source_node=node_x, target_node=node_z)  # ориентировать (только) X --> Z

    def _to_general_graph(self):
        """Преобразовать внутренний PDAG в GeneralGraph whydra.

        :return: GeneralGraph с endpoint-матрицей (TAIL=-1, ARROW=1).
        """
        cg = CausalGraph(self.n_vars, self.node_names)
        cg.G.graph[:] = 0
        for node in self.graph.nodes_set:
            for parent in self.graph.parents(node):  # parent --> node
                cg.G.graph[parent, node] = Endpoint.ARROW.value
                cg.G.graph[node, parent] = Endpoint.TAIL.value
            for nb in self.graph.undirected_neighbors(node):  # nb --- node
                cg.G.graph[nb, node] = Endpoint.TAIL.value
                cg.G.graph[node, nb] = Endpoint.TAIL.value
        return cg.G


class ResultWrapper:
    """Минимальная обёртка результата с атрибутом G (GeneralGraph)."""

    def __init__(self, graph):
        """Сохранить граф.

        :param graph: итоговый GeneralGraph.
        """
        self.G = graph


def rai_stable(
    data: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    node_names: Optional[List[str]] = None,
    n_jobs: int = -1,
    verbose: bool = False,
    **kwargs,
):
    """Запустить RAI (recursive autonomy identification), порт референса IntelLabs.

    Фоновые знания (``background_knowledge``) RAI не поддерживает: этот и прочие
    неизвестные kwargs молча игнорируются.

    :param data: матрица данных (n_samples x n_vars).
    :param alpha: уровень значимости CI-теста.
    :param indep_test: название CI-теста.
    :param node_names: имена переменных (None — X1..Xn).
    :param n_jobs: число процессов для CI-тестов (отрицательное — все ядра CPU).
    :param verbose: логировать статистику фаз.
    :param kwargs: обратная совместимость (``n_procs``) и игнорируемые параметры.
    :return: ResultWrapper с итоговым графом в атрибуте G.
    """
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
    if "n_procs" in kwargs:
        n_jobs = kwargs["n_procs"]
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    learner = RAIStableLearner(
        data=data,
        alpha=alpha,
        indep_test=indep_test,
        node_names=node_names,
        verbose=verbose,
    )
    final_graph = learner.learn_structure(n_jobs=n_jobs)
    return ResultWrapper(final_graph)
