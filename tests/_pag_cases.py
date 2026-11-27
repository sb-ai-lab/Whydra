from dataclasses import dataclass, field

import numpy as np

CPU_MARKS = {"o": 2, "<": 1, ">": 1, "-": -1}
CUDA_MARKS = {"o": 1, "<": 2, ">": 2, "-": 3}


def edge_matrix(size, edges, marks, graph=None):
    matrix = np.zeros((size, size), dtype=np.int8) if graph is None else graph.copy()
    for edge in edges.split(","):
        left, endpoints, right = edge.split()
        x, y = int(left), int(right)
        matrix[y, x] = marks[endpoints[0]]
        matrix[x, y] = marks[endpoints[-1]]
    return matrix


@dataclass
class OrientationCase:
    name: str
    rule: int
    size: int
    edges: str
    changed_edges: str = ""
    sepset: dict = field(default_factory=dict)
    ambiguous: tuple = ()

    def matrices(self, marks):
        before = edge_matrix(self.size, self.edges, marks)
        if self.changed_edges:
            after = edge_matrix(self.size, self.changed_edges, marks, before)
        else:
            after = before.copy()
        return before, after


DISCRIMINATING = "0 --> 1, 1 <-> 2, 2 <-o 3, 1 --> 4, 2 --> 4, 3 o-o 4"
CIRCLE_CYCLE = "0 o-o 1, 1 o-o 2, 2 o-o 3, 3 o-o 4, 4 o-o 0"
DIRECTED_ROUTE = "0 o-> 3, 0 --> 1, 1 --> 2, 2 --> 3"
TWO_ROUTES = (
    "0 o-> 7, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 7, 0 --> 4, 4 --> 5, 5 --> 6, 6 --> 7"
)


CASES = (
    OrientationCase("r1-unshielded", 1, 3, "0 --> 1, 1 o-o 2", "1 --> 2"),
    OrientationCase("r1-ambiguous", 1, 3, "0 --> 1, 1 o-o 2", ambiguous=((0, 1, 2),)),
    OrientationCase("r1-shielded", 1, 3, "0 --> 1, 1 o-o 2, 0 o-o 2"),
    OrientationCase("r2-directed-chain", 2, 3, "0 --> 1, 1 --> 2, 0 o-o 2", "0 o-> 2"),
    OrientationCase("r2-missing-tail", 2, 3, "0 o-> 1, 1 o-> 2, 0 o-o 2"),
    OrientationCase(
        "r3-unshielded-parents",
        3,
        4,
        "0 --> 2, 1 --> 2, 0 o-o 3, 1 o-o 3, 3 o-o 2",
        "3 o-> 2",
    ),
    OrientationCase(
        "r3-shielded-parents",
        3,
        4,
        "0 --> 2, 1 --> 2, 0 o-o 3, 1 o-o 3, 3 o-o 2, 0 --- 1",
    ),
    OrientationCase("r4-collider", 4, 5, DISCRIMINATING, "2 <-> 3, 3 <-> 4"),
    OrientationCase(
        "r4-noncollider", 4, 5, DISCRIMINATING, "3 --> 4", sepset={(0, 4): {3}}
    ),
    OrientationCase("r4-no-witness", 4, 5, DISCRIMINATING + ", 0 --> 4"),
    OrientationCase(
        "r5-uncovered-cycle",
        5,
        5,
        CIRCLE_CYCLE,
        "0 --- 1, 1 --- 2, 2 --- 3, 3 --- 4, 4 --- 0",
    ),
    OrientationCase("r5-ambiguous-cycle", 5, 5, CIRCLE_CYCLE, ambiguous=((1, 2, 3),)),
    OrientationCase("r5-triangle", 5, 3, "0 o-o 1, 1 o-o 2, 2 o-o 0"),
    OrientationCase("r6-undirected-edge", 6, 3, "0 --- 1, 1 o-o 2", "1 --o 2"),
    OrientationCase("r7-unshielded", 7, 3, "0 --o 1, 1 o-o 2", "1 --o 2"),
    OrientationCase("r7-shielded", 7, 3, "0 --o 1, 1 o-o 2, 0 o-o 2"),
    OrientationCase("r8-directed-chain", 8, 3, "0 --> 1, 1 --> 2, 0 o-> 2", "0 --> 2"),
    OrientationCase("r9-uncovered-route", 9, 4, DIRECTED_ROUTE, "0 --> 3"),
    OrientationCase("r9-shielded-first-hop", 9, 4, DIRECTED_ROUTE + ", 1 --> 3"),
    OrientationCase(
        "r9-reversed-middle-arrow", 9, 4, "0 o-> 3, 0 --> 1, 2 --> 1, 2 --> 3"
    ),
    OrientationCase("r10-distinct-first-hops", 10, 8, TWO_ROUTES, "0 --> 7"),
    OrientationCase(
        "r10-ambiguous-first-hops", 10, 8, TWO_ROUTES, ambiguous=((1, 0, 4),)
    ),
    OrientationCase("r10-adjacent-first-hops", 10, 8, TWO_ROUTES + ", 1 --- 4"),
    OrientationCase(
        "r10-two-direct-paths",
        10,
        8,
        "0 o-> 7, 0 --> 3, 0 --> 6, 3 --> 7, 6 --> 7",
        "0 --> 7",
    ),
    OrientationCase(
        "r10-ambiguous-direct-pair-with-long-paths",
        10,
        8,
        TWO_ROUTES + ", 0 --> 3, 0 --> 6",
        ambiguous=((3, 0, 6),),
    ),
    OrientationCase(
        "r10-another-parent-pair-still-has-witnesses",
        10,
        11,
        TWO_ROUTES + ", 0 --> 3, 0 --> 6, 0 --> 8, 8 --> 9, 9 --> 10, 10 --> 7",
        "0 --> 7",
        ambiguous=((3, 0, 6),),
    ),
    OrientationCase(
        "r10-one-direct-and-one-long-path",
        10,
        8,
        "0 o-> 7, 0 --> 3, 3 --> 7, 0 --> 4, 4 --> 5, 5 --> 6, 6 --> 7",
    ),
    OrientationCase(
        "r10-same-first-hop",
        10,
        8,
        "0 o-> 7, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 7, 1 --> 4, 4 --> 5, 5 --> 6, 6 --> 7",
    ),
    OrientationCase(
        "r10-no-witness-for-second-parent",
        10,
        8,
        "0 o-> 7, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 7, 0 --> 4, 4 --> 5, 6 --> 7",
    ),
    # The witness found via the earlier hop (36) reaches parent 35, whose bit
    # must be read from the second word when the witness via hop 37 is found.
    OrientationCase(
        "r10-parent-mask-spans-two-words",
        10,
        41,
        ", ".join(
            [
                "0 o-> 40",
                "0 --> 36",
                "36 --> 38",
                "38 --> 35",
                "0 --> 37",
                "37 --> 39",
                "39 --> 1",
            ]
            + [f"{parent} --> 40" for parent in range(1, 36)]
        ),
        "0 --> 40",
    ),
    # Orienting 0 o-> 7 makes 0 a parent of 7 for the next candidate 3 o-> 7.
    OrientationCase(
        "r10-new-parent-in-same-pass",
        10,
        8,
        "0 o-> 7, 3 o-> 7, 1 --> 7, 2 --> 7, 4 --> 7, 0 --> 1, 0 --> 2, 3 --> 0, 3 --> 4",
        "0 --> 7, 3 --> 7",
    ),
)


PACKED_PAIRS = {
    1: [(0, 2)],
    2: [(2, 0), (2, 1), (1, 2)],
    3: [(2, 1), (0, 2), (1, 2)],
    5: [(2, 1), (1, 2)],
    8: [(0, 2)],
}


def check_cuda_rules(*rules):
    from whydra.algorithms.graph_core import cuda_udag2pag
    from whydra.algorithms.graph_core.cuda_pdsep import SepsetStore, sepset_dict_to_csr

    for case in CASES:
        if case.rule not in rules:
            continue
        graph, expected = case.matrices(CUDA_MARKS)
        apply = getattr(cuda_udag2pag, f"apply_r{case.rule}")
        if case.rule == 4:
            apply(graph, SepsetStore(*sepset_dict_to_csr(case.sepset, case.size)))
        elif case.ambiguous:
            apply(graph, unfVect=case.ambiguous)
        else:
            apply(graph)
        np.testing.assert_array_equal(graph, expected, err_msg=case.name)

    for rule in rules:
        if rule in PACKED_PAIRS:
            block = edge_matrix(3, "0 o-> 2, 1 o-o 2", CUDA_MARKS)
            pack = getattr(cuda_udag2pag, f"pack_r{rule}")
            for copies in (1, 257):
                graph = np.kron(np.eye(copies, dtype=np.int8), block)
                expected = [
                    (x + 3 * i, y + 3 * i)
                    for i in range(copies)
                    for x, y in PACKED_PAIRS[rule]
                ]
                np.testing.assert_array_equal(pack(graph), expected)
