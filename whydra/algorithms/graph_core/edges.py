from .nodes import Node
from .endpoints import Endpoint


class Edge:
    def __init__(self, node1: Node, node2: Node, end1: Endpoint, end2: Endpoint):
        self.node1 = node1
        self.node2 = node2
        self.endpoint1 = end1
        self.endpoint2 = end2

    def get_node1(self) -> Node:
        return self.node1

    def get_node2(self) -> Node:
        return self.node2

    def get_endpoint1(self) -> Endpoint:
        return self.endpoint1

    def get_endpoint2(self) -> Endpoint:
        return self.endpoint2

    def __str__(self):
        marks = {
            Endpoint.TAIL: "-",
            Endpoint.ARROW: ">",
            Endpoint.CIRCLE: "o",
        }
        left = marks.get(self.endpoint1, "?")
        right = marks.get(self.endpoint2, "?")
        # Keep the familiar causal-graph spelling while using only local types.
        if left == "-" and right == ">":
            connector = "-->"
        elif left == ">" and right == "-":
            connector = "<--"
        elif left == ">" and right == ">":
            connector = "<->"
        elif left == "-" and right == "-":
            connector = "---"
        elif left == "o" and right == ">":
            connector = "o->"
        elif left == ">" and right == "o":
            connector = "<-o"
        elif left == "o" and right == "o":
            connector = "o-o"
        else:
            connector = f"{left}-{right}"
        return f"{self.node1.name} {connector} {self.node2.name}"


__all__ = ["Edge"]
