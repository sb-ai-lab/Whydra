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
        return f"{self.node1.name} {self.endpoint1.name}--{self.endpoint2.name} {self.node2.name}"


__all__ = ["Edge"]
