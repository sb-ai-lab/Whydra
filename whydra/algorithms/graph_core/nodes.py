class Node:
    """A graph node identified by its name."""

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        """Return the node's name."""
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


__all__ = ["Node"]
