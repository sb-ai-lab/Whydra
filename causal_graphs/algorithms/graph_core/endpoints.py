from enum import Enum


class Endpoint(Enum):
    TAIL = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2


__all__ = ["Endpoint"]
