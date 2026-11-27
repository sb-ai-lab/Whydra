from enum import Enum


class Endpoint(Enum):
    """Integer codes for the marks in an endpoint matrix."""

    TAIL = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2


__all__ = ["Endpoint"]
