"""Attach the timings from the latest run to a callable."""

from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar, cast

_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class _TimedFunction(Protocol[_P, _R_co]):
    last_times: dict[str, Any]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...


def records_timings(function: Callable[_P, _R_co]) -> _TimedFunction[_P, _R_co]:
    timed = cast("_TimedFunction[_P, _R_co]", function)
    timed.last_times = {}
    return timed
