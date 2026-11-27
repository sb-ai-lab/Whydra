"""Call-throughs for search branches that actually need optional runtime tools.

Keep the usual module-level names callable and monkeypatchable without loading
joblib or notebook progress integrations for a sequential, silent search.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

if TYPE_CHECKING:
    import pandas
    from typing_extensions import Self

    DataFrame = pandas.DataFrame

_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")
_Task = tuple[Callable[..., _Result], tuple, dict]


class _ParallelRunner(Protocol):
    def __call__(self, tasks: Iterable[_Task[_Result]], /) -> Iterable[_Result]: ...
    def __enter__(self) -> "Self": ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...


def __getattr__(name: str):
    """Resolve pandas annotations when callers inspect public type hints."""
    if name == "DataFrame":
        from pandas import DataFrame

        return DataFrame
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def Parallel(*args, **kwargs) -> _ParallelRunner:
    from joblib import Parallel as implementation

    return cast("_ParallelRunner", implementation(*args, **kwargs))


def delayed(function: Callable[_Params, _Result]) -> Callable[_Params, _Task[_Result]]:
    from joblib import delayed as implementation

    return implementation(function)


def tqdm(*args, **kwargs):
    from tqdm import tqdm as implementation

    return implementation(*args, **kwargs)


def tqdm_auto(*args, **kwargs):
    from tqdm.auto import tqdm as implementation

    return implementation(*args, **kwargs)
