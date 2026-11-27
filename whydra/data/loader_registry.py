from collections.abc import Callable
from typing import Any, ClassVar

from .base import DatasetLoader


class LoaderRegistry:
    """Map dataset names and aliases to loader classes."""

    _registry: ClassVar[dict[str, type[DatasetLoader]]] = {}
    _aliases: ClassVar[dict[str, str]] = {
        "real": "feedback",
        "simulated": "feedback",
        "car-evaluation": "feedback",
    }

    @classmethod
    def register(
        cls, name: str, *aliases: str
    ) -> Callable[[type[DatasetLoader]], type[DatasetLoader]]:
        """Register a loader class under its canonical name and optional aliases."""

        def decorator(loader_cls: type[DatasetLoader]) -> type[DatasetLoader]:
            key = name.lower()
            cls._registry[key] = loader_cls
            for a in aliases:
                cls._aliases[a.lower()] = key
            return loader_cls

        return decorator

    @classmethod
    def _resolve_name(cls, name: str) -> str:
        key = name.lower()
        return cls._aliases.get(key, key)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DatasetLoader:
        """Build the loader for a registered name or alias with the given arguments."""
        key = cls._resolve_name(name)
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            msg = (
                f"Unknown loader '{name}' (resolved as '{key}'). Available: {available}"
            )
            raise ValueError(msg)
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return the registered canonical names in sorted order."""
        return sorted(cls._registry.keys())
