from __future__ import annotations

from typing import Any, Callable, Dict, Type, List
from .base import DatasetLoader


class LoaderRegistry:
    _registry: Dict[str, Type[DatasetLoader]] = {}
    _aliases: Dict[str, str] = {
        "real": "feedback",
        "simulated": "feedback",
        "car-evaluation": "feedback",
    }
    @classmethod
    def register(cls, name: str, *aliases: str) -> Callable[[Type[DatasetLoader]], Type[DatasetLoader]]:
        def decorator(loader_cls: Type[DatasetLoader]) -> Type[DatasetLoader]:
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
        key = cls._resolve_name(name)
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown loader '{name}' (resolved as '{key}'). Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._registry.keys())
