"""Load the generator's bundled scenarios (specification C-2).

The YAML files live beside this module. Upstream maintainers regenerate them
with ``scripts/build_scenarios.py``. Use a YAML filename without its extension
to load a bundled scenario:

    from whydra.scmgen import scenarios
    cfg = scenarios.load("rct-clean")
    scenarios.list_scenarios()  # -> ['ctl-binary-rd', ..., 'rct-zeroinfl-smalltau']
"""

import pathlib
from importlib import resources

from ..config import ScenarioConfig


def _root() -> pathlib.Path:
    return pathlib.Path(str(resources.files(__name__)))


def list_scenarios() -> list[str]:
    """List the names of all bundled scenarios in sorted order."""
    return sorted(p.stem for p in _root().glob("*.yaml"))


def path(name: str) -> pathlib.Path:
    """Return the scenario's YAML path or raise KeyError for an unknown name.

    Version 0.6.0 removed the rct-zeroinfl-lowsnr alias.
    """
    if name not in list_scenarios():
        msg = f"unknown scenario {name!r}; available: {', '.join(list_scenarios())}"
        raise KeyError(msg)
    return _root() / f"{name}.yaml"


def load(name: str) -> ScenarioConfig:
    """Load a bundled scenario by name."""
    return ScenarioConfig.from_yaml(path(name))
