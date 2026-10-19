"""Scene registry.

Each scene is a module exposing:
  NAME, LABEL             -- identifiers for the scene-select menu
  PARAM_SCHEMA            -- list of control descriptors (see base.param)
  defaults()             -- dict of default parameter values
  build(params) -> Built  -- the physical layer: recording grid, depth axis,
                             complex field per frequency, emitter, geometry

The HoloEngine holds the shared scan-grid + reconstruction machinery and is
agnostic to which scene produced the fields.
"""

from . import cross, room

REGISTRY = {m.NAME: m for m in (cross, room)}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown scene {name!r}; have {list(REGISTRY)}")
    return REGISTRY[name]


def listing() -> list[dict]:
    return [{"name": m.NAME, "label": m.LABEL} for m in REGISTRY.values()]
