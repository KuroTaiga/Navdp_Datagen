from __future__ import annotations

from typing import Any


def render(*args: Any, **kwargs: Any) -> Any:
    from .gaussian_renderer import render as _render

    return _render(*args, **kwargs)


__all__ = ["render"]
