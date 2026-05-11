"""Gorit Lab DarkTech Generator.

Public entry points:
- :func:`launch_gradio`: starts the Gradio UI (used by the Colab notebook).
- :func:`generate_track`: programmatic one-shot generation.

Imports are lazy so that ``import darktech_generator.schemas`` works without the
heavy audio/ML stack installed.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["generate_track", "launch_gradio", "__version__"]

if TYPE_CHECKING:
    from darktech_generator.api import generate_track, launch_gradio


def __getattr__(name: str) -> Any:
    if name in ("generate_track", "launch_gradio"):
        from darktech_generator import api

        return getattr(api, name)
    raise AttributeError(f"module 'darktech_generator' has no attribute {name!r}")
