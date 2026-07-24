"""Model registry with decorator-based auto-registration."""

from __future__ import annotations

from typing import Type

_registry: dict[str, Type] = {}


def register_model(name: str):
    """Class decorator: register a BaseModel subclass by its canonical name."""

    def decorator(cls: Type) -> Type:
        _registry[name] = cls
        return cls

    return decorator


def get_all_models() -> dict[str, Type]:
    """Return the full registry {model_name: model_class}."""
    # Import model modules to trigger registration side-effects
    _import_models()
    return dict(_registry)


def get_model(name: str) -> Type | None:
    """Look up a single model by name, or None."""
    _import_models()
    return _registry.get(name)


# ---------------------------------------------------------------------------
# Lazy import — each model module calls @register_model at import time
# ---------------------------------------------------------------------------
_imported = False


def _import_models() -> None:
    global _imported
    if _imported:
        return
    # flake8 / isort disabled so the side-effect import order is explicit
    import benchmark.models.flux  # noqa: F401
    import benchmark.models.qwen_image  # noqa: F401
    import benchmark.models.wan  # noqa: F401
    import benchmark.models.hunyuan  # noqa: F401
    import benchmark.models.ltx2  # noqa: F401

    _imported = True
