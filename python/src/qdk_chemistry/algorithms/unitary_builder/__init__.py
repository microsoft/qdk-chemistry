"""QDK/Chemistry unitary builder module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings
from typing import Any

from .base import UnitaryBuilderFactory

__all__: list[str] = ["UnitaryBuilderFactory"]

# Deprecated public names mapped to their replacements. Accessing an alias emits a
# DeprecationWarning but returns the new class object, so existing code keeps working.
_DEPRECATED_ALIASES = {
    "HamiltonianUnitaryBuilderFactory": "UnitaryBuilderFactory",
}


def __getattr__(name: str) -> Any:
    """Resolve deprecated class names to their replacements."""
    target = _DEPRECATED_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"'{__name__}.{name}' is deprecated and will be removed in a "
        f"future release; use '{__name__}.{target}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return globals()[target]


def __dir__() -> list[str]:
    """Ensure dir() lists the deprecated aliases alongside the current names."""
    return sorted(set(globals()) | set(_DEPRECATED_ALIASES))
