"""QDK/Chemistry state preparation algorithms module.

This module provides quantum state preparation algorithms for preparing
quantum states from classical wavefunctions.
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings
from typing import Any

from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.algorithms.state_preparation.dense_pure_state import DensePureStatePreparation
from qdk_chemistry.algorithms.state_preparation.identity import identity_state_prep
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.algorithms.state_preparation.sparse_isometry import (
    SparseIsometryStatePreparation,
)
from qdk_chemistry.algorithms.state_preparation.state_preparation import (
    StatePreparation,
    StatePreparationFactory,
    StatePreparationSettings,
)

# ``StatePreparationSettings`` is deprecated: it is re-exported so existing imports keep
# working, but is intentionally omitted from ``__all__`` so ``import *`` does not pull it in.
__all__ = [
    "AliasSamplingStatePreparation",
    "DensePureStatePreparation",
    "QROMStatePreparation",
    "SparseIsometryStatePreparation",
    "StatePreparationFactory",
    "identity_state_prep",
]

# Deprecated public names mapped to their replacements. Accessing an alias emits a
# DeprecationWarning but returns the new class object, so existing code keeps working.
_DEPRECATED_ALIASES = {
    "SparseIsometryGF2XStatePreparation": "SparseIsometryStatePreparation",
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
