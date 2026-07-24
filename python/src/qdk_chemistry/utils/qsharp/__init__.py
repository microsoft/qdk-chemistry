"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS", "get_qsharp_utils"]

_PROJECT_ROOT = str(Path(__file__).parent)
_initialized = False


def _ensure_qsharp_session():
    """Ensure the interpreter has the chemistry Q# project loaded."""
    global _initialized  # noqa: PLW0603
    try:
        _ = qdk.code.MPSSequential
    except AttributeError:
        _initialized = False
    if not _initialized:
        qsharp.init(project_root=_PROJECT_ROOT)
        _initialized = True


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded).

    Initializes the global Q# interpreter with the chemistry project on first call.
    """
    _ensure_qsharp_session()
    return qdk.code.QDKChemistry.Utils


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Falls through to MPS namespace for names not found in QDKChemistry.Utils.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        if name == "MPSSequential":
            _ensure_qsharp_session()
            return qdk.code.MPSSequential
        if name == "MPSSparse":
            _ensure_qsharp_session()
            return qdk.code.MPSSparse
        _ensure_qsharp_session()
        return getattr(qdk.code.QDKChemistry.Utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
