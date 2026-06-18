"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import qsharp

__all__ = ["QSHARP_UTILS"]

_PROJECT_ROOT = str(Path(__file__).parent)


def get_qsharp_utils():
    """Returns the Q# namespace for chemistry operations (lazy-loaded)."""
    try:
        return qdk.code.QDKChemistry.Utils
    except AttributeError:
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)
        return qdk.code.QDKChemistry.Utils


class _QSharpUtilsProxy:
    """Lightweight proxy that lazily resolves the Q# utilities namespace."""

    def __getattr__(self, name: str):
        """Load Q# code (if necessary) and resolve *name* on the utilities namespace.

        Args:
            name: The name of the attribute being accessed on the Q# utilities namespace.

        """
        utils = get_qsharp_utils()
        return getattr(utils, name)


QSHARP_UTILS = _QSharpUtilsProxy()
