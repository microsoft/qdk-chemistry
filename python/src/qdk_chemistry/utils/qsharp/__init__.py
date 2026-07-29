"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import TargetProfile

__all__ = ["QSHARP_UTILS", "get_qsharp_context"]

_PROJECT_ROOT = str(Path(__file__).parent)


def get_qsharp_context(target_profile: TargetProfile = TargetProfile.Adaptive_RIF) -> qdk.Context:
    """Create a QDK context for the vendored Q# utility project.

    Args:
        target_profile: The target profile for the context. Defaults to Adaptive RIF.

    Returns:
        A :class:`qdk.Context` with the Q# chemistry utilities loaded.

    """
    return qdk.Context(project_root=_PROJECT_ROOT, target_profile=target_profile)


QSHARP_UTILS = get_qsharp_context().code.QDKChemistry.Utils
