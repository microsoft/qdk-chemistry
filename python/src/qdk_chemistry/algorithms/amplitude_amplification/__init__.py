"""QDK/Chemistry amplitude amplification algorithms module.

This module provides amplitude amplification and the subspace oracles that
name the good subspace it amplifies.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .amplitude_amplification import (
    AmplitudeAmplificationFactory,
    AmplitudeAmplificationSettings,
)
from .qpe_subspace import (
    QPESubspaceMarkingSettings,
)

__all__: list[str] = [
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
    "QPESubspaceMarkingSettings",
]
