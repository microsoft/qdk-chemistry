"""QDK/Chemistry amplitude amplification algorithms module.

This module provides amplitude amplification and the subspace oracles that
name the good subspace it amplifies.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .amplitude_amplification import (
    AmplitudeAmplification,
    AmplitudeAmplificationFactory,
    AmplitudeAmplificationSettings,
)
from .qpe_subspace import (
    QPESubspaceMarking,
    QPESubspaceMarkingSettings,
)

__all__: list[str] = [
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
    "QPESubspaceMarkingSettings",
]

# ``AmplitudeAmplification`` and ``QPESubspaceMarking`` are imported above so that
# ``qdk_chemistry.algorithms`` can re-export them, but they are deliberately left out of
# ``__all__``: they are already listed in ``qdk_chemistry.algorithms.__all__``, and naming a
# class in both places makes Sphinx autosummary emit a "duplicate object description" warning,
# which the docs build treats as an error. The sibling subpackages (for example
# ``state_preparation``, which imports but does not export ``StatePreparation``) follow the
# same rule.
