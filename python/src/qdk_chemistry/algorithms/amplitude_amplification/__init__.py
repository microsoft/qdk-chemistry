"""QDK/Chemistry amplitude amplification module.

The quantum primitives live in the Q# module
``QDKChemistry.Utils.AmplitudeAmplification``; the registry algorithm class
:class:`~qdk_chemistry.algorithms.amplitude_amplification.base.AmplitudeAmplification`
assembles them into a circuit and carries the round-scheduling closed forms.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .base import AmplitudeAmplification, AmplitudeAmplificationFactory

__all__: list[str] = [
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
]
