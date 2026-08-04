"""QDK/Chemistry amplitude amplification module.

Amplitude amplification boosts the probability that a prepared state is found in
a marked subspace.  The quantum primitives live in the Q# module
``QDKChemistry.Utils.AmplitudeAmplification``; this package supplies the registry
algorithm class :class:`~qdk_chemistry.algorithms.amplitude_amplification.base.AmplitudeAmplification`
that drives them.  The classical round-scheduling closed forms that decide how
many amplification rounds to run are methods on that class.

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
