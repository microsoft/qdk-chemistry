"""QDK/Chemistry amplitude amplification module.

Amplitude amplification boosts the probability that a prepared state is found in
a marked subspace.  The quantum primitives live in the Q# module
``QDKChemistry.Utils.AmplitudeAmplification``; this package supplies the registry
algorithm class that drives them, plus the classical policy in
:mod:`~qdk_chemistry.algorithms.amplitude_amplification.schedule` that decides
how many amplification rounds to run.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.amplitude_amplification.schedule import (
    DEFAULT_GROWTH_RATE,
    expected_rounds_exponential,
    exponential_schedule,
    fixed_point_phases,
    fixed_point_rounds,
    fixed_point_success_probability,
    optimal_rounds,
    overshoot_overlap,
    robust_rounds,
    rotation_angle,
    safe_rounds,
    success_probability,
    success_probability_with_assumed_overlap,
    worst_case_success_probability,
)

from .base import AmplitudeAmplificationFactory

__all__: list[str] = [
    "DEFAULT_GROWTH_RATE",
    "AmplitudeAmplificationFactory",
    "expected_rounds_exponential",
    "exponential_schedule",
    "fixed_point_phases",
    "fixed_point_rounds",
    "fixed_point_success_probability",
    "optimal_rounds",
    "overshoot_overlap",
    "robust_rounds",
    "rotation_angle",
    "safe_rounds",
    "success_probability",
    "success_probability_with_assumed_overlap",
    "worst_case_success_probability",
]
