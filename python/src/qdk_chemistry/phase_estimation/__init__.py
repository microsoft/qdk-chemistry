"""QDK/Chemistry phase estimation module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.utils.phase import (
    accumulated_phase_from_bits,
    energy_alias_candidates,
    energy_from_phase,
    iterative_phase_feedback_update,
    phase_fraction_from_feedback,
    resolve_energy_aliases,
)

from .base import PhaseEstimation, PhaseEstimationAlgorithm
from .iterative_qpe import IterativePhaseEstimation, IterativePhaseEstimationIteration
from .traditional_qpe import TraditionalPhaseEstimation

__all__ = [
    "IterativePhaseEstimation",
    "IterativePhaseEstimationIteration",
    "PhaseEstimation",
    "PhaseEstimationAlgorithm",
    "TraditionalPhaseEstimation",
    "accumulated_phase_from_bits",
    "energy_alias_candidates",
    "energy_from_phase",
    "iterative_phase_feedback_update",
    "phase_fraction_from_feedback",
    "resolve_energy_aliases",
]
