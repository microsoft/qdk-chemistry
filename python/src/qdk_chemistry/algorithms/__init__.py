"""A package for algorithms in the quantum applications toolkit.

This module is primarily intended for developers who want to implement
custom algorithms that can be registered and used within the QDK/Chemistry framework.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import (
    ActiveSpaceSelector,
    HamiltonianConstructor,
    Localizer,
    MultiConfigurationCalculator,
    MultiConfigurationScf,
    ProjectedMultiConfigurationCalculator,
    ReferenceDerivedCalculator,
    ScfSolver,
    StabilityChecker,
)
from qdk_chemistry.algorithms.energy_estimator import EnergyEstimator
from qdk_chemistry.algorithms.qubit_mapper import QubitMapper
from qdk_chemistry.algorithms.registry import (
    available,
    create,
    register,
    show_settings,
    unregister,
)
from qdk_chemistry.algorithms.state_preparation import StatePreparation
from qdk_chemistry.phase_estimation import (
    IterativePhaseEstimation,
    IterativePhaseEstimationIteration,
    PhaseEstimation,
    PhaseEstimationAlgorithm,
    TraditionalPhaseEstimation,
    energy_from_phase,
)

__all__ = [
    # Classes
    "ActiveSpaceSelector",
    "EnergyEstimator",
    "HamiltonianConstructor",
    "IterativePhaseEstimation",
    "IterativePhaseEstimationIteration",
    "Localizer",
    "MultiConfigurationCalculator",
    "MultiConfigurationScf",
    "PhaseEstimation",
    "PhaseEstimationAlgorithm",
    "ProjectedMultiConfigurationCalculator",
    "QubitMapper",
    "ReferenceDerivedCalculator",
    "ScfSolver",
    "StabilityChecker",
    "StatePreparation",
    "TraditionalPhaseEstimation",
    # Factory functions
    "available",
    "create",
    "energy_from_phase",
    "register",
    "show_settings",
    "unregister",
]
