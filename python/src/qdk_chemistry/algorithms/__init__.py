"""A package for algorithms in the quantum applications toolkit.

This module is primarily intended for developers who want to implement
custom algorithms that can be registered and used within the QDK/Chemistry framework.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from qdk_chemistry.algorithms.active_space_selector import (
    ActiveSpaceSelector,
    QdkAutocasActiveSpaceSelector,
    QdkAutocasEosActiveSpaceSelector,
    QdkOccupationActiveSpaceSelector,
    QdkValenceActiveSpaceSelector,
)
from qdk_chemistry.algorithms.coupled_cluster_calculator import CoupledClusterCalculator
from qdk_chemistry.algorithms.energy_estimator import EnergyEstimator
from qdk_chemistry.algorithms.hamiltonian_constructor import (
    HamiltonianConstructor,
    QdkHamiltonianConstructor,
)
from qdk_chemistry.algorithms.multi_configuration_calculator import (
    MultiConfigurationCalculator,
    QdkMacisAsci,
    QdkMacisCas,
)
from qdk_chemistry.algorithms.multi_configuration_scf import MultiConfigurationScf
from qdk_chemistry.algorithms.orbital_localizer import (
    OrbitalLocalizer,
    QdkMP2NaturalOrbitalLocalizer,
    QdkPipekMezeyLocalizer,
)
from qdk_chemistry.algorithms.projected_multi_configuration_calculator import (
    ProjectedMultiConfigurationCalculator,
    QdkMacisPmc,
)
from qdk_chemistry.algorithms.qubit_mapper import QubitMapper
from qdk_chemistry.algorithms.scf_solver import QdkScfSolver, ScfSolver
from qdk_chemistry.algorithms.stability_checker import StabilityChecker
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
    "CoupledClusterCalculator",
    "EnergyEstimator",
    "HamiltonianConstructor",
    "IterativePhaseEstimation",
    "IterativePhaseEstimationIteration",
    "MultiConfigurationCalculator",
    "MultiConfigurationScf",
    "OrbitalLocalizer",
    "PhaseEstimation",
    "PhaseEstimationAlgorithm",
    "ProjectedMultiConfigurationCalculator",
    "QdkAutocasActiveSpaceSelector",
    "QdkAutocasEosActiveSpaceSelector",
    "QdkHamiltonianConstructor",
    "QdkMP2NaturalOrbitalLocalizer",
    "QdkMacisAsci",
    "QdkMacisCas",
    "QdkMacisPmc",
    "QdkOccupationActiveSpaceSelector",
    "QdkPipekMezeyLocalizer",
    "QdkScfSolver",
    "QdkValenceActiveSpaceSelector",
    "QubitMapper",
    "ScfSolver",
    "StabilityChecker",
    "StatePreparation",
    "TraditionalPhaseEstimation",
    # Factory functions
    "available",
    "create",
    "energy_from_phase",
    "register",
    "show_default",
    "show_settings",
    "unregister",
]


_REGISTRY_EXPORTS = frozenset(
    {
        "available",
        "create",
        "register",
        "show_default",
        "show_settings",
        "unregister",
    }
)

_registry_module: ModuleType | None = None

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from qdk_chemistry.algorithms import registry as _registry_type

    available = _registry_type.available
    create = _registry_type.create
    register = _registry_type.register
    show_default = _registry_type.show_default
    show_settings = _registry_type.show_settings
    unregister = _registry_type.unregister


def _load_registry() -> ModuleType:
    """Import the registry module lazily to avoid circular imports."""
    global _registry_module  # noqa: PLW0603
    if _registry_module is None:
        _registry_module = importlib.import_module("qdk_chemistry.algorithms.registry")
    return _registry_module


def __getattr__(name: str) -> Any:
    """Provide registry helpers on first access while keeping imports lazy."""
    if name in _REGISTRY_EXPORTS:
        attr = getattr(_load_registry(), name)
        globals()[name] = attr  # cache for subsequent lookups
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Ensure dir() lists lazily resolved registry helpers."""
    return sorted(set(globals()) | _REGISTRY_EXPORTS)
