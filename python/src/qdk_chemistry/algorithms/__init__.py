"""A package for algorithms in the quantum applications toolkit.

This module is primarily intended for developers who want to implement
custom algorithms that can be registered and used within the QDK/Chemistry framework.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import contextlib
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
from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.dynamical_correlation_calculator import DynamicalCorrelationCalculator
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
    QdkVVHVLocalizer,
)
from qdk_chemistry.algorithms.phase_estimation.base import PhaseEstimation
from qdk_chemistry.algorithms.projected_multi_configuration_calculator import (
    ProjectedMultiConfigurationCalculator,
    QdkMacisPmc,
)
from qdk_chemistry.algorithms.qubit_hamiltonian_solver import QubitHamiltonianSolver
from qdk_chemistry.algorithms.qubit_mapper import QdkQubitMapper, QubitMapper
from qdk_chemistry.algorithms.scf_solver import QdkScfSolver, ScfSolver
from qdk_chemistry.algorithms.stability_checker import QdkStabilityChecker, StabilityChecker
from qdk_chemistry.algorithms.state_preparation import StatePreparation
from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.base import ControlledEvolutionCircuitMapper
from qdk_chemistry.utils.telemetry import TELEMETRY_ENABLED
from qdk_chemistry.utils.telemetry_events import telemetry_tracker

__all__ = [
    # Classes
    "ActiveSpaceSelector",
    "CircuitExecutor",
    "ControlledEvolutionCircuitMapper",
    "DynamicalCorrelationCalculator",
    "EnergyEstimator",
    "HamiltonianConstructor",
    "MultiConfigurationCalculator",
    "MultiConfigurationScf",
    "OrbitalLocalizer",
    "PhaseEstimation",
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
    "QdkQubitMapper",
    "QdkScfSolver",
    "QdkStabilityChecker",
    "QdkVVHVLocalizer",
    "QdkValenceActiveSpaceSelector",
    "QubitHamiltonianSolver",
    "QubitMapper",
    "ScfSolver",
    "StabilityChecker",
    "StatePreparation",
    "TimeEvolutionBuilder",
    # Factory functions
    "available",
    "create",
    "inspect_settings",
    "print_settings",
    "register",
    "show_default",
    "unregister",
]

_REGISTRY_EXPORTS = frozenset(
    {
        "available",
        "create",
        "inspect_settings",
        "print_settings",
        "register",
        "show_default",
        "unregister",
    }
)

_registry_module: ModuleType | None = None

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from qdk_chemistry.algorithms import registry as _registry_type

    available = _registry_type.available
    create = _registry_type.create
    inspect_settings = _registry_type.inspect_settings
    print_settings = _registry_type.print_settings
    register = _registry_type.register
    show_default = _registry_type.show_default
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


if TELEMETRY_ENABLED:

    def apply_telemetry_to_classes():
        """Apply telemetry tracking to the 'run' methods of all algorithm classes."""
        with contextlib.suppress(NameError):
            for name in __all__:
                cls = globals().get(name)
                if isinstance(cls, type) and hasattr(cls, "run"):
                    cls.run = telemetry_tracker()(cls.run)

    apply_telemetry_to_classes()
    # Delete the function to avoid namespace pollution
    del apply_telemetry_to_classes
