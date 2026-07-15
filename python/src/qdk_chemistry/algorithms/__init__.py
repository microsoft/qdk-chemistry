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
from qdk_chemistry.algorithms.controlled_circuit_mapper.base import ControlledCircuitMapper
from qdk_chemistry.algorithms.dynamical_correlation_calculator import DynamicalCorrelationCalculator, QdkMP2Calculator
from qdk_chemistry.algorithms.expectation_estimator.expectation_estimator import ExpectationEstimator
from qdk_chemistry.algorithms.expectation_estimator.qdk import QdkExpectationEstimator
from qdk_chemistry.algorithms.hadamard_test.hadamard_test import HadamardTest
from qdk_chemistry.algorithms.hamiltonian_constructor import (
    HamiltonianConstructor,
    QdkHamiltonianConstructor,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import HamiltonianUnitaryBuilder, TimeEvolutionBuilder
from qdk_chemistry.algorithms.multi_configuration_calculator import (
    MultiConfigurationCalculator,
    QdkMacisAsci,
    QdkMacisCas,
)
from qdk_chemistry.algorithms.multi_configuration_scf import MultiConfigurationScf
from qdk_chemistry.algorithms.nuclear_derivative import (
    FiniteDifferenceNuclearDerivativeCalculator,
    NuclearDerivativeCalculator,
    QdkNuclearDerivativeCalculator,
)
from qdk_chemistry.algorithms.orbital_localizer import (
    OrbitalLocalizer,
    QdkMP2NaturalOrbitalLocalizer,
    QdkNaturalOrbitalLocalizer,
    QdkPipekMezeyLocalizer,
    QdkVVHVLocalizer,
)
from qdk_chemistry.algorithms.phase_estimation.base import PhaseEstimation
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import QpeCircuitBuilder
from qdk_chemistry.algorithms.projected_multi_configuration_calculator import (
    ProjectedMultiConfigurationCalculator,
    QdkMacisPmc,
)
from qdk_chemistry.algorithms.qubit_hamiltonian_solver import QubitHamiltonianSolver
from qdk_chemistry.algorithms.qubit_mapper import QdkQubitMapper, QubitMapper
from qdk_chemistry.algorithms.scf_solver import QdkScfSolver, QdkStabilizedScfSolver, ScfSolver
from qdk_chemistry.algorithms.stability_checker import QdkStabilityChecker, StabilityChecker
from qdk_chemistry.algorithms.state_preparation import StatePreparation
from qdk_chemistry.utils.telemetry import TELEMETRY_ENABLED
from qdk_chemistry.utils.telemetry_events import telemetry_tracker

__all__ = [
    # Classes
    "ActiveSpaceSelector",
    "CircuitExecutor",
    "ControlledCircuitMapper",
    "DynamicalCorrelationCalculator",
    "ExpectationEstimator",
    "FiniteDifferenceNuclearDerivativeCalculator",
    "HadamardTest",
    "HamiltonianConstructor",
    "HamiltonianUnitaryBuilder",
    "MultiConfigurationCalculator",
    "MultiConfigurationScf",
    "NuclearDerivativeCalculator",
    "OrbitalLocalizer",
    "PhaseEstimation",
    "ProjectedMultiConfigurationCalculator",
    "QdkAutocasActiveSpaceSelector",
    "QdkAutocasEosActiveSpaceSelector",
    "QdkExpectationEstimator",
    "QdkHamiltonianConstructor",
    "QdkMP2Calculator",
    "QdkMP2NaturalOrbitalLocalizer",
    "QdkMacisAsci",
    "QdkMacisCas",
    "QdkMacisPmc",
    "QdkNaturalOrbitalLocalizer",
    "QdkNuclearDerivativeCalculator",
    "QdkOccupationActiveSpaceSelector",
    "QdkPipekMezeyLocalizer",
    "QdkQubitMapper",
    "QdkScfSolver",
    "QdkStabilityChecker",
    "QdkStabilizedScfSolver",
    "QdkVVHVLocalizer",
    "QdkValenceActiveSpaceSelector",
    "QpeCircuitBuilder",
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

# Deprecated public names mapped to their replacements. Accessing an alias emits a
# DeprecationWarning but returns the new class object, so existing code keeps working.
_DEPRECATED_ALIASES = {
    "EnergyEstimator": "ExpectationEstimator",
    "QdkEnergyEstimator": "QdkExpectationEstimator",
}

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
    target = _DEPRECATED_ALIASES.get(name)
    if target is not None:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"'qdk_chemistry.algorithms.{name}' is deprecated and will be removed in a "
            f"future release; use 'qdk_chemistry.algorithms.{target}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[target]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Ensure dir() lists lazily resolved registry helpers."""
    return sorted(set(globals()) | _REGISTRY_EXPORTS | set(_DEPRECATED_ALIASES))


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
