"""Tests for orbital and active-space optimizer scaffolding."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import ActiveSpaceOptimizer, OrbitalOptimizer, QdkQIOOrbitalOptimizer
from qdk_chemistry.data import (
    ActiveSpaceOptimizationResult,
    OrbitalOptimizationResult,
    Settings,
)

from .test_helpers import create_test_orbitals, create_test_wavefunction


class MockOrbitalOptimizer(OrbitalOptimizer):
    """Test orbital optimizer."""

    def __init__(self) -> None:
        super().__init__()
        self._settings = Settings()

    def name(self) -> str:
        """Return the registered name."""
        return "mock_orbital_optimizer"

    def _run_impl(self, wavefunction):
        """Return rotated orbitals with mock convergence data.

        The optimizer consumes a wavefunction but proposes orbitals; here the
        mock simply echoes the input orbitals unchanged.
        """
        return OrbitalOptimizationResult(wavefunction.get_orbitals(), 2.0, 1.0, 3, True)


class MockActiveSpaceOptimizer(ActiveSpaceOptimizer):
    """Test active-space optimizer."""

    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        """Return the registered name."""
        return "mock_active_space_optimizer"

    def _run_impl(self, orbitals, n_active_alpha_electrons, n_active_beta_electrons):  # noqa: ARG002
        """Return a mock self-consistent optimization result.

        Builds a trivial wavefunction from the supplied orbitals and reports a
        two-macro-iteration history whose length matches ``macro_iterations``.
        """
        wavefunction = create_test_wavefunction()
        return ActiveSpaceOptimizationResult(
            -1.0,
            wavefunction,
            True,
            2,
            [-0.9, -1.0],
            [2.0, 1.0],
        )


def test_optimizer_factories_support_python_registration() -> None:
    """Both new algorithm types participate in the public registry."""
    algorithms.register(MockOrbitalOptimizer)
    algorithms.register(MockActiveSpaceOptimizer)
    try:
        assert isinstance(
            algorithms.create("orbital_optimizer", "mock_orbital_optimizer"),
            OrbitalOptimizer,
        )
        assert isinstance(
            algorithms.create("active_space_optimizer", "mock_active_space_optimizer"),
            ActiveSpaceOptimizer,
        )
        optimizer = algorithms.create("orbital_optimizer", "mock_orbital_optimizer")
        result = optimizer.run(create_test_wavefunction())
        assert isinstance(result, OrbitalOptimizationResult)
        assert result.converged
        assert result.initial_objective == 2.0
        assert result.final_objective == 1.0
        assert result.iterations == 3
        assert result.orbitals is not None
    finally:
        algorithms.unregister("orbital_optimizer", "mock_orbital_optimizer")
        algorithms.unregister("active_space_optimizer", "mock_active_space_optimizer")


def test_qdk_qio_optimizer_is_the_default_variant() -> None:
    """The concrete QIO optimizer is registered as the default orbital optimizer."""
    optimizer = algorithms.create("orbital_optimizer")

    assert isinstance(optimizer, QdkQIOOrbitalOptimizer)
    assert optimizer.name() == "qdk_qio"
    assert optimizer.settings().get("max_cycles") == 200


def test_active_space_optimizer_trampoline_executes() -> None:
    """Running a Python ActiveSpaceOptimizer dispatches through the trampoline."""
    optimizer = MockActiveSpaceOptimizer()
    orbitals = create_test_orbitals(2)
    result = optimizer.run(orbitals, 1, 1)

    assert isinstance(result, ActiveSpaceOptimizationResult)
    assert result.energy == -1.0
    assert result.converged
    assert result.macro_iterations == 2
    assert list(result.energy_history) == [-0.9, -1.0]
    assert list(result.objective_history) == [2.0, 1.0]
    assert result.wavefunction is not None


def test_active_space_optimizer_default_settings() -> None:
    """The base active-space optimizer owns workflow convergence settings."""
    optimizer = MockActiveSpaceOptimizer()
    assert optimizer.type_name() == "active_space_optimizer"
    assert optimizer.settings().get("max_macro_iterations") == 20
    assert optimizer.settings().get("energy_tolerance") == 1e-8
    assert optimizer.settings().get("objective_tolerance") == 1e-8

    orbital_optimizer_ref = optimizer.settings().get("orbital_optimizer")
    assert orbital_optimizer_ref.algorithm_type == "orbital_optimizer"
    assert orbital_optimizer_ref.algorithm_name == "qdk_qio"


def test_result_types_are_exported() -> None:
    """Result types are available from the public data module."""
    assert ActiveSpaceOptimizationResult is not None
    assert OrbitalOptimizationResult is not None
