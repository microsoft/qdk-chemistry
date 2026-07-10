"""Tests for deprecated public-name aliases.

These tests guard the backward-compatibility shims added when
``QubitHamiltonian`` was renamed to :class:`~qdk_chemistry.data.QubitOperator`,
``EnergyEstimator`` was renamed to
:class:`~qdk_chemistry.algorithms.ExpectationEstimator`, and the
``"energy_estimator"`` algorithm-type key was renamed to
``"expectation_estimator"``. The old names must keep working while emitting a
``DeprecationWarning`` so downstream users are not broken immediately.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import ExpectationEstimator
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian, QubitOperator


class TestQubitHamiltonianDeprecation:
    """Deprecation of ``QubitHamiltonian`` -> :class:`~qdk_chemistry.data.QubitOperator`."""

    def test_qubit_hamiltonian_is_subclass_of_qubit_operator(self):
        """The deprecated ``QubitHamiltonian`` is a subclass of :class:`QubitOperator`."""
        assert issubclass(QubitHamiltonian, QubitOperator)

    def test_qubit_hamiltonian_construction_warns_and_builds_operator(self):
        """Constructing ``QubitHamiltonian`` warns and yields a working ``QubitOperator``."""
        with pytest.warns(DeprecationWarning, match="QubitHamiltonian"):
            qh = QubitHamiltonian(["ZZ"], np.array([1.0]))
        assert isinstance(qh, QubitOperator)
        assert qh.equiv(QubitOperator(["ZZ"], np.array([1.0])))

    def test_qubit_operator_instance_is_instance_of_deprecated_alias(self):
        """Backward-compat: ``isinstance(op, QubitHamiltonian)`` still matches ``QubitOperator``."""
        op = QubitOperator(["ZZ"], np.array([1.0]))
        assert isinstance(op, QubitHamiltonian)

    def test_deprecated_qubit_hamiltonian_works_when_passed_to_an_algorithm(self):
        """A deprecated ``QubitHamiltonian`` can be passed straight into an algorithm.

        Uses the expectation estimator with a pure-identity Hamiltonian, whose energy
        is deterministic (equal to the coefficient), so no stochastic tolerance is
        needed. This proves the old name is accepted anywhere a ``QubitOperator`` is.
        """
        with pytest.warns(DeprecationWarning, match="QubitHamiltonian"):
            observable = QubitHamiltonian(["IIII"], np.array([3.5]))

        estimator = algorithms.create(
            "expectation_estimator",
            "qdk",
            circuit_executor=AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        circuit = Circuit(qasm='OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[4] q;\n')
        energy_result, _ = estimator.run(circuit, observable, total_shots=100)

        assert np.isclose(energy_result.energy_expectation_value, 3.5)

    def test_referencing_qubit_hamiltonian_does_not_warn(self):
        """Merely importing/referencing the alias must not warn; only construction does."""
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _ = QubitHamiltonian  # referencing the class object must not warn


class TestEnergyEstimatorDeprecation:
    """Deprecation of ``EnergyEstimator`` -> :class:`~qdk_chemistry.algorithms.ExpectationEstimator`.

    Covers both the deprecated class alias and the deprecated ``"energy_estimator"``
    registry type key.
    """

    def test_energy_estimator_alias_resolves_to_expectation_estimator(self):
        """``algorithms.EnergyEstimator`` is the same object as ``ExpectationEstimator``."""
        with pytest.warns(DeprecationWarning, match="EnergyEstimator"):
            alias = algorithms.EnergyEstimator
        assert alias is ExpectationEstimator

    def test_energy_estimator_alias_import_warns(self):
        """Importing the old algorithm name emits a ``DeprecationWarning``."""
        with pytest.warns(DeprecationWarning, match="EnergyEstimator"):
            from qdk_chemistry.algorithms import EnergyEstimator  # noqa: PLC0415
        assert EnergyEstimator is ExpectationEstimator

    def test_deprecated_registry_type_key_create(self):
        """``create("energy_estimator")`` still works and warns, matching the new key."""
        with pytest.warns(DeprecationWarning, match="energy_estimator"):
            old = algorithms.create("energy_estimator", "qdk")
        new = algorithms.create("expectation_estimator", "qdk")
        assert type(old) is type(new)

    def test_deprecated_registry_type_key_create_returns_runnable_estimator(self):
        """``create("energy_estimator", ...)`` returns a real estimator that can run.

        Uses a pure-identity Hamiltonian so the energy is deterministic (equal to the
        coefficient), proving the estimator obtained via the deprecated key is fully
        functional.
        """
        with pytest.warns(DeprecationWarning, match="energy_estimator"):
            estimator = algorithms.create(
                "energy_estimator",
                "qdk",
                circuit_executor=AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
            )
        assert isinstance(estimator, ExpectationEstimator)

        observable = QubitOperator(["IIII"], np.array([2.0]))
        circuit = Circuit(qasm='OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[4] q;\n')
        energy_result, _ = estimator.run(circuit, observable, total_shots=100)

        assert np.isclose(energy_result.energy_expectation_value, 2.0)

    def test_deprecated_registry_type_key_available(self):
        """``available("energy_estimator")`` resolves to the new key's implementations."""
        with pytest.warns(DeprecationWarning, match="energy_estimator"):
            old = algorithms.available("energy_estimator")
        new = algorithms.available("expectation_estimator")
        assert old == new
        assert new  # non-empty: the qdk implementation is registered

    def test_deprecated_registry_type_key_show_default(self):
        """``show_default("energy_estimator")`` resolves to the new key's default."""
        with pytest.warns(DeprecationWarning, match="energy_estimator"):
            old = algorithms.show_default("energy_estimator")
        assert old == algorithms.show_default("expectation_estimator")
