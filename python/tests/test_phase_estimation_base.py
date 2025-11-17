"""Tests for phase estimation base factory utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qdk_chemistry.algorithms import (
    IterativePhaseEstimation,
    PhaseEstimation,
    PhaseEstimationAlgorithm,
    TraditionalPhaseEstimation,
)
from qdk_chemistry.data import QubitHamiltonian


@pytest.fixture
def two_qubit_problem_data() -> tuple[QubitHamiltonian, QuantumCircuit, float]:
    """Return Hamiltonian/state data used across phase estimation tests."""
    sparse_pauli_op = SparsePauliOp.from_list([("XX", 0.25), ("ZZ", 0.5)])
    hamiltonian = QubitHamiltonian(
        pauli_strings=sparse_pauli_op.paulis.to_labels(), coefficients=sparse_pauli_op.coeffs
    )
    state_prep = QuantumCircuit(2, name="psi")
    state_prep.initialize([0.6, 0.0, 0.0, 0.8], [0, 1])
    evolution_time = float(np.pi / 2.0)
    return hamiltonian, state_prep, evolution_time


def test_factory_defaults_to_iterative(two_qubit_problem_data):
    hamiltonian, _, evolution_time = two_qubit_problem_data
    algorithm = PhaseEstimation.from_algorithm(
        None,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
    )

    assert isinstance(algorithm, IterativePhaseEstimation)


def test_factory_accepts_enum(two_qubit_problem_data):
    hamiltonian, _, evolution_time = two_qubit_problem_data
    algorithm = PhaseEstimation.from_algorithm(
        PhaseEstimationAlgorithm.TRADITIONAL,
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
    )

    assert isinstance(algorithm, TraditionalPhaseEstimation)


def test_factory_accepts_string(two_qubit_problem_data):
    hamiltonian, _, evolution_time = two_qubit_problem_data
    algorithm = PhaseEstimation.from_algorithm(
        "traditional",
        hamiltonian=hamiltonian,
        evolution_time=evolution_time,
    )

    assert isinstance(algorithm, TraditionalPhaseEstimation)


def test_factory_rejects_unknown_identifier(two_qubit_problem_data):
    hamiltonian, _, evolution_time = two_qubit_problem_data

    with pytest.raises(ValueError, match="Unrecognized phase estimation algorithm 'invalid'"):
        PhaseEstimation.from_algorithm(
            "invalid",
            hamiltonian=hamiltonian,
            evolution_time=evolution_time,
        )
