"""Unit tests for time-evolution circuit helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp

from qdk_chemistry.utils.time_evolution import (
    PauliEvolutionTerm,
    append_controlled_time_evolution,
    controlled_pauli_rotation,
    extract_terms_from_hamiltonian,
    pauli_evolution_terms,
)

from .reference_tolerances import pauli_coefficient_imaginary_tolerance


def test_pauli_evolution_terms_extracts_mapping() -> None:
    """Decomposition should capture non-identity Pauli components."""
    operator = SparsePauliOp.from_list([("XI", 0.5), ("ZZ", 1.0)])
    terms = pauli_evolution_terms(operator)

    assert terms == [
        PauliEvolutionTerm(pauli_map={1: "X"}, coefficient=0.5),
        PauliEvolutionTerm(pauli_map={0: "Z", 1: "Z"}, coefficient=1.0),
    ]


def test_pauli_evolution_terms_rejects_imaginary_coefficients() -> None:
    """Imaginary components beyond tolerance should raise an error."""
    operator = SparsePauliOp.from_list([("Z", 1.0 + 1e-6j)])
    with pytest.raises(ValueError, match="real coefficients"):
        pauli_evolution_terms(operator, atol=pauli_coefficient_imaginary_tolerance)


def test_controlled_pauli_rotation_identity_adds_phase_gate() -> None:
    """Identity terms should translate into a controlled phase on the ancilla."""
    ancilla = QuantumRegister(1, "anc")
    system = QuantumRegister(1, "sys")
    circuit = QuantumCircuit(ancilla, system)

    term = PauliEvolutionTerm(pauli_map={}, coefficient=1.0)
    returned = controlled_pauli_rotation(circuit, ancilla[0], list(system), term, angle=np.pi / 4)

    assert returned is circuit

    ops = circuit.count_ops()
    assert ops == {"p": 1}
    phase_param = float(circuit.data[0][0].params[0])
    assert phase_param == pytest.approx(-np.pi / 4)


def test_append_controlled_time_evolution_repeats_for_power() -> None:
    """Repeated applications should accumulate the correct controlled rotations."""
    ancilla = QuantumRegister(1, "anc")
    system = QuantumRegister(1, "sys")
    circuit = QuantumCircuit(ancilla, system)

    terms = [PauliEvolutionTerm(pauli_map={0: "Z"}, coefficient=0.5)]
    append_controlled_time_evolution(circuit, ancilla[0], list(system), terms, time=np.pi, power=2)

    ops = circuit.count_ops()
    assert ops == {"crz": 2}
    for instruction, _, _ in circuit.data:
        assert instruction.name == "crz"
        assert float(instruction.params[0]) == pytest.approx(np.pi)


def test_append_controlled_time_evolution_rejects_invalid_power() -> None:
    """Power less than one should raise a validation error."""
    ancilla = QuantumRegister(1, "anc")
    system = QuantumRegister(1, "sys")
    circuit = QuantumCircuit(ancilla, system)

    with pytest.raises(ValueError, match="power must be at least 1"):
        append_controlled_time_evolution(
            circuit,
            ancilla[0],
            list(system),
            [PauliEvolutionTerm(pauli_map={}, coefficient=1.0)],
            time=1.0,
            power=0,
        )


def test_extract_terms_from_hamiltonian_matches_direct_decomposition() -> None:
    """Helper should defer to pauli_evolution_terms for decomposition."""
    operator = SparsePauliOp.from_list([("XX", 0.3), ("YY", -0.7)])

    class _StubHamiltonian:
        def __init__(self, pauli_ops: SparsePauliOp) -> None:
            self.pauli_ops = pauli_ops

    hamiltonian = _StubHamiltonian(operator)

    direct_terms = pauli_evolution_terms(operator)
    extracted_terms = extract_terms_from_hamiltonian(hamiltonian)
    assert extracted_terms == direct_terms
