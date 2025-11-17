"""Test Qubit Mapper functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import QubitMapper, available, create
from qdk_chemistry.data import Hamiltonian, QubitHamiltonian

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_test_hamiltonian


@pytest.mark.parametrize("encoding", ["jordan-wigner", "bravyi-kitaev", "parity"])
def test_qiskit_qubit_mappers(encoding) -> None:
    """Basic test for mapping a Hamiltonian to a Qubit Hamiltonian using Qiskit."""
    assert "qiskit" in available("qubit_mapper")
    qubit_mapper = create("qubit_mapper", "qiskit", encoding="test")
    assert isinstance(qubit_mapper, QubitMapper)
    assert qubit_mapper.settings().get("encoding") == "test"
    qubit_mapper.settings().set("encoding", encoding)
    assert qubit_mapper.settings().get("encoding") == encoding

    hamiltonian = create_test_hamiltonian(2)
    assert isinstance(hamiltonian, Hamiltonian)
    qubit_hamiltonian = qubit_mapper.run(hamiltonian)
    assert isinstance(qubit_hamiltonian, QubitHamiltonian)
    assert qubit_hamiltonian.pauli_ops.num_qubits == 4
    assert isinstance(qubit_hamiltonian.pauli_strings, list)
    assert (
        qubit_hamiltonian.pauli_strings
        == {
            "jordan-wigner": ["IIII", "IIIZ", "IIZI", "IZII", "ZIII"],
            "bravyi-kitaev": ["IIII", "IIIZ", "IIZZ", "IZII", "ZZZI"],
            "parity": ["IIII", "IIIZ", "IIZZ", "IZZI", "ZZII"],
        }[encoding]
    )
    assert isinstance(qubit_hamiltonian.coefficients, np.ndarray)
    assert np.allclose(
        qubit_hamiltonian.coefficients,
        np.array([2.0, -0.5, -0.5, -0.5, -0.5]),
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )
