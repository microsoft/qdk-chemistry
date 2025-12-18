"""Tests for FirstOrderTrotterConstructor and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.time_evolution.constructor.trotter.first_order_trotter import (
    FirstOrderTrotterConstructor,
    _decompose_trotter_step,
    _pauli_label_to_map,
)
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestPauliLabelToMap:
    """Tests for the _pauli_label_to_map helper function."""

    def test_identity_only(self):
        """Test that identity-only labels return an empty mapping."""
        assert _pauli_label_to_map("III") == {}

    def test_single_pauli(self):
        """Test labels with a single non-identity Pauli."""
        assert _pauli_label_to_map("X") == {0: "X"}
        assert _pauli_label_to_map("IZ") == {0: "Z"}

    def test_multiple_paulis(self):
        """Test labels with multiple non-identity Paulis."""
        # label is little-endian: rightmost char -> qubit 0
        mapping = _pauli_label_to_map("XYZ")
        assert mapping == {0: "Z", 1: "Y", 2: "X"}


class TestDecomposeTrotterStep:
    """Tests for the _decompose_trotter_step helper function."""

    def test_basic_decomposition(self):
        """Test basic decomposition of a qubit Hamiltonian."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])

        terms, ordering = _decompose_trotter_step(hamiltonian, time=2.0)

        assert len(terms) == 2
        assert ordering.indices == [0, 1]

        assert terms[0] == ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=2.0)
        assert terms[1] == ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.0)

    def test_filters_small_coefficients(self):
        """Test that terms with small coefficients are filtered out."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[1e-15, 1.0],
        )

        terms, ordering = _decompose_trotter_step(hamiltonian, time=1.0, atol=1e-12)

        assert len(terms) == 1
        assert ordering.indices == [0]
        assert terms[0].pauli_term == {0: "Z"}

    def test_rejects_non_hermitian(self):
        """Test that non-Hermitian Hamiltonians raise a ValueError."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X"],
            coefficients=[1.0 + 1.0j],
        )

        with pytest.raises(ValueError, match="Non-Hermitian"):
            _decompose_trotter_step(hamiltonian, time=1.0)


class TestFirstOrderTrotterConstructor:
    """Tests for the FirstOrderTrotterConstructor class."""

    def test_name(self):
        """Test the name method of FirstOrderTrotterConstructor."""
        ctor = FirstOrderTrotterConstructor()
        assert ctor.name() == "first_order_trotter"

    def test_single_step_construction(self):
        """Test construction of time evolution unitary with a single Trotter step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        ctor = FirstOrderTrotterConstructor(num_trotter_steps=1)
        unitary = ctor.run(hamiltonian, time=0.2)

        assert isinstance(unitary, TimeEvolutionUnitary)
        container = unitary._container

        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 1
        assert container.step_reps == 1
        assert len(container.step_terms) == 2

    def test_multiple_trotter_steps(self):
        """Test construction of time evolution unitary with multiple Trotter steps."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "ZZ"],
            coefficients=[2.0, 1.0],
        )

        ctor = FirstOrderTrotterConstructor(num_trotter_steps=4)
        unitary = ctor.run(hamiltonian, time=0.2)

        container = unitary._container

        # dt = 0.2 / 4 = 0.05
        assert container.step_reps == 4
        assert np.isclose(
            container.step_terms[0].angle,
            0.1,  # angle = dt * coeff = 0.05 * 2.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            container.step_terms[1].angle,
            0.05,  # angle = dt * coeff = 0.05 * 1.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
