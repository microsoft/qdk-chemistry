"""Tests for Pauli string commutation utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms.time_evolution.builder.pauli_commutation import (
    commutator_bound_first_order,
    do_pauli_strings_commute,
)


class TestDoPauliStringsCommute:
    """Tests for the do_pauli_strings_commute function."""

    @pytest.mark.parametrize(
        ("label_a", "label_b", "expected"),
        [
            # Identity commutes with everything
            ("I", "I", True),
            ("I", "X", True),
            ("I", "Y", True),
            ("I", "Z", True),
            # Same Pauli commutes with itself
            ("X", "X", True),
            ("Y", "Y", True),
            ("Z", "Z", True),
            # Different non-identity single-qubit Paulis anticommute
            ("X", "Y", False),
            ("Y", "X", False),
            ("X", "Z", False),
            ("Z", "X", False),
            ("Y", "Z", False),
            ("Z", "Y", False),
        ],
    )
    def test_single_qubit(self, label_a, label_b, expected):
        """Test commutation for single-qubit Pauli operators."""
        assert do_pauli_strings_commute(label_a, label_b) is expected

    @pytest.mark.parametrize(
        ("label_a", "label_b", "expected"),
        [
            # Operators on different qubits always commute
            ("XI", "IX", True),
            ("IY", "ZI", True),
            # Same Pauli on same qubit commutes
            ("XI", "XI", True),
            # Even number of anticommuting positions -> commute
            ("XX", "YY", True),
            ("XY", "YX", True),
            ("XZ", "ZX", True),
            ("YZ", "ZY", True),
            # Odd number of anticommuting positions -> anticommute
            ("XI", "YI", False),
            ("XI", "ZI", False),
            ("IY", "IZ", False),
            ("XY", "YI", False),
        ],
    )
    def test_two_qubit(self, label_a, label_b, expected):
        """Test commutation for two-qubit Pauli strings."""
        assert do_pauli_strings_commute(label_a, label_b) is expected

    @pytest.mark.parametrize(
        ("label_a", "label_b", "expected"),
        [
            # All-identity commutes
            ("III", "III", True),
            # Three anticommuting positions (odd) -> anticommute
            ("XYZ", "YZX", False),
            # Two anticommuting positions (even) -> commute
            ("XYI", "YXI", True),
            # Mixed identity padding
            ("XIZ", "YIZ", False),
            ("XIZ", "YIX", True),
        ],
    )
    def test_multi_qubit(self, label_a, label_b, expected):
        """Test commutation for multi-qubit Pauli strings."""
        assert do_pauli_strings_commute(label_a, label_b) is expected

    def test_different_length_raises(self):
        """Test that different-length labels raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            do_pauli_strings_commute("XI", "XII")


class TestCommutatorBoundFirstOrder:
    """Tests for the commutator_bound_first_order function."""

    def test_all_commuting_terms(self):
        """Test that commuting terms give zero bound."""
        # XI and IX commute
        bound = commutator_bound_first_order(["XI", "IX"], [1.0, 1.0])
        assert bound == 0.0

    def test_anticommuting_pair(self):
        """Test a single anticommuting pair."""
        # X and Z anticommute -> bound = 2 * |a1| * |a2| = 2 * 1 * 1 = 2
        bound = commutator_bound_first_order(["X", "Z"], [1.0, 1.0])
        assert bound == 2.0

    def test_anticommuting_pair_with_coefficients(self):
        """Test an anticommuting pair with non-unit coefficients."""
        # X and Z anticommute -> bound = 2 * |2| * |3| = 12
        bound = commutator_bound_first_order(["X", "Z"], [2.0, 3.0])
        assert bound == 12.0

    def test_mixed_commuting_and_anticommuting(self):
        """Test a mix of commuting and anticommuting pairs."""
        # XI, IX, ZI: XI and IX commute, XI and ZI anticommute, IX and ZI commute
        bound = commutator_bound_first_order(
            ["XI", "IX", "ZI"], [1.0, 1.0, 1.0]
        )
        # Only XI/ZI anticommute -> 2 * 1 * 1 = 2
        assert bound == 2.0

    def test_negative_coefficients(self):
        """Test that negative coefficients are handled via absolute values."""
        bound = commutator_bound_first_order(["X", "Z"], [-2.0, -3.0])
        assert bound == 12.0

    def test_single_term(self):
        """Test that a single term gives zero bound."""
        bound = commutator_bound_first_order(["X"], [1.0])
        assert bound == 0.0

    def test_empty_hamiltonian(self):
        """Test that an empty Hamiltonian gives zero bound."""
        bound = commutator_bound_first_order([], [])
        assert bound == 0.0

    def test_mismatched_lengths_raises(self):
        """Test that mismatched labels/coefficients raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            commutator_bound_first_order(["X", "Z"], [1.0])
