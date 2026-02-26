"""Tests for Pauli string commutation utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.utils.pauli_commutation import (
    commutator_bound_first_order,
    do_pauli_labels_commute,
    do_pauli_labels_qw_commute,
    do_pauli_maps_commute,
    do_pauli_maps_qw_commute,
    get_commutation_checker,
)


class TestDoPauliLabelsCommute:
    """Tests for the do_pauli_labels_commute function."""

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
        assert do_pauli_labels_commute(label_a, label_b) is expected

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
        assert do_pauli_labels_commute(label_a, label_b) is expected

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
        assert do_pauli_labels_commute(label_a, label_b) is expected

    def test_different_length_raises(self):
        """Test that different-length labels raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            do_pauli_labels_commute("XI", "XII")


class TestCommutatorBoundFirstOrder:
    """Tests for the commutator_bound_first_order function."""

    def test_all_commuting_terms(self):
        """Test that commuting terms give zero bound."""
        # XI and IX commute
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        bound = commutator_bound_first_order(h)
        assert bound == 0.0

    def test_anticommuting_pair(self):
        """Test a single anticommuting pair."""
        # X and Z anticommute -> bound = 2 * |a1| * |a2| = 2 * 1 * 1 = 2
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        bound = commutator_bound_first_order(h)
        assert bound == 2.0

    def test_anticommuting_pair_with_coefficients(self):
        """Test an anticommuting pair with non-unit coefficients."""
        # X and Z anticommute -> bound = 2 * |2| * |3| = 12
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[2.0, 3.0])
        bound = commutator_bound_first_order(h)
        assert bound == 12.0

    def test_mixed_commuting_and_anticommuting(self):
        """Test a mix of commuting and anticommuting pairs."""
        # XI, IX, ZI: XI and IX commute, XI and ZI anticommute, IX and ZI commute
        h = QubitHamiltonian(pauli_strings=["XI", "IX", "ZI"], coefficients=[1.0, 1.0, 1.0])
        bound = commutator_bound_first_order(h)
        # Only XI/ZI anticommute -> 2 * 1 * 1 = 2
        assert bound == 2.0

    def test_negative_coefficients(self):
        """Test that negative coefficients are handled via absolute values."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[-2.0, -3.0])
        bound = commutator_bound_first_order(h)
        assert bound == 12.0

    def test_single_term(self):
        """Test that a single term gives zero bound."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        bound = commutator_bound_first_order(h)
        assert bound == 0.0


class TestDoPauliLabelsQwCommute:
    """Tests for the do_pauli_labels_qw_commute function."""

    def test_identity_commutes(self):
        """Test that identity strings qubit-wise commute."""
        assert do_pauli_labels_qw_commute("II", "II") is True

    def test_different_qubits_commute(self):
        """Test operators on different qubits qw-commute."""
        assert do_pauli_labels_qw_commute("XI", "IX") is True

    def test_same_pauli_same_qubit(self):
        """Test same Pauli on same qubit qw-commutes."""
        assert do_pauli_labels_qw_commute("XI", "XI") is True

    def test_different_pauli_same_qubit(self):
        """Test different Paulis on same qubit do not qw-commute."""
        assert do_pauli_labels_qw_commute("XI", "YI") is False

    def test_commuting_but_not_qw_commuting(self):
        """Test that XY and YX commute globally but NOT qubit-wise."""
        # They commute (even number of anticommuting positions)
        assert do_pauli_labels_commute("XY", "YX") is True
        # But NOT qubit-wise (two positions differ)
        assert do_pauli_labels_qw_commute("XY", "YX") is False

    def test_different_length_raises(self):
        """Test that different-length labels raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            do_pauli_labels_qw_commute("XI", "XII")


class TestDoPauliMapsCommute:
    """Tests for the map-based do_pauli_maps_commute function."""

    def test_disjoint_qubits_commute(self):
        """Operators on different qubits commute."""
        assert do_pauli_maps_commute({0: "X"}, {1: "Y"}) is True

    def test_same_pauli_commutes(self):
        """Same Pauli on same qubit commutes."""
        assert do_pauli_maps_commute({0: "X"}, {0: "X"}) is True

    def test_single_anticommuting(self):
        """Different Paulis on same qubit anticommute."""
        assert do_pauli_maps_commute({0: "X"}, {0: "Y"}) is False

    def test_even_anticommuting_commutes(self):
        """Two anticommuting positions → commute."""
        a = {0: "X", 1: "Y"}
        b = {0: "Y", 1: "X"}
        assert do_pauli_maps_commute(a, b) is True

    def test_empty_terms_commute(self):
        """Empty terms commute with anything."""
        assert do_pauli_maps_commute({}, {0: "X"}) is True


class TestDoPauliMapsQwCommute:
    """Tests for the map-based do_pauli_maps_qw_commute function."""

    def test_disjoint_qubits(self):
        """Operators on different qubits qw-commute."""
        assert do_pauli_maps_qw_commute({0: "X"}, {1: "Y"}) is True

    def test_same_pauli(self):
        """Same Pauli on same qubit qw-commutes."""
        assert do_pauli_maps_qw_commute({0: "X"}, {0: "X"}) is True

    def test_different_pauli(self):
        """Different Paulis on same qubit do NOT qw-commute."""
        assert do_pauli_maps_qw_commute({0: "X"}, {0: "Y"}) is False

    def test_commuting_but_not_qw(self):
        """Even number of anticommuting positions: commute but NOT qw-commute."""
        a = {0: "X", 1: "Y"}
        b = {0: "Y", 1: "X"}
        assert do_pauli_maps_commute(a, b) is True
        assert do_pauli_maps_qw_commute(a, b) is False

    def test_same_paulis_overlapping(self):
        """Same Paulis on overlapping qubits qw-commute."""
        assert do_pauli_maps_qw_commute({0: "X", 1: "Z"}, {0: "X", 1: "Z"}) is True

    def test_one_differing_one_matching(self):
        """One differing, one matching position does not qw-commute."""
        assert do_pauli_maps_qw_commute({0: "X", 1: "Z"}, {0: "Y", 1: "Z"}) is False


class TestGetCommutationChecker:
    """Tests for get_commutation_checker factory."""

    def test_general(self):
        """Test that 'general' returns do_pauli_maps_commute."""
        fn = get_commutation_checker("general")
        assert fn is do_pauli_maps_commute

    def test_qubit_wise(self):
        """Test that 'qubit_wise' returns do_pauli_maps_qw_commute."""
        fn = get_commutation_checker("qubit_wise")
        assert fn is do_pauli_maps_qw_commute

    def test_invalid_raises(self):
        """Test that an invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown commutation_type"):
            get_commutation_checker("invalid")
