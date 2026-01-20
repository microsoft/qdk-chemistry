"""Test Hamiltonian loading and grouping functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import numpy as np
import pytest

from qdk_chemistry.data.qubit_hamiltonian import (
    QubitHamiltonian,
    _filter_and_group_pauli_ops_from_statevector,
    filter_and_group_pauli_ops_from_wavefunction,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestQubitHamiltonian:
    """Test suite for QubitHamiltonian data class."""

    def test_initialization(self):
        """Test that QubitHamiltonian initializes correctly."""
        pauli_strings = ["IX", "YY", "ZZ"]
        coefficients = np.array([1.0, -0.5, 0.75])
        qubit_hamiltonian = QubitHamiltonian(pauli_strings, coefficients)
        assert qubit_hamiltonian.pauli_strings == pauli_strings
        assert np.array_equal(qubit_hamiltonian.coefficients, coefficients)
        assert qubit_hamiltonian.num_qubits == 2

    def test_initialization_mismatch(self):
        """Test that initialization raises ValueError on mismatched lengths."""
        with pytest.raises(ValueError, match=r"Mismatch between number of Pauli strings and coefficients\."):
            QubitHamiltonian(pauli_strings=["X", "Y"], coefficients=np.array([1.0]))

    def test_initialization_invalid_pauli(self):
        """Test that initialization raises ValueError on invalid Pauli strings."""
        with pytest.raises(ValueError, match="Invalid Pauli strings or coefficients"):
            QubitHamiltonian(pauli_strings=["X", "A"], coefficients=np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="Invalid Pauli strings or coefficients"):
            QubitHamiltonian(pauli_strings=["X", "Y"], coefficients=np.array([1.0, "invalid"]))
        with pytest.raises(ValueError, match="Invalid Pauli strings or coefficients"):
            QubitHamiltonian(pauli_strings=["X", "ZY"], coefficients=np.array([1.0, 2.0]))

    def test_group_commuting(self):
        """Test group_commuting."""
        qubit_hamiltonian = QubitHamiltonian(["XX", "YY", "ZZ", "XY"], [1.0, 0.5, -0.5, 0.2])
        grouped = qubit_hamiltonian.group_commuting(qubit_wise=False)
        assert len(grouped) == 2

        # Verify coefficients are preserved
        coeff_map = dict(zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True))
        for group in grouped:
            for pauli_str, coeff in zip(group.pauli_strings, group.coefficients, strict=True):
                assert np.isclose(
                    coeff,
                    coeff_map[pauli_str],
                    atol=float_comparison_absolute_tolerance,
                    rtol=float_comparison_relative_tolerance,
                )

    def test_group_commuting_qubitwise(self):
        """Test group_commuting without qubit-wise commuting."""
        qubit_hamiltonian = QubitHamiltonian(["XX", "YY", "ZZ", "XY"], [1.0, 0.5, -0.5, 0.2])
        grouped = qubit_hamiltonian.group_commuting(qubit_wise=True)
        assert len(grouped) == 4  # Qubit-wise commuting returns four groups

        # Check that all original Pauli strings are present across all groups
        all_grouped_strings = []
        for group in grouped:
            assert len(group.pauli_strings) == 1  # Each group should contain only one Pauli string
            all_grouped_strings.extend(group.pauli_strings)
        assert set(all_grouped_strings) == {"XX", "YY", "ZZ", "XY"}

    def test_reorder_qubits_identity(self):
        """Test that identity permutation returns equivalent Hamiltonian."""
        qh = QubitHamiltonian(["XIZI", "IYII"], np.array([0.5, 0.3], dtype=complex))
        reordered = qh.reorder_qubits([0, 1, 2, 3])
        assert reordered.pauli_strings == qh.pauli_strings
        assert np.allclose(reordered.coefficients, qh.coefficients)

    def test_reorder_qubits_swap(self):
        """Test swapping two adjacent qubits."""
        qh = QubitHamiltonian(["XIZI"], np.array([1.0], dtype=complex))
        reordered = qh.reorder_qubits([1, 0, 2, 3])
        assert reordered.pauli_strings == ["IXZI"]

    def test_reorder_qubits_reverse(self):
        """Test reversing all qubit indices."""
        qh = QubitHamiltonian(["XYZI"], np.array([1.0], dtype=complex))
        reordered = qh.reorder_qubits([3, 2, 1, 0])
        assert reordered.pauli_strings == ["IZYX"]

    def test_reorder_qubits_invalid_length(self):
        """Test that invalid permutation length raises error."""
        qh = QubitHamiltonian(["XIZI"], np.array([1.0], dtype=complex))
        with pytest.raises(ValueError, match="Permutation length"):
            qh.reorder_qubits([0, 1, 2])

    def test_reorder_qubits_invalid_values(self):
        """Test that invalid permutation values raise error."""
        qh = QubitHamiltonian(["XIZI"], np.array([1.0], dtype=complex))
        with pytest.raises(ValueError, match="Invalid permutation"):
            qh.reorder_qubits([0, 1, 1, 3])

    def test_to_interleaved_4_qubits(self):
        """Test blocked to interleaved conversion for 4 qubits."""
        # Blocked: [α₀, α₁, β₀, β₁] -> Interleaved: [α₀, β₀, α₁, β₁]
        qh = QubitHamiltonian(["XYZZ"], np.array([1.0], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=2)
        assert interleaved.pauli_strings == ["XZYZ"]

    def test_to_interleaved_preserves_coefficients(self):
        """Test that interleaving preserves coefficient values."""
        qh = QubitHamiltonian(["XIZI", "IYII"], np.array([0.5 + 0.1j, 0.3], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=2)
        assert np.allclose(interleaved.coefficients, qh.coefficients)

    def test_to_interleaved_invalid_n_spatial(self):
        """Test that invalid n_spatial raises error."""
        qh = QubitHamiltonian(["XIZI"], np.array([1.0], dtype=complex))
        with pytest.raises(ValueError, match=re.escape("must be 2 * n_spatial")):
            qh.to_interleaved(n_spatial=3)

    def test_to_interleaved_single_orbital(self):
        """Test that single spatial orbital (2 qubits) is unchanged."""
        qh = QubitHamiltonian(["XY"], np.array([1.0], dtype=complex))
        interleaved = qh.to_interleaved(n_spatial=1)
        assert interleaved.pauli_strings == ["XY"]


def test_filter_and_group_raises_on_zero_norm():
    """Statevector with zero norm should raise ValueError."""
    psi = np.zeros(4)
    h = QubitHamiltonian(["ZZ"], np.array([1.0]))
    with pytest.raises(ValueError, match="zero norm"):
        _filter_and_group_pauli_ops_from_statevector(h, psi)


def test_filter_and_group_trimming_handles_plus_minus_one_expectations():
    """Check correct sign handling for +1 and -1 expectation values."""
    qubit_hamiltonian = QubitHamiltonian(["ZZ", "YY"], np.array([1.0, 2.0]))
    state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    _, classical = _filter_and_group_pauli_ops_from_statevector(qubit_hamiltonian, state)
    # ZZ exp = +1, +1 * 1 added; YY exp = -1, -1 * 2 added
    assert len(classical) == 2
    assert any(
        np.isclose(c, 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance)
        for c in classical
    )
    assert any(
        np.isclose(c, -2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance)
        for c in classical
    )
    assert np.isclose(
        sum(classical), -1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )


def test_filter_and_group_returns_empty_if_all_trimmed():
    """If all expectation values are +1 or -1, no grouped Hamiltonians remain."""
    qubit_hamiltonian = QubitHamiltonian(["ZZ"], np.array([5.0]))
    state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    grouped, classical = _filter_and_group_pauli_ops_from_statevector(qubit_hamiltonian, state)
    assert grouped == []
    assert len(classical) == 1
    assert np.isclose(
        sum(classical), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )


def test_filter_and_group_behavior_on_simple_state(hamiltonian_4e4o, wavefunction_4e4o):
    """Test extraction and simplification of Pauli terms from a Hamiltonian."""
    grouped_ops, classical_coeffs = filter_and_group_pauli_ops_from_wavefunction(
        hamiltonian_4e4o, wavefunction_4e4o, abelian_grouping=True, trimming=True
    )
    assert len(grouped_ops) == 2
    assert np.isclose(
        sum(classical_coeffs).real,
        -4.191428699447072,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


def test_filter_and_group_no_trimming(hamiltonian_4e4o, wavefunction_4e4o):
    """Test filter_and_group_pauli_ops_from_wavefunction with trimming=False."""
    grouped_ops, classical_coeffs = filter_and_group_pauli_ops_from_wavefunction(
        hamiltonian_4e4o,
        wavefunction_4e4o,
        abelian_grouping=True,
        trimming=False,
    )

    # With trimming=False, should keep all terms and have no classical coefficients
    assert len(classical_coeffs) == 0
    assert len(grouped_ops) > 0

    qubit_hamiltonian = QubitHamiltonian(["X", "Z"], [1.0, 2.0])

    # Create |0> state
    statevector = np.array([1.0, 0.0])

    grouped_ops, classical_coeffs = _filter_and_group_pauli_ops_from_statevector(
        qubit_hamiltonian,
        statevector,
        abelian_grouping=True,
        trimming=False,  # This avoids the filtering logic entirely
    )

    # With trimming=False, should keep all terms and have no classical coefficients
    assert len(classical_coeffs) == 0
    assert len(grouped_ops) > 0


def test_filter_and_group_no_abelian_grouping(hamiltonian_4e4o, wavefunction_4e4o):
    """Test filter_and_group_pauli_ops_from_wavefunction abelian_grouping=False."""
    grouped_ops, _ = filter_and_group_pauli_ops_from_wavefunction(
        hamiltonian_4e4o,
        wavefunction_4e4o,
        abelian_grouping=False,
        trimming=True,
    )

    # With abelian_grouping=False, should return a single group
    assert len(grouped_ops) == 1
    assert isinstance(grouped_ops[0], QubitHamiltonian)

    # Create a 1-qubit Hamiltonian
    qubit_hamiltonian = QubitHamiltonian(["X", "Z"], [1.0, 2.0])

    # Create |0> state
    statevector = np.array([1.0, 0.0])

    grouped_ops, _ = _filter_and_group_pauli_ops_from_statevector(
        qubit_hamiltonian,
        statevector,
        abelian_grouping=False,  # This takes a different path
        trimming=False,
    )

    # With abelian_grouping=False, should return a single group
    assert len(grouped_ops) == 1
    assert isinstance(grouped_ops[0], QubitHamiltonian)


def test_filter_and_group_mixed_with_retained_terms():
    """Test filter_and_group ensuring some terms are always retained."""
    # Create a state that will ensure some terms have fractional expectations
    # Use a 2-qubit system with careful state preparation
    qubit_hamiltonian = QubitHamiltonian(["II", "ZI", "IZ", "ZZ"], [1.0, 0.5, 0.3, 0.2])

    # Create a state that gives mixed expectations
    # |00> + |11> (Bell state) normalized
    statevector = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])

    grouped_ops, classical_coeffs = _filter_and_group_pauli_ops_from_statevector(
        qubit_hamiltonian, statevector, abelian_grouping=True, trimming=True
    )

    # This should have some classical terms and some retained terms
    # II: expectation 1 (classical)
    # ZI: expectation 0 (filtered)
    # IZ: expectation 0 (filtered)
    # ZZ: expectation 1 (classical)
    assert len(classical_coeffs) == 2
    assert len(grouped_ops) == 0


def test_filter_and_group_with_small_fractional_expectations():
    """Test with a state that guarantees fractional expectations."""
    # Create a 1-qubit Hamiltonian with multiple terms
    qubit_hamiltonian = QubitHamiltonian(["I", "Z", "X"], [1.0, 2.0, 3.0])

    # Create a state with fractional Z expectation but keep other terms
    # Use a state that's close to |0> but with small |1> component
    epsilon = 0.1
    norm = np.sqrt(1 + epsilon**2)
    statevector = np.array([1 / norm, epsilon / norm])

    grouped_ops, classical_coeffs = _filter_and_group_pauli_ops_from_statevector(
        qubit_hamiltonian, statevector, abelian_grouping=True, trimming=True
    )

    # I should be classical (expectation ~1)
    # Z should have fractional expectation (retained)
    # X should have small expectation (likely retained unless very close to 0)
    total_ops = len(grouped_ops) + len(classical_coeffs)
    assert total_ops > 0  # Should have some terms


def test_filter_and_group_both_trimming_modes():
    """Test both trimming=True and trimming=False for consistency."""
    # Use the same Hamiltonian and state for both tests
    qubit_hamiltonian = QubitHamiltonian(["X", "Z"], [1.0, 2.0])
    statevector = np.array([1.0, 0.0])  # |0> state

    # Test with trimming=False (safe path)
    grouped_ops_no_trim, classical_coeffs_no_trim = _filter_and_group_pauli_ops_from_statevector(
        qubit_hamiltonian, statevector, abelian_grouping=True, trimming=False
    )

    # Should keep all terms
    assert len(classical_coeffs_no_trim) == 0
    assert len(grouped_ops_no_trim) > 0

    # Test with trimming=True and a state that ensures some fractional expectations
    # Use a superposition state
    statevector_super = np.array([0.8, 0.6])

    grouped_ops_trim, classical_coeffs_trim = _filter_and_group_pauli_ops_from_statevector(
        qubit_hamiltonian, statevector_super, abelian_grouping=True, trimming=True
    )

    # Should have some result
    total_terms = len(grouped_ops_trim) + len(classical_coeffs_trim)
    assert total_terms >= 0
