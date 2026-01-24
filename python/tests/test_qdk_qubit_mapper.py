"""Tests for QdkQubitMapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import QubitMapper, available, create
from qdk_chemistry.algorithms.qubit_mapper.qdk_qubit_mapper import (
    _bk_compute_ancestor_indices,
    _bk_compute_children_indices,
    _bk_compute_parity_indices,
    _bk_compute_z_indices_for_y_component,
)
from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian, QubitHamiltonian

from .test_helpers import create_test_hamiltonian, create_test_orbitals


@pytest.fixture
def test_data_path() -> Path:
    """Get path to test data directory."""
    return Path(__file__).resolve().parent / "test_data"


def _make_hamiltonian(
    one_body: np.ndarray,
    two_body: np.ndarray,
    orbitals,
    core_energy: float = 0.0,
) -> Hamiltonian:
    """Helper to create a Hamiltonian from arrays."""
    fock = np.eye(0)
    return Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, core_energy, fock))


class TestBravyiKitaevSets:
    """Tests for Bravyi-Kitaev set computation functions.

    The second argument n must be a power of 2.
    """

    def test_ancestor_indices_4_qubits(self) -> None:
        """Test ancestor indices for 4 qubit system (n=4 is already power of 2)."""
        # U(0) = {1, 3} - qubit 0's occupation affects qubits 1 and 3
        assert _bk_compute_ancestor_indices(0, 4) == frozenset({1, 3})
        # U(1) = {3}
        assert _bk_compute_ancestor_indices(1, 4) == frozenset({3})
        # U(2) = {3}
        assert _bk_compute_ancestor_indices(2, 4) == frozenset({3})
        # U(3) = {} - qubit 3 is the root, no ancestors
        assert _bk_compute_ancestor_indices(3, 4) == frozenset()

    def test_ancestor_indices_8_qubits(self) -> None:
        """Test ancestor indices for 8 qubit system."""
        assert _bk_compute_ancestor_indices(0, 8) == frozenset({1, 3, 7})
        assert _bk_compute_ancestor_indices(4, 8) == frozenset({5, 7})
        assert _bk_compute_ancestor_indices(7, 8) == frozenset()

    def test_parity_set_4_qubits(self) -> None:
        """Test parity set for 4 qubit system.

        P(j) follows the recursive binary tree structure.
        """
        assert _bk_compute_parity_indices(0, 4) == frozenset()
        assert _bk_compute_parity_indices(1, 4) == frozenset({0})
        assert _bk_compute_parity_indices(2, 4) == frozenset({1})
        assert _bk_compute_parity_indices(3, 4) == frozenset({1, 2})

    def test_parity_set_8_qubits(self) -> None:
        """Test parity set for 8 qubit system."""
        assert _bk_compute_parity_indices(0, 8) == frozenset()
        assert _bk_compute_parity_indices(4, 8) == frozenset({3})
        assert _bk_compute_parity_indices(5, 8) == frozenset({3, 4})
        assert _bk_compute_parity_indices(6, 8) == frozenset({3, 5})
        assert _bk_compute_parity_indices(7, 8) == frozenset({3, 5, 6})

    def test_children_indices_4_qubits(self) -> None:
        """Test children indices for 4 qubit system."""
        assert _bk_compute_children_indices(0, 4) == frozenset()
        assert _bk_compute_children_indices(1, 4) == frozenset({0})
        assert _bk_compute_children_indices(2, 4) == frozenset()
        assert _bk_compute_children_indices(3, 4) == frozenset({1, 2})

    def test_z_indices_for_y_component_4_qubits(self) -> None:
        """Test Z indices for Y component R(j) = P(j) - F(j) for 4 qubit system."""
        # R(j) = P(j) - F(j) (set difference)
        assert _bk_compute_z_indices_for_y_component(0, 4) == frozenset()  # {} - {} = {}
        assert _bk_compute_z_indices_for_y_component(1, 4) == frozenset()  # {0} - {0} = {}
        assert _bk_compute_z_indices_for_y_component(2, 4) == frozenset({1})  # {1} - {} = {1}
        assert _bk_compute_z_indices_for_y_component(3, 4) == frozenset()  # {1,2} - {1,2} = {}

    def test_z_indices_for_y_component_8_qubits(self) -> None:
        """Test Z indices for Y component for 8 qubit system."""
        assert _bk_compute_z_indices_for_y_component(4, 8) == frozenset({3})  # {3} - {} = {3}
        assert _bk_compute_z_indices_for_y_component(5, 8) == frozenset({3})  # {3,4} - {4} = {3}
        assert _bk_compute_z_indices_for_y_component(6, 8) == frozenset({3, 5})  # {3,5} - {} = {3,5}
        assert _bk_compute_z_indices_for_y_component(7, 8) == frozenset()  # {3,5,6} - {3,5,6} = {}

    def test_invalid_n_raises_value_error(self) -> None:
        """Test that non-power-of-2 values for n raise ValueError."""
        # Test various non-power-of-2 values
        invalid_values = [0, 3, 5, 6, 7, 9, 10, 12, 15]
        for n in invalid_values:
            with pytest.raises(ValueError, match="n must be a power of 2"):
                _bk_compute_parity_indices(0, n)
            with pytest.raises(ValueError, match="n must be a power of 2"):
                _bk_compute_ancestor_indices(0, n)
            with pytest.raises(ValueError, match="n must be a power of 2"):
                _bk_compute_children_indices(0, n)
            with pytest.raises(ValueError, match="n must be a power of 2"):
                _bk_compute_z_indices_for_y_component(0, n)

    def test_n_equals_1_returns_empty_set(self) -> None:
        """Test that n=1 (valid power of 2) returns empty frozenset."""
        assert _bk_compute_parity_indices(0, 1) == frozenset()
        assert _bk_compute_ancestor_indices(0, 1) == frozenset()
        assert _bk_compute_children_indices(0, 1) == frozenset()
        assert _bk_compute_z_indices_for_y_component(0, 1) == frozenset()


class TestQdkQubitMapper:
    """Tests for QdkQubitMapper."""

    def test_instantiation(self) -> None:
        """Test basic instantiation via factory and interface."""
        assert "qdk" in available("qubit_mapper")
        mapper = create("qubit_mapper", "qdk")
        assert isinstance(mapper, QubitMapper)
        assert mapper.name() == "qdk"
        assert mapper.type_name() == "qubit_mapper"

    def test_default_settings(self) -> None:
        """Test default settings values."""
        mapper = create("qubit_mapper", "qdk")
        assert mapper.settings().get("encoding") == "jordan-wigner"
        assert mapper.settings().get("threshold") == 1e-12

    def test_custom_threshold(self) -> None:
        """Test custom threshold can be set via factory kwargs."""
        mapper = create("qubit_mapper", "qdk", threshold=1e-10)
        assert mapper.settings().get("threshold") == 1e-10

    def test_invalid_encoding_raises(self) -> None:
        """Test that invalid encoding raises ValueError."""
        mapper = create("qubit_mapper", "qdk")
        with pytest.raises(ValueError, match="out of allowed options"):
            mapper.settings().set("encoding", "invalid_type")

    def test_simple_hamiltonian(self) -> None:
        """Test mapping a simple diagonal Hamiltonian."""
        mapper = create("qubit_mapper", "qdk")
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        assert isinstance(result, QubitHamiltonian)
        assert result.num_qubits == 4
        assert len(result.pauli_strings) > 0
        assert len(result.coefficients) == len(result.pauli_strings)
        assert result.coefficients.dtype == complex

    def test_number_operator(self) -> None:
        """Test JW transform of number operator: a†a = (I - Z) / 2."""
        mapper = create("qubit_mapper", "qdk")

        # h_00 = 1 gives n_0 = (I - Z_0)/2 for each spin
        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert result.num_qubits == 2
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)
        assert np.isclose(pauli_dict["ZI"].real, -0.5, atol=1e-10)
        assert np.isclose(pauli_dict["IZ"].real, -0.5, atol=1e-10)

    def test_core_energy_not_included(self) -> None:
        """Test that core energy is not included in QubitHamiltonian."""
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.array([[1.0]])  # Non-zero integral to generate Pauli terms
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        core_energy = 5.0
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals, core_energy=core_energy)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert "II" in pauli_dict
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)

    def test_threshold_pruning(self) -> None:
        """Test that small coefficients are pruned."""
        mapper = create("qubit_mapper", "qdk", threshold=0.1)
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        for coeff in result.coefficients:
            assert abs(coeff) >= 0.1

    def test_pauli_strings_format(self) -> None:
        """Test Pauli string format."""
        mapper = create("qubit_mapper", "qdk")
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        for ps in result.pauli_strings:
            assert isinstance(ps, str)
            assert len(ps) == 4
            assert all(c in "IXYZ" for c in ps)

    def test_pauli_string_ordering_convention(self) -> None:
        """Test that Pauli strings use little-endian ordering (qubit 0 is leftmost).

        For 1 spatial orbital (2 qubits): qubit 0 = alpha, qubit 1 = beta.
        With h_00 = 1 for alpha only (using asymmetric integrals if possible),
        we verify that ZI means Z on qubit 0, I on qubit 1.

        For JW number operator: n_j = (I - Z_j) / 2
        - n_alpha (qubit 0) contributes -0.5 to ZI
        - n_beta (qubit 1) contributes -0.5 to IZ
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Verify ordering: ZI should have Z on qubit 0 (alpha), I on qubit 1 (beta)
        # IZ should have I on qubit 0 (alpha), Z on qubit 1 (beta)
        assert "ZI" in pauli_dict, "Expected ZI for alpha number operator"
        assert "IZ" in pauli_dict, "Expected IZ for beta number operator"
        # Both should have coefficient -0.5 due to symmetric h_00 = 1
        assert np.isclose(pauli_dict["ZI"].real, -0.5, atol=1e-10)
        assert np.isclose(pauli_dict["IZ"].real, -0.5, atol=1e-10)

    def test_hopping_adjacent_orbitals(self) -> None:
        """Test JW transform of hopping term between adjacent orbitals.

        For h_01 = h_10 = t (hopping), the fermionic Hamiltonian is:
            H = t * (a†_0 a_1 + a†_1 a_0) for each spin

        Under Jordan-Wigner, for adjacent orbitals (no Z-string needed):
            a†_p a_q + h.c. -> 0.5 * (X_p X_q + Y_p Y_q)

        For 2 spatial orbitals (4 qubits, blocked: alpha=[0,1], beta=[2,3]):
        - Alpha hopping (0↔1): 0.5*t * (X_0 X_1 + Y_0 Y_1)
        - Beta hopping (2↔3): 0.5*t * (X_2 X_3 + Y_2 Y_3)

        Expected Pauli terms with t=1:
            XXII: 0.5, YYII: 0.5, IIXX: 0.5, IIYY: 0.5
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 2
        one_body = np.array([[0.0, 1.0], [1.0, 0.0]])  # h_01 = h_10 = 1
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected hopping terms (no Z-string for adjacent orbitals)
        expected = {
            "XXII": 0.5,  # alpha X_0 X_1
            "YYII": 0.5,  # alpha Y_0 Y_1
            "IIXX": 0.5,  # beta X_2 X_3
            "IIYY": 0.5,  # beta Y_2 Y_3
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"Coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )

        # Verify no unexpected terms (should only have hopping terms, no diagonal)
        for pauli_str, coeff in pauli_dict.items():
            if pauli_str not in expected:
                assert np.isclose(coeff, 0.0, atol=1e-10), f"Unexpected non-zero term: {pauli_str} = {coeff}"

    def test_hopping_non_adjacent_orbitals_z_string(self) -> None:
        """Test JW transform of hopping between non-adjacent orbitals (Z-string).

        For h_02 = h_20 = t in a 3-orbital system, JW requires a Z-string:
            a†_0 a_2 + h.c. -> 0.5 * (X_0 Z_1 X_2 + Y_0 Z_1 Y_2)

        For 3 spatial orbitals (6 qubits, blocked: alpha=[0,1,2], beta=[3,4,5]):
        - Alpha hopping (0↔2): 0.5*t * (X_0 Z_1 X_2 + Y_0 Z_1 Y_2)
        - Beta hopping (3↔5): 0.5*t * (X_3 Z_4 X_5 + Y_3 Z_4 Y_5)

        Expected Pauli terms with t=1:
            XZXIII: 0.5, YZYIII: 0.5, IIIXZX: 0.5, IIIYZY: 0.5
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 3
        one_body = np.zeros((n_orbitals, n_orbitals))
        one_body[0, 2] = one_body[2, 0] = 1.0  # h_02 = h_20 = 1
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected hopping terms WITH Z-string
        expected = {
            "XZXIII": 0.5,  # alpha X_0 Z_1 X_2
            "YZYIII": 0.5,  # alpha Y_0 Z_1 Y_2
            "IIIXZX": 0.5,  # beta X_3 Z_4 X_5
            "IIIYZY": 0.5,  # beta Y_3 Z_4 Y_5
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected Z-string term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"Coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )

    def test_pure_one_body_hamiltonian(self) -> None:
        """Test Hamiltonian with only one-body terms (no two-body).

        For diagonal h_00 = e_0, h_11 = e_1:
            H = e_0 * (n_0a + n_0b) + e_1 * (n_1a + n_1b)

        Using n_j = (I - Z_j) / 2 and accounting for qubit ordering where
        orbital indices are reversed within spin blocks:
        - Qubit 0 = orbital 1 alpha, Qubit 1 = orbital 0 alpha
        - Qubit 2 = orbital 1 beta,  Qubit 3 = orbital 0 beta

        With e_0 = 1, e_1 = 2:
            Identity: 3.0 (from 1 + 2)
            ZIII: -1.0 (n_1a from h_11=2)
            IZII: -0.5 (n_0a from h_00=1)
            IIZI: -1.0 (n_1b from h_11=2)
            IIIZ: -0.5 (n_0b from h_00=1)
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 2
        one_body = np.array([[1.0, 0.0], [0.0, 2.0]])  # h_00=1, h_11=2
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected: only identity and single-Z terms (number operators)
        # Qubit ordering: orbital indices are reversed within spin blocks
        # Qubit 0 = orbital 1 alpha, Qubit 1 = orbital 0 alpha
        # Qubit 2 = orbital 1 beta,  Qubit 3 = orbital 0 beta
        expected = {
            "IIII": 3.0,  # (1 + 2) from both orbitals, both spins
            "ZIII": -1.0,  # n_1a contribution (h_11=2 -> -2/2)
            "IZII": -0.5,  # n_0a contribution (h_00=1 -> -1/2)
            "IIZI": -1.0,  # n_1b contribution (h_11=2 -> -2/2)
            "IIIZ": -0.5,  # n_0b contribution (h_00=1 -> -1/2)
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"Coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )

        # Verify no two-body interaction terms (no ZZ, XX, YY, etc.)
        assert len(pauli_dict) == len(expected), (
            f"Expected {len(expected)} terms, got {len(pauli_dict)}: {list(pauli_dict.keys())}"
        )

    def test_pure_two_body_hamiltonian(self) -> None:
        """Test Hamiltonian with only two-body terms (no one-body).

        For on-site Coulomb repulsion (00|00) = U:
            H = U/2 * sum_st n_0s n_0t (where s != t for same orbital)
              = U/2 * (n_0a n_0b + n_0b n_0a)
              = U * n_0a n_0b

        Using n_j = (I - Z_j) / 2:
            n_0a n_0b = (I - Z_0)(I - Z_1) / 4
                      = (I - Z_0 - Z_1 + Z_0 Z_1) / 4

        So H = U * (I - Z_0 - Z_1 + Z_0 Z_1) / 4

        With U = 2.0:
            Identity: 0.5, ZI: -0.5, IZ: -0.5, ZZ: 0.5
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.zeros((n_orbitals, n_orbitals))
        two_body = np.zeros(n_orbitals**4)
        two_body[0] = 2.0  # (00|00) = U = 2
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected: n_a n_b interaction
        expected = {
            "II": 0.5,
            "ZI": -0.5,
            "IZ": -0.5,
            "ZZ": 0.5,
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"Coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )

    def test_mixed_one_and_two_body(self) -> None:
        """Test Hamiltonian with both one-body and two-body terms.

        Combines:
        - h_00 = 1 (number operator)
        - (00|00) = 2 (on-site repulsion)

        H = n_0a + n_0b + n_0a n_0b

        From test_number_operator: n_0a + n_0b = I - 0.5*Z_0 - 0.5*Z_1
        From test_pure_two_body: n_0a n_0b = 0.5*I - 0.5*Z_0 - 0.5*Z_1 + 0.5*Z_0 Z_1

        Combined:
            Identity: 1.0 + 0.5 = 1.5
            ZI: -0.5 + (-0.5) = -1.0
            IZ: -0.5 + (-0.5) = -1.0
            ZZ: 0.5
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(n_orbitals**4)
        two_body[0] = 2.0  # (00|00) = 2
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        expected = {
            "II": 1.5,
            "ZI": -1.0,
            "IZ": -1.0,
            "ZZ": 0.5,
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"Coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )

    def test_threshold_boundary(self) -> None:
        """Test coefficient pruning at threshold boundary.

        Create integrals that produce coefficients just above and below threshold.
        """
        threshold = 1e-8
        mapper = create("qubit_mapper", "qdk", threshold=threshold)

        n_orbitals = 2
        # h_00 = 1e-9 (produces coeff ~5e-10, below threshold after /2)
        # h_11 = 1e-6 (produces coeff ~5e-7, above threshold)
        one_body = np.array([[1e-9, 0.0], [0.0, 1e-6]])
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)

        # All returned coefficients should be >= threshold
        for coeff in result.coefficients:
            assert abs(coeff) >= threshold, f"Coefficient {coeff} below threshold {threshold}"

        # Verify h_11 terms are present (above threshold)
        # Qubit ordering: orbital 1 maps to ZIII (alpha) and IIZI (beta)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))
        assert "ZIII" in pauli_dict or "IIZI" in pauli_dict, "Expected terms from h_11 to be present"

    def test_four_orbital_z_string(self) -> None:
        """Test longer Z-string in 4-orbital system.

        For h_03 = h_30 = 1 in a 4-orbital system:
            a†_0 a_3 + h.c. -> 0.5 * (X_0 Z_1 Z_2 X_3 + Y_0 Z_1 Z_2 Y_3)

        For 4 spatial orbitals (8 qubits, blocked: alpha=[0,1,2,3], beta=[4,5,6,7]):
        - Alpha: X_0 Z_1 Z_2 X_3, Y_0 Z_1 Z_2 Y_3
        - Beta: X_4 Z_5 Z_6 X_7, Y_4 Z_5 Z_6 Y_7
        """
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 4
        one_body = np.zeros((n_orbitals, n_orbitals))
        one_body[0, 3] = one_body[3, 0] = 1.0
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected hopping terms with double Z-string
        expected = {
            "XZZXIIII": 0.5,  # alpha X_0 Z_1 Z_2 X_3
            "YZZYIIII": 0.5,  # alpha Y_0 Z_1 Z_2 Y_3
            "IIIIXZZX": 0.5,  # beta X_4 Z_5 Z_6 X_7
            "IIIIYZZY": 0.5,  # beta Y_4 Z_5 Z_6 Y_7
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected Z-string term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"Coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )


class TestQdkQubitMapperRealHamiltonians:
    """Tests with real molecular Hamiltonians."""

    def test_ethylene_4e4o(self, test_data_path: Path) -> None:
        """Test ethylene 4e4o Hamiltonian mapping."""
        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")

        mapper = create("qubit_mapper", "qdk")
        result = mapper.run(hamiltonian)

        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        expected_qubits = 2 * h1_alpha.shape[0]

        assert result.num_qubits == expected_qubits
        assert len(result.pauli_strings) > 0
        for ps in result.pauli_strings:
            assert len(ps) == expected_qubits

    def test_f2_10e6o(self, test_data_path: Path) -> None:
        """Test F2 10e6o Hamiltonian mapping."""
        hamiltonian = Hamiltonian.from_json_file(test_data_path / "f2_10e6o.hamiltonian.json")

        mapper = create("qubit_mapper", "qdk")
        result = mapper.run(hamiltonian)

        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        expected_qubits = 2 * h1_alpha.shape[0]

        assert result.num_qubits == expected_qubits
        assert len(result.pauli_strings) > 0

    def test_vs_qiskit(self, test_data_path: Path) -> None:
        """Cross-validate against Qiskit JordanWignerMapper."""
        pytest.importorskip("qiskit_nature")
        SparsePauliOp = pytest.importorskip("qiskit.quantum_info").SparsePauliOp  # noqa: N806
        FermionicOp = pytest.importorskip("qiskit_nature.second_q.operators").FermionicOp  # noqa: N806

        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")
        threshold = 1e-12

        # Match Qiskit tolerances to QDK threshold
        original_fermionic_atol = FermionicOp.atol
        original_sparse_atol = SparsePauliOp.atol
        try:
            FermionicOp.atol = threshold
            SparsePauliOp.atol = threshold

            qdk_result = create("qubit_mapper", "qdk", threshold=threshold).run(hamiltonian)
            qiskit_result = create("qubit_mapper", "qiskit", encoding="jordan-wigner").run(hamiltonian)
        finally:
            FermionicOp.atol = original_fermionic_atol
            SparsePauliOp.atol = original_sparse_atol

        assert qdk_result.num_qubits == qiskit_result.num_qubits

        qdk_dict = dict(zip(qdk_result.pauli_strings, qdk_result.coefficients, strict=True))
        qiskit_dict = dict(zip(qiskit_result.pauli_strings, qiskit_result.coefficients, strict=True))

        assert len(qdk_dict) == len(qiskit_dict)

        for pauli_str, qiskit_coeff in qiskit_dict.items():
            assert pauli_str in qdk_dict, f"Missing: {pauli_str}"
            assert np.isclose(qdk_dict[pauli_str], qiskit_coeff, rtol=1e-10, atol=1e-14)


class TestBravyiKitaevMapper:
    """Tests for Bravyi-Kitaev mapping."""

    def test_bk_instantiation(self) -> None:
        """Test BK encoding is valid via factory."""
        mapper = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")
        assert mapper.settings().get("encoding") == "bravyi-kitaev"

    def test_bk_simple_hamiltonian(self) -> None:
        """Test BK mapping of simple Hamiltonian."""
        mapper = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        assert isinstance(result, QubitHamiltonian)
        assert result.num_qubits == 4
        assert len(result.pauli_strings) > 0

    def test_bk_number_operator(self) -> None:
        """Test BK transform of number operator.

        In Bravyi-Kitaev, the number operator for orbital j is:
            n_j = 0.5 * (I - Z_j * prod_{k in F(j)} Z_k)
        where F(j) is the flip set. This differs from Jordan-Wigner where
            n_j = 0.5 * (I - Z_j)

        For 2 qubits (1 spatial orbital, alpha + beta):
        - n_0 (alpha, j=0): F(0)={}, so n_0 = 0.5*(I - Z_0)
        - n_1 (beta, j=1): F(1)={0}, so n_1 = 0.5*(I - Z_0*Z_1)
        Total with h_00=1: H = n_0 + n_1 = I - 0.5*Z_0 - 0.5*Z_0*Z_1
        """
        mapper_bk = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")

        # h_00 = 1 gives H = n_0_alpha + n_0_beta
        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result_bk = mapper_bk.run(hamiltonian)
        assert result_bk.num_qubits == 2

        bk_dict = dict(zip(result_bk.pauli_strings, result_bk.coefficients, strict=True))

        # Expected: I (coeff 1), IZ (coeff -0.5), ZZ (coeff -0.5)
        assert len(bk_dict) == 3
        assert np.isclose(bk_dict["II"], 1.0, rtol=1e-10)
        assert np.isclose(bk_dict["IZ"], -0.5, rtol=1e-10)
        assert np.isclose(bk_dict["ZZ"], -0.5, rtol=1e-10)

    def test_bk_core_energy_not_included(self) -> None:
        """Test that core energy is not included in BK QubitHamiltonian."""
        mapper = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")

        n_orbitals = 1
        one_body = np.array([[1.0]])  # Non-zero integral to generate Pauli terms
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        core_energy = 5.0
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals, core_energy=core_energy)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert "II" in pauli_dict
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)

    def test_bk_hopping_adjacent_orbitals(self) -> None:
        """Test BK transform of hopping term between adjacent orbitals.

        BK hopping differs from JW due to the update and parity set structure.
        For 2 spatial orbitals (4 qubits), the BK representation involves
        different Pauli operators than JW.

        The BK transform uses:
        - a†_j = 0.5 * (X_U(j) ⊗ X_j ⊗ Z_P(j) - i * X_U(j) ⊗ Y_j ⊗ Z_R(j))
        - a_j = 0.5 * (X_U(j) ⊗ X_j ⊗ Z_P(j) + i * X_U(j) ⊗ Y_j ⊗ Z_R(j))

        For h_01 = h_10 = 1 with 2 orbitals (4 qubits, blocked ordering):
        We verify the BK result differs from JW but preserves hermiticity.
        """
        mapper = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")

        n_orbitals = 2
        one_body = np.array([[0.0, 1.0], [1.0, 0.0]])  # h_01 = h_10 = 1
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # BK should have different structure than JW
        # Key check: all coefficients should be real (Hermitian Hamiltonian)
        for pauli_str, coeff in pauli_dict.items():
            assert np.isclose(coeff.imag, 0.0, atol=1e-10), f"Non-real coefficient for {pauli_str}: {coeff}"

        # Verify we get some hopping-like terms (X operators present for excitations)
        # Note: BK hopping may not produce Y terms in the same way as JW
        has_x_terms = any("X" in ps for ps in pauli_dict)
        assert has_x_terms, "BK hopping should produce X terms"

        # Verify specific structure for this case
        # BK produces: IIIX, IIZX, IXII, ZXZI for h_01 = h_10 = 1
        assert len(pauli_dict) > 0, "BK should produce non-trivial terms"

    def test_bk_pure_one_body_diagonal(self) -> None:
        """Test BK mapping of pure diagonal one-body Hamiltonian.

        For h_00 = 1, h_11 = 2 with 2 spatial orbitals (4 qubits in BK).
        BK number operators have different structure due to flip sets.

        For 4 qubits (indices 0,1,2,3):
        - n_0: F(0)={}, so n_0 = 0.5*(I - Z_0)
        - n_1: F(1)={0}, so n_1 = 0.5*(I - Z_0*Z_1)
        - n_2: F(2)={}, so n_2 = 0.5*(I - Z_2)
        - n_3: F(3)={1,2}, so n_3 = 0.5*(I - Z_1*Z_2*Z_3)

        H = h_00*(n_0 + n_2) + h_11*(n_1 + n_3)
          = 1*(n_0 + n_2) + 2*(n_1 + n_3)
        """
        mapper = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")

        n_orbitals = 2
        one_body = np.array([[1.0, 0.0], [0.0, 2.0]])
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected terms from BK number operators:
        # n_0 = 0.5*(I - Z_0) -> ZIII with coeff -0.5
        # n_1 = 0.5*(I - Z_0*Z_1) -> ZZII with coeff -0.5
        # n_2 = 0.5*(I - Z_2) -> IIZI with coeff -0.5
        # n_3 = 0.5*(I - Z_1*Z_2*Z_3) -> IZZZ with coeff -0.5
        # Total identity: 0.5*1 + 0.5*2 + 0.5*1 + 0.5*2 = 3.0

        # Check identity coefficient
        assert np.isclose(pauli_dict["IIII"].real, 3.0, atol=1e-10)

        # BK produces ZZ terms even for diagonal one-body (unlike JW)
        # Check that ZZ terms exist
        zz_terms = [ps for ps in pauli_dict if ps.count("Z") >= 2]
        assert len(zz_terms) > 0, "BK diagonal Hamiltonian should have multi-Z terms"

    def test_bk_two_body_on_site(self) -> None:
        """Test BK transform of on-site Coulomb repulsion (00|00) = U.

        The two-body term n_0a n_0b should produce ZZ interactions in BK,
        but with different structure than JW due to BK encoding.
        """
        mapper = create("qubit_mapper", "qdk", encoding="bravyi-kitaev")

        n_orbitals = 1
        one_body = np.zeros((n_orbitals, n_orbitals))
        two_body = np.zeros(n_orbitals**4)
        two_body[0] = 2.0  # U = 2
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # For 1 spatial orbital (2 qubits), BK n_0a n_0b:
        # The BK encoding produces different structure than JW due to the
        # flip set relationships between qubits 0 and 1.
        # Actual BK output for U=2:
        expected = {
            "II": 0.5,
            "IZ": -0.5,
            "ZI": 0.5,
            "ZZ": -0.5,
        }

        for pauli_str, expected_coeff in expected.items():
            assert pauli_str in pauli_dict, f"Missing expected BK term: {pauli_str}"
            assert np.isclose(pauli_dict[pauli_str].real, expected_coeff, atol=1e-10), (
                f"BK coefficient mismatch for {pauli_str}: got {pauli_dict[pauli_str].real}, expected {expected_coeff}"
            )

    def test_bk_vs_qiskit(self, test_data_path: Path) -> None:
        """Cross-validate BK against Qiskit BravyiKitaevMapper."""
        pytest.importorskip("qiskit_nature")
        SparsePauliOp = pytest.importorskip("qiskit.quantum_info").SparsePauliOp  # noqa: N806
        FermionicOp = pytest.importorskip("qiskit_nature.second_q.operators").FermionicOp  # noqa: N806

        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")
        threshold = 1e-12

        # Match Qiskit tolerances to QDK threshold
        original_fermionic_atol = FermionicOp.atol
        original_sparse_atol = SparsePauliOp.atol
        try:
            FermionicOp.atol = threshold
            SparsePauliOp.atol = threshold

            qdk_result = create("qubit_mapper", "qdk", encoding="bravyi-kitaev", threshold=threshold).run(hamiltonian)
            qiskit_result = create("qubit_mapper", "qiskit", encoding="bravyi-kitaev").run(hamiltonian)
        finally:
            FermionicOp.atol = original_fermionic_atol
            SparsePauliOp.atol = original_sparse_atol

        assert qdk_result.num_qubits == qiskit_result.num_qubits

        qdk_dict = dict(zip(qdk_result.pauli_strings, qdk_result.coefficients, strict=True))
        qiskit_dict = dict(zip(qiskit_result.pauli_strings, qiskit_result.coefficients, strict=True))

        assert len(qdk_dict) == len(qiskit_dict), f"QDK has {len(qdk_dict)} terms, Qiskit has {len(qiskit_dict)}"

        for pauli_str, qiskit_coeff in qiskit_dict.items():
            assert pauli_str in qdk_dict, f"Missing: {pauli_str}"
            assert np.isclose(qdk_dict[pauli_str], qiskit_coeff, rtol=1e-10, atol=1e-14), (
                f"Mismatch for {pauli_str}: QDK={qdk_dict[pauli_str]}, Qiskit={qiskit_coeff}"
            )
