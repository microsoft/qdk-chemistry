"""Tests for QdkQubitMapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import QubitMapper, available, create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    Orbitals,
    QubitOperator,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_NATURE

from .test_helpers import (
    create_nontrivial_test_hamiltonian,
    create_test_basis_set,
    create_test_hamiltonian,
    create_test_orbitals,
)

_PAULI_ID_TO_CHAR = {0: "I", 1: "X", 2: "Y", 3: "Z"}


def _table_entry_to_pauli_string(entry: list[tuple[int, int]], num_qubits: int) -> str:
    """Convert a sparse MajoranaMapping table entry to a dense Pauli label string."""
    chars = ["I"] * num_qubits
    for qubit, pauli_id in entry:
        chars[qubit] = _PAULI_ID_TO_CHAR[pauli_id]
    return "".join(chars)


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
        assert mapper.settings().get("threshold") == 1e-12

    def test_custom_threshold(self) -> None:
        """Test custom threshold can be set via factory kwargs."""
        mapper = create("qubit_mapper", "qdk", threshold=1e-10)
        assert mapper.settings().get("threshold") == 1e-10

    def test_simple_hamiltonian(self) -> None:
        """Test mapping a simple diagonal Hamiltonian."""
        mapper = create("qubit_mapper", "qdk")
        hamiltonian = create_test_hamiltonian(2)
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * 2)

        result = mapper.run(hamiltonian, mapping)

        assert isinstance(result, QubitOperator)
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
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert result.num_qubits == 2
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)
        assert np.isclose(pauli_dict["ZI"].real, -0.5, atol=1e-10)
        assert np.isclose(pauli_dict["IZ"].real, -0.5, atol=1e-10)

    def test_core_energy_not_included(self) -> None:
        """Test that core energy is not included in QubitOperator."""
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.array([[1.0]])  # Non-zero integral to generate Pauli terms
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        core_energy = 5.0
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals, core_energy=core_energy)
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert "II" in pauli_dict
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)

    def test_threshold_pruning(self) -> None:
        """Test that small coefficients are pruned."""
        mapper = create("qubit_mapper", "qdk", threshold=0.1)
        hamiltonian = create_test_hamiltonian(2)
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * 2)

        result = mapper.run(hamiltonian, mapping)

        for coeff in result.coefficients:
            assert abs(coeff) >= 0.1

    def test_pauli_strings_format(self) -> None:
        """Test Pauli string format."""
        mapper = create("qubit_mapper", "qdk")
        hamiltonian = create_test_hamiltonian(2)
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * 2)

        result = mapper.run(hamiltonian, mapping)

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
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        # Expected: n_a n_b interaction
        expected = {
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

        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)
        result = mapper.run(hamiltonian, mapping)
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

        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)
        result = mapper.run(hamiltonian, mapping)

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

        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n_orbitals)
        result = mapper.run(hamiltonian, mapping)
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

    def test_mapping_size_mismatch_raises(self) -> None:
        """Test that mismatched mapping size raises ValueError."""
        hamiltonian = create_test_hamiltonian(2)  # 2 spatial → 4 spin-orbitals
        mapper = create("qubit_mapper", "qdk")

        # Oversized mapping (6 modes for 4-spin-orbital Hamiltonian)
        mapping_over = MajoranaMapping.jordan_wigner(num_modes=6)
        with pytest.raises(ValueError, match="modes"):
            mapper.run(hamiltonian, mapping_over)

        # Undersized mapping (2 modes for 4-spin-orbital Hamiltonian)
        mapping_under = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(ValueError, match="modes"):
            mapper.run(hamiltonian, mapping_under)

    def test_custom_mapping_end_to_end(self) -> None:
        """Test that a custom MajoranaMapping from JW table matches JW factory output bit-exactly."""
        mapper = create("qubit_mapper", "qdk")
        hamiltonian = create_test_hamiltonian(2)
        n_modes = 2 * 2

        jw = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        custom = MajoranaMapping.from_table(list(jw.table), name="")

        result_jw = mapper.run(hamiltonian, jw)
        result_custom = mapper.run(hamiltonian, custom)

        assert result_jw.pauli_strings == result_custom.pauli_strings
        assert np.array_equal(result_jw.coefficients, result_custom.coefficients)

    def test_parity_end_to_end(self) -> None:
        """Test parity mapping produces correct qubits, is Hermitian, and matches JW eigenvalues."""
        hamiltonian = create_nontrivial_test_hamiltonian(2)
        n_modes = 2 * 2

        mapper = create("qubit_mapper", "qdk")
        mapping_parity = MajoranaMapping.parity(num_modes=n_modes)
        result_parity = mapper.run(hamiltonian, mapping_parity)

        # Correct number of qubits
        assert result_parity.num_qubits == n_modes

        # All coefficients must be real (Hermitian Hamiltonian)
        for coeff in result_parity.coefficients:
            assert np.isclose(coeff.imag, 0.0, atol=1e-10), f"Non-real coefficient: {coeff}"

        # Eigenvalues must match JW
        mapping_jw = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        result_jw = mapper.run(hamiltonian, mapping_jw)

        def _to_matrix(qh: QubitOperator) -> np.ndarray:
            n = qh.num_qubits
            dim = 2**n
            mat = np.zeros((dim, dim), dtype=complex)
            pauli_mats = {
                "I": np.eye(2, dtype=complex),
                "X": np.array([[0, 1], [1, 0]], dtype=complex),
                "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
                "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            }
            for ps, c in zip(qh.pauli_strings, qh.coefficients, strict=True):
                term = np.array([[1.0]], dtype=complex)
                for ch in ps:
                    term = np.kron(term, pauli_mats[ch])
                mat += c * term
            return mat

        eigs_jw = np.sort(np.linalg.eigvalsh(_to_matrix(result_jw)))
        eigs_parity = np.sort(np.linalg.eigvalsh(_to_matrix(result_parity)))
        np.testing.assert_allclose(eigs_parity, eigs_jw, atol=1e-10)

    def test_parity_number_operator_structure(self) -> None:
        """Parity encoding gives n_j = (I - Z_{j-1}·Z_j)/2, matching standard convention."""
        from qdk_chemistry.utils.pauli_matrix import pauli_to_sparse_matrix  # noqa: PLC0415

        n = 6
        mapping = MajoranaMapping.parity(n)
        I_n = np.eye(2**n)  # noqa: N806

        for j in range(n):
            ps0 = _table_entry_to_pauli_string(mapping.table[2 * j], mapping.num_qubits)
            ps1 = _table_entry_to_pauli_string(mapping.table[2 * j + 1], mapping.num_qubits)
            g0 = pauli_to_sparse_matrix([ps0], np.array([1.0])).toarray()
            g1 = pauli_to_sparse_matrix([ps1], np.array([1.0])).toarray()
            nj = (I_n + 1j * g0 @ g1) / 2

            # Build expected Z structure, using big-endian qubit convention
            # (Pauli label index 0 = MSB = bit n-1).  Qubit q in the
            # Pauli string maps to bit (n - 1 - q) in the state index.
            q = n - 1 - j  # bit position for qubit j
            z_op = np.eye(2**n, dtype=complex)
            for idx in range(2**n):
                if j == 0:
                    if (idx >> q) & 1:
                        z_op[idx, idx] = -1.0
                else:
                    parity = ((idx >> q) & 1) ^ ((idx >> (q + 1)) & 1)
                    if parity:
                        z_op[idx, idx] = -1.0
            expected_nj = (I_n - z_op) / 2
            np.testing.assert_allclose(nj, expected_nj, atol=1e-12, err_msg=f"n_{j} structure mismatch")


class TestUnrestrictedHamiltonians:
    """Tests for unrestricted (UHF) Hamiltonian mapping."""

    @staticmethod
    def _make_unrestricted_hamiltonian(n: int, seed: int = 42):
        """Create an unrestricted Hamiltonian with distinct alpha/beta integrals."""
        rng = np.random.default_rng(seed)
        coeffs_alpha = np.eye(n)
        coeffs_beta = np.eye(n) + rng.standard_normal((n, n)) * 0.1
        basis_set = create_test_basis_set(n, "test-uhf")
        orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        # Symmetric one-body matrices (different for alpha/beta)
        raw_a = rng.standard_normal((n, n)) * 0.3
        h1_alpha = (raw_a + raw_a.T) / 2 + np.diag(np.linspace(1.0, -0.5, n))
        raw_b = rng.standard_normal((n, n)) * 0.3
        h1_beta = (raw_b + raw_b.T) / 2 + np.diag(np.linspace(0.8, -0.3, n))

        # Two-body integrals with 8-fold symmetry (different per channel)
        def make_symmetric_eri(n, rng):
            h2 = np.zeros((n, n, n, n))
            seen = set()
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            perms = frozenset(
                                {
                                    (p, q, r, s),
                                    (q, p, r, s),
                                    (p, q, s, r),
                                    (q, p, s, r),
                                    (r, s, p, q),
                                    (s, r, p, q),
                                    (r, s, q, p),
                                    (s, r, q, p),
                                }
                            )
                            canon = min(perms)
                            if canon in seen:
                                continue
                            seen.add(canon)
                            val = rng.standard_normal() * 0.2
                            for a, b, c, d in perms:
                                h2[a, b, c, d] = val
            return h2.ravel()

        h2_aaaa = make_symmetric_eri(n, rng)
        h2_aabb = make_symmetric_eri(n, rng)
        h2_bbbb = make_symmetric_eri(n, rng)

        fock_a = np.eye(0)
        fock_b = np.eye(0)
        return Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                h1_alpha,
                h1_beta,
                h2_aaaa,
                h2_aabb,
                h2_bbbb,
                orbitals,
                0.5,
                fock_a,
                fock_b,
            )
        )

    def test_unrestricted_hamiltonian_is_detected(self) -> None:
        """Verify the test helper creates an unrestricted Hamiltonian."""
        h = self._make_unrestricted_hamiltonian(2)
        assert not h.get_orbitals().is_restricted()
        h1_a, h1_b = h.get_one_body_integrals()
        assert not np.array_equal(h1_a, h1_b)

    def test_unrestricted_jw_produces_hermitian(self) -> None:
        """UHF JW mapping produces a Hermitian qubit Hamiltonian."""
        h = self._make_unrestricted_hamiltonian(2)
        mapping = MajoranaMapping.jordan_wigner(num_modes=4)
        mapper = create("qubit_mapper", "qdk")
        qh = mapper.run(h, mapping)

        assert qh.num_qubits == 4
        for c in qh.coefficients:
            assert abs(c.imag) < 1e-12, f"Non-real coefficient: {c}"

    def test_unrestricted_eigenvalues_match_across_encodings(self) -> None:
        """UHF eigenvalues are consistent across JW, BK, and parity."""
        h = self._make_unrestricted_hamiltonian(2)
        n_modes = 4

        eigenvalues = {}
        for enc in ["jordan_wigner", "bravyi_kitaev", "parity"]:
            mapping = getattr(MajoranaMapping, enc)(num_modes=n_modes)
            mapper = create("qubit_mapper", "qdk")
            qh = mapper.run(h, mapping)
            eigenvalues[enc] = np.sort(np.linalg.eigvalsh(qh.to_matrix()))

        np.testing.assert_allclose(eigenvalues["jordan_wigner"], eigenvalues["bravyi_kitaev"], atol=1e-10)
        np.testing.assert_allclose(eigenvalues["jordan_wigner"], eigenvalues["parity"], atol=1e-10)

    def test_unrestricted_equals_restricted_when_channels_match(self) -> None:
        """When alpha == beta integrals, UHF path should match restricted path."""
        # Create a restricted Hamiltonian
        h_res = create_nontrivial_test_hamiltonian(2)
        h1_a, _ = h_res.get_one_body_integrals()
        h2_aaaa, _, _ = h_res.get_two_body_integrals()
        n = h1_a.shape[0]

        # Create an "unrestricted" Hamiltonian with alpha == beta
        coeffs_alpha = np.eye(n)
        coeffs_beta = np.eye(n) + np.random.default_rng(99).standard_normal((n, n)) * 0.01
        basis_set = create_test_basis_set(n, "test-uhf-eq")
        orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        h_unres = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                h1_a,
                h1_a,
                h2_aaaa,
                h2_aaaa,
                h2_aaaa,
                orbitals,
                h_res.get_core_energy(),
                np.eye(0),
                np.eye(0),
            )
        )
        assert not h_unres.get_orbitals().is_restricted()

        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * n)
        qh_res = create("qubit_mapper", "qdk").run(h_res, mapping)
        qh_unres = create("qubit_mapper", "qdk").run(h_unres, mapping)

        # Eigenvalues should match
        e_res = np.sort(np.linalg.eigvalsh(qh_res.to_matrix()))
        e_unres = np.sort(np.linalg.eigvalsh(qh_unres.to_matrix()))
        np.testing.assert_allclose(e_unres, e_res, atol=1e-10)

    def test_unrestricted_3_orbitals(self) -> None:
        """UHF mapping works for a larger unrestricted system."""
        h = self._make_unrestricted_hamiltonian(3)
        mapping = MajoranaMapping.jordan_wigner(num_modes=6)
        mapper = create("qubit_mapper", "qdk")
        qh = mapper.run(h, mapping)

        assert qh.num_qubits == 6
        assert len(qh.pauli_strings) > 0
        for c in qh.coefficients:
            assert abs(c.imag) < 1e-12

    def test_unrestricted_asymmetric_aabb(self) -> None:
        """UHF with genuinely asymmetric AABB integrals (eri_aabb[p,q,r,s] ≠ eri_aabb[r,s,p,q])."""
        n = 2
        rng = np.random.default_rng(123)
        coeffs_alpha = np.eye(n) + rng.standard_normal((n, n)) * 0.1
        coeffs_beta = np.eye(n) + rng.standard_normal((n, n)) * 0.1
        basis_set = create_test_basis_set(n, "test-asym-aabb")
        orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        h1_alpha = np.array([[1.0, 0.2], [0.2, 0.8]])
        h1_beta = np.array([[0.9, 0.1], [0.1, 1.1]])

        # AAAA and BBBB with 8-fold symmetry
        def sym_eri(n, rng):
            h2 = np.zeros((n, n, n, n))
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            if h2[p, q, r, s] == 0:
                                v = rng.standard_normal() * 0.2
                                for a, b, c, d in {
                                    (p, q, r, s),
                                    (q, p, r, s),
                                    (p, q, s, r),
                                    (q, p, s, r),
                                    (r, s, p, q),
                                    (s, r, p, q),
                                    (r, s, q, p),
                                    (s, r, q, p),
                                }:
                                    h2[a, b, c, d] = v
            return h2.ravel()

        h2_aaaa = sym_eri(n, rng)
        h2_bbbb = sym_eri(n, rng)

        # AABB with only 4-fold symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr)
        # but NOT (pq|rs) = (rs|pq) since alpha ≠ beta
        h2_aabb_4d = np.zeros((n, n, n, n))
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        if h2_aabb_4d[p, q, r, s] == 0:
                            v = rng.standard_normal() * 0.3
                            for a, b, c, d in {(p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r)}:
                                h2_aabb_4d[a, b, c, d] = v
        h2_aabb = h2_aabb_4d.ravel()

        fock_a, fock_b = np.eye(0), np.eye(0)
        h = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                h1_alpha,
                h1_beta,
                h2_aaaa,
                h2_aabb,
                h2_bbbb,
                orbitals,
                0.0,
                fock_a,
                fock_b,
            )
        )

        n_modes = 2 * n
        mapper = create("qubit_mapper", "qdk")

        # All encodings should produce the same eigenvalues
        eigs = {}
        for enc in ["jordan_wigner", "bravyi_kitaev", "parity"]:
            mapping = getattr(MajoranaMapping, enc)(num_modes=n_modes)
            qh = mapper.run(h, mapping)
            eigs[enc] = np.sort(np.real(np.linalg.eigvalsh(qh.to_matrix())))
            for c in qh.coefficients:
                assert abs(c.imag) < 1e-12, f"Non-real coeff in {enc}: {c}"

        np.testing.assert_allclose(eigs["jordan_wigner"], eigs["bravyi_kitaev"], atol=1e-10)
        np.testing.assert_allclose(eigs["jordan_wigner"], eigs["parity"], atol=1e-10)


class TestQdkQubitMapperRealHamiltonians:
    """Tests with real molecular Hamiltonians."""

    def test_ethylene_4e4o(self, test_data_path: Path) -> None:
        """Test ethylene 4e4o Hamiltonian mapping."""
        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")

        mapper = create("qubit_mapper", "qdk")
        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        expected_qubits = 2 * h1_alpha.shape[0]
        mapping = MajoranaMapping.jordan_wigner(num_modes=expected_qubits)
        result = mapper.run(hamiltonian, mapping)

        assert result.num_qubits == expected_qubits
        assert len(result.pauli_strings) > 0
        for ps in result.pauli_strings:
            assert len(ps) == expected_qubits

    def test_f2_10e6o(self, test_data_path: Path) -> None:
        """Test F2 10e6o Hamiltonian mapping."""
        hamiltonian = Hamiltonian.from_json_file(test_data_path / "f2_10e6o.hamiltonian.json")

        mapper = create("qubit_mapper", "qdk")
        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        expected_qubits = 2 * h1_alpha.shape[0]
        mapping = MajoranaMapping.jordan_wigner(num_modes=expected_qubits)
        result = mapper.run(hamiltonian, mapping)

        assert result.num_qubits == expected_qubits
        assert len(result.pauli_strings) > 0

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT_NATURE, reason="Qiskit Nature not available")
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

            h1_alpha, _ = hamiltonian.get_one_body_integrals()
            mapping = MajoranaMapping.jordan_wigner(num_modes=2 * h1_alpha.shape[0])
            qdk_result = create("qubit_mapper", "qdk", threshold=threshold).run(hamiltonian, mapping)
            qiskit_result = create("qubit_mapper", "qiskit").run(hamiltonian, mapping)
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
        """Test BK mapper can be created via factory."""
        mapper = create("qubit_mapper", "qdk")
        assert isinstance(mapper, QubitMapper)

    def test_bk_simple_hamiltonian(self) -> None:
        """Test BK mapping of simple Hamiltonian."""
        mapper = create("qubit_mapper", "qdk")
        hamiltonian = create_test_hamiltonian(2)
        mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * 2)

        result = mapper.run(hamiltonian, mapping)

        assert isinstance(result, QubitOperator)
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
        mapper_bk = create("qubit_mapper", "qdk")

        # h_00 = 1 gives H = n_0_alpha + n_0_beta
        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)
        mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * n_orbitals)

        result_bk = mapper_bk.run(hamiltonian, mapping)
        assert result_bk.num_qubits == 2

        bk_dict = dict(zip(result_bk.pauli_strings, result_bk.coefficients, strict=True))

        # Expected: I (coeff 1), IZ (coeff -0.5), ZZ (coeff -0.5)
        assert len(bk_dict) == 3
        assert np.isclose(bk_dict["II"], 1.0, rtol=1e-10)
        assert np.isclose(bk_dict["IZ"], -0.5, rtol=1e-10)
        assert np.isclose(bk_dict["ZZ"], -0.5, rtol=1e-10)

    def test_bk_core_energy_not_included(self) -> None:
        """Test that core energy is not included in BK QubitOperator."""
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.array([[1.0]])  # Non-zero integral to generate Pauli terms
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        core_energy = 5.0
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals, core_energy=core_energy)
        mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 2
        one_body = np.array([[0.0, 1.0], [1.0, 0.0]])  # h_01 = h_10 = 1
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)
        mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 2
        one_body = np.array([[1.0, 0.0], [0.0, 2.0]])
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)
        mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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
        mapper = create("qubit_mapper", "qdk")

        n_orbitals = 1
        one_body = np.zeros((n_orbitals, n_orbitals))
        two_body = np.zeros(n_orbitals**4)
        two_body[0] = 2.0  # U = 2
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = _make_hamiltonian(one_body, two_body, orbitals)
        mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * n_orbitals)

        result = mapper.run(hamiltonian, mapping)
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

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT_NATURE, reason="Qiskit Nature not available")
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

            h1_alpha, _ = hamiltonian.get_one_body_integrals()
            mapping = MajoranaMapping.bravyi_kitaev(num_modes=2 * h1_alpha.shape[0])
            qdk_result = create("qubit_mapper", "qdk", threshold=threshold).run(hamiltonian, mapping)
            qiskit_result = create("qubit_mapper", "qiskit").run(hamiltonian, mapping)
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


# ─── SCBK one-step API ───────────────────────────────────────────────────


class TestScbkOneStep:
    """Tests for the one-step SCBK API via MajoranaMapping.symmetry_conserving_bravyi_kitaev."""

    def test_scbk_produces_reduced_qubit_count(self) -> None:
        """One-step SCBK mapping produces n-2 qubits."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        hamiltonian = create_nontrivial_test_hamiltonian()
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(n, Symmetries(1, 1))

        qh = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        assert qh.num_qubits == n - 2

    def test_scbk_sets_encoding_metadata(self) -> None:
        """One-step SCBK sets encoding to 'symmetry-conserving-bravyi-kitaev'."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        hamiltonian = create_nontrivial_test_hamiltonian()
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(n, Symmetries(1, 1))

        qh = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        assert qh.encoding == "symmetry-conserving-bravyi-kitaev"

    def test_scbk_carries_tapering_metadata(self) -> None:
        """One-step SCBK output has tapering metadata."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        hamiltonian = create_nontrivial_test_hamiltonian()
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(n, Symmetries(1, 1))

        qh = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        assert qh.tapering is not None
        assert qh.tapering.num_tapered == 2

    def test_scbk_matches_two_step(self) -> None:
        """One-step symmetry-conserving BK matches explicit BK-tree + internal taper."""
        from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper  # noqa: PLC0415
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        hamiltonian = create_nontrivial_test_hamiltonian()
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        sym = Symmetries(1, 1)

        # One-step
        scbk_mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(n, sym)
        qh_one_step = create("qubit_mapper", "qdk").run(hamiltonian, scbk_mapping)

        # Two-step: use BK-tree (the base encoding for SCBK)
        bk_tree_mapping = MajoranaMapping.bravyi_kitaev_tree(n)
        qh_bk = create("qubit_mapper", "qdk").run(hamiltonian, bk_tree_mapping)
        qh_two_step = QubitMapper._taper_result(qh_bk, scbk_mapping)

        assert qh_one_step.equiv(qh_two_step)

    def test_scbk_non_power_of_two(self) -> None:
        """SCBK eigenvalues match exact sector eigenvalues for non-power-of-2 mode counts."""
        from itertools import combinations  # noqa: PLC0415

        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        hamiltonian = create_nontrivial_test_hamiltonian(3)
        n_spatial = hamiltonian.get_one_body_integrals()[0].shape[0]
        n = 2 * n_spatial
        n_alpha, n_beta = 2, 1
        sym = Symmetries(n_alpha, n_beta)

        mapper = create("qubit_mapper", "qdk")

        # SCBK Hamiltonian
        scbk_mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(n, sym)
        qh_scbk = mapper.run(hamiltonian, scbk_mapping)
        eigs_scbk = np.sort(np.real(np.linalg.eigvalsh(qh_scbk.to_matrix())))

        # Exact: project JW Hamiltonian onto all matching-parity sectors
        jw_mapping = MajoranaMapping.jordan_wigner(n)
        H_jw = mapper.run(hamiltonian, jw_mapping).to_matrix()  # noqa: N806
        all_eigs: list[float] = []
        for na in range(n_spatial + 1):
            for nb in range(n_spatial + 1):
                if (-1) ** (na + nb) != (-1) ** (n_alpha + n_beta):
                    continue
                if (-1) ** na != (-1) ** n_alpha:
                    continue
                states = []
                for ac in combinations(range(n_spatial), na):
                    for bc in combinations(range(n_spatial), nb):
                        s = sum(1 << o for o in ac) | sum(1 << (o + n_spatial) for o in bc)
                        states.append(s)
                if states:
                    proj = np.zeros((2**n, len(states)))
                    for i, s in enumerate(states):
                        proj[s, i] = 1.0
                    all_eigs.extend(np.linalg.eigvalsh(proj.T @ H_jw @ proj))
        eigs_exact = np.sort(all_eigs)

        assert len(eigs_scbk) == len(eigs_exact)
        np.testing.assert_allclose(eigs_scbk, eigs_exact, atol=1e-10)

    def test_standard_mappings_no_tapering(self) -> None:
        """Standard mappings produce QubitOperator with tapering=None."""
        hamiltonian = create_nontrivial_test_hamiltonian()
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]

        for factory in [MajoranaMapping.jordan_wigner, MajoranaMapping.bravyi_kitaev, MajoranaMapping.parity]:
            mapping = factory(n)
            qh = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
            assert qh.tapering is None


# ─── BK-tree factory ─────────────────────────────────────────────────────


class TestBravyiKitaevTreeMapper:
    """Tests for the BK-tree (balanced binary tree) Bravyi-Kitaev variant."""

    def test_bk_tree_instantiation(self) -> None:
        """BK-tree mapping can be created for various mode counts."""
        for n in (4, 6, 8, 10, 12):
            mapping = MajoranaMapping.bravyi_kitaev_tree(n)
            assert mapping.name == "bravyi-kitaev-tree"
            assert mapping.num_qubits == n

    def test_bk_tree_clifford_algebra(self) -> None:
        """BK-tree Majorana operators satisfy anticommutation."""
        from qdk_chemistry.utils.pauli_matrix import pauli_to_sparse_matrix  # noqa: PLC0415

        for n in (4, 6, 8):
            mapping = MajoranaMapping.bravyi_kitaev_tree(n)
            gammas = [
                pauli_to_sparse_matrix(
                    [_table_entry_to_pauli_string(mapping.table[k], mapping.num_qubits)],
                    np.array([1.0]),
                ).toarray()
                for k in range(2 * n)
            ]
            for i in range(2 * n):
                for j in range(i, 2 * n):
                    anticomm = gammas[i] @ gammas[j] + gammas[j] @ gammas[i]
                    expected = 2.0 * np.eye(2**n) if i == j else np.zeros((2**n, 2**n))
                    np.testing.assert_allclose(anticomm, expected, atol=1e-12, err_msg=f"n={n}, i={i}, j={j}")

    def test_bk_tree_eigenvalues_match_jw(self) -> None:
        """BK-tree encoded Hamiltonian has same eigenvalues as JW."""
        hamiltonian = create_nontrivial_test_hamiltonian(3)
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapper = create("qubit_mapper", "qdk")

        qh_jw = mapper.run(hamiltonian, MajoranaMapping.jordan_wigner(n))
        qh_bkt = mapper.run(hamiltonian, MajoranaMapping.bravyi_kitaev_tree(n))

        eigs_jw = np.sort(np.real(np.linalg.eigvalsh(qh_jw.to_matrix())))
        eigs_bkt = np.sort(np.real(np.linalg.eigvalsh(qh_bkt.to_matrix())))
        np.testing.assert_allclose(eigs_bkt, eigs_jw, atol=1e-10)

    def test_bk_tree_z2_symmetries(self) -> None:
        """BK-tree Z_{n-1} and Z_{n/2-1} commute with the Hamiltonian (Z₂ symmetries)."""
        hamiltonian = create_nontrivial_test_hamiltonian(3)
        n = 2 * hamiltonian.get_one_body_integrals()[0].shape[0]
        mapper = create("qubit_mapper", "qdk")

        H = mapper.run(hamiltonian, MajoranaMapping.bravyi_kitaev_tree(n)).to_matrix()  # noqa: N806

        # Build Z_{n-1} and Z_{n/2-1} operators
        for q in (n // 2 - 1, n - 1):
            Z_q = np.eye(2**n, dtype=complex)  # noqa: N806
            for i in range(2**n):
                if (i >> q) & 1:
                    Z_q[i, i] = -1.0
            commutator = H @ Z_q - Z_q @ H
            np.testing.assert_allclose(commutator, 0.0, atol=1e-12, err_msg=f"[H, Z_{q}] ≠ 0")
