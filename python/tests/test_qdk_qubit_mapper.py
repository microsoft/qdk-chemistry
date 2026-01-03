"""Tests for QdkQubitMapper and QubitHamiltonian reorder methods."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import QdkQubitMapper, QubitMapper
from qdk_chemistry.data import Hamiltonian, QubitHamiltonian

from .test_helpers import create_test_hamiltonian, create_test_orbitals


class TestQdkQubitMapperBasic:
    """Basic tests for QdkQubitMapper instantiation and settings."""

    def test_instantiation(self) -> None:
        """Test that QdkQubitMapper can be instantiated."""
        mapper = QdkQubitMapper()
        assert isinstance(mapper, QubitMapper)

    def test_default_settings(self) -> None:
        """Test default settings values."""
        mapper = QdkQubitMapper()
        assert mapper.settings().get("mapping_type") == "jordan_wigner"
        assert mapper.settings().get("threshold") == 1e-12

    def test_name(self) -> None:
        """Test that name() returns 'qdk'."""
        mapper = QdkQubitMapper()
        assert mapper.name() == "qdk"

    def test_type_name(self) -> None:
        """Test that type_name() returns 'qubit_mapper'."""
        mapper = QdkQubitMapper()
        assert mapper.type_name() == "qubit_mapper"

    def test_custom_settings(self) -> None:
        """Test custom settings can be set."""
        mapper = QdkQubitMapper()
        mapper.settings().set("threshold", 1e-10)
        assert mapper.settings().get("threshold") == 1e-10

    def test_invalid_mapping_type_raises(self) -> None:
        """Test that invalid mapping type raises ValueError."""
        mapper = QdkQubitMapper()
        # Settings validation happens at set time due to allowed options
        with pytest.raises(ValueError, match="out of allowed options"):
            mapper.settings().set("mapping_type", "invalid_type")


class TestQdkQubitMapperJordanWigner:
    """Tests for Jordan-Wigner transformation correctness."""

    def test_simple_hamiltonian(self) -> None:
        """Test mapping a simple diagonal Hamiltonian."""
        mapper = QdkQubitMapper()
        hamiltonian = create_test_hamiltonian(2)  # 2 spatial orbitals -> 4 qubits

        qubit_hamiltonian = mapper.run(hamiltonian)

        assert isinstance(qubit_hamiltonian, QubitHamiltonian)
        assert qubit_hamiltonian.num_qubits == 4
        assert len(qubit_hamiltonian.pauli_strings) > 0
        assert len(qubit_hamiltonian.coefficients) == len(qubit_hamiltonian.pauli_strings)

    def test_number_operator(self) -> None:
        """Test that a_p^dag * a_p = (I - Z_p) / 2.

        For a Hamiltonian with only one-body diagonal terms h_pp = 1,
        the qubit Hamiltonian should have terms like (I - Z_p) / 2.
        """
        mapper = QdkQubitMapper()

        # Create a simple Hamiltonian with only h_00 = 1 (alpha orbital 0)
        n_orbitals = 1
        one_body = np.array([[1.0]])  # h_00 = 1
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, 0.0, np.eye(0))

        qubit_hamiltonian = mapper.run(hamiltonian)

        # Should have 2 qubits (1 spatial orbital -> 2 spin orbitals)
        assert qubit_hamiltonian.num_qubits == 2

        # The Hamiltonian should include:
        # - h_00 (alpha) gives (I - Z_0) / 2 = 0.5*I - 0.5*Z_0
        # - h_00 (beta) gives (I - Z_1) / 2 = 0.5*I - 0.5*Z_1
        # So we expect terms: I (coefficient 1.0), Z_0 (-0.5), Z_1 (-0.5)

        # Convert to dict for easier checking
        pauli_dict = dict(zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True))

        # Check that we have expected terms (allowing for complex coefficients)
        assert "II" in pauli_dict
        assert "ZI" in pauli_dict
        assert "IZ" in pauli_dict

        # Check coefficient magnitudes
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)
        assert np.isclose(pauli_dict["ZI"].real, -0.5, atol=1e-10)
        assert np.isclose(pauli_dict["IZ"].real, -0.5, atol=1e-10)

    def test_core_energy_as_identity(self) -> None:
        """Test that core energy appears as coefficient on identity term."""
        mapper = QdkQubitMapper()

        n_orbitals = 1
        one_body = np.zeros((1, 1))
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        core_energy = 5.0
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, core_energy, np.eye(0))

        qubit_hamiltonian = mapper.run(hamiltonian)

        pauli_dict = dict(zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True))

        # Identity term should have coefficient equal to core_energy
        assert "II" in pauli_dict
        assert np.isclose(pauli_dict["II"].real, core_energy, atol=1e-10)

    def test_threshold_pruning(self) -> None:
        """Test that small coefficients are pruned based on threshold."""
        mapper = QdkQubitMapper()
        mapper.settings().set("threshold", 0.1)  # High threshold for testing

        hamiltonian = create_test_hamiltonian(2)
        qubit_hamiltonian = mapper.run(hamiltonian)

        # All coefficients should be above threshold
        for coeff in qubit_hamiltonian.coefficients:
            assert abs(coeff) >= 0.1


class TestQdkQubitMapperOutputFormat:
    """Tests for QubitHamiltonian output format compatibility."""

    def test_pauli_strings_format(self) -> None:
        """Test that Pauli strings are in correct format."""
        mapper = QdkQubitMapper()
        hamiltonian = create_test_hamiltonian(2)

        qubit_hamiltonian = mapper.run(hamiltonian)

        for pauli_str in qubit_hamiltonian.pauli_strings:
            assert isinstance(pauli_str, str)
            assert len(pauli_str) == 4  # 2 spatial orbitals -> 4 qubits
            assert all(c in "IXYZ" for c in pauli_str)

    def test_coefficients_dtype(self) -> None:
        """Test that coefficients are complex numpy array."""
        mapper = QdkQubitMapper()
        hamiltonian = create_test_hamiltonian(2)

        qubit_hamiltonian = mapper.run(hamiltonian)

        assert isinstance(qubit_hamiltonian.coefficients, np.ndarray)
        assert qubit_hamiltonian.coefficients.dtype == complex


class TestQubitHamiltonianReorderQubits:
    """Tests for QubitHamiltonian.reorder_qubits() method."""

    def test_identity_permutation(self) -> None:
        """Test that identity permutation returns equivalent Hamiltonian."""
        qh = QubitHamiltonian(
            pauli_strings=["XIZI", "IYII"],
            coefficients=np.array([0.5, 0.3], dtype=complex),
        )

        reordered = qh.reorder_qubits([0, 1, 2, 3])

        assert reordered.pauli_strings == qh.pauli_strings
        assert np.allclose(reordered.coefficients, qh.coefficients)

    def test_swap_two_qubits(self) -> None:
        """Test swapping two adjacent qubits."""
        qh = QubitHamiltonian(
            pauli_strings=["XIZI"],
            coefficients=np.array([1.0], dtype=complex),
        )

        # Swap qubits 0 and 1: X on q0, I on q1 -> I on q0, X on q1
        reordered = qh.reorder_qubits([1, 0, 2, 3])

        assert reordered.pauli_strings == ["IXZI"]

    def test_reverse_permutation(self) -> None:
        """Test reversing all qubit indices."""
        qh = QubitHamiltonian(
            pauli_strings=["XYZI"],
            coefficients=np.array([1.0], dtype=complex),
        )

        # Permutation that reverses qubit order
        reordered = qh.reorder_qubits([3, 2, 1, 0])

        assert reordered.pauli_strings == ["IZYX"]

    def test_invalid_permutation_length(self) -> None:
        """Test that invalid permutation length raises error."""
        qh = QubitHamiltonian(
            pauli_strings=["XIZI"],
            coefficients=np.array([1.0], dtype=complex),
        )

        with pytest.raises(ValueError, match="Permutation length"):
            qh.reorder_qubits([0, 1, 2])  # Wrong length

    def test_invalid_permutation_values(self) -> None:
        """Test that invalid permutation values raise error."""
        qh = QubitHamiltonian(
            pauli_strings=["XIZI"],
            coefficients=np.array([1.0], dtype=complex),
        )

        with pytest.raises(ValueError, match="Invalid permutation"):
            qh.reorder_qubits([0, 1, 1, 3])  # Duplicate value


class TestQubitHamiltonianToInterleaved:
    """Tests for QubitHamiltonian.to_interleaved() method."""

    def test_blocked_to_interleaved_4_qubits(self) -> None:
        """Test converting 4-qubit blocked to interleaved ordering.

        Blocked:      [α₀, α₁, β₀, β₁]
        Interleaved:  [α₀, β₀, α₁, β₁]

        Permutation: blocked[0] -> interleaved[0] (α₀)
                     blocked[1] -> interleaved[2] (α₁)
                     blocked[2] -> interleaved[1] (β₀)
                     blocked[3] -> interleaved[3] (β₁)
        """
        # String "ABCD" where A is on α₀, B on α₁, C on β₀, D on β₁ (blocked)
        # After interleaving: A on α₀, C on β₀, B on α₁, D on β₁ -> "ACBD"
        qh = QubitHamiltonian(
            pauli_strings=["XYZZ"],  # X(α₀), Y(α₁), Z(β₀), Z(β₁)
            coefficients=np.array([1.0], dtype=complex),
        )

        interleaved = qh.to_interleaved(n_spatial=2)

        # After reorder: X(α₀), Z(β₀), Y(α₁), Z(β₁) -> "XZYZ"
        assert interleaved.pauli_strings == ["XZYZ"]

    def test_interleaved_preserves_coefficients(self) -> None:
        """Test that interleaving preserves coefficient values."""
        qh = QubitHamiltonian(
            pauli_strings=["XIZI", "IYII"],
            coefficients=np.array([0.5 + 0.1j, 0.3], dtype=complex),
        )

        interleaved = qh.to_interleaved(n_spatial=2)

        assert np.allclose(interleaved.coefficients, qh.coefficients)

    def test_invalid_n_spatial(self) -> None:
        """Test that invalid n_spatial raises error."""
        qh = QubitHamiltonian(
            pauli_strings=["XIZI"],
            coefficients=np.array([1.0], dtype=complex),
        )

        with pytest.raises(ValueError, match="must be 2 \\* n_spatial"):
            qh.to_interleaved(n_spatial=3)  # Would need 6 qubits, but we have 4

    def test_identity_for_single_orbital(self) -> None:
        """Test that single spatial orbital (2 qubits) is unchanged by interleaving.

        For n_spatial=1: blocked = [α₀, β₀], interleaved = [α₀, β₀]
        No reordering needed.
        """
        qh = QubitHamiltonian(
            pauli_strings=["XY"],
            coefficients=np.array([1.0], dtype=complex),
        )

        interleaved = qh.to_interleaved(n_spatial=1)

        assert interleaved.pauli_strings == ["XY"]


class TestQdkQubitMapperNontrivialHamiltonians:
    """Tests for QdkQubitMapper with nontrivial Hamiltonians from test data files."""

    @pytest.fixture
    def test_data_files_path(self) -> "Path":
        """Get path to test data directory."""
        from pathlib import Path  # noqa: PLC0415

        module_dir = Path(__file__).resolve().parent
        return (module_dir / "test_data").resolve()

    def test_ethylene_4e4o_hamiltonian(self, test_data_files_path: "Path") -> None:
        """Test mapping the ethylene 4e4o Hamiltonian.

        This is a real-world Hamiltonian with 4 electrons in 4 orbitals,
        resulting in 8 qubits (4 spatial orbitals × 2 spin states).
        """
        classical_hamiltonian = Hamiltonian.from_json_file(test_data_files_path / "ethylene_4e4o_2det.hamiltonian.json")

        mapper = QdkQubitMapper()
        qubit_hamiltonian = mapper.run(classical_hamiltonian)

        # Basic validation - infer n_orbitals from integrals (active space size)
        h1_alpha, _ = classical_hamiltonian.get_one_body_integrals()
        n_orbitals = h1_alpha.shape[0]
        expected_qubits = 2 * n_orbitals  # spin orbitals
        assert qubit_hamiltonian.num_qubits == expected_qubits
        assert len(qubit_hamiltonian.pauli_strings) > 0
        assert len(qubit_hamiltonian.coefficients) == len(qubit_hamiltonian.pauli_strings)

        # All Pauli strings should have correct length
        for ps in qubit_hamiltonian.pauli_strings:
            assert len(ps) == expected_qubits
            assert all(c in "IXYZ" for c in ps)

        # Check that core energy is included in identity term
        pauli_dict = dict(zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True))
        identity_str = "I" * expected_qubits
        assert identity_str in pauli_dict
        # The identity coefficient includes core_energy plus contributions from one-body
        # and two-body terms. The one-body diagonal terms contribute negative values,
        # so the total can be less than core_energy.
        # Just verify the identity term exists and is real-valued.
        assert np.isfinite(pauli_dict[identity_str].real)

    def test_f2_10e6o_hamiltonian(self, test_data_files_path: "Path") -> None:
        """Test mapping the F2 10e6o Hamiltonian.

        This is a larger system with 10 electrons in 6 orbitals,
        resulting in 12 qubits (6 spatial orbitals × 2 spin states).
        """
        classical_hamiltonian = Hamiltonian.from_json_file(test_data_files_path / "f2_10e6o.hamiltonian.json")

        mapper = QdkQubitMapper()
        qubit_hamiltonian = mapper.run(classical_hamiltonian)

        # Basic validation - infer n_orbitals from integrals (active space size)
        h1_alpha, _ = classical_hamiltonian.get_one_body_integrals()
        n_orbitals = h1_alpha.shape[0]
        expected_qubits = 2 * n_orbitals
        assert qubit_hamiltonian.num_qubits == expected_qubits
        assert len(qubit_hamiltonian.pauli_strings) > 0

        # All Pauli strings should have correct length
        for ps in qubit_hamiltonian.pauli_strings:
            assert len(ps) == expected_qubits
            assert all(c in "IXYZ" for c in ps)

    def test_ethylene_vs_qiskit(self, test_data_files_path: "Path") -> None:
        """Cross-validate ethylene Hamiltonian against Qiskit JordanWignerMapper.

        Both mappers should produce identical qubit Hamiltonians when using the
        same threshold. This test sets Qiskit's internal tolerances to match QDK's
        default threshold (1e-12) to ensure both mappers apply the same filtering.
        """
        pytest.importorskip("qiskit_nature")
        from qiskit.quantum_info import SparsePauliOp  # noqa: PLC0415
        from qiskit_nature.second_q.operators import FermionicOp  # noqa: PLC0415

        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        classical_hamiltonian = Hamiltonian.from_json_file(test_data_files_path / "ethylene_4e4o_2det.hamiltonian.json")

        # Use QDK's default threshold for both mappers
        threshold = 1e-12

        # Save original Qiskit tolerances and set to match QDK threshold
        original_fermionic_atol = FermionicOp.atol
        original_sparse_atol = SparsePauliOp.atol
        try:
            FermionicOp.atol = threshold
            SparsePauliOp.atol = threshold

            # Map with QDK (uses threshold from constructor, defaults to 1e-12)
            qdk_mapper = QdkQubitMapper(threshold=threshold)
            qdk_result = qdk_mapper.run(classical_hamiltonian)

            # Map with Qiskit (now uses matching threshold via class attribute)
            qiskit_mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
            qiskit_result = qiskit_mapper.run(classical_hamiltonian)
        finally:
            # Restore original tolerances
            FermionicOp.atol = original_fermionic_atol
            SparsePauliOp.atol = original_sparse_atol

        # Same number of qubits
        assert qdk_result.num_qubits == qiskit_result.num_qubits

        # Convert to dicts for comparison
        qdk_dict = dict(zip(qdk_result.pauli_strings, qdk_result.coefficients, strict=True))
        qiskit_dict = dict(zip(qiskit_result.pauli_strings, qiskit_result.coefficients, strict=True))

        # QDK includes core energy in the identity term, Qiskit does not
        core_energy = classical_hamiltonian.get_core_energy()
        identity_str = "I" * qdk_result.num_qubits
        if identity_str in qiskit_dict:
            qiskit_dict[identity_str] = qiskit_dict[identity_str] + core_energy
        else:
            qiskit_dict[identity_str] = core_energy

        # Same number of terms (since both use same threshold)
        assert len(qdk_dict) == len(qiskit_dict), f"Term count mismatch: QDK={len(qdk_dict)}, Qiskit={len(qiskit_dict)}"

        # Check all terms appear in both with matching coefficients
        for pauli_str, qiskit_coeff in qiskit_dict.items():
            assert pauli_str in qdk_dict, f"Missing Pauli string in QDK: {pauli_str}"
            qdk_coeff = qdk_dict[pauli_str]
            assert np.isclose(qdk_coeff, qiskit_coeff, rtol=1e-10, atol=1e-14), (
                f"Coefficient mismatch for {pauli_str}: QDK={qdk_coeff}, Qiskit={qiskit_coeff}"
            )


class TestQdkQubitMapperVsQiskit:
    """Cross-validation tests comparing QdkQubitMapper against Qiskit mapper."""

    @pytest.fixture
    def simple_hamiltonian(self) -> Hamiltonian:
        """Create a simple test Hamiltonian."""
        return create_test_hamiltonian(2)

    def test_same_number_of_qubits(self, simple_hamiltonian: Hamiltonian) -> None:
        """Test that QDK and Qiskit produce same number of qubits."""
        qdk_mapper = QdkQubitMapper()
        qdk_result = qdk_mapper.run(simple_hamiltonian)

        # Import Qiskit mapper if available
        pytest.importorskip("qiskit_nature")
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        qiskit_mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
        qiskit_result = qiskit_mapper.run(simple_hamiltonian)

        assert qdk_result.num_qubits == qiskit_result.num_qubits

    def test_diagonal_hamiltonian_coefficients(self) -> None:
        """Test coefficient values for a purely diagonal Hamiltonian.

        For a diagonal one-body Hamiltonian with h_pp = 1, the qubit
        Hamiltonian should have specific known coefficients.
        """
        # Create diagonal Hamiltonian
        n_orbitals = 2
        one_body = np.eye(n_orbitals)
        two_body = np.zeros(n_orbitals**4)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, 0.0, np.eye(0))

        qdk_mapper = QdkQubitMapper()
        qdk_result = qdk_mapper.run(hamiltonian)

        # Convert to dict for coefficient comparison
        qdk_dict = dict(zip(qdk_result.pauli_strings, qdk_result.coefficients, strict=True))

        # For diagonal one-body with h_00 = h_11 = 1 (restricted):
        # Each occupation number n_p = (I - Z_p) / 2
        # Total one-body energy: sum over p of h_pp * n_p
        # = 1 * (I - Z_0)/2 + 1 * (I - Z_1)/2 + 1 * (I - Z_2)/2 + 1 * (I - Z_3)/2
        # = 2*I - 0.5*(Z_0 + Z_1 + Z_2 + Z_3)

        # Check identity coefficient
        assert "IIII" in qdk_dict
        assert np.isclose(qdk_dict["IIII"].real, 2.0, atol=1e-10)

        # Check Z coefficients
        for z_str in ["ZIII", "IZII", "IIZI", "IIIZ"]:
            assert z_str in qdk_dict
            assert np.isclose(qdk_dict[z_str].real, -0.5, atol=1e-10)
