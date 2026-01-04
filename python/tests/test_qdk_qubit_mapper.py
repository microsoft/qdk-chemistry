"""Tests for QdkQubitMapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import QdkQubitMapper, QubitMapper
from qdk_chemistry.algorithms.qubit_mapper.qdk_qubit_mapper import (
    _bk_flip_set,
    _bk_parity_set,
    _bk_remainder_set,
    _bk_update_set,
)
from qdk_chemistry.data import Hamiltonian, QubitHamiltonian

from .test_helpers import create_test_hamiltonian, create_test_orbitals


class TestBravyiKitaevSets:
    """Tests for Bravyi-Kitaev set computation functions.

    The second argument n must be a power of 2.
    """

    def test_update_set_4_qubits(self) -> None:
        """Test update set for 4 qubit system (n=4 is already power of 2)."""
        # U(0) = {1, 3} - qubit 0's occupation affects qubits 1 and 3
        assert _bk_update_set(0, 4) == frozenset({1, 3})
        # U(1) = {3}
        assert _bk_update_set(1, 4) == frozenset({3})
        # U(2) = {3}
        assert _bk_update_set(2, 4) == frozenset({3})
        # U(3) = {} - qubit 3 is the root, no ancestors
        assert _bk_update_set(3, 4) == frozenset()

    def test_update_set_8_qubits(self) -> None:
        """Test update set for 8 qubit system."""
        assert _bk_update_set(0, 8) == frozenset({1, 3, 7})
        assert _bk_update_set(4, 8) == frozenset({5, 7})
        assert _bk_update_set(7, 8) == frozenset()

    def test_parity_set_4_qubits(self) -> None:
        """Test parity set for 4 qubit system.

        P(j) follows the recursive binary tree structure.
        """
        assert _bk_parity_set(0, 4) == frozenset()
        assert _bk_parity_set(1, 4) == frozenset({0})
        assert _bk_parity_set(2, 4) == frozenset({1})
        assert _bk_parity_set(3, 4) == frozenset({1, 2})

    def test_parity_set_8_qubits(self) -> None:
        """Test parity set for 8 qubit system."""
        assert _bk_parity_set(0, 8) == frozenset()
        assert _bk_parity_set(4, 8) == frozenset({3})
        assert _bk_parity_set(5, 8) == frozenset({3, 4})
        assert _bk_parity_set(6, 8) == frozenset({3, 5})
        assert _bk_parity_set(7, 8) == frozenset({3, 5, 6})

    def test_flip_set_4_qubits(self) -> None:
        """Test flip set for 4 qubit system."""
        assert _bk_flip_set(0, 4) == frozenset()
        assert _bk_flip_set(1, 4) == frozenset({0})
        assert _bk_flip_set(2, 4) == frozenset()
        assert _bk_flip_set(3, 4) == frozenset({1, 2})

    def test_remainder_set_4_qubits(self) -> None:
        """Test remainder set R(j) = P(j) - F(j) for 4 qubit system."""
        # R(j) = P(j) - F(j) (set difference)
        assert _bk_remainder_set(0, 4) == frozenset()  # {} - {} = {}
        assert _bk_remainder_set(1, 4) == frozenset()  # {0} - {0} = {}
        assert _bk_remainder_set(2, 4) == frozenset({1})  # {1} - {} = {1}
        assert _bk_remainder_set(3, 4) == frozenset()  # {1,2} - {1,2} = {}

    def test_remainder_set_8_qubits(self) -> None:
        """Test remainder set for 8 qubit system."""
        assert _bk_remainder_set(4, 8) == frozenset({3})  # {3} - {} = {3}
        assert _bk_remainder_set(5, 8) == frozenset({3})  # {3,4} - {4} = {3}
        assert _bk_remainder_set(6, 8) == frozenset({3, 5})  # {3,5} - {} = {3,5}
        assert _bk_remainder_set(7, 8) == frozenset()  # {3,5,6} - {3,5,6} = {}


class TestQdkQubitMapper:
    """Tests for QdkQubitMapper."""

    def test_instantiation(self) -> None:
        """Test basic instantiation and interface."""
        mapper = QdkQubitMapper()
        assert isinstance(mapper, QubitMapper)
        assert mapper.name() == "qdk"
        assert mapper.type_name() == "qubit_mapper"

    def test_default_settings(self) -> None:
        """Test default settings values."""
        mapper = QdkQubitMapper()
        assert mapper.settings().get("mapping_type") == "jordan_wigner"
        assert mapper.settings().get("threshold") == 1e-12

    def test_custom_threshold(self) -> None:
        """Test custom threshold can be set."""
        mapper = QdkQubitMapper(threshold=1e-10)
        assert mapper.settings().get("threshold") == 1e-10

    def test_invalid_mapping_type_raises(self) -> None:
        """Test that invalid mapping type raises ValueError."""
        mapper = QdkQubitMapper()
        with pytest.raises(ValueError, match="out of allowed options"):
            mapper.settings().set("mapping_type", "invalid_type")

    def test_simple_hamiltonian(self) -> None:
        """Test mapping a simple diagonal Hamiltonian."""
        mapper = QdkQubitMapper()
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        assert isinstance(result, QubitHamiltonian)
        assert result.num_qubits == 4
        assert len(result.pauli_strings) > 0
        assert len(result.coefficients) == len(result.pauli_strings)
        assert result.coefficients.dtype == complex

    def test_number_operator(self) -> None:
        """Test JW transform of number operator: a†a = (I - Z) / 2."""
        mapper = QdkQubitMapper()

        # h_00 = 1 gives n_0 = (I - Z_0)/2 for each spin
        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, 0.0, np.eye(0))

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert result.num_qubits == 2
        assert np.isclose(pauli_dict["II"].real, 1.0, atol=1e-10)
        assert np.isclose(pauli_dict["ZI"].real, -0.5, atol=1e-10)
        assert np.isclose(pauli_dict["IZ"].real, -0.5, atol=1e-10)

    def test_core_energy_as_identity(self) -> None:
        """Test that core energy appears as identity coefficient."""
        mapper = QdkQubitMapper()

        n_orbitals = 1
        one_body = np.zeros((1, 1))
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, 5.0, np.eye(0))

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert np.isclose(pauli_dict["II"].real, 5.0, atol=1e-10)

    def test_threshold_pruning(self) -> None:
        """Test that small coefficients are pruned."""
        mapper = QdkQubitMapper(threshold=0.1)
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        for coeff in result.coefficients:
            assert abs(coeff) >= 0.1

    def test_pauli_strings_format(self) -> None:
        """Test Pauli string format."""
        mapper = QdkQubitMapper()
        hamiltonian = create_test_hamiltonian(2)

        result = mapper.run(hamiltonian)

        for ps in result.pauli_strings:
            assert isinstance(ps, str)
            assert len(ps) == 4
            assert all(c in "IXYZ" for c in ps)


class TestQdkQubitMapperRealHamiltonians:
    """Tests with real molecular Hamiltonians."""

    @pytest.fixture
    def test_data_path(self) -> Path:
        """Get path to test data directory."""
        return Path(__file__).resolve().parent / "test_data"

    def test_ethylene_4e4o(self, test_data_path: Path) -> None:
        """Test ethylene 4e4o Hamiltonian mapping."""
        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")

        mapper = QdkQubitMapper()
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

        mapper = QdkQubitMapper()
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
        create = pytest.importorskip("qdk_chemistry.algorithms").create

        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")
        threshold = 1e-12

        # Match Qiskit tolerances to QDK threshold
        original_fermionic_atol = FermionicOp.atol
        original_sparse_atol = SparsePauliOp.atol
        try:
            FermionicOp.atol = threshold
            SparsePauliOp.atol = threshold

            qdk_result = QdkQubitMapper(threshold=threshold).run(hamiltonian)
            qiskit_result = create("qubit_mapper", "qiskit", encoding="jordan-wigner").run(hamiltonian)
        finally:
            FermionicOp.atol = original_fermionic_atol
            SparsePauliOp.atol = original_sparse_atol

        assert qdk_result.num_qubits == qiskit_result.num_qubits

        qdk_dict = dict(zip(qdk_result.pauli_strings, qdk_result.coefficients, strict=True))
        qiskit_dict = dict(zip(qiskit_result.pauli_strings, qiskit_result.coefficients, strict=True))

        # QDK includes core energy in identity term, Qiskit does not
        core_energy = hamiltonian.get_core_energy()
        identity_str = "I" * qdk_result.num_qubits
        if identity_str in qiskit_dict:
            qiskit_dict[identity_str] = qiskit_dict[identity_str] + core_energy
        else:
            qiskit_dict[identity_str] = core_energy

        assert len(qdk_dict) == len(qiskit_dict)

        for pauli_str, qiskit_coeff in qiskit_dict.items():
            assert pauli_str in qdk_dict, f"Missing: {pauli_str}"
            assert np.isclose(qdk_dict[pauli_str], qiskit_coeff, rtol=1e-10, atol=1e-14)


class TestBravyiKitaevMapper:
    """Tests for Bravyi-Kitaev mapping."""

    def test_bk_instantiation(self) -> None:
        """Test BK mapping type is valid."""
        mapper = QdkQubitMapper(mapping_type="bravyi_kitaev")
        assert mapper.settings().get("mapping_type") == "bravyi_kitaev"

    def test_bk_simple_hamiltonian(self) -> None:
        """Test BK mapping of simple Hamiltonian."""
        mapper = QdkQubitMapper(mapping_type="bravyi_kitaev")
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
        mapper_bk = QdkQubitMapper(mapping_type="bravyi_kitaev")

        # h_00 = 1 gives H = n_0_alpha + n_0_beta
        n_orbitals = 1
        one_body = np.array([[1.0]])
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, 0.0, np.eye(0))

        result_bk = mapper_bk.run(hamiltonian)
        assert result_bk.num_qubits == 2

        bk_dict = dict(zip(result_bk.pauli_strings, result_bk.coefficients, strict=True))

        # Expected: I (coeff 1), IZ (coeff -0.5), ZZ (coeff -0.5)
        assert len(bk_dict) == 3
        assert np.isclose(bk_dict["II"], 1.0, rtol=1e-10)
        assert np.isclose(bk_dict["IZ"], -0.5, rtol=1e-10)
        assert np.isclose(bk_dict["ZZ"], -0.5, rtol=1e-10)

    def test_bk_core_energy(self) -> None:
        """Test that core energy appears as identity coefficient in BK."""
        mapper = QdkQubitMapper(mapping_type="bravyi_kitaev")

        n_orbitals = 1
        one_body = np.zeros((1, 1))
        two_body = np.zeros(1)
        orbitals = create_test_orbitals(n_orbitals)
        hamiltonian = Hamiltonian(one_body, two_body, orbitals, 5.0, np.eye(0))

        result = mapper.run(hamiltonian)
        pauli_dict = dict(zip(result.pauli_strings, result.coefficients, strict=True))

        assert np.isclose(pauli_dict["II"].real, 5.0, atol=1e-10)

    @pytest.fixture
    def test_data_path(self) -> Path:
        """Get path to test data directory."""
        return Path(__file__).resolve().parent / "test_data"

    def test_bk_vs_qiskit(self, test_data_path: Path) -> None:
        """Cross-validate BK against Qiskit BravyiKitaevMapper."""
        pytest.importorskip("qiskit_nature")
        SparsePauliOp = pytest.importorskip("qiskit.quantum_info").SparsePauliOp  # noqa: N806
        FermionicOp = pytest.importorskip("qiskit_nature.second_q.operators").FermionicOp  # noqa: N806
        create = pytest.importorskip("qdk_chemistry.algorithms").create

        hamiltonian = Hamiltonian.from_json_file(test_data_path / "ethylene_4e4o_2det.hamiltonian.json")
        threshold = 1e-12

        # Match Qiskit tolerances to QDK threshold
        original_fermionic_atol = FermionicOp.atol
        original_sparse_atol = SparsePauliOp.atol
        try:
            FermionicOp.atol = threshold
            SparsePauliOp.atol = threshold

            qdk_result = QdkQubitMapper(mapping_type="bravyi_kitaev", threshold=threshold).run(hamiltonian)
            qiskit_result = create("qubit_mapper", "qiskit", encoding="bravyi-kitaev").run(hamiltonian)
        finally:
            FermionicOp.atol = original_fermionic_atol
            SparsePauliOp.atol = original_sparse_atol

        assert qdk_result.num_qubits == qiskit_result.num_qubits

        qdk_dict = dict(zip(qdk_result.pauli_strings, qdk_result.coefficients, strict=True))
        qiskit_dict = dict(zip(qiskit_result.pauli_strings, qiskit_result.coefficients, strict=True))

        # QDK includes core energy in identity term, Qiskit does not
        core_energy = hamiltonian.get_core_energy()
        identity_str = "I" * qdk_result.num_qubits
        if identity_str in qiskit_dict:
            qiskit_dict[identity_str] = qiskit_dict[identity_str] + core_energy
        else:
            qiskit_dict[identity_str] = core_energy

        assert len(qdk_dict) == len(qiskit_dict), f"QDK has {len(qdk_dict)} terms, Qiskit has {len(qiskit_dict)}"

        for pauli_str, qiskit_coeff in qiskit_dict.items():
            assert pauli_str in qdk_dict, f"Missing: {pauli_str}"
            assert np.isclose(qdk_dict[pauli_str], qiskit_coeff, rtol=1e-10, atol=1e-14), (
                f"Mismatch for {pauli_str}: QDK={qdk_dict[pauli_str]}, Qiskit={qiskit_coeff}"
            )
