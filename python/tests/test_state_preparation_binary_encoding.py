"""Tests for the sparse isometry with binary encoding state preparation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import qsharp

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import SparseIsometryBinaryEncodingStatePreparation
from qdk_chemistry.algorithms.state_preparation.sparse_isometry import gf2x_with_tracking
from qdk_chemistry.algorithms.state_preparation.sparse_isometry_binary_encoding import _encode_gf2x_ops_for_qs
from qdk_chemistry.data import Circuit, Wavefunction
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.binary_encoding import BinaryEncodingSynthesizer

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_random_wavefunction


def _matrix_qubit_counts(wf: Wavefunction) -> tuple[int, int]:
    """Derive qubit counts from the determinant matrix.

    Returns:
        ``(n_system, n_ancilla)`` where

        - *n_system* is the number of system qubits
        - *n_ancilla* is the number of ancilla qubits.

    """
    num_orbitals = len(wf.get_orbitals().get_active_space_indices()[0])
    dets = wf.get_active_determinants()
    bitstrings = []
    for det in dets:
        alpha_str, beta_str = det.to_binary_strings(num_orbitals)
        bitstrings.append(beta_str[::-1] + alpha_str[::-1])

    n_system = len(bitstrings[0])
    matrix = np.array([[int(b) for b in bs] for bs in bitstrings], dtype=np.int8).T
    gf2x_result = gf2x_with_tracking(matrix, skip_diagonal_reduction=True, forward_only=True)

    synthesizer = BinaryEncodingSynthesizer.from_matrix(gf2x_result.reduced_matrix)
    ops = synthesizer.to_gf2x_operations(
        num_local_qubits=n_system,
        active_qubit_indices=gf2x_result.row_map,
        ancilla_start=n_system,
    )
    max_select_ancilla = 0
    for op_name, op_args in ops:
        if op_name in ("select", "select_and"):
            _, address_qubits, _ = op_args
            n_addr = len(address_qubits)
            max_select_ancilla = max(max_select_ancilla, n_addr - 1)

    return n_system, max_select_ancilla


@pytest.fixture
def ozone_wf(test_data_files_path) -> Wavefunction:
    """Load the ozone SCI wavefunction from test data."""
    return Wavefunction.from_json_file(str(test_data_files_path / "ozone_sparse_ci_wavefunction.wavefunction.json"))


class TestSparseIsometryBinaryEncoding:
    """Tests for the sparse isometry binary encoding state preparation."""

    def test_ozone(self, ozone_wf):
        """End-to-end: ozone SCI wavefunction → run() → Circuit → estimate()."""
        binary_encoding_prep = create("state_prep", "sparse_isometry_binary_encoding")
        circuit = binary_encoding_prep.run(ozone_wf)
        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"

        result = circuit.estimate()
        assert isinstance(result, qsharp.estimator.EstimatorResult)
        lc = result["logicalCounts"]
        assert lc["numQubits"] == 12  # 10 system qubits + 2 ancilla qubits
        assert lc["tCount"] == 7
        assert lc["rotationCount"] == 7
        assert lc["cczCount"] == 9
        assert lc["measurementCount"] == 0

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    def test_ozone_statevector(self, ozone_wf):
        """Simulate the ozone circuit and verify the statevector matches.

        The circuit may use ancilla qubits beyond the system register.
        Ancilla qubits sit on the high-index qubits and are returned
        to |0⟩ after uncomputation, so the system-register amplitudes
        live in the first 2^n_system entries of the full statevector.
        """
        from qiskit.quantum_info import Statevector  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        binary_encoding_prep = create("state_prep", "sparse_isometry_binary_encoding")
        circuit = binary_encoding_prep.run(ozone_wf)
        expected_sv = create_statevector_from_wavefunction(ozone_wf, normalize=True)
        n_system = int(np.log2(len(expected_sv)))

        qc = circuit.get_qiskit_circuit()
        sim_data = np.array(Statevector.from_instruction(qc))

        # Extract system-register amplitudes (ancilla qubits should be |0⟩).
        system_sv = sim_data[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    @pytest.mark.parametrize(
        ("n_electrons", "n_orbitals", "n_dets", "seed"),
        [
            (6, 6, 20, 42),
            (8, 8, 50, 99),
        ],
        ids=["6e6o_20det", "8e8o_50det"],
    )
    def test_random_wavefunction(self, n_electrons, n_orbitals, n_dets, seed):
        """End-to-end: random wavefunction → run() → Circuit → estimate().

        The expected qubit count is decomposed into system qubits (from the
        matrix dimensions) and ancilla qubits (from the compiled Q# circuit).
        """
        wf = create_random_wavefunction(
            n_electrons=n_electrons,
            n_orbitals=n_orbitals,
            n_dets=n_dets,
            seed=seed,
        )

        binary_encoding_prep = create("state_prep", "sparse_isometry_binary_encoding")
        circuit = binary_encoding_prep.run(wf)
        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"

        # Derive qubit counts from the matrix.
        # Dense register qubits are system qubits (via rowMap); the extra
        # dense_size - 1 qubits are PreparePureStateD's internal scratch.
        n_system, n_ancilla = _matrix_qubit_counts(wf)
        assert n_system == 2 * n_orbitals
        expected_total = n_system + n_ancilla

        # Resource estimate must agree.
        lc = circuit.estimate()["logicalCounts"]
        assert lc["numQubits"] == expected_total
        assert lc["cczCount"] > 0

    def test_default_settings(self):
        """Default settings: include_negative_controls=True, measurement_based_uncompute=False."""
        state_prep = SparseIsometryBinaryEncodingStatePreparation()
        assert state_prep.settings().get("include_negative_controls") is True
        assert state_prep.settings().get("measurement_based_uncompute") is False

    def test_ozone_negative_controls_disabled(self, ozone_wf):
        """Ozone with include_negative_controls=False produces different resource counts."""
        prep = create("state_prep", "sparse_isometry_binary_encoding", include_negative_controls=False)
        circuit = prep.run(ozone_wf)
        assert isinstance(circuit, Circuit)
        lc = circuit.estimate()["logicalCounts"]
        assert lc["numQubits"] == 11
        assert lc["tCount"] == 7
        assert lc["rotationCount"] == 7
        assert lc["cczCount"] == 5

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.parametrize(
        ("n_electrons", "n_orbitals", "n_dets", "seed"),
        [
            (6, 6, 20, 42),
            (6, 6, 30, 7),
        ],
        ids=["6e6o_20det", "6e6o_30det"],
    )
    def test_random_wavefunction_statevector(self, n_electrons, n_orbitals, n_dets, seed):
        """Simulate random-wavefunction circuits and verify the statevector matches."""
        from qiskit.quantum_info import Statevector  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        wf = create_random_wavefunction(
            n_electrons=n_electrons,
            n_orbitals=n_orbitals,
            n_dets=n_dets,
            seed=seed,
        )
        circuit = create("state_prep", "sparse_isometry_binary_encoding").run(wf)
        expected_sv = create_statevector_from_wavefunction(wf, normalize=True)
        n_system = 2 * n_orbitals

        qc = circuit.get_qiskit_circuit()
        sim_data = np.array(Statevector.from_instruction(qc))

        system_sv = sim_data[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    def test_encode_empty_ops(self):
        """Empty ops list produces empty encoded list."""
        assert _encode_gf2x_ops_for_qs([]) == []

    def test_encode_cx(self):
        """CX op is encoded as MatrixCompressionOp('CX', ...)."""
        encoded = _encode_gf2x_ops_for_qs([("cx", (0, 1))])
        assert len(encoded) == 1
        assert encoded[0].name == "CX"
        assert encoded[0].qubits == [0, 1]

    def test_encode_x(self):
        """X op is encoded as MatrixCompressionOp('X', [qubit])."""
        encoded = _encode_gf2x_ops_for_qs([("x", 5)])
        assert encoded[0].name == "X"
        assert encoded[0].qubits == [5]

    def test_encode_swap(self):
        """SWAP op is encoded correctly."""
        encoded = _encode_gf2x_ops_for_qs([("swap", (2, 3))])
        assert encoded[0].name == "SWAP"
        assert encoded[0].qubits == [2, 3]

    def test_encode_ccx(self):
        """CCX op encodes [ctrl1, ctrl2, target]."""
        encoded = _encode_gf2x_ops_for_qs([("ccx", (4, 0, 1))])
        assert encoded[0].name == "CCX"
        assert encoded[0].qubits == [0, 1, 4]

    def test_encode_select(self):
        """SELECT op preserves data table and records address qubit count."""
        data = [[True, False], [False, True]]
        encoded = _encode_gf2x_ops_for_qs([("select", (data, [0, 1], [2, 3]))])
        assert encoded[0].name == "SELECT"
        assert encoded[0].qubits == [0, 1, 2, 3]
        assert encoded[0].control_state == 2
        assert encoded[0].lookup_data is data

    def test_encode_select_and(self):
        """SELECT_AND is used for measurement-based ops."""
        encoded = _encode_gf2x_ops_for_qs([("select_and", ([[True]], [0], [1]))])
        assert encoded[0].name == "SELECT_AND"

    def test_encode_reversed_order(self):
        """Encoded ops appear in reversed order relative to input."""
        encoded = _encode_gf2x_ops_for_qs([("x", 0), ("cx", (1, 2)), ("x", 3)])
        assert [op.name for op in encoded] == ["X", "CX", "X"]
        assert encoded[0].qubits == [3]
        assert encoded[2].qubits == [0]

    def test_encode_to_dict_keys(self):
        """to_dict produces camelCase keys required by Q# bridge."""
        encoded = _encode_gf2x_ops_for_qs([("cx", (0, 1))])
        assert set(encoded[0].to_dict().keys()) == {"name", "qubits", "controlState", "lookupData"}
