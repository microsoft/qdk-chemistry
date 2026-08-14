"""Tests for the sparse isometry with binary encoding state preparation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk import TargetProfile
from qdk.estimator import EstimatorResult

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation._binary_encoding_utils import (
    MatrixCompressionType,
    RefTableau,
    _BinaryEncodingSynthesizer,
)
from qdk_chemistry.algorithms.state_preparation.sparse_isometry import gf2x_with_tracking
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Configuration,
    QubitOperator,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.symmetry import SymmetryLabel, axes
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.qsharp import (
    QSHARP_UTILS,
    create_qsharp_context,
    get_qsharp_context,
    use_qsharp_context,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_random_wavefunction


def _matrix_qubit_counts(wf: Wavefunction) -> tuple[int, int]:
    """Derive qubit counts from the determinant matrix, accounting for the ancilla pool.

    Returns:
        ``(n_system, n_ancilla)`` where

        - *n_system* is the number of system qubits
        - *n_ancilla* is the number of extra ancilla qubits beyond the system register
          after subtracting Pool A (idle GF2X qubits: system qubits absent from row_map).

    """
    num_orbitals = len(list(wf.get_orbitals().active_indices().indices(SymmetryLabel([axes.alpha()]))))
    dets = wf.get_active_determinants()
    bitstrings = []
    for det in dets:
        alpha_str, beta_str = det.to_binary_strings(num_orbitals)
        bitstrings.append(beta_str[::-1] + alpha_str[::-1])

    n_system = len(bitstrings[0])
    matrix = np.array([[int(b) for b in bs] for bs in bitstrings], dtype=np.int8).T
    gf2x_result = gf2x_with_tracking(matrix, skip_diagonal_reduction=True, forward_only=True)

    ops, _, _ = _BinaryEncodingSynthesizer(
        RefTableau(gf2x_result.reduced_matrix),
    ).synthesize(
        num_local_qubits=n_system,
        active_qubit_indices=gf2x_result.row_map,
        ancilla_start=n_system,
    )
    naive_ancilla = max(
        (
            op.control_state - 1
            for op in ops
            if op.name in (MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND)
        ),
        default=0,
    )

    # Pool A: system qubits not present in row_map (never touched by binary encoding ops)
    active_set = {int(q) for q in gf2x_result.row_map}
    pool_a = len(set(range(n_system)) - active_set)

    # Actual ancilla = max(0, naive - pool_a)
    actual_ancilla = max(0, naive_ancilla - pool_a)
    return n_system, actual_ancilla


@pytest.fixture
def ozone_wf(test_data_files_path) -> Wavefunction:
    """Load the ozone SCI wavefunction from test data."""
    return Wavefunction.from_hdf5_file(str(test_data_files_path / "ozone_sparse_ci_wavefunction.wavefunction.h5"))


@pytest.fixture(scope="session")
def _adaptive_context():
    """Build one Adaptive Q# interpreter for the whole session.

    Each ``qdk.Context`` owns an unsendable Rust interpreter, so building one per test
    leaves interpreters to be dropped on whichever thread happens to run the collection.
    """
    return create_qsharp_context(target_profile=TargetProfile.Adaptive_RIF)


@pytest.fixture
def adaptive_profile(_adaptive_context):
    """Compile Q# for an Adaptive profile for the duration of the test.

    The package default is ``TargetProfile.Base``, which has no mid-circuit measurement;
    ``Std.Intrinsic.AND`` then uncomputes unitarily instead of by measurement.
    """
    with use_qsharp_context(_adaptive_context):
        # Assert the override actually took effect; otherwise these tests would silently
        # run on the Base default and pass for the wrong reason.
        active = get_qsharp_context().get_target_profile()
        assert "adaptive" in str(active).lower(), f"Expected an Adaptive profile, got {active}."
        yield


def _assert_uses_binary_encoding(circuit: Circuit) -> None:
    """Assert the circuit was built by the binary-encoding composition, not the dense fallback.

    Without this the suite cannot distinguish a working binary encoding from a silent
    fallback to the plain sparse-isometry path in :meth:`_run_binary_encoding`.

    Args:
        circuit: Circuit returned by the sparse isometry state preparation.

    """
    factory = circuit._qsharp_factory
    assert factory is not None, "Binary encoding must produce a Q# circuit, not a QASM fallback."
    assert factory.program is QSHARP_UTILS.BinaryEncoding.MakeComposeBinaryEncodingCircuit
    assert factory.parameter["binaryEncodingOps"], "Binary-encoding op sequence must not be empty."


class TestSparseIsometryBinaryEncoding:
    """Tests for the sparse isometry binary encoding state preparation."""

    def test_ozone(self, ozone_wf):
        """End-to-end: ozone SCI wavefunction → run() → Circuit → estimate()."""
        binary_encoding_prep = create("state_prep", "sparse_isometry", binary_encoding=True)
        circuit = binary_encoding_prep.run(ozone_wf)
        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"
        _assert_uses_binary_encoding(circuit)

        result = circuit.estimate()
        assert isinstance(result, EstimatorResult)
        lc = result["logicalCounts"]
        assert lc["numQubits"] == 10  # 10 system qubits; pool covers all ancilla
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

        binary_encoding_prep = create("state_prep", "sparse_isometry", binary_encoding=True)
        circuit = binary_encoding_prep.run(ozone_wf)
        expected_sv = create_statevector_from_wavefunction(ozone_wf, normalize=True)
        n_system = int(np.log2(len(expected_sv)))
        _assert_uses_binary_encoding(circuit)

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

        binary_encoding_prep = create("state_prep", "sparse_isometry", binary_encoding=True)
        circuit = binary_encoding_prep.run(wf)
        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"
        _assert_uses_binary_encoding(circuit)

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
        state_prep = create("state_prep", "sparse_isometry", binary_encoding=True)
        assert state_prep.settings().get("include_negative_controls") is True
        assert state_prep.settings().get("measurement_based_uncompute") is False

    def test_ozone_negative_controls_disabled(self, ozone_wf):
        """Ozone with include_negative_controls=False produces different resource counts."""
        prep = create("state_prep", "sparse_isometry", binary_encoding=True, include_negative_controls=False)
        circuit = prep.run(ozone_wf)
        assert isinstance(circuit, Circuit)
        _assert_uses_binary_encoding(circuit)
        lc = circuit.estimate()["logicalCounts"]
        assert lc["numQubits"] == 10  # 10 system qubits; pool covers the 1 ancilla
        assert lc["tCount"] == 7
        assert lc["rotationCount"] == 7
        assert lc["cczCount"] == 5

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.parametrize(
        ("n_electrons", "n_orbitals", "n_dets", "seed", "binary_encoding", "include_negative_controls"),
        [
            (6, 6, 20, 42, True, True),
            (6, 6, 20, 42, True, False),
            (6, 6, 20, 42, False, True),
            (6, 6, 30, 7, True, True),
            (6, 6, 30, 7, False, True),
        ],
        ids=[
            "6e6o_20det_binenc_negctrl",
            "6e6o_20det_binenc_no_negctrl",
            "6e6o_20det_no_binenc",
            "6e6o_30det_binenc_negctrl",
            "6e6o_30det_no_binenc",
        ],
    )
    def test_qiskit_isometry_dense_prep_statevector(
        self, n_electrons, n_orbitals, n_dets, seed, binary_encoding, include_negative_controls
    ):
        """Sparse isometry with qiskit_regular_isometry as dense prep, with/without binary encoding."""
        from qiskit.quantum_info import Statevector  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        wf = create_random_wavefunction(
            n_electrons=n_electrons,
            n_orbitals=n_orbitals,
            n_dets=n_dets,
            seed=seed,
        )
        prep = create(
            "state_prep",
            "sparse_isometry",
            dense_state_prep=AlgorithmRef("state_prep", "qiskit_regular_isometry"),
            binary_encoding=binary_encoding,
            include_negative_controls=include_negative_controls,
        )
        circuit = prep.run(wf)
        assert isinstance(circuit, Circuit)

        expected_sv = create_statevector_from_wavefunction(wf, normalize=True)
        n_system = 2 * n_orbitals

        qc = circuit.get_qiskit_circuit()
        sim_data = np.array(Statevector.from_instruction(qc))
        system_sv = sim_data[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

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
        circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(wf)
        expected_sv = create_statevector_from_wavefunction(wf, normalize=True)
        n_system = 2 * n_orbitals
        _assert_uses_binary_encoding(circuit)

        qc = circuit.get_qiskit_circuit()
        sim_data = np.array(Statevector.from_instruction(qc))

        system_sv = sim_data[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    @pytest.mark.parametrize(
        ("n_electrons", "n_orbitals", "n_dets", "seed", "expected_n_qubits"),
        [
            # 4 electrons, 3 orbitals, 9 determinants: the full space has only
            # ceil(6 choose 4) = 15 states.  After GF2+X (forward-only, no
            # diagonal reduction) the REF matrix has rank 4 (4 rows) but still
            # 9 columns, so dense_size = RefTableau.dense_register_width(9) = 4 = num_rows.
            # The condition dense_size >= num_rows triggers the fallback.
            (4, 3, 9, 0, 6),
            (4, 3, 9, 1, 6),
        ],
        ids=["4e3o_9det_seed0", "4e3o_9det_seed1"],
    )
    def test_fallback_to_dense_gf2x(self, n_electrons, n_orbitals, n_dets, seed, expected_n_qubits):
        """Wavefunction where after GF2+X the REF matrix is already dense falls back to dense+GF2X."""
        wf = create_random_wavefunction(
            n_electrons=n_electrons,
            n_orbitals=n_orbitals,
            n_dets=n_dets,
            seed=seed,
        )

        # Confirm this case is genuinely a fallback case before testing.
        num_orbitals = len(list(wf.get_orbitals().active_indices().indices(SymmetryLabel([axes.alpha()]))))
        bitstrings = []
        for det in wf.get_active_determinants():
            a, b = det.to_binary_strings(num_orbitals)
            bitstrings.append(b[::-1] + a[::-1])
        mat = np.array([[int(c) for c in bs] for bs in bitstrings], dtype=np.int8).T
        gf2x_result = gf2x_with_tracking(mat, skip_diagonal_reduction=True, forward_only=True)
        num_rows, num_cols = gf2x_result.reduced_matrix.shape
        dense_size = 1 if num_cols < 2 else math.ceil(math.log2(num_cols))
        assert dense_size >= num_rows, f"Expected fallback: dense_size={dense_size} must be >= num_rows={num_rows}"

        circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(wf)
        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"

        # The fallback must not go through the binary-encoding composition.
        assert circuit._qsharp_factory.program is QSHARP_UTILS.StatePreparation.MakeComposeSparseIsometryCircuit

        lc = circuit.estimate()["logicalCounts"]
        # No binary-encoding SELECT/SELECT_AND ops in the fallback path.
        assert lc["cczCount"] == 0
        # System qubits only — PreparePureStateD does not need external ancilla.
        assert lc["numQubits"] == expected_n_qubits

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.parametrize(
        ("n_electrons", "n_orbitals", "n_dets", "seed"),
        [
            (4, 3, 9, 0),
            (4, 3, 9, 1),
        ],
        ids=["4e3o_9det_seed0", "4e3o_9det_seed1"],
    )
    def test_fallback_statevector(self, n_electrons, n_orbitals, n_dets, seed):
        """Fallback circuit produces the correct statevector (Qiskit simulation).

        Validates that the fallback dense+GF2X path correctly encodes the target
        wavefunction amplitudes, not merely that it runs without error.
        """
        from qiskit.quantum_info import Statevector  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        wf = create_random_wavefunction(
            n_electrons=n_electrons,
            n_orbitals=n_orbitals,
            n_dets=n_dets,
            seed=seed,
        )
        circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(wf)
        expected_sv = create_statevector_from_wavefunction(wf, normalize=True)
        n_system = 2 * n_orbitals

        qc = circuit.get_qiskit_circuit()
        sim_data = np.array(Statevector.from_instruction(qc))
        system_sv = sim_data[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )


class TestCreateDense:
    """Tests for :meth:`SparseIsometryStatePreparation.create_dense`.

    ``create_dense`` returns the dense-loading stage alone, on the same full-width
    register as ``run``, so the isometry cost can be recovered by subtracting the two
    resource estimates.
    """

    ISOMETRY_OP_KEYS = ("expansionOps", "binaryEncodingOps", "gaussianEliminationOps")

    @pytest.mark.parametrize("binary_encoding", [False, True], ids=["no_binenc", "binenc"])
    def test_strips_isometry_ops(self, ozone_wf, binary_encoding):
        """The dense stage keeps the composition program but empties every isometry op list."""
        prep = create("state_prep", "sparse_isometry", binary_encoding=binary_encoding)
        full = prep.run(ozone_wf)
        dense = prep.create_dense(ozone_wf)

        full_factory = full._qsharp_factory
        dense_factory = dense._qsharp_factory
        assert dense_factory is not None
        assert dense_factory.program is full_factory.program

        present = [key for key in self.ISOMETRY_OP_KEYS if key in full_factory.parameter]
        assert present, "The composed circuit must carry at least one isometry op list."
        assert any(full_factory.parameter[key] for key in present), "The full circuit must apply isometry gates."
        assert all(dense_factory.parameter[key] == [] for key in present)

        # Dense loading is untouched and still acts on the full register.
        assert dense_factory.parameter["numQubits"] == full_factory.parameter["numQubits"]
        assert dense_factory.parameter["embeddingMap"] == full_factory.parameter["embeddingMap"]
        assert dense_factory.parameter["denseParams"].stateVector == full_factory.parameter["denseParams"].stateVector

    def test_binary_encoding_shrinks_the_dense_stage(self, ozone_wf):
        """Binary encoding loads 2^ceil(log2(d)) amplitudes instead of 2^rank, so its dense stage is smaller."""
        gf2x_prep = create("state_prep", "sparse_isometry", binary_encoding=False)
        binenc_prep = create("state_prep", "sparse_isometry", binary_encoding=True)

        gf2x_dense = gf2x_prep.create_dense(ozone_wf)
        binenc_dense = binenc_prep.create_dense(ozone_wf)
        _assert_uses_binary_encoding(binenc_prep.run(ozone_wf))

        n_dets = len(ozone_wf.get_active_determinants())
        gf2x_sv = gf2x_dense._qsharp_factory.parameter["denseParams"].stateVector
        binenc_sv = binenc_dense._qsharp_factory.parameter["denseParams"].stateVector
        assert len(binenc_sv) == 2 ** math.ceil(math.log2(n_dets))
        assert len(binenc_sv) < len(gf2x_sv)

        # A smaller amplitude register means a cheaper dense load.
        gf2x_counts = gf2x_dense.estimate()["logicalCounts"]
        binenc_counts = binenc_dense.estimate()["logicalCounts"]
        assert binenc_counts["rotationCount"] < gf2x_counts["rotationCount"]
        assert binenc_counts["numQubits"] == gf2x_counts["numQubits"]

    def test_binary_encoding_isometry_cost_by_subtraction(self, ozone_wf):
        """Under binary encoding the isometry contributes the multi-controlled gates."""
        prep = create("state_prep", "sparse_isometry", binary_encoding=True)
        full = prep.run(ozone_wf).estimate()["logicalCounts"]
        dense = prep.create_dense(ozone_wf).estimate()["logicalCounts"]

        assert dense["cczCount"] == 0
        assert full["cczCount"] > 0
        assert full["rotationCount"] == dense["rotationCount"]

    def test_gf2x_isometry_is_clifford_only(self, ozone_wf):
        """Without binary encoding the isometry is only CX/X, so it adds no non-Clifford cost."""
        prep = create("state_prep", "sparse_isometry", binary_encoding=False)
        full = prep.run(ozone_wf).estimate()["logicalCounts"]
        dense = prep.create_dense(ozone_wf).estimate()["logicalCounts"]

        for key in ("tCount", "rotationCount", "cczCount"):
            assert full[key] == dense[key]

    def test_single_determinant_has_no_dense_stage(self):
        """A single reference is pure expansion, so its dense stage is the empty circuit."""
        wf = create_random_wavefunction(n_electrons=4, n_orbitals=4, n_dets=1, seed=11)
        prep = create("state_prep", "sparse_isometry")

        full = prep.run(wf)
        dense = prep.create_dense(wf)
        assert any(full._qsharp_factory.parameter["bitStrings"])
        assert not any(dense._qsharp_factory.parameter["bitStrings"])

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.parametrize("binary_encoding", [False, True], ids=["no_binenc", "binenc"])
    def test_qiskit_dense_prep(self, ozone_wf, binary_encoding):
        """The Qiskit composition path drops the isometry gates but keeps the dense load."""
        prep = create(
            "state_prep",
            "sparse_isometry",
            dense_state_prep=AlgorithmRef("state_prep", "qiskit_regular_isometry"),
            binary_encoding=binary_encoding,
        )
        full_qc = prep.run(ozone_wf).get_qiskit_circuit()
        dense_qc = prep.create_dense(ozone_wf).get_qiskit_circuit()

        assert dense_qc.num_qubits == full_qc.num_qubits
        assert dense_qc.size() < full_qc.size()
        assert dense_qc.size() > 0

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    def test_qiskit_binary_encoding_shrinks_the_dense_stage(self, ozone_wf):
        """Binary encoding's smaller amplitude register makes its dense load cheaper on Qiskit too."""
        kwargs = {"dense_state_prep": AlgorithmRef("state_prep", "qiskit_regular_isometry")}
        gf2x_dense = create("state_prep", "sparse_isometry", binary_encoding=False, **kwargs).create_dense(ozone_wf)
        binenc_dense = create("state_prep", "sparse_isometry", binary_encoding=True, **kwargs).create_dense(ozone_wf)

        gf2x_qc = gf2x_dense.get_qiskit_circuit()
        binenc_qc = binenc_dense.get_qiskit_circuit()
        assert binenc_qc.num_qubits == gf2x_qc.num_qubits
        assert binenc_qc.size() < gf2x_qc.size()


class TestMatrixCompressionQsharpInterop:
    """Regression tests for Q# compilation of serialized matrix-compression operations."""

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    def test_base_nonbinary_f2_expansion_compiles(self, wavefunction_10e6o):
        """The coherent three-determinant F2 expansion compiles through QIR and Qiskit."""
        with use_qsharp_context(create_qsharp_context(target_profile=TargetProfile.Base)):
            circuit = create(
                "state_prep",
                "sparse_isometry",
                binary_encoding=False,
                dense_state_prep=AlgorithmRef("state_prep", "dense_pure_state"),
                measurement_based_uncompute=False,
            ).run(wavefunction_10e6o)
            assert str(circuit.get_qir())
            qc = circuit.get_qiskit_circuit()

        assert qc.num_qubits == 12
        assert qc.num_clbits == 0


class TestMeasurementBasedUncompute:
    """Tests for binary encoding with measurement-based AND uncomputation.

    This path replaces Toffoli uncomputation with a measurement plus a classically
    controlled correction, so the circuit carries mid-circuit measurement and
    feedforward and can only be simulated by a shot-based simulator.
    """

    @pytest.mark.usefixtures("adaptive_profile")
    def test_trades_toffolis_for_measurements(self, ozone_wf):
        """measurement_based_uncompute=True lowers the CCZ count and introduces measurements."""
        toffoli_circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(ozone_wf)
        measured_circuit = create(
            "state_prep", "sparse_isometry", binary_encoding=True, measurement_based_uncompute=True
        ).run(ozone_wf)
        _assert_uses_binary_encoding(measured_circuit)

        toffoli_counts = toffoli_circuit.estimate()["logicalCounts"]
        measured_counts = measured_circuit.estimate()["logicalCounts"]
        assert toffoli_counts["measurementCount"] == 0
        assert measured_counts["measurementCount"] > 0
        assert measured_counts["cczCount"] < toffoli_counts["cczCount"]

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.usefixtures("adaptive_profile")
    def test_adaptive_qir_preserves_measurement_uncompute(self, ozone_wf):
        """Adaptive QIR imports each measured uncompute as Qiskit control flow."""
        circuit = create("state_prep", "sparse_isometry", binary_encoding=True, measurement_based_uncompute=True).run(
            ozone_wf
        )
        measurement_count = circuit.estimate()["logicalCounts"]["measurementCount"]
        assert measurement_count > 0
        assert str(circuit.get_qir())

        qc = circuit.get_qiskit_circuit()
        ops = qc.count_ops()
        assert qc.num_clbits == measurement_count
        assert ops.get("measure", 0) == measurement_count
        assert ops.get("reset", 0) == measurement_count
        assert ops.get("if_else", 0) == measurement_count

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.usefixtures("adaptive_profile")
    @pytest.mark.parametrize("seed_simulator", [1, 7, 13, 21])
    def test_statevector_independent_of_measurement_outcome(self, ozone_wf, seed_simulator):
        """The prepared state must be the target state whichever way the mid-circuit measurements fall."""
        aer = pytest.importorskip("qiskit_aer")

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        circuit = create("state_prep", "sparse_isometry", binary_encoding=True, measurement_based_uncompute=True).run(
            ozone_wf
        )
        expected_sv = create_statevector_from_wavefunction(ozone_wf, normalize=True)
        n_system = int(np.log2(len(expected_sv)))

        qc = circuit.get_qiskit_circuit().copy()
        # Mid-circuit measurement rules out Statevector.from_instruction, so run a shot on Aer.
        qc.save_statevector()
        result = aer.AerSimulator(method="statevector").run(qc, shots=1, seed_simulator=seed_simulator).result()

        system_sv = np.asarray(result.get_statevector())[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    @pytest.mark.usefixtures("adaptive_profile")
    def test_qiskit_dense_prep_path_prepares_correct_state(self, ozone_wf):
        """SELECT_AND must stay correct on the Qiskit path, which decomposes it without ancilla."""
        from qiskit.quantum_info import Statevector  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        circuit = create(
            "state_prep",
            "sparse_isometry",
            dense_state_prep=AlgorithmRef("state_prep", "qiskit_regular_isometry"),
            binary_encoding=True,
            measurement_based_uncompute=True,
        ).run(ozone_wf)
        expected_sv = create_statevector_from_wavefunction(ozone_wf, normalize=True)
        n_system = int(np.log2(len(expected_sv)))

        qc = circuit.get_qiskit_circuit()
        # The ancilla-free decomposition leaves nothing to uncompute, so no measurement is emitted.
        assert qc.count_ops().get("measure", 0) == 0
        system_sv = np.array(Statevector.from_instruction(qc))[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )

    def test_base_profile_warns_and_uncomputes_unitarily(self, ozone_wf, monkeypatch):
        """On a Base profile the request is honoured unitarily, with a warning instead of a failure."""
        emitted: list[str] = []
        monkeypatch.setattr(Logger, "warn", emitted.append)

        with use_qsharp_context(create_qsharp_context(target_profile=TargetProfile.Base)):
            circuit = create(
                "state_prep", "sparse_isometry", binary_encoding=True, measurement_based_uncompute=True
            ).run(ozone_wf)
            _assert_uses_binary_encoding(circuit)
            counts = circuit.estimate()["logicalCounts"]

        assert any("measurement_based_uncompute" in message and "Base" in message for message in emitted), (
            f"Expected a fallback warning naming the profile, got: {emitted}"
        )
        # Std.Intrinsic.AND uncomputes unitarily off-Adaptive, so no measurement survives.
        assert counts["measurementCount"] == 0

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
    def test_base_profile_fallback_prepares_correct_state(self, ozone_wf):
        """The unitary fallback must still prepare the target state, not merely compile."""
        from qiskit.quantum_info import Statevector  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

        with use_qsharp_context(create_qsharp_context(target_profile=TargetProfile.Base)):
            circuit = create(
                "state_prep", "sparse_isometry", binary_encoding=True, measurement_based_uncompute=True
            ).run(ozone_wf)
            qc = circuit.get_qiskit_circuit()

        expected_sv = create_statevector_from_wavefunction(ozone_wf, normalize=True)
        n_system = int(np.log2(len(expected_sv)))
        # No mid-circuit measurement, so the statevector is well defined without a shot-based run.
        assert qc.count_ops().get("measure", 0) == 0
        system_sv = np.array(Statevector.from_instruction(qc))[: 2**n_system]
        overlap = np.abs(np.vdot(expected_sv, system_sv))
        assert np.isclose(
            overlap, 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
        )


class TestBinaryEncodingWithQPE:
    """Tests for binary encoding state preparation integrated with QPE circuit builders."""

    def test_ancilla_overflow_iterative_qpe_qubit_allocation(self):
        """Verify QPE qubit allocation when binary encoding state prep has extra ancilla.

        Uses a 6e6o 20-determinant wavefunction where the binary encoding needs 3 ancilla
        qubits beyond the idle pool. The state prep alone uses 15 qubits (12 system + 3
        extra ancilla). When composed with iterative QPE, the total qubit count should be
        1 (QPE control) + 15 (state prep total) = 16, because QPE only manages the system
        register mapping but the state prep independently allocates its own ancilla.
        """
        wf = create_random_wavefunction(n_electrons=6, n_orbitals=6, n_dets=20, seed=42)

        # Build binary encoding state prep — requires ancilla beyond the pool
        state_prep_circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(wf)
        _assert_uses_binary_encoding(state_prep_circuit)
        sp_lc = state_prep_circuit.estimate()["logicalCounts"]
        num_system_qubits = 2 * 6  # 12 system qubits
        state_prep_total_qubits = sp_lc["numQubits"]
        extra_ancilla = state_prep_total_qubits - num_system_qubits
        assert extra_ancilla > 0, "This test requires a case with ancilla overflow"

        # Create a 12-qubit Hamiltonian matching the system size
        qubit_hamiltonian = QubitOperator(
            pauli_strings=["Z" + "I" * (num_system_qubits - 1), "I" * (num_system_qubits - 1) + "Z"],
            coefficients=np.array([0.5, 0.25]),
        )

        # Build iterative QPE circuits
        num_bits = 4
        circuit_builder = create(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=num_bits,
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        iqpe_circuits = circuit_builder.run(
            state_preparation=state_prep_circuit,
            qubit_hamiltonian=qubit_hamiltonian,
        )

        # Iterative QPE: 1 control + state_prep_total_qubits
        # QPE maps systems=[1..12] but state prep internally allocates 3 more ancilla
        # beyond the system register, so total = 1 + 15 = 16
        expected_qpe_qubits = 1 + state_prep_total_qubits
        for i, qpe_circuit in enumerate(iqpe_circuits):
            lc = qpe_circuit.estimate()["logicalCounts"]
            assert lc["numQubits"] == expected_qpe_qubits, (
                f"Iteration {i}: expected {expected_qpe_qubits} qubits "
                f"(1 control + {state_prep_total_qubits} state_prep), got {lc['numQubits']}. "
                f"State prep has {extra_ancilla} extra ancilla beyond {num_system_qubits} system qubits."
            )

    def test_ancilla_overflow_standard_qpe_qubit_allocation(self):
        """Verify QPE qubit allocation with standard QPE when state prep has extra ancilla.

        Uses the same 6e6o wavefunction with ancilla overflow. Standard QPE uses num_bits
        phase qubits. The total should be num_bits + state_prep_total_qubits because the
        state prep's ancilla are allocated independently of QPE's system qubit mapping.
        """
        wf = create_random_wavefunction(n_electrons=6, n_orbitals=6, n_dets=20, seed=42)

        state_prep_circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(wf)
        _assert_uses_binary_encoding(state_prep_circuit)
        sp_lc = state_prep_circuit.estimate()["logicalCounts"]
        num_system_qubits = 12
        state_prep_total_qubits = sp_lc["numQubits"]
        extra_ancilla = state_prep_total_qubits - num_system_qubits
        assert extra_ancilla > 0, "This test requires a case with ancilla overflow"

        qubit_hamiltonian = QubitOperator(
            pauli_strings=["Z" + "I" * (num_system_qubits - 1), "I" * (num_system_qubits - 1) + "Z"],
            coefficients=np.array([0.5, 0.25]),
        )

        # Build standard QPE circuit
        num_bits = 4
        circuit_builder = create(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=num_bits,
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        qpe_circuits = circuit_builder.run(
            state_preparation=state_prep_circuit,
            qubit_hamiltonian=qubit_hamiltonian,
        )

        # Standard QPE: num_bits phase qubits + state_prep_total_qubits
        # = 4 + 15 = 19
        expected_qpe_qubits = num_bits + state_prep_total_qubits
        lc = qpe_circuits[0].estimate()["logicalCounts"]
        assert lc["numQubits"] == expected_qpe_qubits, (
            f"Expected {expected_qpe_qubits} qubits "
            f"(num_bits={num_bits} + state_prep={state_prep_total_qubits}), got {lc['numQubits']}. "
            f"State prep has {extra_ancilla} extra ancilla beyond {num_system_qubits} system qubits."
        )

    def test_model_hamiltonian_iqpe_with_ancilla(self):
        """End-to-end iterative QPE with binary encoding state prep that requires ancilla.

        Uses a disordered Heisenberg model (8 qubits, open boundary, local Z fields)
        whose ground state is truncated to 26 determinants — enough to require 3 extra
        ancilla qubits in binary encoding (state prep uses 11 total qubits).

        Validates that:
        1. The state prep circuit needs ancilla beyond the system register.
        2. Iterative QPE runs successfully with the enlarged qubit space.
        3. The recovered energy is within the QPE resolution of the exact ground energy.
        """
        # --- Model Hamiltonian: disordered Heisenberg (XX + ZZ) + local Z fields ---
        n = 8
        n_orbitals = 4
        pauli_strings: list[str] = []
        coefficients_list: list[float] = []
        for i in range(n - 1):  # open boundary
            j_coupling = 1.0 + 0.3 * (i % 3 - 1)
            for pauli in ["X", "Z"]:
                s = ["I"] * n
                s[i] = pauli
                s[i + 1] = pauli
                pauli_strings.append("".join(s))
                coefficients_list.append(j_coupling)
        for i in range(n):
            s = ["I"] * n
            s[i] = "Z"
            pauli_strings.append("".join(s))
            coefficients_list.append(0.3 * (i - n / 2))
        coefficients_arr = np.array(coefficients_list)

        # Exact diagonalization
        h_dense = pauli_to_dense_matrix(pauli_strings, coefficients_arr)
        eigenvalues, eigenvectors = np.linalg.eigh(h_dense)
        gs = eigenvectors[:, 0].real
        gs_energy = float(eigenvalues[0])

        # Truncate to top 26 determinants (~94% overlap with ground state)
        sorted_indices = np.argsort(-np.abs(gs))
        n_dets = 26
        top_indices = sorted_indices[:n_dets]
        top_amps = gs[top_indices]
        top_amps = top_amps / np.linalg.norm(top_amps)

        # Build wavefunction from bitstrings
        mapping = {(1, 1): "2", (1, 0): "u", (0, 1): "d", (0, 0): "0"}
        configs = []
        for idx in top_indices:
            bits = format(idx, f"0{n}b")
            alpha_bits = [int(bits[n - 1 - i]) for i in range(n_orbitals)]
            beta_bits = [int(bits[n - 1 - i]) for i in range(n_orbitals, n)]
            config_str = "".join(mapping[alpha_bits[k], beta_bits[k]] for k in range(n_orbitals))
            configs.append(Configuration.from_spin_half_string(config_str))
        from .test_helpers import create_test_orbitals  # noqa: PLC0415

        orbitals = create_test_orbitals(n_orbitals)
        wf = Wavefunction(StateVectorContainer(top_amps, configs, orbitals))

        # Binary encoding state prep — must require ancilla
        state_prep_circuit = create("state_prep", "sparse_isometry", binary_encoding=True).run(wf)
        _assert_uses_binary_encoding(state_prep_circuit)
        sp_lc = state_prep_circuit.estimate()["logicalCounts"]
        state_prep_qubits = sp_lc["numQubits"]
        extra_ancilla = state_prep_qubits - n
        assert extra_ancilla > 0, f"Expected ancilla overflow but state prep uses only {state_prep_qubits} qubits"

        # Qubit Hamiltonian
        qubit_hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients_arr)

        # Run iterative QPE
        num_bits = 8
        evolution_time = float(np.pi / qubit_hamiltonian.schatten_norm)
        iqpe = create("phase_estimation", "qdk_iterative", shots_per_bit=5)
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))
        iqpe.settings().set(
            "qpe_circuit_builder",
            AlgorithmRef(
                "qpe_circuit_builder",
                "qdk_iterative",
                num_bits=num_bits,
                unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=evolution_time),
                controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            ),
        )

        result = iqpe.run(state_preparation=state_prep_circuit, qubit_hamiltonian=qubit_hamiltonian)

        # Energy resolution with num_bits: 2*pi / (t * 2^num_bits)
        energy_resolution = 2 * np.pi / (evolution_time * 2**num_bits)
        assert abs(result.raw_energy - gs_energy) < energy_resolution, (
            f"QPE energy {result.raw_energy:.6f} deviates from ground energy {gs_energy:.6f} "
            f"by more than the {num_bits}-bit resolution ({energy_resolution:.4f}). "
            f"State prep used {state_prep_qubits} qubits ({extra_ancilla} ancilla)."
        )
