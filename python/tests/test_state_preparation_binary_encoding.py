"""Tests for the sparse isometry with binary encoding state preparation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.estimator import EstimatorResult

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import SparseIsometryBinaryEncodingStatePreparation
from qdk_chemistry.algorithms.state_preparation.sparse_isometry import gf2x_with_tracking
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Configuration,
    QubitHamiltonian,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.binary_encoding import MatrixCompressionType, RefTableau, _BinaryEncodingSynthesizer
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.phase import energy_from_phase

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
    num_orbitals = len(wf.get_orbitals().get_active_space_indices()[0])
    dets = wf.get_active_determinants()
    bitstrings = []
    for det in dets:
        alpha_str, beta_str = det.to_binary_strings(num_orbitals)
        bitstrings.append(beta_str[::-1] + alpha_str[::-1])

    n_system = len(bitstrings[0])
    matrix = np.array([[int(b) for b in bs] for bs in bitstrings], dtype=np.int8).T
    gf2x_result = gf2x_with_tracking(matrix, skip_diagonal_reduction=True, forward_only=True)

    ops, bijection, dense_size = _BinaryEncodingSynthesizer(
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
        assert lc["numQubits"] == 10  # 10 system qubits; pool covers the 1 ancilla
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
        num_orbitals = len(wf.get_orbitals().get_active_space_indices()[0])
        bitstrings = []
        for det in wf.get_active_determinants():
            a, b = det.to_binary_strings(num_orbitals)
            bitstrings.append(b[::-1] + a[::-1])
        mat = np.array([[int(c) for c in bs] for bs in bitstrings], dtype=np.int8).T
        gf2x_result = gf2x_with_tracking(mat, skip_diagonal_reduction=True, forward_only=True)
        num_rows, num_cols = gf2x_result.reduced_matrix.shape
        dense_size = 1 if num_cols < 2 else math.ceil(math.log2(num_cols))
        assert dense_size >= num_rows, f"Expected fallback: dense_size={dense_size} must be >= num_rows={num_rows}"

        circuit = create("state_prep", "sparse_isometry_binary_encoding").run(wf)
        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"

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
        state_prep_circuit = create("state_prep", "sparse_isometry_binary_encoding").run(wf)
        sp_lc = state_prep_circuit.estimate()["logicalCounts"]
        num_system_qubits = 2 * 6  # 12 system qubits
        state_prep_total_qubits = sp_lc["numQubits"]
        extra_ancilla = state_prep_total_qubits - num_system_qubits
        assert extra_ancilla > 0, "This test requires a case with ancilla overflow"

        # Create a 12-qubit Hamiltonian matching the system size
        qubit_hamiltonian = QubitHamiltonian(
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

        state_prep_circuit = create("state_prep", "sparse_isometry_binary_encoding").run(wf)
        sp_lc = state_prep_circuit.estimate()["logicalCounts"]
        num_system_qubits = 12
        state_prep_total_qubits = sp_lc["numQubits"]
        extra_ancilla = state_prep_total_qubits - num_system_qubits
        assert extra_ancilla > 0, "This test requires a case with ancilla overflow"

        qubit_hamiltonian = QubitHamiltonian(
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
        state_prep_circuit = create("state_prep", "sparse_isometry_binary_encoding").run(wf)
        sp_lc = state_prep_circuit.estimate()["logicalCounts"]
        state_prep_qubits = sp_lc["numQubits"]
        extra_ancilla = state_prep_qubits - n
        assert extra_ancilla > 0, f"Expected ancilla overflow but state prep uses only {state_prep_qubits} qubits"

        # QubitHamiltonian
        qubit_hamiltonian = QubitHamiltonian(pauli_strings=pauli_strings, coefficients=coefficients_arr)

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

        # Resolve phase ambiguity and verify energy
        phase_candidates = [result.phase_fraction % 1.0, (1.0 - result.phase_fraction) % 1.0]
        energies = [energy_from_phase(p, evolution_time=evolution_time) for p in phase_candidates]
        resolved_energy = energies[int(np.argmin([abs(e - gs_energy) for e in energies]))]

        # Energy resolution with num_bits: 2*pi / (t * 2^num_bits)
        energy_resolution = 2 * np.pi / (evolution_time * 2**num_bits)
        assert abs(resolved_energy - gs_energy) < energy_resolution, (
            f"QPE energy {resolved_energy:.6f} deviates from ground energy {gs_energy:.6f} "
            f"by more than the {num_bits}-bit resolution ({energy_resolution:.4f}). "
            f"State prep used {state_prep_qubits} qubits ({extra_ancilla} ancilla)."
        )
