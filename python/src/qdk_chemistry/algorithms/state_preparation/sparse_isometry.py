"""Sparse isometry module for quantum state preparation.

This module implements sparse isometry algorithms for efficient quantum circuit
generation from electronic structure wavefunctions. Sparse isometry methods
leverage the sparsity of quantum states to create optimized circuits that
prepare only the non-zero amplitude components, significantly reducing circuit
depth and gate count compared to dense state preparation methods.

**SparseIsometryStatePrep**: Enhanced sparse isometry method.
This method performs duplicate row removal, all-ones row removal, and diagonal
matrix rank reduction besides standard GF2 Gaussian elimination. It tracks both
CNOT and X operations for optimal circuit reconstruction and can be more
efficient than standard GF2 for matrices with specific structural patterns.

The sparse isometry algorithms are particularly well-suited for quantum chemistry
applications where electronic structure wavefunctions often have a small number of
dominant determinants.

The implementations prepare the same quantum state with much more efficient
circuits, featuring significantly reduced gate counts and circuit depths
compared to traditional isometry methods.

Algorithm Details:

* SparseIsometry: Applies enhanced preprocessing, GF2 Gaussian elimination,
  and postprocessing, performs dense state preparation on the reduced space,
  then applies recorded operations (CX and X) in reverse to expand back to
  the full space.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass

import numpy as np

from qdk_chemistry.algorithms.state_preparation.state_preparation import StatePreparation, StatePreparationSettings
from qdk_chemistry.data import (
    AlgorithmRef,
    BasisSet,
    Circuit,
    Configuration,
    Orbitals,
    OrbitalType,
    Shell,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from ._binary_encoding_utils import MatrixCompressionOp, MatrixCompressionType, RefTableau, _BinaryEncodingSynthesizer

__all__: list[str] = ["SparseIsometryStatePreparationSettings"]


class SparseIsometryStatePreparationSettings(StatePreparationSettings):
    """Settings for SparseIsometryStatePreparation."""

    def __init__(self):
        """Initialize the StatePreparationSettings."""
        super().__init__()
        self._set_default(
            "dense_state_prep",
            "algorithm_ref",
            AlgorithmRef("state_prep", "dense_pure_state"),
            "State preparation algorithm used for the dense subspace.",
        )
        self._set_default(
            "binary_encoding",
            "bool",
            False,
            "Use binary encoding instead of dense state preparation for the reduced subspace.",
        )
        self._set_default(
            "include_negative_controls",
            "bool",
            True,
            "Include both positive and negative fixed controls in PUI construction.",
        )
        self._set_default(
            "measurement_based_uncompute",
            "bool",
            False,
            "Use measurement-based AND uncomputation in PUI blocks.",
        )


class SparseIsometryStatePreparation(StatePreparation):
    """State preparation using sparse isometry with enhanced GF2 Gaussian elimination.

    This class implements sparse isometry state preparation for electronic structure problems.
    The preprocessing includes:

        1. Removing duplicate rows using CX operations
        2. Removing all-ones rows using X operations
        3. Then performing standard GF2 Gaussian elimination
        4. Apply the additional rank reduction if the reduced row-echelon matrix is diagonal

    This enhanced approach can be more efficient than standard GF2 Gaussian elimination,
    particularly for matrices with duplicate rows or all-ones rows. The algorithm
    tracks both CX and X operations for proper circuit reconstruction.

    Key References:

        * Sparse isometry: Malvetti, Iten, and Colbeck (arXiv:2006.00016) :cite:`Malvetti2021`

    """

    def __init__(self) -> None:
        """Initialize the SparseIsometryStatePreparation."""
        Logger.trace_entering()
        super().__init__()
        self._settings = SparseIsometryStatePreparationSettings()

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit that encodes the given wavefunction using sparse isometry over GF(2^x).

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            A Circuit object containing the quantum circuit that prepares the desired state.

        """
        Logger.trace_entering()

        dets = wavefunction.get_active_determinants()
        coeffs = np.asarray(wavefunction.get_coefficients())
        config_set = wavefunction.get_configuration_set()
        n_bits = config_set.num_modes() * dets[0].bits_per_mode()
        state_vector = [det.to_bits(n_bits) for det in dets]
        n_qubits = len(state_vector[0])

        # Check for single determinant case after filtering
        if len(state_vector) == 1:
            Logger.info("After filtering, only 1 determinant remains, using single reference state preparation")
            return self._prepare_single_reference_state(state_vector[0])

        Logger.debug(f"Using {len(state_vector)} determinants for state preparation")

        if self._settings.get("binary_encoding"):
            circuit = self._run_binary_encoding(state_vector, coeffs, n_qubits)
            if circuit is not None:
                return circuit

        # Perform GF2 Gaussian elimination with tracking
        gf2x_operation_results, statevector_data = self._perform_gf2x(state_vector, coeffs)
        Logger.debug(f"gf2x_operation_results dense qubit: {gf2x_operation_results.row_map}")
        Logger.debug(f"gf2x_operation_results state vector: {statevector_data}")

        # Build reduced wavefunction and delegate dense preparation to nested algorithm
        reduced_wf = self._create_reduced_wavefunction(statevector_data, gf2x_operation_results.rank)
        dense_algo = self._create_nested("dense_state_prep")
        dense_circuit = dense_algo.run(reduced_wf)

        # Build expansion ops and compose with dense circuit
        expansion_ops = self._build_expansion_ops(gf2x_operation_results)
        return self._compose_with_expansion(
            dense_circuit, expansion_ops, gf2x_operation_results.row_map, n_qubits, dense_algo
        )

    def _run_binary_encoding(self, state_vector: list[list[int]], coeffs: np.ndarray, n_qubits: int) -> Circuit | None:
        """Prepare a quantum circuit using binary encoding.

        Args:
            state_vector: List of bit vectors (determinants), each a list of 0/1 ints.
            coeffs: Wavefunction coefficients aligned with the determinants.
            n_qubits: Total number of qubits in the system.

        Returns:
            A Circuit if binary encoding is applicable, or None to fall back to the standard path.

        """
        bitstring_matrix = self._bitstrings_to_binary_matrix(state_vector)
        gf2x_result = gf2x_with_tracking(bitstring_matrix, skip_diagonal_reduction=True, forward_only=True)

        num_rows, num_cols = gf2x_result.reduced_matrix.shape
        dense_register_width = 1 if num_cols < 2 else math.ceil(math.log2(num_cols))
        if not dense_register_width < num_rows:
            Logger.info(
                "Binary encoding is not applicable for this wavefunction; falling back to standard sparse isometry."
            )
            return None

        synthesis = self._synthesize_binary_encoding(gf2x_result, coeffs, n_qubits, len(state_vector))

        # Build reduced wavefunction and delegate dense prep to nested algorithm
        reduced_wf = self._create_reduced_wavefunction(synthesis["compressed_sv"], synthesis["dense_size"])
        dense_algo = self._create_nested("dense_state_prep")
        dense_circuit = dense_algo.run(reduced_wf)

        # Compose with binary encoding + Gaussian elimination expansion.
        # Reverse dense_row_map because DensePureStatePreparation internally reverses
        # its rowMap; the reversed embedding cancels out that reversal so the net effect
        # matches the original ApplyDensePreparation(dense_row_map, sv, qs) behavior.
        return self._compose_binary_encoding(
            dense_circuit,
            list(reversed(synthesis["dense_row_map"])),
            synthesis["binary_encoding_ops"],
            synthesis["gaussian_elimination_ops"],
            synthesis["ancilla_pool"],
            n_qubits,
            dense_algo,
        )

    def _synthesize_binary_encoding(
        self,
        gf2x_result: "GF2XEliminationResult",
        coeffs: np.ndarray,
        n_qubits: int,
        n_dets: int,
    ) -> dict:
        """Build binary-encoding state preparation parameters from an already-computed REF result.

        Args:
            gf2x_result: Forward-only REF result from GF2 Gaussian elimination.
            coeffs: Wavefunction coefficients aligned with matrix columns.
            n_qubits: Total number of qubits in the original space.
            n_dets: Number of determinants (used for logging only).

        Returns:
            A dict with synthesis results for composing the binary-encoding circuit.

        """
        include_negative_controls = self._settings.get("include_negative_controls")
        encoded_ops, bijection, dense_size = _BinaryEncodingSynthesizer(
            RefTableau(gf2x_result.reduced_matrix),
            include_negative_controls=include_negative_controls,
            measurement_based_uncompute=self._settings.get("measurement_based_uncompute"),
        ).synthesize(
            num_local_qubits=n_qubits,
            active_qubit_indices=gf2x_result.row_map,
            ancilla_start=n_qubits,
        )

        compressed_sv = np.zeros(2**dense_size, dtype=float)
        for dense_val, orig_col in bijection:
            if orig_col < len(coeffs):
                compressed_sv[dense_val] = coeffs[orig_col]
        norm = np.linalg.norm(compressed_sv)
        if norm > 0:
            compressed_sv /= norm

        dense_row_map = list(gf2x_result.row_map[:dense_size])

        gaussian_elimination_ops: list[MatrixCompressionOp] = []
        for operation in reversed(gf2x_result.operations):
            if operation[0] in ("cx", "cnot"):
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    gaussian_elimination_ops.append(MatrixCompressionOp(MatrixCompressionType("CX"), [control, target]))
            elif operation[0] == "x" and isinstance(operation[1], int):
                gaussian_elimination_ops.append(MatrixCompressionOp(MatrixCompressionType("X"), [operation[1]]))

        active_qubits_set = {int(q) for q in gf2x_result.row_map}
        ancilla_pool = sorted(set(range(n_qubits)) - active_qubits_set)

        Logger.info(
            f"Binary encoding produced {len(encoded_ops)} operations "
            f"for {n_qubits}-qubit system with {n_dets} determinants "
            f"using {len(ancilla_pool)} pre-existing qubits as ancilla pool"
        )
        return {
            "compressed_sv": compressed_sv,
            "dense_size": dense_size,
            "dense_row_map": dense_row_map,
            "binary_encoding_ops": list(encoded_ops),
            "gaussian_elimination_ops": gaussian_elimination_ops,
            "ancilla_pool": ancilla_pool,
        }

    def _create_reduced_wavefunction(self, statevector_data: np.ndarray, rank: int) -> Wavefunction:
        """Construct a reduced Wavefunction from the statevector.

        Creates a synthetic Wavefunction using 1-bit-per-mode configurations
        that represents the dense subspace after GF2 Gaussian elimination.

        Args:
            statevector_data: Amplitude vector of length 2^rank (normalized).
            rank: Number of qubits in the reduced space.

        Returns:
            A Wavefunction suitable for passing to a dense state preparation algorithm.

        """
        configs = []
        coeffs_list = []
        for idx in range(len(statevector_data)):
            if statevector_data[idx] != 0.0:
                bits_str = "".join(str((idx >> i) & 1) for i in range(rank))
                configs.append(Configuration.from_bitstring(bits_str))
                coeffs_list.append(statevector_data[idx])

        coeffs_arr = np.array(coeffs_list, dtype=float)
        shells = [Shell(0, OrbitalType.S, np.array([1.0]), np.array([1.0])) for _ in range(rank)]
        basis_set = BasisSet("reduced", shells)
        orbitals = Orbitals(np.eye(rank), None, None, basis_set)
        return Wavefunction(StateVectorContainer(coeffs_arr, configs, orbitals))

    def _build_expansion_ops(self, gf2x_operation_results: "GF2XEliminationResult") -> list[MatrixCompressionOp]:
        """Build expansion operations from GF2 Gaussian elimination results.

        Args:
            gf2x_operation_results: The result of GF2 Gaussian elimination.

        Returns:
            List of MatrixCompressionOp representing the expansion gates.

        """
        expansion_ops: list[MatrixCompressionOp] = []
        for operation in reversed(gf2x_operation_results.operations):
            if operation[0] == "cx":
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    expansion_ops.append(MatrixCompressionOp(MatrixCompressionType("CX"), [control, target]))
            elif operation[0] == "x" and isinstance(operation[1], int):
                expansion_ops.append(MatrixCompressionOp(MatrixCompressionType("X"), [operation[1]]))
        return expansion_ops

    def _compose_with_expansion(
        self,
        dense_circuit: Circuit,
        expansion_ops: list[MatrixCompressionOp],
        embedding_map: list[int],
        n_qubits: int,
        dense_algo: "StatePreparation",
    ) -> Circuit:
        """Compose a dense preparation circuit with expansion operations.

        Embeds the dense circuit (operating on the reduced qubit subset) into
        the full register, then applies GF2 Gaussian elimination expansion operations.

        Args:
            dense_circuit: Circuit from the nested dense state prep algorithm.
            expansion_ops: GF2 Gaussian elimination expansion operations for the full register.
            embedding_map: Maps reduced qubit indices to full register positions.
            n_qubits: Total number of qubits in the full register.
            dense_algo: The nested dense state prep algorithm instance (for transpile settings).

        Returns:
            Composed Circuit operating on the full qubit register.

        """
        if dense_circuit._qsharp_op is not None:  # noqa: SLF001
            return self._compose_qsharp(dense_circuit, expansion_ops, embedding_map, n_qubits)
        return self._compose_qiskit(dense_circuit, expansion_ops, embedding_map, n_qubits, dense_algo)

    def _compose_qsharp(
        self,
        dense_circuit: Circuit,
        expansion_ops: list[MatrixCompressionOp],
        embedding_map: list[int],
        n_qubits: int,
    ) -> Circuit:
        """Compose via Q# — embed dense op on subregister, then apply expansion."""
        serialized_ops = [op.to_dict() for op in expansion_ops]
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeComposeSparseIsometryOp(
            dense_circuit._qsharp_op,  # noqa: SLF001
            embedding_map,
            serialized_ops,
        )
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeComposeSparseIsometryCircuit,
            parameter={
                "denseOp": dense_circuit._qsharp_op,  # noqa: SLF001
                "embeddingMap": embedding_map,
                "expansionOps": serialized_ops,
                "numQubits": n_qubits,
            },
        )
        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op, encoding="jordan-wigner")

    def _compose_qiskit(
        self,
        dense_circuit: Circuit,
        expansion_ops: list[MatrixCompressionOp],
        embedding_map: list[int],
        n_qubits: int,
        dense_algo: "StatePreparation",
    ) -> Circuit:
        """Compose via Qiskit — embed dense circuit on subregister, then apply expansion gates."""
        from qiskit import QuantumCircuit, qasm3  # noqa: PLC0415
        from qiskit.compiler import transpile  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import apply_matrix_compression_ops  # noqa: PLC0415

        dense_qc = dense_circuit.get_qiskit_circuit()
        full_qc = QuantumCircuit(n_qubits)
        full_qc.compose(dense_qc, qubits=embedding_map, inplace=True)
        apply_matrix_compression_ops(full_qc, expansion_ops)

        # Transpile using the dense prep algorithm's settings to decompose
        if dense_algo.settings().get("transpile"):
            basis_gates = dense_algo.settings().get("basis_gates")
            opt_level = dense_algo.settings().get("transpile_optimization_level")
            full_qc = transpile(full_qc, basis_gates=basis_gates, optimization_level=opt_level)

        return Circuit(qasm=qasm3.dumps(full_qc), encoding="jordan-wigner")

    def _compose_binary_encoding(
        self,
        dense_circuit: Circuit,
        dense_row_map: list[int],
        binary_encoding_ops: list[MatrixCompressionOp],
        gaussian_elimination_ops: list[MatrixCompressionOp],
        ancilla_pool: list[int],
        n_qubits: int,
        dense_algo: "StatePreparation",
    ) -> Circuit:
        """Compose a dense circuit with binary-encoding and GF2 Gaussian elimination expansion operations.

        Args:
            dense_circuit: Circuit from the nested dense state prep algorithm.
            dense_row_map: Maps reduced qubit indices to full register positions.
            binary_encoding_ops: Binary-encoding gate sequence.
            gaussian_elimination_ops: GF2 Gaussian elimination expansion operations.
            ancilla_pool: Idle qubit indices available as ancillas.
            n_qubits: Total number of qubits in the full register.
            dense_algo: The nested dense state prep algorithm instance (for transpile settings).

        Returns:
            Composed Circuit operating on the full qubit register.

        """
        if dense_circuit._qsharp_op is not None:  # noqa: SLF001
            serialized_be = [op.to_qsharp_parameter() for op in binary_encoding_ops]
            serialized_ge = [op.to_qsharp_parameter() for op in gaussian_elimination_ops]
            qsharp_op = QSHARP_UTILS.BinaryEncoding.MakeComposeBinaryEncodingOp(
                dense_circuit._qsharp_op,  # noqa: SLF001
                dense_row_map,
                serialized_be,
                serialized_ge,
                ancilla_pool,
            )
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.BinaryEncoding.MakeComposeBinaryEncodingCircuit,
                parameter={
                    "denseOp": dense_circuit._qsharp_op,  # noqa: SLF001
                    "embeddingMap": dense_row_map,
                    "binaryEncodingOps": serialized_be,
                    "gaussianEliminationOps": serialized_ge,
                    "numQubits": n_qubits,
                    "ancillaPool": ancilla_pool,
                },
            )
            return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op, encoding="jordan-wigner")

        # Qiskit path
        from qiskit import QuantumCircuit, qasm3  # noqa: PLC0415
        from qiskit.compiler import transpile  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit.conversion import apply_matrix_compression_ops  # noqa: PLC0415

        dense_qc = dense_circuit.get_qiskit_circuit()
        full_qc = QuantumCircuit(n_qubits)
        full_qc.compose(dense_qc, qubits=dense_row_map, inplace=True)
        apply_matrix_compression_ops(full_qc, binary_encoding_ops)
        apply_matrix_compression_ops(full_qc, gaussian_elimination_ops)

        # Transpile using the dense prep algorithm's settings to decompose
        # multi-controlled gates (e.g. MCX) whose QASM3 export has a Qiskit bug.
        if dense_algo.settings().get("transpile"):
            basis_gates = dense_algo.settings().get("basis_gates")
            opt_level = dense_algo.settings().get("transpile_optimization_level")
            full_qc = transpile(full_qc, basis_gates=basis_gates, optimization_level=opt_level)

        return Circuit(qasm=qasm3.dumps(full_qc), encoding="jordan-wigner")

    def _perform_gf2x(
        self, bitstrings: list[list[int]], coeffs: np.ndarray
    ) -> tuple["GF2XEliminationResult", np.ndarray]:
        """Perform Gaussian elimination over GF(2^x) on the given bitstrings.

        Args:
            bitstrings: The list of bit vectors (0/1 ints) representing the wavefunction.
            coeffs: The coefficients corresponding to each determinant.

        Returns:
            A tuple containing the GF2X elimination result and the statevector.

        """
        Logger.trace_entering()
        Logger.debug(f"Using {len(bitstrings)} determinants for state preparation")

        # Step 1: Convert bitstrings to binary matrix
        bitstring_matrix = self._bitstrings_to_binary_matrix(bitstrings)

        # Step 2: Apply enhanced GF2 Gaussian elimination with tracking of operations
        # (includes duplicate removal, all-ones removal, and GF2)
        gf2x_operation_results = gf2x_with_tracking(bitstring_matrix)

        Logger.debug(f"Original matrix shape: {bitstring_matrix.shape}")
        Logger.debug(f"Reduced matrix shape: {gf2x_operation_results.reduced_matrix.shape}")
        Logger.debug(f"Matrix rank: {gf2x_operation_results.rank}")
        Logger.debug(f"Total operations: {len(gf2x_operation_results.operations)}")

        # Log operations by type
        Logger.debug(f"CX operations: {[op for op in gf2x_operation_results.operations if op[0] == 'cx']}")
        Logger.debug(f"X operations: {[op for op in gf2x_operation_results.operations if op[0] == 'x']}")

        # Step 3: Create statevector for the reduced matrix
        if gf2x_operation_results.rank > 0:
            # Create statevector correctly preserving coefficient-determinant correspondence.
            # Each coefficient corresponds to a specific determinant (column in reduced matrix).
            # We need to map each coefficient to the correct basis state in the reduced space.

            statevector_data = np.zeros(2**gf2x_operation_results.rank, dtype=float)

            # For each determinant (column in reduced matrix), map it to the correct statevector index
            for det_idx in range(gf2x_operation_results.reduced_matrix.shape[1]):
                # Get the reduced column for this determinant
                reduced_column = gf2x_operation_results.reduced_matrix[:, det_idx]

                # Convert reduced column to binary string (reverse for little-endian)
                bitstring = "".join(str(bit) for bit in reversed(reduced_column))

                # Calculate the statevector index for this bitstring
                statevector_index = int(bitstring, 2)

                # Assign the coefficient to the correct statevector index
                statevector_data[statevector_index] = coeffs[det_idx]

                Logger.debug(
                    f"Determinant {det_idx}: coeff={coeffs[det_idx]:.6f}, "
                    f"reduced_column={reduced_column.tolist()}, "
                    f"bitstring='{bitstring}', sv_index={statevector_index}"
                )

            # Normalize the statevector
            norm = np.linalg.norm(statevector_data)
            if norm > 0:
                statevector_data /= norm

            Logger.debug(f"Statevector created for reduced matrix with rank {gf2x_operation_results.rank}")
            Logger.debug(f"Statevector shape: {len(statevector_data)}")
            Logger.debug("Non-zero elements in statevector:")
            for i, amp in enumerate(statevector_data):
                bitstring_repr = format(i, f"0{gf2x_operation_results.rank}b")
                Logger.debug(f"  |{bitstring_repr}⟩: {amp:.6f}")

            Logger.debug(f"Target indices are {gf2x_operation_results.row_map}")
        else:
            # If reduced matrix has zero rank, all determinants are identical
            raise ValueError(
                "Cannot perform sparse isometry on identical determinants. All determinants must be distinct. "
                "Please check your wavefunction data - you may have duplicate determinants or "
                "need to use a single-determinant state preparation method."
            )

        return gf2x_operation_results, statevector_data

    def _bitstrings_to_binary_matrix(self, bitstrings: list[list[int]]) -> np.ndarray:
        """Convert a list of bit vectors to a binary matrix.

        This function converts a list of bit vectors (determinants) into a binary matrix
        where each column represents a determinant and each row represents a qubit.

        Args:
            bitstrings (list[list[int]]): List of bit vectors, each a list of 0/1 ints
                in qubit order q[0]...q[N-1] (as returned by Configuration.to_bits()).

        Returns:
            Binary matrix M of shape (N, k) where

                * N is the number of qubits (rows)
                * k is the number of determinants (columns)

            The matrix follows top-down convention with row ordering "q[0]...q[N-1]"
            (qubit 0 at the top).

        Example:
            >>> bitstrings = [[1, 0, 1], [0, 1, 0]]
            >>> matrix = _bitstrings_to_binary_matrix(bitstrings)
            >>> print(matrix)
            [[1 0]  # q[0]
            [0 1]  # q[1]
            [1 0]] # q[2]

        """
        if not bitstrings:
            raise ValueError("Bitstrings list cannot be empty")

        n_qubits = len(bitstrings[0])
        n_dets = len(bitstrings)

        # Validate all bit vectors have the same length
        for i, bitstring in enumerate(bitstrings):
            if len(bitstring) != n_qubits:
                raise ValueError(
                    f"All bit vectors must have the same length. "
                    f"Bit vector {i} has length {len(bitstring)}, expected {n_qubits}"
                )

        # Create binary matrix: each bit vector is already in q[0]...q[N-1] order
        bitstring_matrix = np.zeros((n_qubits, n_dets), dtype=np.int8)
        for i, bitstring in enumerate(bitstrings):
            bitstring_matrix[:, i] = np.array(bitstring, dtype=np.int8)

        return bitstring_matrix

    def _prepare_single_reference_state(self, bitstring: list[int]) -> Circuit:
        r"""Prepare a single reference state on a quantum circuit based on a bitstring.

        The input bitstring is in big-endian order: ``bitstring[0]`` corresponds to
        the highest-indexed qubit (MSB) and ``bitstring[N-1]`` corresponds to qubit 0
        (LSB). The function reverses the bitstring internally so that in the circuit,
        ``bitstring[i]`` maps to qubit ``N-1-i``.

        Args:
            bitstring: List of 0/1 ints in big-endian order (MSB first).
                1 means apply X gate, 0 means leave in |0⟩ state.

        Returns:
                A Circuit object with the prepared single reference state

        Example:
                bitstring = [1, 0, 1, 0] produces X gates on qubit 0 and qubit 2:

                * qubit 0 ← bitstring[0] = 1 → X → :math:`\left| 1 \right\rangle`
                * qubit 1 ← bitstring[1] = 0 → :math:`\left| 0 \right\rangle`
                * qubit 2 ← bitstring[2] = 1 → X → :math:`\left| 1 \right\rangle`
                * qubit 3 ← bitstring[3] = 0 → :math:`\left| 0 \right\rangle`

        """
        # Input validation
        if not bitstring:
            raise ValueError("Bitstring cannot be empty")

        if not all(bit in (0, 1) for bit in bitstring):
            raise ValueError("Bitstring must contain only 0 and 1 values")

        num_qubits = len(bitstring)
        params = QSHARP_UTILS.StatePreparation.SingleReferenceParams(bitStrings=bitstring, numQubits=num_qubits)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeSingleReferenceStateCircuit, parameter=vars(params)
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakePrepareSingleReferenceStateOp(params)

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            encoding="jordan-wigner",
        )

    def name(self) -> str:
        """Return the name of the state preparation method."""
        Logger.trace_entering()
        return "sparse_isometry"


@dataclass
class GF2XEliminationResult:
    """Data class to hold the results of GF2+X elimination."""

    reduced_matrix: np.ndarray
    """Reduced row-echelon binary matrix with zero rows removed."""

    row_map: list[int]
    """Map of reduced matrix row i to original row index."""

    col_map: list[int]
    """Map of reduced matrix col j to original column index."""

    operations: list[tuple[str, int | tuple[int, int]]]
    """List of operations in the form:

        * ('cx', (target_row, control_row)) for CX operations
        * ('x', row_index) for X operations on entire rows

    All indices refer to original matrix positions.
    """

    rank: int
    """Rank of the reduced matrix (number of non-zero rows)."""


def gf2x_with_tracking(
    matrix: np.ndarray,
    *,
    skip_diagonal_reduction: bool = False,
    forward_only: bool = False,
) -> GF2XEliminationResult:
    """Perform enhanced GF2+X Gaussian elimination with smart preprocessing and X operations.

    This function implements a smarter approach to GF2 Gaussian elimination by:

        1. First removing duplicate rows using CX operations
        2. Removing all-ones rows using X operations
        3. Then performing standard Gaussian elimination
        4. Performing further reduction if the resulting matrix is diagonal

    This approach can be more efficient than standard Gaussian elimination alone,
    especially for certain types of matrices.

    Args:
        matrix: shape (m, n), binary (0/1) matrix
        skip_diagonal_reduction: If True, skip the optional diagonal-to-upper-
            staircase rank reduction (step 4).  Binary encoding handles the
            identity pivot block natively, so the extra CX + X expansion ops
            produced by the diagonal reduction are redundant Cliffords.
        forward_only: If True, perform forward-only GF2 elimination into row echelon form (REF),
            skipping back-substitution entirely.

    Returns:
        A dataclass containing GF2+X elimination results.

    """
    Logger.trace_entering()
    n_rows, n_cols = matrix.shape
    row_map = list(range(n_rows))
    col_map = list(range(n_cols))
    operations: list[tuple[str, int | tuple[int, int]]] = []

    # Handle empty matrix case early
    if n_rows == 0:
        raise ValueError("Input matrix has no rows (no qubits). Please check your input data.")
    if n_cols == 0:
        raise ValueError("Input matrix has no columns (no determinants). Please check your input data.")

    # Log the original matrix rank
    original_rank = np.linalg.matrix_rank(matrix)
    Logger.info(f"Original matrix rank: {original_rank}")

    # Check for zero rank matrix (all zero rows)
    if original_rank == 0:
        raise ValueError(
            "Input matrix has rank 0 (all rows are zero). This indicates no valid quantum states. "
            "Please check your wavefunction data - you may have invalid determinants or coefficients."
        )

    # Work on a copy to avoid modifying the input
    matrix_work = matrix.copy()

    # Step 1: Remove duplicate rows using CX operations
    matrix_work, row_map, operations = _remove_duplicate_rows_with_cnot(matrix_work, row_map, operations)

    # Step 2: Remove all-ones rows using X operations
    matrix_work, row_map, operations = _remove_all_ones_rows_with_x(matrix_work, row_map, operations)

    # Step 3: Perform standard Gaussian elimination on the remaining matrix
    if matrix_work.shape[0] > 0:  # Only if there are rows left
        matrix_processed, updated_row_map, new_cnot_ops = _perform_gaussian_elimination(
            matrix_work, row_map, [], forward_only=forward_only
        )

        for target, control in new_cnot_ops:
            operations.append(("cx", (target, control)))

        # Remove zero rows and update row_map accordingly
        matrix_reduced, reduced_row_map, rank = _remove_zero_rows(matrix_processed, updated_row_map)

        gf2x_results = GF2XEliminationResult(
            reduced_matrix=matrix_reduced,
            row_map=reduced_row_map,
            col_map=col_map,
            operations=operations,
            rank=rank,
        )

        # Step 4: Check for diagonal matrix and apply further reduction if possible
        if not forward_only and not skip_diagonal_reduction and rank > 1 and _is_diagonal_matrix(matrix_reduced):
            Logger.info(f"Detected diagonal matrix with rank {rank}, applying further reduction")
            gf2x_results = _reduce_diagonal_matrix(matrix_reduced, reduced_row_map, col_map, operations)

        # Log the final reduced matrix rank
        Logger.info(f"Final reduced matrix rank: {gf2x_results.rank}")

        return gf2x_results

    # If no rows left after preprocessing, return empty matrix
    Logger.info("Final reduced matrix rank: 0")
    return GF2XEliminationResult(
        reduced_matrix=np.empty((0, n_cols), dtype=matrix.dtype),
        row_map=row_map,
        col_map=col_map,
        operations=operations,
        rank=0,
    )


def _remove_duplicate_rows_with_cnot(
    matrix: np.ndarray,
    row_map: list[int],
    operations: list[tuple[str, int | tuple[int, int]]],
) -> tuple[np.ndarray, list[int], list[tuple[str, int | tuple[int, int]]]]:
    """Remove duplicate rows using CNOT operations.

    This function identifies duplicate rows and eliminates them by applying CNOT operations.
    When two rows are identical, a CNOT operation from one to the other will make the target row all zeros.

    Args:
        matrix: Binary matrix to process
        row_map: Current row mapping to original indices
        operations: List to append operations to

    Returns:
        A tuple containing ``(updated_matrix, updated_row_map, updated_operations)``.

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    operations_work = operations.copy()

    n_rows, _ = matrix_work.shape
    rows_to_eliminate: set[int] = set()

    # Find duplicate rows and XOR them to zero immediately
    for i in range(n_rows):
        if i in rows_to_eliminate:
            continue

        if not np.any(matrix_work[i]):
            continue

        for j in range(i + 1, n_rows):
            if j in rows_to_eliminate:
                continue

            if np.array_equal(matrix_work[i], matrix_work[j]):
                operations_work.append(("cx", (row_map_work[j], row_map_work[i])))
                matrix_work[j] ^= matrix_work[i]
                rows_to_eliminate.add(j)

                Logger.info(
                    f"Found duplicate row {j} identical to row {i}, adding CX({row_map_work[i]}, {row_map_work[j]})"
                )

    # Remove eliminated rows (now all zeros)
    if rows_to_eliminate:
        Logger.info(f"Eliminating {len(rows_to_eliminate)} duplicate rows: {sorted(rows_to_eliminate)}")

        rows_to_keep = [i for i in range(n_rows) if i not in rows_to_eliminate]
        matrix_work = matrix_work[rows_to_keep]
        row_map_work = [row_map_work[i] for i in rows_to_keep]

    return matrix_work, row_map_work, operations_work


def _remove_all_ones_rows_with_x(
    matrix: np.ndarray,
    row_map: list[int],
    operations: list[tuple[str, int | tuple[int, int]]],
) -> tuple[np.ndarray, list[int], list[tuple[str, int | tuple[int, int]]]]:
    """Remove all-ones rows using X operations.

    This function identifies rows that contain all ones and eliminates them
    by applying X operations to flip all bits in those rows to zeros.

    Args:
        matrix: Binary matrix to process
        row_map: Current row mapping to original indices
        operations: List to append operations to

    Returns:
        A tuple containing ``(updated_matrix, updated_row_map, updated_operations)``

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    operations_work = operations.copy()

    n_rows, n_cols = matrix_work.shape
    rows_to_eliminate = []

    # Find all-ones rows
    for i in range(n_rows):
        if np.all(matrix_work[i] == 1):
            # Apply X operation to flip all bits to zero
            operations_work.append(("x", row_map_work[i]))
            rows_to_eliminate.append(i)

            Logger.info(f"Found all-ones row {i}, adding X operation on row {row_map_work[i]}")

    # Apply X operations to eliminate all-ones rows
    for i in rows_to_eliminate:
        matrix_work[i] = np.zeros(n_cols, dtype=matrix_work.dtype)
    # Remove eliminated rows (which are now all zeros)
    if rows_to_eliminate:
        Logger.info(f"Eliminating {len(rows_to_eliminate)} all-ones rows: {rows_to_eliminate}")

        # Create mask for rows to keep
        rows_to_keep = [i for i in range(n_rows) if i not in rows_to_eliminate]

        # Update matrix and row mapping
        matrix_work = matrix_work[rows_to_keep]
        row_map_work = [row_map_work[i] for i in rows_to_keep]

    return matrix_work, row_map_work, operations_work


def _perform_gaussian_elimination(
    matrix: np.ndarray,
    row_map: list[int],
    cnot_ops: list[tuple[int, int]],
    *,
    forward_only: bool = False,
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """Perform GF2 Gaussian elimination.

    Args:
        matrix: Binary matrix to reduce (copied internally).
        row_map: Current-to-original row index mapping (copied internally).
        cnot_ops: Existing CNOT operation list (copied internally).
        forward_only: If True, perform forward-only elimination into
            row echelon form (REF), skipping back-substitution.  If
            False (default), perform full elimination into RREF.

    Returns:
        ``(reduced_matrix, updated_row_map, updated_cnot_ops)``

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    cnot_ops_work = cnot_ops.copy()
    num_rows, num_cols = matrix_work.shape

    pivot_row = 0
    for col in range(num_cols):
        sel = _find_pivot_row(matrix_work, pivot_row, col)
        if sel is None:
            continue

        if sel != pivot_row:
            matrix_work[[pivot_row, sel]] = matrix_work[[sel, pivot_row]]
            row_map_work[pivot_row], row_map_work[sel] = row_map_work[sel], row_map_work[pivot_row]

        _eliminate_column(matrix_work, pivot_row, col, row_map_work, cnot_ops_work, forward_only=forward_only)

        pivot_row += 1
        if pivot_row == num_rows:
            break

    return matrix_work, row_map_work, cnot_ops_work


def _find_pivot_row(matrix: np.ndarray, start_row: int, col: int) -> int | None:
    """Find the first row at or below ``start_row`` with a 1 in ``col``.

    Args:
        matrix: Binary matrix (read-only).
        start_row: First row index to consider (inclusive).
        col: Column to search.

    Returns:
        Row index of the first 1-entry, or ``None`` if the column is
        all-zero from ``start_row`` downward.

    """
    candidates = np.flatnonzero(matrix[start_row:, col])
    return start_row + int(candidates[0]) if candidates.size > 0 else None


def _eliminate_column(
    matrix: np.ndarray,
    pivot_row: int,
    col: int,
    row_map: list[int],
    cnot_ops: list[tuple[int, int]],
    *,
    forward_only: bool = False,
) -> None:
    """Eliminate rows in ``col`` using XOR with the pivot row.

    Modifies ``matrix`` and ``cnot_ops`` **in place**.

    Args:
        matrix: Binary matrix (modified in place).
        pivot_row: Index of the pivot row (unchanged).
        col: Column to eliminate.
        row_map: Current-to-original row index mapping (read-only).
        cnot_ops: Destination list for recorded CNOT operations.
        forward_only: If True, only eliminate rows below the pivot
            (forward elimination / REF).  If False, eliminate all
            other rows (full back-substitution / RREF).

    """
    if forward_only:
        targets = np.flatnonzero(matrix[pivot_row + 1 :, col]) + pivot_row + 1
    else:
        targets = np.flatnonzero(matrix[:, col])
        targets = targets[targets != pivot_row]
    for r in targets:
        matrix[r] ^= matrix[pivot_row]
        cnot_ops.append((row_map[r], row_map[pivot_row]))


def _remove_zero_rows(matrix: np.ndarray, row_map: list[int]) -> tuple[np.ndarray, list[int], int]:
    """Remove all-zero rows from the matrix and update the row mapping.

    Args:
        matrix: Binary matrix (read-only).
        row_map: Current-to-original row index mapping (read-only).

    Returns:
        ``(matrix_reduced, reduced_row_map, rank)`` where ``rank`` is the
        number of retained (non-zero) rows.

    """
    non_zero_indices = np.flatnonzero(np.any(matrix, axis=1))
    return (
        matrix[non_zero_indices],
        [row_map[i] for i in non_zero_indices],
        int(non_zero_indices.size),
    )


def _reduce_diagonal_matrix(
    matrix: np.ndarray,
    row_map: list[int],
    col_map: list[int],
    operations: list[tuple[str, int | tuple[int, int]]],
) -> GF2XEliminationResult:
    """Reduce a diagonal (identity) matrix by one rank via CX cascade + X.

    Applies CX(i, i+1) for i = 0…rank-2, making the last row all-ones,
    then X on the last row to zero it, and finally removes that row.

    The caller is responsible for verifying ``_is_diagonal_matrix`` first.

    Args:
        matrix: Diagonal binary matrix to reduce.
        row_map: Current row mapping to original indices.
        col_map: Column mapping (passed through unchanged).
        operations: Operations list to extend.

    Returns:
        GF2XEliminationResult with rank decremented by 1.

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    operations_work = operations.copy()
    rank = matrix_work.shape[0]

    Logger.info(f"Applying diagonal matrix reduction on {rank}x{matrix_work.shape[1]} matrix")

    # Sequential CX(i, i+1) accumulates all 1s into the last row
    for i in range(rank - 1):
        operations_work.append(("cx", (row_map_work[i + 1], row_map_work[i])))
        matrix_work[i + 1] ^= matrix_work[i]

    # X on the all-ones last row zeroes it
    operations_work.append(("x", row_map_work[rank - 1]))

    Logger.info(f"Diagonal reduction complete: rank reduced from {rank} to {rank - 1}")

    return GF2XEliminationResult(
        reduced_matrix=matrix_work[:-1],
        row_map=row_map_work[:-1],
        col_map=col_map,
        operations=operations_work,
        rank=rank - 1,
    )


def _is_diagonal_matrix(matrix: np.ndarray) -> bool:
    """Check if a binary matrix is diagonal and safe for rank reduction.

    Two accepted shapes:

    1. **Square identity**: ``matrix == np.eye(r)``.
    2. **Pseudo-diagonal** (more columns than rows, odd row count):
       the leading ``r x r`` block is identity and every extra column
       is all-ones.

    Args:
        matrix: Binary matrix to check.

    Returns:
        ``True`` if the matrix matches one of the accepted shapes.

    """
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        return False

    num_rows, num_cols = matrix.shape
    identity = np.eye(num_rows, dtype=matrix.dtype)

    if num_rows == num_cols:
        return bool(np.array_equal(matrix, identity))

    return (
        num_cols > num_rows
        and num_rows % 2 == 1
        and bool(np.array_equal(matrix[:, :num_rows], identity))
        and bool(np.all(matrix[:, num_rows:] == 1))
    )
