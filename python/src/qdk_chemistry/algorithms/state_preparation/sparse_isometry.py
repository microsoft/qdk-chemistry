"""Sparse isometry module for quantum state preparation.

This module implements sparse isometry algorithms for efficient quantum circuit
generation from electronic structure wavefunctions. Sparse isometry methods
leverage the sparsity of quantum states to create optimized circuits that
prepare only the non-zero amplitude components, significantly reducing circuit
depth and gate count compared to dense state preparation methods.

**SparseIsometryGF2XStatePrep**: Enhanced sparse isometry using GF2+X elimination.
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

* SparseIsometryGF2X: Applies enhanced GF2+X elimination (preprocessing + GF2
  + postprocessing), performs dense state preparation on the reduced space,
  then applies recorded operations (CX and X) in reverse to expand back to
  the full space.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import qdk_chemistry.plugins.qiskit
from qdk_chemistry.algorithms.state_preparation.state_preparation import StatePreparation, StatePreparationSettings
from qdk_chemistry.data import Circuit, Wavefunction
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = []


class SparseIsometryGF2XStatePreparationSettings(StatePreparationSettings):
    """Settings for SparseIsometryGF2XStatePreparation."""

    def __init__(self):
        """Initialize the StatePreparationSettings."""
        super().__init__()
        self._set_default(
            "dense_preparation_method", "string", "qdk", "The dense state preparation method to use.", ["qdk", "qiskit"]
        )


class SparseIsometryGF2XStatePreparation(StatePreparation):
    """State preparation using sparse isometry with enhanced GF2+X elimination.

    This class implements "GF2+X" state preparation for electronic structure problems using
    the ``gf2x_with_tracking`` function which performs smart preprocessing
    before GF2 Gaussian elimination. The preprocessing includes:

        1. Removing duplicate rows using CX operations
        2. Removing all-ones rows using X operations
        3. Then performing standard GF2 Gaussian elimination
        4. Apply the additional rank reduction if the reduced row-echelon matrix is diagonal

    This enhanced approach can be more efficient than standard GF2 Gaussian elimination,
    particularly for matrices with duplicate rows or all-ones rows. The algorithm
    tracks both CX and X operations for proper circuit reconstruction.

    The algorithm:

        1. Reads the wavefunction to get coefficients and bitstrings
        2. Converts bitstrings to a binary matrix
        3. Applies enhanced GF2+X elimination (duplicate removal + all-ones removal + GF2)
        4. Performs dense state preparation on the reduced space
        5. Applies recorded operations (both CX and X) in reverse order to expand back to full space

    Key References:

        * Sparse isometry: Malvetti, Iten, and Colbeck (arXiv:2006.00016) :cite:`Malvetti2021`

    """

    def __init__(self) -> None:
        """Initialize the SparseIsometryGF2XStatePreparation."""
        Logger.trace_entering()
        super().__init__()
        self._settings = SparseIsometryGF2XStatePreparationSettings()

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit that encodes the given wavefunction using sparse isometry over GF(2^x).

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            A Circuit object containing the quantum circuit that prepares the desired state.

        """
        Logger.trace_entering()

        if (
            self._settings.get("dense_preparation_method") == "qiskit"
            and not qdk_chemistry.plugins.qiskit.QDK_CHEMISTRY_HAS_QISKIT
        ):
            raise ImportError(
                "Qiskit is not available. Please install Qiskit to use the 'qiskit' dense preparation method."
            )

        # Active Space Consistency Check
        alpha_indices, beta_indices = wavefunction.get_orbitals().get_active_space_indices()
        if alpha_indices != beta_indices:
            raise ValueError(
                f"Active space contains {len(alpha_indices)} alpha orbitals and "
                f"{len(beta_indices)} beta orbitals. Asymmetric active spaces for "
                "alpha and beta orbitals are not supported for state preparation."
            )

        coeffs = wavefunction.get_coefficients()
        dets = wavefunction.get_active_determinants()
        num_orbitals = len(wavefunction.get_orbitals().get_active_space_indices()[0])
        bitstrings = []
        for det in dets:
            alpha_str, beta_str = det.to_binary_strings(num_orbitals)
            bitstring = beta_str[::-1] + alpha_str[::-1]  # Qiskit uses little-endian convention
            bitstrings.append(bitstring)

        # Check for single determinant case after filtering
        if len(bitstrings) == 1:
            Logger.info("After filtering, only 1 determinant remains, using single reference state preparation")
            return self._prepare_single_reference_state(bitstrings[0])

        n_qubits = len(bitstrings[0])
        Logger.debug(f"Using {len(bitstrings)} determinants for state preparation")

        # Perform GF2+X elimination with tracking
        gf2x_operation_results, statevector_data = self._perform_gf2x(bitstrings, coeffs)
        Logger.debug(f"gf2x_operation_results dense qubit: {gf2x_operation_results.row_map}")
        Logger.debug(f"gf2x_operation_results state vector: {statevector_data}")

        if self._settings.get("dense_preparation_method") == "qiskit":
            return self._qiskit_dense_preparation(gf2x_operation_results, statevector_data, n_qubits)

        # Use QDK dense state preparation
        expansion_ops: list[MatrixCompressionOp] = []
        for operation in reversed(gf2x_operation_results.operations):
            if operation[0] == "cx":
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    expansion_ops.append(MatrixCompressionOp("CX", [control, target]))
            elif operation[0] == "x" and isinstance(operation[1], int):
                expansion_ops.append(MatrixCompressionOp("X", [operation[1]]))

        # State vector indexing is in little-endian order, the row map is reversed for Q# convention
        state_prep_params = QSHARP_UTILS.StatePreparation.StatePreparationParams(
            rowMap=gf2x_operation_results.row_map[::-1],
            stateVector=statevector_data.tolist(),
            expansionOps=[op.to_dict() for op in expansion_ops],
            numQubits=n_qubits,
        )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=vars(state_prep_params),
        )

        state_prep_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=state_prep_op, encoding="jordan-wigner")

    def _qiskit_dense_preparation(
        self, gf2x_operation_results: "GF2XEliminationResult", statevector_data: np.ndarray, num_qubits: int
    ) -> Circuit:
        """Perform dense state preparation using Qiskit and apply GF2+X operations in reverse.

        Args:
            gf2x_operation_results: The result of GF2+X elimination containing the reduced matrix and operations.
            statevector_data: The statevector corresponding to the reduced matrix.
            num_qubits: The total number of qubits in the original space.

        Returns:
            A Circuit object containing the quantum circuit that prepares the desired state using Qiskit
            for dense preparation.

        """
        from qiskit import QuantumCircuit, qasm3, transpile  # noqa: PLC0415
        from qiskit.circuit.library import (  # noqa: PLC0415
            StatePreparation as QiskitStatePreparation,
        )
        from qiskit.quantum_info import Statevector  # noqa: PLC0415
        from qiskit.transpiler import PassManager  # noqa: PLC0415

        from qdk_chemistry.plugins.qiskit._interop.transpiler import (  # noqa: PLC0415
            MergeZBasisRotations,
            RemoveZBasisOnZeroState,
            SubstituteCliffordRz,
        )

        # Use Qiskit dense state preparation
        qc = QuantumCircuit(num_qubits)
        statevector = Statevector(statevector_data)
        qc.append(QiskitStatePreparation(statevector, normalize=False), gf2x_operation_results.row_map)
        for operation in reversed(gf2x_operation_results.operations):
            if operation[0] == "cx":
                # operation[1] should be a tuple for CX operations
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    qc.cx(control, target)
            elif operation[0] == "x" and isinstance(operation[1], int):
                # operation[1] should be an int for X operations
                qubit = operation[1]
                qc.x(qubit)

        basis_gates = self._settings.get("basis_gates")
        do_transpile = self._settings.get("transpile")
        if do_transpile and basis_gates:
            opt_level = self._settings.get("transpile_optimization_level")
            qc = transpile(qc, basis_gates=basis_gates, optimization_level=opt_level)
            pass_manager = PassManager([MergeZBasisRotations(), SubstituteCliffordRz(), RemoveZBasisOnZeroState()])
            qc = pass_manager.run(qc)

            Logger.info(
                f"Final circuit after transpilation: {qc.num_qubits} qubits, depth {qc.depth()}, {qc.size()} gates"
            )
        return Circuit(qasm=qasm3.dumps(qc), encoding="jordan-wigner")

    def _perform_gf2x(self, bitstrings: list[str], coeffs: np.ndarray) -> tuple["GF2XEliminationResult", np.ndarray]:
        """Perform Gaussian elimination over GF(2^x) on the given bitstrings.

        Args:
            bitstrings: The list of bitstrings representing the wavefunction.
            coeffs: The coefficients corresponding to each determinant.

        Returns:
            A tuple containing the GF2X elimination result and the statevector.

        """
        Logger.trace_entering()
        Logger.debug(f"Using {len(bitstrings)} determinants for state preparation")

        # Step 1: Convert bitstrings to binary matrix
        bitstring_matrix = self._bitstrings_to_binary_matrix(bitstrings)

        # Step 2: Apply enhanced GF2+X
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

    def _bitstrings_to_binary_matrix(self, bitstrings: list[str]) -> np.ndarray:
        """Convert a list of bitstrings to a binary matrix.

        This function converts a list of bitstrings (determinants) into a binary matrix
        where each column represents a determinant and each row represents a qubit.

        Args:
            bitstrings (list[str]): List of bitstrings in little-endian order.
                Each bitstring represents a determinant where the string is ordered
                as "q[N-1]...q[0]" (most significant bit first in the string).

        Returns:
            Binary matrix M of shape (N, k) where

                * N is the number of qubits (rows)
                * k is the number of determinants (columns)

            The matrix follows top-down convention with row ordering "q[0]...q[N-1]"
            (qubit 0 at the top).

        Note:
            The input bitstrings are in little-endian order ("q[N-1]...q[0]"),
            but the output binary matrix follows the top-down convention with
            row ordering "q[0]...q[N-1]". This means each bitstring is reversed
            when converting to a column in the matrix.

        Example:
            >>> bitstrings = ["101", "010"]  # q[2]q[1]q[0] format
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

        # Validate all bitstrings have the same length
        for i, bitstring in enumerate(bitstrings):
            if len(bitstring) != n_qubits:
                raise ValueError(
                    f"All bitstrings must have the same length. "
                    f"Bitstring {i} has length {len(bitstring)}, expected {n_qubits}"
                )

        # Create binary matrix with correct row ordering (reverse each bitstring)
        bitstring_matrix = np.zeros((n_qubits, n_dets), dtype=np.int8)
        for i, bitstring in enumerate(bitstrings):
            # Reverse the bitstring to get correct qubit ordering
            # Input: "q[N-1]...q[0]" -> Output: column with q[0] at top
            reversed_bitstring = bitstring[::-1]
            bitstring_matrix[:, i] = np.array(list(map(int, reversed_bitstring)), dtype=np.int8)

        return bitstring_matrix

    def _prepare_single_reference_state(self, bitstring: str) -> Circuit:
        r"""Prepare a single reference state on a quantum circuit based on a bitstring.

        Args:
            bitstring: Binary string representing the occupation of qubits.

                '1' means apply X gate, '0' means leave in |0⟩ state.

        Returns:
                A Circuit object containing an OpenQASM3 string with the prepared single reference state

        Example:
                bitstring = "1010" creates a circuit with X gates on qubits 1 and 3:

                * :math:`\left| 0 \right\rangle \rightarrow I \rightarrow \left| 0 \right\rangle`
                (qubit 0, corresponds to rightmost bit '0')
                * :math:`\left| 0 \right\rangle \rightarrow X \rightarrow \left| 1 \right\rangle`
                (qubit 1, corresponds to bit '1')
                * :math:`\left| 0 \right\rangle \rightarrow I \rightarrow \left| 0 \right\rangle`
                (qubit 2, corresponds to bit '0')
                * :math:`\left| 0 \right\rangle \rightarrow X \rightarrow \left| 1 \right\rangle`
                (qubit 3, corresponds to leftmost bit '1')

        """
        # Input validation
        if not bitstring:
            raise ValueError("Bitstring cannot be empty")

        if not all(bit in "01" for bit in bitstring):
            raise ValueError("Bitstring must contain only '0' and '1' characters")

        bitstring_array = [int(bit) for bit in bitstring]
        num_qubits = len(bitstring_array)
        params = QSHARP_UTILS.StatePreparation.SingleReferenceParams(
            bitStrings=bitstring_array[::-1], numQubits=num_qubits
        )
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
        return "sparse_isometry_gf2x"


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


@dataclass
class MatrixCompressionOp:
    """A single gate in the compressed matrix-encoding circuit.

    Mirrors the Q# ``MatrixCompressionOp`` struct.  Use :meth:`to_dict` to
    produce a camelCase dict consumable by the Q# bridge.

    Attributes:
        name: Gate name (e.g. ``"CX"``, ``"CCX"``, ``"SELECT"``).
        qubits: Qubit indices involved in the operation.
        control_state: Integer encoding of the control state for multi-
            controlled gates.  For ``SELECT``/``SELECT_AND``, this stores
            the number of address qubits.
        lookup_data: Boolean lookup table for ``SELECT`` operations;
            empty list for all other gate types.

    """

    name: str
    qubits: list[int]
    control_state: int = 0
    lookup_data: list[list[bool]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase dict matching the Q# ``MatrixCompressionOp`` struct."""
        return {
            "name": self.name,
            "qubits": self.qubits,
            "controlState": self.control_state,
            "lookupData": self.lookup_data,
        }


def gf2x_with_tracking(
    matrix: np.ndarray,
    *,
    skip_diagonal_reduction: bool = False,
    staircase_mode: bool = False,
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
        staircase_mode: If True, perform forward-only GF2 elimination (REF)
            then convert the pivot block directly to upper-staircase form
            with minimal CX ops — skipping back-substitution entirely.
            This produces a matrix that binary encoding's diagonal path
            recognises as ``is_diagonal_reduced``, saving ``r-1`` CX gates
            vs. the full RREF + staircase cascade.  Implies
            ``skip_diagonal_reduction=True``.

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
        if staircase_mode:
            # Forward-only elimination → REF → direct staircase fill
            matrix_processed, updated_row_map, new_cnot_ops = _perform_gaussian_elimination_forward_only(
                matrix_work, row_map, []
            )

            for target, control in new_cnot_ops:
                operations.append(("cx", (target, control)))

            # Remove zero rows
            matrix_reduced, reduced_row_map, rank = _remove_zero_rows(matrix_processed, updated_row_map)

            if rank > 1:
                # Convert REF pivot block directly to staircase form
                matrix_reduced, reduced_row_map, operations = _ref_to_staircase(
                    matrix_reduced, reduced_row_map, operations
                )

            gf2x_results = GF2XEliminationResult(
                reduced_matrix=matrix_reduced,
                row_map=reduced_row_map,
                col_map=col_map,
                operations=operations,
                rank=rank,
            )
        else:
            matrix_processed, updated_row_map, new_cnot_ops = _perform_gaussian_elimination(matrix_work, row_map, [])

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
            if not skip_diagonal_reduction and rank > 1 and _is_diagonal_matrix(matrix_reduced):
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
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """Perform full GF2 Gaussian elimination (forward + back-substitution).

    Args:
        matrix: Binary matrix to reduce (copied internally).
        row_map: Current-to-original row index mapping (copied internally).
        cnot_ops: Existing CNOT operation list (copied internally).

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

        _eliminate_column(matrix_work, pivot_row, col, row_map_work, cnot_ops_work)

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
) -> None:
    """Eliminate all other rows in ``col`` using XOR with the pivot row.

    Modifies ``matrix`` and ``cnot_ops`` **in place**.

    Args:
        matrix: Binary matrix (modified in place).
        pivot_row: Index of the pivot row (unchanged).
        col: Column to eliminate.
        row_map: Current-to-original row index mapping (read-only).
        cnot_ops: Destination list for recorded CNOT operations.

    """
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


def _perform_gaussian_elimination_forward_only(
    matrix: np.ndarray,
    row_map: list[int],
    cnot_ops: list[tuple[int, int]],
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """Perform forward-only GF2 Gaussian elimination (no back-substitution).

    Produces an upper-triangular (row echelon form) matrix rather than RREF.
    Back-substitution is skipped so that binary encoding's staircase
    conversion can be applied directly to the REF pivot block.

    Args:
        matrix: Binary matrix to reduce (copied internally).
        row_map: Current-to-original row index mapping (copied internally).
        cnot_ops: Existing CNOT operation list (copied internally).

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

        # Eliminate only rows BELOW the pivot (forward elimination only)
        below = np.flatnonzero(matrix_work[pivot_row + 1 :, col]) + pivot_row + 1
        for r in below:
            matrix_work[r] ^= matrix_work[pivot_row]
            cnot_ops_work.append((row_map_work[r], row_map_work[pivot_row]))

        pivot_row += 1
        if pivot_row == num_rows:
            break

    return matrix_work, row_map_work, cnot_ops_work


def _ref_to_staircase(
    matrix: np.ndarray,
    row_map: list[int],
    operations: list[tuple[str, int | tuple[int, int]]],
) -> tuple[np.ndarray, list[int], list[tuple[str, int | tuple[int, int]]]]:
    """Convert a REF (upper-triangular) pivot block to upper-staircase form.

    For each above-diagonal entry ``(i, pivot_col_j)`` where ``j > i``:

    * If the entry is already 1, it's correct — do nothing.
    * If the entry is 0, apply CX(control=row_j, target=row_i) to set it to 1.

    Processing columns left-to-right guarantees that side effects on later
    columns are absorbed when we reach them.

    After this transform the pivot sub-matrix equals ``np.triu(ones(r, r))``
    which binary encoding's ``_is_diagonal_reduction_shape`` recognises,
    allowing it to skip its own CX cascade.

    Args:
        matrix: REF binary matrix (rank x n_cols).
        row_map: Current row-to-original-qubit mapping.
        operations: Existing operations list to extend.

    Returns:
        ``(matrix, row_map, operations)`` — updated in place.

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    operations_work = operations.copy()

    rank = matrix_work.shape[0]

    # Identify pivot columns
    pivot_cols: list[int] = []
    for r in range(rank):
        nz = np.flatnonzero(matrix_work[r])
        if nz.size > 0:
            pivot_cols.append(int(nz[0]))

    # Fill above-diagonal entries in pivot columns to reach staircase form
    for j_idx in range(1, len(pivot_cols)):
        pc = pivot_cols[j_idx]
        for i_idx in range(j_idx):
            if not matrix_work[i_idx, pc]:
                # Need to set this entry to 1: CX(control=row_j, target=row_i)
                matrix_work[i_idx] ^= matrix_work[j_idx]
                operations_work.append(("cx", (row_map_work[i_idx], row_map_work[j_idx])))

    return matrix_work, row_map_work, operations_work


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
        :class:`GF2XEliminationResult` with rank decremented by 1.

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
