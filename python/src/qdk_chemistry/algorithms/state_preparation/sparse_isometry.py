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
  then applies recorded operations (CNOT and X) in reverse to expand back to
  the full space.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager

from qdk_chemistry.algorithms.state_preparation.state_preparation import StatePreparation, StatePreparationSettings
from qdk_chemistry.data import Circuit, Wavefunction
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class SparseIsometryGF2XStatePreparation(StatePreparation):
    """State preparation using sparse isometry with enhanced GF2+X elimination.

    This class implements "GF2+X" state preparation for electronic structure problems using
    the ``gf2x_with_tracking`` function which performs smart preprocessing
    before GF2 Gaussian elimination. The preprocessing includes:

        1. Removing duplicate rows using CNOT operations
        2. Removing all-ones rows using X operations
        3. Then performing standard GF2 Gaussian elimination
        4. Apply the additional rank reduction if the reduced row-echelon matrix is diagonal

    This enhanced approach can be more efficient than standard GF2 Gaussian elimination,
    particularly for matrices with duplicate rows or all-ones rows. The algorithm
    tracks both CNOT and X operations for proper circuit reconstruction.

    The algorithm:

        1. Reads the wavefunction to get coefficients and bitstrings
        2. Converts bitstrings to a binary matrix
        3. Applies enhanced GF2+X elimination (duplicate removal + all-ones removal + GF2)
        4. Performs dense state preparation on the reduced space
        5. Applies recorded operations (both CNOT and X) in reverse order to expand back to full space

    Key References:

        * Sparse isometry: Malvetti, Iten, and Colbeck (arXiv:2006.00016) :cite:`Malvetti2021`

    """

    def __init__(self):
        """Initialize the SparseIsometryGF2XStatePreparation."""
        Logger.trace_entering()
        super().__init__()
        self._settings = StatePreparationSettings()

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit that encodes the given wavefunction using sparse isometry over GF(2^x).

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            A Circuit object containing an OpenQASM3 string of the quantum circuit that prepares the wavefunction.

        """
        Logger.trace_entering()
        # Imported here to avoid circular import issues
        from qdk_chemistry.plugins.qiskit._interop.transpiler import (  # noqa: PLC0415
            MergeZBasisRotations,
            RemoveZBasisOnZeroState,
            SubstituteCliffordRz,
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
        Logger.debug(f"CNOT operations: {[op for op in gf2x_operation_results.operations if op[0] == 'cnot']}")
        Logger.debug(f"X operations: {[op for op in gf2x_operation_results.operations if op[0] == 'x']}")

        # Step 3: Create quantum circuit
        qc = QuantumCircuit(
            n_qubits,
            name=f"sparse_isometry_gf2x_{len(bitstrings)}_dets",
        )

        # Step 4: Create statevector for the reduced matrix
        if gf2x_operation_results.rank > 0:
            # Create statevector correctly preserving coefficient-determinant correspondence.
            # Each coefficient corresponds to a specific determinant (column in reduced matrix).
            # We need to map each coefficient to the correct basis state in the reduced space.

            statevector_data = np.zeros(2**gf2x_operation_results.rank, dtype=complex)

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

            # Create Statevector object for StatePreparation
            statevector = Statevector(statevector_data)

            # Step 5: Apply dense state preparation on reduced space
            Logger.debug(f"Target indices are {gf2x_operation_results.row_map}")
            qc.append(QiskitStatePreparation(statevector, normalize=False), gf2x_operation_results.row_map)
        else:
            # If reduced matrix has zero rank, all determinants are identical
            raise ValueError(
                "Cannot perform sparse isometry on identical determinants. All determinants must be distinct. "
                "Please check your wavefunction data - you may have duplicate determinants or "
                "need to use a single-determinant state preparation method."
            )

        # Step 6: Apply recorded operations in reverse order to expand back to full space.
        # Note: GF2+X can have both CNOT and X operations
        for operation in reversed(gf2x_operation_results.operations):
            if operation[0] == "cnot":
                # operation[1] should be a tuple for CNOT operations
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    qc.cx(control, target)
            elif operation[0] == "x" and isinstance(operation[1], int):
                # operation[1] should be an int for X operations
                qubit = operation[1]
                qc.x(qubit)

        Logger.info(
            f"Final circuit before transpilation: {qc.num_qubits} qubits, depth {qc.depth()}, {qc.size()} gates"
        )

        # Transpile the circuit if needed
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

    def _bitstrings_to_binary_matrix(self, bitstrings: list[str]) -> np.ndarray:
        """Convert a list of bitstrings to a binary matrix.

        This function converts a list of bitstrings (determinants) into a binary matrix
        where each column represents a determinant and each row represents a qubit.

        Args:
            bitstrings (list[str]): List of bitstrings in Qiskit little endian order.
                Each bitstring represents a determinant where the string is ordered
                as "q[N-1]...q[0]" (most significant bit first in the string).

        Returns:
            Binary matrix M of shape (N, k) where

                * N is the number of qubits (rows)
                * k is the number of determinants (columns)

            The matrix follows Qiskit circuit top-down convention with row ordering "q[0]...q[N-1]"
            (qubit 0 at the top).

        Note:
            The input bitstrings are in Qiskit little endian order ("q[N-1]...q[0]"),
            but the output binary matrix follows the Qiskit circuit convention with
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

        num_qubits = len(bitstring)
        circuit = QuantumCircuit(num_qubits, name=f"SingleRef_{bitstring}")

        # Apply X gates for positions with '1'
        # Note: bitstring is in little-endian format (rightmost bit = qubit 0)
        for i, bit in enumerate(reversed(bitstring)):
            if bit == "1":
                circuit.x(i)

        return Circuit(qasm=qasm3.dumps(circuit), encoding="jordan-wigner")

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

        * ('cnot', (target_row, control_row)) for CNOT operations
        * ('x', row_index) for X operations on entire rows

    All indices refer to original matrix positions.
    """

    rank: int
    """Rank of the reduced matrix (number of non-zero rows)."""


def gf2x_with_tracking(matrix: np.ndarray) -> GF2XEliminationResult:
    """Perform enhanced GF2+X Gaussian elimination with smart preprocessing and X operations.

    This function implements a smarter approach to GF2 Gaussian elimination by:

        1. First removing duplicate rows using CNOT operations
        2. Removing all-ones rows using X operations
        3. Then performing standard Gaussian elimination
        4. Performing further reduction if the resulting matrix is diagonal

    This approach can be more efficient than standard Gaussian elimination alone,
    especially for certain types of matrices.

    Args:
        matrix: shape (m, n), binary (0/1) matrix

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

    # Step 1: Remove duplicate rows using CNOT operations
    matrix_work, row_map, operations = _remove_duplicate_rows_with_cnot(matrix_work, row_map, operations)

    # Step 2: Remove all-ones rows using X operations
    matrix_work, row_map, operations = _remove_all_ones_rows_with_x(matrix_work, row_map, operations)

    # Step 3: Perform standard Gaussian elimination on the remaining matrix
    if matrix_work.shape[0] > 0:  # Only if there are rows left
        # Convert CNOT operations to the format expected by Gaussian elimination
        cnot_ops = []
        for op in operations:
            if op[0] == "cnot" and isinstance(op[1], tuple):
                cnot_ops.append((op[1][0], op[1][1]))

        # Perform Gaussian elimination
        m_current, n_current = matrix_work.shape
        matrix_processed, updated_row_map, updated_cnot_ops = _perform_gaussian_elimination(
            matrix_work, m_current, n_current, row_map, cnot_ops
        )

        # Update operations list with new CNOT operations from Gaussian elimination
        for target, control in updated_cnot_ops[len(cnot_ops) :]:  # Only add new operations
            operations.append(("cnot", (target, control)))

        # Remove zero rows and update row_map accordingly
        matrix_reduced, reduced_row_map, rank = _remove_zero_rows(matrix_processed, updated_row_map)

        gf2x_results = GF2XEliminationResult(
            reduced_matrix=matrix_reduced, row_map=reduced_row_map, col_map=col_map, operations=operations, rank=rank
        )

        # Step 4: Check for diagonal matrix and apply further reduction if possible
        if rank > 1 and _is_diagonal_matrix(matrix_reduced):
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
    rows_to_eliminate = []

    # Find duplicate rows
    for i in range(n_rows):
        # Skip rows that are already marked for elimination
        if i in rows_to_eliminate:
            continue

        row_i = matrix_work[i]

        # Skip all-zero rows as they don't need CNOT operations
        if np.all(row_i == 0):
            continue

        # Look for duplicates of this row
        for j in range(i + 1, n_rows):
            if j in rows_to_eliminate:
                continue

            row_j = matrix_work[j]

            # If rows are identical, eliminate the later one
            if np.array_equal(row_i, row_j):
                # CNOT(control=i, target=j) will make row j become all zeros
                operations_work.append(("cnot", (row_map_work[j], row_map_work[i])))
                rows_to_eliminate.append(j)

                Logger.info(
                    f"Found duplicate row {j} identical to row {i}, adding CNOT({row_map_work[i]}, {row_map_work[j]})"
                )

    # Apply CNOT operations to eliminate duplicate rows
    for op in operations_work:
        if op[0] == "cnot" and isinstance(op[1], tuple):
            # Find the current positions of the target and control rows
            target_orig, control_orig = op[1]
            target_current = row_map_work.index(target_orig)
            control_current = row_map_work.index(control_orig)

            # Apply CNOT: target row = target row XOR control row
            matrix_work[target_current] = matrix_work[target_current] ^ matrix_work[control_current]

    # Remove eliminated rows (which should now be all zeros)
    if rows_to_eliminate:
        Logger.info(f"Eliminating {len(rows_to_eliminate)} duplicate rows: {rows_to_eliminate}")

        # Create mask for rows to keep
        rows_to_keep = [i for i in range(n_rows) if i not in rows_to_eliminate]

        # Update matrix and row mapping
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
    num_rows: int,
    num_cols: int,
    row_map: list[int],
    cnot_ops: list[tuple[int, int]],
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """Perform the main GF2 Gaussian elimination steps on a binary matrix.

    This function implements the core algorithm of GF2 Gaussian elimination by iterating through columns,
    finding pivot rows, swapping rows when necessary, and eliminating other entries in each column using XOR operations.

    Args:
        matrix: Binary matrix to reduce (copied, not modified in-place)
        num_rows: Number of rows in the matrix
        num_cols: Number of columns in the matrix
        row_map: Mapping from current to original row indices (copied, not modified)
        cnot_ops: List to record CNOT operations (copied, not modified)

    Returns:
        A tuple containing ``(updated_matrix, updated_row_map, updated_operations)``

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    cnot_ops_work = cnot_ops.copy()

    row = 0  # current row
    for col in range(num_cols):
        # Find the first row (row >= current) with a 1 in this column
        sel = _find_pivot_row(matrix_work, row, num_rows, col)
        if sel is None:
            continue

        # Swap current row and selected row if needed
        if sel != row:
            matrix_work[[row, sel], :] = matrix_work[[sel, row], :]
            row_map_work[row], row_map_work[sel] = row_map_work[sel], row_map_work[row]

        # Eliminate all other rows (except the pivot row) in this column
        matrix_work, cnot_ops_work = _eliminate_column(matrix_work, num_rows, row, col, row_map_work, cnot_ops_work)

        # Move to next row
        row += 1
        if row == num_rows:
            break

    return matrix_work, row_map_work, cnot_ops_work


def _find_pivot_row(matrix: np.ndarray, row: int, num_rows: int, col: int) -> int | None:
    """Find the first row with a 1 in the given column for pivot selection.

    This function searches for a suitable pivot row starting from the current row position downward.
    It looks for the first row that has a 1 in the specified column,
    which can be used as a pivot for Gaussian elimination.

    Args:
        matrix: Binary matrix to search (read-only, not modified)
        row: Starting row index to search from (inclusive)
        num_rows: Total number of rows in the matrix
        col: Column index to check for pivot candidates

    Returns:
        Index of the first row with a 1 in the column, or None if no suitable pivot is found in the remaining rows.

    Note:
        This function only reads the matrix and does not modify any arguments.
        It returns None when no pivot can be found, indicating the column should be skipped.

    """
    for r in range(row, num_rows):
        if matrix[r, col]:
            return r
    return None


def _eliminate_column(
    matrix: np.ndarray,
    num_rows: int,
    pivot_row: int,
    col: int,
    row_map: list[int],
    cnot_ops: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Eliminate all other rows in the given column using XOR operations.

    This function performs the elimination step of GF2 Gaussian elimination
    by XORing the pivot row with all other rows that have a 1 in the current column.

    Args:
        matrix: Binary matrix to modify (copied, not modified in-place)
        num_rows: Number of rows in the matrix
        pivot_row: Index of the pivot row (remains unchanged)
        col: Column index to eliminate
        row_map: Mapping from current to original row indices (read-only)
        cnot_ops: List to record CNOT operations (copied, not modified)

    Returns:
        tuple[np.ndarray, list[tuple[int, int]]]: Tuple containing:

            * ``updated_matrix``: Matrix after column elimination.
            * ``updated_cnot_ops``: Updated list of CNOT operations.

    """
    matrix_work = matrix.copy()
    cnot_ops_work = cnot_ops.copy()

    for r in range(num_rows):
        if r != pivot_row and matrix_work[r, col]:
            matrix_work[r, :] ^= matrix_work[pivot_row, :]
            # Record CNOT operation using original matrix indices
            cnot_ops_work.append((row_map[r], row_map[pivot_row]))

    return matrix_work, cnot_ops_work


def _remove_zero_rows(matrix: np.ndarray, row_map: list[int]) -> tuple[np.ndarray, list[int], int]:
    """Remove zero rows from the matrix and update row mapping.

    This function creates a new matrix containing only the non-zero rows from the input matrix,
    along with an updated row mapping that tracks which original rows correspond to the rows in the reduced matrix.

    Args:
        matrix: Binary matrix to process (read-only, not modified)
        row_map: Current mapping from matrix rows to original indices (read-only)

    Returns:
        tuple[np.ndarray, list[int], int]: Tuple containing:

            * ``matrix_reduced``: New matrix with only non-zero rows.
            * ``reduced_row_map``: Updated mapping from reduced matrix rows to original indices.
            * ``rank``: Number of non-zero rows (matrix rank).

    Note:
        This function does not modify its input arguments. It creates and returns
        new objects containing only the non-zero rows and their corresponding mappings.

    """
    n_rows, _ = matrix.shape
    non_zero_rows = []
    reduced_row_map = []

    for i in range(n_rows):
        if not np.all(matrix[i, :] == 0):  # Keep non-zero rows
            non_zero_rows.append(i)
            reduced_row_map.append(row_map[i])

    # Extract only non-zero rows
    matrix_reduced = matrix[non_zero_rows, :]
    rank = len(non_zero_rows)

    return matrix_reduced, reduced_row_map, rank


def _reduce_diagonal_matrix(
    matrix: np.ndarray,
    row_map: list[int],
    col_map: list[int],
    operations: list[tuple[str, int | tuple[int, int]]],
) -> GF2XEliminationResult:
    """Further reduce a diagonal matrix using CNOT and X operations.

    This function handles the special case where the matrix is diagonal
    (square matrix with 1s on diagonal and 0s elsewhere). It applies
    sequential CNOT operations to create an all-ones row, then uses
    an X operation to eliminate it, reducing the rank by 1.

    Procedure:

        1. Apply CNOT(i, i+1) sequentially for i = 0 to rank-2
        2. This makes the last row (rank-1) become all 1s
        3. Apply X on the last row to make it all 0s
        4. Remove the zero row to reduce rank by 1

    Args:
        matrix: Diagonal binary matrix to reduce
        row_map: Current row mapping to original indices
        col_map: Column mapping to original indices (unchanged)
        operations: List of operations performed so far

    Returns:
        A tuple containing:

            * matrix_reduced: Further reduced matrix with rank decreased by 1
            * reduced_row_map: Updated row mapping (last row removed)
            * col_map: Unchanged column mapping
            * updated_operations: Operations list with new CNOT and X operations
            * new_rank: Original rank - 1

    """
    matrix_work = matrix.copy()
    row_map_work = row_map.copy()
    operations_work = operations.copy()

    rank = matrix_work.shape[0]

    # Verify this is actually a diagonal matrix
    if not _is_diagonal_matrix(matrix_work):
        Logger.warn("Matrix is not diagonal, skipping diagonal reduction")
        return GF2XEliminationResult(
            reduced_matrix=matrix_work, row_map=row_map_work, col_map=col_map, operations=operations_work, rank=rank
        )

    Logger.info(f"Applying diagonal matrix reduction on {rank}x{rank} matrix")

    # Step 1: Apply sequential CNOT operations CNOT(i, i+1) for i = 0 to rank-2
    for i in range(rank - 1):
        control_idx = i
        target_idx = i + 1

        # Record CNOT operation using original row indices
        operations_work.append(
            (
                "cnot",
                (row_map_work[target_idx], row_map_work[control_idx]),
            )
        )

        # Apply CNOT: target row = target row XOR control row
        matrix_work[target_idx] = matrix_work[target_idx] ^ matrix_work[control_idx]

        Logger.info(f"Applied CNOT({row_map_work[control_idx]}, {row_map_work[target_idx]})")

    # After all CNOTs, the last row should be all 1s
    last_row = rank - 1
    Logger.info(f"Last row after CNOTs: {matrix_work[last_row]}")

    # Step 2: Apply X operation on the last row to make it all 0s
    operations_work.append(("x", row_map_work[last_row]))
    matrix_work[last_row] = np.zeros(matrix_work.shape[1], dtype=matrix_work.dtype)

    Logger.info(f"Applied X operation on row {row_map_work[last_row]}")

    # Step 3: Remove the last row (which is now all zeros)
    matrix_reduced = matrix_work[:-1, :]  # Remove last row
    reduced_row_map = row_map_work[:-1]  # Remove last row mapping
    new_rank = rank - 1

    Logger.info(f"Diagonal reduction complete: rank reduced from {rank} to {new_rank}")

    return GF2XEliminationResult(
        reduced_matrix=matrix_reduced,
        row_map=reduced_row_map,
        col_map=col_map,
        operations=operations_work,
        rank=new_rank,
    )


def _is_diagonal_matrix(matrix: np.ndarray) -> bool:
    """Check if a binary matrix is diagonal and safe for a further rank reduction.

    The diagonal reduction optimization is mathematically valid in two scenarios:

        1. True diagonal matrix: Square matrix with 1s on diagonal, 0s elsewhere
        2. Pseudo-diagonal: Rectangular matrix where:
            * The square part (min(rows,cols) x min(rows,cols)) is diagonal
            * ALL remaining columns are all 1s
            * We have an odd number of rows

    The rank reduction works by applying sequential CNOTs and an X operation,
    which is only valid when these specific structural conditions are met.
    We also require rank > 1 since rank 1 matrices are already minimal.

    Args:
        matrix: Binary matrix to check

    Returns:
        True if matrix is diagonal and safe for rank reduction, False otherwise

    """
    # Check basic requirements
    if matrix.ndim != 2 or matrix.shape[0] <= 1 or not np.array_equal(matrix & 1, matrix):
        return False

    num_rows, num_cols = matrix.shape

    # Scenario 1: True diagonal matrix (square)
    if num_rows == num_cols:
        is_diagonal = True
        for row_idx in range(num_rows):
            for col_idx in range(num_rows):
                expected_value = 1 if row_idx == col_idx else 0
                if matrix[row_idx, col_idx] != expected_value:
                    is_diagonal = False
                    break
            if not is_diagonal:
                break
        if is_diagonal:
            return True

    # Scenario 2: Pseudo-diagonal (rectangular with more columns than rows)
    # ONLY valid when: remaining columns are ALL 1s AND odd number of rows
    elif num_cols > num_rows and num_rows % 2 == 1:
        # Check the square part is diagonal
        square_part = matrix[:num_rows, :num_rows]
        is_square_diagonal = True
        for row_idx in range(num_rows):
            for col_idx in range(num_rows):
                expected_value = 1 if row_idx == col_idx else 0
                if square_part[row_idx, col_idx] != expected_value:
                    is_square_diagonal = False
                    break
            if not is_square_diagonal:
                break

        # If square part is diagonal, check remaining columns are all 1s
        if is_square_diagonal:
            remaining_columns = matrix[:, num_rows:]
            if np.all(remaining_columns == 1):
                return True

    # All other cases: not safe for diagonal reduction
    return False
