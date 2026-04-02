"""Sparse isometry with binary encoding for quantum state preparation.

This module implements a state preparation algorithm that combines GF2+X
elimination with batched binary encoding.  Instead of delegating the reduced
subspace to a dense state preparation routine (as the base
:class:`SparseIsometryGF2XStatePreparation` does), this algorithm feeds the
RREF matrix directly into the binary-encoding solver which synthesises the
full circuit using batched Toffoli gates and Partial Unary Iteration (PUI)
lookup blocks.

The approach is particularly effective for wavefunctions with many
determinants whose binary matrix has favourable sparsity structure,
since the entire expansion is expressed in terms of CX, Toffoli, and
lookup gates with no intermediate dense state preparation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import numpy as np
import qdk

from qdk_chemistry.data import Circuit, Wavefunction
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.binary_encoding import BinaryEncodingSynthesizer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .sparse_isometry import (
    GF2XEliminationResult,
    MatrixCompressionOp,
    SparseIsometryGF2XStatePreparation,
    SparseIsometryGF2XStatePreparationSettings,
    gf2x_with_tracking,
)


class SparseIsometryBinaryEncodingSettings(SparseIsometryGF2XStatePreparationSettings):
    """Settings for SparseIsometryBinaryEncodingStatePreparation."""

    def __init__(self):
        """Initialize with parent defaults plus binary-encoding controls."""
        super().__init__()
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


class SparseIsometryBinaryEncodingStatePreparation(SparseIsometryGF2XStatePreparation):
    """State preparation using sparse isometry with binary encoding.

    This class extends :class:`SparseIsometryGF2XStatePreparation` by replacing
    the dense state preparation step with a binary-encoding circuit synthesiser.
    After GF2+X elimination produces a reduced RREF matrix, the binary-encoding
    synthesiser (:class:`~qdk_chemistry.algorithms.state_preparation.binary_encoding.BinaryEncodingSynthesizer`)
    compresses the matrix into an efficient circuit using:

        1. Stage 1 — diagonal (unary-to-binary) encoding of pivot columns
        2. Stage 2 — non-pivot column processing with batched PUI lookup blocks

    The resulting circuit consists entirely of CX, Toffoli, X, SWAP, and
    PUI-lookup gates — no dense state preparation is used.

    Key References:

        * Sparse isometry: Malvetti, Iten, and Colbeck (arXiv:2006.00016) :cite:`Malvetti2021`
    """

    def __init__(self) -> None:
        """Initialize the SparseIsometryBinaryEncodingStatePreparation."""
        Logger.trace_entering()
        super().__init__()
        self._settings = SparseIsometryBinaryEncodingSettings()

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit using GF2+X elimination followed by binary encoding.

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            A Circuit object containing the quantum circuit.

        """
        Logger.trace_entering()

        # Active Space Consistency Check (same as parent)
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
            bitstrings.append(beta_str[::-1] + alpha_str[::-1])

        if len(bitstrings) == 1:
            Logger.info("After filtering, only 1 determinant remains, using single reference state preparation")
            return self._prepare_single_reference_state(bitstrings[0])

        n_qubits = len(bitstrings[0])
        Logger.debug(f"Using {len(bitstrings)} determinants for state preparation")

        # Step 1: GF2+X elimination — skip the diagonal reduction because
        # binary encoding's stage-1 handles the identity pivot block natively;
        # the extra CX + X ops from diagonal reduction would be redundant.
        bitstring_matrix = self._bitstrings_to_binary_matrix(bitstrings)
        gf2x_result = gf2x_with_tracking(bitstring_matrix, skip_diagonal_reduction=True, staircase_mode=True)

        # Step 2: Binary encoding on the reduced RREF matrix
        binary_ops, num_ancilla, bijection, dense_size = self._perform_binary_encoding(gf2x_result, n_qubits)

        # Step 2b: Build compressed statevector reindexed by the bijection.
        # The bijection maps (dense_val, orig_col) where orig_col is the
        # determinant index and dense_val is the binary-register label.
        compressed_sv = np.zeros(2**dense_size, dtype=float)
        for dense_val, orig_col in bijection:
            if orig_col < len(coeffs):
                compressed_sv[dense_val] = coeffs[orig_col]
        norm = np.linalg.norm(compressed_sv)
        if norm > 0:
            compressed_sv /= norm

        # The dense register consists of the first dense_size rows of the
        # tableau, which map to the first dense_size entries of row_map.
        dense_row_map = gf2x_result.row_map[:dense_size]

        # Step 3: Build expansion operations from GF2+X elimination
        expansion_ops: list[MatrixCompressionOp] = []
        for operation in reversed(gf2x_result.operations):
            if operation[0] in ("cx", "cnot"):
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    expansion_ops.append(MatrixCompressionOp("CX", [control, target]))
            elif operation[0] == "x" and isinstance(operation[1], int):
                expansion_ops.append(MatrixCompressionOp("X", [operation[1]]))

        # Step 4: Pre-process binary-encoding ops into MatrixCompressionOp instances
        encoded_ops = _encode_gf2x_ops_for_qs(binary_ops)

        # Step 4b: Elide redundant CX pair at the boundary.
        # Binary encoding's unary staircase always starts with CX(row_map[rank-1], row_map[rank-2])
        # which, after reversal, becomes the LAST encoded_op.
        # GF2+X back-substitution often ends with the same CX (clearing the
        # entry above the last pivot), which becomes the FIRST expansion_op.
        # Since CX is self-inverse, the pair cancels — remove both.
        if (
            encoded_ops
            and expansion_ops
            and encoded_ops[-1].name == "CX"
            and expansion_ops[0].name == "CX"
            and encoded_ops[-1].qubits == expansion_ops[0].qubits
        ):
            Logger.debug(f"Eliding redundant boundary CX pair on qubits {encoded_ops[-1].qubits}")
            encoded_ops.pop()
            expansion_ops.pop(0)

        # Build circuit using QDK Q# factory with binary-encoding entry point
        # dense_val from the bijection uses row 0 = MSB (_bits_to_int is MSB-first).
        # PreparePureStateD treats qubits[0] as MSB, so pass dense_row_map
        # as-is (row 0 first) — do NOT reverse like the parent sparse isometry
        # (which uses the opposite convention: row rank-1 = MSB).
        state_prep_params = qdk.code.BinaryEncodingStatePreparationParams(
            rowMap=list(dense_row_map),
            stateVector=compressed_sv.tolist(),
            expansionOps=[op.to_dict() for op in expansion_ops],
            binaryEncodingOps=[op.to_dict() for op in encoded_ops],
            numQubits=n_qubits,
            numAncilla=num_ancilla,
        )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.BinaryEncoding.MakeBinaryEncodingStatePreparationCircuit,
            parameter=vars(state_prep_params),
        )

        Logger.info(
            f"Binary encoding produced {len(binary_ops)} operations ({len(encoded_ops)} encoded) "
            f"using {num_ancilla} ancillae for {n_qubits}-qubit system with {len(bitstrings)} determinants"
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            encoding="jordan-wigner",
        )

    def _perform_binary_encoding(
        self, gf2x_result: GF2XEliminationResult, n_qubits: int
    ) -> tuple[list[tuple[str, Any]], int, list[tuple[int, int]], int]:
        """Run binary-encoding synthesis on the reduced RREF matrix.

        Args:
            gf2x_result: Result from GF2+X elimination containing the reduced matrix.
            n_qubits: Total number of qubits in the original space.

        Returns:
            Tuple of ``(gf2x_ops, num_ancilla, bijection, dense_size)``:

            - ``gf2x_ops``: gate operations from the binary-encoding solver.
            - ``num_ancilla``: number of ancilla qubits consumed.
            - ``bijection``: list of ``(dense_val, orig_col)`` mapping each
              original matrix column to its compressed binary-register label.
            - ``dense_size``: number of qubits in the compressed dense register.

        """
        include_negative_controls = self._settings.get("include_negative_controls")

        Logger.debug(
            f"Binary encoding input: {gf2x_result.reduced_matrix.shape} matrix, "
            f"include_negative_controls={include_negative_controls}"
        )

        synthesizer = BinaryEncodingSynthesizer.from_matrix(
            gf2x_result.reduced_matrix,
            include_negative_controls=include_negative_controls,
            measurement_based_uncompute=self._settings.get("measurement_based_uncompute"),
        )

        gf2x_ops, num_ancilla = synthesizer.to_gf2x_operations(
            num_local_qubits=n_qubits,
            active_qubit_indices=gf2x_result.row_map,
            ancilla_start=n_qubits,
        )

        Logger.debug(
            f"Binary encoding output: {len(gf2x_ops)} ops, "
            f"{num_ancilla} ancillae, bijection size {len(synthesizer.bijection)}, "
            f"dense_size {synthesizer.dense_size}"
        )

        return gf2x_ops, num_ancilla, synthesizer.bijection, synthesizer.dense_size

    def name(self) -> str:
        """Return the algorithm identifier string."""
        return "sparse_isometry_binary_encoding"


def _encode_gf2x_ops_for_qs(
    operations: list[tuple[str, Any]],
) -> list[MatrixCompressionOp]:
    """Pre-process GF2+X operations into :class:`MatrixCompressionOp` instances for Q#.

    The resulting list is already in *reversed* order so that Q# can iterate
    forward.

    Args:
        operations: Sequence of ``(op_name, op_args)`` tuples as produced by
            :meth:`BinaryEncodingSynthesizer.to_gf2x_operations`.

    Returns:
        List of :class:`MatrixCompressionOp` ready to be serialised for Q#.

    """
    ops: list[MatrixCompressionOp] = []

    for op_name, op_args in reversed(operations):
        if op_name == "x":
            ops.append(MatrixCompressionOp("X", [int(op_args)]))

        elif op_name == "cx":
            ops.append(MatrixCompressionOp("CX", [int(op_args[0]), int(op_args[1])]))

        elif op_name == "swap":
            ops.append(MatrixCompressionOp("SWAP", [int(op_args[0]), int(op_args[1])]))

        elif op_name == "ccx":
            target, ctrl1, ctrl2 = op_args
            ops.append(MatrixCompressionOp("CCX", [int(ctrl1), int(ctrl2), int(target)]))

        elif op_name in ("select", "select_and"):
            data_table, addr_qubits, dat_qubits = op_args
            qubits = [int(q) for q in addr_qubits] + [int(q) for q in dat_qubits]
            qs_name = "SELECT_AND" if op_name == "select_and" else "SELECT"
            ops.append(
                MatrixCompressionOp(
                    qs_name,
                    qubits,
                    control_state=len(addr_qubits),
                    lookup_data=data_table,
                )
            )

    return ops
