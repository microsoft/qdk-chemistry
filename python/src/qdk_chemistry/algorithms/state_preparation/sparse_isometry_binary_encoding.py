"""Sparse isometry with binary encoding for quantum state preparation.

This module implements a state preparation algorithm that combines GF2+X
elimination with batched binary encoding.  Instead of delegating the reduced
subspace to a dense state preparation routine, this algorithm feeds the
REF matrix directly into the binary-encoding solver which synthesises the
full circuit using batched Toffoli gates and Partial Unary Iteration (PUI)
lookup blocks.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import Circuit, Wavefunction
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.binary_encoding import BinaryEncodingSynthesizer, MatrixCompressionOp, MatrixCompressionType
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .sparse_isometry import (
    GF2XEliminationResult,
    SparseIsometryGF2XStatePreparation,
    SparseIsometryGF2XStatePreparationSettings,
    gf2x_with_tracking,
)

__all__ = [
    "SparseIsometryBinaryEncodingSettings",
]


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

    This class extends sparse isometry with GF2+X elimination by replacing
    the dense state preparation step with a binary-encoding circuit synthesiser.
    After GF2+X elimination produces a REF matrix, the binary-encoding
    synthesiser compresses the matrix into an efficient circuit using:

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

        params = self._build_binary_encoding_params(wavefunction)

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.BinaryEncoding.MakeBinaryEncodingStatePreparationCircuit,
            parameter=params,
        )
        qsharp_op = QSHARP_UTILS.BinaryEncoding.MakeBinaryEncodingStatePreparationOp(*params.values())

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            encoding="jordan-wigner",
        )

    def _build_binary_encoding_params(self, wavefunction: Wavefunction) -> dict:
        """Build binary-encoding state preparation parameters from a wavefunction.

        Extracts coefficients and determinants, performs GF2+X elimination and
        binary-encoding synthesis, and returns the parameter dict for Q# circuit
        construction.

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            A dict of parameters for Q# circuit construction.

        """
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
        gf2x_result = gf2x_with_tracking(bitstring_matrix, skip_diagonal_reduction=True, forward_only=True)

        # Step 2: Binary encoding on the REF matrix
        encoded_ops, bijection, dense_size = self._perform_binary_encoding(gf2x_result, n_qubits)

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
        gaussian_elimination_ops: list[MatrixCompressionOp] = []
        for operation in reversed(gf2x_result.operations):
            if operation[0] in ("cx", "cnot"):
                if isinstance(operation[1], tuple):
                    target, control = operation[1]
                    gaussian_elimination_ops.append(MatrixCompressionOp(MatrixCompressionType.CX, [control, target]))
            elif operation[0] == "x" and isinstance(operation[1], int):
                gaussian_elimination_ops.append(MatrixCompressionOp(MatrixCompressionType.X, [operation[1]]))

        # Build circuit using QDK Q# factory with binary-encoding entry point
        # dense_val from the bijection uses row 0 = MSB (_bits_to_int is MSB-first).
        # PreparePureStateD treats qubits[0] as MSB, so pass dense_row_map
        # as-is (row 0 first) — do NOT reverse like the parent sparse isometry
        # (which uses the opposite convention: row rank-1 = MSB).
        # Create the ancilla pool from the original qubits that are not touched by binary encoding (i.e. not in row_map)
        # since they are idle until the expansion stage and can be borrowed as ancillas during SparseOneHotSCS.
        active_qubits_set = {int(q) for q in gf2x_result.row_map}
        ancilla_pool = sorted(set(range(n_qubits)) - active_qubits_set)

        state_prep_params = QSHARP_UTILS.BinaryEncoding.BinaryEncodingStatePreparationParams(
            rowMap=list(dense_row_map),
            stateVector=compressed_sv.tolist(),
            gaussianEliminationOps=[op.to_qsharp_parameter() for op in gaussian_elimination_ops],
            binaryEncodingOps=[op.to_qsharp_parameter() for op in encoded_ops],
            numQubits=n_qubits,
            ancillaPool=ancilla_pool,
        )
        params = vars(state_prep_params)

        Logger.info(
            f"Binary encoding produced {len(params['binaryEncodingOps'])} operations "
            f"for {n_qubits}-qubit system with {len(bitstrings)} determinants "
            f"using {len(params['ancillaPool'])} pre-existing qubits as ancilla pool"
        )
        return params

    def _create_dense(self, params: dict) -> Circuit:
        """Create a standalone dense state preparation circuit.

        Args:
            params: The parameter dict for Q# circuit construction.

        Returns:
            A dense state preparation circuit on the reduced qubit subset.

        """
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeDenseStatePreparation,
            parameter={
                "rowMap": params["rowMap"],
                "stateVector": params["stateVector"],
                "numQubits": params["numQubits"],
            },
        )
        return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")

    def _create_isometry(self, params: dict) -> Circuit:
        """Create a standalone isometry circuit (binary encoding + GF2+X expansion).

        Args:
            params: The parameter dict for Q# circuit construction.

        Returns:
            A Circuit containing the binary-encoding operations followed by
            the GF2+X expansion operations.

        """
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.BinaryEncoding.MakeBinaryEncodingExpansion,
            parameter={
                "binaryEncodingOps": params["binaryEncodingOps"],
                "gaussianEliminationOps": params["gaussianEliminationOps"],
                "numQubits": params["numQubits"],
                "ancillaPool": params["ancillaPool"],
            },
        )
        return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")

    def _perform_binary_encoding(
        self, gf2x_result: GF2XEliminationResult, n_qubits: int
    ) -> tuple[list[MatrixCompressionOp], list[tuple[int, int]], int]:
        """Run binary-encoding synthesis and return Q#-ready ops.

        Runs the synthesiser, translates qubit indices from local to global,
        and converts operations directly into MatrixCompressionOp
        instances in reversed order (so Q# can iterate forward).

        Args:
            gf2x_result: Result from GF2+X elimination containing the reduced matrix.
            n_qubits: Total number of qubits in the original space.

        Returns:
            Tuple of ``(ops, bijection, dense_size)``:

            - ``ops``: MatrixCompressionOp list ready for Q#.
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

        ops = synthesizer.to_operations(
            num_local_qubits=n_qubits,
            active_qubit_indices=gf2x_result.row_map,
            ancilla_start=n_qubits,
            reverse=True,
        )

        Logger.debug(
            f"Binary encoding output: {len(ops)} ops, "
            f"bijection size {len(synthesizer.bijection)}, "
            f"dense_size {synthesizer.dense_size}"
        )

        return ops, synthesizer.bijection, synthesizer.dense_size

    def name(self) -> str:
        """Return the algorithm identifier string."""
        return "sparse_isometry_binary_encoding"
