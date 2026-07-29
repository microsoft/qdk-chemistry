"""SOSSA qubit mapper for factorized SOS Hamiltonians."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import sqrt

import numpy as np

from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import FactorizedHamiltonianContainer, Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers import SOSSAContainer

_INV_SQRT_TWO = 1.0 / sqrt(2.0)


class SOSSAQubitMapper(QubitMapper):
    """Map a factorized Hamiltonian to a SOSSA qubit operator.

    The mapper computes the SOSSA block-encoding data directly: the outer and
    inner LCU coefficients, the D1/Q1 and spin-free Givens rotation angles, and
    the block-encoding normalization. These are stored in the
    :class:`SOSSAContainer` for the circuit builder to consume.
    """

    def name(self) -> str:
        """Return the algorithm variant name."""
        return "sossa"

    def _run_impl(self, hamiltonian: Hamiltonian, mapping: MajoranaMapping) -> QubitOperator:
        """Convert a factorized Hamiltonian to a structured SOS qubit operator."""
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("SOSSAQubitMapper requires a Hamiltonian")
        container = hamiltonian.get_container()
        if not isinstance(container, FactorizedHamiltonianContainer):
            raise TypeError("SOSSAQubitMapper requires a Hamiltonian backed by FactorizedHamiltonianContainer")
        return self._map_factorized_container(container, mapping)

    @classmethod
    def _map_factorized_container(
        cls,
        container: FactorizedHamiltonianContainer,
        mapping: MajoranaMapping,
    ) -> QubitOperator:
        """Map a validated factorized container to a SOSSA qubit operator."""
        num_orbitals = container.get_num_orbitals()
        if mapping.base_encoding != "jordan-wigner":
            raise ValueError("SOSSAQubitMapper requires a Jordan-Wigner mapping")
        if not mapping.is_majorana_atomic or mapping.num_modes != 2 * num_orbitals:
            raise ValueError("mapping must provide atomic Majoranas for 2N spin orbitals")
        if mapping.tapering is not None:
            raise ValueError("SOSSAQubitMapper does not support tapered mappings")

        num_ranks = container.get_num_ranks()
        num_bases = container.get_num_bases()
        num_copies = container.get_num_copies()
        u_matrices = np.asarray(container.get_u_matrices(), dtype=float)
        weights = np.asarray(container.get_w_matrices(), dtype=float)
        identity_weights = np.asarray(container.get_wb_matrix(), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(container.get_h1_majorana(), dtype=float))

        # The spin-free bilinear coefficient is common to both spin channels.
        sf_coefficient = float(mapping.bilinear(1, 0)[0].real)

        # One-body (D1 then Q1) outer/inner coefficients and rotation vectors.
        one_body_vectors: list[np.ndarray] = []
        outer_coefficients: list[float] = []
        inner_coefficients: list[list[float]] = []
        one_norm_squares = 0.0
        num_positive = 0
        for positive in (True, False):
            for index, eigenvalue in enumerate(eigenvalues):
                if (positive and eigenvalue <= 0.0) or (not positive and eigenvalue >= 0.0):
                    continue
                weight = abs(float(eigenvalue))
                one_body_vectors.append(np.asarray(eigenvectors[:, index], dtype=float))
                outer_coefficients.append(sqrt(2.0 * weight))
                inner_coefficients.append([1.0] + [0.0] * num_bases)
                one_norm_squares += 2.0 * weight  # two spin channels, each one-norm^2 == weight
                num_positive += positive

        # Spin-free rotation modes (per rank, per basis).
        basis_vectors = np.empty((num_ranks, num_bases, num_orbitals))
        for rank in range(num_ranks):
            for basis in range(num_bases):
                offset = (rank * num_bases + basis) * num_orbitals
                basis_vectors[rank, basis] = u_matrices[offset : offset + num_orbitals]

        # Spin-free outer/inner coefficients, ordered rank-major then copy-minor.
        for rank in range(num_ranks):
            for copy in range(num_copies):
                identity_weight = abs(float(identity_weights[rank, copy]))
                basis_weights = [
                    float(weights[(rank * num_bases + basis) * num_copies + copy]) for basis in range(num_bases)
                ]
                row = [weight * sf_coefficient for weight in basis_weights]
                one_norm = _INV_SQRT_TWO * (identity_weight + abs(sf_coefficient) * sum(abs(w) for w in basis_weights))
                outer_coefficients.append(one_norm)
                inner_coefficients.append([*row, identity_weight])
                one_norm_squares += one_norm**2

        one_body_rotation_angles, two_body_rotation_angles = cls._compute_rotation_angles(
            np.asarray(one_body_vectors), basis_vectors, num_orbitals, num_ranks, num_bases
        )

        negative_sum = float(-np.sum(eigenvalues[eigenvalues < 0.0]))
        w0_square_sum = 0.0
        for rank in range(num_ranks):
            for copy in range(num_copies):
                w0 = identity_weights[rank, copy]
                for basis in range(num_bases):
                    w0 -= weights[(rank * num_bases + basis) * num_copies + copy]
                w0_square_sum += w0 * w0
        energy_shift = (
            container.get_core_energy() + container.get_bliss_shift() - 2.0 * negative_sum - 0.5 * w0_square_sum
        )

        return QubitOperator(
            SOSSAContainer(
                num_orbitals,
                mapping.num_qubits,
                energy_shift,
                0.5 * one_norm_squares,
                num_ranks,
                num_bases,
                num_copies,
                num_positive,
                np.asarray(outer_coefficients),
                np.asarray(inner_coefficients),
                one_body_rotation_angles,
                two_body_rotation_angles,
                mapping.name,
                "blocked",
            )
        )

    @classmethod
    def _compute_rotation_angles(
        cls,
        one_body_vectors: np.ndarray,
        basis_vectors: np.ndarray,
        num_orbitals: int,
        num_ranks: int,
        num_bases: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute Givens rotation angles for the D1/Q1 and spin-free generators.

        Returns:
            ``(one_body_angles, two_body_angles)`` where ``one_body_angles`` has
            shape :math:`[N, N-1]` (D1 then Q1) and ``two_body_angles`` has shape
            :math:`[R \cdot (B+1), N]` (Givens angles plus a trailing ``b == B`` flag).

        Reference: Appendix B.5, Eq. 115 in :cite:`Low2025`.

        """
        one_body_angles = cls._batch_vector_to_givens_angles(one_body_vectors)

        n_bp1 = num_bases + 1
        n_sf = num_ranks * n_bp1
        sf_vectors = np.zeros((n_sf, num_orbitals))
        for rank in range(num_ranks):
            sf_vectors[rank * n_bp1 : rank * n_bp1 + num_bases] = basis_vectors[rank, :num_bases]
        sf_angles = cls._batch_vector_to_givens_angles(sf_vectors)

        b_eq_b_flags = np.zeros(n_sf)
        for rank in range(num_ranks):
            b_eq_b_flags[rank * n_bp1 + num_bases] = 1.0
        sf_with_flag = np.column_stack([sf_angles, b_eq_b_flags])

        # Reorder to basis-major, rank-minor addressing for the Q# QROM.
        flat_indices = [rank * n_bp1 + basis for basis in range(n_bp1) for rank in range(num_ranks)]
        return one_body_angles, sf_with_flag[flat_indices]

    @staticmethod
    def _batch_vector_to_givens_angles(vectors: np.ndarray) -> np.ndarray:
        """Convert unit vectors to Givens rotation angles via batch bottom-up elimination.

        Args:
            vectors: Array of shape ``[M, N]``.

        Returns:
            Angles of shape ``[M, N-1]``.

        """
        n = vectors.shape[1]
        v = vectors.copy()
        angles = np.empty((vectors.shape[0], n - 1))
        for j in range(n - 2, -1, -1):
            angles[:, j] = np.arctan2(v[:, j + 1], v[:, j])
            v[:, j] = np.hypot(v[:, j], v[:, j + 1])
        return angles
