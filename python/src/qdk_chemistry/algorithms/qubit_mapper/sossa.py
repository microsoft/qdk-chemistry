"""SOSSA qubit mapper for factorized SOS Hamiltonians."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import FactorizedHamiltonianContainer, Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.sossa import (
    FactorizedHamiltonianMetadata,
    RotatedPaulis,
    SOSContainer,
)
from qdk_chemistry.utils import Logger

__all__ = ["SOSQubitMapper"]


class SOSQubitMapper(QubitMapper):
    """Map a factorized Hamiltonian to a SOSSA qubit operator."""

    def name(self) -> str:
        """Return the algorithm variant name."""
        return "sossa"

    def _run_impl(self, hamiltonian: Hamiltonian, _mapping: MajoranaMapping) -> QubitOperator:
        """Convert a factorized Hamiltonian to a structured SOS qubit operator."""
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("SOSQubitMapper requires a Hamiltonian")

        container = hamiltonian.get_container()
        if not isinstance(container, FactorizedHamiltonianContainer):
            raise TypeError("SOSQubitMapper requires a Hamiltonian backed by FactorizedHamiltonianContainer")

        Logger.warn("SOSQubitMapper ignores the provided mapping and uses a single-mode Jordan-Wigner encoding")
        return self._map_factorized_container(container, MajoranaMapping.jordan_wigner(1))

    @classmethod
    def _map_factorized_container(
        cls,
        container: FactorizedHamiltonianContainer,
        mapping: MajoranaMapping,
    ) -> QubitOperator:
        """Map a validated factorized container to a SOSSA qubit operator."""
        if np.any(np.asarray(container.get_signs(), dtype=float) < 0.0):
            raise ValueError(
                "SOSQubitMapper requires a positive semi-definite factorization: the container "
                "has negative per-rank signs, so it is not a sum of squares"
            )

        num_orbitals = container.get_num_orbitals()
        num_ranks = container.get_num_ranks()
        num_bases = container.get_num_bases()
        num_copies = container.get_num_copies()
        u_matrices = np.asarray(container.get_u_matrices(), dtype=float)
        weights = np.asarray(container.get_w_matrices(), dtype=float)
        identity_weights = np.asarray(container.get_wb_matrix(), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(container.get_h1_majorana(), dtype=float))

        x_pauli = sparse_pauli_word_to_label(mapping.majorana(0), 1)
        y_pauli = sparse_pauli_word_to_label(mapping.majorana(1), 1)
        sf_bilinear_coefficient, sf_word = mapping.bilinear(1, 0)
        sf_coefficient = float(sf_bilinear_coefficient.real)
        sf_pauli = sparse_pauli_word_to_label(sf_word, 1)

        pos_mask = eigenvalues > 0.0
        neg_mask = eigenvalues < 0.0
        one_body_vectors = np.concatenate([eigenvectors[:, pos_mask].T, eigenvectors[:, neg_mask].T], axis=0)
        sqrt_lambdas = np.sqrt(np.concatenate([eigenvalues[pos_mask], -eigenvalues[neg_mask]]))
        signs = np.concatenate([np.ones(int(pos_mask.sum())), -np.ones(int(neg_mask.sum()))])
        num_positive = int(pos_mask.sum())
        one_body_angles = (
            cls._batch_vector_to_givens_angles(one_body_vectors)
            if one_body_vectors.shape[0]
            else np.empty((0, max(num_orbitals - 1, 0)))
        )
        one_body_coeffs = 0.5 * np.stack([sqrt_lambdas, 1j * signs * sqrt_lambdas], axis=1)

        weights_rbc = weights.reshape(num_ranks, num_bases, num_copies)
        basis_vectors = u_matrices.reshape(num_ranks * num_bases, num_orbitals)
        two_body_angles = (
            cls._batch_vector_to_givens_angles(basis_vectors)
            if basis_vectors.shape[0]
            else np.empty((0, max(num_orbitals - 1, 0)))
        )
        sf_basis = np.transpose(weights_rbc, (0, 2, 1)).reshape(num_ranks * num_copies, num_bases) * sf_coefficient
        two_body_coeffs = np.concatenate([sf_basis, identity_weights.reshape(-1, 1)], axis=1).astype(complex)

        negative_sum = float(-np.sum(eigenvalues[neg_mask]))
        w0 = identity_weights - weights_rbc.sum(axis=1)
        w0_square_sum = float(np.sum(w0**2))
        energy_shift = container.get_core_energy() - 2.0 * negative_sum - 0.5 * w0_square_sum

        return QubitOperator(
            SOSContainer(
                RotatedPaulis(one_body_angles, one_body_coeffs, (x_pauli, y_pauli)),
                RotatedPaulis(two_body_angles, two_body_coeffs, (sf_pauli,)),
                mapping.name,
                "blocked",
                FactorizedHamiltonianMetadata(
                    num_spatial_orbitals=num_orbitals,
                    num_ranks=num_ranks,
                    num_bases=num_bases,
                    num_copies=num_copies,
                    num_positive_one_body_terms=num_positive,
                    energy_shift=energy_shift,
                ),
            )
        )

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
