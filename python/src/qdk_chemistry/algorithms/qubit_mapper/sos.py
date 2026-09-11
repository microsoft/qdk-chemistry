"""SOSSA qubit mapper for factorized SOS Hamiltonians."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data import FactorizedHamiltonianContainer, Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.sos import (
    FactorizedHamiltonianMetadata,
    RotatedPaulis,
    SOSContainer,
)

__all__ = ["SOSQubitMapper", "SOSQubitMapperSettings"]


class SOSQubitMapperSettings(QubitMapperSettings):
    """Settings for the SOS qubit mapper."""

    def __init__(self) -> None:
        """Initialize the settings with the package-wide screening default."""
        super().__init__()
        self._set_default(
            "threshold",
            "double",
            1e-12,
            "Magnitude below which a one-body eigenvalue counts as zero. The mode keeps its outer "
            "register slot but carries no amplitude, so eigensolver noise in the nullspace cannot "
            "become a generator. Must be non-negative.",
        )


class SOSQubitMapper(QubitMapper):
    """Map a factorized Hamiltonian to a SOSSA qubit operator."""

    def __init__(self) -> None:
        """Initialize the mapper with its settings."""
        super().__init__()
        self._settings = SOSQubitMapperSettings()

    def name(self) -> str:
        """Return the algorithm variant name."""
        return "sos"

    def _run_impl(self, hamiltonian: Hamiltonian, mapping: MajoranaMapping) -> QubitOperator:
        """Convert a factorized Hamiltonian to a structured SOS qubit operator.

        Args:
            hamiltonian: The factorized Hamiltonian to map.
            mapping: The Majorana mapping to encode under. Must be an untapered
                Jordan-Wigner mapping over the Hamiltonian's spin orbitals.

        Returns:
            The sum-of-squares qubit operator.

        Raises:
            TypeError: If the Hamiltonian is not backed by a factorized container.
            ValueError: If the mapping is not one this construction can honour.

        """
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("SOSQubitMapper requires a Hamiltonian")

        container = hamiltonian.get_container()
        if not isinstance(container, FactorizedHamiltonianContainer):
            raise TypeError("SOSQubitMapper requires a Hamiltonian backed by FactorizedHamiltonianContainer")

        self._validate_mapping(mapping, container.get_num_orbitals())
        return self._map_factorized_container(
            container, MajoranaMapping.jordan_wigner(1), float(self._settings.get("threshold"))
        )

    @staticmethod
    def _validate_mapping(mapping: MajoranaMapping, num_orbitals: int) -> None:
        """Reject mappings other than blocked Jordan-Wigner over ``2 * num_orbitals`` modes.

        Args:
            mapping: The mapping supplied by the caller.
            num_orbitals: Number of spatial orbitals in the Hamiltonian.

        Raises:
            ValueError: If the encoding, mode count, or tapering is unsupported.

        """
        if mapping.name != "jordan-wigner":
            raise ValueError(f"SOSQubitMapper supports the jordan-wigner encoding only; got {mapping.name!r}.")
        num_modes = 2 * num_orbitals
        if mapping.num_modes != num_modes:
            raise ValueError(
                f"SOSQubitMapper requires a mapping over {num_modes} spin orbitals; got {mapping.num_modes}."
            )
        if mapping.tapering is not None:
            raise ValueError("SOSQubitMapper does not support tapered mappings")

    @classmethod
    def _map_factorized_container(
        cls,
        container: FactorizedHamiltonianContainer,
        mapping: MajoranaMapping,
        threshold: float = 1e-12,
    ) -> QubitOperator:
        """Map a validated factorized container to a SOSSA qubit operator.

        Args:
            container: The factorized Hamiltonian to map.
            mapping: The single-mode Majorana mapping supplying the Pauli labels.
            threshold: Magnitude below which a one-body eigenvalue counts as zero.

        Returns:
            The sum-of-squares qubit operator.

        Raises:
            ValueError: If the threshold is negative or not a number.

        """
        if not threshold >= 0.0:
            raise ValueError(f"threshold must be non-negative; got {threshold!r}")

        num_orbitals = container.get_num_orbitals()
        num_ranks = container.get_num_ranks()
        num_bases = container.get_num_bases()
        num_copies = container.get_num_copies()

        one_body, num_d1_terms, negative_eigenvalue_sum = cls._map_one_body_terms(
            np.asarray(container.get_h1_prime(), dtype=float), mapping, threshold
        )
        two_body, rank_copy_weight_square_sum = cls._map_two_body_terms(container, mapping)
        one_body_shift = 2.0 * negative_eigenvalue_sum
        two_body_shift = 0.5 * rank_copy_weight_square_sum
        energy_shift = container.get_core_energy() - one_body_shift - two_body_shift

        return QubitOperator(
            SOSContainer(
                one_body,
                two_body,
                mapping.name,
                "blocked",
                FactorizedHamiltonianMetadata(
                    num_spatial_orbitals=num_orbitals,
                    num_ranks=num_ranks,
                    num_bases=num_bases,
                    num_copies=num_copies,
                    num_positive_one_body_terms=num_d1_terms,
                    energy_shift=energy_shift,
                ),
            )
        )

    @classmethod
    def _map_one_body_terms(
        cls, h1_prime: np.ndarray, mapping: MajoranaMapping, threshold: float
    ) -> tuple[RotatedPaulis, int, float]:
        """Build D1 and Q1 generators from the effective one-body matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(h1_prime)

        positive = eigenvalues > threshold
        negative = eigenvalues < -threshold
        screened = ~(positive | negative)
        num_screened = int(screened.sum())
        num_d1_terms = int(positive.sum()) + num_screened

        # D1 rows occupy the positive side of the outer register. Screened modes stay
        # in that partition with zero amplitude, followed by the Q1 rows.
        ordered_vectors = np.concatenate(
            [eigenvectors[:, positive].T, eigenvectors[:, screened].T, eigenvectors[:, negative].T]
        )
        magnitudes = np.concatenate([eigenvalues[positive], np.zeros(num_screened), -eigenvalues[negative]])
        phase_signs = np.concatenate([np.ones(num_d1_terms), -np.ones(int(negative.sum()))])
        amplitudes = np.sqrt(magnitudes)

        angles = cls._batch_vector_to_givens_angles(ordered_vectors)
        # The Y phase distinguishes D1 (+i) from Q1 (-i); the builder later adds
        # the sqrt(2) factor for the two spin channels.
        coeffs = 0.5 * np.stack([amplitudes, 1j * phase_signs * amplitudes], axis=1)
        paulis = (
            sparse_pauli_word_to_label(mapping.majorana(0), 1),
            sparse_pauli_word_to_label(mapping.majorana(1), 1),
        )
        negative_eigenvalue_sum = float(-np.sum(eigenvalues[negative]))
        return RotatedPaulis(angles, coeffs, paulis), num_d1_terms, negative_eigenvalue_sum

    @classmethod
    def _map_two_body_terms(
        cls, container: FactorizedHamiltonianContainer, mapping: MajoranaMapping
    ) -> tuple[RotatedPaulis, float]:
        """Build spin-free generators and their contribution to the energy shift."""
        num_orbitals = container.get_num_orbitals()
        num_ranks = container.get_num_ranks()
        num_bases = container.get_num_bases()
        num_copies = container.get_num_copies()

        weights_by_rank_basis_copy = np.asarray(container.get_w_matrices(), dtype=float).reshape(
            num_ranks, num_bases, num_copies
        )
        basis_vectors = np.asarray(container.get_u_matrices(), dtype=float).reshape(num_ranks * num_bases, num_orbitals)
        identity_weights = np.asarray(container.get_wb_matrix(), dtype=float)

        bilinear_coefficient, bilinear_word = mapping.bilinear(1, 0)
        # Since n = (I - i gamma_1 gamma_0) / 2, the rotated-Z term has the
        # opposite sign from the Majorana bilinear.
        spin_free_coefficient = -float(bilinear_coefficient.real)
        spin_free_pauli = sparse_pauli_word_to_label(bilinear_word, 1)

        angles = cls._batch_vector_to_givens_angles(basis_vectors)
        # Each row represents (rank, copy), with one rotated-Z coefficient per basis.
        basis_coeffs = (
            np.transpose(weights_by_rank_basis_copy, (0, 2, 1)).reshape(num_ranks * num_copies, num_bases)
            * spin_free_coefficient
        )
        coeffs = np.concatenate([basis_coeffs, identity_weights.reshape(-1, 1)], axis=1).astype(complex)

        # These derived W^(r,c) values enter only the energy shift, not the LCU rows above.
        rank_copy_weights = identity_weights - weights_by_rank_basis_copy.sum(axis=1)
        rank_copy_weight_square_sum = float(np.sum(rank_copy_weights**2))
        return RotatedPaulis(angles, coeffs, (spin_free_pauli,)), rank_copy_weight_square_sum

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
