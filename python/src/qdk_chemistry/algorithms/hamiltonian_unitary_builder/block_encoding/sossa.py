r"""QDK/Chemistry implementation of the SOSSA (Sum of Squares Spectral Amplification) block encoding.

The SOSSA block encoding implements the walk operator from the DFTHC
(Density-Fitted Tensor Hypercontraction) decomposition for quantum chemistry
Hamiltonians, as described in :cite:`Low2025`.

The walk operator is:
    W = Ref_{a,B} · U† · Ref_B · U
where
    U = OuterPREP · within{InnerPREP} apply{SELECT}
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import ceil, log2, sqrt

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    HamiltonianUnitaryBuilder,
    HamiltonianUnitaryBuilderSettings,
)
from qdk_chemistry.data import (
    Configuration,
    FactorizedHamiltonianContainer,
    ModelOrbitals,
    StateVectorContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAContainer,
    SOSSAInnerPrepare,
    SOSSASelect,
)

__all__: list[str] = ["SOSSABuilder", "SOSSASettings"]


class SOSSASettings(HamiltonianUnitaryBuilderSettings):
    """Settings for the SOSSA block encoding builder."""

    def __init__(self):
        """Initialize SOSSASettings with default values."""
        super().__init__()
        self._set_default(
            "tolerance",
            "float",
            1e-12,
            "Minimum normalization below which the SOSSA decomposition is ill-defined.",
        )


class SOSSABuilder(HamiltonianUnitaryBuilder):
    r"""SOSSA (Sum of Squares Spectral Amplification) block encoding builder.

    Constructs a SOSSA block encoding from a FactorizedHamiltonianContainer.
    Unlike LCU which operates on a QubitHamiltonian (Pauli decomposition),
    SOSSA takes the molecular factorized Hamiltonian directly.

    """

    def __init__(
        self,
        power: int = 1,
    ):
        r"""Initialize the SOSSA builder.

        Args:
            power: The power to raise the walk operator to. Defaults to 1.

        """
        super().__init__()
        self._settings = SOSSASettings()
        self._settings.set("power", power)

    def _run_impl(self, factorized_hamiltonian: FactorizedHamiltonianContainer) -> UnitaryRepresentation:
        """Build the SOSSA block encoding from a factorized Hamiltonian.

        Args:
            factorized_hamiltonian: The factorized Hamiltonian container with
                U, W, WB matrices and metadata.

        Returns:
            UnitaryRepresentation wrapping the SOSSAContainer.

        """
        # Extract dimensions
        n_orbitals = factorized_hamiltonian.get_num_orbitals()
        n_ranks = factorized_hamiltonian.get_num_ranks()
        n_bases = factorized_hamiltonian.get_num_bases()
        n_copies = factorized_hamiltonian.get_num_copies()

        # Extract tensors and reshape from flat storage
        u_flat = np.array(factorized_hamiltonian.get_u_matrices())  # [R*B*N]
        w_flat = np.array(factorized_hamiltonian.get_w_matrices())  # [R*B*C]
        wb = np.array(factorized_hamiltonian.get_wb_matrix())  # [R, C]

        basis_vectors = u_flat.reshape(n_ranks, n_bases, n_orbitals)  # U[r, b, p]
        two_body_weights = w_flat.reshape(n_ranks, n_bases, n_copies)  # W[r, b, c]

        # Compute adjusted one-body matrix (Majorana representation)
        h1_majorana = np.array(factorized_hamiltonian.get_h1_majorana())  # [N, N]

        # Eigendecompose one-body matrix
        w_plus, w_minus, u_plus, u_minus = self._compute_one_body_weights(h1_majorana)
        num_d1 = len(w_plus)

        # Reshape for coefficient computation: [R][C][B] and [R][C]
        tbw: list[list[list[float]]] = []
        for r in range(n_ranks):
            r_list: list[list[float]] = []
            for c in range(n_copies):
                r_list.append([float(two_body_weights[r, b, c]) for b in range(n_bases)])
            tbw.append(r_list)

        iw: list[list[float]] = []
        for r in range(n_ranks):
            iw.append([float(wb[r, c]) for c in range(n_copies)])

        # Compute PREPARE coefficients
        outer_coeffs = self._compute_outer_coefficients(
            w_plus.tolist(), w_minus.tolist(), tbw, iw, n_orbitals, n_ranks, n_copies
        )
        inner_coeffs = self._compute_inner_coefficients(tbw, iw, n_orbitals, n_ranks, n_copies, n_bases)

        # Compute rotation angles
        dq_angles, sf_angles = self._compute_rotation_angles(
            u_plus.T, u_minus.T, basis_vectors, num_d1, n_orbitals, n_ranks, n_bases
        )

        # Compute free-rider data (G, r encoding for QROM)
        free_rider = self._compute_free_rider_data(num_d1, n_orbitals, n_ranks, n_copies)

        # Compute normalization
        outer_arr = np.array(outer_coeffs)
        inner_arr = np.array(inner_coeffs)
        normalization = self._compute_normalization(outer_arr, inner_arr)

        # Build sub-oracles
        xo_dim = n_orbitals + n_ranks * n_copies
        num_outer_qubits = ceil(log2(xo_dim)) if xo_dim > 1 else 1
        num_inner_qubits = ceil(log2(n_bases + 1)) if n_bases + 1 > 1 else 1

        outer_prepare = self._build_outer_prepare(outer_arr, num_outer_qubits)

        inner_prepare = SOSSAInnerPrepare(
            conditional_coefficients=inner_arr,
            num_inner_qubits=num_inner_qubits,
            num_bases=n_bases,
            free_rider_data=np.array(free_rider, dtype=bool) if free_rider else None,
        )

        select = SOSSASelect(
            one_body_rotation_angles=np.array(dq_angles),
            two_body_rotation_angles=np.array(sf_angles),
            num_orbitals=n_orbitals,
            num_ranks=n_ranks,
            num_copies=n_copies,
            num_bases=n_bases,
            num_positive_one_body_terms=num_d1,
        )

        # Energy shift: E_SOS + E_nuc for full energy recovery (Eq. 29, :cite:`Low2025`)
        # E_SOS = -2·Σw⁻ - ½·Σ|W₀|² + E_BLISS
        w0 = wb - np.sum(two_body_weights, axis=1)  # [R, C]
        e_sos = -2.0 * np.sum(w_minus) - 0.5 * float(np.sum(w0**2)) + factorized_hamiltonian.get_bliss_core_shift()
        energy_shift = e_sos + factorized_hamiltonian.get_core_energy()

        container = SOSSAContainer(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=normalization,
            power=self._settings.get("power"),
            energy_shift=energy_shift,
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _build_outer_prepare(statevector: np.ndarray, num_qubits: int) -> Wavefunction:
        """Build a Wavefunction encoding the outer PREPARE statevector.

        Args:
            statevector: Array of amplitudes for the outer PREPARE oracle.
            num_qubits: Number of qubits in the prepare register.

        Returns:
            Wavefunction whose coefficients encode the outer PREPARE amplitudes.

        """
        coeffs_list: list[float] = []
        dets: list[Configuration] = []
        for idx, amp in enumerate(statevector):
            if amp != 0.0:
                bitstring = format(idx, f"0{num_qubits}b")
                dets.append(Configuration.from_bitstring(bitstring))
                coeffs_list.append(float(amp))
        orbitals = ModelOrbitals(num_qubits)
        coeffs_arr = np.array(coeffs_list)
        norm = np.linalg.norm(coeffs_arr)
        if norm > 0:
            coeffs_arr = coeffs_arr / norm
        container = StateVectorContainer(coeffs_arr, dets, orbitals)
        return Wavefunction(container)

    # =========================================================================
    # Circuit synthesis helpers
    # =========================================================================

    @staticmethod
    def _compute_one_body_weights(
        h1_majorana: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Eigendecompose the adjusted one-body matrix into D1/Q1 generators.

        Args:
            h1_majorana: Adjusted one-body matrix h'(1), shape [N, N].

        Returns:
            (w_plus, w_minus, u_plus, u_minus) where:
                w_plus: positive eigenvalues [N_D1]
                w_minus: absolute negative eigenvalues [N_Q1]
                u_plus: eigenvectors for positive eigenvalues [N, N_D1]
                u_minus: eigenvectors for negative eigenvalues [N, N_Q1]

        """
        eigvals, eigvecs = np.linalg.eigh(h1_majorana)
        pos_mask = eigvals > 0
        neg_mask = eigvals < 0
        return (
            eigvals[pos_mask],
            -eigvals[neg_mask],
            eigvecs[:, pos_mask],
            eigvecs[:, neg_mask],
        )

    @staticmethod
    def _compute_outer_coefficients(
        one_body_weights_plus: list[float],
        one_body_weights_minus: list[float],
        two_body_weights: list[list[list[float]]],
        identity_weight: list[list[float]],
        n_orbitals: int,
        n_ranks: int,
        n_copies: int,
    ) -> list[float]:
        r"""Compute the outer PREP amplitudes for the :math:`x_o` register.

        Reference: Eq. 84-87, B9 in :cite:`Low2025`.
        """
        xo_dim = n_orbitals + n_ranks * n_copies
        coefficients = [0.0] * xo_dim

        idx = 0
        for coeff in one_body_weights_plus:
            coefficients[idx] = sqrt(2.0 * abs(coeff))
            idx += 1
        for coeff in one_body_weights_minus:
            coefficients[idx] = sqrt(2.0 * abs(coeff))
            idx += 1
        for r in range(n_ranks):
            for c in range(n_copies):
                wb = abs(identity_weight[r][c])
                ws = sum(abs(w) for w in two_body_weights[r][c])
                coefficients[idx] = (wb + ws) / sqrt(2)
                idx += 1

        return coefficients

    @staticmethod
    def _compute_inner_coefficients(
        two_body_weights: list[list[list[float]]],
        identity_weight: list[list[float]],
        n_orbitals: int,
        n_ranks: int,
        n_copies: int,
        n_bases: int,
    ) -> list[list[float]]:
        r"""Compute the inner PREP amplitudes for the b register.

        Shape: [Xo][B+1]. Reference: Appendix B.4 in :cite:`Low2025`.
        """
        xo_dim = n_orbitals + n_ranks * n_copies
        coefficients: list[list[float]] = []

        for _ in range(n_orbitals):
            row = [0.0] * (n_bases + 1)
            row[0] = 1.0
            coefficients.append(row)

        for r in range(n_ranks):
            for c in range(n_copies):
                row = [0.0] * (n_bases + 1)
                for b in range(n_bases):
                    row[b] = float(two_body_weights[r][c][b])
                row[n_bases] = float(abs(identity_weight[r][c]))
                coefficients.append(row)

        assert len(coefficients) == xo_dim
        return coefficients

    @staticmethod
    def _compute_rotation_angles(
        one_body_basis_plus: np.ndarray,
        one_body_basis_minus: np.ndarray,
        basis_vectors: np.ndarray,
        num_one_body_plus: int,
        n_orbitals: int,
        n_ranks: int,
        n_bases: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        r"""Compute Givens rotation angles for D1/Q1 and SF generators.

        Returns:
            (dq_angles, sf_angles) where:
                dq_angles: shape [N][N-1], D1 then Q1 angles.
                sf_angles: shape [R*(B+1)][N], SF Givens angles + bEqB flag.

        Reference: Appendix B.5, Eq. 115 in :cite:`Low2025`.

        """
        # Stack all DQ vectors into a single [N, n_orbitals] matrix for batch processing
        num_q1 = n_orbitals - num_one_body_plus
        dq_vectors = np.empty((n_orbitals, n_orbitals))
        if num_one_body_plus > 0:
            dq_vectors[:num_one_body_plus] = one_body_basis_plus[:num_one_body_plus]
        if num_q1 > 0:
            dq_vectors[num_one_body_plus:] = one_body_basis_minus[:num_q1]
        dq_angles_arr = SOSSABuilder._batch_vector_to_givens_angles(dq_vectors)
        dq_angles = dq_angles_arr.tolist()

        # Stack all SF vectors: R*B real vectors + R zero vectors for b==B
        n_sf = n_ranks * (n_bases + 1)
        sf_vectors = np.zeros((n_sf, n_orbitals))
        for r in range(n_ranks):
            sf_vectors[r * (n_bases + 1) : r * (n_bases + 1) + n_bases] = basis_vectors[r, :n_bases]
            # row r*(B+1)+B stays zero (b==B case)

        sf_angles_arr = SOSSABuilder._batch_vector_to_givens_angles(sf_vectors)

        # Append bEqB flag column and reorder to [b*R + r] addressing
        b_eq_b_flags = np.zeros(n_sf)
        for r in range(n_ranks):
            b_eq_b_flags[r * (n_bases + 1) + n_bases] = 1.0
        sf_with_flag = np.column_stack([sf_angles_arr, b_eq_b_flags])

        # Flatten: iterate b in outer loop, r in inner (Q# QROM addressing)
        n_bp1 = n_bases + 1
        flat_indices = [r * n_bp1 + b for b in range(n_bp1) for r in range(n_ranks)]
        sf_flat = sf_with_flag[flat_indices].tolist()

        return dq_angles, sf_flat

    @staticmethod
    def _batch_vector_to_givens_angles(vectors: np.ndarray) -> np.ndarray:
        """Convert multiple unit vectors to Givens rotation angles via batch bottom-up elimination.

        Args:
            vectors: shape [M, N] where M is number of vectors.

        Returns:
            angles: shape [M, N-1].

        """
        n = vectors.shape[1]
        v = vectors.copy()
        angles = np.empty((vectors.shape[0], n - 1))
        for j in range(n - 2, -1, -1):
            angles[:, j] = np.arctan2(v[:, j + 1], v[:, j])
            v[:, j] = np.hypot(v[:, j], v[:, j + 1])
        return angles

    @staticmethod
    def _compute_normalization(
        outer_coefficients: np.ndarray,
        inner_coefficients: np.ndarray,
    ) -> float:
        r"""Compute the SOSSA normalization :math:`\Lambda`.

        .. math::

            \Lambda = \frac{1}{2} \lambda_\text{sqrt}^2

        where :math:`\lambda_\text{sqrt} = \sum_{x_o} |\alpha_{x_o}| \cdot (\sum_b |\beta_{x_o,b}|)`.

        """
        inner_l1 = np.sum(np.abs(inner_coefficients), axis=1)
        lambda_sqrt = float(np.sum(np.abs(outer_coefficients) * inner_l1))
        return 0.5 * lambda_sqrt**2

    @staticmethod
    def _compute_free_rider_data(
        num_one_body_plus: int,
        n_orbitals: int,
        n_ranks: int,
        n_copies: int,
    ) -> list[list[bool]]:
        r"""Compute QROM free-rider data encoding (G, r) for each outer index.

        Shape: ``[Xo][2 + R_bits]``.

        Each entry ``data[x_o]`` encodes the generator type G (2 bits) and the
        rank index r in little-endian binary.

        G encoding (2 bits = ``[sf_vs_dq, d_vs_q]``):
            - D1 (particle): ``[False, False]``
            - Q1 (hole):     ``[False, True]``
            - SF (two-body): ``[True,  True]``

        Reference: Eq. 82 in :cite:`Low2025`.

        """
        xo_dim = n_orbitals + n_ranks * n_copies
        n_d1 = num_one_body_plus
        r_bits = ceil(log2(n_ranks)) if n_ranks > 1 else 0

        data: list[list[bool]] = []
        for x_o in range(xo_dim):
            if x_o < n_d1:
                g_bits = [False, False]
                r_val = 0
            elif x_o < n_orbitals:
                g_bits = [False, True]
                r_val = 0
            else:
                g_bits = [True, True]
                sf_idx = x_o - n_orbitals
                r_val = sf_idx // n_copies

            r_enc = [(r_val >> k) & 1 == 1 for k in range(r_bits)]
            data.append(g_bits + r_enc)

        return data

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "hamiltonian_unitary_builder"
