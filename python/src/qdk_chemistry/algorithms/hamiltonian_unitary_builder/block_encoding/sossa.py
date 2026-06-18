r"""QDK/Chemistry implementation of the SOSSA (Sum of Squares with Ancilla) block encoding.

The SOSSA block encoding implements the walk operator from the DFTHC
(Density-Fitted Tensor Hypercontraction) decomposition for quantum chemistry
Hamiltonians, as described in arXiv:2502.15882v1 (Low et al. 2025).

The walk operator is:
    W = Ref_{a,B} · U† · Ref_B · U
where
    U = OuterPREP · within{InnerPREP} apply{SELECT}

References:
    Low, G. H. et al. "Quantum simulation of chemistry with sublinear scaling
    in basis size." arXiv:2502.15882v1 (2025).

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import atan2, ceil, log2, sqrt

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
            "quantum_walk",
            "bool",
            True,
            "If True, wrap block encoding with quantum walk operator (use with QPE).",
        )
        self._set_default(
            "tolerance",
            "float",
            1e-12,
            "Minimum normalization below which the SOSSA decomposition is ill-defined.",
        )


class SOSSABuilder(HamiltonianUnitaryBuilder):
    r"""SOSSA (Sum of Squares with Ancilla) block encoding builder.

    Constructs a SOSSA block encoding from a FactorizedHamiltonianContainer.
    Unlike LCU which operates on a QubitHamiltonian (Pauli decomposition),
    SOSSA takes the molecular factorized Hamiltonian directly.

    """

    def __init__(
        self,
        power: int = 1,
        quantum_walk: bool = True,
    ):
        r"""Initialize the SOSSA builder.

        Args:
            power: The power to raise the walk operator to. Defaults to 1.
            quantum_walk: If True, produce a quantum walk operator for QPE.
                Defaults to True.

        """
        super().__init__()
        self._settings = SOSSASettings()
        self._settings.set("power", power)
        self._settings.set("quantum_walk", quantum_walk)

    def _run_impl(self, factorized_hamiltonian: FactorizedHamiltonianContainer) -> UnitaryRepresentation:
        """Build the SOSSA block encoding from a factorized Hamiltonian.

        Args:
            factorized_hamiltonian: The factorized Hamiltonian container with
                U, W, WB matrices and metadata.

        Returns:
            UnitaryRepresentation wrapping the SOSSAContainer.

        """
        # Extract dimensions
        N = factorized_hamiltonian.get_num_orbitals()
        R = factorized_hamiltonian.get_num_ranks()
        B = factorized_hamiltonian.get_num_bases()
        C = factorized_hamiltonian.get_num_copies()

        # Extract tensors and reshape from flat storage
        u_flat = np.array(factorized_hamiltonian.get_u_matrices())  # [R*B*N]
        w_flat = np.array(factorized_hamiltonian.get_w_matrices())  # [R*B*C]
        wb = np.array(factorized_hamiltonian.get_wb_matrix())       # [R, C]

        basis_vectors = u_flat.reshape(R, B, N)     # U[r, b, p]
        two_body_weights = w_flat.reshape(R, B, C)  # W[r, b, c]

        # Compute adjusted one-body matrix (Majorana representation)
        h1_majorana = np.array(factorized_hamiltonian.get_h1_majorana())  # [N, N]

        # Eigendecompose one-body matrix
        w_plus, w_minus, u_plus, u_minus = self._compute_one_body_weights(h1_majorana)
        num_d1 = len(w_plus)

        # Reshape for coefficient computation: [R][C][B] and [R][C]
        tbw: list[list[list[float]]] = []
        for r in range(R):
            r_list: list[list[float]] = []
            for c in range(C):
                r_list.append([float(two_body_weights[r, b, c]) for b in range(B)])
            tbw.append(r_list)

        iw: list[list[float]] = []
        for r in range(R):
            iw.append([float(wb[r, c]) for c in range(C)])

        # Compute PREPARE coefficients
        outer_coeffs = self._compute_outer_coefficients(
            w_plus.tolist(), w_minus.tolist(), tbw, iw, N, R, C
        )
        inner_coeffs = self._compute_inner_coefficients(tbw, iw, N, R, C, B)

        # Compute rotation angles
        dq_angles, sf_angles = self._compute_rotation_angles(
            u_plus.T, u_minus.T, basis_vectors, num_d1, N, R, B
        )

        # Compute normalization
        outer_arr = np.array(outer_coeffs)
        inner_arr = np.array(inner_coeffs)
        normalization = self._compute_normalization(outer_arr, inner_arr)

        # Build sub-oracles
        Xo = N + R * C
        num_outer_qubits = int(ceil(log2(Xo))) if Xo > 1 else 1
        num_inner_qubits = int(ceil(log2(B + 1))) if B + 1 > 1 else 1

        outer_prepare = self._build_outer_prepare(outer_arr, num_outer_qubits)

        inner_prepare = SOSSAInnerPrepare(
            conditional_coefficients=inner_arr,
            num_inner_qubits=num_inner_qubits,
            num_bases=B,
        )

        select = SOSSASelect(
            rotation_angles=np.array(dq_angles),
            sf_rotation_angles=np.array(sf_angles),
            num_orbitals=N,
            num_ranks=R,
            num_copies=C,
            num_bases=B,
            num_d1=num_d1,
        )

        container = SOSSAContainer(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=normalization,
            power=self._settings.get("power"),
            quantum_walk=self._settings.get("quantum_walk"),
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
    # Circuit synthesis helpers (from arXiv:2502.15882, Appendix B)
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
        N: int,
        R: int,
        C: int,
    ) -> list[float]:
        r"""Compute the outer PREP amplitudes for the :math:`x_o` register.

        Reference: Eq. 84-87, B9 in arXiv:2502.15882v1.
        """
        Xo = N + R * C
        coefficients = [0.0] * Xo

        idx = 0
        for coeff in one_body_weights_plus:
            coefficients[idx] = sqrt(2.0 * abs(coeff))
            idx += 1
        for coeff in one_body_weights_minus:
            coefficients[idx] = sqrt(2.0 * abs(coeff))
            idx += 1
        for r in range(R):
            for c in range(C):
                wb = abs(identity_weight[r][c])
                ws = sum(abs(w) for w in two_body_weights[r][c])
                coefficients[idx] = (wb + ws) / sqrt(2)
                idx += 1

        return coefficients

    @staticmethod
    def _compute_inner_coefficients(
        two_body_weights: list[list[list[float]]],
        identity_weight: list[list[float]],
        N: int,
        R: int,
        C: int,
        B: int,
    ) -> list[list[float]]:
        r"""Compute the inner PREP amplitudes for the b register.

        Shape: [Xo][B+1]. Reference: Appendix B.4 in arXiv:2502.15882v1.
        """
        Xo = N + R * C
        coefficients: list[list[float]] = []

        for _ in range(N):
            row = [0.0] * (B + 1)
            row[0] = 1.0
            coefficients.append(row)

        for r in range(R):
            for c in range(C):
                row = [0.0] * (B + 1)
                for b in range(B):
                    row[b] = float(two_body_weights[r][c][b])
                row[B] = float(abs(identity_weight[r][c]))
                coefficients.append(row)

        assert len(coefficients) == Xo
        return coefficients

    @staticmethod
    def _compute_rotation_angles(
        one_body_basis_plus: np.ndarray,
        one_body_basis_minus: np.ndarray,
        basis_vectors: np.ndarray,
        num_one_body_plus: int,
        N: int,
        R: int,
        B: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        r"""Compute Givens rotation angles for D1/Q1 and SF generators.

        Returns:
            (dq_angles, sf_angles) where:
                dq_angles: shape [N][N-1], D1 then Q1 angles.
                sf_angles: shape [R*(B+1)][N], SF Givens angles + bEqB flag.

        Reference: Appendix B.5, Eq. 115 in arXiv:2502.15882v1.
        """
        dq_angles: list[list[float]] = []
        for x_o in range(N):
            if x_o < num_one_body_plus:
                angles = SOSSABuilder._vector_to_givens_angles(one_body_basis_plus[x_o])
            else:
                angles = SOSSABuilder._vector_to_givens_angles(
                    one_body_basis_minus[x_o - num_one_body_plus]
                )
            dq_angles.append(angles)

        sf_angles_3d: list[list[list[float]]] = []
        for r in range(R):
            b_entries: list[list[float]] = []
            for b in range(B + 1):
                if b < B:
                    angles = SOSSABuilder._vector_to_givens_angles(basis_vectors[r, b])
                else:
                    angles = [0.0] * (N - 1)
                angles.append(1.0 if b == B else 0.0)
                b_entries.append(angles)
            sf_angles_3d.append(b_entries)

        # Flatten: iterate b in outer loop, r in inner (Q# QROM addressing).
        n_r = len(sf_angles_3d)
        n_bp1 = len(sf_angles_3d[0]) if n_r > 0 else 0
        sf_flat = [sf_angles_3d[r][b] for b in range(n_bp1) for r in range(n_r)]

        return dq_angles, sf_flat

    @staticmethod
    def _vector_to_givens_angles(vec: np.ndarray) -> list[float]:
        """Convert a unit vector to Givens rotation angles via bottom-up elimination."""
        N = len(vec)
        v = vec.copy().astype(float)
        angles: list[float] = [0.0] * (N - 1)
        for j in range(N - 2, -1, -1):
            angles[j] = float(atan2(v[j + 1], v[j]))
            v[j] = float(np.sqrt(v[j] ** 2 + v[j + 1] ** 2))
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

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "hamiltonian_unitary_builder"
