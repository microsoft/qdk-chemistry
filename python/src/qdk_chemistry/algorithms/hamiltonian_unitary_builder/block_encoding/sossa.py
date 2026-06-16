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

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    HamiltonianUnitaryBuilder,
    HamiltonianUnitaryBuilderSettings,
)
from qdk_chemistry.data import QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAContainer,
    SOSSAInnerPrepare,
    SOSSAOuterPrepare,
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

    Constructs a SOSSA block encoding from pre-computed DFTHC tensors.
    Unlike LCU which operates on a QubitHamiltonian (Pauli decomposition),
    SOSSA takes the molecular DFTHC factorization directly.

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

    def build_from_dfthc(
        self,
        *,
        num_orbitals: int,
        num_ranks: int,
        num_bases: int,
        num_copies: int,
        outer_coefficients: np.ndarray,
        inner_coefficients: np.ndarray,
        dq_rotation_angles: np.ndarray,
        sf_rotation_angles: np.ndarray,
        num_d1: int,
    ) -> UnitaryRepresentation:
        r"""Build the SOSSA block encoding from DFTHC tensors.

        This is the primary entry point for constructing the SOSSA block encoding
        from the classical preprocessing pipeline output.

        Args:
            num_orbitals: Number of spatial orbitals N.
            num_ranks: Number of DFTHC ranks R.
            num_bases: Number of bases B.
            num_copies: Number of copies C.
            outer_coefficients: Outer PREP amplitudes, length :math:`X_o = N + R \cdot C`.
            inner_coefficients: Inner PREP amplitudes, shape :math:`[X_o, B+1]`.
            dq_rotation_angles: D1/Q1 Givens angles, shape :math:`[N, N-1]`.
            sf_rotation_angles: SF Givens angles, shape :math:`[R \cdot (B+1), N-1]`.
            num_d1: Number of D1 entries in the outer register.

        Returns:
            UnitaryRepresentation wrapping the SOSSAContainer.

        """
        power: int = self._settings.get("power")
        quantum_walk: bool = self._settings.get("quantum_walk")
        tolerance: float = self._settings.get("tolerance")

        outer_coefficients = np.asarray(outer_coefficients, dtype=float)
        inner_coefficients = np.asarray(inner_coefficients, dtype=float)
        dq_rotation_angles = np.asarray(dq_rotation_angles, dtype=float)
        sf_rotation_angles = np.asarray(sf_rotation_angles, dtype=float)

        xo = num_orbitals + num_ranks * num_copies
        if len(outer_coefficients) != xo:
            raise ValueError(
                f"outer_coefficients length {len(outer_coefficients)} != X_o = N + R*C = {xo}"
            )
        if inner_coefficients.shape != (xo, num_bases + 1):
            raise ValueError(
                f"inner_coefficients shape {inner_coefficients.shape} != ({xo}, {num_bases + 1})"
            )

        # Compute normalization Lambda = (1/2) * lambda_sqrt^2
        normalization = self._compute_normalization(outer_coefficients, inner_coefficients)
        if normalization < tolerance:
            raise ValueError("Normalization is too small, cannot build SOSSA block encoding.")

        # Normalize outer coefficients for state preparation
        l1_outer = float(np.sum(np.abs(outer_coefficients)))
        outer_statevector = np.sqrt(np.abs(outer_coefficients) / l1_outer)

        num_outer_qubits = int(np.ceil(np.log2(xo))) if xo > 1 else 1
        num_inner_qubits = int(np.ceil(np.log2(num_bases + 1))) if num_bases > 0 else 1

        outer_prepare = SOSSAOuterPrepare(
            statevector=outer_statevector,
            num_outer_qubits=num_outer_qubits,
        )
        inner_prepare = SOSSAInnerPrepare(
            conditional_coefficients=inner_coefficients,
            num_inner_qubits=num_inner_qubits,
            num_bases=num_bases,
        )
        select = SOSSASelect(
            rotation_angles=dq_rotation_angles,
            sf_rotation_angles=sf_rotation_angles,
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_copies=num_copies,
            num_bases=num_bases,
            num_d1=num_d1,
        )

        container = SOSSAContainer(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=normalization,
            power=power,
            quantum_walk=quantum_walk,
        )

        return UnitaryRepresentation(container=container)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Not supported for SOSSA — use build_from_dfthc() instead.

        SOSSA requires DFTHC tensor data, not a Pauli-decomposed Hamiltonian.

        Raises:
            NotImplementedError: Always. Use build_from_dfthc() instead.

        """
        raise NotImplementedError(
            "SOSSABuilder does not support building from a QubitHamiltonian. "
            "Use build_from_dfthc() with pre-computed DFTHC tensors."
        )

    @staticmethod
    def _compute_normalization(
        outer_coefficients: np.ndarray,
        inner_coefficients: np.ndarray,
    ) -> float:
        r"""Compute the SOSSA normalization :math:`\Lambda`.

        .. math::

            \Lambda = \frac{1}{2} \lambda_\text{sqrt}^2

        where :math:`\lambda_\text{sqrt} = \sum_{x_o} |\alpha_{x_o}| \cdot (\sum_b |\beta_{x_o,b}|)`.

        Args:
            outer_coefficients: Outer amplitudes of length X_o.
            inner_coefficients: Inner amplitudes of shape [X_o, B+1].

        Returns:
            The normalization constant Lambda.

        """
        # lambda_sqrt = sum over x_o of (outer_coeff[xo] * sum_b inner_coeff[xo, b])
        inner_l1 = np.sum(np.abs(inner_coefficients), axis=1)  # shape [Xo]
        lambda_sqrt = float(np.sum(np.abs(outer_coefficients) * inner_l1))
        return 0.5 * lambda_sqrt**2

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "hamiltonian_unitary_builder"
