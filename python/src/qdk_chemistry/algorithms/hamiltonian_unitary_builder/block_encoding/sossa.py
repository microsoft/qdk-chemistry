r"""QDK/Chemistry implementation of the SOSSA (Sum of Squares Spectral Amplification) block encoding."""

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
    ModelOrbitals,
    QubitOperator,
    SOSSAContainer,
    StateVectorContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAInnerPrepare,
    SOSSASelect,
    SOSSAWalkContainer,
)

__all__: list[str] = ["SOSSABuilder", "SOSSASettings"]

_SQRT_TWO = sqrt(2.0)
_INV_SQRT_TWO = 1.0 / sqrt(2.0)


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
    """SOSSA (Sum of Squares Spectral Amplification) block encoding builder."""

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

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        """Build the SOSSA block encoding from qubit operator.

        Args:
            qubit_hamiltonian: Qubit operator with SOSSAContainer.

        Returns:
            UnitaryRepresentation wrapping the SOSSAWalkContainer.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("SOSSABuilder requires a QubitOperator containing an SOSSAContainer")
        sossa = qubit_hamiltonian.get_container()
        if not isinstance(sossa, SOSSAContainer):
            raise TypeError("SOSSABuilder requires a QubitOperator containing an SOSSAContainer")
        if sossa.encoding != "jordan-wigner" or sossa.fermion_mode_order != "blocked":
            raise ValueError("the SOSSA circuit builder currently supports blocked Jordan-Wigner operators only")

        n_orbitals = sossa.num_spatial_orbitals
        num_positive = sossa.num_positive_one_body_terms

        outer_coefficients = self._outer_coefficients(sossa)
        normalization = 0.5 * float(np.sum(outer_coefficients**2))
        if normalization <= self._settings.get("tolerance"):
            raise ValueError("the SOSSA operator normalization is below the configured tolerance")

        one_body_rotation_angles = np.array([*sossa.d1.rotations, *sossa.q1.rotations], dtype=float)
        two_body_rotation_angles = self._two_body_rotation_angles(
            sossa.sf.rotations, sossa.num_ranks, sossa.num_bases, n_orbitals
        )

        xo_dim = n_orbitals + sossa.num_ranks * sossa.num_copies
        num_outer_qubits = ceil(log2(xo_dim)) if xo_dim > 1 else 1
        num_inner_qubits = ceil(log2(sossa.num_bases + 1)) if sossa.num_bases + 1 > 1 else 1

        free_rider = self._compute_free_rider_data(num_positive, n_orbitals, sossa.num_ranks, sossa.num_copies)

        container = SOSSAWalkContainer(
            outer_prepare=self._build_outer_prepare(outer_coefficients, num_outer_qubits),
            inner_prepare=SOSSAInnerPrepare(
                conditional_coefficients=sossa.inner_coefficients,
                num_inner_qubits=num_inner_qubits,
                num_bases=sossa.num_bases,
                free_rider_data=np.array(free_rider, dtype=bool) if free_rider else None,
            ),
            select=SOSSASelect(
                one_body_rotation_angles=one_body_rotation_angles,
                two_body_rotation_angles=two_body_rotation_angles,
                num_orbitals=n_orbitals,
                num_ranks=sossa.num_ranks,
                num_copies=sossa.num_copies,
                num_bases=sossa.num_bases,
                num_positive_one_body_terms=num_positive,
            ),
            normalization=normalization,
            power=self._settings.get("power"),
            energy_shift=sossa.energy_shift,
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _outer_coefficients(sossa: SOSSAContainer) -> np.ndarray:
        r"""Compute the outer PREPARE LCU coefficients from the container generators.

        The one-body coefficients are :math:`\sqrt{2}` times the D1/Q1 term
        one-norms; the spin-free coefficients are the per-``(rank, copy)`` inner
        one-norms scaled by :math:`1/\sqrt{2}`.
        """
        one_body = [_SQRT_TWO * abs(coefficient) for coefficient in (*sossa.d1.coefficients, *sossa.q1.coefficients)]
        num_one_body = len(one_body)
        spin_free = [
            _INV_SQRT_TWO * (abs(row[-1]) + float(np.sum(np.abs(row[:-1]))))
            for row in sossa.inner_coefficients[num_one_body:]
        ]
        return np.array(one_body + spin_free, dtype=float)

    @staticmethod
    def _two_body_rotation_angles(
        sf_rotations: tuple[np.ndarray, ...],
        num_ranks: int,
        num_bases: int,
        num_orbitals: int,
    ) -> np.ndarray:
        r"""Assemble the spin-free SELECT angles ``[R (B+1), N]`` from per-``(rank, basis)`` angles.

        Each rank block holds its ``B`` basis Givens angle vectors followed by a
        zero ``b == B`` row carrying a trailing flag; the blocks are reordered to
        basis-major, rank-minor addressing for the Q# QROM.
        """
        n_bp1 = num_bases + 1
        angles = np.zeros((num_ranks * n_bp1, num_orbitals - 1))
        flags = np.zeros(num_ranks * n_bp1)
        for rank in range(num_ranks):
            for basis in range(num_bases):
                angles[rank * n_bp1 + basis] = sf_rotations[rank * num_bases + basis]
            flags[rank * n_bp1 + num_bases] = 1.0
        with_flag = np.column_stack([angles, flags])
        order = [rank * n_bp1 + basis for basis in range(n_bp1) for rank in range(num_ranks)]
        return with_flag[order]

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
        r_bits = ceil(log2(n_ranks)) if n_ranks > 1 else 0

        data: list[list[bool]] = []
        for x_o in range(xo_dim):
            if x_o < num_one_body_plus:
                g_bits = [False, False]
                r_val = 0
            elif x_o < n_orbitals:
                g_bits = [False, True]
                r_val = 0
            else:
                g_bits = [True, True]
                r_val = (x_o - n_orbitals) // n_copies

            r_enc = [(r_val >> k) & 1 == 1 for k in range(r_bits)]
            data.append(g_bits + r_enc)

        return data

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "hamiltonian_unitary_builder"
