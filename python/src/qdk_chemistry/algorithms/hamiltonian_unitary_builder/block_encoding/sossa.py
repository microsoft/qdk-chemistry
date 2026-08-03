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
    sossa_register_bits,
)

__all__: list[str] = ["SOSSABuilder", "SOSSASettings"]

_SQRT_TWO = sqrt(2.0)
_INV_SQRT_TWO = 1.0 / sqrt(2.0)


def _row_l1_norms(coeffs: np.ndarray) -> np.ndarray:
    """Return the per-row L1 norm of a ``[M, T]`` coefficient block (empty-safe)."""
    magnitudes = np.abs(np.asarray(coeffs))
    if magnitudes.ndim < 2:
        return np.zeros(len(magnitudes))
    return magnitudes.sum(axis=1)


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

        one_body_rotation_angles = sossa.one_body.angles
        two_body_rotation_angles = self._two_body_rotation_angles(
            sossa.two_body.angles, sossa.num_ranks, sossa.num_bases, n_orbitals
        )

        reg_bits = sossa_register_bits(n_orbitals, sossa.num_ranks, sossa.num_bases, sossa.num_copies)
        num_outer_qubits = reg_bits["xo_bits"]
        num_inner_qubits = reg_bits["b_bits"]

        free_rider = self._compute_free_rider_data(num_positive, n_orbitals, sossa.num_ranks, sossa.num_copies)

        container = SOSSAWalkContainer(
            outer_prepare=self._build_outer_prepare(outer_coefficients, num_outer_qubits),
            inner_prepare=SOSSAInnerPrepare(
                conditional_coefficients=self._inner_conditional_coefficients(sossa, len(one_body_rotation_angles)),
                num_inner_qubits=num_inner_qubits,
                free_rider_data=np.array(free_rider, dtype=bool) if free_rider else None,
            ),
            select=SOSSASelect(
                one_body_rotation_angles=one_body_rotation_angles,
                two_body_rotation_angles=two_body_rotation_angles,
                num_positive_one_body_terms=num_positive,
            ),
            num_orbitals=n_orbitals,
            num_ranks=sossa.num_ranks,
            num_bases=sossa.num_bases,
            num_copies=sossa.num_copies,
            normalization=normalization,
            power=self._settings.get("power"),
            energy_shift=sossa.energy_shift,
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _outer_coefficients(sossa: SOSSAContainer) -> np.ndarray:
        r"""Compute the outer PREPARE LCU coefficients from the container generators.

        The one-body coefficients are :math:`\sqrt{2}` times the D1/Q1 generator
        one-norms; each generator contributes two Pauli terms (X and Y) whose
        magnitudes are summed. The spin-free coefficients are the per-``(rank,
        copy)`` two-body row one-norms scaled by :math:`1/\sqrt{2}`.
        """
        one_body = _SQRT_TWO * _row_l1_norms(sossa.one_body.coeffs)
        spin_free = [_INV_SQRT_TWO * (abs(row[-1]) + float(np.sum(np.abs(row[:-1])))) for row in sossa.two_body.coeffs]
        return np.concatenate([one_body, np.asarray(spin_free, dtype=float)])

    @staticmethod
    def _inner_conditional_coefficients(sossa: SOSSAContainer, num_one_body: int) -> np.ndarray:
        r"""Assemble the inner-PREPARE conditional distribution ``[Xo, B+1]``.

        One delta row (``b = 0``) per one-body generator, then one spin-free row
        per ``(rank, copy)``: the rotated-``Z`` coefficients followed by the
        absolute identity weight (the ``b == B`` free-rider magnitude).
        """
        b_plus_1 = sossa.num_bases + 1
        delta = np.zeros((num_one_body, b_plus_1))
        if num_one_body:
            delta[:, 0] = 1.0
        sf = np.asarray(sossa.two_body.coeffs)
        sf_rows = np.zeros((sf.shape[0], b_plus_1))
        if sf.size:
            sf_rows[:, :-1] = sf[:, :-1].real
            sf_rows[:, -1] = np.abs(sf[:, -1])
        return np.concatenate([delta, sf_rows], axis=0)

    @staticmethod
    def _two_body_rotation_angles(
        sf_angles: np.ndarray,
        num_ranks: int,
        num_bases: int,
        num_orbitals: int,
    ) -> np.ndarray:
        r"""Assemble the spin-free SELECT angles ``[R (B+1), N-1]`` from per-``(rank, basis)`` angles.

        Each rank block holds its ``B`` basis Givens angle vectors followed by a
        zero ``b == B`` (identity) row; the blocks are reordered to basis-major,
        rank-minor addressing for the Q# QROM, which recomputes the ``b == B`` flag.
        """
        n_bp1 = num_bases + 1
        angles = np.zeros((num_ranks * n_bp1, num_orbitals - 1))
        for rank in range(num_ranks):
            for basis in range(num_bases):
                angles[rank * n_bp1 + basis] = sf_angles[rank * num_bases + basis]
        order = [rank * n_bp1 + basis for basis in range(n_bp1) for rank in range(num_ranks)]
        return angles[order]

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
