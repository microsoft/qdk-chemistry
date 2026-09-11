r"""QDK/Chemistry implementation of the SOSSA (Sum of Squares Spectral Amplification) block encoding."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import ceil, isnan, log2, nan, sqrt

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    HamiltonianUnitaryBuilder,
    HamiltonianUnitaryBuilderSettings,
)
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    QubitOperator,
    StateVectorContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.qubit_operator.containers.sos import SOSContainer
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAInnerPrepare,
    SOSSARegisterLayout,
    SOSSASelect,
    SOSSAWalkContainer,
)

__all__: list[str] = ["SOSSABuilder", "SOSSASettings"]


class SOSSASettings(HamiltonianUnitaryBuilderSettings):
    """Settings for the SOSSA block encoding builder."""

    def __init__(self):
        r"""Initialize SOSSASettings with default values.

        Attributes:
            reference_ground_state_energy: Reference total ground-state energy :math:`E_{\text{gs}}`.
            reference_energy_gap: Reference gap :math:`E_{\text{gap}}` above the sum-of-squares shift.

        Both default to NaN, meaning unset. They are alternative ways to state the same
        reference point and are mutually exclusive; ``_resolve_lambda_eff`` turns whichever
        was supplied into ``lambda_eff``.

        """
        super().__init__()
        self._set_default(
            "reference_ground_state_energy",
            "float",
            nan,
            "Reference total ground-state energy E_gs, including the core/nuclear contribution, "
            "on the same convention as the sum-of-squares shift. Used only to derive lambda_eff. "
            "NaN leaves lambda_eff unset. Mutually exclusive with 'reference_energy_gap'.",
        )
        self._set_default(
            "reference_energy_gap",
            "float",
            nan,
            "Reference energy gap E_gap = E_gs - E_SOS, the ground-state energy measured from the "
            "sum-of-squares shift. Used only to derive lambda_eff. NaN leaves lambda_eff unset. "
            "Mutually exclusive with 'reference_ground_state_energy'.",
        )


def _resolve_lambda_eff(
    normalization: float,
    energy_shift: float,
    reference_ground_state_energy: float,
    reference_energy_gap: float,
) -> float | None:
    r"""Derive :math:`\lambda_{\text{eff}}` from whichever reference energy was supplied.

    The sum-of-squares Hamiltonian is positive semidefinite and block encoded with
    normalization :math:`\Lambda`, so its spectrum lies in :math:`[0, 2\Lambda]` once the
    origin is moved to :math:`E_{\text{SOS}}`. The ground state sits at
    :math:`E_{\text{gap}}` within that window, and (:cite:`Low2025`, Eq. (11))

    .. math::

        E_{\text{gap}} = E_{\text{gs}} - E_{\text{SOS}}, \qquad
        \lambda_{\text{eff}} = \sqrt{E_{\text{gap}}(2\Lambda - E_{\text{gap}})}

    Args:
        normalization: Block-encoding normalization :math:`\Lambda`.
        energy_shift: The sum-of-squares shift :math:`E_{\text{SOS}}` (includes core energy).
        reference_ground_state_energy: Total :math:`E_{\text{gs}}`, or NaN when unset.
        reference_energy_gap: :math:`E_{\text{gap}}`, or NaN when unset.

    Returns:
        The effective normalization, or ``None`` when neither reference energy was supplied.

    Raises:
        ValueError: If both reference energies are supplied, or if the resulting gap falls
            outside the open interval :math:`(0, 2\Lambda)`, where
            :math:`\lambda_{\text{eff}}` is undefined.

    """
    has_energy = not isnan(reference_ground_state_energy)
    has_gap = not isnan(reference_energy_gap)
    if has_energy and has_gap:
        raise ValueError(
            f"the SOSSA builder accepts 'reference_ground_state_energy' or 'reference_energy_gap', not both; got "
            f"reference_ground_state_energy={reference_ground_state_energy!r} and "
            f"reference_energy_gap={reference_energy_gap!r}"
        )
    if not has_energy and not has_gap:
        return None

    gap = reference_energy_gap if has_gap else reference_ground_state_energy - energy_shift
    two_lambda = 2.0 * normalization
    if not 0.0 < gap < two_lambda:
        source = (
            f"reference_energy_gap {reference_energy_gap!r}"
            if has_gap
            else (
                f"reference_ground_state_energy {reference_ground_state_energy!r} relative to the "
                f"sum-of-squares shift {energy_shift!r}"
            )
        )
        raise ValueError(
            f"{source} gives an energy gap of {gap!r}, outside the representable window "
            f"(0, {two_lambda!r}); lambda_eff is undefined there"
        )
    return float(sqrt(gap * (two_lambda - gap)))


class SOSSABuilder(HamiltonianUnitaryBuilder):
    """SOSSA (Sum of Squares Spectral Amplification) block encoding builder."""

    def __init__(
        self,
        power: int = 1,
        reference_ground_state_energy: float = nan,
        reference_energy_gap: float = nan,
    ):
        r"""Initialize the SOSSA builder.

        Args:
            power: The power to raise the walk operator to. Defaults to 1.
            reference_ground_state_energy: Reference total ground-state energy :math:`E_{\text{gs}}`,
                including the core/nuclear contribution, used to derive
                :attr:`~qdk_chemistry.data.unitary_representation.containers.sossa.SOSSAWalkContainer.lambda_eff`.
                Defaults to NaN, which leaves ``lambda_eff`` unset. Mutually exclusive with
                ``reference_energy_gap``.
            reference_energy_gap: Reference gap :math:`E_{\text{gap}} = E_{\text{gs}} - E_{\text{SOS}}`,
                an alternative to ``reference_ground_state_energy`` for the same purpose. Defaults to NaN.

        """
        super().__init__()
        self._settings = SOSSASettings()
        self._settings.set("power", power)
        self._settings.set("reference_ground_state_energy", reference_ground_state_energy)
        self._settings.set("reference_energy_gap", reference_energy_gap)

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        """Build the SOSSA block encoding from qubit operator.

        Args:
            qubit_hamiltonian: Qubit operator with SOSContainer.

        Returns:
            UnitaryRepresentation wrapping the SOSSAWalkContainer.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("SOSSABuilder requires a QubitOperator containing an SOSContainer")
        sossa = qubit_hamiltonian.get_container()
        if not isinstance(sossa, SOSContainer):
            raise TypeError("SOSSABuilder requires a QubitOperator containing an SOSContainer")
        if sossa.encoding != "jordan-wigner" or sossa.fermion_mode_order != "blocked":
            raise ValueError("the SOSSA circuit builder currently supports blocked Jordan-Wigner operators only")

        meta = sossa.metadata
        n_orbitals = meta.num_spatial_orbitals
        num_positive = meta.num_positive_one_body_terms

        outer_coefficients = self._outer_coefficients(sossa)
        normalization = 0.5 * float(np.sum(a=outer_coefficients**2))
        lambda_eff = _resolve_lambda_eff(
            normalization,
            meta.energy_shift,
            self._settings.get("reference_ground_state_energy"),
            self._settings.get("reference_energy_gap"),
        )

        one_body_rotation_angles = sossa.one_body.angles
        two_body_rotation_angles = self._two_body_rotation_angles(
            sossa.two_body.angles, meta.num_ranks, meta.num_bases, n_orbitals
        )

        reg_bits = self._sossa_register_bits(n_orbitals, meta.num_ranks, meta.num_bases, meta.num_copies)
        num_outer_qubits = reg_bits.outer_prep_bits

        free_rider = self._compute_free_rider_data(
            num_positive, n_orbitals, meta.num_ranks, meta.num_copies, reg_bits.rank_bits
        )

        container = SOSSAWalkContainer(
            outer_prepare=self._build_outer_prepare(outer_coefficients, num_outer_qubits),
            inner_prepare=SOSSAInnerPrepare(
                conditional_coefficients=self._inner_conditional_coefficients(sossa, len(one_body_rotation_angles)),
                free_rider_data=np.array(free_rider, dtype=bool) if free_rider else None,
            ),
            select=SOSSASelect(
                one_body_rotation_angles=one_body_rotation_angles,
                two_body_rotation_angles=two_body_rotation_angles,
            ),
            metadata=meta,
            layout=reg_bits,
            normalization=normalization,
            power=self._settings.get("power"),
            lambda_eff=lambda_eff,
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _sossa_register_bits(
        num_orbitals: int,
        num_ranks: int,
        num_bases: int,
        num_copies: int,
    ) -> SOSSARegisterLayout:
        r"""Derive the structural ancilla register widths implied by ``(N, R, B, C)``.

        Args:
            num_orbitals: Number of spatial orbitals ``N``.
            num_ranks: Number of DFTHC ranks ``R``.
            num_bases: Number of bases ``B`` (``B + 1`` inner entries including the identity term).
            num_copies: Number of copies ``C``.

        Returns:
            SOSSARegisterLayout: The widths the walk container carries for its consumers.

        """
        outer_prep_dim = num_orbitals + num_ranks * num_copies
        rank_bits = ceil(log2(num_ranks)) if num_ranks > 1 else 0
        return SOSSARegisterLayout(
            outer_prep_bits=ceil(log2(outer_prep_dim)) if outer_prep_dim > 1 else 1,
            inner_prep_bits=ceil(log2(num_bases + 1)) if num_bases + 1 > 1 else 1,
            rank_bits=rank_bits,
            num_free_rider_bits=2 + rank_bits,
        )

    @staticmethod
    def _outer_coefficients(sossa: SOSContainer) -> np.ndarray:
        r"""Compute the outer PREPARE LCU coefficients from the container generators.

        The one-body coefficients are :math:`\sqrt{2}` times the D1/Q1 generator
        one-norms; each generator contributes two Pauli terms (X and Y) whose
        magnitudes are summed. The spin-free coefficients are the per-``(rank,
        copy)`` two-body row one-norms scaled by :math:`1/\sqrt{2}`.
        """
        magnitudes = np.abs(np.asarray(sossa.one_body.coeffs))
        row_l1 = magnitudes.sum(axis=1) if magnitudes.ndim >= 2 else np.zeros(len(magnitudes))
        one_body = sqrt(2.0) * row_l1
        spin_free = [(abs(row[-1]) + float(np.sum(np.abs(row[:-1])))) / sqrt(2.0) for row in sossa.two_body.coeffs]
        return np.concatenate([one_body, np.asarray(spin_free, dtype=float)])

    @staticmethod
    def _inner_conditional_coefficients(sossa: SOSContainer, num_one_body: int) -> np.ndarray:
        r"""Assemble the inner-PREPARE conditional distribution ``[Xo, B+1]``.

        One delta row (``b = 0``) per one-body generator, then one spin-free row
        per ``(rank, copy)``: the rotated-``Z`` coefficients followed by the
        absolute identity weight (the ``b == B`` free-rider magnitude).
        """
        b_plus_1 = sossa.metadata.num_bases + 1
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
                bitstring = format(idx, f"0{num_qubits}b")[::-1]
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
        rank_bits: int,
    ) -> list[list[bool]]:
        r"""Compute QROM free-rider data encoding (G, r) for each outer index.

        Shape: ``[Xo][2 + R_bits]``.

        Each entry ``data[x_o]`` encodes the generator type G (2 bits) and the
        rank index r in little-endian binary.

        G encoding (2 bits = ``[sf_vs_dq, d_vs_q]``):
            - D1 (particle): ``[False, False]``
            - Q1 (hole):     ``[False, True]``
            - SF (two-body): ``[True,  True]``

        The D/Q/SF generator taxonomy is Eqs. (28)-(32) of :cite:`Low2025`; the
        two-bit packing above is this implementation's own layout choice.

        """
        xo_dim = n_orbitals + n_ranks * n_copies

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

            r_enc = [(r_val >> k) & 1 == 1 for k in range(rank_bits)]
            data.append(g_bits + r_enc)

        return data

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "hamiltonian_unitary_builder"
