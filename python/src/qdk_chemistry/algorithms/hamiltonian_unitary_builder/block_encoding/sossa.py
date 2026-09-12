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
from qdk_chemistry.data.qubit_operator.containers.sum_of_squares import SumOfSquaresContainer
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
        """Initialize SOSSASettings with default values.

        Attributes:
            reference_ground_state_energy: Reference total ground-state energy E_gs.
            reference_energy_gap: Reference gap E_gap above the sum-of-squares shift.

        Both settings are optional and mutually exclusive: each defaults to NaN, meaning unset,
        and they are alternative ways to compute the `lambda_eff`. Leaving both unset leaves
        `lambda_eff` unset.

        """
        super().__init__()
        self._set_default(
            "reference_ground_state_energy",
            "float",
            nan,
            "Reference total ground-state energy of the original hamiltonian, including the core/nuclear contribution. "
            "NaN leaves lambda_eff unset. Mutually exclusive with 'reference_energy_gap'.",
        )
        self._set_default(
            "reference_energy_gap",
            "float",
            nan,
            "Reference energy gap. Mutually exclusive with 'reference_ground_state_energy'.",
        )


class SOSSABuilder(HamiltonianUnitaryBuilder):
    """SOSSA (Sum of Squares Spectral Amplification) block encoding builder."""

    def __init__(
        self,
        power: int = 1,
        reference_ground_state_energy: float = nan,
        reference_energy_gap: float = nan,
    ):
        """Initialize the SOSSA builder.

        Args:
            power: The power to raise the walk operator to. Defaults to 1.
            reference_ground_state_energy: Reference total ground-state energy E_gs, including
                the core/nuclear contribution, used to derive
                :attr:`~qdk_chemistry.data.unitary_representation.containers.sossa.SOSSAWalkContainer.lambda_eff`.
                Defaults to NaN, which leaves ``lambda_eff`` unset. Mutually exclusive with
                ``reference_energy_gap``.
            reference_energy_gap: Reference gap E_gap = E_gs - E_SOS, an alternative to
                ``reference_ground_state_energy`` for the same purpose. Defaults to NaN.

        """
        super().__init__()
        self._settings = SOSSASettings()
        self._settings.set("power", power)
        self._settings.set("reference_ground_state_energy", reference_ground_state_energy)
        self._settings.set("reference_energy_gap", reference_energy_gap)

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        """Build the SOSSA block encoding from qubit operator.

        Args:
            qubit_hamiltonian: Qubit operator with SumOfSquaresContainer.

        Returns:
            UnitaryRepresentation wrapping the SOSSAWalkContainer.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("SOSSABuilder requires a QubitOperator containing a SumOfSquaresContainer")
        sossa = qubit_hamiltonian.get_container()
        if not isinstance(sossa, SumOfSquaresContainer):
            raise TypeError("SOSSABuilder requires a QubitOperator containing a SumOfSquaresContainer")
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

        outer_prep_dim = n_orbitals + meta.num_ranks * meta.num_copies
        rank_bits = ceil(log2(meta.num_ranks)) if meta.num_ranks > 1 else 0
        reg_bits = SOSSARegisterLayout(
            outer_prep_bits=ceil(log2(outer_prep_dim)) if outer_prep_dim > 1 else 1,
            inner_prep_bits=ceil(log2(meta.num_bases + 1)) if meta.num_bases + 1 > 1 else 1,
            rank_bits=rank_bits,
            num_free_rider_bits=2 + rank_bits,
        )
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
    def _outer_coefficients(sossa: SumOfSquaresContainer) -> np.ndarray:
        """Compute the outer PREPARE LCU coefficients from the container generators.

        The outer PREPARE is Eq. (B10) of :cite:`Low2025`::

            PREP|0⟩ = ∑_(xₒ=0)^(Xₒ-1) λ^(G_(xₒ),r_(xₒ),c_(xₒ))/√(2Λ)
                       |xₒ⟩|garbage_(xₒ)⟩.

        The generator weights are::

            λ_(SF,rc) = 1/√2 (|w_B^(rc)| + ∑_(b=0)^(B-1) |w_b^(rc)|),
            λ_(D₁,r) = √(w₊^(r)),
            λ_(Q₁,r) = √(w₋^(r)).
        """
        magnitudes = np.abs(np.asarray(sossa.one_body.coeffs))
        row_l1 = magnitudes.sum(axis=1) if magnitudes.ndim >= 2 else np.zeros(len(magnitudes))
        one_body = sqrt(2.0) * row_l1
        spin_free = [(abs(row[-1]) + float(np.sum(np.abs(row[:-1])))) / sqrt(2.0) for row in sossa.two_body.coeffs]
        return np.concatenate([one_body, np.asarray(spin_free, dtype=float)])

    @staticmethod
    def _inner_conditional_coefficients(sossa: SumOfSquaresContainer, num_one_body: int) -> np.ndarray:
        """Assemble the inner-PREPARE conditional amplitudes ``[Xo, B+1]``.

        One delta row (``b = 0``) per one-body generator, then one spin-free row
        per ``(rank, copy)``. The PREPARE backends square and normalize each row,
        so every spin-free entry is sign(w_b)√|w_b|; SELECT consumes that sign
        for both rotated-``Z`` and identity entries.
        """
        b_plus_1 = sossa.metadata.num_bases + 1
        delta = np.zeros((num_one_body, b_plus_1))
        if num_one_body:
            delta[:, 0] = 1.0
        sf = np.asarray(sossa.two_body.coeffs)
        sf_rows = np.zeros((sf.shape[0] if sf.ndim == 2 else 0, b_plus_1))
        if sf.size:
            if not np.allclose(sf.imag, 0.0):
                raise ValueError("SOSSA requires real two-body coefficients; got a complex-valued block")
            weights = np.real(sf)
            sf_rows = np.sign(weights) * np.sqrt(np.abs(weights))
            empty_rows = ~sf_rows.any(axis=1)
            if empty_rows.any():
                sf_rows[empty_rows, -1] = 1.0
        return np.concatenate([delta, sf_rows], axis=0)

    @staticmethod
    def _two_body_rotation_angles(
        sf_angles: np.ndarray,
        num_ranks: int,
        num_bases: int,
        num_orbitals: int,
    ) -> np.ndarray:
        r"""Assemble the spin-free SELECT angles ``[R (B+1), N-1]`` from per-``(rank, basis)`` angles.

        Rows use basis-major, rank-minor addressing for the Q# QROM. The final
        basis block is zero because ``b == B`` selects the identity term.
        """
        n_bp1 = num_bases + 1
        angles = np.zeros((num_ranks * n_bp1, num_orbitals - 1))
        for basis in range(num_bases):
            for rank in range(num_ranks):
                angles[basis * num_ranks + rank] = sf_angles[rank * num_bases + basis]
        return angles

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


def _resolve_lambda_eff(
    normalization: float,
    energy_shift: float,
    reference_ground_state_energy: float,
    reference_energy_gap: float,
) -> float | None:
    """Derive λ_eff from whichever reference energy was supplied.

    The sum-of-squares Hamiltonian is positive semidefinite and block encoded with
    normalization Λ, so its spectrum lies in [0, 2Λ] once the origin is moved to
    E_SOS. The ground state sits at E_gap within that window, and
    (:cite:`Low2025`, Eq. (11))::

        E_gap = E_gs - E_SOS,
        λ_eff = √(E_gap (2Λ - E_gap)).

    Args:
        normalization: Block-encoding normalization Λ.
        energy_shift: The sum-of-squares shift E_SOS (includes core energy).
        reference_ground_state_energy: Total E_gs, or NaN when unset.
        reference_energy_gap: E_gap, or NaN when unset.

    Returns:
        The effective normalization, or ``None`` when neither reference energy was supplied.

    Raises:
        ValueError: If both reference energies are supplied, or if the resulting gap falls
            outside the open interval (0, 2Λ), where λ_eff is undefined.

    """
    has_energy = not isnan(reference_ground_state_energy)
    has_gap = not isnan(reference_energy_gap)
    if has_energy and has_gap:
        raise ValueError(
            "The SOSSA builder accepts 'reference_ground_state_energy' or 'reference_energy_gap', not both."
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
            f"(0, {two_lambda!r}); lambda_eff is undefined."
        )
    return float(sqrt(gap * (two_lambda - gap)))
