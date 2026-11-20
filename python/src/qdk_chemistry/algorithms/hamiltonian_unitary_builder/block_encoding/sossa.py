r"""QDK/Chemistry implementation of the Sum of Squares Spectral Amplification (SOSSA) block encoding."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import ceil, isfinite, isnan, log2, nan, sqrt

import numpy as np
from scipy.sparse import csr_matrix

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
    SOSSABlockEncodingContainer,
    SOSSAInnerPrepare,
    SOSSARegisterLayout,
    SOSSASelect,
)

__all__: list[str] = ["SOSSABuilder", "SOSSABuilderSettings"]


class SOSSABuilderSettings(HamiltonianUnitaryBuilderSettings):
    """Settings for the SOSSA block encoding builder."""

    def __init__(self):
        """Initialize SOSSABuilderSettings with default values.

        Attributes:
            reference_ground_state_energy: Reference total ground-state energy E_gs.
            reference_energy_gap: Reference gap E_gap above the sum-of-squares shift.
            variational_bound_determinants: Determinants spanning a Rayleigh-Ritz upper bound on E_gs.

        The three settings are optional and mutually exclusive: the first two default to NaN and
        the last to an empty list, meaning unset, and they are alternative ways to compute the
        `lambda_eff`. Leaving all unset leaves `lambda_eff` unset.

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
        self._set_default(
            "variational_bound_determinants",
            "vector<string>",
            [],
            "Occupation strings over '0', 'u', 'd' and '2' (one character per spatial orbital of the "
            "factorized Hamiltonian) whose span gives a Rayleigh-Ritz upper bound E_ub >= E_gs on the "
            "encoded Hamiltonian; lambda_eff then uses E_gap = min(E_ub - E_SOS, Lambda), which keeps it an "
            "upper bound on the true lambda_eff. Empty disables it. "
            "Mutually exclusive with the two reference settings.",
        )


class SOSSABuilder(HamiltonianUnitaryBuilder):
    """Sum of Squares Spectral Amplification (SOSSA) block encoding builder."""

    def __init__(
        self,
        power: int = 1,
        reference_ground_state_energy: float = nan,
        reference_energy_gap: float = nan,
        variational_bound_determinants: list[str] | None = None,
    ):
        """Initialize the SOSSA builder.

        Args:
            power: The power to raise the walk operator to. Defaults to 1.
            reference_ground_state_energy: Reference total ground-state energy E_gs, including
                the core/nuclear contribution, used to derive
                :attr:`~qdk_chemistry.data.unitary_representation.containers.sossa.SOSSABlockEncodingContainer.lambda_eff`.
                Defaults to NaN, which leaves ``lambda_eff`` unset. Mutually exclusive with
                ``reference_energy_gap``.
            reference_energy_gap: Reference gap E_gap = E_gs - E_SOS, an alternative to
                ``reference_ground_state_energy`` for the same purpose. Defaults to NaN.
            variational_bound_determinants: Occupation strings (e.g. ``"2u00"``) whose span gives
                a variational upper bound on E_gs of the encoded Hamiltonian, the tighter bound of
                :cite:`Shirobokov2026`. Defaults to ``None``, which disables it.

        """
        super().__init__()
        self._settings = SOSSABuilderSettings()
        self._settings.set("power", power)
        self._settings.set("reference_ground_state_energy", reference_ground_state_energy)
        self._settings.set("reference_energy_gap", reference_energy_gap)
        if variational_bound_determinants is not None:
            self._settings.set("variational_bound_determinants", list(variational_bound_determinants))

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        """Build the SOSSA block encoding from qubit operator.

        Args:
            qubit_hamiltonian: Qubit operator with SumOfSquaresContainer.

        Returns:
            UnitaryRepresentation wrapping the SOSSABlockEncodingContainer.

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
        self._validate_one_body_generators(sossa, n_orbitals, num_positive)
        self._validate_two_body_generators(sossa, n_orbitals, meta.num_ranks, meta.num_bases, meta.num_copies)

        outer_coefficients = self._outer_coefficients(sossa)
        normalization = 0.5 * float(np.sum(a=outer_coefficients**2))
        if not isfinite(normalization) or normalization <= 0.0:
            raise ValueError(f"SOSSA block encoding requires a positive normalization, got {normalization}.")
        lambda_eff = _resolve_lambda_eff(
            normalization,
            meta.energy_shift,
            self._settings.get("reference_ground_state_energy"),
            self._settings.get("reference_energy_gap"),
            sossa=sossa,
            variational_bound_determinants=list(self._settings.get("variational_bound_determinants")),
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

        container = SOSSABlockEncodingContainer(
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
    def _validate_one_body_generators(
        sossa: SumOfSquaresContainer,
        num_orbitals: int,
        num_positive: int,
    ) -> None:
        """Validate one-body rows before they are packed into the fixed outer layout."""
        one_body = sossa.one_body
        angles = np.asarray(one_body.angles)
        coeffs = np.asarray(one_body.coeffs, dtype=complex)
        if angles.shape[0] != num_orbitals or coeffs.shape[0] != num_orbitals:
            raise ValueError(
                f"SOSSA requires {num_orbitals} one-body slots; got {angles.shape[0]} angle rows "
                f"and {coeffs.shape[0]} coefficient rows."
            )
        if one_body.paulis != ("X", "Y"):
            raise ValueError(f"SOSSA one-body generators must use Pauli labels ('X', 'Y'); got {one_body.paulis!r}.")

        signs = np.ones(num_orbitals)
        signs[num_positive:] = -1.0
        expected_y = 1j * signs * coeffs[:, 0]
        matches = np.isclose(coeffs[:, 1], expected_y, rtol=1e-12, atol=1e-12)
        if not np.all(matches):
            row = int(np.flatnonzero(~matches)[0])
            generator = "particle" if row < num_positive else "hole"
            relation = "c_Y = +i*c_X" if row < num_positive else "c_Y = -i*c_X"
            raise ValueError(
                f"SOSSA one-body coefficient row {row} generator {generator} and must satisfy "
                f"{relation}; got c_X={coeffs[row, 0]!r}, c_Y={coeffs[row, 1]!r}."
            )

    @staticmethod
    def _validate_two_body_generators(
        sossa: SumOfSquaresContainer,
        num_orbitals: int,
        num_ranks: int,
        num_bases: int,
        num_copies: int,
    ) -> None:
        """Validate spin-free rows before they are packed into the fixed outer layout."""
        angles = np.asarray(sossa.two_body.angles)
        coeffs = np.asarray(sossa.two_body.coeffs, dtype=complex)
        if sossa.two_body.paulis != ("Z",):
            raise ValueError(f"SOSSA two-body generators must use Pauli label ('Z',); got {sossa.two_body.paulis!r}.")
        if angles.shape[0] != num_ranks * num_bases or (angles.shape[0] and angles.shape[1] != num_orbitals - 1):
            raise ValueError(
                f"SOSSA requires two-body angles of shape {(num_ranks * num_bases, num_orbitals - 1)}; "
                f"got {angles.shape}."
            )
        if coeffs.shape[0] != num_ranks * num_copies or (coeffs.shape[0] and coeffs.shape[1] != num_bases + 1):
            raise ValueError(
                f"SOSSA requires two-body coefficients of shape {(num_ranks * num_copies, num_bases + 1)}; "
                f"got {coeffs.shape}."
            )
        if not np.allclose(coeffs.imag, 0.0):
            raise ValueError("SOSSA requires real two-body coefficients; got a complex-valued block")

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
    *,
    sossa: SumOfSquaresContainer,
    variational_bound_determinants: list[str],
) -> float | None:
    """Derive λ_eff from whichever reference energy was supplied.

    The sum-of-squares Hamiltonian is positive semidefinite and block encoded with
    normalization Λ, so its spectrum lies in [0, 2Λ] once the origin is moved to
    E_SOS. The ground state sits at E_gap within that window, and
    (:cite:`Low2025`, Eq. (11))::

        E_gap = E_gs - E_SOS,
        λ_eff = √(E_gap (2Λ - E_gap)).

    With ``variational_bound_determinants``, E_gs is replaced by the variational upper bound
    E_ub ≥ E_gs of :cite:`Shirobokov2026`: the lowest Rayleigh-Ritz eigenvalue of the encoded
    Hamiltonian over the span of those determinants, evaluated directly from the SOS factors, with the
    gap clamped to Λ so that λ_eff stays an upper bound.

    Args:
        normalization: Block-encoding normalization Λ.
        energy_shift: The sum-of-squares shift E_SOS (includes core energy).
        reference_ground_state_energy: Total E_gs, or NaN when unset.
        reference_energy_gap: E_gap, or NaN when unset.
        sossa: The sum-of-squares operator whose factors define the encoded Hamiltonian.
        variational_bound_determinants: Occupation strings spanning the Rayleigh-Ritz space, or empty.

    Returns:
        The effective normalization, or ``None`` when no reference energy was supplied.

    Raises:
        ValueError: If more than one reference is supplied, if a determinant string is malformed,
            repeated or in another particle sector, or if the resulting gap falls outside the open
            interval (0, 2Λ), where λ_eff is undefined.

    """
    has_energy = not isnan(reference_ground_state_energy)
    has_gap = not isnan(reference_energy_gap)
    has_determinants = bool(variational_bound_determinants)
    if has_energy + has_gap + has_determinants > 1:
        raise ValueError(
            "The SOSSA builder accepts only one of 'reference_ground_state_energy', 'reference_energy_gap' "
            "and 'variational_bound_determinants'."
        )
    if not has_energy and not has_gap and not has_determinants:
        return None

    if has_determinants:
        num_orbitals = sossa.metadata.num_spatial_orbitals
        determinants: list[tuple[int, int]] = []
        for occupation in variational_bound_determinants:
            if len(occupation) != num_orbitals or set(occupation) - set("0ud2"):
                raise ValueError(
                    f"variational_bound_determinants entry {occupation!r} must have one of '0', 'u', 'd', '2' "
                    f"per spatial orbital ({num_orbitals})."
                )
            determinants.append(
                (
                    sum(1 << p for p, s in enumerate(occupation) if s in "u2"),
                    sum(1 << p for p, s in enumerate(occupation) if s in "d2"),
                )
            )
        if len(set(determinants)) != len(determinants):
            raise ValueError("variational_bound_determinants must not repeat a determinant.")
        if len({(a.bit_count(), b.bit_count()) for a, b in determinants}) != 1:
            raise ValueError("variational_bound_determinants must all share one (alpha, beta) electron count.")
        index = {det: i for i, det in enumerate(determinants)}

        # A Givens angle row θ encodes the unit vector u(θ) with u_j = cos θ_j ∏_{k<j} sin θ_k.
        def orbital(angles: np.ndarray) -> np.ndarray:
            """Return the unit vector whose Givens elimination gives ``angles``."""
            vector = np.zeros(num_orbitals)
            remainder = 1.0
            for j, theta in enumerate(angles):
                vector[j] = remainder * np.cos(theta)
                remainder *= np.sin(theta)
            vector[-1] = remainder
            return vector

        def number(vector: np.ndarray, det: tuple[int, int]) -> dict[tuple[int, int], float]:
            """Apply the spin-summed number operator of orbital ``vector`` to determinant ``det``."""
            result: dict[tuple[int, int], float] = {}
            for spin in (0, 1):
                occupied = det[spin]
                for q in range(num_orbitals):
                    if not occupied >> q & 1:
                        continue
                    result[det] = result.get(det, 0.0) + vector[q] ** 2
                    for p in range(num_orbitals):
                        if occupied >> p & 1:
                            continue
                        low, high = min(p, q), max(p, q)
                        sign = -1.0 if (occupied & ((1 << high) - (1 << (low + 1)))).bit_count() % 2 else 1.0
                        moved = occupied ^ (1 << q) ^ (1 << p)
                        target = (moved, det[1]) if spin == 0 else (det[0], moved)
                        result[target] = result.get(target, 0.0) + sign * vector[p] * vector[q]
            return result

        hamiltonian = np.zeros((len(determinants), len(determinants)))
        num_positive = sossa.metadata.num_positive_one_body_terms
        # One-body squares are w n_k (D1) or w (2 - n_k) (Q1), with w = 4|c_X|^2 the eigenvalue magnitude.
        for row, (angles, coeffs) in enumerate(zip(sossa.one_body.angles, sossa.one_body.coeffs, strict=True)):
            weight = 4.0 * abs(coeffs[0]) ** 2
            sign = 1.0 if row < num_positive else -1.0
            for j, det in enumerate(determinants):
                if row >= num_positive:
                    hamiltonian[j, j] += 2.0 * weight
                for target, amplitude in number(orbital(angles), det).items():
                    if target in index:
                        hamiltonian[index[target], j] += sign * weight * amplitude
        # Spin-free squares are ½ G², G = (c_B + Σ_b c_b) I - Σ_b c_b L_b, so ⟨i|G²|j⟩ = ⟨G i|G j⟩.
        num_bases, num_copies = sossa.metadata.num_bases, sossa.metadata.num_copies
        basis_orbitals = [orbital(angles) for angles in sossa.two_body.angles]
        for row, coeffs in enumerate(np.real(sossa.two_body.coeffs)):
            rank = row // num_copies
            targets: dict[tuple[int, int], int] = {}
            entries: list[tuple[int, int, float]] = []
            for j, det in enumerate(determinants):
                image = {det: coeffs[num_bases] + float(np.sum(coeffs[:num_bases]))}
                for basis in range(num_bases):
                    for target, amplitude in number(basis_orbitals[rank * num_bases + basis], det).items():
                        image[target] = image.get(target, 0.0) - coeffs[basis] * amplitude
                entries.extend(
                    (targets.setdefault(target, len(targets)), j, amplitude) for target, amplitude in image.items()
                )
            rows, cols, amplitudes = zip(*entries, strict=True)
            images = csr_matrix((amplitudes, (rows, cols)), shape=(len(targets), len(determinants)))
            hamiltonian += 0.5 * (images.T @ images).toarray()
        # E_ub ≥ E_gs only bounds λ_eff from above while E_ub - E_SOS ≤ Λ, where λ_eff peaks at Λ.
        gap = min(float(np.linalg.eigvalsh(0.5 * (hamiltonian + hamiltonian.T))[0]), normalization)
    else:
        gap = reference_energy_gap if has_gap else reference_ground_state_energy - energy_shift
    two_lambda = 2.0 * normalization
    if not 0.0 < gap < two_lambda:
        source = (
            f"variational_bound_determinants {variational_bound_determinants!r}"
            if has_determinants
            else f"reference_energy_gap {reference_energy_gap!r}"
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
