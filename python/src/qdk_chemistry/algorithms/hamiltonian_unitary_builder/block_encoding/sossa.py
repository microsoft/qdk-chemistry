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
    RotatedPauliContainer,
    SOSSAContainer,
    StateVectorContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.sossa_qubit_operator import RotatedMode, SpinPolicy
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAInnerPrepare,
    SOSSASelect,
    SOSSAWalkContainer,
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
        sossa_container = self._require_sos_container(qubit_hamiltonian)
        if sossa_container.encoding != "jordan-wigner" or sossa_container.fermion_mode_order != "blocked":
            raise ValueError("the SOSSA circuit builder currently supports blocked Jordan-Wigner operators only")
        if sossa_container.normalization <= self._settings.get("tolerance"):
            raise ValueError("the SOSSA operator normalization is below the configured tolerance")

        (
            outer_arr,
            inner_arr,
            one_body_vectors,
            basis_vectors,
            num_d1,
            n_ranks,
            n_bases,
            n_copies,
        ) = self._extract_oracle_data(sossa_container)
        n_orbitals = sossa_container.num_spatial_orbitals

        dq_angles, sf_angles = self._compute_rotation_angles(
            one_body_vectors[:num_d1],
            one_body_vectors[num_d1:],
            basis_vectors,
            num_d1,
            n_orbitals,
            n_ranks,
            n_bases,
        )

        # Compute free-rider data (G, r encoding for QROM)
        free_rider = self._compute_free_rider_data(num_d1, n_orbitals, n_ranks, n_copies)

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

        container = SOSSAWalkContainer(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=sossa_container.normalization,
            power=self._settings.get("power"),
            energy_shift=sossa_container.energy_shift,
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _require_sos_container(value: QubitOperator) -> SOSSAContainer:
        if not isinstance(value, QubitOperator):
            raise TypeError("SOSSABuilder requires a QubitOperator containing an SOSSAContainer")
        container = value.get_container()
        if not isinstance(container, SOSSAContainer):
            raise TypeError("SOSSABuilder requires a QubitOperator containing an SOSSAContainer")
        return container

    @staticmethod
    def _require_rotated_pauli_container(generator: QubitOperator) -> RotatedPauliContainer:
        container = generator.get_container()
        if not isinstance(container, RotatedPauliContainer):
            raise TypeError("SOSSABuilder requires each SOS generator to contain a RotatedPauliContainer")
        return container

    @staticmethod
    def _rotated_modes(generator: QubitOperator) -> list[RotatedMode]:
        modes: list[RotatedMode] = []
        for term in SOSSABuilder._require_rotated_pauli_container(generator).terms:
            mode = term.mode
            if mode is not None:
                modes.append(mode)
        return modes

    @staticmethod
    def _is_positive_one_body(container: RotatedPauliContainer) -> bool:
        """Classify a one-body generator as D1 (positive) from its imaginary term sign."""
        dominant = max(container.terms, key=lambda term: abs(term.coefficient.imag))
        return dominant.coefficient.imag > 0.0

    @staticmethod
    def _extract_oracle_data(
        sossa_container: SOSSAContainer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
        """Extract the current SOSSA Q# oracle layout from structured generators."""
        one_body: dict[int, list[QubitOperator]] = {}
        spin_free: dict[tuple[int, int], QubitOperator] = {}

        for generator in sossa_container.generators:
            container = SOSSABuilder._require_rotated_pauli_container(generator)
            source_index = tuple(container.source_index)
            if container.spin_policy == SpinPolicy.Specific:
                if len(source_index) != 1:
                    raise ValueError("one-body generators require a one-dimensional source index")
                one_body.setdefault(source_index[0], []).append(generator)
            elif container.spin_policy == SpinPolicy.Summed:
                if len(source_index) != 2 or source_index in spin_free:
                    raise ValueError("SF generators require a unique (rank, copy) source index")
                spin_free[source_index] = generator
            else:
                raise ValueError("each SOS generator requires a spin policy")

        d1_keys: list[int] = []
        q1_keys: list[int] = []
        for key, generators in one_body.items():
            container = SOSSABuilder._require_rotated_pauli_container(generators[0])
            (d1_keys if SOSSABuilder._is_positive_one_body(container) else q1_keys).append(key)
        d1_keys.sort()
        q1_keys.sort()
        ordered_one_body = d1_keys + q1_keys
        n_orbitals = sossa_container.num_spatial_orbitals
        if len(ordered_one_body) != n_orbitals or not spin_free:
            raise ValueError("the SOSSA circuit layout requires N one-body modes and at least one SF generator")

        one_body_vectors: list[np.ndarray] = []
        outer_coefficients: list[float] = []
        for key in ordered_one_body:
            generators_by_spin: dict[int, QubitOperator] = {}
            for generator in one_body[key]:
                modes = SOSSABuilder._rotated_modes(generator)
                if not modes:
                    raise ValueError("a one-body generator requires a rotated mode")
                spin = modes[0].spin
                if spin in generators_by_spin:
                    raise ValueError("each one-body mode requires one generator per spin channel")
                generators_by_spin[spin] = generator
            if set(generators_by_spin) != {0, 1}:
                raise ValueError("each one-body mode requires one generator per spin channel")
            spin_pair = [generators_by_spin[0], generators_by_spin[1]]
            basis_vector = np.asarray(SOSSABuilder._rotated_modes(spin_pair[0])[0].basis_vector, dtype=float)
            partner_modes = SOSSABuilder._rotated_modes(spin_pair[1])
            if not partner_modes or not np.allclose(basis_vector, partner_modes[0].basis_vector):
                raise ValueError("spin-paired one-body generators must use the same rotated mode")
            norm = SOSSABuilder._require_rotated_pauli_container(spin_pair[0]).lcu_normalization
            partner_norm = SOSSABuilder._require_rotated_pauli_container(spin_pair[1]).lcu_normalization
            if not np.isclose(norm, partner_norm):
                raise ValueError("spin-paired one-body generators must have equal normalization")
            one_body_vectors.append(basis_vector)
            outer_coefficients.append(sqrt(2.0) * norm)

        n_ranks = max(rank for rank, _ in spin_free) + 1
        n_copies = max(copy for _, copy in spin_free) + 1
        if len(spin_free) != n_ranks * n_copies:
            raise ValueError("SF source indices must form a complete rank-by-copy grid")

        first_terms = SOSSABuilder._require_rotated_pauli_container(spin_free[(0, 0)]).terms
        if len(first_terms) < 3 or (len(first_terms) - 1) % 2 != 0:
            raise ValueError("an SF generator requires one identity term and spin-paired rotated terms")
        n_bases = (len(first_terms) - 1) // 2
        basis_vectors = np.empty((n_ranks, n_bases, n_orbitals))
        inner_coefficients = [[1.0] + [0.0] * n_bases for _ in range(n_orbitals)]

        for rank in range(n_ranks):
            for copy in range(n_copies):
                generator = spin_free[(rank, copy)]
                terms = SOSSABuilder._require_rotated_pauli_container(generator).terms
                if len(terms) != 1 + 2 * n_bases or terms[0].mode is not None:
                    raise ValueError("all SF generators must share the same spin-paired basis layout")
                identity_weight = abs(float(terms[0].coefficient.real)) * sqrt(2.0)
                row: list[float] = []
                for basis in range(n_bases):
                    spin_terms = terms[1 + 2 * basis : 3 + 2 * basis]
                    first_mode = spin_terms[0].mode
                    second_mode = spin_terms[1].mode
                    if first_mode is None or second_mode is None or [first_mode.spin, second_mode.spin] != [0, 1]:
                        raise ValueError("SF rotated terms must be ordered as spin pairs")
                    vector = np.asarray(first_mode.basis_vector, dtype=float)
                    if not np.allclose(vector, second_mode.basis_vector):
                        raise ValueError("spin-paired SF terms must use the same rotated mode")
                    if copy == 0:
                        basis_vectors[rank, basis] = vector
                    elif not np.allclose(basis_vectors[rank, basis], vector):
                        raise ValueError("SF rotated modes must be independent of the copy index")
                    coefficient = spin_terms[0].coefficient
                    if not np.isclose(coefficient.imag, 0.0) or not np.isclose(coefficient, spin_terms[1].coefficient):
                        raise ValueError("Jordan-Wigner SF spin pairs require equal real coefficients")
                    row.append(float(coefficient.real) * 2.0 * sqrt(2.0))
                inner_coefficients.append([*row, identity_weight])
                outer_coefficients.append(SOSSABuilder._require_rotated_pauli_container(generator).lcu_normalization)

        return (
            np.asarray(outer_coefficients),
            np.asarray(inner_coefficients),
            np.asarray(one_body_vectors),
            basis_vectors,
            len(d1_keys),
            n_ranks,
            n_bases,
            n_copies,
        )

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
