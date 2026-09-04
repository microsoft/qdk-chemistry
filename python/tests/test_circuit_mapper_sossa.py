"""Tests for the SOSSA controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms.circuit_mapper import SOSSAMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import AlgorithmRef, Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, create_qsharp_context, get_qsharp_context

from .test_helpers import create_random_factorized_hamiltonian, to_sossa_operator


def _reverse_bits(x: int, n: int) -> int:
    """Reverse the bit order of *x* within an *n*-bit field."""
    return int(format(x, f"0{n}b")[::-1], 2)


def _alias_atol(num_coefficients: int, bits_precision: int) -> float:
    """Tolerance on a marginal probability for an L-term, mu-bit alias table.

    Same bound the dedicated alias-sampling tests assert against; see
    ``_alias_atol`` in ``test_state_preparation_alias.py``.
    """
    return 1.0 / (num_coefficients * 2**bits_precision)


def _with_prepared_gradient(op, num_gradient: int):
    """Wrap an outer PREPARE so it prepares and restores the gradient it reads.

    The walk hands the outer PREPARE a slice of the persistent gradient register that
    phase estimation prepares once; a standalone simulation has to supply it.
    """
    if not num_gradient:
        return op
    return QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
        op, QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, num_gradient
    )


def _build_sossa_unitary(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
) -> UnitaryRepresentation:
    """Helper: build UnitaryRepresentation with SOSSAWalkContainer from random factorized data."""
    fh = create_random_factorized_hamiltonian(
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_bases=num_bases,
        num_copies=num_copies,
        seed=seed,
    )
    builder = SOSSABuilder()
    return builder.run(to_sossa_operator(fh))


def _make_sossa_mapper(
    outer_algorithm: str = "alias_sampling",
    inner_algorithm: str = "controlled_alias_sampling",
    select_algorithm: str = "qrom_phase_gradient",
    coefficient_bit_precision: int = 10,
    rotation_bit_precision: int = 10,
) -> SOSSAMapper:
    """Create a SOSSAMapper with the given algorithm settings."""
    mapper = SOSSAMapper()
    mapper.settings().set("outer_prepare", AlgorithmRef("state_prep", outer_algorithm))
    mapper.settings().set("inner_prepare_algorithm", inner_algorithm)
    mapper.settings().set("select_algorithm", select_algorithm)
    mapper.settings().set("coefficient_bit_precision", coefficient_bit_precision)
    mapper.settings().set("rotation_bit_precision", rotation_bit_precision)
    return mapper


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-operation builder tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOuterPrep:
    """Tests for SOSSAMapper._build_outer_prep."""

    @pytest.mark.parametrize("algorithm", ["alias_sampling", "dense_pure_state", "qrom"])
    def test_build_outer_prep_returns_callable(self, algorithm):
        """Verify _build_outer_prep produces a Q# callable for each algorithm."""
        sossa_unitary = _build_sossa_unitary()
        container = sossa_unitary.get_container()
        mapper = _make_sossa_mapper(outer_algorithm=algorithm)
        op, num_gradient = mapper._build_outer_prep(container)
        assert op is not None
        assert num_gradient == (10 if algorithm == "qrom" else 0)

    @pytest.mark.parametrize("algorithm", ["dense_pure_state", "qrom"])
    def test_build_outer_prep_fidelity(self, algorithm):
        """Verify _build_outer_prep's callable prepares the correct statevector.

        Simulates the Q# callable in the global Q# session and checks fidelity
        against the expected normalized state:
          |ψ⟩ = Σ_j (a_j / ||a||) |j⟩

        Every backend writes the outer index register little-endian, which is how SELECT
        reads it back, so the backend a caller picks cannot change which generator an
        amplitude belongs to. ``DumpRegister`` reports big-endian, so coefficient ``j``
        is expected at the bit-reversed dump index for all of them.
        """
        sossa_unitary = _build_sossa_unitary()
        container = sossa_unitary.get_container()
        mapper = _make_sossa_mapper(outer_algorithm=algorithm)
        op, num_gradient = mapper._build_outer_prep(container)

        coefficients = np.asarray(container.outer_prepare.get_coefficients())
        num_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1

        full_sv = np.array(
            dump_operation_on_state(
                _with_prepared_gradient(op, num_gradient),
                num_qubits + num_gradient,
                context=get_qsharp_context(),
            )
        )
        actual_sv = full_sv.reshape(2**num_qubits, 2**num_gradient)[:, 0]

        n_states = 2**num_qubits
        expected = np.zeros(n_states)
        for j, amp in enumerate(coefficients):
            if j < n_states:
                expected[_reverse_bits(j, num_qubits)] = amp
        expected /= np.linalg.norm(expected)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

    def test_build_outer_prep_alias_sampling_marginal_probs(self):
        r"""Verify alias sampling prepares the SOS outer distribution, not its square root.

        The SOS block encoding needs amplitudes proportional to the generator one-norms
        :math:`c_l` (Eqs. (7) and (9) of Low et al. 2025), because the normalization
        :math:`\Lambda = \frac{1}{2}\sum_l c_l^2` and the energy decoding
        :math:`E = \Lambda(1 + \cos 2\pi\varphi)` are both read off those amplitudes.
        One-dimensional alias sampling discretizes its input as a *probability*
        distribution, so the marginal on the index register must come out as
        :math:`p(l) = c_l^2 / \sum_j c_j^2` rather than :math:`|c_l| / \sum_j |c_j|`.
        """
        sossa_unitary = _build_sossa_unitary()
        container = sossa_unitary.get_container()
        bit_precision = 10
        mapper = _make_sossa_mapper(outer_algorithm="alias_sampling", coefficient_bit_precision=bit_precision)
        op, num_gradient = mapper._build_outer_prep(container)
        assert num_gradient == 0

        coefficients = np.asarray(container.outer_prepare.get_coefficients())
        num_index_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1
        total_qubits = 2 * num_index_qubits + 2 * bit_precision + 1

        full_sv = np.array(dump_operation_on_state(op, total_qubits, context=get_qsharp_context()))

        n_index = 2**num_index_qubits
        shift = total_qubits - num_index_qubits
        probs = np.zeros(n_index)
        for i in range(len(full_sv)):
            probs[(i >> shift) & (n_index - 1)] += abs(full_sv[i]) ** 2

        squared_coeffs = np.abs(coefficients) ** 2
        expected_probs = np.zeros(n_index)
        for j, p in enumerate(squared_coeffs / np.sum(squared_coeffs)):
            expected_probs[_reverse_bits(j, num_index_qubits)] = p

        atol = _alias_atol(len(coefficients), bit_precision)
        np.testing.assert_allclose(probs, expected_probs, atol=atol)


class TestInnerPrep:
    """Tests for SOSSAMapper._build_inner_prep."""

    @pytest.mark.parametrize("algorithm", ["controlled_alias_sampling", "direct"])
    def test_build_inner_prep_fidelity(self, algorithm):
        """Verify inner prep conditional marginals when combined with outer prep.

        Applies outer prep (dense_pure, exact) then inner prep on the combined
        register.  For each outer index l with non-negligible amplitude, checks
        that the conditional marginal probabilities on the inner index register
        match:
            P(b|l) ≈ |c_{l,b}|² / Σ_j |c_{l,j}|²
        """
        # Use num_bases=2 for a non-trivial inner dimension (B+1=3)
        sossa_unitary = _build_sossa_unitary(num_orbitals=2, num_ranks=2, num_bases=2, num_copies=1)
        container = sossa_unitary.get_container()

        # Build outer prep (exact, dense_pure)
        outer_mapper = _make_sossa_mapper(outer_algorithm="dense_pure_state")
        outer_op, _ = outer_mapper._build_outer_prep(container)

        # Build inner prep
        bit_precision = 6
        inner_mapper = _make_sossa_mapper(inner_algorithm=algorithm, coefficient_bit_precision=bit_precision)
        inner_op = inner_mapper._build_inner_prep(container)

        # Compute register sizes
        outer_coeffs = np.asarray(container.outer_prepare.get_coefficients())
        num_outer_qubits = math.ceil(math.log2(len(outer_coeffs))) if len(outer_coeffs) > 1 else 1

        inner_coeffs = container.inner_prepare.conditional_coefficients
        n_coeffs = inner_coeffs.shape[1]
        n_index_bits = math.ceil(math.log2(n_coeffs)) if n_coeffs > 1 else 1

        if algorithm == "controlled_alias_sampling":
            fr = container.inner_prepare.free_rider_data
            n_fr = fr.shape[1] if fr is not None and fr.size > 0 else 0
            num_inner_qubits = 2 * n_index_bits + 2 * bit_precision + 3 + n_fr
        else:  # direct
            fr = container.inner_prepare.free_rider_data
            n_fr = fr.shape[1] if fr is not None and fr.size > 0 else 0
            num_inner_qubits = n_index_bits + n_fr

        # Apply outer + inner prep
        full_sv = np.array(
            dump_operation_on_state(
                QSHARP_UTILS.SOSSAWalk.MakeOuterInnerPrepOp(outer_op, inner_op, num_outer_qubits),
                num_outer_qubits + num_inner_qubits,
                context=get_qsharp_context(),
            )
        )

        # Check conditional marginals for each outer value l
        total_qubits = num_outer_qubits + num_inner_qubits
        n_inner_index = 2**n_index_bits

        for ell in range(len(outer_coeffs)):
            if abs(outer_coeffs[ell]) < 1e-10:
                continue

            # Compute conditional marginal probs on inner index register
            probs = np.zeros(n_inner_index)
            for i in range(len(full_sv)):
                amp = full_sv[i]
                if abs(amp) < 1e-15:
                    continue
                bits = format(i, f"0{total_qubits}b")
                outer_be = bits[:num_outer_qubits]
                outer_val = int(outer_be[::-1], 2)  # LE
                if outer_val != ell:
                    continue
                inner_be = bits[num_outer_qubits : num_outer_qubits + n_index_bits]
                inner_val = int(inner_be[::-1], 2)
                probs[inner_val] += abs(amp) ** 2

            # Normalize to conditional probability
            total_prob = np.sum(probs)
            if total_prob < 1e-10:
                continue
            probs /= total_prob

            # Expected: |c_{l,b}|² / Σ|c_{l,j}|²
            abs_coeffs = np.abs(inner_coeffs[ell])
            expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

            atol = _alias_atol(n_coeffs, bit_precision) if algorithm == "controlled_alias_sampling" else 1e-3
            np.testing.assert_allclose(
                probs[:n_coeffs], expected_probs, atol=atol, err_msg=f"outer={ell}, algorithm={algorithm}"
            )


class TestSOSSAMapper:
    """Tests for the SOSSA block-encoding circuit mapper."""

    def test_rejects_non_sossa_container(self):
        """Verify SOSSAMapper raises ValueError for non-SOSSAWalkContainer containers."""

        class MockContainer:
            """Mock container that is not a SOSSAWalkContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="not supported"):
            mapper.run(unitary_rep)

    @pytest.mark.parametrize(
        ("outer_alg", "inner_alg", "select_alg"),
        [
            ("alias_sampling", "controlled_alias_sampling", "qrom_phase_gradient"),
            ("dense_pure_state", "direct", "direct"),
            ("qrom", "controlled_alias_sampling", "direct"),
            ("alias_sampling", "direct", "qrom_phase_gradient"),
            ("dense_pure_state", "controlled_alias_sampling", "qrom_phase_gradient"),
        ],
        ids=[
            "default_all",
            "dense_direct_direct",
            "qrom_alias_direct",
            "alias_direct_phase",
            "dense_alias_phase",
        ],
    )
    def test_all_algorithm_combinations_produce_circuit(self, outer_alg, inner_alg, select_alg):
        """Test that all valid algorithm combinations produce a Circuit."""
        unitary = _build_sossa_unitary()
        mapper = _make_sossa_mapper(
            outer_algorithm=outer_alg,
            inner_algorithm=inner_alg,
            select_algorithm=select_alg,
        )
        circuit = mapper.run(unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_declares_the_register_the_walk_reflects_about(self):
        """The block encoding reports a flat register the caller can size the reflection from.

        Unary QPE reads the reflected width off ``num_qubits`` minus the system register and
        the gradient tail, so those three have to be consistent for the reflection it builds
        with ``MakeAncillaReflectionOp`` to cover exactly the flagging ancillas.
        """
        unitary = _build_sossa_unitary()
        container = unitary.get_container()
        mapper = _make_sossa_mapper()
        circuit = mapper.run(unitary)

        num_system_qubits = 2 * container.metadata.num_spatial_orbitals
        num_gradient = circuit.metadata.num_phase_gradient_ancillas
        assert num_gradient == mapper._num_phase_gradient_qubits
        assert circuit.num_qubits == num_system_qubits + mapper._num_ancilla_qubits(container)
        assert circuit.num_qubits - num_system_qubits - num_gradient > 0

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 2, 2, 1),
        ],
        ids=["N2R1B1C1", "N2R2B1C1", "N3R2B2C1"],
    )
    def test_mapping_parametrized_dimensions(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Test mapping for various (N, R, B, C) configurations."""
        unitary = _build_sossa_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = SOSSAMapper()
        circuit = mapper.run(unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None


def _vector_to_givens_angles(vec: np.ndarray) -> list[float]:
    """Convert a unit vector to Givens rotation angles (same as SOSSABuilder)."""
    N = len(vec)  # noqa: N806
    v = vec.copy().astype(float)
    angles = [0.0] * (N - 1)
    for j in range(N - 2, -1, -1):
        angles[j] = float(np.arctan2(v[j + 1], v[j]))
        v[j] = float(np.sqrt(v[j] ** 2 + v[j + 1] ** 2))
    return angles


class TestSelectFullFidelity:
    """Tests for the full SELECT operation fidelity with known rotation angles."""

    @staticmethod
    def _select_data(
        N: int,  # noqa: N803
        rotation_bit_precision: int,
        num_ranks: int = 1,
        num_bases: int = 1,
        num_copies: int = 1,
    ) -> dict:
        rng = np.random.default_rng(42 + N)

        def unit_angles() -> list[float]:
            v = rng.standard_normal(N)
            return _vector_to_givens_angles(v / np.linalg.norm(v))

        rank_bits = math.ceil(math.log2(num_ranks)) if num_ranks > 1 else 0
        return {
            "numOrbitals": N,
            "numRanks": num_ranks,
            "numBases": num_bases,
            "numCopies": num_copies,
            "numPositiveOneBody": N,
            "OneBodyRotationAngles": [unit_angles() for _ in range(N)],
            # Indexed b * R + r, matching both BuildSFBulkRotationData and the direct path.
            "TwoBodyRotationAngles": [unit_angles() for _ in range(num_ranks * (num_bases + 1))],
            "rotationBitPrecision": rotation_bit_precision,
            "numFreeRiderBits": 2 + rank_bits,
        }

    @staticmethod
    def _run_select(select_data: dict, use_phase_gradient: bool, xo_value: int = 0, b_value: int = 0) -> np.ndarray:
        """Run TestSelectDQ in its own context and return the resulting state vector.

        A fresh context per run is required: ``TestSelectDQ`` allocates through the QIR
        runtime without releasing, so a second run in the same context would report both
        runs' qubits.
        """
        ctx = create_qsharp_context()
        ctx.code.QDKChemistry.Utils.SOSSAWalk.TestSelectDQ(select_data, xo_value, b_value, use_phase_gradient)
        return np.array(ctx.dump_machine().as_dense_state())

    @pytest.mark.parametrize("N", [2, 3])
    def test_select_dq_givens_fidelity(self, N):  # noqa: N803
        """Verify SELECT with a DQ entry produces a non-trivial rotation."""
        select_data = self._select_data(N, rotation_bit_precision=10)

        sv = self._run_select(select_data, use_phase_gradient=False)

        assert np.sum(np.abs(sv) ** 2) > 0.99, "State normalization check failed"

        single_qubit_probs = np.abs(sv) ** 2
        assert np.max(single_qubit_probs) < 0.99, (
            "State is too concentrated; Givens rotation may not be applied correctly"
        )

    @pytest.mark.parametrize("rotation_bit_precision", [8, 12])
    @pytest.mark.parametrize("N", [2, 3])
    def test_select_dq_rotation_backends_agree(self, N, rotation_bit_precision):  # noqa: N803
        """The QROM/phase-gradient rotation path must match the direct-rotation path."""
        select_data = self._select_data(N, rotation_bit_precision)

        direct = self._run_select(select_data, use_phase_gradient=False)
        qrom = self._run_select(select_data, use_phase_gradient=True)

        self._assert_backends_agree(direct, qrom, N, rotation_bit_precision)

    @pytest.mark.parametrize("N", [2, 3])
    def test_select_sf_rotation_backends_agree(self, N):  # noqa: N803
        """The same agreement, on a spin-free entry with b and r both nonzero."""
        select_data = self._select_data(N, 8, num_ranks=2, num_bases=2, num_copies=1)
        # x_o = N + 1 is the second SF generator, so r = 1; b = 1 is a non-identity basis.
        xo_value, b_value = N + 1, 1

        direct = self._run_select(select_data, False, xo_value, b_value)
        qrom = self._run_select(select_data, True, xo_value, b_value)

        self._assert_backends_agree(direct, qrom, N, 8)

    @staticmethod
    def _assert_backends_agree(direct: np.ndarray, qrom: np.ndarray, N: int, rotation_bit_precision: int) -> None:  # noqa: N803
        """Compare the two rotation backends up to angle quantization and a global phase."""
        stride = len(qrom) // len(direct)
        subspace = qrom[::stride]
        leaked = 1.0 - float(np.vdot(subspace, subspace).real)
        assert abs(leaked) < 1e-9, f"phase gradient register did not return to |0...0> (leaked {leaked:.3e})"

        # Compare up to an irrelevant global phase.
        overlap = abs(complex(np.vdot(direct, subspace)))
        # Angle quantization error is O(2^-b) per rotation, and infidelity is quadratic in it.
        tolerance = 40 * (N - 1) * (2.0**-rotation_bit_precision) ** 2
        assert 1.0 - overlap < tolerance, (
            f"rotation backends disagree for N={N}, b_rot={rotation_bit_precision}: "
            f"infidelity {1.0 - overlap:.3e} exceeds {tolerance:.3e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Walk operator logical resource count tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAWalkLogicalCounts:
    """Verify logical resource counts of the SOSSA walk operator match paper formulas.

    The walk operator W = Ref_{a,B} . U-adj . Ref_B . U (defined inline above Eq. (9)
    of :cite:`Low2025`, derived in its Appendix A 2)
    has resource costs that depend on the problem parameters (N, R, B, C) and the
    chosen sub-algorithms. These tests verify that:

    1. Qubit counts match: 2N + n_Xo + n_B' + 2(spin) + 1(control) + ancilla
    2. Toffoli counts scale correctly with problem size
    3. Walk power multiplies the Toffoli cost linearly

    Reference: :cite:`Low2025`, Appendix B 7 b (Table III).
    """

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 2, 2, 1),
            (4, 2, 2, 2),
        ],
        ids=["N2R1B1C1", "N2R2B1C1", "N3R2B2C1", "N4R2B2C2"],
    )
    def test_qubit_count_matches_formula(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Verify numQubits matches the paper formula bounds."""
        sossa_unitary = _build_sossa_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure_state",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )
        circuit = mapper.run(sossa_unitary)

        factory = circuit._qsharp_factory
        ctx = factory.program._qdk_context
        lc = ctx.logical_counts(factory.program, *factory.parameter.values())

        actual_qubits = lc["numQubits"]

        N = num_orbitals  # noqa: N806
        R, B, C = num_ranks, num_bases, num_copies  # noqa: N806
        Xo = N + R * C  # noqa: N806
        n_xo = math.ceil(math.log2(Xo)) if Xo > 1 else 1
        n_b = math.ceil(math.log2(B + 1)) if B + 1 > 1 else 1
        min_qubits = 2 * N + n_xo + n_b + 2 + 1
        select_ancilla = 3

        assert actual_qubits >= min_qubits + select_ancilla, (
            f"N={N},R={R},B={B},C={C}: qubits={actual_qubits} < min={min_qubits}+select_anc={select_ancilla}"
        )
        max_overhead = n_xo + n_b + N + 10
        assert actual_qubits <= min_qubits + select_ancilla + max_overhead, (
            f"N={N},R={R},B={B},C={C}: qubits={actual_qubits} > max={min_qubits + select_ancilla + max_overhead}"
        )


def _int_to_bools(value: int, width: int) -> list[bool]:
    """Convert integer to little-endian Bool array (matching Q# IntAsBoolArray)."""
    return [(value >> i) & 1 == 1 for i in range(width)]


def _bools_to_qs(data: list) -> str:
    """Convert nested Python bool list to Q# literal string."""
    if isinstance(data[0], list):
        return "[" + ", ".join(_bools_to_qs(row) for row in data) + "]"
    return "[" + ", ".join("true" if b else "false" for b in data) + "]"


def _make_random_data_1d(n_data: int, n_bits: int, seed: int = 42) -> list[list[bool]]:
    """Generate random Bool[][] data for 1D SelectSwap tests."""
    rng = np.random.default_rng(seed)
    return [_int_to_bools(int(rng.integers(0, 2**n_bits)), n_bits) for _ in range(n_data)]


def _make_random_data_2d(n_outer: int, n_inner: int, n_bits: int, seed: int = 42) -> list[list[list[bool]]]:
    """Generate random Bool[][][] data for 2D Select2DLoad tests."""
    rng = np.random.default_rng(seed)
    return [[_int_to_bools(int(rng.integers(0, 2**n_bits)), n_bits) for _ in range(n_inner)] for _ in range(n_outer)]


_NS = "QDKChemistry.Utils.SelectSwap"


class TestSelectSwapCorrectness:
    """Verify SelectSwap loads the correct data for each address."""

    @pytest.mark.parametrize(
        ("n_data", "n_bits", "num_swap_bits"),
        [
            (4, 3, 0),  # no swap (plain Select)
            (4, 3, 1),  # 1 swap bit
            (8, 4, 0),  # 8 entries, no swap
            (8, 4, 1),  # 8 entries, 1 swap bit
            (8, 4, 2),  # 8 entries, 2 swap bits
        ],
    )
    def test_1d_all_addresses(self, n_data, n_bits, num_swap_bits):
        """For each address |i⟩, SelectSwap should load data[i] into output."""
        data = _make_random_data_1d(n_data, n_bits)
        result = create_qsharp_context().eval(
            f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, {num_swap_bits})"
        )
        assert result, f"SelectSwap 1D failed: n_data={n_data}, n_bits={n_bits}, num_swap_bits={num_swap_bits}"

    def test_1d_auto_lambda(self):
        """SelectSwap with numSwapBits=-1 (auto-optimal) should produce correct results."""
        data = _make_random_data_1d(8, 4)
        result = create_qsharp_context().eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, -1)")
        assert result, "SelectSwap 1D with auto lambda failed"

    @pytest.mark.parametrize(
        ("n_outer", "n_inner", "n_bits", "num_swap_bits"),
        [
            (2, 4, 3, 0),  # no swap
            (2, 4, 3, 1),  # 1 swap bit
            (3, 4, 4, 0),  # non-power-of-2 outer
        ],
    )
    def test_2d_all_addresses(self, n_outer, n_inner, n_bits, num_swap_bits):
        """For each (i, j), SelectSwap2D should load data[i][j] into target."""
        data = _make_random_data_2d(n_outer, n_inner, n_bits)
        result = create_qsharp_context().eval(
            f"{_NS}.TestSelectSwap2DCorrectness({_bools_to_qs(data)}, {num_swap_bits}, false)"
        )
        assert result, (
            f"SelectSwap2D failed: n_outer={n_outer}, n_inner={n_inner}, n_bits={n_bits}, num_swap_bits={num_swap_bits}"
        )
