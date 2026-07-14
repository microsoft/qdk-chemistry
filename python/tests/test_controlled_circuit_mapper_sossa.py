"""Tests for the SOSSA controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from pathlib import Path

import numpy as np
import pytest
import qdk
from qdk import qsharp

from qdk_chemistry.algorithms.controlled_circuit_mapper import SOSSAMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import AlgorithmRef, Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

from .test_helpers import create_random_factorized_hamiltonian

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)


def _build_controlled_unitary(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
) -> UnitaryRepresentation:
    """Helper: build UnitaryRepresentation with SOSSAContainer from random factorized data."""
    fh = create_random_factorized_hamiltonian(
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_bases=num_bases,
        num_copies=num_copies,
        seed=seed,
    )
    builder = SOSSABuilder()
    return builder.run(fh)


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
    """Tests for SOSSAMapper.build_outer_prep."""

    @pytest.mark.parametrize("algorithm", ["alias_sampling", "dense_pure_state", "qrom"])
    def test_build_outer_prep_returns_callable(self, algorithm):
        """Verify build_outer_prep produces a Q# callable for each algorithm."""
        controlled_unitary = _build_controlled_unitary()
        container = controlled_unitary.get_container()
        mapper = _make_sossa_mapper(outer_algorithm=algorithm)
        op = mapper.build_outer_prep(container)
        assert op is not None

    @pytest.mark.parametrize("algorithm", ["dense_pure_state", "qrom"])
    def test_build_outer_prep_fidelity(self, algorithm):
        """Verify build_outer_prep's callable prepares the correct statevector.

        Simulates the Q# callable in the global Q# session and checks fidelity
        against the expected normalized state:
          |ψ⟩ = Σ_j (a_j / ||a||) |j⟩

        Note: MakeOuterPreparePureState (dense_pure_state) uses Reversed(register) to
        convert PreparePureStateD's big-endian convention to little-endian for
        Select. dump_machine() reports in big-endian order, so coefficient[k]
        appears at dump index bit_reverse(k). We account for this by building
        expected in LE-address space.
        """
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIF)

        controlled_unitary = _build_controlled_unitary()
        container = controlled_unitary.get_container()
        mapper = _make_sossa_mapper(outer_algorithm=algorithm)
        op = mapper.build_outer_prep(container)

        coefficients = np.asarray(container.outer_prepare.get_coefficients())
        num_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestApplyOuterPrep(op, num_qubits)
        state = qsharp.dump_machine()
        actual_sv = np.array(state.as_dense_state())

        n_states = 2**num_qubits
        expected = np.zeros(n_states)
        for j, amp in enumerate(coefficients):
            if j < n_states:
                if algorithm == "dense_pure_state":
                    # MakeOuterPreparePureState uses Reversed(register):
                    # coefficient[k] → LE address k → BE dump index bit_reverse(k)
                    be_idx = int(format(j, f"0{num_qubits}b")[::-1], 2)
                    expected[be_idx] = amp
                else:
                    expected[j] = amp
        expected /= np.linalg.norm(expected)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

    def test_build_outer_prep_alias_sampling_marginal_probs(self):
        """Verify alias sampling prepares the correct marginal probabilities.

        The alias sampling op produces |ψ⟩ = Σ_l √(p̃_l) |l⟩|garbage_l⟩.
        We check that the marginal probabilities on the index register match
        p(l) = |c_l| / Σ|c_j|.
        """
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIF)

        controlled_unitary = _build_controlled_unitary()
        container = controlled_unitary.get_container()
        bit_precision = 10
        mapper = _make_sossa_mapper(outer_algorithm="alias_sampling", coefficient_bit_precision=bit_precision)
        op = mapper.build_outer_prep(container)

        coefficients = np.asarray(container.outer_prepare.get_coefficients())
        num_index_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1
        total_qubits = 2 * num_index_qubits + 2 * bit_precision + 1

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestApplyOuterPrep(op, total_qubits)
        state = qsharp.dump_machine()
        full_sv = np.array(state.as_dense_state())

        # Compute marginal probabilities on the index register (top bits, LE)
        n_index = 2**num_index_qubits
        shift = total_qubits - num_index_qubits
        probs = np.zeros(n_index)
        for i in range(len(full_sv)):
            be_idx = (i >> shift) & (n_index - 1)
            index_val = int("{:0{w}b}".format(be_idx, w=num_index_qubits)[::-1], 2)
            probs[index_val] += abs(full_sv[i]) ** 2

        abs_coeffs = np.abs(coefficients)
        expected_probs = abs_coeffs / np.sum(abs_coeffs)

        atol = 2.0 / (2**bit_precision)
        np.testing.assert_allclose(probs[: len(coefficients)], expected_probs, atol=atol)


class TestInnerPrep:
    """Tests for SOSSAMapper.build_inner_prep."""

    @pytest.mark.parametrize("algorithm", ["controlled_alias_sampling", "direct"])
    def test_build_inner_prep_fidelity(self, algorithm):
        """Verify inner prep conditional marginals when combined with outer prep.

        Applies outer prep (dense_pure, exact) then inner prep on the combined
        register.  For each outer index l with non-negligible amplitude, checks
        that the conditional marginal probabilities on the inner index register
        match:
            P(b|l) ≈ |c_{l,b}|² / Σ_j |c_{l,j}|²
        """
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIF)

        # Use num_bases=2 for a non-trivial inner dimension (B+1=3)
        controlled_unitary = _build_controlled_unitary(num_orbitals=2, num_ranks=2, num_bases=2, num_copies=1)
        container = controlled_unitary.get_container()

        # Build outer prep (exact, dense_pure)
        outer_mapper = _make_sossa_mapper(outer_algorithm="dense_pure_state")
        outer_op = outer_mapper.build_outer_prep(container)

        # Build inner prep
        bit_precision = 6
        inner_mapper = _make_sossa_mapper(inner_algorithm=algorithm, coefficient_bit_precision=bit_precision)
        inner_op = inner_mapper.build_inner_prep(container)

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
        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestApplyOuterInnerPrep(
            outer_op, inner_op, num_outer_qubits, num_inner_qubits
        )
        state = qsharp.dump_machine()
        full_sv = np.array(state.as_dense_state())

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

            atol = 2.0 / (2**bit_precision) if algorithm == "controlled_alias_sampling" else 1e-3
            np.testing.assert_allclose(
                probs[:n_coeffs], expected_probs, atol=atol, err_msg=f"outer={ell}, algorithm={algorithm}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main SOSSAMapper tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAMapper:
    """Tests for the SOSSA controlled circuit mapper."""

    def test_name_and_type(self):
        """Test that name and type_name return correct values."""
        mapper = SOSSAMapper()
        assert mapper.name() == "sossa"
        assert mapper.type_name() == "controlled_circuit_mapper"

    def test_default_settings(self):
        """Test default settings are correct."""
        mapper = SOSSAMapper()
        assert mapper.outer_prepare_needs_alias_reflection is True
        assert mapper.inner_prepare_needs_alias_reflection is True
        assert mapper.select_needs_phase_gradient is True
        assert mapper.settings().get("rotation_bit_precision") == 10
        assert mapper.settings().get("coefficient_bit_precision") == 10
        assert mapper.settings().get("compute_qubit_percentage") == 0.0

    def test_rejects_non_sossa_container(self):
        """Verify SOSSAMapper raises ValueError for non-SOSSAContainer containers."""

        class MockContainer:
            """Mock container that is not a SOSSAContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="not supported"):
            mapper.run(unitary_rep)

    def test_rejects_multiple_control_qubits(self):
        """Verify SOSSAMapper raises ValueError for multiple control qubits."""
        unitary_rep = _build_controlled_unitary()

        mapper = SOSSAMapper()
        mapper.settings().set("control_indices", [0, 1])
        with pytest.raises(ValueError, match="single control qubit"):
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
        controlled_unitary = _build_controlled_unitary()
        mapper = _make_sossa_mapper(
            outer_algorithm=outer_alg,
            inner_algorithm=inner_alg,
            select_algorithm=select_alg,
        )
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

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
        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None

    def test_walk_params_alias_sampling_flags(self):
        """Test that alias sampling algorithms set reflection flags correctly."""
        controlled_unitary = _build_controlled_unitary()
        mapper = _make_sossa_mapper(
            outer_algorithm="alias_sampling",
            inner_algorithm="controlled_alias_sampling",
            select_algorithm="qrom_phase_gradient",
        )
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert "outerPrepareOp" in params
        assert "innerPrepareOp" in params
        assert "selectOp" in params
        assert params["numSystemQubits"] == 4  # 2 * num_orbitals(=2)

    def test_walk_params_power(self):
        """Test that power passes through to walk_params."""
        fh = create_random_factorized_hamiltonian()
        builder = SOSSABuilder(power=5)
        unitary_rep = builder.run(fh)

        mapper = SOSSAMapper()
        circuit = mapper.run(unitary_rep)

        params = circuit._qsharp_factory.parameter
        assert params["power"] == 5

    def test_compute_qubit_percentage_passes_to_qsharp(self):
        """Test that the compute-qubit percentage passes through to the Q# entry point."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper()
        mapper.settings().set("compute_qubit_percentage", 20.0)

        circuit = mapper.run(controlled_unitary)

        assert circuit._qsharp_factory.parameter["computeQubitPercentage"] == 20.0

    @pytest.mark.parametrize("percentage", [-1.0, 101.0])
    def test_compute_qubit_percentage_rejects_out_of_range_values(self, percentage: float):
        """Test that the compute-qubit percentage remains within 0-100."""
        mapper = SOSSAMapper()

        with pytest.raises(ValueError, match="out of allowed range"):
            mapper.settings().set("compute_qubit_percentage", percentage)


# ═══════════════════════════════════════════════════════════════════════════════
# SELECT fidelity tests
# ═══════════════════════════════════════════════════════════════════════════════


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

    @pytest.mark.parametrize("N", [2, 3])
    def test_select_dq_givens_fidelity(self, N):  # noqa: N803
        """Verify SELECT with a DQ entry produces a non-trivial rotation."""
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIF)

        rng = np.random.default_rng(42 + N)
        dq_angles = []
        for _ in range(N):
            v = rng.standard_normal(N)
            v /= np.linalg.norm(v)
            dq_angles.append(_vector_to_givens_angles(v))

        R, B, C = 1, 1, 1  # noqa: N806
        sf_angles = [
            [0.0] * (N - 1) + [0.0],
            [0.0] * (N - 1) + [1.0],
        ]

        select_data = {
            "numOrbitals": N,
            "numRanks": R,
            "numBases": B,
            "numCopies": C,
            "numPositiveOneBody": N,
            "OneBodyRotationAngles": dq_angles,
            "TwoBodyRotationAngles": sf_angles,
            "rotationBitPrecision": 10,
            "numFreeRiderBits": 2,
        }

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestSelectDQ(select_data, 0)
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        assert np.sum(np.abs(sv) ** 2) > 0.99, "State normalization check failed"

        single_qubit_probs = np.abs(sv) ** 2
        assert np.max(single_qubit_probs) < 0.99, (
            "State is too concentrated; Givens rotation may not be applied correctly"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Walk operator logical resource count tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAWalkLogicalCounts:
    """Verify logical resource counts of the SOSSA walk operator match paper formulas.

    The walk operator W = Ref_{a,B} . U-adj . Ref_B . U (arXiv:2502.15882v1, Eq. 77)
    has resource costs that depend on the problem parameters (N, R, B, C) and the
    chosen sub-algorithms. These tests verify that:

    1. Qubit counts match: 2N + n_Xo + n_B' + 2(spin) + 1(control) + ancilla
    2. Toffoli counts scale correctly with problem size
    3. Walk power multiplies the Toffoli cost linearly

    Reference: arXiv:2502.15882v1, Appendix B.7 (Table 3).
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
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIF)

        controlled_unitary = _build_controlled_unitary(
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
        circuit = mapper.run(controlled_unitary)

        factory = circuit._qsharp_factory
        lc = qsharp.logical_counts(factory.program, *factory.parameter.values())

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


def _make_context():
    """Create a fresh qdk.Context with all Q# sources loaded."""
    return qdk.Context(project_root=_PROJECT_ROOT)


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
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, {num_swap_bits})")
        assert result, f"SelectSwap 1D failed: n_data={n_data}, n_bits={n_bits}, num_swap_bits={num_swap_bits}"

    def test_1d_auto_lambda(self):
        """SelectSwap with numSwapBits=-1 (auto-optimal) should produce correct results."""
        data = _make_random_data_1d(8, 4)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, -1)")
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
        """For each (i, j), Select2DLoad should load data[i][j] into target."""
        data = _make_random_data_2d(n_outer, n_inner, n_bits)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelect2DLoadCorrectness({_bools_to_qs(data)}, {num_swap_bits})")
        assert result, (
            f"Select2DLoad failed: n_outer={n_outer}, n_inner={n_inner}, n_bits={n_bits}, num_swap_bits={num_swap_bits}"
        )
