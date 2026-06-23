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
from qdk_chemistry.data import AlgorithmRef, Circuit, FactorizedHamiltonianContainer
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

from .test_helpers import create_test_orbitals

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)

# Short name → registry name for outer_prepare AlgorithmRef
_OUTER_PREP_MAP = {
    "alias_sampling": "alias_sampling",
    "dense_pure": "dense_pure_state",
    "qrom": "qrom_state_prep",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_random_factorized_hamiltonian(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
):
    """Create a random FactorizedHamiltonianContainer for testing."""
    rng = np.random.default_rng(seed)
    n, r, b, c = num_orbitals, num_ranks, num_bases, num_copies

    h1 = rng.standard_normal((n, n))
    h1 = (h1 + h1.T) / 2

    u_matrices = np.zeros(r * b * n)
    for ri in range(r):
        for bi in range(b):
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            u_matrices[ri * b * n + bi * n : ri * b * n + (bi + 1) * n] = v

    w_matrices = rng.standard_normal(r * b * c)
    wb_matrix = rng.standard_normal((r, c))

    orbitals = create_test_orbitals(n)
    inactive_fock = np.zeros((n, n))

    return FactorizedHamiltonianContainer(
        h1,
        u_matrices,
        w_matrices,
        wb_matrix,
        r,
        b,
        c,
        orbitals,
        0.0,
        inactive_fock,
    )


def _build_controlled_unitary(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
    quantum_walk: bool = True,
):
    """Helper: build ControlledUnitary with SOSSAContainer from random factorized data."""
    fh = _make_random_factorized_hamiltonian(
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_bases=num_bases,
        num_copies=num_copies,
        seed=seed,
    )
    builder = SOSSABuilder(quantum_walk=quantum_walk)
    unitary_rep = builder.run(fh)
    return ControlledUnitary(unitary=unitary_rep, control_indices=[0])


def _make_sossa_mapper(
    outer_algorithm: str = "alias_sampling",
    inner_algorithm: str = "controlled_alias_sampling",
    select_algorithm: str = "qrom_phase_gradient",
    coefficient_bit_precision: int = 10,
    rotation_bit_precision: int = 10,
) -> SOSSAMapper:
    """Create a SOSSAMapper with the given algorithm settings."""
    mapper = SOSSAMapper()
    ref_name = _OUTER_PREP_MAP.get(outer_algorithm, outer_algorithm)
    mapper.settings().set("outer_prepare", AlgorithmRef("state_prep", ref_name))
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

    @pytest.mark.parametrize("algorithm", ["alias_sampling", "dense_pure", "qrom"])
    def test_build_outer_prep_returns_callable(self, algorithm):
        """Verify build_outer_prep produces a Q# callable for each algorithm."""
        controlled_unitary = _build_controlled_unitary()
        container = controlled_unitary.unitary.get_container()
        mapper = _make_sossa_mapper(outer_algorithm=algorithm)
        op = mapper.build_outer_prep(container)
        assert op is not None

    @pytest.mark.parametrize("algorithm", ["dense_pure", "qrom"])
    def test_build_outer_prep_fidelity(self, algorithm):
        """Verify build_outer_prep's callable prepares the correct statevector.

        Simulates the Q# callable in the global Q# session and checks fidelity
        against the expected normalized state:
          |ψ⟩ = Σ_j (a_j / ||a||) |j⟩
        """
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        controlled_unitary = _build_controlled_unitary()
        container = controlled_unitary.unitary.get_container()
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
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        controlled_unitary = _build_controlled_unitary()
        container = controlled_unitary.unitary.get_container()
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
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        # Use num_bases=2 for a non-trivial inner dimension (B+1=3)
        controlled_unitary = _build_controlled_unitary(num_orbitals=2, num_ranks=2, num_bases=2, num_copies=1)
        container = controlled_unitary.unitary.get_container()

        # Build outer prep (exact, dense_pure)
        outer_mapper = _make_sossa_mapper(outer_algorithm="dense_pure")
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


class TestSelectSettings:
    """Tests for SELECT algorithm settings on SOSSAMapper."""

    def test_default_algorithm(self):
        """Test default algorithm is qrom_phase_gradient."""
        mapper = SOSSAMapper()
        assert mapper.settings().get("select_algorithm") == "qrom_phase_gradient"
        assert mapper.settings().get("rotation_bit_precision") == 10

    @pytest.mark.parametrize("algorithm", ["qrom_phase_gradient", "direct"])
    def test_valid_algorithms(self, algorithm):
        """Test all valid algorithms are accepted."""
        mapper = SOSSAMapper()
        mapper.settings().set("select_algorithm", algorithm)
        assert mapper.settings().get("select_algorithm") == algorithm


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

    def test_custom_settings(self):
        """Test custom settings are accepted."""
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
        )
        assert mapper.outer_prepare_needs_alias_reflection is False
        assert mapper.inner_prepare_needs_alias_reflection is False
        assert mapper.select_needs_phase_gradient is False

    def test_basic_mapping_produces_circuit_with_factory(self):
        """Test that mapping produces a Circuit with both qsharp_op and qsharp_factory."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_rejects_non_sossa_container(self):
        """Verify SOSSAMapper raises ValueError for non-SOSSAContainer containers."""

        class MockContainer:
            """Mock container that is not a SOSSAContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="not supported"):
            mapper.run(controlled_unitary)

    def test_rejects_multiple_control_qubits(self):
        """Verify SOSSAMapper raises ValueError for multiple control qubits."""
        controlled_unitary = _build_controlled_unitary()
        controlled_unitary = ControlledUnitary(unitary=controlled_unitary.unitary, control_indices=[0, 1])

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="single control qubit"):
            mapper.run(controlled_unitary)

    @pytest.mark.parametrize(
        ("outer_alg", "inner_alg", "select_alg"),
        [
            ("alias_sampling", "controlled_alias_sampling", "qrom_phase_gradient"),
            ("dense_pure", "direct", "direct"),
            ("qrom", "controlled_alias_sampling", "direct"),
            ("alias_sampling", "direct", "qrom_phase_gradient"),
            ("dense_pure", "controlled_alias_sampling", "qrom_phase_gradient"),
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

    def test_circuit_has_qsharp_factory_program_and_parameter(self):
        """Test that the circuit's qsharp_factory has correct structure."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        factory = circuit._qsharp_factory
        assert factory is not None
        assert factory.program is not None
        assert isinstance(factory.parameter, dict)

        expected_keys = {
            "outerPrepareOp",
            "innerPrepareOp",
            "selectOp",
            "numSystemQubits",
            "numOuterQubits",
            "numOuterIndexQubits",
            "numInnerQubits",
            "numReflectInner",
            "numPhaseGradientQubits",
            "power",
        }
        assert expected_keys == set(factory.parameter.keys())

    def test_walk_params_reflect_mapper_settings(self):
        """Test that walk_params correctly reflect mapper settings."""
        controlled_unitary = _build_controlled_unitary()
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=12,
        )
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert "outerPrepareOp" in params
        assert "innerPrepareOp" in params
        assert "selectOp" in params
        assert params["numSystemQubits"] == 4  # 2 * num_orbitals(=2)

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

    def test_walk_params_system_qubit_count(self):
        """Test that numSystemQubits = 2 * num_orbitals."""
        n = 3
        controlled_unitary = _build_controlled_unitary(num_orbitals=n)
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert params["numSystemQubits"] == 2 * n

    def test_walk_params_power(self):
        """Test that power passes through to walk_params."""
        fh = _make_random_factorized_hamiltonian()
        builder = SOSSABuilder(power=5)
        unitary_rep = builder.run(fh)
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])

        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert params["power"] == 5


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


class TestSelectGivensFidelity:
    """Tests for the Givens rotation chain fidelity.

    NOTE: The CNOT·Ry·CNOT Givens gate preserves parity within each adjacent
    qubit pair ({|01⟩,|10⟩} and {|00⟩,|11⟩} are invariant subspaces).
    For N=2 this gives exact single-excitation decomposition.  For N>2 there
    is leakage to multi-excitation states, which is acceptable for the block
    encoding (the within{Givens}apply{Majorana} pattern relies only on
    round-trip correctness: Givens†·Givens = I).

    Qubit ordering: Q# dump_machine uses BIG-ENDIAN convention where
    qs[0] = MSB.  Single excitation on qs[j] → state index 2^(N-1-j).
    """

    def test_givens_rotation_n2_exact(self):
        """Verify ApplyGivensSequence maps |10⟩ → target vector exactly for N=2."""
        N = 2  # noqa: N806
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        rng = np.random.default_rng(42)
        vec = rng.standard_normal(N)
        vec /= np.linalg.norm(vec)

        angles = _vector_to_givens_angles(vec)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestGivensRotation(angles, N)
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        # Big-endian: qs[j] occupied → sv[1 << (N-1-j)]
        for j in range(N):
            idx = 1 << (N - 1 - j)
            assert abs(sv[idx] - vec[j]) < 1e-10, f"Mismatch at qubit {j}: expected {vec[j]:.6f}, got {sv[idx]}"

        # All other amplitudes should be zero (no leakage for N=2)
        single_excitation_indices = {1 << (N - 1 - j) for j in range(N)}
        for i in range(2**N):
            if i not in single_excitation_indices:
                assert abs(sv[i]) < 1e-10, f"Non-zero amplitude at index {i}: {sv[i]}"

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_givens_round_trip(self, N):  # noqa: N803
        """Verify Givens rotation and its adjoint cancel: G†G|ψ⟩ = |ψ⟩."""
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        rng = np.random.default_rng(123 + N)
        vec = rng.standard_normal(N)
        vec /= np.linalg.norm(vec)

        angles = _vector_to_givens_angles(vec)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestGivensRoundTrip(angles, N)
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        # Initial state X(qs[0]) → sv[2^(N-1)] (big-endian: qs[0] = MSB)
        initial_idx = 1 << (N - 1)
        assert abs(sv[initial_idx]) > 1 - 1e-10, f"Round trip failed: sv[{initial_idx}] = {sv[initial_idx]}"
        for i in range(2**N):
            if i != initial_idx:
                assert abs(sv[i]) < 1e-10, f"Non-zero at {i}: {sv[i]}"

    @pytest.mark.parametrize(
        ("vec_desc", "vec"),
        [
            ("e0_2", np.array([1.0, 0.0])),
            ("e1_2", np.array([0.0, 1.0])),
            ("half_2", np.array([1.0, 1.0]) / np.sqrt(2)),
            ("random_2", np.array([0.6, 0.8])),
        ],
        ids=["e0_2", "e1_2", "half_2", "random_2"],
    )
    def test_givens_specific_vectors_n2(self, vec_desc, vec):
        """Verify Givens rotation for specific N=2 vectors (exact, no leakage)."""
        N = len(vec)  # noqa: N806
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        angles = _vector_to_givens_angles(vec)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestGivensRotation(angles, N)
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        # Big-endian: qs[j] occupied → sv[1 << (N-1-j)]
        for j in range(N):
            idx = 1 << (N - 1 - j)
            actual_amp = sv[idx]
            assert abs(actual_amp - vec[j]) < 1e-10, (
                f"Vector {vec_desc}: mismatch at qubit {j}: expected {vec[j]}, got {actual_amp}"
            )

    @pytest.mark.parametrize(
        ("vec_desc", "vec"),
        [
            ("e0_3", np.array([1.0, 0.0, 0.0])),
            ("e1_3", np.array([0.0, 1.0, 0.0])),
            ("e2_3", np.array([0.0, 0.0, 1.0])),
        ],
        ids=["e0_3", "e1_3", "e2_3"],
    )
    def test_givens_basis_vectors_n3(self, vec_desc, vec):
        """Verify Givens with basis vectors for N=3 (no leakage for basis vectors).

        When the target is a basis vector, only one Givens rotation is non-trivial
        and the others have angle 0 or π/2, avoiding the leakage issue.
        """
        N = len(vec)  # noqa: N806
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        angles = _vector_to_givens_angles(vec)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestGivensRotation(angles, N)
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        # For basis vectors, check the expected qubit is occupied
        j_nonzero = int(np.argmax(np.abs(vec)))
        idx = 1 << (N - 1 - j_nonzero)
        assert abs(abs(sv[idx]) - 1.0) < 1e-10, (
            f"Vector {vec_desc}: expected unit amplitude at sv[{idx}], got {sv[idx]}"
        )


class TestSelectSpinsFidelity:
    """Tests for SelectSpins operation correctness.

    Register layout: isSF(1) + spinDQ(1) + spinSF(1) + spin(1) + sysDown(N) + sysUp(N)
    Big-endian convention: qs[k] → bit position (total-1-k) in state vector index.
    """

    @staticmethod
    def _bit_of(qubit_pos: int, total: int) -> int:
        """Convert qubit array position to bit position in state vector (big-endian)."""
        return total - 1 - qubit_pos

    def test_dq_mode_spin_up(self):
        """In DQ mode (isSF=0) with spinDQ=0, no SWAP should occur."""
        N = 2  # noqa: N806
        total = 2 * N + 4  # 8 qubits
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestSelectSpins(False, False, N)  # spinDQ=0, spinSF=0
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        # Initial: sysDown[0]=1 (qubit pos 4). Expected: unchanged (spin=0, no SWAP).
        sysdown0_bit = self._bit_of(4, total)  # bit 3
        expected_idx = 1 << sysdown0_bit
        assert abs(abs(sv[expected_idx]) - 1.0) < 1e-10, (
            f"Expected all amplitude at index {expected_idx}, got {abs(sv[expected_idx])}"
        )

    def test_dq_mode_spin_down(self):
        """In DQ mode (isSF=0) with spinDQ=1, SWAP should move sysDown to sysUp."""
        N = 2  # noqa: N806
        total = 2 * N + 4  # 8 qubits
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestSelectSpins(True, False, N)  # spinDQ=1, spinSF=0
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        spinDQ_bit = self._bit_of(1, total)  # noqa: N806  # bit 6
        spin_bit = self._bit_of(3, total)  # bit 4
        sysUp0_bit = self._bit_of(4 + N, total)  # noqa: N806  # bit 1
        expected_idx = (1 << spinDQ_bit) | (1 << spin_bit) | (1 << sysUp0_bit)
        assert abs(abs(sv[expected_idx]) - 1.0) < 1e-10, (
            f"Expected all amplitude at index {expected_idx}, got {abs(sv[expected_idx])}"
        )

    def test_sf_mode_spin_down(self):
        """In SF mode (isSF=1) with spinSF=1, SWAP should occur."""
        N = 2  # noqa: N806
        total = 2 * N + 4  # 8 qubits
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestSelectSpinsSF(True, N)  # spinSF=1
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        isSF_bit = self._bit_of(0, total)  # noqa: N806  # bit 7
        spinSF_bit = self._bit_of(2, total)  # noqa: N806  # bit 5
        spin_bit = self._bit_of(3, total)  # bit 4
        sysUp0_bit = self._bit_of(4 + N, total)  # noqa: N806  # bit 1
        expected_idx = (1 << isSF_bit) | (1 << spinSF_bit) | (1 << spin_bit) | (1 << sysUp0_bit)
        assert abs(abs(sv[expected_idx]) - 1.0) < 1e-10, (
            f"Expected all amplitude at index {expected_idx}, got {abs(sv[expected_idx])}"
        )


class TestSelectFullFidelity:
    """Tests for the full SELECT operation fidelity with known rotation angles."""

    @pytest.mark.parametrize("N", [2, 3])
    def test_select_round_trip(self, N):  # noqa: N803
        """Verify SELECT†·SELECT = Identity (unitarity check)."""
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

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
            "numD1": N,
            "dqRotationAngles": dq_angles,
            "sfRotationAngles": sf_angles,
            "rotationBitPrecision": 10,
            "numFreeRiderBits": 2,
        }

        qdk.code.QDKChemistry.Utils.SOSSAWalk.TestSelectRT(select_data, 0)
        state = qsharp.dump_machine()
        sv = np.array(state.as_dense_state())

        max_amp = np.max(np.abs(sv))
        num_nonzero = np.sum(np.abs(sv) > 1e-10)
        assert max_amp > 1 - 1e-10, f"Round trip max amp = {max_amp} (expected ~1)"
        assert num_nonzero == 1, f"Round trip has {num_nonzero} non-zero amps (expected 1)"

    @pytest.mark.parametrize("N", [2, 3])
    def test_select_dq_givens_fidelity(self, N):  # noqa: N803
        """Verify SELECT with a DQ entry produces a non-trivial rotation."""
        qsharp.init(project_root=_PROJECT_ROOT, target_profile=qsharp.TargetProfile.Adaptive_RIFLA)

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
            "numD1": N,
            "dqRotationAngles": dq_angles,
            "sfRotationAngles": sf_angles,
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
        qsharp.init(project_root=_PROJECT_ROOT)

        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
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

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (3, 2, 2, 1),
            (4, 2, 2, 2),
        ],
        ids=["N2R1B1C1", "N3R2B2C1", "N4R2B2C2"],
    )
    def test_toffoli_scaling_with_problem_size(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Verify Toffoli count scales correctly with problem size for direct SELECT."""
        qsharp.init(project_root=_PROJECT_ROOT)

        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )
        circuit = mapper.run(controlled_unitary)

        factory = circuit._qsharp_factory
        lc = qsharp.logical_counts(factory.program, *factory.parameter.values())

        tof = lc["cczCount"]

        N = num_orbitals  # noqa: N806
        R, B, C = num_ranks, num_bases, num_copies  # noqa: N806
        num_sf = R * C
        num_rotations = N - 1
        gates_per_rotation = N + num_sf * (B + 1)
        dominant_term = 2 * num_rotations * gates_per_rotation

        assert tof >= dominant_term, f"N={N},R={R},B={B},C={C}: tof={tof} < dominant={dominant_term}"
        Xo = N + num_sf  # noqa: N806
        xo_bits = math.ceil(math.log2(Xo)) if Xo > 1 else 1
        b_bits = math.ceil(math.log2(B + 1)) if B + 1 > 1 else 1
        max_ctrl_cost = xo_bits + b_bits + 5
        max_tof = 10 * dominant_term * max_ctrl_cost
        assert tof <= max_tof, f"N={N},R={R},B={B},C={C}: tof={tof} > max={max_tof}"

    def test_power_multiplies_toffoli_linearly(self):
        """Verify that power=p multiplies the walk step Toffoli cost by p."""
        qsharp.init(project_root=_PROJECT_ROOT)

        N, R, B, C = 2, 1, 1, 1  # noqa: N806
        fh = _make_random_factorized_hamiltonian(num_orbitals=N, num_ranks=R, num_bases=B, num_copies=C, seed=42)

        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )

        # Build with power=1
        builder1 = SOSSABuilder(power=1)
        unitary1 = builder1.run(fh)
        cu1 = ControlledUnitary(unitary=unitary1, control_indices=[0])
        circuit1 = mapper.run(cu1)
        factory1 = circuit1._qsharp_factory
        lc1 = qsharp.logical_counts(factory1.program, *factory1.parameter.values())

        # Build with power=3
        builder3 = SOSSABuilder(power=3)
        unitary3 = builder3.run(fh)
        cu3 = ControlledUnitary(unitary=unitary3, control_indices=[0])
        circuit3 = mapper.run(cu3)
        factory3 = circuit3._qsharp_factory
        lc3 = qsharp.logical_counts(factory3.program, *factory3.parameter.values())

        tof_1 = lc1["cczCount"]
        tof_3 = lc3["cczCount"]

        assert tof_3 == 3 * tof_1, f"power=3 Toffoli={tof_3} != 3 * power=1 Toffoli={tof_1}"
        assert lc3["numQubits"] == lc1["numQubits"], (
            f"power=3 qubits={lc3['numQubits']} != power=1 qubits={lc1['numQubits']}"
        )

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (3, 2, 2, 1),
        ],
        ids=["N2R1B1C1", "N3R2B2C1"],
    )
    def test_majorana_op_contributes_14_toffoli_per_walk(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Verify MajoranaOp contributes at least 2*7 = 14 Toffoli per walk step."""
        qsharp.init(project_root=_PROJECT_ROOT)

        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )

        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )
        circuit = mapper.run(controlled_unitary)
        factory = circuit._qsharp_factory
        lc = qsharp.logical_counts(factory.program, *factory.parameter.values())

        majorana_min_tof = 2 * 7
        assert lc["cczCount"] >= majorana_min_tof, f"Total tof={lc['cczCount']} < majorana_min={majorana_min_tof}"

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 2, 1, 1),
            (3, 2, 2, 1),
            (4, 3, 2, 2),
        ],
        ids=["N2R2B1C1", "N3R2B2C1", "N4R3B2C2"],
    )
    def test_spin_copy_contributes_2n_toffoli(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Verify SelectSpins contributes at least 2*N Toffoli per walk step."""
        qsharp.init(project_root=_PROJECT_ROOT)

        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )
        circuit = mapper.run(controlled_unitary)
        factory = circuit._qsharp_factory
        lc = qsharp.logical_counts(factory.program, *factory.parameter.values())

        N = num_orbitals  # noqa: N806
        spin_copy_tof = 2 * N
        assert lc["cczCount"] >= spin_copy_tof, f"N={N}: total tof={lc['cczCount']} < spin_copy_min={spin_copy_tof}"

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (3, 2, 1, 1),
            (4, 2, 2, 2),
        ],
        ids=["N2R1B1C1", "N3R2B1C1", "N4R2B2C2"],
    )
    def test_reflection_qubit_count(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Verify reflections act on the correct number of qubits."""
        qsharp.init(project_root=_PROJECT_ROOT)

        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )
        circuit = mapper.run(controlled_unitary)

        factory = circuit._qsharp_factory
        lc = qsharp.logical_counts(factory.program, *factory.parameter.values())

        N = num_orbitals  # noqa: N806
        R, B, C = num_ranks, num_bases, num_copies  # noqa: N806
        Xo = N + R * C  # noqa: N806
        n_xo = math.ceil(math.log2(Xo)) if Xo > 1 else 1
        n_b = math.ceil(math.log2(B + 1)) if B + 1 > 1 else 1

        outer_ref_qubits = n_xo + n_b + 2
        inner_ref_qubits = n_b + 2
        min_ref_tof = max(0, outer_ref_qubits - 1) + max(0, inner_ref_qubits - 1)
        assert lc["cczCount"] >= min_ref_tof, (
            f"N={N},R={R},B={B},C={C}: tof={lc['cczCount']} < min_reflection_tof={min_ref_tof}"
        )
