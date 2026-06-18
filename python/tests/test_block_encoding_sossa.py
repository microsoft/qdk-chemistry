"""Tests for the SOSSA block encoding builder, container, and Q# sub-operations."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from math import ceil, log2
from pathlib import Path

import h5py
import numpy as np
import pytest
from qdk import qsharp

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import (
    Configuration,
    FactorizedHamiltonianContainer,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAContainer,
    SOSSAInnerPrepare,
    SOSSASelect,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import float_comparison_absolute_tolerance
from .test_helpers import create_test_orbitals

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
    """Create a random FactorizedHamiltonianContainer for testing.

    Args:
        num_orbitals: Number of spatial orbitals (N).
        num_ranks: Number of ranks (R).
        num_bases: Number of bases (B).
        num_copies: Number of copies (C).
        seed: Random seed for reproducibility.

    Returns:
        FactorizedHamiltonianContainer from C++ pybind11.

    """
    rng = np.random.default_rng(seed)
    n, r, b, c = num_orbitals, num_ranks, num_bases, num_copies

    # Symmetric one-body integrals
    h1 = rng.standard_normal((n, n))
    h1 = (h1 + h1.T) / 2

    # Random orthogonal basis vectors (U), flattened [R*B*N]
    u_matrices = np.zeros(r * b * n)
    for ri in range(r):
        for bi in range(b):
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            u_matrices[ri * b * n + bi * n : ri * b * n + (bi + 1) * n] = v

    # Two-body weights W [R*B*C]
    w_matrices = rng.standard_normal(r * b * c)

    # Identity weights WB [R, C]
    wb_matrix = rng.standard_normal((r, c))

    orbitals = create_test_orbitals(n)
    inactive_fock = np.zeros((n, n))

    return FactorizedHamiltonianContainer(
        h1, u_matrices, w_matrices, wb_matrix,
        r, b, c,
        orbitals, 0.0, inactive_fock,
    )


def _make_h2_sossa_unitary_representation():
    """Build a UnitaryRepresentation with SOSSAContainer from H2-like test data.

    Directly constructs the container from known H2-like data (N=2, R=2, B=1, C=1)
    without going through the builder or C++ container. Used for serialization tests.
    """
    num_orbitals = 2
    num_ranks = 2
    num_bases = 1
    num_copies = 1
    num_d1 = 1

    # Outer statevector (already normalized for Prepare)
    outer_coefficients = np.array([0.3, 0.2, 0.5, 0.4])
    l1 = np.sum(np.abs(outer_coefficients))
    outer_statevector = np.sqrt(np.abs(outer_coefficients) / l1)

    # Inner coefficients: [Xo=4, B+1=2]
    inner_coefficients = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.6, 0.4],
        [0.7, 0.3],
    ])

    # Rotation angles
    dq_rotation_angles = np.array([[0.3], [0.5]])
    sf_rotation_angles = np.array([[0.1], [0.2], [0.15], [0.25]])

    x_o_dim = num_orbitals + num_ranks * num_copies
    num_outer_qubits = ceil(log2(x_o_dim)) if x_o_dim > 1 else 1
    num_inner_qubits = ceil(log2(num_bases + 1)) if num_bases + 1 > 1 else 1

    # Build outer prepare Wavefunction
    coeffs_list = []
    dets = []
    for idx, amp in enumerate(outer_statevector):
        if amp != 0.0:
            bitstring = format(idx, f"0{num_outer_qubits}b")
            dets.append(Configuration.from_bitstring(bitstring))
            coeffs_list.append(float(amp))
    orbitals = ModelOrbitals(num_outer_qubits)
    sv_container = StateVectorContainer(np.array(coeffs_list), dets, orbitals)
    outer_prepare = Wavefunction(sv_container)
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

    # Compute normalization
    inner_l1 = np.sum(np.abs(inner_coefficients), axis=1)
    lambda_sqrt = np.sum(np.abs(outer_coefficients) * inner_l1)
    normalization = 0.5 * lambda_sqrt**2

    container = SOSSAContainer(
        outer_prepare=outer_prepare,
        inner_prepare=inner_prepare,
        select=select,
        normalization=normalization,
        power=1,
        quantum_walk=True,
    )

    return UnitaryRepresentation(container=container)


# ═══════════════════════════════════════════════════════════════════════════════
# Container tests (serialization round-trips)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAContainer:
    """Tests for the SOSSA container serialization."""

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization round-trip."""
        result = _make_h2_sossa_unitary_representation()
        container = result.get_container()

        json_data = container.to_json()
        restored = SOSSAContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.power == container.power
        assert restored.quantum_walk == container.quantum_walk
        assert np.isclose(restored.normalization, container.normalization)
        assert np.allclose(
            restored.outer_prepare.get_coefficients(),
            container.outer_prepare.get_coefficients(),
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            restored.inner_prepare.conditional_coefficients,
            container.inner_prepare.conditional_coefficients,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            restored.select.rotation_angles,
            container.select.rotation_angles,
            atol=float_comparison_absolute_tolerance,
        )

    def test_hdf5_roundtrip(self):
        """Test HDF5 serialization/deserialization round-trip."""
        result = _make_h2_sossa_unitary_representation()
        container = result.get_container()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sossa.h5"
            with h5py.File(filepath, "w") as f:
                container.to_hdf5(f)
            with h5py.File(filepath, "r") as f:
                restored = SOSSAContainer.from_hdf5(f)

        assert restored.type == container.type
        assert restored.power == container.power
        assert np.isclose(restored.normalization, container.normalization)
        assert np.allclose(
            restored.outer_prepare.get_coefficients(), container.outer_prepare.get_coefficients()
        )
        assert np.allclose(
            restored.select.sf_rotation_angles, container.select.sf_rotation_angles
        )

    def test_unitary_representation_json_dispatch(self):
        """Test that UnitaryRepresentation correctly dispatches SOSSA from JSON."""
        result = _make_h2_sossa_unitary_representation()

        json_data = result.to_json()
        restored = UnitaryRepresentation.from_json(json_data)

        assert restored.get_container_type() == "sossa"
        assert isinstance(restored.get_container(), SOSSAContainer)

    def test_get_summary(self):
        """Test that get_summary returns a non-empty string."""
        result = _make_h2_sossa_unitary_representation()
        summary = result.get_container().get_summary()

        assert "SOSSA" in summary
        assert "N=2" in summary
        assert "R=2" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# Builder tests (using builder.run() with FactorizedHamiltonianContainer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSABuilder:
    """Tests for the SOSSA block encoding builder algorithm."""

    def test_name_and_type(self):
        """Test algorithm name and type."""
        builder = SOSSABuilder()
        assert builder.name() == "sossa"
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_run_produces_sossa_container(self):
        """Test builder.run() with FactorizedHamiltonianContainer produces SOSSAContainer."""
        fh = _make_random_factorized_hamiltonian(num_orbitals=2, num_ranks=2, num_bases=1, num_copies=1)
        builder = SOSSABuilder()
        result = builder.run(fh)

        assert isinstance(result, UnitaryRepresentation)
        container = result.get_container()
        assert isinstance(container, SOSSAContainer)
        assert container.quantum_walk is True

    def test_run_produces_correct_dimensions(self):
        """Test that run() produces a container with correct dimensions."""
        n, r, b, c = 4, 3, 2, 2
        fh = _make_random_factorized_hamiltonian(num_orbitals=n, num_ranks=r, num_bases=b, num_copies=c)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        assert container.select.num_orbitals == n
        assert container.select.num_ranks == r
        assert container.select.num_bases == b
        assert container.select.num_copies == c

    def test_outer_statevector_normalized(self):
        """Test that outer PREPARE statevector is properly normalized."""
        fh = _make_random_factorized_hamiltonian(num_orbitals=2, num_ranks=2, num_bases=1, num_copies=1)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        sv = container.outer_prepare.get_coefficients()
        assert np.isclose(np.sum(sv**2), 1.0, atol=1e-10)

    def test_normalization_positive(self):
        """Test that normalization Lambda > 0."""
        fh = _make_random_factorized_hamiltonian(num_orbitals=2, num_ranks=2, num_bases=1, num_copies=1)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        assert container.normalization > 0

    def test_inner_coefficients_shape(self):
        """Test inner coefficients have correct shape [Xo, B+1]."""
        n, r, b, c = 3, 2, 2, 1
        fh = _make_random_factorized_hamiltonian(num_orbitals=n, num_ranks=r, num_bases=b, num_copies=c)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        x_o_dim = n + r * c
        inner = container.inner_prepare.conditional_coefficients
        assert inner.shape == (x_o_dim, b + 1)

    def test_rotation_angles_shape(self):
        """Test DQ and SF rotation angles have correct shapes."""
        n, r, b, c = 3, 2, 2, 1
        fh = _make_random_factorized_hamiltonian(num_orbitals=n, num_ranks=r, num_bases=b, num_copies=c)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        # DQ angles: [N, N-1]
        assert container.select.rotation_angles.shape == (n, n - 1)
        # SF angles: [R*(B+1), N] (N-1 Givens + 1 bEqB flag)
        assert container.select.sf_rotation_angles.shape == (r * (b + 1), n)

    def test_power_setting(self):
        """Test power parameter passes through to container."""
        fh = _make_random_factorized_hamiltonian()
        builder = SOSSABuilder(power=3)
        result = builder.run(fh)
        assert result.get_container().power == 3

    def test_quantum_walk_setting(self):
        """Test quantum_walk parameter passes through to container."""
        fh = _make_random_factorized_hamiltonian()
        builder = SOSSABuilder(quantum_walk=False)
        result = builder.run(fh)
        assert result.get_container().quantum_walk is False

    def test_run_impl_requires_factorized_hamiltonian(self):
        """Test that _run_impl raises on invalid input."""
        builder = SOSSABuilder()
        with pytest.raises((TypeError, AttributeError)):
            builder._run_impl(None)

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 2, 2, 1),
            (4, 3, 2, 2),
        ],
        ids=["N2R1B1C1", "N2R2B1C1", "N3R2B2C1", "N4R3B2C2"],
    )
    def test_run_parametrized(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Test builder.run() for various (N, R, B, C) configurations."""
        fh = _make_random_factorized_hamiltonian(
            num_orbitals=num_orbitals, num_ranks=num_ranks,
            num_bases=num_bases, num_copies=num_copies,
        )
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        assert isinstance(container, SOSSAContainer)
        x_o_dim = num_orbitals + num_ranks * num_copies
        assert len(container.outer_prepare.get_coefficients()) == x_o_dim
        assert container.inner_prepare.conditional_coefficients.shape[0] == x_o_dim
        assert container.normalization > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Q# component tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOuterPrepareQSharp:
    """Test the Q# OuterPrepare sub-operations via dump_machine."""

    @pytest.fixture(autouse=True)
    def _reinit_qsharp(self):
        """Re-initialize Q# interpreter for each test."""
        qsharp.init()
        _ = QSHARP_UTILS.SOSSAWalk

    def test_pure_state_preparation(self):
        """Test MakeOuterPreparePureState produces the correct statevector.

        Applies PreparePureStateD to |0⟩ and verifies amplitudes via dump_machine.
        """
        coefficients = [0.5, 0.3, 0.7, 0.1]
        norm = np.sqrt(sum(c**2 for c in coefficients))
        expected = [c / norm for c in coefficients]
        n_qubits = 2

        sv_str = "[" + ", ".join(f"{c:.16f}" for c in coefficients) + "]"
        qsharp.eval(f"use qs = Qubit[{n_qubits}];")
        qsharp.eval(
            f"let op = QDKChemistry.Utils.SOSSAWalk.MakeOuterPreparePureState({sv_str});"
            f" op(qs);"
        )
        state = qsharp.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        # Check amplitudes match expected (up to global phase)
        assert np.allclose(
            np.abs(amplitudes[: len(expected)]),
            np.abs(expected),
            atol=float_comparison_absolute_tolerance,
        )
        qsharp.eval("ResetAll(qs)")

    def test_outer_prepare_identity_vector(self):
        """Test outer prepare with a single-element statevector (delta function)."""
        # Only first element is nonzero → should prepare |0⟩
        coefficients = [1.0, 0.0, 0.0, 0.0]
        n_qubits = 2

        sv_str = "[" + ", ".join(f"{c:.16f}" for c in coefficients) + "]"
        qsharp.eval(f"use qs = Qubit[{n_qubits}];")
        qsharp.eval(
            f"let op = QDKChemistry.Utils.SOSSAWalk.MakeOuterPreparePureState({sv_str});"
            f" op(qs);"
        )
        state = qsharp.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        assert np.abs(amplitudes[0]) > 0.99
        qsharp.eval("ResetAll(qs)")


class TestInnerPrepareQSharp:
    """Test the Q# InnerPrepare sub-operations via dump_machine."""

    @pytest.fixture(autouse=True)
    def _reinit_qsharp(self):
        """Re-initialize Q# interpreter for each test."""
        qsharp.init()
        _ = QSHARP_UTILS.SOSSAWalk

    def test_direct_inner_prepare_conditioned_on_xo(self):
        """Test InnerPrepareDirect: for a fixed x_o, inner register gets correct state.

        Prepares outer register in |x_o⟩, applies inner prepare, checks inner register.
        """
        # 2 outer states, 2 inner states (B+1=2)
        inner_coefficients = [[0.8, 0.6], [0.3, 0.95]]
        n_outer = 1  # ceil(log2(2))
        n_inner = 1  # ceil(log2(2))

        ic_str = "[[0.8, 0.6], [0.3, 0.95]]"

        # Test x_o=0: inner should be proportional to [0.8, 0.6]
        qsharp.eval(f"use outer = Qubit[{n_outer}];")
        qsharp.eval(f"use inner = Qubit[{n_inner}];")
        qsharp.eval(
            f"let op = QDKChemistry.Utils.SOSSAWalk.MakeInnerPrepareDirect({ic_str});"
            f" op(outer, inner);"
        )
        state = qsharp.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        # With outer=|0⟩, the state is |0⟩_outer ⊗ PreparedState(inner_coefficients[0])
        # 2 qubits total: |outer, inner⟩ = |00⟩, |01⟩, |10⟩, |11⟩
        # |0⟩_outer contributes to indices 0 (|00⟩) and 1 (|01⟩)
        expected_inner = np.array(inner_coefficients[0])
        expected_inner = expected_inner / np.linalg.norm(expected_inner)

        actual_inner = amplitudes[:2]  # |00⟩ and |01⟩
        actual_inner_norm = np.abs(actual_inner)

        assert np.allclose(
            actual_inner_norm,
            np.abs(expected_inner),
            atol=float_comparison_absolute_tolerance,
        )
        qsharp.eval("ResetAll(outer); ResetAll(inner)")


class TestSelectQSharp:
    """Test the Q# Select (Givens rotation) sub-operations via dump_machine."""

    @pytest.fixture(autouse=True)
    def _reinit_qsharp(self):
        """Re-initialize Q# interpreter for each test."""
        qsharp.init()
        _ = QSHARP_UTILS.SOSSAWalk

    def test_givens_sequence_single_angle(self):
        """Test ApplyGivensSequence with a single angle applies Ry(2θ)."""
        theta = 0.3
        qsharp.eval("use q = Qubit[1];")
        qsharp.eval(
            f"QDKChemistry.Utils.SOSSAWalk.ApplyGivensSequence([{theta:.16f}], q);"
        )
        state = qsharp.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        # Ry(2θ)|0⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
        expected = [np.cos(theta), np.sin(theta)]
        assert np.allclose(
            np.abs(amplitudes),
            np.abs(expected),
            atol=float_comparison_absolute_tolerance,
        )
        qsharp.eval("ResetAll(q)")

    def test_givens_sequence_two_angles(self):
        """Test ApplyGivensSequence with two angles on 2 qubits."""
        theta0, theta1 = 0.3, 0.5
        qsharp.eval("use q = Qubit[2];")
        qsharp.eval(
            f"QDKChemistry.Utils.SOSSAWalk.ApplyGivensSequence("
            f"[{theta0:.16f}, {theta1:.16f}], q);"
        )
        state = qsharp.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        # Ry(2θ₁) on q[1], then Ry(2θ₀) on q[0], applied to |00⟩
        # q[0]: cos(θ₀)|0⟩ + sin(θ₀)|1⟩
        # q[1]: cos(θ₁)|0⟩ + sin(θ₁)|1⟩
        # State: cos(θ₀)cos(θ₁)|00⟩ + cos(θ₀)sin(θ₁)|01⟩ + sin(θ₀)cos(θ₁)|10⟩ + sin(θ₀)sin(θ₁)|11⟩
        expected = np.array([
            np.cos(theta0) * np.cos(theta1),
            np.cos(theta0) * np.sin(theta1),
            np.sin(theta0) * np.cos(theta1),
            np.sin(theta0) * np.sin(theta1),
        ])
        assert np.allclose(
            np.abs(amplitudes),
            np.abs(expected),
            atol=float_comparison_absolute_tolerance,
        )
        qsharp.eval("ResetAll(q)")


class TestReflectAboutZeroQSharp:
    """Test the Q# ReflectAboutZero operation via dump_machine."""

    @pytest.fixture(autouse=True)
    def _reinit_qsharp(self):
        """Re-initialize Q# interpreter for each test."""
        qsharp.init()
        _ = QSHARP_UTILS.SOSSAWalk

    def test_reflect_zero_state(self):
        """R|0⟩ = +|0⟩ (eigenvalue +1)."""
        qsharp.eval("use q = Qubit[1];")
        qsharp.eval("QDKChemistry.Utils.SOSSAWalk.ReflectAboutZero([q[0]]);")
        state = qsharp.dump_machine()
        assert state.check_eq([1.0, 0.0])
        qsharp.eval("ResetAll(q)")

    def test_reflect_one_state(self):
        """R|1⟩ = -|1⟩ (eigenvalue -1)."""
        qsharp.eval("use q = Qubit[1];")
        qsharp.eval("X(q[0]); QDKChemistry.Utils.SOSSAWalk.ReflectAboutZero([q[0]]);")
        state = qsharp.dump_machine()
        assert state.check_eq([0.0, -1.0])
        qsharp.eval("Reset(q[0])")

    def test_reflect_two_qubit(self):
        """R|00⟩ = +|00⟩ and R|01⟩ = -|01⟩ for two-qubit register."""
        # |00⟩ case
        qsharp.eval("use q = Qubit[2];")
        qsharp.eval("QDKChemistry.Utils.SOSSAWalk.ReflectAboutZero(q);")
        state = qsharp.dump_machine()
        assert state.check_eq([1.0, 0.0, 0.0, 0.0])
        qsharp.eval("ResetAll(q)")

        # |01⟩ case
        qsharp.eval("use q = Qubit[2];")
        qsharp.eval("X(q[1]); QDKChemistry.Utils.SOSSAWalk.ReflectAboutZero(q);")
        state = qsharp.dump_machine()
        assert state.check_eq([0.0, -1.0, 0.0, 0.0])
        qsharp.eval("ResetAll(q)")


class TestMajoranaQSharp:
    """Test the Q# Majorana operations via dump_machine."""

    @pytest.fixture(autouse=True)
    def _reinit_qsharp(self):
        """Re-initialize Q# interpreter for each test."""
        qsharp.init()
        _ = QSHARP_UTILS.SOSSAWalk

    def test_majorana_d1_on_zero(self):
        """MajoranaD1(spin, |0⟩) = X|0⟩ = |1⟩ (ignoring spin CZ when spin=|0⟩)."""
        qsharp.eval("use spin = Qubit[1];")
        qsharp.eval("use sys = Qubit[1];")
        qsharp.eval("QDKChemistry.Utils.SOSSAWalk.MajoranaD1(spin, sys);")
        state = qsharp.dump_machine()
        # 2 qubits: |spin, sys⟩. MajoranaD1 does X(sys[0]) then CZ(spin, sys).
        # spin=|0⟩, sys=|0⟩ → X(sys) → |0,1⟩, CZ(|0⟩,|1⟩) = |0,1⟩
        amplitudes = np.array(state.as_dense_state())
        assert np.abs(amplitudes[1]) > 0.99  # |01⟩
        qsharp.eval("ResetAll(spin); ResetAll(sys)")

    def test_majorana_sf_on_zero(self):
        """MajoranaSF(|0⟩) = Z|0⟩ = |0⟩."""
        qsharp.eval("use sys = Qubit[1];")
        qsharp.eval("QDKChemistry.Utils.SOSSAWalk.MajoranaSF(sys);")
        state = qsharp.dump_machine()
        assert state.check_eq([1.0, 0.0])
        qsharp.eval("ResetAll(sys)")

    def test_majorana_sf_on_one(self):
        """MajoranaSF(|1⟩) = Z|1⟩ = -|1⟩."""
        qsharp.eval("use sys = Qubit[1];")
        qsharp.eval("X(sys[0]); QDKChemistry.Utils.SOSSAWalk.MajoranaSF(sys);")
        state = qsharp.dump_machine()
        assert state.check_eq([0.0, -1.0])
        qsharp.eval("Reset(sys[0])")
