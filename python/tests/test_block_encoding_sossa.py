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
from .test_helpers import create_random_factorized_hamiltonian

# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_sossa_unitary_representation():
    """Build a UnitaryRepresentation with SOSSAContainer"""
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
    inner_coefficients = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.6, 0.4],
            [0.7, 0.3],
        ]
    )

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
        one_body_rotation_angles=dq_rotation_angles,
        two_body_rotation_angles=sf_rotation_angles,
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_copies=num_copies,
        num_bases=num_bases,
        num_positive_one_body_terms=num_d1,
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
    )

    return UnitaryRepresentation(container=container)


# ═══════════════════════════════════════════════════════════════════════════════
# Container tests (serialization round-trips)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAContainer:
    """Tests for the SOSSA container serialization."""

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization round-trip."""
        result = _make_sossa_unitary_representation()
        container = result.get_container()

        json_data = container.to_json()
        restored = SOSSAContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.power == container.power
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
            restored.select.one_body_rotation_angles,
            container.select.one_body_rotation_angles,
            atol=float_comparison_absolute_tolerance,
        )

    def test_hdf5_roundtrip(self):
        """Test HDF5 serialization/deserialization round-trip."""
        result = _make_sossa_unitary_representation()
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
        assert np.allclose(restored.outer_prepare.get_coefficients(), container.outer_prepare.get_coefficients())
        assert np.allclose(restored.select.two_body_rotation_angles, container.select.two_body_rotation_angles)

    def test_unitary_representation_json_dispatch(self):
        """Test that UnitaryRepresentation correctly dispatches SOSSA from JSON."""
        result = _make_sossa_unitary_representation()

        json_data = result.to_json()
        restored = UnitaryRepresentation.from_json(json_data)

        assert restored.get_container_type() == "sossa"
        assert isinstance(restored.get_container(), SOSSAContainer)

    def test_get_summary(self):
        """Test that get_summary returns a non-empty string."""
        result = _make_sossa_unitary_representation()
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


    def test_run_produces_correct_dimensions(self):
        """Test that run() produces a container with correct dimensions."""
        n, r, b, c = 4, 3, 2, 2
        fh = create_random_factorized_hamiltonian(num_orbitals=n, num_ranks=r, num_bases=b, num_copies=c)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        assert container.select.num_orbitals == n
        assert container.select.num_ranks == r
        assert container.select.num_bases == b
        assert container.select.num_copies == c

    def test_outer_statevector_normalized(self):
        """Test that outer PREPARE statevector is properly normalized."""
        fh = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=2, num_bases=1, num_copies=1)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        sv = container.outer_prepare.get_coefficients()
        assert np.isclose(np.sum(sv**2), 1.0, atol=1e-10)

    def test_inner_coefficients_shape(self):
        """Test inner coefficients have correct shape [Xo, B+1]."""
        n, r, b, c = 3, 2, 2, 1
        fh = create_random_factorized_hamiltonian(num_orbitals=n, num_ranks=r, num_bases=b, num_copies=c)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        x_o_dim = n + r * c
        inner = container.inner_prepare.conditional_coefficients
        assert inner.shape == (x_o_dim, b + 1)

    def test_rotation_angles_shape(self):
        """Test DQ and SF rotation angles have correct shapes."""
        n, r, b, c = 3, 2, 2, 1
        fh = create_random_factorized_hamiltonian(num_orbitals=n, num_ranks=r, num_bases=b, num_copies=c)
        builder = SOSSABuilder()
        result = builder.run(fh)
        container = result.get_container()

        # DQ angles: [N, N-1]
        assert container.select.one_body_rotation_angles.shape == (n, n - 1)
        # SF angles: [R*(B+1), N] (N-1 Givens + 1 bEqB flag)
        assert container.select.two_body_rotation_angles.shape == (r * (b + 1), n)

    def test_power_setting(self):
        """Test power parameter passes through to container."""
        fh = create_random_factorized_hamiltonian()
        builder = SOSSABuilder(power=3)
        result = builder.run(fh)
        assert result.get_container().power == 3

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
        fh = create_random_factorized_hamiltonian(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
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
        MakeOuterPreparePureState uses Reversed(register) so coefficient[k]
        appears at bit-reversed dump index.
        """
        coefficients = [0.5, 0.3, 0.7, 0.1]
        norm = np.sqrt(sum(c**2 for c in coefficients))
        n_qubits = 2

        # Build expected in dump_machine order (big-endian):
        # coefficient[k] → bit_reverse(k) in dump output
        n_states = 2**n_qubits
        expected = np.zeros(n_states)
        for k, c in enumerate(coefficients):
            be_idx = int(format(k, f"0{n_qubits}b")[::-1], 2)
            expected[be_idx] = c / norm

        sv_str = "[" + ", ".join(f"{c:.16f}" for c in coefficients) + "]"
        qsharp.eval(f"use qs = Qubit[{n_qubits}];")
        qsharp.eval(f"let op = QDKChemistry.Utils.SOSSAWalk.MakeOuterPreparePureState({sv_str}); op(qs);")
        state = qsharp.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        # Check amplitudes match expected (up to global phase)
        assert np.allclose(
            np.abs(amplitudes[: len(expected)]),
            np.abs(expected),
            atol=float_comparison_absolute_tolerance,
        )
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
        # No free-rider data for this unit test (empty array)
        fr_str = "[]"

        ic_str = "[[0.8, 0.6], [0.3, 0.95]]"

        # Test x_o=0: inner should be proportional to [0.8, 0.6]
        qsharp.eval(f"use outer = Qubit[{n_outer}];")
        qsharp.eval(f"use inner = Qubit[{n_inner}];")
        qsharp.eval(
            f"let op = QDKChemistry.Utils.SOSSAWalk.MakeInnerPrepareDirect({ic_str}, {fr_str}); op(outer, inner);"
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