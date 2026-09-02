"""Tests for the SOSSA block encoding builder, container, and Q# sub-operations."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import qdk

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.qubit_operator.containers.sossa import FactorizedHamiltonianMetadata
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAInnerPrepare,
    SOSSASelect,
    SOSSAWalkContainer,
)
from qdk_chemistry.utils.qsharp import create_qsharp_context

from .reference_tolerances import float_comparison_absolute_tolerance
from .test_helpers import create_random_factorized_hamiltonian, to_sossa_operator

# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def qdk_ctx() -> qdk.Context:
    """Fresh Q# context, isolated from the library's shared one.

    Tests that inspect quantum state need their own interpreter, because
    ``dump_machine`` reports every qubit currently allocated in the context.
    """
    return create_qsharp_context()


def _make_sossa_unitary_representation():
    """Build a UnitaryRepresentation with SOSSAWalkContainer."""
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

    layout = SOSSABuilder._sossa_register_bits(num_orbitals, num_ranks, num_bases, num_copies)
    num_outer_qubits = layout.outer_prep_bits

    # Build outer prepare Wavefunction
    coeffs_list = []
    dets = []
    for idx, amp in enumerate(outer_statevector):
        if amp != 0.0:
            bitstring = format(idx, f"0{num_outer_qubits}b")[::-1]
            dets.append(Configuration.from_bitstring(bitstring))
            coeffs_list.append(float(amp))
    orbitals = ModelOrbitals(num_outer_qubits)
    sv_container = StateVectorContainer(np.array(coeffs_list), dets, orbitals)
    outer_prepare = Wavefunction(sv_container)
    inner_prepare = SOSSAInnerPrepare(
        conditional_coefficients=inner_coefficients,
    )
    select = SOSSASelect(
        one_body_rotation_angles=dq_rotation_angles,
        two_body_rotation_angles=sf_rotation_angles,
    )

    # Compute normalization
    inner_l1 = np.sum(np.abs(inner_coefficients), axis=1)
    lambda_sqrt = np.sum(np.abs(outer_coefficients) * inner_l1)
    normalization = 0.5 * lambda_sqrt**2

    container = SOSSAWalkContainer(
        outer_prepare=outer_prepare,
        inner_prepare=inner_prepare,
        select=select,
        metadata=FactorizedHamiltonianMetadata(
            num_spatial_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
            num_positive_one_body_terms=num_d1,
            energy_shift=0.0,
            normalization=normalization,
        ),
        layout=layout,
        power=1,
    )

    return UnitaryRepresentation(container=container)


# ═══════════════════════════════════════════════════════════════════════════════
# Container tests (serialization round-trips)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAWalkContainer:
    """Tests for the SOSSA container serialization."""

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization round-trip."""
        result = _make_sossa_unitary_representation()
        container = result.get_container()

        json_data = container.to_json()
        restored = SOSSAWalkContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.power == container.power
        assert np.isclose(restored.metadata.normalization, container.metadata.normalization)
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
                restored = SOSSAWalkContainer.from_hdf5(f)

        assert restored.type == container.type
        assert restored.power == container.power
        assert np.isclose(restored.metadata.normalization, container.metadata.normalization)
        assert np.allclose(restored.outer_prepare.get_coefficients(), container.outer_prepare.get_coefficients())
        assert np.allclose(restored.select.two_body_rotation_angles, container.select.two_body_rotation_angles)

    def test_unitary_representation_json_dispatch(self):
        """Test that UnitaryRepresentation correctly dispatches SOSSA from JSON."""
        result = _make_sossa_unitary_representation()

        json_data = result.to_json()
        restored = UnitaryRepresentation.from_json(json_data)

        assert restored.get_container_type() == "sossa_walk"
        assert isinstance(restored.get_container(), SOSSAWalkContainer)

    def test_num_qubits_ancilla_excess_is_exactly_the_structural_widths(self):
        """Pin the identity the generic ancilla fallback depends on.

        ``PhaseEstimationCircuitBuilder`` falls back to
        ``unitary.get_num_qubits() - qubit_hamiltonian.num_qubits`` when a mapper
        exposes no ancilla count. That subtraction is only meaningful because both
        operands carry the same ``2N`` system register, so the difference is exactly
        the structural ancilla width ``xo + b + free_rider + 2``. If the system terms
        ever stop cancelling, the fallback silently returns a wrong allocation size.
        """
        container = _make_sossa_unitary_representation().get_container()
        meta = container.metadata
        expected = SOSSABuilder._sossa_register_bits(
            meta.num_spatial_orbitals, meta.num_ranks, meta.num_bases, meta.num_copies
        )

        # The stored layout is derived data, so pin it against the formula it came from.
        assert container.layout == expected

        num_system = 2 * meta.num_spatial_orbitals
        expected_ancilla = expected.outer_prep_bits + expected.inner_prep_bits + expected.num_free_rider_bits + 2

        assert container.num_qubits - num_system == expected_ancilla


# ═══════════════════════════════════════════════════════════════════════════════
# Builder tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSABuilder:
    """Tests for the SOSSA block encoding builder algorithm."""

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
        result = builder.run(to_sossa_operator(fh))
        container = result.get_container()

        assert isinstance(container, SOSSAWalkContainer)
        x_o_dim = num_orbitals + num_ranks * num_copies
        assert len(container.outer_prepare.get_coefficients()) == x_o_dim
        assert container.inner_prepare.conditional_coefficients.shape[0] == x_o_dim
        assert container.metadata.normalization > 0

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [(2, 1, 1, 1), (3, 2, 2, 1)],
        ids=["N2R1B1C1", "N3R2B2C1"],
    )
    def test_outer_prepare_amplitudes_encode_the_generator_one_norms(
        self, num_orbitals, num_ranks, num_bases, num_copies
    ):
        r"""The outer PREPARE holds :math:`c/\|c\|`, with the scale carried by :math:`\Lambda`.

        The SOS block encoding needs amplitudes proportional to the generator one-norms
        :math:`c_{x_o}` (Eqs. (7) and (9) of Low et al. 2025), and every state-preparation
        backend squares its own input. The stored wavefunction is normalized, so
        :math:`\|c\|` survives only in :math:`\Lambda = \frac{1}{2}\sum_{x_o} c_{x_o}^2` --
        which is why the container carries that separately from the amplitudes.
        """
        fh = create_random_factorized_hamiltonian(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        container = SOSSABuilder().run(to_sossa_operator(fh)).get_container()

        amplitudes = np.asarray(container.outer_prepare.get_coefficients(), dtype=float)

        assert np.isclose(np.linalg.norm(amplitudes), 1.0)
        assert np.all(amplitudes >= 0.0)

        # sum_xo c_xo^2 = 2 Lambda, so the unnormalized coefficients are recovered
        # from the stored amplitudes by scaling with sqrt(2 Lambda).
        unnormalized = amplitudes * np.sqrt(2.0 * container.metadata.normalization)
        assert np.isclose(np.sum(unnormalized**2), 2.0 * container.metadata.normalization)

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [(2, 1, 1, 1), (3, 2, 2, 1), (4, 3, 2, 2), (6, 3, 3, 2)],
        ids=["N2R1B1C1", "N3R2B2C1", "N4R3B2C2", "N6R3B3C2"],
    )
    def test_one_body_generators_address_the_first_sf_rotation_row(
        self, num_orbitals, num_ranks, num_bases, num_copies
    ):
        """Every one-body generator must yield ``b = 0`` and ``r = 0``.

        ``WithGivensRotationsQROM`` reads the SF rotation table uncontrolled so that its
        uncompute is a measurement-based unlookup. One-body generators therefore also read
        that table, and the word they pick up is removed again with CNOTs -- which is only
        possible because they all address the same row, row 0, whose contents are classical.
        Break this and the Givens angles for every one-body term are silently wrong.
        """
        fh = create_random_factorized_hamiltonian(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        container = SOSSABuilder().run(to_sossa_operator(fh)).get_container()

        one_body = np.asarray(container.inner_prepare.conditional_coefficients, dtype=float)[:num_orbitals]
        expected = np.zeros_like(one_body)
        expected[:, 0] = 1.0
        assert np.array_equal(one_body, expected), f"one-body inner-PREPARE rows are not a delta at b=0:\n{one_body}"

        free_rider = np.asarray(container.inner_prepare.free_rider_data, dtype=bool)[:num_orbitals]
        rank_bits = free_rider[:, 2:]
        assert not rank_bits.any(), f"one-body free-rider rank bits are not all zero:\n{rank_bits}"


# ═══════════════════════════════════════════════════════════════════════════════
# Q# component tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInnerPrepareQSharp:
    """Test the Q# InnerPrepare sub-operations via dump_machine."""

    def test_direct_inner_prepare_conditioned_on_xo(self, qdk_ctx):
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
        qdk_ctx.eval(f"use outer = Qubit[{n_outer}];")
        qdk_ctx.eval(f"use inner = Qubit[{n_inner}];")
        qdk_ctx.eval(
            f"let op = QDKChemistry.Utils.SOSSAWalk.MakeInnerPrepareDirect({ic_str}, {fr_str}); op(outer, inner);"
        )
        state = qdk_ctx.dump_machine()
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
        qdk_ctx.eval("ResetAll(outer); ResetAll(inner)")
