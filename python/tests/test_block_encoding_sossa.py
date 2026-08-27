"""Tests for the SOSSA block encoding builder, container, and Q# sub-operations."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.qubit_mapper.sossa import SOSSAQubitMapper
from qdk_chemistry.data import (
    Configuration,
    Hamiltonian,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAInnerPrepare,
    SOSSASelect,
    SOSSAWalkContainer,
    sossa_register_bits,
)

from .reference_tolerances import float_comparison_absolute_tolerance
from .test_helpers import create_random_factorized_hamiltonian

# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _to_sossa_operator(factorized_hamiltonian):
    hamiltonian = Hamiltonian(factorized_hamiltonian)
    return SOSSAQubitMapper().run(hamiltonian, None)


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

    _reg_bits = sossa_register_bits(num_orbitals, num_ranks, num_bases, num_copies)
    num_outer_qubits = _reg_bits["xo_bits"]
    num_inner_qubits = _reg_bits["b_bits"]

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
    squared = np.array(coeffs_list) ** 2
    squared /= np.linalg.norm(squared)
    outer_prepare_probabilities = Wavefunction(StateVectorContainer(squared, dets, orbitals))
    inner_prepare = SOSSAInnerPrepare(
        conditional_coefficients=inner_coefficients,
        num_inner_qubits=num_inner_qubits,
    )
    select = SOSSASelect(
        one_body_rotation_angles=dq_rotation_angles,
        two_body_rotation_angles=sf_rotation_angles,
        num_positive_one_body_terms=num_d1,
    )

    # Compute normalization
    inner_l1 = np.sum(np.abs(inner_coefficients), axis=1)
    lambda_sqrt = np.sum(np.abs(outer_coefficients) * inner_l1)
    normalization = 0.5 * lambda_sqrt**2

    container = SOSSAWalkContainer(
        outer_prepare=outer_prepare,
        outer_prepare_probabilities=outer_prepare_probabilities,
        inner_prepare=inner_prepare,
        select=select,
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_bases=num_bases,
        num_copies=num_copies,
        normalization=normalization,
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
        assert np.isclose(restored.normalization, container.normalization)
        assert np.allclose(
            restored.outer_prepare.get_coefficients(),
            container.outer_prepare.get_coefficients(),
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            restored.outer_prepare_probabilities.get_coefficients(),
            container.outer_prepare_probabilities.get_coefficients(),
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
        assert np.isclose(restored.normalization, container.normalization)
        assert np.allclose(restored.outer_prepare.get_coefficients(), container.outer_prepare.get_coefficients())
        assert np.allclose(
            restored.outer_prepare_probabilities.get_coefficients(),
            container.outer_prepare_probabilities.get_coefficients(),
        )
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
        reg_bits = sossa_register_bits(
            container.num_orbitals, container.num_ranks, container.num_bases, container.num_copies
        )

        num_system = 2 * container.num_orbitals
        expected_ancilla = reg_bits["xo_bits"] + reg_bits["b_bits"] + reg_bits["num_free_rider_bits"] + 2

        assert container.num_qubits - num_system == expected_ancilla

    def test_diverging_stored_inner_width_is_rejected(self):
        """A stored inner width that disagrees with (N, R, B, C) must fail loudly.

        ``num_inner_qubits`` is persisted on the inner PREPARE *and* derivable from
        ``sossa_register_bits``. Nothing forces a deserialized value to agree with the
        formula, so version skew could reintroduce a divergence that would otherwise
        corrupt ``num_qubits`` silently.
        """
        container = _make_sossa_unitary_representation().get_container()
        skewed = replace(
            container.inner_prepare,
            num_inner_qubits=container.inner_prepare.num_inner_qubits + 1,
        )

        with pytest.raises(ValueError, match="num_inner_qubits"):
            SOSSAWalkContainer(
                outer_prepare=container.outer_prepare,
                outer_prepare_probabilities=container.outer_prepare_probabilities,
                inner_prepare=skewed,
                select=container.select,
                num_orbitals=container.num_orbitals,
                num_ranks=container.num_ranks,
                num_bases=container.num_bases,
                num_copies=container.num_copies,
                normalization=container.normalization,
            )

    def test_deserializing_a_skewed_inner_width_is_rejected(self):
        """The same guard must hold on the JSON reload path, not just construction."""
        container = _make_sossa_unitary_representation().get_container()
        json_data = container.to_json()
        json_data["inner_prepare"]["num_inner_qubits"] += 1

        with pytest.raises(ValueError, match="num_inner_qubits"):
            SOSSAWalkContainer.from_json(json_data)


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
        result = builder.run(_to_sossa_operator(fh))
        container = result.get_container()

        assert isinstance(container, SOSSAWalkContainer)
        x_o_dim = num_orbitals + num_ranks * num_copies
        assert len(container.outer_prepare.get_coefficients()) == x_o_dim
        assert container.inner_prepare.conditional_coefficients.shape[0] == x_o_dim
        assert container.normalization > 0

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [(2, 1, 1, 1), (3, 2, 2, 1)],
        ids=["N2R1B1C1", "N3R2B2C1"],
    )
    def test_outer_prepare_probabilities_are_the_squared_amplitudes(
        self, num_orbitals, num_ranks, num_bases, num_copies
    ):
        r"""Verify the builder emits both conventions of the outer PREPARE distribution.

        The SOS block encoding needs the outer PREPARE to produce amplitudes
        proportional to the generator one-norms :math:`c_{x_o}` (Eqs. (7) and (9) of Low et
        al. 2025), so ``outer_prepare`` holds :math:`c/\|c\|`. Backends that
        discretize their input as a probability distribution have to be handed
        :math:`c^2` instead, which is what ``outer_prepare_probabilities`` holds.
        Also check the normalization identity :math:`\sum_{x_o} c_{x_o}^2 = 2\Lambda`.
        """
        fh = create_random_factorized_hamiltonian(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        container = SOSSABuilder().run(_to_sossa_operator(fh)).get_container()

        amplitudes = np.asarray(container.outer_prepare.get_coefficients(), dtype=float)
        probabilities = np.asarray(container.outer_prepare_probabilities.get_coefficients(), dtype=float)

        assert np.isclose(np.linalg.norm(amplitudes), 1.0)
        assert np.isclose(np.linalg.norm(probabilities), 1.0)
        assert np.all(amplitudes >= 0.0)

        expected = amplitudes**2
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(probabilities, expected, atol=float_comparison_absolute_tolerance)

        # sum_xo c_xo^2 = 2 Lambda, so the unnormalized coefficients are recovered
        # from the stored amplitudes by scaling with sqrt(2 Lambda).
        unnormalized = amplitudes * np.sqrt(2.0 * container.normalization)
        assert np.isclose(np.sum(unnormalized**2), 2.0 * container.normalization)


# ═══════════════════════════════════════════════════════════════════════════════
# Q# component tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOuterPrepareQSharp:
    """Test the Q# OuterPrepare sub-operations via dump_machine."""

    def test_pure_state_preparation(self, qdk_ctx):
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
        qdk_ctx.eval(f"use qs = Qubit[{n_qubits}];")
        qdk_ctx.eval(f"let op = QDKChemistry.Utils.SOSSAWalk.MakeOuterPreparePureState({sv_str}); op(qs);")
        state = qdk_ctx.dump_machine()
        amplitudes = np.array(state.as_dense_state())

        # Check amplitudes match expected (up to global phase)
        assert np.allclose(
            np.abs(amplitudes[: len(expected)]),
            np.abs(expected),
            atol=float_comparison_absolute_tolerance,
        )
        qdk_ctx.eval("ResetAll(qs)")


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
