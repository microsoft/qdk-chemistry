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

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    QubitOperator,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.qubit_operator.containers.sos import FactorizedHamiltonianMetadata, SOSContainer
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAInnerPrepare,
    SOSSARegisterLayout,
    SOSSASelect,
    SOSSAWalkContainer,
)

from .test_helpers import create_random_factorized_hamiltonian, to_sossa_operator


def _make_sossa_unitary_representation(*, power: int = 1, lambda_eff: float | None = None):
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

    outer_prep_dim = num_orbitals + num_ranks * num_copies
    rank_bits = ceil(log2(num_ranks)) if num_ranks > 1 else 0
    layout = SOSSARegisterLayout(
        outer_prep_bits=ceil(log2(outer_prep_dim)) if outer_prep_dim > 1 else 1,
        inner_prep_bits=ceil(log2(num_bases + 1)) if num_bases + 1 > 1 else 1,
        rank_bits=rank_bits,
        num_free_rider_bits=2 + rank_bits,
    )
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
        free_rider_data=np.array(
            [
                [False, False, False],  # D1, rank 0
                [False, True, False],  # Q1, rank 0
                [True, True, False],  # SF, rank 0
                [True, True, True],  # SF, rank 1
            ]
        ),
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
        ),
        layout=layout,
        normalization=normalization,
        power=power,
        lambda_eff=lambda_eff,
    )

    return UnitaryRepresentation(container=container)


def _assert_sossa_containers_equal(actual: SOSSAWalkContainer, expected: SOSSAWalkContainer) -> None:
    """Assert equality of every serialized SOSSA walk field."""
    assert actual.type == expected.type
    assert actual.power == expected.power
    assert actual.normalization == pytest.approx(expected.normalization)
    assert actual.has_lambda_eff == expected.has_lambda_eff
    if expected.has_lambda_eff:
        assert actual.lambda_eff == pytest.approx(expected.lambda_eff)
    assert actual.metadata == expected.metadata
    assert actual.layout == expected.layout

    assert actual.outer_prepare.get_container_type() == expected.outer_prepare.get_container_type()
    assert actual.outer_prepare.to_json() == expected.outer_prepare.to_json()
    np.testing.assert_allclose(actual.outer_prepare.get_coefficients(), expected.outer_prepare.get_coefficients())
    assert list(actual.outer_prepare.get_active_determinants()) == list(
        expected.outer_prepare.get_active_determinants()
    )
    assert actual.outer_prepare.get_orbitals().num_modes() == expected.outer_prepare.get_orbitals().num_modes()
    np.testing.assert_allclose(
        actual.inner_prepare.conditional_coefficients,
        expected.inner_prepare.conditional_coefficients,
    )
    np.testing.assert_array_equal(actual.inner_prepare.free_rider_data, expected.inner_prepare.free_rider_data)
    np.testing.assert_allclose(actual.select.one_body_rotation_angles, expected.select.one_body_rotation_angles)
    np.testing.assert_allclose(actual.select.two_body_rotation_angles, expected.select.two_body_rotation_angles)
    assert actual.to_json() == expected.to_json()


class TestSOSSAWalkContainer:
    """Tests for the SOSSA container serialization."""

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization round-trip."""
        result = _make_sossa_unitary_representation(power=3, lambda_eff=0.75)
        container = result.get_container()

        json_data = container.to_json()
        restored = SOSSAWalkContainer.from_json(json_data)

        _assert_sossa_containers_equal(restored, container)

    def test_hdf5_roundtrip(self):
        """Test HDF5 serialization/deserialization round-trip."""
        result = _make_sossa_unitary_representation(power=3, lambda_eff=0.75)
        container = result.get_container()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sossa.h5"
            with h5py.File(filepath, "w") as f:
                container.to_hdf5(f)
            with h5py.File(filepath, "r") as f:
                restored = SOSSAWalkContainer.from_hdf5(f)

        _assert_sossa_containers_equal(restored, container)

    def test_unitary_representation_json_dispatch(self):
        """Test that UnitaryRepresentation correctly dispatches SOSSA from JSON."""
        result = _make_sossa_unitary_representation()

        json_data = result.to_json()
        restored = UnitaryRepresentation.from_json(json_data)

        assert restored.get_container_type() == "sossa_walk"
        assert isinstance(restored.get_container(), SOSSAWalkContainer)

    def test_num_qubits_ancilla_excess_is_exactly_the_structural_widths(self):
        """Pin every register width for the fixed N=2, R=2, B=1, C=1 fixture."""
        container = _make_sossa_unitary_representation().get_container()
        expected_layout = SOSSARegisterLayout(
            outer_prep_bits=2,
            inner_prep_bits=1,
            rank_bits=1,
            num_free_rider_bits=3,
        )

        assert container.layout == expected_layout
        assert container.num_qubits == 12

    def test_a_layout_reserving_free_rider_bits_requires_a_matching_table(self):
        """SELECT reads those bits whether or not anything wrote them.

        ``MakeFreeRiderLoadOp([])`` is a no-op and the alias PREPARE skips the word when the
        table is empty, so an absent table leaves ``isSF`` at zero and routes every spin-free
        term as DQ -- a wrong operator that still builds, simulates and resource-estimates.
        A table of the wrong width misaligns ``isSF``/``dvsq`` the same way.
        """
        container = _make_sossa_unitary_representation().get_container()
        coefficients = container.inner_prepare.conditional_coefficients
        table = container.inner_prepare.free_rider_data
        assert container.layout.num_free_rider_bits > 0, "fixture must reserve the bits under test"

        def rebuild(inner_prepare):
            return SOSSAWalkContainer(
                outer_prepare=container.outer_prepare,
                inner_prepare=inner_prepare,
                select=container.select,
                metadata=container.metadata,
                layout=container.layout,
                normalization=container.normalization,
            )

        with pytest.raises(ValueError, match="no free_rider_data"):
            rebuild(SOSSAInnerPrepare(coefficients))
        with pytest.raises(ValueError, match="must have shape"):
            rebuild(SOSSAInnerPrepare(coefficients, table[:, :-1]))
        with pytest.raises(ValueError, match="must have shape"):
            rebuild(SOSSAInnerPrepare(coefficients, table[:-1]))

    def test_lambda_eff_raises_when_no_reference_energy_was_supplied(self):
        """An unset ``lambda_eff`` must announce itself, not masquerade as a number."""
        container = _make_sossa_unitary_representation().get_container()

        assert container.has_lambda_eff is False
        with pytest.raises(ValueError, match="lambda_eff is unset"):
            _ = container.lambda_eff


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
        assert container.normalization > 0

    def test_outer_prepare_amplitudes_encode_hand_calculated_generator_weights(self):
        r"""The outer PREPARE holds :math:`c/\|c\|`, with the scale carried by :math:`\Lambda`."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 1, 1, 1))
        sossa = operator.get_container()
        sossa.one_body.coeffs[...] = np.array([[1.0, -2.0j], [-3.0, 4.0j]])
        sossa.two_body.coeffs[...] = np.array([[-5.0, 6.0]])

        container = SOSSABuilder().run(operator).get_container()

        amplitudes = np.asarray(container.outer_prepare.get_coefficients(), dtype=float)
        expected_weights = np.array([3.0 * np.sqrt(2.0), 7.0 * np.sqrt(2.0), 11.0 / np.sqrt(2.0)])
        expected_normalization = 0.5 * np.sum(expected_weights**2)
        expected_amplitudes = expected_weights / np.linalg.norm(expected_weights)

        np.testing.assert_allclose(amplitudes, expected_amplitudes)
        assert container.normalization == pytest.approx(expected_normalization)

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [(2, 1, 1, 1), (3, 2, 2, 1), (4, 3, 2, 2), (6, 3, 3, 2)],
        ids=["N2R1B1C1", "N3R2B2C1", "N4R3B2C2", "N6R3B3C2"],
    )
    def test_one_body_generators_address_the_first_sf_rotation_row(
        self, num_orbitals, num_ranks, num_bases, num_copies
    ):
        """Every one-body generator must yield ``b = 0`` and ``r = 0``."""
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

    def test_free_rider_data_encodes_generator_flags_copies_and_nonzero_ranks(self):
        """Pin D1/Q1/SF flags and little-endian ranks for multiple copies."""
        source = to_sossa_operator(create_random_factorized_hamiltonian(3, 3, 1, 2)).get_container()
        operator = QubitOperator(
            SOSContainer(
                source.one_body,
                source.two_body,
                source.encoding,
                source.fermion_mode_order,
                FactorizedHamiltonianMetadata(
                    num_spatial_orbitals=3,
                    num_ranks=3,
                    num_bases=1,
                    num_copies=2,
                    num_positive_one_body_terms=1,
                    energy_shift=source.metadata.energy_shift,
                ),
            )
        )

        container = SOSSABuilder().run(operator).get_container()
        actual = np.asarray(container.inner_prepare.free_rider_data, dtype=bool)
        expected = np.array(
            [
                [False, False, False, False],  # D1, rank 0
                [False, True, False, False],  # Q1, rank 0
                [False, True, False, False],  # Q1, rank 0
                [True, True, False, False],  # SF, rank 0, copy 0
                [True, True, False, False],  # SF, rank 0, copy 1
                [True, True, True, False],  # SF, rank 1, copy 0
                [True, True, True, False],  # SF, rank 1, copy 1
                [True, True, False, True],  # SF, rank 2, copy 0
                [True, True, False, True],  # SF, rank 2, copy 1
            ],
            dtype=bool,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_inner_prepare_uses_signed_square_roots_of_all_spin_free_weights(self):
        """Inner PREPARE must linearize every SF weight and preserve its SELECT sign."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 2, 1))
        sossa = operator.get_container()
        weights = np.array([[4.0, -9.0, 16.0], [-1.0, 0.0, -25.0]])
        sossa.two_body.coeffs[...] = weights

        container = SOSSABuilder().run(operator).get_container()
        actual = np.asarray(container.inner_prepare.conditional_coefficients, dtype=float)[-len(weights) :]
        expected = np.sign(weights) * np.sqrt(np.abs(weights))

        np.testing.assert_allclose(actual, expected)
        probabilities = actual**2 / np.sum(actual**2, axis=1, keepdims=True)
        expected_probabilities = np.abs(weights) / np.sum(np.abs(weights), axis=1, keepdims=True)
        np.testing.assert_allclose(probabilities, expected_probabilities)

    def test_inner_prepare_pads_zero_spin_free_rows_with_identity(self):
        """Unreachable SF rows must still define a valid conditional distribution."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 2, 1))
        sossa = operator.get_container()
        sossa.two_body.coeffs[...] = np.array([[0.0, 0.0, 0.0], [4.0, -9.0, 16.0]])

        container = SOSSABuilder().run(operator).get_container()
        actual = np.asarray(container.inner_prepare.conditional_coefficients, dtype=float)[-2:]

        np.testing.assert_allclose(actual, np.array([[0.0, 0.0, 1.0], [2.0, -3.0, 4.0]]))

    def test_ground_state_energy_and_energy_gap_agree_through_the_shift(self):
        r"""The two settings must be the same statement of the same reference point."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(3, 2, 2, 1))
        probe = SOSSABuilder().run(operator).get_container()
        shift = probe.metadata.energy_shift
        gap = 0.4 * probe.normalization
        assert shift != 0.0, "a zero shift would make this conversion check vacuous"

        from_gap = SOSSABuilder(reference_energy_gap=gap).run(operator).get_container().lambda_eff
        from_energy = SOSSABuilder(reference_ground_state_energy=shift + gap).run(operator).get_container().lambda_eff

        assert from_energy == pytest.approx(from_gap, abs=1e-12)

        # Positive control: forgetting to apply the shift must be detectable -- either it
        # lands outside the window (rejected) or it lands on a different value. Passing a
        # bare gap as if it were an absolute energy must never silently agree.
        def lambda_eff_or_none(**settings):
            try:
                return SOSSABuilder(**settings).run(operator).get_container().lambda_eff
            except ValueError:
                return None

        assert lambda_eff_or_none(reference_ground_state_energy=gap) != from_gap

    def test_supplying_both_reference_energies_is_rejected(self):
        """``reference_ground_state_energy`` and ``reference_energy_gap`` are two spellings of one input."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 1, 1))
        lam = SOSSABuilder().run(operator).get_container().normalization

        with pytest.raises(ValueError, match="not both"):
            SOSSABuilder(reference_ground_state_energy=0.0, reference_energy_gap=lam).run(operator)

    @pytest.mark.parametrize(
        "gap_fraction",
        [-0.5, 0.0, 2.0, 2.5],
        ids=["below_shift", "at_lower_band_edge", "at_upper_band_edge", "above_upper_band_edge"],
    )
    def test_lambda_eff_rejects_gaps_outside_the_band(self, gap_fraction):
        """A gap outside ``(0, 2 Lambda)`` is a caller error, not a value to be clamped."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 1, 1))
        lam = SOSSABuilder().run(operator).get_container().normalization

        with pytest.raises(ValueError, match="outside the representable window"):
            SOSSABuilder(reference_energy_gap=gap_fraction * lam).run(operator)
