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
        power=1,
    )

    return UnitaryRepresentation(container=container)


def _assert_sossa_containers_equal(actual: SOSSAWalkContainer, expected: SOSSAWalkContainer) -> None:
    """Assert equality of every serialized SOSSA walk field."""
    assert actual.type == expected.type
    assert actual.power == expected.power
    assert actual.normalization == pytest.approx(expected.normalization)
    assert actual.has_lambda_eff == expected.has_lambda_eff
    assert actual.metadata == expected.metadata
    assert actual.layout == expected.layout

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


class TestSOSSAWalkContainer:
    """Tests for the SOSSA container serialization."""

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization round-trip."""
        result = _make_sossa_unitary_representation()
        container = result.get_container()

        json_data = container.to_json()
        restored = SOSSAWalkContainer.from_json(json_data)

        _assert_sossa_containers_equal(restored, container)

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

    def test_lambda_eff_raises_when_no_reference_energy_was_supplied(self):
        """An unset ``lambda_eff`` must announce itself, not masquerade as a number.

        The block encoding is fully buildable without a reference energy -- the circuit
        never uses ``lambda_eff`` -- so the container has to represent "not supplied"
        somehow. Returning ``0.0`` or ``Lambda`` would silently mis-size a query
        schedule, which is the one thing this quantity exists to do correctly.
        """
        container = _make_sossa_unitary_representation().get_container()

        assert container.has_lambda_eff is False
        with pytest.raises(ValueError, match="lambda_eff is unset"):
            _ = container.lambda_eff

    def test_lambda_eff_round_trips_through_json_and_hdf5(self):
        """A stored ``lambda_eff`` must survive serialization in both formats.

        ``lambda_eff`` is derived from an input the factorization does not carry, so if
        serialization dropped it a reloaded container would be indistinguishable from one
        that was never given a reference energy -- and would raise instead of returning
        the value the builder computed.
        """
        container = _make_sossa_unitary_representation().get_container()
        rebuilt = SOSSAWalkContainer(
            outer_prepare=container.outer_prepare,
            inner_prepare=container.inner_prepare,
            select=container.select,
            metadata=container.metadata,
            layout=container.layout,
            normalization=container.normalization,
            lambda_eff=0.75,
        )

        from_json = SOSSAWalkContainer.from_json(rebuilt.to_json())
        assert from_json.lambda_eff == pytest.approx(0.75, abs=1e-12)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "walk.h5"
            with h5py.File(path, "w") as handle:
                rebuilt.to_hdf5(handle.create_group("walk"))
            with h5py.File(path, "r") as handle:
                from_hdf5 = SOSSAWalkContainer.from_hdf5(handle["walk"])
        assert from_hdf5.lambda_eff == pytest.approx(0.75, abs=1e-12)

        # Positive control: the unset case must round trip as unset, not as 0.0.
        assert SOSSAWalkContainer.from_json(container.to_json()).has_lambda_eff is False


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
        r"""The outer PREPARE holds :math:`c/\|c\|`, with the scale carried by :math:`\Lambda`.

        The SOS block encoding needs amplitudes proportional to the generator one-norms
        :math:`c_{x_o}` (Eqs. (7) and (9) of Low et al. 2025), and every state-preparation
        backend squares its own input. The stored wavefunction is normalized, so
        :math:`\|c\|` survives only in :math:`\Lambda = \frac{1}{2}\sum_{x_o} c_{x_o}^2` --
        which is why the container carries that separately from the amplitudes.
        """
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

    def test_two_body_rotation_angles_are_padded_in_basis_major_order(self):
        """The QROM table must be basis-major with one trailing identity block."""
        operator = to_sossa_operator(create_random_factorized_hamiltonian(3, 2, 2, 1))
        sossa = operator.get_container()
        sf_angles = np.array(
            [
                [0.0, 0.1],  # rank 0, basis 0
                [1.0, 1.1],  # rank 0, basis 1
                [2.0, 2.1],  # rank 1, basis 0
                [3.0, 3.1],  # rank 1, basis 1
            ]
        )
        sossa.two_body.angles[...] = sf_angles

        container = SOSSABuilder().run(operator).get_container()
        actual = container.select.two_body_rotation_angles

        expected = np.array(
            [
                [0.0, 0.1],
                [2.0, 2.1],
                [1.0, 1.1],
                [3.0, 3.1],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_lambda_eff_at_band_centre_equals_the_normalization(self):
        r"""At the middle of the band, :math:`\lambda_{\text{eff}}` must collapse to :math:`\Lambda`.

        ``sqrt(E_gap (2L - E_gap))`` is a semicircle over ``E_gap in [0, 2L]`` peaking at
        ``E_gap = L``, where it equals ``L`` exactly. That single point pins the whole
        closed form without diagonalizing anything: an implementation that dropped the
        ``2`` would return ``0`` here, and the asymptotic ``sqrt(2 L E_gap)`` of the
        paper's abstract would return ``sqrt(2) L``.
        """
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 1, 1))
        lam = SOSSABuilder().run(operator).get_container().normalization

        container = SOSSABuilder(reference_energy_gap=lam).run(operator).get_container()

        assert container.lambda_eff == pytest.approx(lam, abs=1e-12)
        # Discriminate against the two plausible wrong forms at the same point.
        assert container.lambda_eff != pytest.approx(np.sqrt(2.0) * lam, abs=1e-6)

    def test_lambda_eff_is_symmetric_about_the_band_centre(self):
        """Gaps mirrored about ``Lambda`` must give the same value.

        ``E_gap`` and ``2L - E_gap`` are interchangeable in ``E_gap (2L - E_gap)``, so a
        term-ordering or sign slip that broke the symmetry would show up here even
        though the band-centre pin above is blind to it.
        """
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 1, 1))
        lam = SOSSABuilder().run(operator).get_container().normalization

        low = SOSSABuilder(reference_energy_gap=0.25 * lam).run(operator).get_container().lambda_eff
        high = SOSSABuilder(reference_energy_gap=1.75 * lam).run(operator).get_container().lambda_eff

        assert low == pytest.approx(high, abs=1e-12)
        # Guard against the degenerate pass where both sides are the band-centre value.
        assert low < lam

    def test_ground_state_energy_and_energy_gap_agree_through_the_shift(self):
        r"""The two settings must be the same statement of the same reference point.

        ``reference_energy_gap`` is :math:`E_{\text{gs}} - E_{\text{SOS}}`, so supplying either one
        has to land on an identical :math:`\lambda_{\text{eff}}`. Only the builder knows
        ``energy_shift``, which is why it offers both spellings rather than making every
        caller do the subtraction. A sign slip in that conversion would pass a test that
        used just one of the settings.
        """
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
        """``reference_ground_state_energy`` and ``reference_energy_gap`` are two spellings of one input.

        Honouring one and ignoring the other would let a caller believe a reference energy
        took effect when it silently did not.
        """
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
        """A gap outside ``(0, 2 Lambda)`` is a caller error, not a value to be clamped.

        ``H_SOS`` is positive semidefinite and block encoded with normalization
        ``Lambda``, so its spectrum lies in ``[0, 2 Lambda]`` by construction
        (:cite:`Low2025`, Sec. II). A reference energy that lands outside that window
        belongs to a different operator, and returning ``0.0`` there would be
        indistinguishable from the genuinely small ``lambda_eff`` of a near
        frustration-free Hamiltonian -- the regime the quantity exists to report.
        """
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 1, 1))
        lam = SOSSABuilder().run(operator).get_container().normalization

        with pytest.raises(ValueError, match="outside the representable window"):
            SOSSABuilder(reference_energy_gap=gap_fraction * lam).run(operator)

    def test_omitting_the_reference_energy_still_builds_the_block_encoding(self):
        """The circuit never consumes ``lambda_eff``, so it must not be required to build one.

        ``lambda_eff`` sizes a query schedule; the SOSSA circuit mapper reads only the
        PREPARE/SELECT data and the register layout. Making the reference energy mandatory
        would block every circuit-only caller -- including a bare
        ``AlgorithmRef("hamiltonian_unitary_builder", "sossa")`` nested inside a QPE
        circuit builder, which has no way to pass one.
        """
        operator = to_sossa_operator(create_random_factorized_hamiltonian(2, 2, 1, 1))

        container = SOSSABuilder().run(operator).get_container()

        assert container.normalization > 0
        assert container.has_lambda_eff is False
