"""Tests for the SOSSA block encoding builder and container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import BlockEncodingContainer
from qdk_chemistry.data.unitary_representation.containers.sossa import (
    SOSSAContainer,
    SOSSAInnerPrepare,
    SOSSAOuterPrepare,
    SOSSASelect,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _make_h2_dfthc_data():
    """Create minimal H2-like DFTHC test data (N=2, R=2, B=1, C=1)."""
    num_orbitals = 2
    num_ranks = 2
    num_bases = 1
    num_copies = 1
    num_d1 = 1  # 1 D1 entry

    # Xo = num_orbitals + num_ranks * num_copies = 4
    # Outer coefficients: [D1, Q1, SF_r0c0, SF_r1c0]
    outer_coefficients = np.array([0.3, 0.2, 0.5, 0.4])
    # Inner coefficients: shape [4, 2] (B+1=2)
    inner_coefficients = np.array([
        [1.0, 0.0],  # D1: all weight on b=0
        [1.0, 0.0],  # Q1: all weight on b=0
        [0.6, 0.4],  # SF r=0: split across b=0, b=1
        [0.7, 0.3],  # SF r=1: split across b=0, b=1
    ])
    # Rotation angles: [N, N-1] = [2, 1]
    dq_rotation_angles = np.array([
        [0.3],  # D1 orbital 0
        [0.5],  # Q1 orbital 1
    ])
    # SF rotation angles: [R*(B+1), N-1] = [4, 1]
    sf_rotation_angles = np.array([
        [0.1],  # r=0, b=0
        [0.2],  # r=0, b=1
        [0.15],  # r=1, b=0
        [0.25],  # r=1, b=1
    ])

    return {
        "num_orbitals": num_orbitals,
        "num_ranks": num_ranks,
        "num_bases": num_bases,
        "num_copies": num_copies,
        "outer_coefficients": outer_coefficients,
        "inner_coefficients": inner_coefficients,
        "dq_rotation_angles": dq_rotation_angles,
        "sf_rotation_angles": sf_rotation_angles,
        "num_d1": num_d1,
    }


class TestSOSSAContainer:
    """Tests for the SOSSA container data class."""

    def test_container_creation(self):
        """Test that SOSSAContainer can be created with valid data."""
        data = _make_h2_dfthc_data()
        outer_prep = SOSSAOuterPrepare(
            statevector=np.sqrt(np.abs(data["outer_coefficients"]) / np.sum(np.abs(data["outer_coefficients"]))),
            num_outer_qubits=2,
        )
        inner_prep = SOSSAInnerPrepare(
            conditional_coefficients=data["inner_coefficients"],
            num_inner_qubits=1,
            num_bases=data["num_bases"],
        )
        select = SOSSASelect(
            rotation_angles=data["dq_rotation_angles"],
            sf_rotation_angles=data["sf_rotation_angles"],
            num_orbitals=data["num_orbitals"],
            num_ranks=data["num_ranks"],
            num_copies=data["num_copies"],
            num_bases=data["num_bases"],
            num_d1=data["num_d1"],
        )
        container = SOSSAContainer(
            outer_prepare=outer_prep,
            inner_prepare=inner_prep,
            select=select,
            normalization=1.5,
        )

        assert container.type == "sossa"
        assert container.power == 1
        assert container.quantum_walk is True
        assert isinstance(container, BlockEncodingContainer)

    def test_container_type_and_properties(self):
        """Test container properties match input parameters."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
        container = result.get_container()

        assert isinstance(container, SOSSAContainer)
        assert container.select.num_orbitals == 2
        assert container.select.num_ranks == 2
        assert container.select.num_bases == 1
        assert container.select.num_copies == 1
        assert container.normalization > 0

    def test_json_roundtrip(self):
        """Test JSON serialization/deserialization round-trip."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
        container = result.get_container()

        json_data = container.to_json()
        restored = SOSSAContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.power == container.power
        assert restored.quantum_walk == container.quantum_walk
        assert np.isclose(restored.normalization, container.normalization)
        assert np.allclose(
            restored.outer_prepare.statevector,
            container.outer_prepare.statevector,
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
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
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
            restored.outer_prepare.statevector, container.outer_prepare.statevector
        )
        assert np.allclose(
            restored.select.sf_rotation_angles, container.select.sf_rotation_angles
        )

    def test_unitary_representation_json_dispatch(self):
        """Test that UnitaryRepresentation correctly dispatches SOSSA from JSON."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)

        json_data = result.to_json()
        restored = UnitaryRepresentation.from_json(json_data)

        assert restored.get_container_type() == "sossa"
        assert isinstance(restored.get_container(), SOSSAContainer)

    def test_get_summary(self):
        """Test that get_summary returns a non-empty string."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
        summary = result.get_container().get_summary()

        assert "SOSSA" in summary
        assert "N=2" in summary
        assert "R=2" in summary


class TestSOSSABuilder:
    """Tests for the SOSSA block encoding builder algorithm."""

    def test_build_from_dfthc_basic(self):
        """Test basic DFTHC data produces a valid SOSSAContainer."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)

        assert isinstance(result, UnitaryRepresentation)
        container = result.get_container()
        assert isinstance(container, SOSSAContainer)
        assert container.quantum_walk is True

    def test_build_from_dfthc_normalization(self):
        """Test that normalization is computed correctly.

        Lambda = (1/2) * lambda_sqrt^2 where
        lambda_sqrt = sum_xo |outer[xo]| * sum_b |inner[xo, b]|
        """
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
        container = result.get_container()

        # Manual computation
        outer = data["outer_coefficients"]
        inner = data["inner_coefficients"]
        inner_l1 = np.sum(np.abs(inner), axis=1)
        lambda_sqrt = np.sum(np.abs(outer) * inner_l1)
        expected_norm = 0.5 * lambda_sqrt**2

        assert np.isclose(
            container.normalization,
            expected_norm,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_outer_statevector_normalized(self):
        """Test that outer PREPARE statevector is properly normalized."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
        container = result.get_container()

        sv = container.outer_prepare.statevector
        # Squared magnitudes should sum to 1 (approximately, within padded length)
        assert np.isclose(np.sum(sv**2), 1.0, atol=1e-10)

    def test_outer_statevector_encodes_sqrt_coefficients(self):
        """Test outer statevector = sqrt(|alpha_xo| / l1_norm)."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder()
        result = builder.build_from_dfthc(**data)
        container = result.get_container()

        outer = data["outer_coefficients"]
        l1 = np.sum(np.abs(outer))
        expected = np.sqrt(np.abs(outer) / l1)

        assert np.allclose(
            container.outer_prepare.statevector,
            expected,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_power_setting(self):
        """Test power parameter passes through to container."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder(power=3)
        result = builder.build_from_dfthc(**data)
        assert result.get_container().power == 3

    def test_quantum_walk_setting(self):
        """Test quantum_walk parameter passes through to container."""
        data = _make_h2_dfthc_data()
        builder = SOSSABuilder(quantum_walk=False)
        result = builder.build_from_dfthc(**data)
        assert result.get_container().quantum_walk is False

    def test_run_impl_raises(self):
        """Test that _run_impl (from QubitHamiltonian) raises NotImplementedError."""
        builder = SOSSABuilder()
        with pytest.raises(NotImplementedError, match="build_from_dfthc"):
            builder.run(None)

    def test_name_and_type(self):
        """Test algorithm name and type."""
        builder = SOSSABuilder()
        assert builder.name() == "sossa"
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_invalid_outer_coefficients_length(self):
        """Test validation of outer_coefficients length."""
        data = _make_h2_dfthc_data()
        data["outer_coefficients"] = np.array([0.5, 0.3])  # Wrong length (should be 4)
        builder = SOSSABuilder()
        with pytest.raises(ValueError, match="outer_coefficients length"):
            builder.build_from_dfthc(**data)

    def test_invalid_inner_coefficients_shape(self):
        """Test validation of inner_coefficients shape."""
        data = _make_h2_dfthc_data()
        data["inner_coefficients"] = np.array([[0.5, 0.3, 0.2]] * 4)  # Wrong B+1
        builder = SOSSABuilder()
        with pytest.raises(ValueError, match="inner_coefficients shape"):
            builder.build_from_dfthc(**data)

    def test_zero_normalization_raises(self):
        """Test that zero normalization raises ValueError."""
        data = _make_h2_dfthc_data()
        data["outer_coefficients"] = np.zeros(4)
        builder = SOSSABuilder()
        with pytest.raises(ValueError, match="too small"):
            builder.build_from_dfthc(**data)
