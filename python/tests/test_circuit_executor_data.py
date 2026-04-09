"""Tests for CircuitExecutorData class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import pytest

from qdk_chemistry.data.circuit_executor_data import CircuitExecutorData


@pytest.fixture
def sample_bitstring_counts() -> dict[str, int]:
    """Sample bitstring counts for testing."""
    return {
        "00": 250,
        "01": 150,
        "10": 300,
        "11": 300,
    }


@pytest.fixture
def sample_executor_data(sample_bitstring_counts) -> CircuitExecutorData:
    """Sample CircuitExecutorData instance for testing."""
    return CircuitExecutorData(
        bitstring_counts=sample_bitstring_counts,
        total_shots=1000,
        executor="test_executor",
        executor_metadata={"backend": "simulator", "version": "1.0"},
    )


@pytest.fixture
def sample_loss_bitstrings() -> dict[str, int]:
    """Sample loss bitstring counts for testing."""
    return {
        "0L": 30,
        "L0": 15,
        "LL": 5,
    }


@pytest.fixture
def sample_executor_data_with_loss(sample_bitstring_counts, sample_loss_bitstrings) -> CircuitExecutorData:
    """Sample CircuitExecutorData with loss bitstrings."""
    return CircuitExecutorData(
        bitstring_counts=sample_bitstring_counts,
        total_shots=1050,
        executor="test_executor",
        executor_metadata={"backend": "simulator", "version": "1.0"},
        loss_bitstrings=sample_loss_bitstrings,
    )


class TestCircuitExecutorDataInitialization:
    """Tests for CircuitExecutorData initialization."""

    def test_initialization_with_all_parameters(self, sample_bitstring_counts):
        """Test initialization with all parameters."""
        metadata = {"key": "value"}
        data = CircuitExecutorData(
            bitstring_counts=sample_bitstring_counts,
            total_shots=1000,
            executor="test_executor",
            executor_metadata=metadata,
        )

        assert data.bitstring_counts == sample_bitstring_counts
        assert data.total_shots == 1000
        assert data.executor == "test_executor"
        assert data.get_executor_metadata() == metadata

    def test_initialization_without_metadata(self, sample_bitstring_counts):
        """Test initialization without executor metadata."""
        data = CircuitExecutorData(
            bitstring_counts=sample_bitstring_counts,
            total_shots=1000,
            executor="test_executor",
        )

        assert data.bitstring_counts == sample_bitstring_counts
        assert data.total_shots == 1000
        assert data.executor == "test_executor"
        assert data.get_executor_metadata() is None
        assert data.loss_bitstrings is None

    def test_initialization_with_loss_bitstrings(self, sample_bitstring_counts, sample_loss_bitstrings):
        """Test initialization with loss bitstrings."""
        data = CircuitExecutorData(
            bitstring_counts=sample_bitstring_counts,
            total_shots=1050,
            executor="test_executor",
            loss_bitstrings=sample_loss_bitstrings,
        )

        assert data.loss_bitstrings == sample_loss_bitstrings

    def test_initialization_with_empty_bitstring_counts(self):
        """Test initialization with empty bitstring counts."""
        data = CircuitExecutorData(
            bitstring_counts={},
            total_shots=0,
            executor="empty_executor",
        )

        assert data.bitstring_counts == {}
        assert data.total_shots == 0
        assert data.executor == "empty_executor"


class TestCircuitExecutorDataMethods:
    """Tests for CircuitExecutorData methods."""

    def test_get_executor_metadata(self, sample_executor_data):
        """Test get_executor_metadata returns correct metadata."""
        metadata = sample_executor_data.get_executor_metadata()
        assert metadata == {"backend": "simulator", "version": "1.0"}

    def test_get_executor_metadata_none(self, sample_bitstring_counts):
        """Test get_executor_metadata returns None when not set."""
        data = CircuitExecutorData(
            bitstring_counts=sample_bitstring_counts,
            total_shots=1000,
            executor="test_executor",
        )
        assert data.get_executor_metadata() is None

    def test_get_summary(self, sample_executor_data):
        """Test get_summary returns expected format."""
        summary = sample_executor_data.get_summary()

        assert "Circuit Executor Data" in summary
        assert "Executor: test_executor" in summary
        assert "Total shots: 1000" in summary
        assert "Bitstring counts: 4" in summary

    def test_get_summary_with_empty_counts(self):
        """Test get_summary with empty bitstring counts."""
        data = CircuitExecutorData(
            bitstring_counts={},
            total_shots=0,
            executor="empty_executor",
        )
        summary = data.get_summary()

        assert "Bitstring counts: 0" in summary

    def test_get_summary_with_loss(self, sample_executor_data_with_loss):
        """Test get_summary includes loss information."""
        summary = sample_executor_data_with_loss.get_summary()

        assert "Loss shots: 50" in summary


class TestCircuitExecutorDataJsonSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_to_json(self, sample_executor_data, sample_bitstring_counts):
        """Test to_json returns correct dictionary."""
        json_data = sample_executor_data.to_json()

        assert json_data["bitstring_counts"] == sample_bitstring_counts
        assert json_data["total_shots"] == 1000
        assert json_data["executor"] == "test_executor"
        assert "version" in json_data
        assert "loss_bitstrings" not in json_data

    def test_to_json_with_loss(self, sample_executor_data_with_loss, sample_loss_bitstrings):
        """Test to_json includes loss_bitstrings when present."""
        json_data = sample_executor_data_with_loss.to_json()

        assert json_data["loss_bitstrings"] == sample_loss_bitstrings

    def test_from_json(self, sample_bitstring_counts):
        """Test from_json creates correct instance."""
        json_data = {
            "version": CircuitExecutorData._serialization_version,
            "bitstring_counts": sample_bitstring_counts,
            "total_shots": 1000,
            "executor": "test_executor",
        }

        data = CircuitExecutorData.from_json(json_data)

        assert data.bitstring_counts == sample_bitstring_counts
        assert data.total_shots == 1000
        assert data.executor == "test_executor"

    def test_json_roundtrip(self, sample_executor_data):
        """Test JSON serialization roundtrip preserves data."""
        json_data = sample_executor_data.to_json()
        restored = CircuitExecutorData.from_json(json_data)

        assert restored.bitstring_counts == sample_executor_data.bitstring_counts
        assert restored.total_shots == sample_executor_data.total_shots
        assert restored.executor == sample_executor_data.executor
        assert restored.loss_bitstrings is None

    def test_json_roundtrip_with_loss(self, sample_executor_data_with_loss):
        """Test JSON serialization roundtrip preserves loss bitstrings."""
        json_data = sample_executor_data_with_loss.to_json()
        restored = CircuitExecutorData.from_json(json_data)

        assert restored.bitstring_counts == sample_executor_data_with_loss.bitstring_counts
        assert restored.total_shots == sample_executor_data_with_loss.total_shots
        assert restored.loss_bitstrings == sample_executor_data_with_loss.loss_bitstrings

    def test_from_json_with_missing_optional_fields(self):
        """Test from_json handles missing optional fields."""
        json_data = {
            "version": CircuitExecutorData._serialization_version,
        }

        data = CircuitExecutorData.from_json(json_data)

        assert data.bitstring_counts == {}
        assert data.total_shots == 0
        assert data.executor == ""

    def test_from_json_invalid_version(self, sample_bitstring_counts):
        """Test from_json raises error for invalid version."""
        json_data = {
            "version": "99.99.99",
            "bitstring_counts": sample_bitstring_counts,
            "total_shots": 1000,
            "executor": "test_executor",
        }

        with pytest.raises(RuntimeError, match="version"):
            CircuitExecutorData.from_json(json_data)


class TestCircuitExecutorDataHdf5Serialization:
    """Tests for HDF5 serialization and deserialization."""

    def test_to_hdf5(self, sample_executor_data):
        """Test to_hdf5 writes correct data."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hdf5_file:
                sample_executor_data.to_hdf5(hdf5_file)

            with h5py.File(filepath, "r") as hdf5_file:
                assert hdf5_file.attrs["total_shots"] == 1000
                assert hdf5_file.attrs["executor"] == "test_executor"
                assert "bitstring_keys" in hdf5_file
                assert "bitstring_counts" in hdf5_file
                assert "loss_bitstrings" not in hdf5_file
        finally:
            filepath.unlink()

    def test_to_hdf5_with_loss(self, sample_executor_data_with_loss):
        """Test to_hdf5 writes loss bitstrings group when present."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hdf5_file:
                sample_executor_data_with_loss.to_hdf5(hdf5_file)

            with h5py.File(filepath, "r") as hdf5_file:
                assert "loss_bitstrings" in hdf5_file
                assert "keys" in hdf5_file["loss_bitstrings"]
                assert "counts" in hdf5_file["loss_bitstrings"]
        finally:
            filepath.unlink()

    def test_from_hdf5(self, sample_bitstring_counts):
        """Test from_hdf5 creates correct instance."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hdf5_file:
                hdf5_file.attrs["version"] = CircuitExecutorData._serialization_version
                hdf5_file.attrs["total_shots"] = 1000
                hdf5_file.attrs["executor"] = "test_executor"
                keys = list(sample_bitstring_counts.keys())
                counts = list(sample_bitstring_counts.values())
                hdf5_file.create_dataset("bitstring_keys", data=[k.encode() for k in keys])
                hdf5_file.create_dataset("bitstring_counts", data=counts)

            with h5py.File(filepath, "r") as hdf5_file:
                data = CircuitExecutorData.from_hdf5(hdf5_file)

            assert data.bitstring_counts == sample_bitstring_counts
            assert data.total_shots == 1000
            assert data.executor == "test_executor"
            assert data.loss_bitstrings is None
        finally:
            filepath.unlink()

    def test_hdf5_roundtrip(self, sample_executor_data):
        """Test HDF5 serialization roundtrip preserves data."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hdf5_file:
                sample_executor_data.to_hdf5(hdf5_file)

            with h5py.File(filepath, "r") as hdf5_file:
                restored = CircuitExecutorData.from_hdf5(hdf5_file)

            assert restored.bitstring_counts == sample_executor_data.bitstring_counts
            assert restored.total_shots == sample_executor_data.total_shots
            assert restored.executor == sample_executor_data.executor
            assert restored.loss_bitstrings is None
        finally:
            filepath.unlink()

    def test_hdf5_roundtrip_with_loss(self, sample_executor_data_with_loss):
        """Test HDF5 serialization roundtrip preserves loss bitstrings."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hdf5_file:
                sample_executor_data_with_loss.to_hdf5(hdf5_file)

            with h5py.File(filepath, "r") as hdf5_file:
                restored = CircuitExecutorData.from_hdf5(hdf5_file)

            assert restored.loss_bitstrings == sample_executor_data_with_loss.loss_bitstrings
            assert restored.total_shots == sample_executor_data_with_loss.total_shots
        finally:
            filepath.unlink()

    def test_from_hdf5_invalid_version(self):
        """Test from_hdf5 raises error for invalid version."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hdf5_file:
                hdf5_file.attrs["version"] = "99.99.99"
                hdf5_file.attrs["total_shots"] = 1000
                hdf5_file.attrs["executor"] = "test"
                hdf5_file.create_dataset("bitstring_keys", data=[b"00"])
                hdf5_file.create_dataset("bitstring_counts", data=[100])

            with h5py.File(filepath, "r") as hdf5_file, pytest.raises(RuntimeError, match="version"):
                CircuitExecutorData.from_hdf5(hdf5_file)
        finally:
            filepath.unlink()
