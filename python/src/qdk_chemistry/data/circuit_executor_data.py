"""QDK/Chemistry Circuit Executor Data module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class CircuitExecutorData(DataClass):
    """Bitstring data and metadata from quantum circuit executions."""

    # Class attribute for filename validation
    _data_type_name = "circuit_executor_data"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        bitstring_counts: dict[str, int],
        total_shots: int,
        executor: str,
        executor_metadata: Any | None = None,
        loss_bitstrings: dict[str, int] | None = None,
    ) -> None:
        """Initialize circuit executor data.

        Args:
            bitstring_counts: Bitstring count dict (clean shots only, 0/1).
            total_shots: Total number of shots used for the measurement.
            executor: Name of the executor used for the measurement.
            executor_metadata: Metadata associated with the executor.
            loss_bitstrings: Bitstring counts for shots where at
                least one qubit was lost. Uses 'L' to mark lost qubits (e.g. '0L10').

        """
        Logger.trace_entering()
        self.bitstring_counts = bitstring_counts
        self.total_shots = total_shots
        self.executor = executor
        self._executor_metadata = executor_metadata
        self.loss_bitstrings = loss_bitstrings
        super().__init__()

    def get_executor_metadata(self) -> Any | None:
        """Get the executor metadata.

        Returns:
            Any | None: Metadata associated with the executor.

        """
        return self._executor_metadata

    def get_summary(self) -> str:
        """Get a human-readable summary of the measurement data.

        Returns:
            str: Summary string describing the circuit executor data.

        """
        summary = (
            f"Circuit Executor Data\n  Executor: {self.executor}\n  "
            f"Total shots: {self.total_shots}\n  Bitstring counts: {len(self.bitstring_counts)}"
        )
        if self.loss_bitstrings:
            summary += f"\n  Loss shots: {sum(self.loss_bitstrings.values())}"
        return summary

    def to_json(self) -> dict[str, Any]:
        """Convert circuit executor data to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the measurement data.

        """
        data = {
            "bitstring_counts": self.bitstring_counts,
            "total_shots": self.total_shots,
            "executor": self.executor,
        }
        if self.loss_bitstrings:
            data["loss_bitstrings"] = self.loss_bitstrings
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the measurement data to an HDF5 group.

        Args:
            group: HDF5 group or file to write the measurement data to.

        """
        self._add_hdf5_version(group)
        group.attrs["total_shots"] = self.total_shots
        group.create_dataset("bitstring_keys", data=np.array(list(self.bitstring_counts.keys()), dtype="S"))
        group.create_dataset("bitstring_counts", data=np.array(list(self.bitstring_counts.values()), dtype=np.int64))
        if self.executor is not None:
            group.attrs["executor"] = self.executor
        if self.loss_bitstrings:
            loss_group = group.create_group("loss_bitstrings")
            loss_group.create_dataset("keys", data=np.array(list(self.loss_bitstrings.keys()), dtype="S"))
            loss_group.create_dataset("counts", data=np.array(list(self.loss_bitstrings.values()), dtype=np.int64))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "CircuitExecutorData":
        """Create measurement data from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data.

        Returns:
            CircuitExecutorData: New instance reconstructed from JSON data.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        bitstring_counts = json_data.get("bitstring_counts", {})
        total_shots = json_data.get("total_shots", 0)
        executor = json_data.get("executor", "")
        loss_bitstrings = json_data.get("loss_bitstrings")

        return cls(
            bitstring_counts=bitstring_counts,
            total_shots=total_shots,
            executor=executor,
            loss_bitstrings=loss_bitstrings,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "CircuitExecutorData":
        """Load measurement data from an HDF5 group.

        Args:
            group: HDF5 group or file containing the data.

        Returns:
            CircuitExecutorData: New instance reconstructed from HDF5 data.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)

        total_shots = int(group.attrs.get("total_shots", 0))
        executor = group.attrs.get("executor", "")
        bitstring_keys = group["bitstring_keys"][()]
        bitstring_counts_values = group["bitstring_counts"][()]
        bitstring_counts = {
            key.decode("utf-8"): int(count) for key, count in zip(bitstring_keys, bitstring_counts_values, strict=False)
        }

        loss_bitstrings = None
        if "loss_bitstrings" in group:
            loss_group = group["loss_bitstrings"]
            loss_keys = loss_group["keys"][()]
            loss_counts = loss_group["counts"][()]
            loss_bitstrings = {
                key.decode("utf-8"): int(count) for key, count in zip(loss_keys, loss_counts, strict=False)
            }

        return cls(
            bitstring_counts=bitstring_counts,
            total_shots=total_shots,
            executor=executor,
            loss_bitstrings=loss_bitstrings,
        )
