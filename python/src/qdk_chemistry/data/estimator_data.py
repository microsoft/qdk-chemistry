"""QDK/Chemistry Energy Estimator Results module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data._hashing import _hash_array, _hash_float, _hash_int, _hash_str, _hash_uint
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.qubit_operator import QubitOperator
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class EnergyExpectationResult(DataClass):
    """Expectation value and variance for a Hamiltonian energy estimate.

    Attributes:
        energy_expectation_value (float): Expectation value of the energy.
        energy_variance (float): Variance of the energy.
        expvals_each_term (list[numpy.ndarray]): Expectation values for each term in the Hamiltonian.
        variances_each_term (list[numpy.ndarray]): Variances for each term in the Hamiltonian.

    """

    # Class attribute for filename validation
    _data_type_name = "energy_expectation_result"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        energy_expectation_value: float,
        energy_variance: float,
        expvals_each_term: list[np.ndarray],
        variances_each_term: list[np.ndarray],
    ) -> None:
        """Initialize an energy expectation result.

        Args:
            energy_expectation_value (float): Expectation value of the energy.
            energy_variance (float): Variance of the energy.
            expvals_each_term (list[numpy.ndarray]): Expectation values for each term in the Hamiltonian.
            variances_each_term (list[numpy.ndarray]): Variances for each term in the Hamiltonian.

        """
        Logger.trace_entering()
        self.energy_expectation_value = energy_expectation_value
        self.energy_variance = energy_variance
        self.expvals_each_term = expvals_each_term
        self.variances_each_term = variances_each_term
        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "energy_expectation_result")
        _hash_float(h, self.energy_expectation_value)
        _hash_float(h, self.energy_variance)
        _hash_uint(h, len(self.expvals_each_term))
        for arr in self.expvals_each_term:
            _hash_array(h, arr)
        _hash_uint(h, len(self.variances_each_term))
        for arr in self.variances_each_term:
            _hash_array(h, arr)

    def get_summary(self) -> str:
        """Get a human-readable summary of the energy expectation result.

        Returns:
            str: Summary string describing the energy expectation result.

        """
        return (
            f"Energy Expectation Result\n"
            f"  Energy: {self.energy_expectation_value:.6f} ± {np.sqrt(self.energy_variance):.6f}\n"
            f"  Variance: {self.energy_variance:.6e}\n"
            f"  Number of terms: {len(self.expvals_each_term)}"
        )

    def to_json(self) -> dict[str, Any]:
        """Convert energy expectation result to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the energy expectation result.

        """
        data = {
            "energy_expectation_value": float(self.energy_expectation_value),
            "energy_variance": float(self.energy_variance),
            "expvals_each_term": [arr.tolist() for arr in self.expvals_each_term],
            "variances_each_term": [arr.tolist() for arr in self.variances_each_term],
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the energy expectation result to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the energy expectation result to.

        """
        self._add_hdf5_version(group)
        group.attrs["energy_expectation_value"] = self.energy_expectation_value
        group.attrs["energy_variance"] = self.energy_variance

        # Store arrays as datasets
        for i, arr in enumerate(self.expvals_each_term):
            group.create_dataset(f"expvals_term_{i}", data=arr)

        for i, arr in enumerate(self.variances_each_term):
            group.create_dataset(f"variances_term_{i}", data=arr)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "EnergyExpectationResult":
        """Create an energy expectation result from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            EnergyExpectationResult: New instance reconstructed from JSON data.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)

        return cls(
            energy_expectation_value=float(json_data["energy_expectation_value"]),
            energy_variance=float(json_data["energy_variance"]),
            expvals_each_term=[np.array(arr) for arr in json_data["expvals_each_term"]],
            variances_each_term=[np.array(arr) for arr in json_data["variances_each_term"]],
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "EnergyExpectationResult":
        """Load an energy expectation result from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file containing the data.

        Returns:
            EnergyExpectationResult: New instance reconstructed from HDF5 data.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)

        energy_expectation_value = group.attrs["energy_expectation_value"]
        energy_variance = group.attrs["energy_variance"]

        # Load arrays from datasets
        expvals_each_term = []
        variances_each_term = []

        i = 0
        while f"expvals_term_{i}" in group:
            expvals_each_term.append(np.array(group[f"expvals_term_{i}"]))
            i += 1

        i = 0
        while f"variances_term_{i}" in group:
            variances_each_term.append(np.array(group[f"variances_term_{i}"]))
            i += 1

        return cls(
            energy_expectation_value=float(energy_expectation_value),
            energy_variance=float(energy_variance),
            expvals_each_term=expvals_each_term,
            variances_each_term=variances_each_term,
        )


class MeasurementData(DataClass):
    """Measurement bitstring data and metadata for a ``QubitOperator``.

    Attributes:
        hamiltonians (list[QubitOperator]): List of QubitOperator corresponding to the measurement data.
        bitstring_counts (list[dict[str, int] | None] | None): Bitstring count dictionaries for each QubitOperator.
        shots_list (list[int] | None): List of number of shots used for each measurement.

    """

    # Class attribute for filename validation
    _data_type_name = "measurement_data"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        hamiltonians: list[QubitOperator],
        bitstring_counts: list[dict[str, int] | None] | None = None,
        shots_list: list[int] | None = None,
    ) -> None:
        """Initialize measurement data.

        Args:
            hamiltonians (list[QubitOperator]): List of QubitOperator objects.
            bitstring_counts (list[dict[str, int] | None] | None): Bitstring count dicts for each QubitOperator.
            shots_list (list[int] | None): List of number of shots used for each measurement.

        """
        Logger.trace_entering()
        self.hamiltonians = hamiltonians
        self.bitstring_counts = bitstring_counts if bitstring_counts is not None else []
        self.shots_list = shots_list if shots_list is not None else []
        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "measurement_data")
        _hash_uint(h, len(self.hamiltonians))
        for ham in self.hamiltonians:
            _hash_str(h, ham.content_hash())
        _hash_uint(h, len(self.bitstring_counts))
        for bc in self.bitstring_counts:
            if bc is None:
                h.update(b"\x00")
            else:
                h.update(b"\x01")
                _hash_uint(h, len(bc))
                for k in sorted(bc.keys()):
                    _hash_str(h, k)
                    _hash_int(h, bc[k])
        _hash_uint(h, len(self.shots_list))
        for s in self.shots_list:
            _hash_int(h, s)

    def get_summary(self) -> str:
        """Get a human-readable summary of the measurement data.

        Returns:
            str: Summary string describing the measurement data.

        """
        total_shots = sum(self.shots_list) if self.shots_list else 0
        return (
            f"Measurement Data\n"
            f"  Number of Hamiltonians: {len(self.hamiltonians)}\n"
            f"  Total shots: {total_shots}\n"
            f"  Measurements collected: {len(self.bitstring_counts)}"
        )

    def to_json(self) -> dict[str, Any]:
        """Convert measurement data to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the measurement data.

        """
        data = {
            str(i): {
                "hamiltonian": {
                    "paulis": hamiltonian.pauli_strings,
                    "coefficients": hamiltonian.coefficients.tolist(),
                },
                "bitstring": self.bitstring_counts[i] if i < len(self.bitstring_counts) else None,
                "shots": self.shots_list[i] if i < len(self.shots_list) else 0,
            }
            for i, hamiltonian in enumerate(self.hamiltonians)
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the measurement data to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the measurement data to.

        """
        self._add_hdf5_version(group)
        group.attrs["num_hamiltonians"] = len(self.hamiltonians)

        # Store each hamiltonian and its measurements
        for i, hamiltonian in enumerate(self.hamiltonians):
            ham_group = group.create_group(f"hamiltonian_{i}")

            # Store Hamiltonian data
            ham_group.create_dataset("pauli_strings", data=np.array(hamiltonian.pauli_strings, dtype="S"))
            ham_group.create_dataset("coefficients", data=hamiltonian.coefficients)

            # Store bitstring counts if available
            if i < len(self.bitstring_counts) and self.bitstring_counts[i] is not None:
                bitstring_dict = self.bitstring_counts[i]
                assert bitstring_dict is not None  # Type narrowing for mypy
                bitstrings = list(bitstring_dict.keys())
                counts = list(bitstring_dict.values())
                ham_group.create_dataset("bitstrings", data=np.array(bitstrings, dtype="S"))
                ham_group.create_dataset("counts", data=np.array(counts))

            # Store shots if available
            if i < len(self.shots_list):
                ham_group.attrs["shots"] = self.shots_list[i]

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "MeasurementData":
        """Create measurement data from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            MeasurementData: New instance reconstructed from JSON data.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)

        hamiltonians: list[QubitOperator] = []
        bitstring_counts: list[dict[str, int] | None] = []
        shots_list: list[int] = []

        # Iterate through the indexed items (skip 'version' key)
        for key in sorted((k for k in json_data if k != "version"), key=int):
            item = json_data[key]

            # Reconstruct QubitOperator
            ham_data = item["hamiltonian"]
            hamiltonian = QubitOperator(
                ham_data["paulis"],
                np.array(ham_data["coefficients"]),
            )
            hamiltonians.append(hamiltonian)

            # Get bitstring counts
            bitstring_counts.append(item.get("bitstring"))

            # Get shots
            shots_list.append(item.get("shots", 0))

        return cls(
            hamiltonians=hamiltonians,
            bitstring_counts=bitstring_counts,
            shots_list=shots_list,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "MeasurementData":
        """Load measurement data from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file containing the data.

        Returns:
            MeasurementData: New instance reconstructed from HDF5 data.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)

        num_hamiltonians = group.attrs["num_hamiltonians"]

        hamiltonians: list[QubitOperator] = []
        bitstring_counts: list[dict[str, int] | None] = []
        shots_list: list[int] = []

        for i in range(num_hamiltonians):
            ham_group = group[f"hamiltonian_{i}"]

            # Load Hamiltonian data
            pauli_strings = [s.decode() for s in ham_group["pauli_strings"][:]]
            coefficients = np.array(ham_group["coefficients"])
            hamiltonian = QubitOperator(pauli_strings, coefficients)
            hamiltonians.append(hamiltonian)

            # Load bitstring counts if available
            if "bitstrings" in ham_group and "counts" in ham_group:
                bitstrings = [s.decode() for s in ham_group["bitstrings"][:]]
                counts = ham_group["counts"][:]
                bitstring_dict: dict[str, int] = dict(zip(bitstrings, counts, strict=True))
                bitstring_counts.append(bitstring_dict)
            else:
                bitstring_counts.append(None)

            # Load shots if available
            if "shots" in ham_group.attrs:
                shots_list.append(int(ham_group.attrs["shots"]))
            else:
                shots_list.append(0)

        return cls(
            hamiltonians=hamiltonians,
            bitstring_counts=bitstring_counts,
            shots_list=shots_list,
        )
