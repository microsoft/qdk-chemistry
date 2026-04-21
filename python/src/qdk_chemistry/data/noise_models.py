"""QDK/Chemistry noise model module for simulating noise in quantum circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any, ClassVar

import h5py
from qsharp._simulation import NoiseConfig
from ruamel.yaml import YAML

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.enum import CaseInsensitiveStrEnum

__all__: list[str] = ["SupportedErrorTypes", "SupportedGate"]


class SupportedGate(CaseInsensitiveStrEnum):
    """An enumeration of quantum gate types with case-insensitive string lookup.

    Gate types gathered from QDK QIR instructions.
    """

    CCX = "ccx"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    H = "h"
    ID = "id"
    MEASURE = "measure"
    RESET = "reset"
    RX = "rx"
    RXX = "rxx"
    RY = "ry"
    RYY = "ryy"
    RZ = "rz"
    RZZ = "rzz"
    S = "s"
    SDG = "sdg"
    SWAP = "swap"
    SX = "sx"
    SXDG = "sxdg"
    T = "t"
    TDG = "tdg"
    X = "x"
    Y = "y"
    Z = "z"

    @classmethod
    def from_string(cls, gate_str: str) -> "SupportedGate":
        """Get a Gate enum value from its string representation.

        Args:
            gate_str: String representation of the gate (case-insensitive).

        Returns:
            SupportedGate: The corresponding SupportedGate enum value.

        Raises:
            ValueError: If no matching gate is found.

        """
        try:
            # Leverage internal _missing_ method for case-insensitive lookup
            return cls(gate_str)
        except ValueError:
            # If the gate_str does not match any enum value, raise an error
            raise ValueError(f"Unknown gate type: {gate_str}") from None


ONE_QUBIT_SUPPORTED_GATES = frozenset(
    [
        SupportedGate.H,
        SupportedGate.ID,
        SupportedGate.RX,
        SupportedGate.RY,
        SupportedGate.RZ,
        SupportedGate.S,
        SupportedGate.SDG,
        SupportedGate.SX,
        SupportedGate.SXDG,
        SupportedGate.T,
        SupportedGate.TDG,
        SupportedGate.X,
        SupportedGate.Y,
        SupportedGate.Z,
    ]
)
TWO_QUBIT_SUPPORTED_GATES = frozenset(
    [
        SupportedGate.CX,
        SupportedGate.CY,
        SupportedGate.CZ,
        SupportedGate.SWAP,
        SupportedGate.RXX,
        SupportedGate.RYY,
        SupportedGate.RZZ,
    ]
)
THREE_QUBIT_SUPPORTED_GATES = frozenset([SupportedGate.CCX])


class SupportedErrorTypes(CaseInsensitiveStrEnum):
    """Supported error types for quantum gates with case-insensitive string lookup."""

    DEPOLARIZING_ERROR = "depolarizing_error"
    QUBIT_LOSS = "qubit_loss"


class QuantumErrorProfile(DataClass):
    """A class representing a quantum error profile containing information about quantum gates and error properties.

    This class provides functionalities to define, load, and save quantum error profiles.

    """

    # Class attribute for filename validation
    _data_type_name = "quantum_error_profile"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    basis_gates_exclusion: ClassVar[set[str]] = {"reset", "barrier", "measure"}
    """Gates to exclude from basis gates in noise model."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        errors: dict[SupportedGate, dict[SupportedErrorTypes, float]] | None = None,
    ) -> None:
        """Initialize a QuantumErrorProfile.

        Args:
            name: Name of the quantum error profile.
            description: Description of what the error profile represents.
            errors: Dictionary mapping gate names to their error properties.

        """
        Logger.trace_entering()
        self.name: str = "default" if name is None else name
        self.description: str = "No description provided" if description is None else description
        self.errors: dict[SupportedGate, dict[SupportedErrorTypes, float]] = {}
        self.one_qubit_gates: set[SupportedGate] = set()
        self.two_qubit_gates: set[SupportedGate] = set()
        self.three_qubit_gates: set[SupportedGate] = set()
        if errors is not None:
            for gate_key, error_rates in errors.items():
                gate = gate_key if isinstance(gate_key, SupportedGate) else SupportedGate(gate_key)
                if gate in ONE_QUBIT_SUPPORTED_GATES:
                    self.one_qubit_gates.add(gate)
                elif gate in TWO_QUBIT_SUPPORTED_GATES:
                    self.two_qubit_gates.add(gate)
                elif gate in THREE_QUBIT_SUPPORTED_GATES:
                    self.three_qubit_gates.add(gate)
                validated_rates: dict[SupportedErrorTypes, float] = {}
                for error_type_key, rate in error_rates.items():
                    error_type = (
                        error_type_key
                        if isinstance(error_type_key, SupportedErrorTypes)
                        else SupportedErrorTypes(error_type_key)
                    )
                    if not isinstance(rate, int | float):
                        raise TypeError(f"Expected rate to be a float, got {type(rate).__name__}")
                    validated_rates[error_type] = float(rate)
                self.errors[gate] = validated_rates

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    def __eq__(self, other: object) -> bool:
        """Check equality between two QuantumErrorProfile instances.

        Args:
            other: Object to compare with.

        Returns:
            bool: True if equal, False otherwise.

        """
        if not isinstance(other, QuantumErrorProfile):
            return False
        return self.name == other.name and self.description == other.description and self.errors == other.errors

    def __hash__(self) -> int:
        """Make QuantumErrorProfile hashable.

        Returns:
            int: Hash value.

        """
        # Convert mutable nested dict to immutable tuple for hashing
        errors_tuple = tuple(
            sorted((str(k), tuple(sorted((str(et), r) for et, r in v.items()))) for k, v in self.errors.items())
        )
        return hash((self.name, self.description, errors_tuple))

    @property
    def basis_gates(self) -> list[str]:
        """Get basis gates from profile.

        Returns:
            list[str]: List of basis gates in noise model.

        """
        return [
            str(gate)
            for gate in self.one_qubit_gates | self.two_qubit_gates | self.three_qubit_gates
            if gate not in self.basis_gates_exclusion
        ]

    def to_yaml_file(self, yaml_file: str | Path) -> None:
        """Save quantum error profile to YAML file.

        Args:
            yaml_file: Path to save YAML file.

        """
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Convert to serializable dict
        data = self.to_json()

        with Path(yaml_file).open("w") as f:
            yaml.dump(data, f)

    @classmethod
    def from_yaml_file(cls, yaml_file: str | Path) -> "QuantumErrorProfile":
        """Load quantum error profile from YAML file.

        Args:
            yaml_file: Path to YAML file.

        Returns:
            QuantumErrorProfile: Loaded profile.

        """
        yaml = YAML(typ="safe")  # type: ignore
        if not Path(yaml_file).exists():
            raise FileNotFoundError(f"File {yaml_file} not found")

        with Path(yaml_file).open("r") as f:
            data = yaml.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML file {yaml_file} is empty or invalid.")

        invalid_keys = set(data.keys()) - {"version", "name", "description", "errors"}
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in YAML file: {invalid_keys}.\n"
                "Only 'version', 'name', 'description', and 'errors' are allowed."
            )

        return cls.from_json(data)

    # DataClass interface implementation
    def get_summary(self) -> str:
        """Get a human-readable summary of the QuantumErrorProfile.

        Returns:
            str: Summary string describing the quantum error profile.

        """
        data = self.to_json()
        lines = [
            "Quantum Error Profile",
            f"  name: {data['name']}",
            f"  description: {data['description']}",
            "  errors:",
        ]
        for gate_str, error_rates in data["errors"].items():
            lines.append(f"    gate: {gate_str}")
            for error_type, rate in error_rates.items():
                lines.append(f"      {error_type}: {rate}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Convert the QuantumErrorProfile to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the quantum error profile.

        """
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "errors": {},
        }

        # Convert enum keys to strings in the errors dictionary
        for gate, error_rates in self.errors.items():
            gate_str = str(gate)
            data["errors"][gate_str] = {str(error_type): rate for error_type, rate in error_rates.items()}

        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the QuantumErrorProfile to an HDF5 group.

        Args:
            group: HDF5 group or file to write the quantum error profile to.

        """
        data = self.to_json()
        group.attrs["version"] = data["version"]
        group.attrs["name"] = data["name"]
        group.attrs["description"] = data["description"]
        # Serialize errors dict as JSON string since HDF5 does not support nested dicts
        group.attrs["errors"] = json.dumps(data["errors"])

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "QuantumErrorProfile":
        """Create a QuantumErrorProfile from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data.

        Returns:
            QuantumErrorProfile: New instance of the QuantumErrorProfile.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)

        name = json_data.get("name")
        description = json_data.get("description")
        errors: dict[SupportedGate, dict[SupportedErrorTypes, float]] = {}

        json_errors = json_data.get("errors")
        if json_errors is not None:
            for gate_str, error_rates in json_errors.items():
                errors[SupportedGate(gate_str)] = {SupportedErrorTypes(et): rate for et, rate in error_rates.items()}

        return cls(name=name, description=description, errors=errors)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "QuantumErrorProfile":
        """Load a QuantumErrorProfile from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from.

        Returns:
            QuantumErrorProfile: New instance of the QuantumErrorProfile.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        data = {
            "version": group.attrs["version"],
            "name": group.attrs["name"],
            "description": group.attrs["description"],
            "errors": json.loads(group.attrs["errors"]),  # Deserialize errors from JSON string
        }
        return cls.from_json(data)

    def to_qdk_noise_config(self) -> NoiseConfig:
        """Convert the QuantumErrorProfile to a QDK-compatible noise configuration dictionary.

        Returns:
            QDK-compatible noise configuration object.

        """
        noise = NoiseConfig()
        for gate, error_rates in self.errors.items():
            gate_name = str(gate)
            gate_name_qdk = gate_name.lower()
            if gate_name_qdk == "sdg":
                gate_name_qdk = "s_adj"
            elif gate_name_qdk == "tdg":
                gate_name_qdk = "t_adj"
            elif gate_name_qdk == "sxdg":
                gate_name_qdk = "sx_adj"
            elif gate_name_qdk == "measure":
                gate_name_qdk = "mresetz"
            try:
                gate_config = getattr(noise, gate_name_qdk)
            except AttributeError:
                raise ValueError(f"Gate {gate_name} is not supported in QDK noise config.") from None
            for error_type, rate in error_rates.items():
                if error_type == SupportedErrorTypes.DEPOLARIZING_ERROR:
                    gate_config.set_depolarizing(rate)
                elif error_type == SupportedErrorTypes.QUBIT_LOSS:
                    gate_config.loss = rate
                else:
                    raise ValueError(f"Error type {error_type} is not currently supported.")
        return noise
