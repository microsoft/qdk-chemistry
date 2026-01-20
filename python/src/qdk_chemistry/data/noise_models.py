"""QDK/Chemistry noise model module for simulating noise in quantum circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import h5py
from qsharp._simulation import NoiseConfig
from ruamel.yaml import YAML

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.enum import CaseInsensitiveStrEnum

__all__: list[str] = ["GateErrorDef", "SupportedErrorTypes", "SupportedGate"]


class SupportedGate(CaseInsensitiveStrEnum):
    """An enumeration of quantum gate types with case-insensitive string lookup.

    Gate types gathered from Qiskit
    https://github.com/Qiskit/qiskit/blob/a88ed60615eeb988f404f9afaf142775478aceb9/qiskit/circuit/quantumcircuit.py#L673C1-L731C1
    """

    BARRIER = "barrier"
    CCX = "ccx"
    CCZ = "ccz"
    CH = "ch"
    CP = "cp"
    CRX = "crx"
    CRY = "cry"
    CRZ = "crz"
    CS = "cs"
    CSDG = "csdg"
    CSWAP = "cswap"
    CSX = "csx"
    CU = "cu"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    DCX = "dcx"
    DELAY = "delay"
    ECR = "ecr"
    H = "h"
    ID = "id"
    INITIALIZE = "initialize"
    ISWAP = "iswap"
    MCP = "mcp"
    MCRX = "mcrx"
    MCRY = "mcry"
    MCRZ = "mcrz"
    MCX = "mcx"
    MEASURE = "measure"
    MS = "ms"
    P = "p"
    PAULI = "pauli"
    R = "r"
    RCCCX = "rcccx"
    RCCX = "rccx"
    RESET = "reset"
    RV = "rv"
    RX = "rx"
    RXX = "rxx"
    RY = "ry"
    RYY = "ryy"
    RZ = "rz"
    RZX = "rzx"
    RZZ = "rzz"
    S = "s"
    SDG = "sdg"
    SWAP = "swap"
    SX = "sx"
    SXDG = "sxdg"
    T = "t"
    TDG = "tdg"
    U = "u"
    UNITARY = "unitary"
    X = "x"
    Y = "y"
    Z = "z"

    @classmethod
    def from_string(cls, gate_str: str) -> "SupportedGate":
        """Get a Gate enum value from its string representation.

        Args:
            gate_str (str): String representation of the gate (case-insensitive).

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


class SupportedErrorTypes(CaseInsensitiveStrEnum):
    """Supported error types for quantum gates with case-insensitive string lookup."""

    DEPOLARIZING_ERROR = "depolarizing_error"


class GateErrorDef(TypedDict):
    """Typed dictionary for error definitions."""

    type: SupportedErrorTypes
    """Error type."""

    rate: float
    """Error rate."""

    num_qubits: int
    """Number of qubits the gate acts on."""


class QuantumErrorProfile(DataClass):
    """A class representing a quantum error profile containing information about quantum gates and error properties.

    This class provides functionalities to define, load, and save quantum error profiles.

    Attributes:
        name (str): Name of the quantum error profile.
        description (str): Description of what the error profile represents.
        errors (dict[SupportedGate, GateErrorDef]): Dictionary mapping gate names to their error properties.
        one_qubit_gates (list[str]): List of gate names that operate on a single qubit.
        two_qubit_gates (list[str]): List of gate names that operate on two qubits.

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
        errors: dict[SupportedGate, GateErrorDef] | None = None,
    ) -> None:
        """Initialize a QuantumErrorProfile.

        Args:
            name (str | None): Name of the quantum error profile.
            description (str | None): Description of what the error profile represents.
            errors (dict | None): Dictionary mapping supported gate names to their error definitions.

        """
        Logger.trace_entering()
        self.name: str = "default" if name is None else name
        self.description: str = "No description provided" if description is None else description
        self.errors: dict[SupportedGate, GateErrorDef] = {}
        if errors is not None:
            # Check types
            for gate_key, error_dict in errors.items():
                gate = gate_key if isinstance(gate_key, SupportedGate) else SupportedGate(gate_key)
                if isinstance(error_dict, dict):
                    if isinstance(error_dict["type"], SupportedErrorTypes):
                        error_type = error_dict["type"]
                    else:
                        error_type = SupportedErrorTypes(error_dict["type"])
                    assert isinstance(error_dict["rate"], float)
                    assert isinstance(error_dict["num_qubits"], int)
                    self.errors[gate] = GateErrorDef(
                        type=error_type,
                        rate=error_dict["rate"],
                        num_qubits=error_dict["num_qubits"],
                    )
                else:
                    self.errors[gate] = error_dict

        # Initialize one_qubit_gates and two_qubit_gates based on errors
        one_qubit_gates: list[str] = []
        two_qubit_gates: list[str] = []
        for gate, error_dict in self.errors.items():
            if error_dict["num_qubits"] == 1:
                one_qubit_gates.append(str(gate))
            elif error_dict["num_qubits"] == 2:
                two_qubit_gates.append(str(gate))
            else:
                raise ValueError(f"Unsupported number of qubits: {error_dict['num_qubits']}")
        self.one_qubit_gates = sorted(set(one_qubit_gates))
        self.two_qubit_gates = sorted(set(two_qubit_gates))

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    def __eq__(self, other: object) -> bool:
        """Check equality between two QuantumErrorProfile instances.

        Args:
            other (object): Object to compare with.

        Returns:
            bool: True if equal, False otherwise.

        """
        if not isinstance(other, QuantumErrorProfile):
            return False
        return (
            self.name == other.name
            and self.description == other.description
            and self.errors == other.errors
            and self.one_qubit_gates == other.one_qubit_gates
            and self.two_qubit_gates == other.two_qubit_gates
        )

    def __hash__(self) -> int:
        """Make QuantumErrorProfile hashable.

        Returns:
            int: Hash value.

        """
        # Convert mutable dict to immutable tuple of items for hashing
        errors_tuple = tuple(sorted((str(k), tuple(v.items())) for k, v in self.errors.items()))
        return hash(
            (self.name, self.description, errors_tuple, tuple(self.one_qubit_gates), tuple(self.two_qubit_gates))
        )

    @property
    def basis_gates(self) -> list[str]:
        """Get basis gates from profile.

        Returns:
            list[str]: List of basis gates in noise model.

        """
        return [gate for gate in self.one_qubit_gates + self.two_qubit_gates if gate not in self.basis_gates_exclusion]

    def to_yaml_file(self, yaml_file: str | Path) -> None:
        """Save quantum error profile to YAML file.

        Args:
            yaml_file (str | pathlib.Path): Path to save YAML file.

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
            yaml_file (str | pathlib.Path): Path to YAML file.

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
        for gate_str, error_dict in data["errors"].items():
            lines.append(f"    gate: {gate_str}")
            lines.append(f"    type: {error_dict['type']}")
            lines.append(f"    rate: {error_dict['rate']}")
            lines.append(f"    num_qubits: {error_dict['num_qubits']}")
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

        # Convert enum keys and values to strings in the errors dictionary
        for gate, error_def in self.errors.items():
            gate_str = str(gate)
            error_dict = dict(error_def)
            error_dict["type"] = str(error_dict["type"])
            data["errors"][gate_str] = error_dict

        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the QuantumErrorProfile to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the quantum error profile to.

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
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            QuantumErrorProfile: New instance of the QuantumErrorProfile.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)

        name = json_data.get("name")
        description = json_data.get("description")
        errors: dict[SupportedGate, GateErrorDef] = {}

        json_errors = json_data.get("errors")
        if json_errors is not None:
            for gate, error_dict in json_errors.items():
                errors[SupportedGate(gate)] = GateErrorDef(
                    type=SupportedErrorTypes(error_dict["type"]),
                    rate=error_dict["rate"],
                    num_qubits=error_dict["num_qubits"],
                )

        return cls(name=name, description=description, errors=errors)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "QuantumErrorProfile":
        """Load a QuantumErrorProfile from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to read data from.

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
        for gate, error_def in self.errors.items():
            gate_name = str(gate)
            if error_def["type"] == SupportedErrorTypes.DEPOLARIZING_ERROR:
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
                    getattr(noise, gate_name_qdk).set_depolarizing(error_def["rate"])
                except AttributeError:
                    # Warn and skip unsupported gates
                    Logger.warn(f"Gate {gate_name} not supported in QDK noise config; skipping.")
            else:
                raise ValueError(f"Error type {error_def['type']} is not currently supported.")
        return noise
