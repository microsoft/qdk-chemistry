"""QDK/Chemistry noise model module for simulating noise in quantum circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, TypedDict

from ruamel.yaml import YAML


class CaseInsensitiveStrEnum(StrEnum):
    """StrEnum that allows case-insensitive lookup of values."""

    @classmethod
    def _missing_(cls, value):  # make input case-insensitive
        if isinstance(value, str):
            for member in cls:
                if member.value.upper() == value.upper():
                    return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class SupportedGate(CaseInsensitiveStrEnum):
    """An enumeration of quantum gate types.

    Gathered from qiskit
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
            gate_str: String representation of the gate (case-insensitive)

        Returns:
            The corresponding SupportedGate enum value

        Raises:
            ValueError: If no matching gate is found

        """
        try:
            # Leverage internal _missing_ method for case-insensitive lookup
            return cls(gate_str)
        except ValueError:
            # If the gate_str does not match any enum value, raise an error
            raise ValueError(f"Unknown gate type: {gate_str}") from None


class SupportedErrorTypes(CaseInsensitiveStrEnum):
    """Supported error types for quantum gates."""

    DEPOLARIZING_ERROR = "depolarizing_error"


class GateErrorDef(TypedDict):
    """Typed dictionary for error definitions."""

    type: SupportedErrorTypes
    """Error type."""

    rate: float
    """Error rate."""

    num_qubits: int
    """Number of qubits the gate acts on."""


#: Gates to exclude from basis gates in noise model.
BASIS_GATES_EXCLUSION = {"reset", "barrier", "measure"}


@dataclass
class QuantumErrorProfile:
    """A class representing a quantum error profile containing information about quantum gates and error properties.

    This class provides functionalities to define, load, and save quantum error profiles
    from YAML files, and convert them to Qiskit noise models for simulations.
    """

    name: str
    """Name of the quantum error profile."""

    description: str
    """Description of what the error profile represents."""

    errors: dict[SupportedGate, GateErrorDef] = field(default_factory=dict)
    """Dictionary mapping gate names to their error properties."""

    one_qubit_gates: list[str] = field(default_factory=list)
    """Set of gate names that operate on a single qubit."""

    two_qubit_gates: list[str] = field(default_factory=list)
    """Set of gate names that operate on two qubits."""

    supported_yaml_keys: ClassVar[set[str]] = {
        "name",
        "description",
        "errors",
    }
    """YAML keys supported in the quantum error profile."""

    def __post_init__(self):
        """Initialize one_qubit_gates and two_qubit_gates based on errors."""
        # Clear existing sets to avoid duplication when loading from YAML
        self.one_qubit_gates.clear()
        self.two_qubit_gates.clear()

        for gate, error_dict in self.errors.items():
            if error_dict["num_qubits"] == 1:
                self.one_qubit_gates.append(str(gate))
            elif error_dict["num_qubits"] == 2:
                self.two_qubit_gates.append(str(gate))
            else:
                raise ValueError(f"Unsupported number of qubits: {error_dict['num_qubits']}")

        self.one_qubit_gates = sorted(set(self.one_qubit_gates))
        self.two_qubit_gates = sorted(set(self.two_qubit_gates))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for YAML serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "errors": {},
        }

        # Convert enum keys and values to strings in the errors dictionary
        for gate, error_def in self.errors.items():
            gate_str = str(gate)
            error_dict = dict(error_def)
            error_dict["type"] = str(error_dict["type"])
            result["errors"][gate_str] = error_dict

        return result

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> "QuantumErrorProfile":
        """Load quantum error profile from YAML file.

        Args:
            yaml_file: Path to YAML file

        Returns:
            Loaded profile

        """
        yaml = YAML(typ="safe")  # type: ignore
        if not Path(yaml_file).exists():
            raise FileNotFoundError(f"File {yaml_file} not found")

        with Path(yaml_file).open("r") as f:
            data = yaml.load(f)

        if data is None:
            raise ValueError(f"YAML file {yaml_file} is empty or invalid.")

        data_keys = data.keys()

        invalid_keys = set(data_keys) - cls.supported_yaml_keys
        if invalid_keys:
            raise ValueError(f"Invalid keys in YAML file: {invalid_keys}.Only {cls.supported_yaml_keys} are allowed.")

        name = data.get("name", "default")
        description = data.get("description", "No description provided")
        errors_raw = data.get("errors", {})

        # Convert string keys back to Enum and ensure error types are Enums
        errors: dict[SupportedGate, GateErrorDef] = {}
        for gate_str, error_dict in errors_raw.items():
            gate = SupportedGate(gate_str)
            errors[gate] = GateErrorDef(
                type=SupportedErrorTypes(error_dict["type"]),
                rate=error_dict["rate"],
                num_qubits=error_dict["num_qubits"],
            )

        return cls(
            name=name,
            description=description,
            errors=errors,
        )

    def to_yaml(self, yaml_file: str | Path) -> None:
        """Save quantum error profile to YAML file.

        Args:
            yaml_file: Path to save YAML file

        """
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Convert to serializable dict
        data = self.to_dict()

        with Path(yaml_file).open("w") as f:
            yaml.dump(data, f)

    @property
    def basis_gates(self) -> list[str]:
        """Get basis gates from profile."""
        gates = []
        for gate in self.one_qubit_gates + self.two_qubit_gates:
            if gate not in BASIS_GATES_EXCLUSION:
                gates.append(gate)
        return gates
