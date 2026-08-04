"""QDK/Chemistry circuit mapper for controlled-unitary abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

__all__: list[str] = [
    "ControlledCircuitMapper",
    "ControlledCircuitMapperFactory",
    "ControlledCircuitMapperSettings",
    "UnaryIterationWalkMapper",
]


@runtime_checkable
class UnaryIterationWalkMapper(Protocol):
    """Capability a controlled circuit mapper needs to drive unary-iteration phase estimation.

    A mapper satisfies this interface when its block encoding is a self-inverse walk
    operator whose reflection can be skipped under the control of a phase register, so
    that branch ``t`` realizes :math:`W^{p - 2t}` while still applying exactly :math:`p`
    blocks. Membership is structural: any mapper defining these three methods qualifies,
    without having to subclass anything.

    """

    def build_walk_op(
        self,
        unitary: UnitaryRepresentation,
        num_queries: int,
        use_unary_iteration: bool = True,
    ) -> Any:
        """Build the walk callable acting on (control register, system + ancilla register).

        Args:
            unitary: The unitary representation holding the block encoding.
            num_queries: Number of walk blocks to apply.
            use_unary_iteration: Whether the control register is a phase register iterated over.

        Returns:
            A Q# callable accepting the control register and the system/ancilla register.

        """
        ...

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the walk operator needs beyond the system register."""
        ...

    def get_ancilla_prep_op(self) -> Any:
        """Return the Q# callable that initializes those ancillas.

        Returns:
            A Q# callable preparing the ancilla register, or a no-op when none is needed.

        """
        ...


class ControlledCircuitMapperSettings(Settings):
    """Settings for the ControlledCircuitMapper.

    Attributes:
        control_indices: The control qubit indices. Defaults to ``[0]``.
        target_indices: The target qubit indices. An empty list means auto-fill
            based on the unitary's qubit count and control indices.

    """

    def __init__(self):
        """Initialize the settings for ControlledCircuitMapper."""
        super().__init__()
        self._set_default(
            "control_indices",
            "vector<int>",
            [0],
            "The control qubit indices.",
        )
        self._set_default(
            "target_indices",
            "vector<int>",
            [],
            "The target qubit indices. Empty means auto-fill.",
        )


class ControlledCircuitMapper(Algorithm):
    """Base class for circuit mapper for controlled-unitary in QDK/Chemistry algorithms."""

    def __init__(self):
        """Initialize the ControlledCircuitMapper."""
        super().__init__()
        self._settings = ControlledCircuitMapperSettings()

    @abstractmethod
    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        """Construct a Circuit representing the controlled unitary.

        Args:
            unitary: The unitary representation to be controlled.
                Control and target indices are read from settings.

        Returns:
            Circuit: A Circuit representing the controlled unitary.

        """

    def _get_control_indices(self) -> list[int]:
        """Get control indices from settings.

        Returns:
            The control qubit indices.

        """
        control_indices = self._settings.get("control_indices")
        if len(control_indices) != len(set(control_indices)):
            raise ValueError("control_indices must not contain duplicates.")
        return control_indices

    def _get_target_indices(self, unitary: UnitaryRepresentation) -> list[int]:
        """Get target indices from settings, auto-filling if empty.

        Args:
            unitary: The unitary representation, used for auto-fill.

        Returns:
            The resolved target qubit indices.

        """
        target_indices = self._settings.get("target_indices")
        if target_indices:
            if len(target_indices) != len(set(target_indices)):
                raise ValueError("target_indices must not contain duplicates.")
            if len(target_indices) != unitary.get_num_qubits():
                raise ValueError(
                    f"target_indices length ({len(target_indices)}) "
                    f"must match unitary qubit count ({unitary.get_num_qubits()})."
                )
            if any(idx in self._get_control_indices() for idx in target_indices):
                raise ValueError("target_indices must not overlap with control_indices.")
            return target_indices
        control_indices = self._get_control_indices()
        control_set = set(control_indices)
        num_target = unitary.get_num_qubits()
        targets: list[int] = []
        i = 0
        while len(targets) < num_target:
            if i not in control_set:
                targets.append(i)
            i += 1
        return targets


class ControlledCircuitMapperFactory(AlgorithmFactory):
    """Factory class for creating ControlledCircuitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def default_algorithm_name(self) -> str:
        """Return pauli_sequence as the default algorithm name."""
        return "pauli_sequence"
