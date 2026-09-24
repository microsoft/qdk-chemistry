"""Controlled circuit mapper for the plaquette Trotterization of the Fermi-Hubbard model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.circuit_mapper.hubbard_plaquette_mapper import (
    plaquette_parameters,
    require_plaquette_container,
)
from qdk_chemistry.data import Circuit, UnitaryRepresentation
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["ControlledHubbardPlaquetteMapper"]


class ControlledHubbardPlaquetteMapper(ControlledCircuitMapper):
    """Map a plaquette evolution to its singly-controlled Q# circuit."""

    def name(self) -> str:
        """Return ``hubbard_plaquette`` as the algorithm name."""
        return "hubbard_plaquette"

    def type_name(self) -> str:
        """Return ``controlled_circuit_mapper`` as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, evolution: UnitaryRepresentation) -> Circuit:
        """Build the controlled circuit for a plaquette evolution.

        Args:
            evolution: The plaquette evolution to lower.

        Returns:
            Circuit: The Q# circuit applying the controlled evolution.

        Raises:
            ValueError: If more than one control qubit is requested.

        """
        container = require_plaquette_container(evolution)
        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("The plaquette mapper currently only supports a single control qubit.")
        params = plaquette_parameters(container)
        targets = self._get_target_indices(evolution)
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit,
                parameter={"params": params, "control": control_indices[0], "systems": targets},
            ),
            qsharp_op=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOp(params),
        )
