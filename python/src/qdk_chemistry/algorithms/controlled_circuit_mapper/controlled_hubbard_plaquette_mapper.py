"""Controlled circuit mapper for the plaquette Trotterization of the Fermi-Hubbard model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Circuit, UnitaryRepresentation
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["ControlledHubbardPlaquetteMapper"]


class ControlledHubbardPlaquetteMapper(ControlledCircuitMapper):
    """Map a plaquette unitary representation to its singly-controlled Q# circuit."""

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
            ValueError: If the unitary container type is not supported, or if more than one control qubit is requested.

        """
        container = evolution.get_container()
        if not isinstance(container, HubbardPlaquetteContainer):
            raise ValueError(
                f"The {evolution.get_container_type()} container type is not supported. "
                "ControlledHubbardPlaquetteMapper only supports HubbardPlaquette container for the unitary."
            )
        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("The plaquette mapper currently only supports a single control qubit.")
        # Only the lattice shape and the layer angles cross the boundary; the tilings and
        # spin pairings are derived in Q# from the shape.
        params = QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
            width=container.width,
            height=container.height,
            interactionAngle=container.interaction_angle,
            onsiteAngle=container.onsite_angle,
            identityAngle=container.identity_angle,
            hoppingAngle=container.hopping_angle,
            repetitions=container.step_reps,
        )
        targets = self._get_target_indices(evolution)
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit,
                parameter={"params": params, "control": control_indices[0], "systems": targets},
            ),
            qsharp_op=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOp(params),
        )
