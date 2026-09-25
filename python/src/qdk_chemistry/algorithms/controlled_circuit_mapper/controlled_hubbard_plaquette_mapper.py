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

__all__: list[str] = ["ControlledHubbardPlaquetteMapper", "plaquette_parameters", "require_plaquette_container"]


def plaquette_parameters(container: HubbardPlaquetteContainer):
    """Lower a plaquette container to its Q# parameter struct.

    Only the lattice shape and the layer angles cross the boundary; the tilings and spin
    pairings are derived in Q# from the shape.

    Args:
        container: The evolution to lower.

    Returns:
        The Q# ``HubbardPlaquetteParams`` describing the evolution.

    """
    return QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
        width=container.width,
        height=container.height,
        interactionAngle=container.interaction_angle,
        onsiteAngle=container.onsite_angle,
        identityAngle=container.identity_angle,
        hoppingAngle=container.hopping_angle,
        repetitions=container.step_reps,
    )


def require_plaquette_container(evolution: UnitaryRepresentation) -> HubbardPlaquetteContainer:
    """Return the evolution's container, rejecting any other representation.

    Args:
        evolution: The unitary representation to lower.

    Returns:
        HubbardPlaquetteContainer: The wrapped container.

    Raises:
        TypeError: If the representation is not a plaquette evolution.

    """
    container = evolution.get_container()
    if not isinstance(container, HubbardPlaquetteContainer):
        raise TypeError(
            f"The plaquette mapper requires a HubbardPlaquetteContainer, but the representation "
            f"wraps a {type(container).__name__}."
        )
    return container


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
