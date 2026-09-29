"""Controlled circuit mapper for the plaquette Trotterization of the Fermi-Hubbard model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Circuit, UnitaryRepresentation
from qdk_chemistry.data.circuit import CircuitMetadata, PhaseGradient, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = ["ControlledHubbardPlaquetteMapper"]


class ControlledHubbardPlaquetteMapperSettings(ControlledCircuitMapperSettings):
    r"""Settings for :class:`ControlledHubbardPlaquetteMapper`.

    Attributes:
        rotation_bit_precision: Width of the binary phase gradient register the Hamming-weight
            rotations are applied through, so each one is exact to :math:`2\pi/2^b`. Defaults to
            10, as :class:`~qdk_chemistry.algorithms.state_preparation.qrom_state_prep` does.

    """

    def __init__(self):
        """Initialize the settings, adding the phase gradient width."""
        super().__init__()
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Width of the phase gradient register the Hamming-weight rotations are applied through. "
            "The upper bound of 30 is a sanity limit as 2^-30 is already far below chemical accuracy.",
            (1, 30),
        )


class ControlledHubbardPlaquetteMapper(ControlledCircuitMapper):
    """Map a plaquette unitary representation to its singly-controlled Q# circuit."""

    def __init__(self):
        """Initialize the mapper with its own settings."""
        super().__init__()
        self._settings = ControlledHubbardPlaquetteMapperSettings()

    def name(self) -> str:
        """Return ``hubbard_plaquette`` as the algorithm name."""
        return "hubbard_plaquette"

    def type_name(self) -> str:
        """Return ``controlled_circuit_mapper`` as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, evolution: UnitaryRepresentation) -> Circuit:
        """Build the controlled circuit for a plaquette evolution.

        Args:
            evolution: The plaquette unitary representation to be mapped.

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
            hoppingAngle=container.hopping_angle,
            repetitions=container.step_reps,
            rotationBitPrecision=int(self._settings.get("rotation_bit_precision")),
        )
        targets = self._get_target_indices(evolution)
        num_gradient = QSHARP_UTILS.HubbardPlaquette.PlaquetteGradientSize(params)
        if num_gradient == 0:
            return Circuit(
                qsharp_factory=QsharpFactoryData(
                    program=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit,
                    parameter={"params": params, "control": control_indices[0], "systems": targets},
                ),
                qsharp_op=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOp(params),
            )

        # The Hamming-weight towers phase through one binary phase gradient, whose state does not
        # depend on any angle. The Q# operation expects it at the end of its targets, so phase
        # estimation prepares it once for every query; the standalone factory program prepares
        # its own.
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit,
                parameter={"params": params, "control": control_indices[0], "systems": targets},
            ),
            qsharp_op=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOp(params),
            num_qubits=len(targets) + num_gradient,
            metadata=CircuitMetadata(phase_gradients=(PhaseGradient.binary(num_gradient),)),
        )
