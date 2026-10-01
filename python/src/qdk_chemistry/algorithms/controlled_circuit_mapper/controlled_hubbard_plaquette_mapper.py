"""Controlled circuit mapper for the plaquette Trotterization of the Fermi-Hubbard model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Circuit, UnitaryRepresentation
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = ["ControlledHubbardPlaquetteMapper"]


class ControlledHubbardPlaquetteMapperSettings(ControlledCircuitMapperSettings):
    r"""Settings for :class:`ControlledHubbardPlaquetteMapper`.

    Attributes:
        max_hwp_batch_size: Largest tower of equal-angle rotations phased through a single
            Hamming-weight register, or ``-1`` for no cap. Defaults to ``-1``.

    """

    def __init__(self):
        """Initialize the settings, adding the Hamming-weight batch cap."""
        super().__init__()
        self._set_default(
            "max_hwp_batch_size",
            "int",
            -1,
            "Largest tower of equal-angle rotations phased through a single Hamming-weight "
            "register. A shorter batch releases its adder-tree scratch sooner, so the peak "
            "ancilla count follows the batch rather than the whole lattice, at the cost of one "
            "extra set of place-value rotations per batch. Set to -1 for no cap.",
            (-1, 1 << 20),
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
            ValueError: If the unitary container type is not supported, if more than one control qubit is
                requested, or if ``max_hwp_batch_size`` is neither -1 nor positive.

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

        max_batch_size = int(self._settings.get("max_hwp_batch_size"))
        if max_batch_size != -1 and max_batch_size < 1:
            raise ValueError(f"max_hwp_batch_size must be -1 or a positive integer. Got {max_batch_size}.")

        # Only the lattice shape, the layer angles and the batch cap cross the boundary; the
        # tilings and spin pairings are derived in Q# from the shape. The scalar shift stays
        # classical and is applied by HubbardPlaquetteContainer.eigenvalue_from_phase.
        params = QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
            width=container.width,
            height=container.height,
            interactionAngle=container.interaction_angle,
            hoppingAngle=container.hopping_angle,
            repetitions=container.step_reps,
            maxBatchSize=max_batch_size,
        )
        targets = self._get_target_indices(evolution)
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit,
                parameter={"params": params, "control": control_indices[0], "systems": targets},
            ),
            qsharp_op=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOp(params),
        )
