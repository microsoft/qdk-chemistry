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

__all__: list[str] = ["ControlledHubbardPlaquetteMapper", "ControlledHubbardPlaquetteMapperSettings"]


class ControlledHubbardPlaquetteMapperSettings(ControlledCircuitMapperSettings):
    r"""Settings for :class:`ControlledHubbardPlaquetteMapper`.

    Attributes:
        use_hamming_weight_phasing: Whether to phase each tower of equal-angle rotations through a
            Hamming-weight register.
            When ``False``, every term is its own rotation.
            Defaults to ``True``.
        max_hamming_weight_phasing_batch_size: Largest tower of equal-angle rotations phased
            through one Hamming-weight register, or ``-1`` for no cap.
            A smaller cap needs fewer scratch qubits but more rotations.
            Every batch is phased as given, so on the ``Rz`` ladder (``hamming_weight_phasing_method``
            ``"rz_ladder"``) use a cap of at least 8, as shorter batches save too few rotations to be
            worth the extra Toffolis and scratch qubits.
            Ignored when ``use_hamming_weight_phasing`` is ``False``.
            Defaults to ``-1``.
        hamming_weight_phasing_method: How each Hamming weight is phased: ``"catalyst"`` adds it into
            a catalyst, a generalized phase gradient the evolution prepares once per angle, and
            ``"rz_ladder"`` synthesizes each of its place values as its own ``Rz``.
            Ignored when ``use_hamming_weight_phasing`` is ``False``.
            Defaults to ``"catalyst"``.

    """

    def __init__(self):
        """Initialize the settings, adding the Hamming-weight phasing options and the phasing choice."""
        super().__init__()
        self._set_default(
            "use_hamming_weight_phasing",
            "bool",
            True,
            "Whether to phase each tower of equal-angle rotations through a Hamming-weight register. "
            "When False, every term is its own rotation.",
        )
        self._set_default(
            "max_hamming_weight_phasing_batch_size",
            "int",
            -1,
            "Largest tower of equal-angle rotations phased through one Hamming-weight register. "
            "A smaller cap needs fewer scratch qubits but more rotations. Every batch is phased as "
            "given, so on the Rz ladder (hamming_weight_phasing_method 'rz_ladder') use a cap of at "
            "least 8, as shorter batches save too few rotations to be worth the extra Toffolis and "
            "scratch qubits. Ignored when use_hamming_weight_phasing is False. Set to -1 for no cap.",
        )
        self._set_default(
            "hamming_weight_phasing_method",
            "string",
            "catalyst",
            "How each Hamming weight is phased. 'catalyst' adds it into a generalized phase gradient "
            "the evolution prepares once per angle, trading the place-value rotations of every tower "
            "for one rotation and Toffolis. 'rz_ladder' synthesizes each place value as its own Rz, "
            "which needs no catalyst. Ignored when use_hamming_weight_phasing is False.",
            ["catalyst", "rz_ladder"],
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
                requested, or if Hamming-weight phasing is on and ``max_hamming_weight_phasing_batch_size``
                is neither -1 nor positive.

        """
        container = evolution.get_container()
        if not isinstance(container, HubbardPlaquetteContainer):
            raise ValueError(f"The {evolution.get_container_type()} container type is not supported.")
        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("The plaquette mapper currently only supports a single control qubit.")

        max_batch_size = int(self._settings.get("max_hamming_weight_phasing_batch_size"))
        use_hamming_weight_phasing = bool(self._settings.get("use_hamming_weight_phasing"))
        if use_hamming_weight_phasing and max_batch_size != -1 and max_batch_size < 1:
            raise ValueError(
                f"max_hamming_weight_phasing_batch_size must be -1 or a positive integer. Got {max_batch_size}."
            )

        # Only the lattice shape, the layer angles and the phasing settings cross the boundary;
        # the tilings and spin pairings are derived in Q# from the shape.
        params = QSHARP_UTILS.HubbardPlaquette.HubbardPlaquetteParams(
            width=container.width,
            height=container.height,
            interactionAngle=container.interaction_angle,
            hoppingAngle=container.hopping_angle,
            repetitions=container.step_reps,
            useHammingWeightPhasing=use_hamming_weight_phasing,
            maxBatchSize=max_batch_size,
            useCatalyst=self._settings.get("hamming_weight_phasing_method") == "catalyst",
        )
        targets = self._get_target_indices(evolution)
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpCircuit,
                parameter={"params": params, "control": control_indices[0], "systems": targets},
            ),
            qsharp_op=QSHARP_UTILS.HubbardPlaquette.MakeRepControlledPlaquetteExpOp(params),
            num_qubits=1 + container.num_qubits,
        )
