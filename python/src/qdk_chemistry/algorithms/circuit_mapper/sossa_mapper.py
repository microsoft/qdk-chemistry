"""QDK/Chemistry Sum of Squares Spectral Amplification (SOSSA) circuit mapper :cite:`Low2025`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

from qdk_chemistry.data import AlgorithmRef, SettingNotFound, Settings
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSABlockEncodingContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = [
    "SOSSAMapper",
    "SOSSAMapperSettings",
]


class SOSSAMapperSettings(Settings):
    """Settings for the SOSSAMapper."""

    def __init__(self):
        """Initialize settings for SOSSAMapper."""
        super().__init__()
        self._set_default(
            "outer_prepare_algorithm",
            "algorithm_ref",
            AlgorithmRef("state_prep", "alias_sampling"),
        )
        self._set_default(
            "inner_prepare_algorithm",
            "string",
            "controlled_alias_sampling",
            "Inner PREPARE algorithm ('controlled_alias_sampling' or 'direct').",
            ["controlled_alias_sampling", "direct"],
        )
        self._set_default(
            "select_algorithm",
            "string",
            "qrom_phase_gradient",
            "SELECT algorithm ('qrom_phase_gradient' or 'direct').",
            ["qrom_phase_gradient", "direct"],
        )
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Number of bits for Givens rotation angle precision, from 1 to 30 inclusive.",
            (1, 30),
        )
        self._set_default(
            "coefficient_bit_precision",
            "int",
            10,
            "Number of bits for alias sampling coefficient precision, from 1 to 30 inclusive.",
            (1, 30),
        )


class SOSSAMapper(CircuitMapper):
    r"""Circuit mapper for the Sum of Squares Spectral Amplification (SOSSA) block encoding :cite:`Low2025`.

    Emits uncontrolled :math:`B = U^\dagger \cdot \mathrm{Ref}_B \cdot U`
    on ``[system | ancillas | phase gradient]``. The phase-gradient suffix is
    shared ancilla and is excluded from the reflected zero register.

    Unary QPE controls walk iterations through its own index register. Iterative and
    standard QPE need a controlled quantum walk mapper, which this class does not provide.
    """

    def __init__(self):
        """Initialize the SOSSAMapper."""
        super().__init__()
        self._settings = SOSSAMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "circuit_mapper"

    def _build_outer_prepare_circuit(self, container: SOSSABlockEncodingContainer) -> Circuit:
        r"""Build the outer PREPARE circuit.

        Args:
            container: The SOSSA container with outer_prepare coefficients.

        Returns:
            The PREPARE circuit to embed in the block encoding.

        """
        prepare_algorithm = self._create_nested("outer_prepare_algorithm")
        prepare_settings = prepare_algorithm.settings()
        # Configure by capability so aliases of these PREPARE implementations keep the same SOSSA contract.
        for key, value in (
            ("bits_precision", self._settings.get("coefficient_bit_precision")),
            ("allocate_phase_gradient", False),
        ):
            try:
                prepare_settings.get(key)
            except SettingNotFound:
                continue
            prepare_settings.set(key, value)
        circuit = prepare_algorithm.run(container.outer_prepare)
        if circuit._qsharp_op is None:  # noqa: SLF001
            raise ValueError("The outer PREPARE circuit has no Q# operation to embed in the SOSSA block encoding.")
        if circuit.num_qubits is None:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' does not declare num_qubits, so the "
                "outer register cannot be sized."
            )
        return circuit

    def _build_inner_oracles(self, container: SOSSABlockEncodingContainer) -> tuple[Any, Any]:
        r"""Build the Q# inner PREPARE and free-rider load callables.

        Creates a superposition over bases :math:`b` conditioned on :math:`x_o`. The
        alias-sampling factory places the free-rider table in the cheaper of the inner
        PREPARE and a separate load performed once per block encoding.

        Algorithms:
            - ``"controlled_alias_sampling"``: 2D alias sampling.
            - ``"direct"``: Direct multiplexed preparation (ControlledPureStatePrep).

        Args:
            container: The SOSSA container with inner_prepare coefficients.

        Returns:
            The inner PREPARE and free-rider load callables.

        """
        algorithm = self._settings.get("inner_prepare_algorithm")
        coeff_bits = self._settings.get("coefficient_bit_precision")
        coefficients = container.inner_prepare.conditional_coefficients.tolist()
        free_rider_data = container.inner_prepare.free_rider_data
        free_rider_data = free_rider_data.tolist() if free_rider_data is not None else []

        if algorithm == "controlled_alias_sampling":
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareAliasSamplingOracles(
                coefficients,
                free_rider_data,
                coeff_bits,
            )
        if algorithm == "direct":
            return (
                QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients, free_rider_data),
                QSHARP_UTILS.SOSSAWalk.MakeFreeRiderLoadOp(free_rider_data),
            )
        raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{algorithm}'.")

    def _build_select(self, container: SOSSABlockEncodingContainer) -> Any:
        r"""Build the SELECT step.

        Args:
            container: The SOSSA container with rotation angles and structure.

        Returns:
            A Q# callable for the SELECT oracle.

        """
        algorithm = self._settings.get("select_algorithm")
        rot_bits = self._settings.get("rotation_bit_precision")

        meta = container.metadata
        num_free_rider_bits = container.layout.num_free_rider_bits
        inner_prep_bits = container.layout.inner_prep_bits
        inner_algorithm = self._settings.get("inner_prepare_algorithm")
        if inner_algorithm == "controlled_alias_sampling":
            sign_qubit_index = 2 * inner_prep_bits + 2 * self._settings.get("coefficient_bit_precision") + 1
        elif inner_algorithm == "direct":
            sign_qubit_index = inner_prep_bits
        else:
            raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{inner_algorithm}'.")

        select_data = {
            "numOrbitals": meta.num_spatial_orbitals,
            "numRanks": meta.num_ranks,
            "numBases": meta.num_bases,
            "numCopies": meta.num_copies,
            "numPositiveOneBody": meta.num_positive_one_body_terms,
            "OneBodyRotationAngles": container.select.one_body_rotation_angles.tolist(),
            "TwoBodyRotationAngles": container.select.two_body_rotation_angles.tolist(),
            "rotationBitPrecision": rot_bits,
            "numFreeRiderBits": num_free_rider_bits,
            "signQubitIndex": sign_qubit_index,
        }
        if algorithm == "qrom_phase_gradient":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectPhaseGradient(select_data)
        if algorithm == "direct":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectDirectRotation(select_data)
        raise ValueError(f"Unsupported SOSSA SELECT algorithm '{algorithm}'.")

    def _compute_register_sizes(
        self, container: SOSSABlockEncodingContainer, outer_prepare_circuit: Circuit
    ) -> tuple[dict[str, int], Any]:
        """Compute the register widths and the Q# ``SOSSAWalkLayout`` describing them.

        Args:
            container: The SOSSA block encoding container.
            outer_prepare_circuit: The outer PREPARE circuit already created for this block encoding.

        Returns:
            The width map, and the Q# ``SOSSAWalkLayout`` built from it.

        """
        meta = container.metadata
        layout = container.layout
        num_orbitals = meta.num_spatial_orbitals
        num_system_qubits = 2 * num_orbitals
        outer_prep_bits = layout.outer_prep_bits
        inner_prep_bits = layout.inner_prep_bits
        num_free_rider_bits = layout.num_free_rider_bits

        outer_prepare_width = outer_prepare_circuit.num_qubits
        if outer_prepare_width is None:
            raise ValueError(
                "The outer PREPARE circuit does not declare num_qubits, so the outer register cannot be sized."
            )
        outer_gradient_bits = outer_prepare_circuit.metadata.num_phase_gradient_ancillas
        num_outer_qubits = outer_prepare_width - outer_gradient_bits
        if num_outer_qubits < outer_prep_bits:
            raise ValueError(
                f"The outer PREPARE circuit owns {num_outer_qubits} non-gradient qubits, but the SOSSA "
                f"layout indexes {outer_prep_bits}."
            )

        inner_algorithm = self._settings.get("inner_prepare_algorithm")
        if inner_algorithm == "controlled_alias_sampling":
            mu_inner = self._settings.get("coefficient_bit_precision")
            num_inner_qubits = 2 * inner_prep_bits + 2 * mu_inner + 3 + num_free_rider_bits
            num_reflect_inner = inner_prep_bits + mu_inner + 1
        elif inner_algorithm == "direct":
            # The extra qubit is the sign bit SELECT phases
            num_inner_qubits = inner_prep_bits + 1 + num_free_rider_bits
            num_reflect_inner = inner_prep_bits
        else:
            raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{inner_algorithm}'.")

        select_algorithm = self._settings.get("select_algorithm")
        if select_algorithm == "qrom_phase_gradient":
            select_gradient_bits = int(self._settings.get("rotation_bit_precision"))
        elif select_algorithm == "direct":
            select_gradient_bits = 0
        else:
            raise ValueError(f"Unsupported SOSSA SELECT algorithm '{select_algorithm}'.")
        if outer_gradient_bits and select_gradient_bits and outer_gradient_bits != select_gradient_bits:
            raise ValueError(
                "The outer PREPARE and SELECT share one phase gradient register and must agree "
                f"on its width. Now, the outer PREPARE: {outer_gradient_bits} and SELECT: {select_gradient_bits}."
            )
        num_phase_gradient_qubits = max(outer_gradient_bits, select_gradient_bits)
        num_spin_qubits = 2  # spinDQ + spinSF, matches Q# SOSSAWalk.qs

        regs = {
            "num_system_qubits": num_system_qubits,
            "num_outer_qubits": num_outer_qubits,
            "num_outer_index_qubits": outer_prep_bits,
            "num_inner_qubits": num_inner_qubits,
            "num_reflect_inner": num_reflect_inner,
            "num_phase_gradient_qubits": num_phase_gradient_qubits,
            "num_outer_prepare_gradient_qubits": outer_gradient_bits,
            "num_ancilla_qubits": num_outer_qubits + num_reflect_inner + num_spin_qubits + num_phase_gradient_qubits,
        }
        register_layout = QSHARP_UTILS.SOSSAWalk.SOSSAWalkLayout(
            numSystemQubits=regs["num_system_qubits"],
            numOuterQubits=regs["num_outer_qubits"],
            numOuterIndexQubits=regs["num_outer_index_qubits"],
            numOuterPrepareGradientQubits=regs["num_outer_prepare_gradient_qubits"],
            numInnerQubits=regs["num_inner_qubits"],
            numReflectInner=regs["num_reflect_inner"],
            numFreeRiderQubits=num_free_rider_bits,
            numPhaseGradientQubits=regs["num_phase_gradient_qubits"],
        )
        return regs, register_layout

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct the SOSSA block encoding on the flat ``[system | ancilla]`` register.

        Args:
            unitary: The unitary representation containing the SOSSA decomposition.

        Returns:
            Circuit: The block encoding :math:`B`, declaring the full register width and the
            phase gradient qubits its caller must prepare.

        Raises:
            ValueError: If the container is not a :class:`SOSSABlockEncodingContainer`.

        """
        container = unitary.get_container()
        if not isinstance(container, SOSSABlockEncodingContainer):
            raise ValueError(f"The {unitary.get_container_type()} container type is not supported.")
        free_rider = container.inner_prepare.free_rider_data
        if container.layout.num_free_rider_bits and (free_rider is None or free_rider.size == 0):
            raise ValueError(
                f"The register layout reserves {container.layout.num_free_rider_bits} free-rider bits "
                "but the container carries no free-rider table."
            )
        if container.power != 1:
            Logger.warn(f"The container's power {container.power} is ignored.")

        outer_prepare_circuit = self._build_outer_prepare_circuit(container)
        regs, register_layout = self._compute_register_sizes(container, outer_prepare_circuit)
        outer_prepare_op = outer_prepare_circuit._qsharp_op  # noqa: SLF001
        inner_prepare_op, free_rider_op = self._build_inner_oracles(container)
        select_op = self._build_select(container)

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.SOSSAWalk.MakeSOSSABlockEncodingCircuit,
            parameter={
                "outerPrepareOp": outer_prepare_op,
                "freeRiderOp": free_rider_op,
                "innerPrepareOp": inner_prepare_op,
                "selectOp": select_op,
                "layout": register_layout,
            },
        )
        qsharp_op = QSHARP_UTILS.SOSSAWalk.MakeSOSSABlockEncodingOp(
            outer_prepare_op,
            free_rider_op,
            inner_prepare_op,
            select_op,
            register_layout,
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            num_qubits=regs["num_system_qubits"] + regs["num_ancilla_qubits"],
            metadata=CircuitMetadata(num_phase_gradient_ancillas=regs["num_phase_gradient_qubits"]),
        )
