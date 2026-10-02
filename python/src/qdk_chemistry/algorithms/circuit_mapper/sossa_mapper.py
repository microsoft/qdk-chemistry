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
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = [
    "SOSSAMapper",
    "SOSSAMapperSettings",
    "rotation_batch_size_for",
]

#: Maps the ``lookup_method`` setting onto the Q# ``Lookup*`` constants.
_LOOKUP_METHODS: dict[str, int] = {
    "select": 0,
    "select_swap": 1,
    "dirty_select_swap": 2,
}


def rotation_batch_size_for(num_orbitals: int, num_batches: int) -> int:
    r"""Smallest ``rotation_batch_size`` that still streams the angles in ``num_batches`` passes.

    Choose the number of passes, not :math:`\lambda`. Streaming costs one extra table
    lookup per batch in each direction, so its Toffoli penalty tracks the batch count
    :math:`\lceil (N-1)/\lambda \rceil` rather than :math:`\lambda` itself. That makes the
    penalty a step function: every :math:`\lambda` inside one step buys exactly the same
    Toffolis, while the rotation register keeps growing at ``b_rot`` qubits per angle. Only
    the smallest member of each step can be optimal, and that is what this returns.

    Picking :math:`\lambda` directly is how callers pay width for nothing. At Fe2S2-20
    (:math:`N = 20`, so 19 angles) :math:`\lambda = 16` and :math:`\lambda = 10` both make
    two passes and cost an identical 60.3M Toffolis, but 16 costs 13 more qubits.

    Args:
        num_orbitals: Number of spatial orbitals :math:`N`. SELECT holds :math:`N - 1`
            Givens angles, so that is the number being split into batches.
        num_batches: Number of passes over the angle table, from 1 to :math:`N - 1`. One
            pass keeps every angle resident, which is the cheapest in Toffolis and the
            widest in qubits; more passes trade the one against the other.

    Returns:
        The value to pass as the ``rotation_batch_size`` setting.

    Raises:
        ValueError: If ``num_orbitals`` is below 2, or ``num_batches`` is outside
            ``1..num_orbitals - 1``.

    """
    if num_orbitals < 2:
        raise ValueError(f"num_orbitals must be at least 2 to hold a rotation angle, got {num_orbitals}")

    num_angles = num_orbitals - 1
    if not 1 <= num_batches <= num_angles:
        raise ValueError(
            f"num_batches must be between 1 and {num_angles} for {num_orbitals} orbitals, got {num_batches}"
        )

    return -(-num_angles // num_batches)


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
            "rotation_batch_size",
            "int",
            0,
            "Number of Givens angles held in the rotation register at once (the SOSSA lambda). "
            "0 keeps all N-1 angles resident, which is the cheapest in Toffolis. Smaller values "
            "stream the angles in batches, cutting the rotation register to lambda*b_rot qubits "
            "at the cost of one extra table lookup per batch, in each direction. The Toffoli "
            "penalty follows the batch count, ceil((N-1)/lambda), so it is a step function of "
            "lambda: every lambda within one step costs the same Toffolis while the register "
            "keeps growing, making all but the smallest of them strictly wasteful. Choose the "
            "number of batches and derive lambda with rotation_batch_size_for() rather than "
            "setting lambda directly.",
            (0, 4096),
        )
        self._set_default(
            "lookup_method",
            "string",
            "select_swap",
            "How every QROM table in the walk is routed. One choice governs both lookups -- the "
            "streamed rotation batches and the inner alias-sampling tables -- because they draw "
            "on the same budget and a caller who is short of qubits is short of them everywhere. "
            "'select' is a plain unary-iteration lookup: no extra qubits, Toffoli cost one per "
            "table row. 'select_swap' is a clean QROAM that cuts those Toffolis but allocates "
            "scratch proportional to 2^k times the loaded word. 'dirty_select_swap' runs the "
            "same network on borrowed qubits that are provably idle across the load, so it costs "
            "no width at all, but it runs Select twice and the butterfly four times and so pays "
            "roughly two to three times the Toffolis of the clean network at equal width. "
            "Every method falls back to a plain lookup at shapes where its own cost model says "
            "no network pays, so naming one can only ever spend qubits that buy something. "
            "Borrowing in particular only undercuts a plain lookup on tables large relative to "
            "the loaded word, roughly numData > 32 * numBits.",
            ["select", "select_swap", "dirty_select_swap"],
        )
        self._set_default(
            "coefficient_bit_precision",
            "int",
            10,
            "Number of bits for alias sampling coefficient precision, from 1 to 30 inclusive.",
            (1, 30),
        )
        self._set_default(
            "inner_prepare_swap_bits",
            "int",
            -1,
            "Swap width k of the QROAM that loads the inner alias-sampling tables. -1 lets the "
            "library pick, 0 forces a plain unary-iteration lookup, and a positive value fixes "
            "k, clamped to the table's address width. The swap network allocates scratch "
            "proportional to 2^k times the loaded word, and that word carries "
            "'coefficient_bit_precision', so k multiplies the cost of every coefficient bit. "
            "The default selector takes the narrowest width within a fifth of the Toffoli "
            "optimum, which declines the last widening or two that the Toffoli minimum would "
            "take; set k explicitly to get that minimum back. Lowering k by one is exact -- it "
            "changes only how identical data is routed, never the state prepared -- which "
            "makes it the one width knob here that costs no accuracy. Expect a modest Toffoli "
            "increase in return, and re-derive 'rotation_batch_size' afterwards, since "
            "narrowing PREPARE can put SELECT back on the critical path.",
            (-1, 30),
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
                self._settings.get("inner_prepare_swap_bits"),
                self._lookup_method(),
            )
        if algorithm == "direct":
            return (
                QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients, free_rider_data),
                QSHARP_UTILS.SOSSAWalk.MakeFreeRiderLoadOp(free_rider_data),
            )
        raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{algorithm}'.")

    def _lookup_method(self) -> int:
        r"""Resolve ``lookup_method`` to the Q# tag both loaders branch on.

        One setting feeds the streamed rotation batches and the inner alias-sampling
        tables alike, so a caller cannot accidentally route one table through borrowed
        qubits and the other through allocated scratch.

        Returns:
            The Q# ``Lookup*`` constant for the configured method.

        Raises:
            ValueError: If the configured method is not one of the three known tags.

        """
        method = self._settings.get("lookup_method")
        if method not in _LOOKUP_METHODS:
            raise ValueError(f"Unsupported SOSSA lookup method '{method}'.")
        return _LOOKUP_METHODS[method]

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

        lookup_method = self._lookup_method()

        select_data = {
            "numOrbitals": meta.num_spatial_orbitals,
            "numRanks": meta.num_ranks,
            "numBases": meta.num_bases,
            "numCopies": meta.num_copies,
            "numPositiveOneBody": meta.num_positive_one_body_terms,
            "OneBodyRotationAngles": container.select.one_body_rotation_angles.tolist(),
            "TwoBodyRotationAngles": container.select.two_body_rotation_angles.tolist(),
            "rotationBitPrecision": rot_bits,
            "rotationBatchSize": int(self._settings.get("rotation_batch_size")),
            "rotationLookupMethod": lookup_method,
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
            ValueError: If the container is not a :class:`SOSSABlockEncodingContainer` or its
                power is not one.

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
            raise ValueError(f"The SOSSA mapper supports only unit power, got {container.power}.")

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
