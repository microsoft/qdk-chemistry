"""QDK/Chemistry SOSSA (Sum of Squares Spectral Amplification) circuit mapper :cite:`Low2025`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAWalkContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = [
    "SOSSAMapper",
    "SOSSAMapperSettings",
]

# Two SELECT calls, each conjugated by the inner PREPARE, so the alias table is read four times.
_INNER_LOOKUPS_PER_BLOCK_ENCODING = 4


class SOSSAMapperSettings(Settings):
    """Settings for the SOSSAMapper."""

    def __init__(self):
        """Initialize settings for SOSSAMapper."""
        super().__init__()
        self._set_default(
            "outer_prepare",
            "algorithm_ref",
            AlgorithmRef("state_prep", "alias_sampling"),
        )
        self._set_default(
            "inner_prepare_algorithm",
            "string",
            "controlled_alias_sampling",
            "Inner PREPARE algorithm: controlled_alias_sampling or direct.",
        )
        self._set_default(
            "select_algorithm",
            "string",
            "qrom_phase_gradient",
            "SELECT algorithm: qrom_phase_gradient or direct.",
        )
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Number of bits for Givens rotation angle precision.",
        )
        self._set_default(
            "coefficient_bit_precision",
            "int",
            10,
            "Number of bits for alias sampling coefficient precision.",
        )


class SOSSAMapper(CircuitMapper):
    r"""Circuit mapper for the SOSSA block encoding :math:`B` :cite:`Low2025`.

    Emits :math:`B = U^\dagger \cdot \mathrm{Ref}_B \cdot U` on the flat register
    ``[system | ancillas | phase gradient]``. The
    walk :math:`W = \mathrm{Ref}_{a,B} \cdot B` is left to the caller, which reflects about
    the ancillas ahead of the phase gradient; that reflection is the generic
    ``MakeAncillaReflectionOp`` because the ancillas the all-zero state flags are the
    leading contiguous block.
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

    def _build_outer_prep(self, container: SOSSAWalkContainer) -> tuple[Any, int]:
        r"""Build the Q# outer PREPARE callable and the gradient width it expects.

        Args:
            container: The SOSSA container with outer_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[]) => Unit is Adj + Ctl`` and the number of shared
            phase-gradient qubits it expects appended to the outer register.

        """
        prepare_algorithm = self._create_nested("outer_prepare")
        ref: AlgorithmRef = self._settings.get("outer_prepare")
        if ref.algorithm_name == "alias_sampling":
            prepare_algorithm.settings().set("bits_precision", self._settings.get("coefficient_bit_precision"))
        elif ref.algorithm_name == "qrom":
            # The walk allocates one persistent gradient and hands it to every PREPARE and
            # SELECT, so the outer PREPARE must read that one rather than allocate its own.
            # `_num_phase_gradient_qubits` sizes the register from this algorithm's
            # `rotation_bit_precision`, and `_build_walk_oracles` checks the two agree.
            prepare_algorithm.settings().set("allocate_phase_gradient", False)
        circuit = prepare_algorithm.run(container.outer_prepare)
        return circuit._qsharp_op, circuit.metadata.num_phase_gradient_ancillas  # noqa: SLF001

    def _hoist_free_rider(self, container: SOSSAWalkContainer) -> bool:
        r"""Whether the free-rider word is loaded once per block encoding rather than per PREPARE.

        Carrying it in the alias table widens the QROAM output word, which the swap network is
        charged for on each of the four lookups a block encoding performs; hoisting replaces
        that with one ``Select`` round trip over the conditions. Which wins depends on the
        shape, so the comparison is left to ``SeparateWordLoadPays``.

        Args:
            container: The SOSSA walk container describing the block encoding.

        Returns:
            True when a separate load is the cheaper of the two.

        """
        if self._settings.get("inner_prepare_algorithm") != "controlled_alias_sampling":
            return True
        free_rider = container.inner_prepare.free_rider_data
        if free_rider is None or free_rider.size == 0:
            return False
        layout = container.layout
        mu = int(self._settings.get("coefficient_bit_precision"))
        return QSHARP_UTILS.SelectSwap.SeparateWordLoadPays(
            container.inner_prepare.conditional_coefficients.shape[0],
            1 << layout.inner_prep_bits,
            mu + layout.inner_prep_bits + 2,
            layout.num_free_rider_bits,
            _INNER_LOOKUPS_PER_BLOCK_ENCODING,
        )

    def _build_inner_prep(self, container: SOSSAWalkContainer) -> Any:
        r"""Build the Q# inner (controlled) PREPARE callable.

        Creates a superposition over bases :math:`b` conditioned on :math:`x_o`. The
        free-rider word is loaded here only when :meth:`_hoist_free_rider` says otherwise.

        Algorithms:
            - ``"controlled_alias_sampling"``: 2D alias sampling.
            - ``"direct"``: Direct multiplexed preparation (ControlledPureStatePrep).

        Args:
            container: The SOSSA container with inner_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[], Qubit[]) => Unit is Adj``.

        """
        algorithm = self._settings.get("inner_prepare_algorithm")
        coeff_bits = self._settings.get("coefficient_bit_precision")
        coefficients = container.inner_prepare.conditional_coefficients.tolist()
        free_rider_data = container.inner_prepare.free_rider_data
        free_rider_data = free_rider_data.tolist() if free_rider_data is not None else []

        if algorithm == "controlled_alias_sampling":
            inline = [] if self._hoist_free_rider(container) else free_rider_data
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareAliasSampling(coefficients, inline, coeff_bits)
        return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients, free_rider_data)

    def _build_free_rider_load(self, container: SOSSAWalkContainer) -> Any:
        r"""Build the Q# callable that loads the free-rider word :math:`(G, r)` for one :math:`x_o`.

        Args:
            container: The SOSSA container carrying the free-rider table.

        Returns:
            A Q# callable ``(Qubit[], Qubit[]) => Unit is Adj + Ctl``, a no-op when the inner
            PREPARE carries the word itself.

        """
        if not self._hoist_free_rider(container):
            return QSHARP_UTILS.SOSSAWalk.MakeFreeRiderLoadOp([])
        free_rider_data = container.inner_prepare.free_rider_data
        free_rider_data = free_rider_data.tolist() if free_rider_data is not None else []
        return QSHARP_UTILS.SOSSAWalk.MakeFreeRiderLoadOp(free_rider_data)

    def _build_select(self, container: SOSSAWalkContainer) -> Any:
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
        }
        if algorithm == "qrom_phase_gradient":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectPhaseGradient(select_data)
        return QSHARP_UTILS.SOSSAWalk.MakeSelectDirectRotation(select_data)

    @property
    def _num_phase_gradient_qubits(self) -> int:
        """Width of the persistent phase gradient register, zero when none is needed.

        Raises:
            ValueError: If the outer PREPARE and SELECT ask for different widths.

        """
        outer_ref: AlgorithmRef = self._settings.get("outer_prepare")
        outer_bits = (
            int(self._create_nested("outer_prepare").settings().get("rotation_bit_precision"))
            if outer_ref.algorithm_name == "qrom"
            else 0
        )
        select_bits = (
            int(self._settings.get("rotation_bit_precision"))
            if self._settings.get("select_algorithm") == "qrom_phase_gradient"
            else 0
        )
        if outer_bits and select_bits and outer_bits != select_bits:
            raise ValueError(
                "The outer PREPARE and SELECT share one phase gradient register and must agree "
                f"on its width, but the outer PREPARE asks for {outer_bits} qubits and SELECT for "
                f"{select_bits}. Set the same rotation_bit_precision on both."
            )
        return max(outer_bits, select_bits)

    def _compute_register_sizes(self, container: SOSSAWalkContainer) -> tuple[dict[str, int], Any]:
        """Compute the register widths and the Q# ``SOSSAWalkLayout`` describing them.

        Args:
            container: The SOSSA walk container describing the block encoding.

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

        outer_ref: AlgorithmRef = self._settings.get("outer_prepare")
        if outer_ref.algorithm_name == "alias_sampling":
            mu_outer = self._settings.get("coefficient_bit_precision")
            num_outer_qubits = 2 * outer_prep_bits + 2 * mu_outer + 1
        else:
            num_outer_qubits = outer_prep_bits

        if self._settings.get("inner_prepare_algorithm") == "controlled_alias_sampling":
            mu_inner = self._settings.get("coefficient_bit_precision")
            num_inner_qubits = 2 * inner_prep_bits + 2 * mu_inner + 3 + num_free_rider_bits
            num_reflect_inner = inner_prep_bits + mu_inner + 1
        else:
            num_inner_qubits = inner_prep_bits + num_free_rider_bits
            num_reflect_inner = inner_prep_bits

        num_phase_gradient_qubits = self._num_phase_gradient_qubits
        num_spin_qubits = 2  # spinDQ + spinSF, matches Q# SOSSAWalk.qs

        regs = {
            "num_system_qubits": num_system_qubits,
            "num_outer_qubits": num_outer_qubits,
            "num_outer_index_qubits": outer_prep_bits,
            "num_inner_qubits": num_inner_qubits,
            "num_reflect_inner": num_reflect_inner,
            "num_phase_gradient_qubits": num_phase_gradient_qubits,
            "num_outer_prepare_gradient_qubits": (
                num_phase_gradient_qubits if outer_ref.algorithm_name == "qrom" else 0
            ),
            "num_ancilla_qubits": num_outer_qubits + num_reflect_inner + num_spin_qubits + num_phase_gradient_qubits,
        }
        walk_layout = QSHARP_UTILS.SOSSAWalk.SOSSAWalkLayout(
            numSystemQubits=regs["num_system_qubits"],
            numOuterQubits=regs["num_outer_qubits"],
            numOuterIndexQubits=regs["num_outer_index_qubits"],
            numInnerQubits=regs["num_inner_qubits"],
            numReflectInner=regs["num_reflect_inner"],
            numPhaseGradientQubits=regs["num_phase_gradient_qubits"],
            numOuterPrepareGradientQubits=regs["num_outer_prepare_gradient_qubits"],
            numFreeRiderQubits=num_free_rider_bits,
        )
        return regs, walk_layout

    def _build_walk_oracles(self, container: SOSSAWalkContainer, regs: dict[str, int]) -> tuple[Any, Any, Any, Any]:
        """Build the outer PREPARE, free-rider load, inner PREPARE and SELECT of the block encoding.

        Raises:
            ValueError: If the outer PREPARE circuit declares a different phase gradient
                width than the walk layout reserves for it.

        """
        outer_prepare_op, outer_gradient_qubits = self._build_outer_prep(container)
        reserved = regs["num_outer_prepare_gradient_qubits"]
        if outer_gradient_qubits != reserved:
            raise ValueError(
                f"The outer PREPARE circuit declares {outer_gradient_qubits} phase gradient ancillas "
                f"but the walk layout reserves {reserved} for it, so the register handed to it would "
                "be the wrong width."
            )
        return (
            outer_prepare_op,
            self._build_free_rider_load(container),
            self._build_inner_prep(container),
            self._build_select(container),
        )

    def _num_ancilla_qubits(self, container: SOSSAWalkContainer) -> int:
        """Ancilla qubits the block encoding acts on, past the system register.

        Args:
            container: The SOSSA walk container describing the block encoding.

        Returns:
            The width of the ancilla part of the ``[system | ancilla]`` register.

        """
        regs, _ = self._compute_register_sizes(container)
        return regs["num_ancilla_qubits"]

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct the SOSSA block encoding on the flat ``[system | ancilla]`` register.

        Args:
            unitary: The unitary representation containing the SOSSA decomposition.

        Returns:
            Circuit: The block encoding :math:`B`, declaring the full register width and the
            phase gradient qubits its caller must prepare.

        Raises:
            ValueError: If the container is not a :class:`SOSSAWalkContainer`.

        """
        container = unitary.get_container()
        if not isinstance(container, SOSSAWalkContainer):
            raise ValueError(f"The {unitary.get_container_type()} container type is not supported.")
        if container.power != 1:
            Logger.warn(f"The container's walk power {container.power} is ignored.")

        regs, walk_layout = self._compute_register_sizes(container)
        outer_prepare_op, free_rider_op, inner_prepare_op, select_op = self._build_walk_oracles(container, regs)

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.SOSSAWalk.MakeSOSSABlockEncodingCircuit,
            parameter={
                "outerPrepareOp": outer_prepare_op,
                "freeRiderOp": free_rider_op,
                "innerPrepareOp": inner_prepare_op,
                "selectOp": select_op,
                "layout": walk_layout,
            },
        )
        qsharp_op = QSHARP_UTILS.SOSSAWalk.MakeSOSSABlockEncodingOp(
            outer_prepare_op,
            free_rider_op,
            inner_prepare_op,
            select_op,
            walk_layout,
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            num_qubits=regs["num_system_qubits"] + regs["num_ancilla_qubits"],
            metadata=CircuitMetadata(num_phase_gradient_ancillas=regs["num_phase_gradient_qubits"]),
        )
