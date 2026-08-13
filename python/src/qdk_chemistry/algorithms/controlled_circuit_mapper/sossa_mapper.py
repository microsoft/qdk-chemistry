"""QDK/Chemistry SOSSA (Sum of Squares Spectral Amplification) controlled circuit mapper :cite:`Low2025`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

from qdk_chemistry.data import AlgorithmRef
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAWalkContainer, sossa_register_bits
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = [
    "SOSSAMapper",
    "SOSSAMapperSettings",
]


class SOSSAMapperSettings(ControlledCircuitMapperSettings):
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


class SOSSAMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper for the SOSSA walk operator."""

    def __init__(self):
        """Initialize the SOSSAMapper."""
        super().__init__()
        self._settings = SOSSAMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "controlled_circuit_mapper"

    def build_outer_prep(self, container: SOSSAWalkContainer) -> Any:
        r"""Build the Q# outer PREPARE callable.

        The outer PREPARE must produce amplitudes proportional to the generator
        one-norms :math:`c_{x_o}` themselves, not to their square roots, because the
        SOS block encoding reads off :math:`\sum_{x_o} c_{x_o}^2 = 2\Lambda` from the
        amplitudes (Eq. 88 of :cite:`Low2025`).

        Args:
            container: The SOSSA container with outer_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[]) => Unit is Adj + Ctl``.

        """
        ref: AlgorithmRef = self._settings.get("outer_prepare")
        if ref.algorithm_name == "dense_pure_state":
            # Use MakeOuterPreparePureState directly to avoid the endianness
            # mismatch
            coeffs = [float(c) for c in container.outer_prepare.get_coefficients()]
            n_qubits = self._compute_register_sizes(container)["num_outer_qubits"]
            n_padded = 1 << n_qubits
            padded = coeffs + [0.0] * (n_padded - len(coeffs))
            return QSHARP_UTILS.SOSSAWalk.MakeOuterPreparePureState(padded)
        prepare_algorithm = self._create_nested("outer_prepare")
        outer_prepare = container.outer_prepare
        if ref.algorithm_name == "alias_sampling":
            # Keep the op's precision in sync with the outer register size (see _compute_register_sizes).
            prepare_algorithm.bits_precision = self._settings.get("coefficient_bit_precision")
            # One-dimensional alias sampling discretizes its input as a probability
            # distribution, so it has to be handed the squared coefficients the
            # builder precomputed. The 2D conditional table used by the inner
            # PREPARE squares its input itself.
            outer_prepare = container.outer_prepare_probabilities
        circuit = prepare_algorithm.run(outer_prepare)
        return circuit._qsharp_op  # noqa: SLF001

    def build_inner_prep(self, container: SOSSAWalkContainer) -> Any:
        r"""Build the Q# inner (controlled) PREPARE callable.

        Creates a superposition over bases :math:`b` conditioned on :math:`x_o`.

        Algorithms:
            - ``"controlled_alias_sampling"``: 2D alias sampling with free-rider data.
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
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareAliasSampling(coefficients, free_rider_data, coeff_bits)
        return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients, free_rider_data)

    def build_select(self, container: SOSSAWalkContainer) -> Any:
        r"""Build the SELECT step.

        Algorithms:
            - ``"qrom_phase_gradient"``: Load angles via QROM, apply via phase gradient adders.
            - ``"direct"``: Direct rotation synthesis.

        Args:
            container: The SOSSA container with rotation angles and structure.

        Returns:
            A Q# callable for the SELECT oracle.

        """
        algorithm = self._settings.get("select_algorithm")
        rot_bits = self._settings.get("rotation_bit_precision")

        num_free_rider_bits = sossa_register_bits(
            container.num_orbitals, container.num_ranks, container.num_bases, container.num_copies
        )["num_free_rider_bits"]

        select_data = {
            "numOrbitals": container.num_orbitals,
            "numRanks": container.num_ranks,
            "numBases": container.num_bases,
            "numCopies": container.num_copies,
            "numPositiveOneBody": container.select.num_positive_one_body_terms,
            "OneBodyRotationAngles": container.select.one_body_rotation_angles.tolist(),
            "TwoBodyRotationAngles": container.select.two_body_rotation_angles.tolist(),
            "rotationBitPrecision": rot_bits,
            "numFreeRiderBits": num_free_rider_bits,
        }
        if algorithm == "qrom_phase_gradient":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectPhaseGradient(select_data)
        return QSHARP_UTILS.SOSSAWalk.MakeSelectDirectRotation(select_data)

    @property
    def uses_phase_gradient(self) -> bool:
        """Whether a persistent phase gradient register must be allocated."""
        return self._settings.get("select_algorithm") == "qrom_phase_gradient"

    def _compute_register_sizes(self, container: SOSSAWalkContainer) -> dict[str, int]:
        """Compute register sizes from container structure and settings."""
        num_orbitals = container.num_orbitals
        num_system_qubits = 2 * num_orbitals
        reg_bits = sossa_register_bits(num_orbitals, container.num_ranks, container.num_bases, container.num_copies)
        xo_bits = reg_bits["xo_bits"]
        b_bits = reg_bits["b_bits"]
        num_free_rider_bits = reg_bits["num_free_rider_bits"]

        outer_ref: AlgorithmRef = self._settings.get("outer_prepare")
        if outer_ref.algorithm_name == "alias_sampling":
            mu_outer = self._settings.get("coefficient_bit_precision")
            num_outer_qubits = 2 * xo_bits + 2 * mu_outer + 1
        else:
            num_outer_qubits = xo_bits

        if self._settings.get("inner_prepare_algorithm") == "controlled_alias_sampling":
            mu_inner = self._settings.get("coefficient_bit_precision")
            num_inner_qubits = 2 * b_bits + 2 * mu_inner + 3 + num_free_rider_bits
            num_reflect_inner = b_bits + mu_inner + 1
        else:
            num_inner_qubits = b_bits + num_free_rider_bits
            num_reflect_inner = b_bits

        num_phase_gradient_qubits = self._settings.get("rotation_bit_precision") if self.uses_phase_gradient else 0

        return {
            "num_system_qubits": num_system_qubits,
            "num_outer_qubits": num_outer_qubits,
            "num_outer_index_qubits": xo_bits,
            "num_inner_qubits": num_inner_qubits,
            "num_reflect_inner": num_reflect_inner,
            "num_phase_gradient_qubits": num_phase_gradient_qubits,
        }

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a controlled SOSSA walk step circuit.

        Args:
            unitary: The unitary representation containing the SOSSA decomposition.

        Returns:
            Circuit: A quantum circuit implementing the controlled SOSSA walk step.

        """
        unitary_container = unitary.get_container()
        if not isinstance(unitary_container, SOSSAWalkContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported. "
                "SOSSAMapper only supports SOSSAWalkContainer."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("SOSSAMapper only supports a single control qubit.")

        power = unitary_container.power

        outer_prepare_op, inner_prepare_op, select_op = self._build_walk_oracles(unitary_container)
        layout = self._build_walk_layout(unitary_container)

        walk_params = {
            "outerPrepareOp": outer_prepare_op,
            "innerPrepareOp": inner_prepare_op,
            "selectOp": select_op,
            "layout": layout,
            "power": power,
        }

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkCircuit,
            parameter=walk_params,
        )
        qsharp_op = QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkOp(
            outer_prepare_op,
            inner_prepare_op,
            select_op,
            layout,
            power,
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

    def _build_walk_oracles(self, container: SOSSAWalkContainer) -> tuple[Any, Any, Any]:
        """Build the outer PREPARE, inner PREPARE and SELECT callables of the block encoding."""
        return (
            self.build_outer_prep(container),
            self.build_inner_prep(container),
            self.build_select(container),
        )

    def _build_walk_layout(self, container: SOSSAWalkContainer) -> Any:
        """Build the Q# ``SOSSAWalkLayout`` describing the register sizes of the walk."""
        regs = self._compute_register_sizes(container)
        return QSHARP_UTILS.SOSSAWalk.SOSSAWalkLayout(
            numSystemQubits=regs["num_system_qubits"],
            numOuterQubits=regs["num_outer_qubits"],
            numOuterIndexQubits=regs["num_outer_index_qubits"],
            numInnerQubits=regs["num_inner_qubits"],
            numReflectInner=regs["num_reflect_inner"],
            numPhaseGradientQubits=regs["num_phase_gradient_qubits"],
        )

    def num_ancilla_qubits(self, container: SOSSAWalkContainer) -> int:
        """The number of ancilla qubits used by external algorithms like phase estimation."""
        regs = self._compute_register_sizes(container)
        num_spin_qubits = 2  # spinDQ + spinSF, matches Q# SOSSAWalk.qs
        return regs["num_outer_qubits"] + regs["num_inner_qubits"] + num_spin_qubits + regs["num_phase_gradient_qubits"]

    def build_walk_op(
        self,
        unitary: UnitaryRepresentation,
        num_queries: int,
        use_unary_iteration: bool = True,
    ) -> Any:
        """Build a SOSSA walk callable acting on (control register, system + ancilla register).

        When ``use_unary_iteration`` is ``True`` the control register is the phase register.
        Unary iteration skips one outer reflection per address, so branch ``t`` applies
        ``W^(num_queries - 2t)``. Otherwise the control register holds a single qubit and the
        controlled walk step is repeated ``num_queries`` times.

        Args:
            unitary: The unitary representation containing the SOSSA decomposition.
            num_queries: Number of SOSSA involution blocks to apply.
            use_unary_iteration: Whether the control register is a phase register iterated over.

        Returns:
            A Q# callable accepting the control register and the combined system/ancilla register.

        Raises:
            TypeError: If the unitary does not contain a SOSSA walk.
            ValueError: If ``num_queries`` is not positive.

        """
        container = unitary.get_container()
        if not isinstance(container, SOSSAWalkContainer):
            raise TypeError("A SOSSA walk callable requires a SOSSAWalkContainer")
        if num_queries <= 0:
            raise ValueError(f"num_queries must be a positive integer. Got {num_queries}.")

        outer_prepare_op, inner_prepare_op, select_op = self._build_walk_oracles(container)

        return QSHARP_UTILS.SOSSAWalk.MakeSOSSAWalkOp(
            outer_prepare_op,
            inner_prepare_op,
            select_op,
            self._build_walk_layout(container),
            num_queries,
            use_unary_iteration,
        )

    def get_ancilla_prep_op(self) -> Any:
        """Return the Q# ancilla preparation op for external algorithms like phase estimation.

        Returns:
            A Q# callable ``Qubit[] => Unit is Adj`` that prepares the phase gradient state
            on the block-encoding ancillas, or a no-op if phase gradient is not needed.

        """
        if self.uses_phase_gradient:
            rot_bits = self._settings.get("rotation_bit_precision")
            return QSHARP_UTILS.PhaseGradient.MakePhaseGradientAncillaPrep(rot_bits)
        return QSHARP_UTILS.StatePreparation.MakeNoOpAncillaPrep()
