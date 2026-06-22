"""QDK/Chemistry SOSSA (Sum of Squares with Ancilla) controlled circuit mapper.

The SOSSAMapper composes the full controlled walk operator from three
sub-operations (outer PREPARE, inner PREPARE, SELECT), each built as a
method on the mapper itself. Configuration is via Settings, matching the
pattern used by PrepSelPrepMapper.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import ceil, log2
from typing import Any

from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

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
            "outer_prepare",
            "algorithm_ref",
            AlgorithmRef("state_prep", "alias_sampling"),
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


class SOSSAMapper(ControlledCircuitMapper):
    r"""Controlled circuit mapper for the SOSSA walk operator.

    Composes a controlled SOSSA walk step from three sub-operations built
    directly as methods on this class:

    1. :meth:`build_outer_prep` — outer PREPARE (amplitude-loading into
       :math:`x_o` register), resolved via an ``AlgorithmRef`` state_prep setting.
    2. :meth:`build_inner_prep` — inner (controlled) PREPARE over bases.
    3. :meth:`build_select` — SELECT (multiplexed Givens rotations + Majorana).

    The walk operator:

    .. math::

        W = \mathrm{Ref}_{a,B} \cdot U^\dagger \cdot \mathrm{Ref}_B \cdot U

    Configuration:
        - ``outer_prepare``: AlgorithmRef for state preparation (like LCU).
          Supports ``"alias_sampling"``, ``"dense_pure_state"``, ``"qrom_state_prep"``.
        - ``inner_prepare_algorithm``: ``"controlled_alias_sampling"`` or ``"direct"``.
        - ``select_algorithm``: ``"qrom_phase_gradient"`` or ``"direct"``.
        - ``rotation_bit_precision``: bits for Givens angle precision (b_rot).
        - ``coefficient_bit_precision``: bits for alias sampling coefficients.

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
        return "controlled_circuit_mapper"

    # ═══════════════════════════════════════════════════════════════════════════
    # Sub-operation builders
    # ═══════════════════════════════════════════════════════════════════════════

    def build_outer_prep(self, container: SOSSAContainer) -> Any:
        r"""Build the Q# outer PREPARE callable.

        Resolves the ``outer_prepare`` AlgorithmRef setting to a state
        preparation algorithm, runs it on the container's outer_prepare
        wavefunction, and returns the Q# callable.

        Args:
            container: The SOSSA container with outer_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[]) => Unit is Adj + Ctl``.

        """
        prepare_algorithm = self._create_nested("outer_prepare")
        circuit = prepare_algorithm.run(container.outer_prepare)
        return circuit._qsharp_op  # noqa: SLF001

    @property
    def outer_prepare_needs_alias_reflection(self) -> bool:
        """Whether the outer prepare uses alias sampling (needs keep/mu reflection)."""
        ref: AlgorithmRef = self._settings.get("outer_prepare")
        return ref.algorithm_name == "alias_sampling"

    @property
    def outer_prepare_coefficient_bits(self) -> int:
        """Get coefficient bit precision for outer prepare."""
        return self._settings.get("coefficient_bit_precision")

    def build_inner_prep(self, container: SOSSAContainer) -> Any:
        r"""Build the Q# inner (controlled) PREPARE callable.

        Creates a superposition over bases :math:`b` conditioned on :math:`x_o`.

        Algorithms:
            - ``"controlled_alias_sampling"``: 2D alias sampling with free-rider data.
            - ``"direct"``: Direct coherent preparation (ControlledPureStatePrep).

        Args:
            container: The SOSSA container with inner_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[], Qubit[]) => Unit is Adj``.

        """
        algorithm = self._settings.get("inner_prepare_algorithm")
        coeff_bits = self._settings.get("coefficient_bit_precision")
        coefficients = container.inner_prepare.conditional_coefficients.tolist()
        fr = container.inner_prepare.free_rider_data
        fr_data = fr.tolist() if fr is not None else []

        if algorithm == "controlled_alias_sampling":
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareAliasSampling(coefficients, fr_data, coeff_bits)
        return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients, fr_data)

    @property
    def inner_prepare_needs_alias_reflection(self) -> bool:
        """Whether the inner prepare uses alias sampling (needs keep reflection)."""
        return self._settings.get("inner_prepare_algorithm") == "controlled_alias_sampling"

    def build_select(self, container: SOSSAContainer) -> Any:
        r"""Build the Q# SELECT callable (multiplexed Givens rotations).

        Algorithms:
            - ``"qrom_phase_gradient"``: Load angles via QROM, apply via phase
              gradient adders. (Paper Tables A-D)
            - ``"direct"``: Direct rotation synthesis. (Paper Table E)

        Args:
            container: The SOSSA container with rotation angles and structure.

        Returns:
            A Q# callable for the SELECT oracle.

        """
        algorithm = self._settings.get("select_algorithm")
        rot_bits = self._settings.get("rotation_bit_precision")

        R = container.select.num_ranks
        rank_bits = ceil(log2(R)) if R > 1 else 0
        num_free_rider_bits = 2 + rank_bits

        select_data = {
            "numOrbitals": container.select.num_orbitals,
            "numRanks": container.select.num_ranks,
            "numBases": container.select.num_bases,
            "numCopies": container.select.num_copies,
            "numD1": container.select.num_d1,
            "dqRotationAngles": container.select.rotation_angles.tolist(),
            "sfRotationAngles": container.select.sf_rotation_angles.tolist(),
            "rotationBitPrecision": rot_bits,
            "numFreeRiderBits": num_free_rider_bits,
        }
        if algorithm == "qrom_phase_gradient":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectPhaseGradient(select_data)
        return QSHARP_UTILS.SOSSAWalk.MakeSelectDirectRotation(select_data)

    @property
    def select_needs_phase_gradient(self) -> bool:
        """Whether a persistent phase gradient register must be allocated."""
        return self._settings.get("select_algorithm") == "qrom_phase_gradient"

    # ═══════════════════════════════════════════════════════════════════════════
    # Register size computation
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_register_sizes(self, container: SOSSAContainer) -> dict:
        """Compute register sizes from container structure and settings."""
        num_orbitals = container.select.num_orbitals
        num_system_qubits = 2 * num_orbitals
        x_o_dim = num_orbitals + container.select.num_ranks * container.select.num_copies
        xo_bits = ceil(log2(x_o_dim)) if x_o_dim > 1 else 1
        num_bases = container.select.num_bases
        b_bits = ceil(log2(num_bases + 1)) if num_bases + 1 > 1 else 1
        R = container.select.num_ranks
        rank_bits = ceil(log2(R)) if R > 1 else 0
        num_free_rider_bits = 2 + rank_bits

        if self.outer_prepare_needs_alias_reflection:
            mu_outer = self.outer_prepare_coefficient_bits
            num_outer_qubits = 2 * xo_bits + 2 * mu_outer + 1
        else:
            num_outer_qubits = xo_bits

        if self.inner_prepare_needs_alias_reflection:
            mu_inner = self._settings.get("coefficient_bit_precision")
            num_inner_qubits = 2 * b_bits + 2 * mu_inner + 3 + num_free_rider_bits
            num_reflect_inner = b_bits + mu_inner + 1
        else:
            num_inner_qubits = b_bits + num_free_rider_bits
            num_reflect_inner = b_bits

        # Phase gradient register: allocated by QPE, used by QROM-based SELECT
        num_phase_gradient_qubits = (
            self._settings.get("rotation_bit_precision") if self.select_needs_phase_gradient else 0
        )

        return {
            "num_system_qubits": num_system_qubits,
            "num_outer_qubits": num_outer_qubits,
            "num_inner_qubits": num_inner_qubits,
            "num_reflect_inner": num_reflect_inner,
            "num_phase_gradient_qubits": num_phase_gradient_qubits,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Circuit construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_impl(self, controlled_unitary: ControlledUnitary) -> Circuit:
        r"""Construct a controlled SOSSA walk step circuit.

        Args:
            controlled_unitary: The controlled unitary containing the SOSSA
                decomposition (outer/inner PREPARE and SELECT data).

        Returns:
            Circuit: A quantum circuit implementing the controlled SOSSA walk step.

        """
        unitary_container = controlled_unitary.unitary.get_container()
        if not isinstance(unitary_container, SOSSAContainer):
            raise ValueError(
                f"The {controlled_unitary.get_unitary_container_type()} container type is not supported. "
                "SOSSAMapper only supports SOSSAContainer."
            )

        if len(controlled_unitary.control_indices) != 1:
            raise ValueError("SOSSAMapper currently only supports a single control qubit.")

        power = unitary_container.power

        outer_prepare_op = self.build_outer_prep(unitary_container)
        inner_prepare_op = self.build_inner_prep(unitary_container)
        select_op = self.build_select(unitary_container)

        regs = self._compute_register_sizes(unitary_container)

        walk_params = {
            "outerPrepareOp": outer_prepare_op,
            "innerPrepareOp": inner_prepare_op,
            "selectOp": select_op,
            "numSystemQubits": regs["num_system_qubits"],
            "numOuterQubits": regs["num_outer_qubits"],
            "numInnerQubits": regs["num_inner_qubits"],
            "numReflectInner": regs["num_reflect_inner"],
            "numPhaseGradientQubits": regs["num_phase_gradient_qubits"],
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
            regs["num_system_qubits"],
            regs["num_outer_qubits"],
            regs["num_inner_qubits"],
            regs["num_reflect_inner"],
            regs["num_phase_gradient_qubits"],
            power,
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

    def num_ancillary_qubits(self, container: SOSSAContainer) -> int:
        """Calculate the number of ancillary qubits required for the SOSSA circuit.

        Args:
            container: The SOSSA container with decomposition data.

        Returns:
            The number of ancillary qubits required.

        """
        regs = self._compute_register_sizes(container)
        return (
            regs["num_outer_qubits"]
            + regs["num_inner_qubits"]
            + regs["num_reflect_inner"]
            + regs["num_phase_gradient_qubits"]
        )

    def get_ancilla_prep_op(self) -> Any:
        """Return the Q# ancilla preparation op for SOSSA (phase gradient init).

        Returns:
            A Q# callable ``Qubit[] => Unit is Adj`` that prepares the phase gradient state
            on the block-encoding ancillas, or a no-op if phase gradient is not needed.

        """
        if self.select_needs_phase_gradient:
            rot_bits = self._settings.get("rotation_bit_precision")
            return QSHARP_UTILS.SOSSAWalk.MakePhaseGradientAncillaPrep(rot_bits)
        return QSHARP_UTILS.SOSSAWalk.MakeNoOpAncillaPrep()

    def build_estimate_circuit(
        self,
        container: SOSSAContainer,
        num_queries: int,
    ) -> Circuit:
        """Build a resource-estimation circuit using RepeatEstimates.

        Instead of tracing through each walk step individually (expensive
        for large query counts), this evaluates a single controlled walk step
        and multiplies the cost by ``num_queries`` via ``RepeatEstimates``.

        Args:
            container: The SOSSA container with decomposition data.
            num_queries: Total number of walk operator queries
                (e.g. ceil(pi * lambda / (2 * sigma)) for Heisenberg-limited QPE).

        Returns:
            Circuit with a ``qsharp_factory`` targeting
            ``EstimateSOSSAWalkCircuit``.

        """
        outer_prepare_op = self.build_outer_prep(container)
        inner_prepare_op = self.build_inner_prep(container)
        select_op = self.build_select(container)

        regs = self._compute_register_sizes(container)

        estimate_params = {
            "outerPrepareOp": outer_prepare_op,
            "innerPrepareOp": inner_prepare_op,
            "selectOp": select_op,
            "numSystemQubits": regs["num_system_qubits"],
            "numOuterQubits": regs["num_outer_qubits"],
            "numInnerQubits": regs["num_inner_qubits"],
            "numReflectInner": regs["num_reflect_inner"],
            "numPhaseGradientQubits": regs["num_phase_gradient_qubits"],
            "numQueries": num_queries,
        }

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.SOSSAWalk.EstimateSOSSAWalkCircuit,
                parameter=estimate_params,
            )
        )
