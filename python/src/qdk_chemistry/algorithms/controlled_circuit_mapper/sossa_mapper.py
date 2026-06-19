"""QDK/Chemistry SOSSA (Sum of Squares with Ancilla) controlled circuit mapper.

Each sub-operation mapper independently produces a Q# callable for its
sub-circuit (outer prepare, inner prepare, select). The walk-step mapper
composes them into the full controlled walk operator, following the same
pattern as PrepSelPrepMapper.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from math import ceil, log2
from typing import Any

import numpy as np

from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.algorithms.state_preparation.dense_pure_state import DensePureStatePreparation
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = [
    "InnerPrepareMapper",
    "OuterPrepareMapper",
    "SOSSAMapper",
    "SOSSAMapperSettings",
    "SelectMapper",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-operation mappers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OuterPrepareMapper:
    r"""Maps the outer PREPARE oracle to a Q# callable.

    The outer PREPARE loads amplitudes into the :math:`x_o` register.
    Each algorithm choice produces a *different Q# op* for the sub-operation.

    Algorithms:
        - ``"alias_sampling"``: Alias sampling with keep/mu registers.
          Requires reflection over keep register. (Paper Tables A-B)
        - ``"dense_pure"``: Coherent pure-state preparation (PreparePureStateD).
          No extra reflection registers needed. (Paper Tables C-E)
        - ``"qrom"``: Direct QROM amplitude loading.

    """

    algorithm: str = "alias_sampling"
    """State preparation algorithm name."""

    coefficient_bit_precision: int = 10
    """Bit precision for alias sampling coefficients (mu register width)."""

    def __post_init__(self):
        """Validate algorithm name."""
        valid = {"alias_sampling", "dense_pure", "qrom"}
        if self.algorithm not in valid:
            raise ValueError(
                f"Unknown outer prepare algorithm '{self.algorithm}'. "
                f"Must be one of: {sorted(valid)}"
            )

    @property
    def needs_alias_reflection(self) -> bool:
        """Whether alias sampling keep/mu registers participate in reflection."""
        return self.algorithm == "alias_sampling"

    def build_op(self, container: SOSSAContainer) -> Any:
        """Build the Q# outer prepare callable from container data.

        Delegates to the corresponding state preparation algorithm
        (alias_sampling, dense_pure_state, or qrom_state_prep) and
        returns the Q# callable from the resulting circuit.

        Args:
            container: The SOSSA container with outer_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[]) => Unit is Adj + Ctl`` for outer prepare.

        """
        if self.algorithm == "dense_pure":
            prep = DensePureStatePreparation()
            circuit = prep.run(container.outer_prepare)
        else:
            statevector = np.asarray(container.outer_prepare.get_coefficients())
            num_qubits = ceil(log2(len(statevector))) if len(statevector) > 1 else 1
            qubit_indices = list(range(num_qubits))
            if self.algorithm == "alias_sampling":
                prep = AliasSamplingStatePreparation(bits_precision=self.coefficient_bit_precision)
            else:
                prep = QROMStatePreparation(rotation_bit_precision=self.coefficient_bit_precision)
            circuit = prep.prepare_from_statevector(statevector, num_qubits, qubit_indices)
        return circuit._qsharp_op


@dataclass(frozen=True)
class InnerPrepareMapper:
    r"""Maps the inner (controlled) PREPARE oracle to a Q# callable.

    The inner PREPARE creates a superposition over bases :math:`b`
    conditioned on :math:`x_o`. Each algorithm choice gives a different Q# op.

    Algorithms:
        - ``"controlled_alias_sampling"``: 2D alias sampling with free-rider data.
          Requires keep register in inner reflection. (Paper Tables A-B)
        - ``"direct"``: Direct coherent preparation (ControlledPureStatePrep).
          No extra keep register needed. (Paper Tables C-E)

    """

    algorithm: str = "controlled_alias_sampling"
    """Controlled state preparation algorithm name."""

    coefficient_bit_precision: int = 10
    """Bit precision for inner alias sampling coefficients."""

    def __post_init__(self):
        """Validate algorithm name."""
        valid = {"controlled_alias_sampling", "direct"}
        if self.algorithm not in valid:
            raise ValueError(
                f"Unknown inner prepare algorithm '{self.algorithm}'. "
                f"Must be one of: {sorted(valid)}"
            )

    @property
    def needs_alias_reflection(self) -> bool:
        """Whether inner alias keep register participates in reflection."""
        return self.algorithm == "controlled_alias_sampling"

    def build_op(self, container: SOSSAContainer) -> Any:
        """Build the Q# inner prepare callable from container data.

        Args:
            container: The SOSSA container with inner_prepare coefficients.

        Returns:
            A Q# callable ``(Qubit[], Qubit[]) => Unit is Adj``
            for inner prepare (takes outer register and inner register).

        """
        coefficients = container.inner_prepare.conditional_coefficients.tolist()
        # TODO: add a setting for non-free-rider version where we compute G, R on the fly instead of storing them
        if self.algorithm == "controlled_alias_sampling":
            fr = container.inner_prepare.free_rider_data
            fr_data = fr.tolist() if fr is not None else []
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareAliasSampling(
                coefficients, fr_data, self.coefficient_bit_precision
            )
        # direct
        return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients)


@dataclass(frozen=True)
class SelectMapper:
    r"""Maps the SELECT oracle (multiplexed rotations) to a Q# callable.

    The SELECT applies Givens rotations controlled on :math:`(x_o, b)`.
    Each algorithm choice gives a different Q# op.

    Algorithms:
        - ``"qrom_phase_gradient"``: Load angles via QROM, apply via phase gradient
          adders. Requires a persistent phase gradient register. (Paper Tables A-D)
        - ``"direct"``: Direct rotation synthesis (no phase gradient register).
          Higher Toffoli cost per rotation but fewer qubits. (Paper Table E)

    """

    multiplexed_rotation: str = "qrom_phase_gradient"
    """Multiplexed rotation algorithm name."""

    rotation_bit_precision: int = 10
    """Number of bits for Givens rotation angle precision (b_rot)."""

    def __post_init__(self):
        """Validate algorithm name."""
        valid = {"qrom_phase_gradient", "direct"}
        if self.multiplexed_rotation not in valid:
            raise ValueError(
                f"Unknown multiplexed rotation '{self.multiplexed_rotation}'. "
                f"Must be one of: {sorted(valid)}"
            )

    @property
    def needs_phase_gradient_register(self) -> bool:
        """Whether a persistent phase gradient register must be allocated."""
        return self.multiplexed_rotation == "qrom_phase_gradient"

    def build_op(self, container: SOSSAContainer) -> Any:
        """Build the Q# select callable from container data.

        Args:
            container: The SOSSA container with rotation angles and structure.

        Returns:
            A Q# callable for the SELECT oracle (Givens rotations + Majorana).

        """
        select_data = {
            "numOrbitals": container.select.num_orbitals,
            "numRanks": container.select.num_ranks,
            "numBases": container.select.num_bases,
            "numCopies": container.select.num_copies,
            "numD1": container.select.num_d1,
            "dqRotationAngles": container.select.rotation_angles.tolist(),
            "sfRotationAngles": container.select.sf_rotation_angles.tolist(),
            "rotationBitPrecision": self.rotation_bit_precision,
        }
        if self.multiplexed_rotation == "qrom_phase_gradient":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectPhaseGradient(select_data)
        # direct
        return QSHARP_UTILS.SOSSAWalk.MakeSelectDirectRotation(select_data)


# ═══════════════════════════════════════════════════════════════════════════════
# Main SOSSA Mapper
# ═══════════════════════════════════════════════════════════════════════════════


class SOSSAMapperSettings(Settings):
    """Settings for the SOSSAMapper."""

    def __init__(self):
        """Initialize settings for SOSSAMapper."""
        super().__init__()
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
    r"""Controlled circuit mapper for the SOSSA walk operator.

    Composes a controlled SOSSA walk step from independently-built sub-ops:

    1. **outer_prepare_op** — built by :class:`OuterPrepareMapper`
    2. **inner_prepare_op** — built by :class:`InnerPrepareMapper`
    3. **select_op** — built by :class:`SelectMapper`

    The walk operator:

    .. math::

        W = \mathrm{Ref}_{a,B} \cdot U^\dagger \cdot \mathrm{Ref}_B \cdot U

    The three ops are passed to the Q# ``MakeControlledSOSSAWalkOp`` which
    composes them with reflections. Register allocation:

    - Phase gradient register allocated if ``select_mapper.needs_phase_gradient_register``
    - Alias keep/mu registers included in outer reflection if
      ``outer_prepare_mapper.needs_alias_reflection``
    - Alias keep register included in inner reflection if
      ``inner_prepare_mapper.needs_alias_reflection``

    """

    def __init__(
        self,
        outer_prepare_mapper: OuterPrepareMapper | None = None,
        inner_prepare_mapper: InnerPrepareMapper | None = None,
        select_mapper: SelectMapper | None = None,
    ):
        """Initialize the SOSSAMapper with sub-operation mappers.

        Args:
            outer_prepare_mapper: Mapper for outer PREPARE strategy.
                Defaults to alias_sampling with 10-bit precision.
            inner_prepare_mapper: Mapper for inner (controlled) PREPARE strategy.
                Defaults to controlled_alias_sampling with 10-bit precision.
            select_mapper: Mapper for SELECT (multiplexed rotation) strategy.
                Defaults to qrom_phase_gradient with 10-bit precision.

        """
        super().__init__()
        self._settings = SOSSAMapperSettings()
        self.outer_prepare_mapper = outer_prepare_mapper or OuterPrepareMapper()
        self.inner_prepare_mapper = inner_prepare_mapper or InnerPrepareMapper()
        self.select_mapper = select_mapper or SelectMapper()

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, controlled_unitary: ControlledUnitary) -> Circuit:
        r"""Construct a controlled SOSSA walk step circuit.

        Each sub-mapper independently builds its Q# callable, then the
        walk step composer stitches them with reflections.

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

        # 1. Build each sub-operation's Q# callable independently.
        outer_prepare_op = self.outer_prepare_mapper.build_op(unitary_container)
        inner_prepare_op = self.inner_prepare_mapper.build_op(unitary_container)
        select_op = self.select_mapper.build_op(unitary_container)

        # 2. Compute register sizes for the compose step.
        num_orbitals = unitary_container.select.num_orbitals
        num_system_qubits = 2 * num_orbitals
        x_o_dim = num_orbitals + unitary_container.select.num_ranks * unitary_container.select.num_copies
        num_outer_qubits = ceil(log2(x_o_dim)) if x_o_dim > 1 else 1
        num_inner_qubits = ceil(log2(unitary_container.select.num_bases + 1))

        # 3. Compose into the walk step via Q#.
        walk_params = {
            "outerPrepareOp": outer_prepare_op,
            "innerPrepareOp": inner_prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system_qubits,
            "numOuterQubits": num_outer_qubits,
            "numInnerQubits": num_inner_qubits,
            "power": power,
            "outerReflectionIncludesKeep": self.outer_prepare_mapper.needs_alias_reflection,
            "innerReflectionIncludesKeep": self.inner_prepare_mapper.needs_alias_reflection,
            "needsPhaseGradient": self.select_mapper.needs_phase_gradient_register,
            "phaseGradientBits": self.select_mapper.rotation_bit_precision,
        }

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkCircuit,
            parameter=walk_params,
        )
        qsharp_op = QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkOp(
            outer_prepare_op, inner_prepare_op, select_op,
            num_system_qubits, num_outer_qubits, num_inner_qubits, power,
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)
