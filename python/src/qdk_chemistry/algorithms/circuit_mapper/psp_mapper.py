"""QDK/Chemistry PREPARE-SELECT-PREPARE circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

from qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_psp_mapper import _build_pauli_select_op
from qdk_chemistry.data import AlgorithmRef, Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = ["PSPMapper", "PSPMapperSettings"]


class PSPMapperSettings(Settings):
    """Settings for the PSPMapper.

    Attributes:
        prepare: Algorithm reference for the PREPARE oracle state preparation.
            Defaults to ``DensePureStatePreparation``.

    """

    def __init__(self):
        """Initialize the settings for PSPMapper."""
        super().__init__()
        self._set_default(
            "prepare",
            "algorithm_ref",
            AlgorithmRef("state_prep", "dense_pure_state"),
            "Algorithm for the PREPARE oracle state preparation. ",
        )


class PSPMapper(CircuitMapper):
    r"""Circuit mapper using the PREPARE-SELECT-PREPARE pattern.

    Composes a block encoding from:

    1. **PREPARE** — amplitude-loading into the ancilla register, resolved via
       the ``prepare`` setting.  Defaults to ``DensePureStatePreparation``.
    2. **SELECT** — Pauli SELECT oracle applied on the system register,
       constructed directly from the block-encoding container's SELECT data.

    The two callables are stitched together by the Q# ``PrepSelPrep`` operation:

    .. math::

        B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}

    No reflection is applied, so a
    :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`
    maps to a *single* application of its underlying block encoding rather than to the walk
    :math:`W = (2|0\rangle\langle 0| - I) \cdot B[H]`: its power counts walk steps, which only
    mean something once the reflections are interleaved. Whoever schedules the walk owns them,
    and pairs the circuit produced here with :meth:`reflection_op` — unary-iteration phase
    estimation interleaves its own reflections so it can omit exactly one, which is what makes
    a non-power-of-two query count possible.

    """

    def __init__(self):
        """Initialize the PSPMapper."""
        super().__init__()
        self._settings = PSPMapperSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"prepare_select_prepare"``.

        """
        return "prepare_select_prepare"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"circuit_mapper"``.

        """
        return "circuit_mapper"

    def _resolve_lcu(self, container: Any) -> tuple[LCUContainer, int, bool]:
        """Unwrap a container into its LCU data, power, and whether it is a quantum walk.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The LCU data, the requested power, and whether the container is a quantum walk.

        Raises:
            ValueError: If the container is neither an LCU nor an LCU walk.

        """
        if isinstance(container, LCUWalkContainer):
            return container.block_encoding, container.power, True
        if isinstance(container, LCUContainer):
            return container, container.power, False
        raise ValueError(
            f"Container type '{type(container).__name__}' is not supported. "
            "PSPMapper requires LCUContainer or LCUWalkContainer."
        )

    def _run_impl(self, evolution: UnitaryRepresentation) -> Circuit:
        r"""Construct the block-encoding circuit on the flat ``[system | ancilla]`` register.

        Args:
            evolution: The unitary representation containing either an
                :class:`LCUContainer` (plain block encoding) or an
                :class:`LCUWalkContainer` (quantum walk).

        Returns:
            Circuit: A quantum circuit implementing the block encoding.

        """
        container = evolution.get_container()
        lcu, power, use_quantum_walk = self._resolve_lcu(container)
        # A walk container's power counts walk steps, not block encodings, and the walk is
        # whoever schedules the reflections' business, so only a plain LCU repeats the block.
        block_power = 1 if use_quantum_walk else power

        if lcu.prepare is not None:
            prepare_op = self._create_nested("prepare").run(lcu.prepare)._qsharp_op  # noqa: SLF001
        else:
            # The 0-ancilla case has a 0-mode wavefunction, so PREPARE is a no-op.
            prepare_op = QSHARP_UTILS.PrepSelPrep.NoOpPrepare
        select_op = _build_pauli_select_op(lcu.select)
        num_system = lcu.select.num_target_qubits

        psp_parameters = {
            "prepareOp": prepare_op,
            "selectOp": select_op,
            "numSystemQubits": num_system,
            "numAncillaQubits": lcu.num_prepare_ancillas,
            "power": block_power,
        }

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepCircuit,
            parameter=psp_parameters,
        )
        qsharp_op = QSHARP_UTILS.PrepSelPrep.MakePrepSelPrepOp(prepare_op, select_op, num_system, block_power)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

    def num_ancillary_qubits(self, container: Any) -> int:
        """The number of ancilla qubits the walk needs beyond the system register.

        Args:
            container: The container held by the unitary representation.

        Returns:
            The size of the PREPARE ancilla register, which is also the register the walk
            reflection acts on.

        """
        lcu, _, _ = self._resolve_lcu(container)
        return lcu.num_prepare_ancillas

    def reflection_op(self, container: Any):
        """Build the reflection a qubitization walk pairs this block encoding with.

        This is the seam that lets a schedule assemble its own walk without knowing that the
        block encoding is a PREPARE-SELECT-PREPARE one: the returned callable acts on the same
        flat ``[system | ancilla]`` register as the circuit :meth:`run` produces.

        Args:
            container: The container held by the unitary representation.

        Returns:
            A Q# callable reflecting about the all-zero state of the block-encoding ancillas.

        """
        lcu, _, _ = self._resolve_lcu(container)
        return QSHARP_UTILS.PrepSelPrep.MakeAncillaReflectionOp(lcu.select.num_target_qubits)
