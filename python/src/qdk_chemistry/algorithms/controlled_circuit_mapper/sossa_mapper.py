"""QDK/Chemistry SOSSA (Sum of Squares with Ancilla) controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["SOSSAMapper", "SOSSAMapperSettings"]


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

    Composes a controlled SOSSA walk step from:

    1. **OuterPREP** — amplitude-loading into the :math:`x_o` register.
    2. **InnerPREP** — conditional alias sampling on :math:`(x_o, b)`.
    3. **SELECT** — Givens rotations + Majorana operator.
    4. **Reflections** — Inner (Ref_B) and outer (Ref_{a,B}).

    The walk operator (Eq. 77):

    .. math::

        W = \mathrm{Ref}_{a,B} \cdot U^\dagger \cdot \mathrm{Ref}_B \cdot U

    Only reflections are controlled for QPE; :math:`U` and :math:`U^\dagger` run unconditionally.

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
        rotation_bit_precision: int = self._settings.get("rotation_bit_precision")
        coefficient_bit_precision: int = self._settings.get("coefficient_bit_precision")

        # Build Q# parameters for the SOSSA walk step
        sossa_params = {
            "numOrbitals": unitary_container.select.num_orbitals,
            "numRanks": unitary_container.select.num_ranks,
            "numBases": unitary_container.select.num_bases,
            "numCopies": unitary_container.select.num_copies,
            "numD1": unitary_container.select.num_d1,
            "outerStatevector": unitary_container.outer_prepare.statevector.tolist(),
            "innerCoefficients": unitary_container.inner_prepare.conditional_coefficients.tolist(),
            "dqRotationAngles": unitary_container.select.rotation_angles.tolist(),
            "sfRotationAngles": unitary_container.select.sf_rotation_angles.tolist(),
            "rotationBitPrecision": rotation_bit_precision,
            "coefficientBitPrecision": coefficient_bit_precision,
            "power": power,
        }

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkCircuit,
            parameter=sossa_params,
        )
        qsharp_op = QSHARP_UTILS.SOSSAWalk.MakeControlledSOSSAWalkOp(sossa_params)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)
