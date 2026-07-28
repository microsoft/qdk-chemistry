"""QDK/Chemistry FOQCS controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.foqcs import FoqcsContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["ControlledFoqcsMapper"]


class ControlledFoqcsMapper(ControlledCircuitMapper):
    r"""Controlled block-encoding mapper for FOQCS spin-model Hamiltonians.

    Builds a controlled block encoding for a
    :class:`~qdk_chemistry.data.unitary_representation.containers.foqcs.FoqcsContainer`
    using the FOQCS (First-Order-Quantized Check-matrix Sparse) construction, in
    which the Pauli-term families are loaded as balanced Dicke states and the
    SELECT oracle is realized as a constant-depth transversal layer of CX gates
    (from the ``xReg`` ancilla) and CZ gates (from the ``zReg`` ancilla):

    .. math::

        B[H] = \mathrm{PREPARE}(c^*)^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}(c)

    The forward **PREPARE** loads a one-hot superposition over the term families
    (weighted by the normalized amplitudes) and spreads each family across the
    system register with a balanced Dicke state.  The conjugate **PREPARE** used
    for the un-preparation negates the per-family phase angles so the block
    encoding supplies the FOQCS Y-term phase correction.  Because the transversal
    SELECT is self-gated by the ancilla one-hot pattern, the outer control is
    routed onto the PREPARE pair rather than the SELECT.

    When the input is an :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`
    wrapping a FOQCS container, the block encoding is additionally wrapped with
    the reflection operator to form a quantum walk:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``FoqcsContainer`` (optionally wrapped in an ``LCUWalkContainer``).

    """

    def __init__(self):
        """Initialize the ControlledFoqcsMapper."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"foqcs"``.

        """
        return "foqcs"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"controlled_circuit_mapper"``.

        """
        return "controlled_circuit_mapper"

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a controlled FOQCS block-encoding circuit.

        Args:
            unitary: The unitary representation holding a :class:`FoqcsContainer` or a wrapping walk container.

        Returns:
            Circuit: A quantum circuit implementing the controlled FOQCS block encoding.

        Raises:
            ValueError: If the container is not a FOQCS container, or more than one control qubit is requested.

        """
        container = unitary.get_container()

        if isinstance(container, LCUWalkContainer):
            block = container.block_encoding
            power = container.power
            use_quantum_walk = True
        else:
            block = container
            power = getattr(container, "power", 1)
            use_quantum_walk = False

        if not isinstance(block, FoqcsContainer):
            raise ValueError(
                f"Container type '{unitary.get_container_type()}' is not supported. "
                "ControlledFoqcsMapper requires a FoqcsContainer, or an LCUWalkContainer wrapping one."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("ControlledFoqcsMapper currently only supports a single control qubit.")

        params = self._build_foqcs_params(block)
        phases = [family.phase for family in block.families]

        # PREPARE axis — forward loads the family phases; the conjugate un-PREP
        # negates them so the block encoding supplies the FOQCS Y-term phase.
        prepare_op = QSHARP_UTILS.Foqcs.MakeFoqcsPrepareOp(params, phases)
        prepare_conj_op = QSHARP_UTILS.Foqcs.MakeFoqcsPrepareOp(params, [-p for p in phases])

        # Transversal CX/CZ SELECT — self-gated, so the outer control drives PREPARE.
        select_op = QSHARP_UTILS.Foqcs.MakeFoqcsSelectOp(block.num_families)
        control_prepare = True

        num_system = block.num_target_qubits
        num_ancilla = block.num_prepare_ancillas

        if use_quantum_walk:
            make_circuit = QSHARP_UTILS.PrepSelPrep.MakeControlledPSPWalkCircuit
            make_op = QSHARP_UTILS.PrepSelPrep.MakeControlledPSPWalkOp
        else:
            make_circuit = QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepCircuit
            make_op = QSHARP_UTILS.PrepSelPrep.MakeControlledPrepSelPrepOp

        psp_parameters = {
            "prepareOp": prepare_op,
            "prepareConjOp": prepare_conj_op,
            "selectOp": select_op,
            "controlPrepare": control_prepare,
            "numSystemQubits": num_system,
            "numAncillaQubits": num_ancilla,
            "power": power,
        }

        qsharp_factory = QsharpFactoryData(program=make_circuit, parameter=psp_parameters)
        qsharp_op = make_op(prepare_op, prepare_conj_op, select_op, control_prepare, num_system, num_ancilla, power)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

    @staticmethod
    def _build_foqcs_params(container: FoqcsContainer):
        """Build the Q# ``FoqcsParams`` struct from a FOQCS container.

        Args:
            container: The FOQCS block-encoding container.

        Returns:
            A Q# ``FoqcsParams`` struct instance.

        """
        paulis_per_family: list[list[qsharp.Pauli]] = [
            [getattr(qsharp.Pauli, p) for p in family.paulis] for family in container.families
        ]
        offsets = [family.offset for family in container.families]
        abs_coeffs = [family.abs_coeff for family in container.families]
        phases = [family.phase for family in container.families]

        return QSHARP_UTILS.Foqcs.FoqcsParams(
            paulisPerFamily=paulis_per_family,
            offsets=offsets,
            absCoeffs=abs_coeffs,
            phases=phases,
            numSites=container.num_sites,
        )
