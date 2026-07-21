"""QDK/Chemistry CSWAP-sandwich controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["ControlledSwapPauliSequenceMapper"]


class ControlledSwapPauliSequenceMapper(ControlledCircuitMapper):
    r"""Controlled evolution circuit mapper using a CSWAP-sandwich construction.

    Given a time-evolution operator expressed as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper constructs
    a controlled version of :math:`U(t)` without controlling every gate of the evolution.
    It uses a *CSWAP sandwich*:

    1. An internally allocated ``vacuum`` register (initialized to :math:`|0\ldots0\rangle`)
       is conditionally swapped with the system register, controlled on the control qubit.
    2. The *uncontrolled* Pauli evolution :math:`U(t)` is applied to the vacuum register
       (repeated ``step_reps`` times).
    3. The controlled swap is uncomputed and the vacuum register is reset.

    When the control qubit is :math:`|0\rangle` the evolution acts on the vacuum reference
    :math:`|0\ldots0\rangle` (leaving the system state untouched); when it is :math:`|1\rangle`
    the system state is parked in the vacuum register and is evolved. The target eigenphase
    therefore accumulates on the :math:`|1\rangle` branch, matching the standard controlled-:math:`U`
    convention.

    This trades the cost of controlling every gate for a single layer of controlled-:math:`\mathrm{SWAP}`
    gates, so the (repeated) evolution is applied with uncontrolled gates only.

    **Residual phase / Hamiltonian restriction.** On the :math:`|0\rangle` branch the evolution acts on
    the vacuum, contributing :math:`U|0\ldots0\rangle = \lambda\,|0\ldots0\rangle` with
    :math:`\lambda = \langle 0\ldots0|U|0\ldots0\rangle`. This leaves a residual phase
    :math:`\varphi_0 = \arg\lambda = -E_0 t` (where :math:`E_0 = \langle 0\ldots0|H|0\ldots0\rangle`) on the
    measured eigenphase, which is a known constant and can be removed by the QPE feedback rotation.
    The construction is exact **only if** :math:`|0\ldots0\rangle` is an eigenstate of :math:`U`
    (equivalently :math:`|\lambda| = 1`); otherwise the vacuum leaks (:math:`|\lambda| < 1`), the control
    qubit decoheres, and the phase cannot be recovered. This holds automatically for
    particle-number-conserving Hamiltonians (e.g. Jordan-Wigner/Bravyi-Kitaev molecular Hamiltonians),
    for which the all-zero register is the exact vacuum eigenstate.

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.
        * The vacuum register is allocated internally by the Q# operation.

    """

    def __init__(self):
        """Initialize the ControlledSwapPauliSequenceMapper."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name."""
        return "cswap_pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled unitary.

        Args:
            unitary: The unitary representation containing the Hamiltonian and evolution parameters.
            Control and target indices are read from settings.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U` via the CSWAP sandwich,
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the unitary container type is not supported.
            ValueError: If multiple control qubits are provided.

        """
        unitary_container = unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported. "
                "ControlledSwapPauliSequenceMapper only supports PauliProductFormula container for the unitary."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("ControlledSwapPauliSequenceMapper currently only supports a single control qubit.")

        target_indices = self._get_target_indices(unitary)

        pauli_terms: list[list[qsharp.Pauli]] = []
        angles: list[float] = []
        for term in unitary_container.step_terms:
            base_terms = [qsharp.Pauli.I] * unitary_container.num_qubits
            for index, pauli in term.pauli_term.items():
                base_terms[index] = getattr(qsharp.Pauli, pauli)
            pauli_terms.append(base_terms.copy())
            angles.append(term.angle)

        controlled_evo_params = QSHARP_UTILS.ControlledSwapPauliExp.ControlledSwapPauliExpParams(
            pauliExponents=pauli_terms,
            pauliCoefficients=angles,
            repetitions=unitary_container.step_reps,
            control=control_indices[0],
            systems=target_indices,
        )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.ControlledSwapPauliExp.MakeRepControlledSwapPauliExpCircuit,
            parameter=vars(controlled_evo_params),
        )

        controlled_unitary_op = QSHARP_UTILS.ControlledSwapPauliExp.MakeRepControlledSwapPauliExpOp(
            controlled_evo_params
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)
