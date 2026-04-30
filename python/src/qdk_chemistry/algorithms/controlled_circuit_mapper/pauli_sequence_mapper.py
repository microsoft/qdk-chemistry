"""QDK/Chemistry sequence structure controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["PauliSequenceMapper"]


class PauliSequenceMapper(ControlledCircuitMapper):
    r"""Controlled evolution circuit mapper using Pauli product formula term sequences.

    Given a time-evolution operator expressed as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper constructs
    a controlled version of :math:`U(t)` using the following pattern:

    1. Each Pauli operator :math:`P_j` is basis-rotated into the :math:`Z` basis.
    2. Qubits involved in :math:`P_j` are entangled into a sequence using CNOT gates.
    3. A controlled :math:`R_z` rotation implements
        :math:`e^{-i\,\theta_j\,P_j} \;\rightarrow\; \text{CRZ}(2 \theta_j)`.
    4. The basis rotations and entangling operations are uncomputed.

    The process repeats for all terms in :math:`U_{\mathrm{step}}`, for :math:`r` step repetitions,
    and for the specified power.

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.

    """

    def __init__(self):
        """Initialize the PauliSequenceMapper."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name."""
        return "pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, controlled_unitary: ControlledUnitary) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled unitary.

        Args:
            controlled_unitary: The controlled unitary containing the Hamiltonian
            and evolution parameters. Target indices are read from
            controlled_unitary.target_indices.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U^{\text{power}}`
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the unitary container type is not supported.
            ValueError: If multiple control qubits are provided.

        """
        unitary_container = controlled_unitary.unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {controlled_unitary.get_unitary_container_type()} container type is not supported. "
                "PauliSequenceMapper only supports PauliProductFormula container for the unitary."
            )

        if len(controlled_unitary.control_indices) != 1:
            raise ValueError("PauliSequenceMapper currently only supports a single control qubit.")

        target_indices = controlled_unitary.target_indices

        pauli_terms: list[list[qsharp.Pauli]] = []
        angles: list[float] = []
        for term in unitary_container.step_terms:
            base_terms = [qsharp.Pauli.I] * unitary_container.num_qubits
            for index, pauli in term.pauli_term.items():
                base_terms[index] = getattr(qsharp.Pauli, pauli)
            pauli_terms.append(base_terms.copy())
            angles.append(term.angle)

        controlled_evo_params = QSHARP_UTILS.ControlledPauliExp.RepControlledPauliExpParams(
            pauliExponents=pauli_terms,
            pauliCoefficients=angles,
            repetitions=unitary_container.step_reps,
            control=controlled_unitary.control_indices[0],
            systems=target_indices,
        )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpCircuit,
            parameter=vars(controlled_evo_params),
        )

        controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpOp(controlled_evo_params)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)
