"""QDK/Chemistry sequence structure evolution circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import EvolutionCircuitMapper

__all__: list[str] = ["PauliSequenceMapper", "PauliSequenceMapperSettings"]


class PauliSequenceMapperSettings(Settings):
    """Settings for PauliSequenceMapper."""

    def __init__(self):
        """Initialize PauliSequenceMapperSettings with default values."""
        super().__init__()


class PauliSequenceMapper(EvolutionCircuitMapper):
    r"""Evolution circuit mapper using Pauli product formula term sequences.

    Given a time-evolution operator expressed as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper constructs
    a :math:`U(t)` using the following pattern:

    1. Each Pauli operator :math:`P_j` is basis-rotated into the :math:`Z` basis.
    2. Qubits involved in :math:`P_j` are entangled into a sequence using CNOT gates.
    3. A :math:`R_z` rotation implements
        :math:`e^{-i\,\theta_j\,P_j} \;\rightarrow\; R_z(2 \theta_j)`.
    4. The basis rotations and entangling operations are uncomputed.

    Notes:
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.

    """

    def __init__(self):
        """Initialize the PauliSequenceMapper."""
        super().__init__()
        self._settings = PauliSequenceMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "pauli_sequence"

    def type_name(self) -> str:
        """Return evolution_circuit_mapper as the algorithm type name."""
        return "evolution_circuit_mapper"

    def _run_impl(self, evolution: TimeEvolutionUnitary) -> Circuit:
        r"""Construct a quantum circuit implementing the time evolution unitary.

        Args:
            evolution: The time evolution unitary containing the Hamiltonian
            and evolution parameters.

        Returns:
            Circuit: A quantum circuit implementing the unitary :math:`U`
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the time evolution unitary container type is not supported.

        """
        unitary_container = evolution.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {evolution.get_container_type()} container type is not supported. "
                "PauliSequenceMapper only supports PauliProductFormula container for time evolution unitary."
            )

        pauli_terms: list[list[qsharp.Pauli]] = []
        angles: list[float] = []
        for term in unitary_container.step_terms:
            base_terms = [qsharp.Pauli.I] * unitary_container.num_qubits
            for index, pauli in term.pauli_term.items():
                base_terms[index] = getattr(qsharp.Pauli, pauli)
            pauli_terms.append(base_terms.copy())
            angles.append(term.angle)

        evo_params = {
            "pauliExponents": pauli_terms,
            "pauliCoefficients": angles,
            "repetitions": unitary_container.step_reps,
        }

        target_indices = list(range(unitary_container.num_qubits))

        qsc = qsharp.circuit(
            QSHARP_UTILS.PauliExp.MakeRepPauliExpCircuit,
            evo_params,
            target_indices,
        )

        qir = qsharp.compile(
            QSHARP_UTILS.PauliExp.MakeRepPauliExpCircuit,
            evo_params,
            target_indices,
        )

        evolution_op = QSHARP_UTILS.PauliExp.MakeRepPauliExpOp(evo_params)

        return Circuit(
            qsharp=qsc,
            qir=qir,
            qsharp_op=evolution_op,
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.PauliExp.MakeRepPauliExpCircuit,
                parameter={"evo_params": evo_params, "target_indices": target_indices},
            ),
        )
