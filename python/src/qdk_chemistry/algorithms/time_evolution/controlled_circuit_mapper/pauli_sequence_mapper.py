"""QDK/Chemistry sequence structure controlled evolution circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence
from pathlib import Path

import qdk
from qdk import qsharp

from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    PauliProductFormulaContainer,
)
from qdk_chemistry.data.time_evolution.controlled_time_evolution import ControlledTimeEvolutionUnitary

from .base import ControlledEvolutionCircuitMapper

__all__: list[str] = ["PauliSequenceMapper", "PauliSequenceMapperSettings"]


class PauliSequenceMapperSettings(Settings):
    """Settings for PauliSequenceMapper."""

    def __init__(self):
        """Initialize PauliSequenceMapperSettings with default values.

        Attributes:
            power: The power of the controlled unitary to be constructed. It controls
                how many times the controlled evolution operator :math:`U` is repeated.

        """
        super().__init__()
        self._set_default("power", "int", 1, "The power of the controlled unitary to be constructed.")


class PauliSequenceMapper(ControlledEvolutionCircuitMapper):
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

    def __init__(self, power: int = 1):
        """Initialize the PauliSequenceMapper.

        Args:
            power: The power of the controlled unitary to be constructed. It controls
                how many times the controlled evolution operator :math:`U` is repeated.

        """
        super().__init__()
        self._settings = PauliSequenceMapperSettings()
        self._settings.set("power", power)

    def name(self) -> str:
        """Return the algorithm name."""
        return "pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_evolution_circuit_mapper as the algorithm type name."""
        return "controlled_evolution_circuit_mapper"

    def _run_impl(
        self, controlled_evolution: ControlledTimeEvolutionUnitary, target_indices: Sequence[int] | None = None
    ) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled time evolution unitary.

        Args:
            controlled_evolution: The controlled time evolution unitary containing the Hamiltonian
            and evolution parameters.
            target_indices: Indices of the target qubits in the circuit. If None, defaults to all
            qubits except the control qubits at controlled_evolution.control_indices.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U^{\text{power}}`
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the time evolution unitary container type is not supported.
            ValueError: If multiple control qubits are provided.

        """
        # Import Q# code for controlled Pauli exponentiation
        code = (Path(__file__).parent / "ControlledPauliExp.qs").read_text()
        qsharp.eval(code)

        unitary_container = controlled_evolution.time_evolution_unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {controlled_evolution.get_unitary_container_type()} container type is not supported. "
                "PauliSequenceMapper only supports PauliProductFormula container for time evolution unitary."
            )

        if len(controlled_evolution.control_indices) != 1:
            raise ValueError("PauliSequenceMapper currently only supports a single control qubit.")

        total_qubits = controlled_evolution.get_num_total_qubits()

        if target_indices is None:
            target_indices = [i for i in range(total_qubits) if i not in controlled_evolution.control_indices]

        pauli_terms: list[list[qsharp.Pauli]] = []
        angles: list[float] = []
        for term in unitary_container.step_terms:
            base_terms = [qsharp.Pauli.I] * unitary_container.num_qubits
            for index, pauli in term.pauli_term.items():
                base_terms[index] = getattr(qsharp.Pauli, pauli)
            pauli_terms.append(base_terms.copy())
            angles.append(term.angle)

        flattened_pauli_terms: list[list[qsharp.Pauli]] = []
        flattened_angles: list[float] = []
        for _ in range(unitary_container.step_reps):
            flattened_pauli_terms.extend(pauli_terms)
            flattened_angles.extend(angles)

        controlled_evo_params = {
            "pauliExponents": flattened_pauli_terms,
            "pauliCoefficients": flattened_angles,
            "repetitions": self._settings.get("power"),
        }

        qsc = qsharp.circuit(
            qdk.code.MakeRepControlledEvolutionCircuit,
            controlled_evo_params,
            controlled_evolution.control_indices[0],
            target_indices,
        )

        qir = qsharp.compile(
            qdk.code.MakeRepControlledEvolutionCircuit,
            controlled_evo_params,
            controlled_evolution.control_indices[0],
            target_indices,
        )

        controlled_evolution_op = qdk.code.MakeRepControlledEvolutionOp(controlled_evo_params)

        return Circuit(qsharp=qsc, qir=qir, qsharp_op=controlled_evolution_op)
