"""Prepare sparse trial states for the stretched-N2 selected active space."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, Hamiltonian, Wavefunction
from qdk_chemistry.data.symmetry import SymmetryLabel, axes
from qdk_chemistry.utils import Logger
from tutorial_choose_active_space import ActiveSpaceResult, run_active_space_workflow


@dataclass
class DeterminantContribution:
    """One determinant's contribution to the reference wavefunction."""

    occupation: str
    amplitude: complex
    weight: float
    cumulative_weight: float


@dataclass
class TrialStateResult:
    """Quality and circuit cost for one determinant truncation."""

    num_determinants: int
    trial_wavefunction: Wavefunction
    fidelity: float
    circuit: Circuit
    num_compute_qubits: int
    num_logical_gates: int
    logical_gate_counts: dict[str, int]


@dataclass
class TrialStateWorkflowResult:
    """Reference data and trial states used by the chapter."""

    active_space_result: ActiveSpaceResult
    active_hamiltonian: Hamiltonian
    reference_determinants: list[DeterminantContribution]
    trial_states: list[TrialStateResult]


################################################################################
# start-cell-determinant-weights
def leading_determinant_contributions(
    wavefunction: Wavefunction, max_determinants: int = 8
) -> list[DeterminantContribution]:
    """Return amplitudes and weights for the leading reference determinants."""
    cumulative_weight = 0.0
    contributions = []
    alpha_channel = SymmetryLabel([axes.alpha()])
    num_active_spatial_orbitals = len(
        wavefunction.get_orbitals().active_indices().indices(alpha_channel)
    )
    for determinant, coefficient in wavefunction.get_top_determinants(
        max_determinants=max_determinants
    ).items():
        amplitude = complex(coefficient)

        # Squared amplitudes contribute to the norm; their running sum shows how
        # much of the reference wavefunction the leading determinants capture.
        weight = float(abs(amplitude) ** 2)
        cumulative_weight += weight
        contributions.append(
            DeterminantContribution(
                # Configuration capacity may include zero-valued storage padding;
                # display only the physical selected active spatial orbitals.
                occupation=determinant.to_string()[:num_active_spatial_orbitals],
                amplitude=amplitude,
                weight=weight,
                cumulative_weight=cumulative_weight,
            )
        )
    return contributions
    # end-cell-determinant-weights


################################################################################


################################################################################
# start-cell-circuit-statistics
def iter_leaf_gates(value: object) -> Iterator[str]:
    """Yield normalized leaf-gate names from the decomposed QDK circuit JSON."""
    # The circuit is a nested tree of dictionaries and lists. yield from flattens
    # recursive results into one stream of leaf-gate names.
    if isinstance(value, dict):
        children = value.get("children")

        # Composite operations contain children; only childless operations are
        # counted as logical gates.
        if isinstance(children, list) and children:
            for child in children:
                yield from iter_leaf_gates(child)
        else:
            gate_name = value.get("gate")
            if isinstance(gate_name, str):
                # Q# represents CNOT as an X gate with controls; distinguish it
                # from a bare X while the full gate record is available.
                yield (
                    "CNOT"
                    if gate_name == "X" and value.get("controls")
                    else gate_name
                )
        # The children branch was already traversed, so skip that key while
        # checking other fields for additional nested circuit structures.
        for key, child in value.items():
            if key != "children":
                yield from iter_leaf_gates(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_leaf_gates(child)


def circuit_statistics(circuit: Circuit) -> tuple[int, int, dict[str, int]]:
    """Return compute-qubit count and logical gate counts."""
    # Convert the Q# circuit JSON to ordinary Python containers, then flatten
    # composite operations to the leaf gates that contribute to the count.
    circuit_data = json.loads(circuit.get_qsharp_circuit().json())
    logical_gate_names = list(iter_leaf_gates(circuit_data))
    logical_gate_counts = Counter(logical_gate_names)
    return (
        len(circuit_data["qubits"]),
        len(logical_gate_names),
        dict(sorted(logical_gate_counts.items())),
    )
    # end-cell-circuit-statistics


################################################################################


def run_trial_state_workflow(
    determinant_counts: tuple[int, ...] = (1, 2, 4),
) -> TrialStateWorkflowResult:
    """Build and compare sparse trial states for the selected N2 Hamiltonian."""
    active_space_result = run_active_space_workflow()
    reference_wavefunction = active_space_result.refined_casci_wavefunction
    selected_orbitals = active_space_result.refined_wavefunction.get_orbitals()
    active_hamiltonian = create("hamiltonian_constructor", "qdk").run(selected_orbitals)
    reference_determinants = leading_determinant_contributions(reference_wavefunction)

    trial_states = []
    for num_determinants in determinant_counts:
        ################################################################################
        # start-cell-sparse-trial
        # The leading reference determinants define the trial-state support;
        # PMC then reoptimizes their amplitudes within that restricted space.
        top_determinants = reference_wavefunction.get_top_determinants(
            max_determinants=num_determinants
        )
        projected_calculator = create(
            "projected_multi_configuration_calculator", "macis_pmc"
        )
        _, trial_wavefunction = projected_calculator.run(
            active_hamiltonian, list(top_determinants)
        )
        retained_determinants = trial_wavefunction.get_active_determinants()

        # Read both coefficient vectors in the PMC determinant order so entries
        # at the same array position always refer to the same determinant.
        reference_coefficients = np.asarray(
            [
                reference_wavefunction.get_coefficient(determinant)
                for determinant in retained_determinants
            ]
        )
        trial_coefficients = np.asarray(
            [
                trial_wavefunction.get_coefficient(determinant)
                for determinant in retained_determinants
            ]
        )

        # The trial vector is normalized on the retained support. The reference
        # entries keep their full-state normalization, so their restricted norm
        # records weight omitted by truncation.
        fidelity = float(abs(np.vdot(reference_coefficients, trial_coefficients)) ** 2)
        # end-cell-sparse-trial
        ################################################################################

        ################################################################################
        # start-cell-preparation-circuit
        state_preparation = create("state_prep", "sparse_isometry_gf2x")
        circuit = state_preparation.run(trial_wavefunction)
        num_compute_qubits, num_logical_gates, logical_gate_counts = circuit_statistics(
            circuit
        )
        # end-cell-preparation-circuit
        ################################################################################

        trial_states.append(
            TrialStateResult(
                num_determinants=trial_wavefunction.size(),
                trial_wavefunction=trial_wavefunction,
                fidelity=fidelity,
                circuit=circuit,
                num_compute_qubits=num_compute_qubits,
                num_logical_gates=num_logical_gates,
                logical_gate_counts=logical_gate_counts,
            )
        )

    return TrialStateWorkflowResult(
        active_space_result=active_space_result,
        active_hamiltonian=active_hamiltonian,
        reference_determinants=reference_determinants,
        trial_states=trial_states,
    )


def print_trial_state_results(result: TrialStateWorkflowResult) -> None:
    """Print the quality and circuit-cost comparison for the lab notebook."""
    print("Leading selected-space CASCI determinants:")
    print("  Symbols: 2 = doubly occupied, u = alpha, d = beta, 0 = unoccupied")
    print("  Occupation     Amplitude       Weight    Cumulative weight")
    for contribution in result.reference_determinants:
        print(
            f"  {contribution.occupation:<10} "
            f"{contribution.amplitude.real:+.12f}  "
            f"{contribution.weight:.12f}  "
            f"{contribution.cumulative_weight:.12f}"
        )

    print("Sparse trial-state comparison:")
    for trial_state in result.trial_states:
        print(f"\nDeterminants: {trial_state.num_determinants}")
        print(f"Fidelity: {trial_state.fidelity:.12f}")
        print(f"Compute qubits: {trial_state.num_compute_qubits}")
        print(
            f"Preparation logical gate count: {trial_state.num_logical_gates}"
        )
        print(f"Logical gate-family counts: {trial_state.logical_gate_counts}")


def main() -> None:
    """Run and report the command-line trial-state workflow."""
    Logger.set_global_level(Logger.LogLevel.off)
    result = run_trial_state_workflow()
    print_trial_state_results(result)


if __name__ == "__main__":
    main()
