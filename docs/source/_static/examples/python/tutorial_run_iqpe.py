"""Estimate the selected-space N2 ground-state energy with native IQPE."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# docs-example: slow

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.utils import Logger
from tutorial_map_n2_to_qubits import QubitMappingResult, run_qubit_mapping_workflow
from tutorial_prepare_trial_state import TrialStateResult, run_trial_state_workflow

DEFAULT_NUM_PHASE_BITS = 6
DEFAULT_SHOTS_PER_BIT = 3
DEFAULT_NUM_COMPLETE_RUNS = 20
DEFAULT_FIRST_SEED = 42
TARGET_ENERGY_ERROR_HARTREE = 1e-3


@dataclass
class EvolutionTimeChoice:
    """Reference-guided evolution-time choice for the teaching workflow."""

    bound_time_hartree_inverse: float
    bound_reference_phase_fraction: float
    time_hartree_inverse: float
    reference_phase_fraction: float
    grid_phase_fraction: float
    grid_bitstring: str
    grid_active_energy_hartree: float


@dataclass
class IqpeProblem:
    """Prepared Hamiltonian, trial state, and IQPE circuit configuration."""

    mapping: QubitMappingResult
    trial_state: TrialStateResult
    evolution_time: EvolutionTimeChoice
    num_phase_bits: int
    shots_per_bit: int
    circuit_builder_reference: AlgorithmRef
    iteration_circuits: list[Circuit]


@dataclass
class IqpeRun:
    """One complete bitwise IQPE run."""

    seed: int
    bitstring: str
    phase_fraction: float
    active_energy_hartree: float
    total_energy_hartree: float
    error_hartree: float
    runtime_seconds: float


@dataclass
class IqpeWorkflowResult:
    """Repeated IQPE results and their unique modal estimate."""

    problem: IqpeProblem
    runs: list[IqpeRun]
    bitstring_counts: dict[str, int]
    modal_run: IqpeRun
    total_runtime_seconds: float


################################################################################
# start-cell-evolution-time
def choose_reference_guided_evolution_time(
    qubit_hamiltonian: QubitOperator,
    reference_active_energy_hartree: float,
    *,
    num_phase_bits: int = DEFAULT_NUM_PHASE_BITS,
    target_energy_error_hartree: float = TARGET_ENERGY_ERROR_HARTREE,
) -> EvolutionTimeChoice:
    """Align a known reference energy with a selected phase-grid error."""
    if num_phase_bits <= 0:
        raise ValueError("num_phase_bits must be positive")

    # Start from a time whose signed, unaliased interval contains the entire
    # Hamiltonian spectrum, then identify the nearest finite-bit grid point.
    # QubitOperator.schatten_norm is the sum of absolute Pauli coefficients,
    # denoted lambda in the chapter's evolution-time derivation.
    bound_time = np.pi / qubit_hamiltonian.schatten_norm
    reference_phase = (bound_time * reference_active_energy_hartree / (2 * np.pi)) % 1.0
    grid_size = 2**num_phase_bits
    grid_index = round(reference_phase * grid_size) % grid_size
    grid_phase = grid_index / grid_size

    # Convert the grid fraction to the signed angle convention used by the QDK
    # product-formula result container.
    grid_angle = 2 * np.pi * grid_phase
    if grid_angle > np.pi:
        grid_angle -= 2 * np.pi

    evolution_time = grid_angle / (
        reference_active_energy_hartree + target_energy_error_hartree
    )
    grid_energy = grid_angle / evolution_time
    aligned_reference_phase = (
        evolution_time * reference_active_energy_hartree / (2 * np.pi)
    ) % 1.0
    return EvolutionTimeChoice(
        bound_time_hartree_inverse=float(bound_time),
        bound_reference_phase_fraction=float(reference_phase),
        time_hartree_inverse=float(evolution_time),
        reference_phase_fraction=float(aligned_reference_phase),
        grid_phase_fraction=float(grid_phase),
        grid_bitstring=f"{grid_index:0{num_phase_bits}b}",
        grid_active_energy_hartree=float(grid_energy),
    )
    # end-cell-evolution-time


################################################################################


################################################################################
# start-cell-iqpe-settings
def iqpe_circuit_builder_reference(
    evolution_time_hartree_inverse: float,
    *,
    num_phase_bits: int = DEFAULT_NUM_PHASE_BITS,
) -> AlgorithmRef:
    """Configure native IQPE with first-order repeated-power Trotter evolution."""
    return AlgorithmRef(
        "qpe_circuit_builder",
        "qdk_iterative",
        num_bits=num_phase_bits,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper", "pauli_sequence"
        ),
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "trotter",
            time=evolution_time_hartree_inverse,
            order=1,
            num_divisions=1,
            power_strategy="repeat",
        ),
    )
    # end-cell-iqpe-settings


################################################################################


def prepare_iqpe_problem(
    *,
    num_phase_bits: int = DEFAULT_NUM_PHASE_BITS,
    shots_per_bit: int = DEFAULT_SHOTS_PER_BIT,
) -> IqpeProblem:
    """Prepare the mapped problem, four-determinant state, and iteration circuits."""
    mapping = run_qubit_mapping_workflow()
    trial_state = run_trial_state_workflow((4,)).trial_states[0]
    evolution_time = choose_reference_guided_evolution_time(
        mapping.qubit_hamiltonian,
        mapping.mapped_active_energy,
        num_phase_bits=num_phase_bits,
    )
    circuit_builder_ref = iqpe_circuit_builder_reference(
        evolution_time.time_hartree_inverse,
        num_phase_bits=num_phase_bits,
    )
    circuit_builder = create(
        "qpe_circuit_builder",
        "qdk_iterative",
        num_bits=num_phase_bits,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper", "pauli_sequence"
        ),
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "trotter",
            time=evolution_time.time_hartree_inverse,
            order=1,
            num_divisions=1,
            power_strategy="repeat",
        ),
    )
    iteration_circuits = circuit_builder.run(
        state_preparation=trial_state.circuit,
        qubit_hamiltonian=mapping.qubit_hamiltonian,
    )
    return IqpeProblem(
        mapping=mapping,
        trial_state=trial_state,
        evolution_time=evolution_time,
        num_phase_bits=num_phase_bits,
        shots_per_bit=shots_per_bit,
        circuit_builder_reference=circuit_builder_ref,
        iteration_circuits=iteration_circuits,
    )


################################################################################
# start-cell-run-iqpe
def run_complete_iqpe(problem: IqpeProblem, *, seed: int) -> IqpeRun:
    """Execute one complete bitwise IQPE run."""
    iqpe = create(
        "phase_estimation",
        "qdk_iterative",
        shots_per_bit=problem.shots_per_bit,
    )
    iqpe.settings().set("qpe_circuit_builder", problem.circuit_builder_reference)
    iqpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=seed),
    )

    start = perf_counter()
    result = iqpe.run(
        state_preparation=problem.trial_state.circuit,
        qubit_hamiltonian=problem.mapping.qubit_hamiltonian,
    )
    runtime_seconds = perf_counter() - start

    # Format the reconstructed grid fraction in canonical most-significant-bit
    # order rather than exposing the order in which iterative bits were measured.
    grid_size = 2**problem.num_phase_bits
    grid_index = round(result.phase_fraction * grid_size) % grid_size
    bitstring = f"{grid_index:0{problem.num_phase_bits}b}"

    total_energy = result.raw_energy + problem.mapping.core_energy
    energy_error = total_energy - problem.mapping.active_space_result.refined_energy
    return IqpeRun(
        seed=seed,
        bitstring=bitstring,
        phase_fraction=result.phase_fraction,
        active_energy_hartree=result.raw_energy,
        total_energy_hartree=total_energy,
        error_hartree=energy_error,
        runtime_seconds=runtime_seconds,
    )
    # end-cell-run-iqpe


################################################################################


################################################################################
# start-cell-aggregate-runs
def select_unique_mode(runs: list[IqpeRun]) -> tuple[dict[str, int], IqpeRun]:
    """Return bitstring frequencies and the run representing their unique mode."""
    if not runs:
        raise ValueError("at least one complete IQPE run is required")
    bitstring_counts = Counter(run.bitstring for run in runs)
    highest_count = max(bitstring_counts.values())
    modal_bitstrings = sorted(
        bitstring
        for bitstring, count in bitstring_counts.items()
        if count == highest_count
    )
    if len(modal_bitstrings) != 1:
        raise RuntimeError(
            f"complete IQPE runs have no unique mode: {bitstring_counts}"
        )
    modal_bitstring = modal_bitstrings[0]
    modal_run = next(run for run in runs if run.bitstring == modal_bitstring)
    return dict(sorted(bitstring_counts.items())), modal_run
    # end-cell-aggregate-runs


################################################################################


def print_iqpe_settings(
    problem: IqpeProblem,
    *,
    num_complete_runs: int,
    first_seed: int,
) -> None:
    """Print all settings students record before the long simulation."""
    last_seed = first_seed + num_complete_runs - 1
    print("IQPE settings:")
    print(f"  Trial determinants: {problem.trial_state.num_determinants}")
    print(f"  Trial fidelity: {problem.trial_state.fidelity:.12f}")
    print(f"  Compute qubits: {problem.mapping.num_compute_qubits}")
    print("  Readout ancillas: 1")
    print(f"  Phase bits: {problem.num_phase_bits}")
    print(f"  Shots per bit: {problem.shots_per_bit}")
    print(f"  Complete runs: {num_complete_runs}")
    print(f"  Simulator seeds: {first_seed}-{last_seed}")
    print("  Hamiltonian simulation: first-order Trotter product formula")
    print("  Trotter divisions: 1")
    print("  Controlled powers: repeated approximate base unitary")
    print(
        "  Hamiltonian coefficient sum (lambda): "
        f"{problem.mapping.qubit_hamiltonian.schatten_norm:.12f} Hartree"
    )
    print(
        "  Initial unaliased time bound: "
        f"{problem.evolution_time.bound_time_hartree_inverse:.12f} Hartree^-1"
    )
    print(
        "  Reference phase at initial time bound: "
        f"{problem.evolution_time.bound_reference_phase_fraction:.12f}"
    )
    print(
        "  Selected evolution time: "
        f"{problem.evolution_time.time_hartree_inverse:.12f} Hartree^-1"
    )


def run_iqpe_workflow(
    *,
    num_complete_runs: int = DEFAULT_NUM_COMPLETE_RUNS,
    first_seed: int = DEFAULT_FIRST_SEED,
    num_phase_bits: int = DEFAULT_NUM_PHASE_BITS,
    shots_per_bit: int = DEFAULT_SHOTS_PER_BIT,
) -> IqpeWorkflowResult:
    """Prepare and execute repeated complete IQPE runs with progress output."""
    if num_complete_runs <= 0:
        raise ValueError("num_complete_runs must be positive")
    problem = prepare_iqpe_problem(
        num_phase_bits=num_phase_bits,
        shots_per_bit=shots_per_bit,
    )
    print_iqpe_settings(
        problem,
        num_complete_runs=num_complete_runs,
        first_seed=first_seed,
    )
    print(
        f"Built {len(problem.iteration_circuits)} IQPE iteration circuits "
        f"for {problem.mapping.num_compute_qubits} compute qubits."
    )
    print(
        f"Reference phase: {problem.evolution_time.reference_phase_fraction:.12f}; "
        f"grid bitstring: {problem.evolution_time.grid_bitstring}"
    )

    workflow_start = perf_counter()
    runs = []
    for run_index in range(num_complete_runs):
        seed = first_seed + run_index
        print(
            f"Running complete IQPE {run_index + 1}/{num_complete_runs} "
            f"(seed={seed})...",
            flush=True,
        )
        run = run_complete_iqpe(problem, seed=seed)
        runs.append(run)
        print(
            f"  bits={run.bitstring}, total energy={run.total_energy_hartree:.12f} "
            f"Hartree, error={run.error_hartree:+.3e} Hartree, "
            f"runtime={run.runtime_seconds:.1f} s",
            flush=True,
        )

    total_runtime = perf_counter() - workflow_start
    bitstring_counts, modal_run = select_unique_mode(runs)
    return IqpeWorkflowResult(
        problem=problem,
        runs=runs,
        bitstring_counts=bitstring_counts,
        modal_run=modal_run,
        total_runtime_seconds=total_runtime,
    )


def print_iqpe_results(result: IqpeWorkflowResult) -> None:
    """Print the final values students record in the lab notebook."""
    print(f"Complete-run bitstring counts: {result.bitstring_counts}")
    print(f"Modal bitstring: {result.modal_run.bitstring}")
    print(
        f"Modal active-space energy: {result.modal_run.active_energy_hartree:.12f} "
        "Hartree"
    )
    print(f"Core energy: {result.problem.mapping.core_energy:.12f} Hartree")
    print(f"Modal total energy: {result.modal_run.total_energy_hartree:.12f} Hartree")
    print(
        "CASCI algorithmic reference: "
        f"{result.problem.mapping.active_space_result.refined_energy:.12f} Hartree"
    )
    print(f"Modal energy error: {result.modal_run.error_hartree:+.12f} Hartree")
    print(f"Total IQPE runtime: {result.total_runtime_seconds:.1f} seconds")


def main() -> None:
    """Run the complete tutorial IQPE workflow."""
    Logger.set_global_level(Logger.LogLevel.off)
    result = run_iqpe_workflow()
    print_iqpe_results(result)


if __name__ == "__main__":
    main()
