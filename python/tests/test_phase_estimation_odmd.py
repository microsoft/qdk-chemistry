"""Tests for observable dynamic mode decomposition phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hadamard_test_generator.base import HadamardTestGenerator
from qdk_chemistry.algorithms.phase_estimation.dynamic_mode_decomposition import DynamicModeDecomposition
from qdk_chemistry.data import Circuit, QpeResult, QubitHamiltonian, Structure
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT, QDK_CHEMISTRY_HAS_QISKIT_NATURE
from qdk_chemistry.utils.phase import energy_from_phase

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import qasm3

_SEED = 42
_EVOLUTION_TIME = float(np.pi / 48.0)
_HANKEL_ROWS = 12
_HANKEL_COLUMNS = 6
_SHOTS_PER_OBSERVABLE = 500

_EXPECTED_PHASE_FRACTION = 0.9802702159864114
_EXPECTED_ENERGY = -1.8940592653045103


@cache
def water_odmd_problem(backend: str) -> tuple[QubitHamiltonian, Circuit, Circuit | None]:
    """Build the active-space water ODMD benchmark inputs for a mapper algorithm name."""
    num_active_electrons = 2
    num_active_orbitals = 3

    water_coords = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [0.758602, 0.000000, 0.504284],
            [-0.758602, 0.000000, 0.504284],
        ],
        dtype=float,
    )
    structure = Structure(water_coords, ["O", "H", "H"])

    try:
        scf_solver = create("scf_solver")
    except KeyError as exc:
        pytest.skip(f"SCF solver is not available in this environment: {exc}")

    _, scf_wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")

    active_space_selector = create(
        "active_space_selector",
        algorithm_name="qdk_valence",
        num_active_electrons=num_active_electrons,
        num_active_orbitals=num_active_orbitals,
    )
    active_wfn = active_space_selector.run(scf_wfn)

    hamiltonian_constructor = create("hamiltonian_constructor")
    active_hamiltonian = hamiltonian_constructor.run(active_wfn.get_orbitals())

    qubit_mapper = create(
        "qubit_mapper",
        algorithm_name=backend,
        encoding="jordan-wigner",
    )
    qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)

    state_prep_builder = create("state_prep", algorithm_name="sparse_isometry_gf2x")
    state_prep = state_prep_builder.run(active_wfn)

    qsharp_state_prep = state_prep
    qiskit_state_prep: Circuit | None = None
    if backend == "qiskit":
        qiskit_circuit = state_prep.get_qiskit_circuit()
        qiskit_state_prep = Circuit(qasm=qasm3.dumps(qiskit_circuit))

    return qubit_hamiltonian, qsharp_state_prep, qiskit_state_prep


def _resolve_energy_alias(phase_fraction: float, expected_energy: float) -> tuple[float, float]:
    """Resolve the phase alias to the energy branch nearest expected_energy."""
    phase_candidates = [phase_fraction % 1.0, (1.0 - phase_fraction) % 1.0]
    energy_candidates = [energy_from_phase(phase, evolution_time=_EVOLUTION_TIME) for phase in phase_candidates]
    best = int(np.argmin([abs(energy - expected_energy) for energy in energy_candidates]))
    return phase_candidates[best], energy_candidates[best]


def _run_odmd(
    qubit_hamiltonian: QubitHamiltonian,
    state_preparation: Circuit,
    hadamard_test_generator: HadamardTestGenerator,
) -> QpeResult:
    """Execute ODMD and return the phase-estimation result."""
    odmd = DynamicModeDecomposition(
        hankel_rows=_HANKEL_ROWS,
        hankel_columns=_HANKEL_COLUMNS,
        evolution_time=_EVOLUTION_TIME,
        shots_per_observable=_SHOTS_PER_OBSERVABLE,
    )

    evolution_builder = create("time_evolution_builder", "trotter")
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    executor = create("circuit_executor", "qdk_full_state_simulator", seed=_SEED)

    result = odmd.run(
        qubit_hamiltonian=qubit_hamiltonian,
        state_preparation=state_preparation,
        hadamard_test_generator=hadamard_test_generator,
        circuit_executor=executor,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
    )
    return result


def test_qsharp_odmd_water_reference() -> None:
    """Validate ODMD execution for the Q# state-preparation path."""
    qubit_hamiltonian, qsharp_state_prep, _ = water_odmd_problem("qdk")
    hadamard_test_generator = create("hadamard_test_generator", "qsharp_hadamard_generator")
    result = _run_odmd(qubit_hamiltonian, qsharp_state_prep, hadamard_test_generator)
    resolved_phase, resolved_energy = _resolve_energy_alias(result.phase_fraction, _EXPECTED_ENERGY)

    assert np.isclose(resolved_phase, _EXPECTED_PHASE_FRACTION)
    assert np.isclose(resolved_energy, _EXPECTED_ENERGY)


@pytest.mark.skipif(not (QDK_CHEMISTRY_HAS_QISKIT and QDK_CHEMISTRY_HAS_QISKIT_NATURE), reason="Qiskit not available")
def test_qiskit_odmd_water_reference() -> None:
    """Validate ODMD execution for the Qiskit state-preparation path."""
    qubit_hamiltonian, _, qiskit_state_prep = water_odmd_problem("qiskit")
    assert qiskit_state_prep is not None
    hadamard_test_generator = create("hadamard_test_generator", "qiskit_hadamard_generator")
    result = _run_odmd(qubit_hamiltonian, qiskit_state_prep, hadamard_test_generator)
    resolved_phase, resolved_energy = _resolve_energy_alias(result.phase_fraction, _EXPECTED_ENERGY)

    assert np.isclose(resolved_phase, _EXPECTED_PHASE_FRACTION)
    assert np.isclose(resolved_energy, _EXPECTED_ENERGY)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hankel_rows": 0, "hankel_columns": 12, "evolution_time": 0.1}, "hankel_rows must be"),
        ({"hankel_rows": 24, "hankel_columns": 0, "evolution_time": 0.1}, "hankel_columns must be"),
        ({"hankel_rows": 24, "hankel_columns": 12, "evolution_time": 0.0}, "evolution_time must be"),
        (
            {"hankel_rows": 24, "hankel_columns": 12, "evolution_time": 0.1, "shots_per_observable": 0},
            "shots_per_observable must be",
        ),
    ],
)
def test_odmd_constructor_invalid_inputs(kwargs: dict[str, float | int], message: str) -> None:
    """Ensure DynamicModeDecomposition constructor validates invalid inputs."""
    with pytest.raises(ValueError, match=message):
        DynamicModeDecomposition(**kwargs)
