"""Tests for observable dynamic mode decomposition phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.dynamic_mode_decomposition import DynamicModeDecomposition
from qdk_chemistry.data import Circuit, QpeResult, QubitHamiltonian, Structure
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT, QDK_CHEMISTRY_HAS_QISKIT_NATURE
from qdk_chemistry.utils.phase import energy_from_phase

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    qpe_phase_fraction_tolerance,
)

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import qasm3


pytestmark = pytest.mark.skipif(
    not (QDK_CHEMISTRY_HAS_QISKIT and QDK_CHEMISTRY_HAS_QISKIT_NATURE),
    reason="ODMD water tests require Qiskit dependencies",
)

_SEED = 42
_EVOLUTION_TIME = float(np.pi / 48.0)
_HANKEL_ROWS = 24
_INITIAL_HANKEL_COLUMNS = 12
_SHOTS_PER_OBSERVABLE = 500
_EIGEN_CONVERGE_TOL = 1e-3

# References captured from test_ODMD/tests/result_*_shot500_tol1e-3.log.
_EXPECTED_PHASE_FRACTION = 0.9800011077309474
_EXPECTED_ENERGY = -1.919893657829055


@pytest.fixture(scope="module")
def water_odmd_problem() -> tuple[QubitHamiltonian, Circuit, Circuit]:
    """Build the active-space water ODMD benchmark inputs once per module."""
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

    qubit_mapper = create("qubit_mapper", algorithm_name="qiskit", encoding="jordan-wigner")
    qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)

    state_prep_builder = create("state_prep", algorithm_name="sparse_isometry_gf2x")
    state_prep = state_prep_builder.run(active_wfn)

    qsharp_state_prep = state_prep
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
    *,
    max_hankel_columns: int = 200,
) -> tuple[QpeResult, DynamicModeDecomposition]:
    """Execute ODMD and return both result and algorithm instance."""
    odmd = DynamicModeDecomposition(
        hankel_rows=_HANKEL_ROWS,
        initial_hankel_columns=_INITIAL_HANKEL_COLUMNS,
        time_step=_EVOLUTION_TIME,
        eigen_converge_tol=_EIGEN_CONVERGE_TOL,
        shots_per_observable=_SHOTS_PER_OBSERVABLE,
        max_hankel_columns=max_hankel_columns,
    )

    evolution_builder = create("time_evolution_builder", "trotter")
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    executor = create("circuit_executor", "qdk_full_state_simulator", seed=_SEED)

    result = odmd.run(
        qubit_hamiltonian=qubit_hamiltonian,
        state_preparation=state_preparation,
        circuit_executor=executor,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
    )
    return result, odmd


def test_qsharp_odmd_water_reference(water_odmd_problem: tuple[QubitHamiltonian, Circuit, Circuit]) -> None:
    """Validate ODMD reference phase and energy for the Q# state-preparation path."""
    qubit_hamiltonian, qsharp_state_prep, _ = water_odmd_problem
    result, odmd = _run_odmd(qubit_hamiltonian, qsharp_state_prep)
    resolved_phase, resolved_energy = _resolve_energy_alias(result.phase_fraction, _EXPECTED_ENERGY)

    assert np.isclose(
        result.phase_fraction,
        _EXPECTED_PHASE_FRACTION,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_phase,
        _EXPECTED_PHASE_FRACTION,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_energy,
        _EXPECTED_ENERGY,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert odmd.is_converged() is True
    assert result.metadata is not None
    assert result.metadata.get("converged") is True
    assert result.metadata.get("stop_reason") == "converged"


def test_qiskit_odmd_water_reference(water_odmd_problem: tuple[QubitHamiltonian, Circuit, Circuit]) -> None:
    """Validate ODMD reference phase and energy for the Qiskit state-preparation path."""
    qubit_hamiltonian, _, qiskit_state_prep = water_odmd_problem
    result, odmd = _run_odmd(qubit_hamiltonian, qiskit_state_prep)
    resolved_phase, resolved_energy = _resolve_energy_alias(result.phase_fraction, _EXPECTED_ENERGY)

    assert np.isclose(
        result.phase_fraction,
        _EXPECTED_PHASE_FRACTION,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_phase,
        _EXPECTED_PHASE_FRACTION,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_energy,
        _EXPECTED_ENERGY,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert odmd.is_converged() is True
    assert result.metadata is not None
    assert result.metadata.get("converged") is True
    assert result.metadata.get("stop_reason") == "converged"


def test_qsharp_odmd_max_hankel_limit(water_odmd_problem: tuple[QubitHamiltonian, Circuit, Circuit]) -> None:
    """Verify ODMD reports non-convergence when max_hankel_columns is too small."""
    qubit_hamiltonian, qsharp_state_prep, _ = water_odmd_problem
    result, odmd = _run_odmd(
        qubit_hamiltonian,
        qsharp_state_prep,
        max_hankel_columns=_INITIAL_HANKEL_COLUMNS + 1,
    )

    assert odmd.is_converged() is False
    assert result.metadata is not None
    assert result.metadata.get("converged") is False
    assert result.metadata.get("stop_reason") == "max_hankel_columns_reached"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hankel_rows": 0, "initial_hankel_columns": 12, "time_step": 0.1}, "hankel_rows must be"),
        (
            {"hankel_rows": 24, "initial_hankel_columns": 1, "time_step": 0.1},
            "initial_hankel_columns must be larger than 1",
        ),
        (
            {"hankel_rows": 24, "initial_hankel_columns": 8, "time_step": 0.1, "max_hankel_columns": 7},
            "initial_hankel_columns must be no more than max_hankel_columns",
        ),
        ({"hankel_rows": 24, "initial_hankel_columns": 12, "time_step": 0.0}, "time_step must be"),
        (
            {"hankel_rows": 24, "initial_hankel_columns": 12, "time_step": 0.1, "eigen_converge_tol": 0.0},
            "eigen_converge_tol must be",
        ),
        (
            {"hankel_rows": 24, "initial_hankel_columns": 12, "time_step": 0.1, "shots_per_observable": 0},
            "shots_per_observable must be",
        ),
    ],
)
def test_odmd_constructor_invalid_inputs(kwargs: dict[str, float | int], message: str) -> None:
    """Ensure DynamicModeDecomposition constructor validates invalid inputs."""
    with pytest.raises(ValueError, match=message):
        DynamicModeDecomposition(**kwargs)
