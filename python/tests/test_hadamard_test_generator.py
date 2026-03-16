"""Tests for Hadamard test generator algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hadamard_test_generator.hadamard_test_generator import (
    QiskitHadamardGenerator,
    QsharpHadamardGenerator,
)
from qdk_chemistry.data import Circuit, ControlledTimeEvolutionUnitary, Structure
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

try:
    from qdk import qsharp

    _HAS_QSHARP = True
except ImportError:
    _HAS_QSHARP = False

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit, qasm3

_SEED = 42
_SHOTS = 100
_EVOLUTION_TIME = float(np.pi / 48.0)
_OBSERVABLE_POWER = 10


@dataclass(frozen=True)
class HadamardWaterBenchmark:
    """Container for the water benchmark used by Hadamard generator tests."""

    num_system_qubits: int
    state_preparation: Circuit
    ctrl_time_evolution_circuit: Circuit


def _measure_observable(
    *,
    state_preparation: Circuit,
    num_system_qubits: int,
    ctrl_time_evolution_circuit: Circuit,
    generator: QiskitHadamardGenerator | QsharpHadamardGenerator,
    test_basis: str = "X",
    shots: int = _SHOTS,
) -> float:
    """Measure Re(<psi|U|psi>) with a Hadamard test circuit and simulator."""
    circuit = generator.run(
        state_preparation,
        num_system_qubits,
        ctrl_time_evolution_circuit,
        test_basis=test_basis,
    )
    simulator = create("circuit_executor", "qdk_full_state_simulator", seed=_SEED)
    result = simulator.run(circuit, shots=shots)
    counts = result.bitstring_counts
    return (counts.get("0", 0) - counts.get("1", 0)) / shots


@pytest.fixture(scope="module")
def water_hadamard_benchmark() -> HadamardWaterBenchmark:
    """Construct water-system state prep and controlled evolution circuits once per module."""
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

    scf_solver = create("scf_solver")
    _, scf_wfn = scf_solver.run(
        structure,
        charge=0,
        spin_multiplicity=1,
        basis_or_guess="cc-pvdz",
    )

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
    state_preparation = state_prep_builder.run(active_wfn)

    evolution_builder = create("time_evolution_builder", "trotter")
    time_evolution = evolution_builder.run(qubit_hamiltonian, _EVOLUTION_TIME)
    controlled_evolution = ControlledTimeEvolutionUnitary(time_evolution_unitary=time_evolution, control_indices=[0])

    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    circuit_mapper.settings().update("power", _OBSERVABLE_POWER)
    ctrl_time_evolution_circuit = circuit_mapper._run_impl(controlled_evolution=controlled_evolution)

    return HadamardWaterBenchmark(
        num_system_qubits=qubit_hamiltonian.num_qubits,
        state_preparation=state_preparation,
        ctrl_time_evolution_circuit=ctrl_time_evolution_circuit,
    )


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
def test_qiskit_hadamard_generator_measures_water_observable(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Qiskit Hadamard generator reproduces the reference observable for water."""
    observable_value = _measure_observable(
        state_preparation=water_hadamard_benchmark.state_preparation,
        num_system_qubits=water_hadamard_benchmark.num_system_qubits,
        ctrl_time_evolution_circuit=water_hadamard_benchmark.ctrl_time_evolution_circuit,
        generator=QiskitHadamardGenerator(),
    )

    assert np.isclose(observable_value, 0.34, atol=1e-12)


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qsharp_hadamard_generator_measures_water_observable(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Q# Hadamard generator reproduces the reference observable for water."""
    observable_value = _measure_observable(
        state_preparation=water_hadamard_benchmark.state_preparation,
        num_system_qubits=water_hadamard_benchmark.num_system_qubits,
        ctrl_time_evolution_circuit=water_hadamard_benchmark.ctrl_time_evolution_circuit,
        generator=QsharpHadamardGenerator(),
    )

    assert np.isclose(observable_value, 0.34, atol=1e-12)


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
def test_qiskit_hadamard_generator_measures_water_observable_in_y_basis(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Qiskit Hadamard generator reproduces the Y-basis reference observable for water."""
    observable_value = _measure_observable(
        state_preparation=water_hadamard_benchmark.state_preparation,
        num_system_qubits=water_hadamard_benchmark.num_system_qubits,
        ctrl_time_evolution_circuit=water_hadamard_benchmark.ctrl_time_evolution_circuit,
        generator=QiskitHadamardGenerator(),
        test_basis="Y",
    )

    assert np.isclose(observable_value, 0.98, atol=1e-12)


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qsharp_hadamard_generator_measures_water_observable_in_y_basis(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Q# Hadamard generator reproduces the Y-basis reference observable for water."""
    observable_value = _measure_observable(
        state_preparation=water_hadamard_benchmark.state_preparation,
        num_system_qubits=water_hadamard_benchmark.num_system_qubits,
        ctrl_time_evolution_circuit=water_hadamard_benchmark.ctrl_time_evolution_circuit,
        generator=QsharpHadamardGenerator(),
        test_basis="Y",
    )

    assert np.isclose(observable_value, 0.98, atol=1e-12)


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
def test_qiskit_hadamard_generator_rejects_invalid_test_basis() -> None:
    """Qiskit generator rejects unsupported Hadamard measurement bases."""
    state_prep_qc = QuantumCircuit(1, name="state")
    ctrl_evol_qc = QuantumCircuit(2, name="ctrl")
    ctrl_evol_qc.cx(0, 1)

    with pytest.raises(ValueError, match="test_basis"):
        QiskitHadamardGenerator().run(
            Circuit(qasm=qasm3.dumps(state_prep_qc)),
            1,
            Circuit(qasm=qasm3.dumps(ctrl_evol_qc)),
            test_basis="BAD",
        )


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qsharp_hadamard_generator_rejects_invalid_test_basis() -> None:
    """Q# generator rejects unsupported Hadamard measurement bases."""
    with pytest.raises(ValueError, match="test_basis"):
        QsharpHadamardGenerator().run(  # type: ignore[arg-type]
            object(),
            1,
            object(),
            test_basis="BAD",
        )


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")
def test_qiskit_hadamard_generator_rejects_incompatible_input_circuits() -> None:
    """Qiskit generator raises errors when inputs cannot produce Qiskit circuits."""
    generator = QiskitHadamardGenerator()

    with pytest.raises(ValueError, match="state_preparation"):
        generator.run(  # type: ignore[arg-type]
            object(),
            1,
            Circuit(qasm=qasm3.dumps(QuantumCircuit(2))),
            test_basis="X",
        )

    with pytest.raises(ValueError, match="ctrl_time_evol_unitary_circuit"):
        generator.run(  # type: ignore[arg-type]
            Circuit(qasm=qasm3.dumps(QuantumCircuit(1))),
            1,
            object(),
            test_basis="X",
        )


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qsharp_hadamard_generator_rejects_incompatible_input_circuits(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Q# generator raises errors when input circuits do not expose Q# operations."""
    generator = QsharpHadamardGenerator()
    bad_state_preparation_circuit = Circuit(qasm=water_hadamard_benchmark.state_preparation.get_qasm())
    bad_ctrl_time_evolution_circuit = Circuit(qasm=water_hadamard_benchmark.ctrl_time_evolution_circuit.get_qasm())

    with pytest.raises(ValueError, match="state_preparation"):
        generator.run(bad_state_preparation_circuit, 1, water_hadamard_benchmark.ctrl_time_evolution_circuit, "X")

    with pytest.raises(ValueError, match="ctrl_time_evol_unitary_circuit"):
        generator.run(
            water_hadamard_benchmark.state_preparation,
            water_hadamard_benchmark.num_system_qubits,
            bad_ctrl_time_evolution_circuit,
            "X",
        )
