"""Tests for EvolveAndMeasure state-preparation composition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import qdk_chemistry.algorithms.time_evolution.measure_simulation.base as measure_base
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter
from qdk_chemistry.algorithms.time_evolution.circuit_mapper import PauliSequenceMapper
from qdk_chemistry.algorithms.time_evolution.measure_simulation import EvolveAndMeasure
from qdk_chemistry.data import Circuit, QubitHamiltonian


def test_prepend_state_prep_circuit_composes_qsharp_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should compose state preparation and evolution through Q# operations."""
    algo = EvolveAndMeasure()
    state_prep_op = object()
    evolution_op = object()

    monkeypatch.setattr(
        measure_base,
        "QSHARP_UTILS",
        SimpleNamespace(
            CircuitComposition=SimpleNamespace(
                MakeSequentialCircuit="sequential-circuit",
                MakeSequentialOp=lambda first, second: ("sequential-op", first, second),
            )
        ),
    )
    monkeypatch.setattr(
        measure_base.qsharp,
        "circuit",
        lambda *args: ("qsharp-circuit", args),
    )
    monkeypatch.setattr(
        measure_base.qsharp,
        "compile",
        lambda *args: ("qir", args),
    )

    state_prep = Circuit(
        qir='attributes #0 = { "required_num_qubits"="1" }',
        qsharp_op=state_prep_op,
        encoding="jordan-wigner",
    )
    evolution = Circuit(
        qir='attributes #0 = { "required_num_qubits"="1" }',
        qsharp_op=evolution_op,
        encoding="jordan-wigner",
    )

    combined = algo._prepend_state_prep_circuit(state_prep, evolution)

    assert combined.qsharp == (
        "qsharp-circuit",
        ("sequential-circuit", state_prep_op, evolution_op, [0]),
    )
    assert combined.qir == (
        "qir",
        ("sequential-circuit", state_prep_op, evolution_op, [0]),
    )
    assert combined._qsharp_op == ("sequential-op", state_prep_op, evolution_op)
    assert combined.encoding == "jordan-wigner"


def test_prepend_state_prep_circuit_rejects_mismatched_qubit_counts() -> None:
    """The helper should reject incompatible circuit widths before composing."""
    algo = EvolveAndMeasure()
    state_prep = Circuit(qir='attributes #0 = { "required_num_qubits"="1" }', qsharp_op=lambda _: None)
    evolution = Circuit(qir='attributes #0 = { "required_num_qubits"="2" }', qsharp_op=lambda _: None)

    with pytest.raises(ValueError, match="same number of qubits"):
        algo._prepend_state_prep_circuit(state_prep, evolution)


def test_prepend_state_prep_circuit_requires_qsharp_operations() -> None:
    """The helper should fail fast when either circuit lacks a Q# operation handle."""
    algo = EvolveAndMeasure()
    state_prep = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n")
    evolution = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")

    with pytest.raises(RuntimeError, match="requires Q# operations"):
        algo._prepend_state_prep_circuit(state_prep, evolution)


def test_evolve_and_measure_eigenvalue_remains_constant() -> None:
    """Run an example EvolveAndMeasure workflow."""
    hamiltonian_p = QubitHamiltonian(["ZZ"], np.array([1.0]))
    hamiltonian_m = QubitHamiltonian(["ZZ"], np.array([-1.0]))

    steps = 100
    hamiltonians = [hamiltonian_p, hamiltonian_m] * (steps // 2)
    time_steps = [float(t + 1) for t in range(steps)]
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    evolution_builder = Trotter(num_divisions=1, order=1, optimize_term_ordering=True)
    algo = EvolveAndMeasure()
    mapper = PauliSequenceMapper()
    circuit_executor = create("energy_estimator", "qiskit_aer_simulator")

    measurements = algo.run(
        hamiltonians,
        times=time_steps,
        observables=[observable],
        evolution_builder=evolution_builder,
        circuit_mapper=mapper,
        circuit_executor=circuit_executor,
        shots=1024,
    )

    for measurement in measurements:
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.2)
