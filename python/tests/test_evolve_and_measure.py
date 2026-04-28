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
from qdk_chemistry.algorithms.time_evolution.measure_simulation import EvolveAndMeasure
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_AER, QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME


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

    combined = algo._prepend_state_prep_circuit(state_prep, evolution, num_qubits=1)

    assert combined._qsharp_factory is not None
    assert combined._qsharp_factory.program == "sequential-circuit"
    assert combined._qsharp_factory.parameter == {
        "first": state_prep_op,
        "second": evolution_op,
        "targets": [0],
    }
    assert combined._qsharp_op == ("sequential-op", state_prep_op, evolution_op)
    assert combined.encoding == "jordan-wigner"


def test_prepend_state_prep_circuit_requires_qsharp_operations() -> None:
    """The helper should fail fast when either circuit lacks a Q# operation handle."""
    algo = EvolveAndMeasure()
    state_prep = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n")
    evolution = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")

    with pytest.raises(RuntimeError, match="requires Q# operations"):
        algo._prepend_state_prep_circuit(state_prep, evolution, num_qubits=1)


def test_evolve_and_measure_eigenvalue_remains_constant() -> None:
    """Run an example EvolveAndMeasure workflow."""
    hamiltonian_p = QubitHamiltonian(["ZZ"], np.array([1.0]))
    hamiltonian_m = QubitHamiltonian(["ZZ"], np.array([-1.0]))

    steps = 100
    hamiltonians = [hamiltonian_p, hamiltonian_m] * (steps // 2)
    time_steps = [float(t + 1) for t in range(steps)]
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EvolveAndMeasure()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("time_evolution_builder", "trotter", num_divisions=1, order=1, optimize_term_ordering=True),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )

    measurements = algo.run(
        hamiltonians,
        times=time_steps,
        observables=[observable],
        shots=1024,
    )

    for measurement in measurements:
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.2)


@pytest.mark.skipif(
    not QDK_CHEMISTRY_HAS_QISKIT_AER or not QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME,
    reason="Qiskit Aer or IBM Runtime not available",
)
def test_evolve_and_measure_with_device_backend() -> None:
    """Run EvolveAndMeasure with a device_backend_name string."""
    hamiltonian = QubitHamiltonian(["ZZ"], np.array([1.0]))
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EvolveAndMeasure()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("time_evolution_builder", "trotter", num_divisions=1, order=1, optimize_term_ordering=True),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qiskit_aer_simulator"),
    )

    measurements = algo.run(
        [hamiltonian],
        times=[1.0],
        observables=[observable],
        shots=1024,
        device_backend_name="fake_manila",
    )

    for measurement in measurements:
        # With device noise the expectation value should still be close to 1.0
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.5)
