"""Regress logical counts and physical resource optima for each algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from functools import cache
from pathlib import Path

import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Configuration,
    DFTHCHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.utils.qsharp import get_qsharp_context

try:
    from qdk.qre import PSSPC, LatticeSurgery, estimate
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode
except ImportError:
    pytest.skip("qdk.qre is not available", allow_module_level=True)

#: Hamiltonian file and alpha/beta electron counts of each active space.
_HAMILTONIANS = {
    "ethylene_4e4o": ("ethylene_4e4o_2det.hamiltonian.json", 2, 2),
    "f2_10e6o": ("f2_10e6o.hamiltonian.json", 5, 5),
    "h2_dfthc": ("h2_dfthc_r2_b2_c1.hamiltonian.json", 1, 1),
}

_NUM_QUERIES = 255


@cache
def _circuit(hamiltonian_name: str, algorithm: str) -> Circuit:
    """Build the phase-estimation circuit of ``algorithm`` for ``hamiltonian_name``."""
    filename, num_alpha, num_beta = _HAMILTONIANS[hamiltonian_name]
    hamiltonian = Hamiltonian.from_json_file(Path(__file__).parent / "test_data" / filename)
    num_orbitals = hamiltonian.get_one_body_integrals()[0].shape[0]
    mapping = MajoranaMapping.jordan_wigner(2 * num_orbitals)
    configuration = Configuration.canonical_hf_configuration(num_alpha, num_beta, num_orbitals)
    wavefunction = Wavefunction(StateVectorContainer(configuration, ModelOrbitals(num_orbitals)))
    state_preparation = create("state_prep", "sparse_isometry").run(wavefunction)

    if algorithm == "trotter":
        operator = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        builder = create(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=8,  # 2^8 - 1 controlled steps match the unary query count.
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
    elif algorithm == "lcu":
        operator = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        builder = create("qpe_circuit_builder", "qdk_unary", num_queries=_NUM_QUERIES)
    else:
        if not isinstance(hamiltonian.get_container(), DFTHCHamiltonianContainer):
            hamiltonian = create("hamiltonian_factorization", "double_factorization").run(hamiltonian)
        operator = create("qubit_mapper", "sum_of_squares").run(hamiltonian, mapping)
        builder = create(
            "qpe_circuit_builder",
            "qdk_unary",
            num_queries=_NUM_QUERIES,
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
            circuit_mapper=AlgorithmRef(
                "circuit_mapper",
                "sossa",
                outer_prepare_algorithm=AlgorithmRef("state_prep", "dense_pure_state"),
                inner_prepare_algorithm="direct",
                select_algorithm="direct",
                coefficient_bit_precision=10,
                rotation_bit_precision=10,
            ),
        )
    return builder.run(state_preparation=state_preparation, qubit_hamiltonian=operator)[0]


@pytest.mark.parametrize("algorithm", ["trotter", "lcu", "sossa"])
def test_resource_estimates(algorithm, num_regression):
    """Pin each system's logical counts and physical optima, excluding nondeterministic trace estimates."""
    resources = {}
    for hamiltonian_name in _HAMILTONIANS:
        circuit = _circuit(hamiltonian_name, algorithm)
        factory = circuit._qsharp_factory
        counts = get_qsharp_context().logical_counts(factory.program, *factory.parameter.values())
        row = {
            key: counts.get(key, 0)
            for key in (
                "numQubits",
                "tCount",
                "rotationCount",
                "rotationDepth",
                "cczCount",
                "ccixCount",
                "measurementCount",
            )
        }
        for backend in ("counts", "trace"):
            application = (
                circuit.get_qre_application(use_trace_backend=False)
                if backend == "counts"
                else circuit.get_qre_application()
            )
            if backend == "trace":
                assert application.get_trace().compute_qubits == row["numQubits"]
            entries = list(
                estimate(
                    application,
                    GateBased(error_rate=1e-4, gate_time=100, measurement_time=500),
                    isa_query=SurfaceCode.q() * RoundBasedFactory.q(),
                    trace_query=PSSPC.q() * LatticeSurgery.q(slow_down_factor=[1.0, 1.5, 2.0, 3.0, 4.0]),
                    max_error=0.01,
                )
            )
            assert entries, f"{hamiltonian_name}/{algorithm}/{backend} has no feasible estimate"
            if backend == "counts" or algorithm == "trotter":
                min_qubits = min(entries, key=lambda entry: (entry.qubits, entry.runtime))
                fastest = min(entries, key=lambda entry: (entry.runtime, entry.qubits))
                row.update(
                    {
                        f"{backend}_min_qubits": min_qubits.qubits,
                        f"{backend}_min_qubits_runtime_ns": min_qubits.runtime,
                        f"{backend}_fastest_qubits": fastest.qubits,
                        f"{backend}_fastest_runtime_ns": fastest.runtime,
                    }
                )
        for key, value in row.items():
            resources.setdefault(key, []).append(value)

    print(f"{algorithm} resource baseline: {resources!r}")
    num_regression.check(
        resources,
        data_index=list(_HAMILTONIANS),
        basename=algorithm,
        default_tolerance={"atol": 0, "rtol": 0},
    )


@pytest.mark.xfail(
    strict=True,
    reason="The qdk.qre trace backend draws measurement outcomes from an unseeded random "
    "generator, so circuits that uncompute with measurements trace differently on every run. "
    "Once it is seeded, pin the LCU and SOSSA trace-backend estimates as well.",
)
def test_the_trace_backend_traces_reproducibly():
    """Canary: tracing a SOSSA circuit, which uncomputes with measurements, is reproducible."""
    circuit = _circuit("h2_dfthc", "sossa")

    traces = [circuit.get_qre_application().get_trace() for _ in range(3)]

    assert len({(trace.num_gates, trace.depth) for trace in traces}) == 1
