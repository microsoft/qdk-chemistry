"""Regress logical counts and dollar-cost estimates against pytest-regressions baselines."""

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
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.utils.qsharp import get_qsharp_context

try:
    from qdk.qre import PSSPC, LatticeSurgery, estimate
    from qdk.qre.dollar_cost import DollarCostModelFromSpec
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode
except ImportError:
    pytest.skip("qdk.qre with dollar_cost is not available", allow_module_level=True)

_TEST_DATA = Path(__file__).resolve().parent / "test_data"

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
    hamiltonian = Hamiltonian.from_json_file(_TEST_DATA / filename)
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
        if not isinstance(hamiltonian.get_container(), FactorizedHamiltonianContainer):
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


def _logical_counts(circuit: Circuit) -> dict[str, int]:
    """Return the pinned subset of the circuit's logical counts."""
    factory = circuit._qsharp_factory
    counts = get_qsharp_context().logical_counts(factory.program, *factory.parameter.values())
    return {
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


def _cheapest_estimate(application) -> dict[str, float]:
    """Estimate as in the ``dollar_cost`` sample and return its cheapest feasible point."""
    results = estimate(
        application,
        GateBased(error_rate=1e-4, gate_time=100, measurement_time=500),
        isa_query=SurfaceCode.q() * RoundBasedFactory.q(),
        trace_query=PSSPC.q() * LatticeSurgery.q(slow_down_factor=[1.0, 1.5, 2.0, 3.0, 4.0]),
        max_error=0.01,
    )
    cost_model = DollarCostModelFromSpec(str(_TEST_DATA / "example_cost_spec.json"))
    entries = [
        {
            "qubits": entry.qubits,
            "runtime_ns": entry.runtime,
            "usd": cost_model.cost_usd(qubits=entry.qubits, runtime_nanos=entry.runtime),
        }
        for entry in results
    ]
    assert entries, "the estimate has no feasible point within the error budget"
    return min(entries, key=lambda entry: (entry["usd"], entry["runtime_ns"], entry["qubits"]))


_CASES = [
    pytest.param(name, algorithm, id=f"{name}-{algorithm}")
    for name in _HAMILTONIANS
    for algorithm in ("trotter", "lcu", "sossa")
]


@pytest.mark.parametrize(("hamiltonian_name", "algorithm"), _CASES)
def test_logical_counts(hamiltonian_name, algorithm, data_regression):
    """The circuit's logical counts must match the pinned values."""
    data_regression.check(
        _logical_counts(_circuit(hamiltonian_name, algorithm)),
        basename=f"logical_counts_{hamiltonian_name}_{algorithm}",
    )


@pytest.mark.parametrize(("hamiltonian_name", "algorithm"), _CASES)
def test_physical_estimate_and_dollar_cost(hamiltonian_name, algorithm, num_regression):
    """The seeded counts backend must reproduce the cheapest point's qubits, runtime, and USD cost."""
    application = _circuit(hamiltonian_name, algorithm).get_qre_application(use_trace_backend=False)

    cheapest = _cheapest_estimate(application)

    num_regression.check(
        cheapest,
        basename=f"physical_estimate_and_dollar_cost_{hamiltonian_name}_{algorithm}",
        default_tolerance={"atol": 0, "rtol": 0},
        tolerances={"usd": {"atol": 1e-12, "rtol": 1e-6}},
    )


@pytest.mark.parametrize(("hamiltonian_name", "algorithm"), _CASES)
def test_the_default_trace_backend_estimates_the_circuit(hamiltonian_name, algorithm, num_regression):
    """Trace every circuit on its logical qubits, pinning only the reproducible Trotter estimates."""
    circuit = _circuit(hamiltonian_name, algorithm)
    application = circuit.get_qre_application()

    trace = application.get_trace()
    cheapest = _cheapest_estimate(application)

    assert trace.compute_qubits == _logical_counts(circuit)["numQubits"]
    if algorithm == "trotter":
        num_regression.check(
            cheapest,
            basename=f"default_trace_backend_{hamiltonian_name}",
            default_tolerance={"atol": 0, "rtol": 0},
            tolerances={"usd": {"atol": 1e-12, "rtol": 1e-6}},
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
