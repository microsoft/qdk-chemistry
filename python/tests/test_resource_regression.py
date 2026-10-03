"""Resource-estimation regression for Trotter, LCU, and SOSSA phase estimation.

Each case builds the phase-estimation circuit of a Hamiltonian in ``test_data`` and pins
its logical counts together with the cheapest point of a ``qdk.qre`` estimate. The
estimate is configured as in the QDK ``samples/qre/dollar_cost.ipynb`` sample and priced
with that sample's ``example_cost_spec.json``. A changed value is a resource regression,
or an improvement, to review and re-pin.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from functools import cache
from pathlib import Path

import pytest
from qdk.qsharp import QSharpError

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

#: The default cost specification of the QDK ``dollar_cost`` sample.
_COST_SPEC = _TEST_DATA / "example_cost_spec.json"

#: Hamiltonian file and alpha/beta electron counts of each active space.
_HAMILTONIANS = {
    "ethylene_4e4o": ("ethylene_4e4o_2det.hamiltonian.json", 2, 2),
    "f2_10e6o": ("f2_10e6o.hamiltonian.json", 5, 5),
    "h2_dfthc": ("h2_dfthc_r2_b2_c1.hamiltonian.json", 1, 1),
}

#: Standard QPE phase bits for Trotter; 2^8 - 1 controlled steps match the unary query count.
_NUM_PHASE_BITS = 8
_NUM_QUERIES = 255
_TROTTER_TIME = 1.0

_PINNED_LOGICAL_COUNT_KEYS = (
    "numQubits",
    "tCount",
    "rotationCount",
    "rotationDepth",
    "cczCount",
    "ccixCount",
    "measurementCount",
)

#: Pinned logical counts and the cheapest estimate (physical qubits, runtime in ns, USD).
_PINNED: dict[tuple[str, str], dict] = {
    ("ethylene_4e4o", "trotter"): {
        "logical_counts": {
            "numQubits": 16,
            "tCount": 21,
            "rotationCount": 94158,
            "rotationDepth": 86979,
            "cczCount": 0,
            "ccixCount": 0,
            "measurementCount": 8,
        },
        "qubits": 28585,
        "runtime_ns": 14938281000,
        "usd": 2.9,
    },
    ("ethylene_4e4o", "lcu"): {
        "logical_counts": {
            "numQubits": 38,
            "tCount": 21,
            "rotationCount": 130250,
            "rotationDepth": 129966,
            "cczCount": 2558421,
            "ccixCount": 0,
            "measurementCount": 2054,
        },
        "qubits": 70035,
        "runtime_ns": 91322784000,
        "usd": 41.72,
    },
    ("ethylene_4e4o", "sossa"): {
        "logical_counts": {
            "numQubits": 54,
            "tCount": 51021,
            "rotationCount": 401570,
            "rotationDepth": 397970,
            "cczCount": 3220402,
            "ccixCount": 0,
            "measurementCount": 72180,
        },
        "qubits": 72450,
        "runtime_ns": 163308393000,
        "usd": 77.11,
    },
    ("f2_10e6o", "trotter"): {
        "logical_counts": {
            "numQubits": 20,
            "tCount": 21,
            "rotationCount": 195138,
            "rotationDepth": 178779,
            "cczCount": 0,
            "ccixCount": 0,
            "measurementCount": 8,
        },
        "qubits": 30034,
        "runtime_ns": 33936723000,
        "usd": 6.89,
    },
    ("f2_10e6o", "lcu"): {
        "logical_counts": {
            "numQubits": 44,
            "tCount": 21,
            "rotationCount": 260810,
            "rotationDepth": 260528,
            "cczCount": 7982017,
            "ccixCount": 0,
            "measurementCount": 2310,
        },
        "qubits": 71108,
        "runtime_ns": 320751464000,
        "usd": 148.72,
    },
    ("f2_10e6o", "sossa"): {
        "logical_counts": {
            "numQubits": 67,
            "tCount": 69381,
            "rotationCount": 1554170,
            "rotationDepth": 1545470,
            "cczCount": 14221613,
            "ccixCount": 0,
            "measurementCount": 129046,
        },
        "qubits": 80179,
        "runtime_ns": 828595196000,
        "usd": 431.84,
    },
    ("h2_dfthc", "trotter"): {
        "logical_counts": {
            "numQubits": 12,
            "tCount": 21,
            "rotationCount": 13578,
            "rotationDepth": 11499,
            "cczCount": 0,
            "ccixCount": 0,
            "measurementCount": 8,
        },
        "qubits": 32495,
        "runtime_ns": 1383137000,
        "usd": 0.3,
    },
    ("h2_dfthc", "lcu"): {
        "logical_counts": {
            "numQubits": 28,
            "tCount": 21,
            "rotationCount": 16010,
            "rotationDepth": 15726,
            "cczCount": 126993,
            "ccixCount": 0,
            "measurementCount": 1286,
        },
        "qubits": 34872,
        "runtime_ns": 5849208000,
        "usd": 1.37,
    },
    ("h2_dfthc", "sossa"): {
        "logical_counts": {
            "numQubits": 35,
            "tCount": 8181,
            "rotationCount": 28250,
            "rotationDepth": 28220,
            "cczCount": 108889,
            "ccixCount": 0,
            "measurementCount": 11487,
        },
        "qubits": 35508,
        "runtime_ns": 7688925000,
        "usd": 1.83,
    },
}


def _hartree_fock_state_prep(num_orbitals: int, num_alpha: int, num_beta: int) -> Circuit:
    configuration = Configuration.canonical_hf_configuration(num_alpha, num_beta, num_orbitals)
    wavefunction = Wavefunction(StateVectorContainer(configuration, ModelOrbitals(num_orbitals)))
    return create("state_prep", "sparse_isometry").run(wavefunction)


@cache
def _circuit(hamiltonian_name: str, algorithm: str) -> Circuit:
    """Build the phase-estimation circuit of ``algorithm`` for ``hamiltonian_name``."""
    filename, num_alpha, num_beta = _HAMILTONIANS[hamiltonian_name]
    hamiltonian = Hamiltonian.from_json_file(_TEST_DATA / filename)
    num_orbitals = hamiltonian.get_one_body_integrals()[0].shape[0]
    mapping = MajoranaMapping.jordan_wigner(2 * num_orbitals)
    state_preparation = _hartree_fock_state_prep(num_orbitals, num_alpha, num_beta)

    if algorithm == "trotter":
        operator = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
        builder = create(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=_NUM_PHASE_BITS,
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=_TROTTER_TIME),
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
    factory = circuit._qsharp_factory
    counts = get_qsharp_context().logical_counts(factory.program, *factory.parameter.values())
    return {key: counts.get(key, 0) for key in _PINNED_LOGICAL_COUNT_KEYS}


def _cheapest_estimate(application) -> dict[str, float]:
    """Estimate as in the ``dollar_cost`` sample and return its cheapest feasible point."""
    results = estimate(
        application,
        GateBased(error_rate=1e-4, gate_time=100, measurement_time=500),
        isa_query=SurfaceCode.q() * RoundBasedFactory.q(),
        trace_query=PSSPC.q() * LatticeSurgery.q(slow_down_factor=[1.0, 1.5, 2.0, 3.0, 4.0]),
        max_error=0.01,
    )
    cost_model = DollarCostModelFromSpec(str(_COST_SPEC))
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
def test_logical_counts(hamiltonian_name, algorithm):
    """The circuit's logical counts must match the pinned values."""
    pinned = _PINNED[(hamiltonian_name, algorithm)]

    assert _logical_counts(_circuit(hamiltonian_name, algorithm)) == pinned["logical_counts"]


@pytest.mark.parametrize(("hamiltonian_name", "algorithm"), _CASES)
def test_physical_estimate_and_dollar_cost(hamiltonian_name, algorithm):
    """The cheapest point's physical qubits, runtime, and USD cost must match the pinned values.

    The trace backend cannot evaluate these circuits yet (see the canary below), so the
    estimate runs on the logical-counts backend.
    """
    pinned = _PINNED[(hamiltonian_name, algorithm)]
    application = _circuit(hamiltonian_name, algorithm).get_qre_application(use_trace_backend=False)

    cheapest = _cheapest_estimate(application)

    assert cheapest["qubits"] == pinned["qubits"]
    assert cheapest["runtime_ns"] == pinned["runtime_ns"]
    assert cheapest["usd"] == pytest.approx(pinned["usd"])


@pytest.mark.xfail(
    raises=QSharpError,
    strict=True,
    reason="The qdk.qre trace backend has no IsResourceEstimating intrinsic, which Loop and the "
    "unary QPE schedule branch on. Once it does, pin the trace-backend estimates and drop "
    "use_trace_backend=False above.",
)
@pytest.mark.parametrize("algorithm", ["trotter", "lcu", "sossa"])
def test_the_default_trace_backend_estimates_the_circuit(algorithm):
    """Canary: the default ``get_qre_application`` uses the trace backend."""
    _cheapest_estimate(_circuit("h2_dfthc", algorithm).get_qre_application())
