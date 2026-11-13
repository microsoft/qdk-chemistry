"""Regress logical counts and physical resource optima for each algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import (
    AlgorithmRef,
    CholeskyHamiltonianContainer,
    Circuit,
    Configuration,
    DFTHCHamiltonianContainer,
    FermiHubbardModelHamiltonianDescription,
    Hamiltonian,
    LatticeGeometry,
    MajoranaMapping,
    ModelOrbitals,
    QubitOperator,
    StateVectorContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .helpers import create_dense_wavefunction, create_random_factorized_hamiltonian, create_test_orbitals

try:
    from qdk.qre import PSSPC, LatticeSurgery
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode

    _HAS_QRE = True
except ImportError:
    _HAS_QRE = False

#: Hamiltonian file and alpha/beta electron counts of each active space.
_HAMILTONIANS = {
    "ethylene_4e4o": (Path(__file__).parent / "test_data" / "ethylene_4e4o_2det.hamiltonian.json", 2, 2),
    "f2_10e6o": (Path(__file__).parent / "test_data" / "f2_10e6o.hamiltonian.json", 5, 5),
    "h2_dfthc": (
        Path(__file__).resolve().parents[2] / "examples" / "data" / "h2_dfthc_r2_b2_c1.hamiltonian.json",
        1,
        1,
    ),
}

_NUM_QUERIES = 255


@cache
def _circuit(hamiltonian_name: str, algorithm: str) -> Circuit:
    """Build the phase-estimation circuit of ``algorithm`` for ``hamiltonian_name``."""
    filename, num_alpha, num_beta = _HAMILTONIANS[hamiltonian_name]
    hamiltonian = Hamiltonian.from_json_file(filename)
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
            one_body = hamiltonian.get_one_body_integrals()[0]
            two_body = hamiltonian.get_two_body_integrals()[0]
            eigenvalues, eigenvectors = np.linalg.eigh(two_body.reshape(one_body.size, one_body.size))
            assert eigenvalues.min() >= -1e-10 * eigenvalues.max()
            keep = eigenvalues > 1e-10 * eigenvalues.max()
            vectors = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
            hamiltonian = Hamiltonian(
                CholeskyHamiltonianContainer(
                    one_body, vectors, hamiltonian.get_orbitals(), hamiltonian.get_core_energy(), np.eye(0)
                )
            )
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
@pytest.mark.skipif(not _HAS_QRE, reason="qdk.qre is not available")
def test_resource_estimates(algorithm, num_regression):
    """Compare counts and trace physical optima while pinning only reproducible estimates."""
    resources: dict[str, list[int | float]] = {}
    for hamiltonian_name in _HAMILTONIANS:
        circuit = _circuit(hamiltonian_name, algorithm)
        counts = circuit.estimate().logical_counts
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
            if backend == "trace":
                application = circuit.get_qre_application(use_trace_backend=True)
                assert application.get_trace().compute_qubits == row["numQubits"]
            entries = list(
                circuit.estimate(
                    GateBased(error_rate=1e-4, gate_time=100, measurement_time=500),
                    isa_query=SurfaceCode.q() * RoundBasedFactory.q(),
                    trace_query=PSSPC.q() * LatticeSurgery.q(slow_down_factor=[1.0, 1.5, 2.0, 3.0, 4.0]),
                    max_error=0.01,
                    use_trace_backend=backend == "trace",
                ).physical_estimates
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

    if algorithm == "trotter":
        assert any(
            resources[f"counts_{metric}"] != resources[f"trace_{metric}"]
            for metric in ("min_qubits", "min_qubits_runtime_ns", "fastest_qubits", "fastest_runtime_ns")
        ), "Counts and trace backends unexpectedly produced identical physical estimates"

    num_regression.check(
        resources,
        data_index=list(_HAMILTONIANS),
        basename=algorithm,
        default_tolerance={"atol": 0, "rtol": 0},
    )


@pytest.mark.parametrize("algorithm", ["alias_sampling", "qrom"])
def test_state_preparation_resources(algorithm, num_regression):
    """Pin the logical cost of preparing the same four amplitudes with each algorithm."""
    settings = {"bits_precision": 4} if algorithm == "alias_sampling" else {"rotation_bit_precision": 4}
    circuit = create("state_prep", algorithm, **settings).run(create_dense_wavefunction([0.5, 0.3, 0.7, 0.1]))
    counts = circuit.estimate().logical_counts
    num_regression.check(
        {key: [counts[key]] for key in ("numQubits", "cczCount", "tCount", "rotationCount", "measurementCount")},
        data_index=["four_amplitudes"],
        basename=algorithm,
        default_tolerance={"atol": 0, "rtol": 0},
    )


def test_alias_lookup_resources(qsharp_test_context, qsharp_test_utils, num_regression):
    """Pin the forward, adjoint, and round-trip costs of the conditional alias lookup."""
    lookup_data = [
        [[bool((17 * outer + 5 * inner + bit) % 7 < 3) for bit in range(21)] for inner in range(16)]
        for outer in range(90)
    ]
    num_swap_bits = qsharp_test_context.code.QDKChemistry.Utils.SelectSwap.ComputeOptimalLambda2D(90, 16, 21, True, -1)
    rows = [
        qsharp_test_context.logical_counts(
            qsharp_test_utils.SelectSwapTests.TestSelectSwap2DResourceProbe,
            lookup_data,
            num_swap_bits,
            True,
            *flags,
        )
        for flags in ((True, False), (False, True), (True, True))
    ]
    num_regression.check(
        {
            key: [row[key] for row in rows]
            for key in ("numQubits", "cczCount", "ccixCount", "tCount", "rotationCount", "measurementCount")
        },
        data_index=["forward", "adjoint", "round_trip"],
        basename="alias_lookup",
        default_tolerance={"atol": 0, "rtol": 0},
    )


def test_binary_encoding_resources(test_data_files_path, num_regression):
    """Pin ozone binary-encoding costs with and without negative controls."""
    wavefunction = Wavefunction.from_hdf5_file(
        str(test_data_files_path / "ozone_sparse_ci_wavefunction.wavefunction.h5")
    )
    rows = [
        create("state_prep", "sparse_isometry", binary_encoding=True, include_negative_controls=enabled)
        .run(wavefunction)
        .estimate()
        .logical_counts
        for enabled in (True, False)
    ]
    num_regression.check(
        {
            key: [row[key] for row in rows]
            for key in ("numQubits", "tCount", "rotationCount", "cczCount", "measurementCount")
        },
        data_index=["ozone", "ozone_without_negative_controls"],
        basename="binary_encoding",
        default_tolerance={"atol": 0, "rtol": 0},
    )


def test_controlled_pauli_sequence_resources(num_regression):
    """Pin the logical width, rotation count, and depth of six disjoint controlled Pauli pairs."""
    terms = [ExponentiatedPauliTerm({2 * i: "X", 2 * i + 1: "Y"}, 0.123) for i in range(6)]
    container = PauliProductFormulaContainer(terms, step_reps=2, num_qubits=12, layer_offsets=(0, 6))
    mapper = create("controlled_circuit_mapper", "pauli_sequence", control_indices=[12])
    counts = mapper.run(UnitaryRepresentation(container=container)).estimate().logical_counts
    num_regression.check(
        {key: [counts[key]] for key in ("numQubits", "rotationCount", "rotationDepth")},
        data_index=["six_disjoint_pairs"],
        basename="controlled_pauli_sequence",
        default_tolerance={"atol": 0, "rtol": 0},
    )


def test_controlled_psp_resources(num_regression):
    """Pin controlled PREPARE-SELECT-PREPARE costs for all four existing mapper configurations."""
    rng = np.random.default_rng(0)
    paulis = sorted({"".join(rng.choice(list("IXYZ"), 6)) for _ in range(60)})[:20]
    operator = QubitOperator(pauli_strings=paulis, coefficients=rng.normal(size=len(paulis)))
    unitary = create("hamiltonian_unitary_builder", "lcu", quantum_walk=False).run(operator)
    rows = []
    for settings in (
        {},
        {"prepare": AlgorithmRef("state_prep", "alias_sampling")},
        {"prepare": AlgorithmRef("state_prep", "qrom", rotation_bit_precision=10)},
        {"select_method": "multi_controlled", "prepare": AlgorithmRef("state_prep", "dense_pure_state")},
    ):
        circuit = create("controlled_circuit_mapper", "prepare_select_prepare", **settings).run(unitary)
        counts = circuit.estimate().logical_counts
        rows.append(
            {"declaredNumQubits": circuit.num_qubits, "numQubits": counts["numQubits"], "cczCount": counts["cczCount"]}
        )
    num_regression.check(
        {key: [row[key] for row in rows] for key in ("declaredNumQubits", "numQubits", "cczCount")},
        data_index=["default", "alias_sampling", "qrom", "multi_controlled"],
        basename="controlled_psp",
        default_tolerance={"atol": 0, "rtol": 0},
    )


def test_fe2s2_resources(dataframe_regression):
    """Pin Fe2S2-20 costs for the original and both streamed SELECT implementations."""
    num_orbitals = 20
    factorized = create_random_factorized_hamiltonian(
        num_orbitals=num_orbitals, num_ranks=14, num_bases=15, num_copies=5
    )
    operator = create("qubit_mapper", "sum_of_squares").run(
        Hamiltonian(factorized), MajoranaMapping.jordan_wigner(2 * num_orbitals)
    )
    configuration = Configuration.canonical_hf_configuration(15, 15, num_orbitals)
    reference = Wavefunction(StateVectorContainer(configuration, create_test_orbitals(num_orbitals)))
    state_preparation = create("state_prep", "sparse_isometry").run(reference)
    rows = []
    for settings in (
        {},
        {"max_givens_rotation_batches": 2},
        {"max_givens_rotation_batches": 2, "lookup_method": "select"},
    ):
        builder = create(
            "qpe_circuit_builder",
            "qdk_unary",
            num_queries=10_162,
            circuit_mapper=AlgorithmRef(
                "circuit_mapper",
                "sossa",
                outer_prepare_algorithm=AlgorithmRef("state_prep", "alias_sampling"),
                inner_prepare_algorithm="controlled_alias_sampling",
                select_algorithm="qrom_phase_gradient",
                coefficient_bit_precision=11,
                rotation_bit_precision=15,
                **settings,
            ),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        )
        counts = (
            builder.run(state_preparation=state_preparation, qubit_hamiltonian=operator)[0].estimate().logical_counts
        )
        rows.append({"numQubits": counts["numQubits"], "toffoliCount": counts["cczCount"] + counts["ccixCount"]})
    # Decimal strings keep tolerated columns floating-point when the fixture rereads its CSV.
    dataframe_regression.check(
        pd.DataFrame(
            {
                "numQubits": [row["numQubits"] for row in rows],
                "toffoliCount": [f"{row['toffoliCount']:.1f}" for row in rows],
            },
            index=["fe2s2", "fe2s2_streamed", "fe2s2_streamed_select"],
        ),
        basename="sossa_fe2s2",
        tolerances={"toffoliCount": {"atol": 0, "rtol": 0.01}},
        default_tolerance={"atol": 0, "rtol": 0},
    )


def test_hubbard_plaquette_resources(dataframe_regression):
    """Pin both Hubbard lattice sizes with the existing tolerance for Clifford-boundary rotations."""
    rows = []
    for size in (2, 4):
        num_bits, energy_budget = 10, 0.0051 * size * size
        qpe_budget = 2.0 / 3.0 * energy_budget  # Optimal split: Campbell arXiv:2012.09238v4, App. F.
        builder = create(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=num_bits,
            phase_state="sine",
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder",
                "hubbard_plaquette",
                order=2,
                time=math.tan(math.pi / (2**num_bits + 1)) / qpe_budget,
                power_strategy="rescale",
                target_accuracy=energy_budget - qpe_budget,
            ),
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "hubbard_plaquette"),
        )
        lattice = LatticeGeometry.square(size, size, periodic_x=True, periodic_y=True)
        model = FermiHubbardModelHamiltonianDescription(lattice, t=1.0, u=8.0, epsilon=0.0)
        rows.append(builder.run(identity_state_prep(num_qubits=2 * size * size), model)[0].estimate().logical_counts)
    # Decimal strings keep tolerated columns floating-point when the fixture rereads its CSV.
    dataframe_regression.check(
        pd.DataFrame(
            {
                key: [f"{row.get(key, 0):.1f}" if key.startswith("rotation") else row.get(key, 0) for row in rows]
                for key in (
                    "numQubits",
                    "rotationCount",
                    "rotationDepth",
                    "tCount",
                    "cczCount",
                    "ccixCount",
                    "measurementCount",
                )
            },
            index=["2", "4"],
        ),
        basename="hubbard_plaquette",
        tolerances={key: {"atol": 16, "rtol": 0} for key in ("rotationCount", "rotationDepth")},
        default_tolerance={"atol": 0, "rtol": 0},
    )


@pytest.mark.skipif(not _HAS_QRE, reason="qdk.qre is not available")
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The qdk.qre trace backend draws measurement outcomes from an unseeded random "
    "generator, so circuits that uncompute with measurements trace differently on every run. "
    "Once it is seeded, pin the LCU and SOSSA trace-backend estimates as well.",
)
def test_the_trace_backend_traces_reproducibly():
    """Canary: tracing a SOSSA circuit, which uncomputes with measurements, is reproducible."""
    circuit = _circuit("h2_dfthc", "sossa")

    traces = [circuit.get_qre_application(use_trace_backend=True).get_trace() for _ in range(3)]

    assert len({(trace.num_gates, trace.depth) for trace in traces}) == 1
