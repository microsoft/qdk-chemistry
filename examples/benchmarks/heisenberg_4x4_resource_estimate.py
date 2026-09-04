"""Estimate resources for time evolution of an open 4x4 J1-J2 Heisenberg model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk.qre import estimate
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DrivenQubitHamiltonian,
    LatticeGraph,
    QubitOperator,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian


def create_lattice() -> LatticeGraph:
    """Create the open 4x4 square lattice."""
    return LatticeGraph.square(
        4,
        4,
        periodic_x=False,
        periodic_y=False,
    )


def create_hamiltonian(graph: LatticeGraph) -> QubitOperator:
    r"""Create the J1-J2 model with physical spins S = sigma / 2."""
    j1 = 1.0
    j2 = 0.5
    pauli_couplings = {1: j1 / 4.0, 2: j2 / 4.0}
    return create_heisenberg_hamiltonian(
        graph,
        jx=pauli_couplings,
        jy=pauli_couplings,
        jz=pauli_couplings,
        include_term_groups=False,
    )


def build_time_evolution_circuit(
    hamiltonian: QubitOperator, *, dt: float, total_time: float
) -> Circuit:
    """Build a fourth-order Euler/Trotter time-evolution circuit."""
    zero_hamiltonian = QubitOperator(
        ["I" * hamiltonian.num_qubits],
        np.array([0.0]),
    )
    time_dependent_hamiltonian = DrivenQubitHamiltonian(
        hamiltonian,
        zero_hamiltonian,
        drive=lambda _time: 0.0,
    )
    circuit_builder = create(
        "evolution_circuit_builder",
        "euler",
        evolution_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "trotter",
            order=4,
            num_divisions=1,
        ),
        propagator=AlgorithmRef("propagator", "magnus", order=1),
        circuit_mapper=AlgorithmRef("circuit_mapper", "pauli_sequence"),
        total_time=total_time,
        dt=dt,
    )
    state_prep = identity_state_prep(num_qubits=hamiltonian.num_qubits)
    return circuit_builder.run(time_dependent_hamiltonian, state_prep)


def main() -> None:
    """Build and estimate the 4x4 J1-J2 Heisenberg circuit with QRE."""
    Logger.set_global_level(Logger.LogLevel.off)

    j1 = 1.0
    j2 = 0.5
    dt = 1.0
    total_time = 100.0
    graph = create_lattice()
    shells = graph.nearest_neighbor_shells([1, 2])
    hamiltonian = create_hamiltonian(graph)

    if {shell: len(pairs) for shell, pairs in shells.items()} != {1: 24, 2: 18}:
        raise RuntimeError("The open 4x4 square lattice has unexpected shell counts.")
    if len(hamiltonian.pauli_strings) != 126:
        raise RuntimeError("The J1-J2 Hamiltonian has an unexpected term count.")

    circuit = build_time_evolution_circuit(
        hamiltonian,
        dt=dt,
        total_time=total_time,
    )
    spin_direction = np.array([0.0, 1.0, 0.0])
    circuit = circuit.with_uniform_spin_basis_rotation(
        spin_direction,
        num_qubits=graph.num_sites,
    )

    print("Open 4x4 J1-J2 Heisenberg benchmark")
    print(f"Sites: {graph.num_sites}")
    print(
        f"Geometric shell counts: { {shell: len(pairs) for shell, pairs in shells.items()} }"
    )
    print(f"Physical couplings: J1={j1}, J2={j2}")
    print(f"Hamiltonian Pauli terms: {len(hamiltonian.pauli_strings)}")
    print(f"Spin basis rotation direction: {spin_direction.tolist()}")
    print(f"Evolution: total_time={total_time}, dt={dt}, Trotter order=4, divisions=1")

    application = circuit.get_qre_application()
    architecture = Majorana(error_rate=1.0e-5)
    isa_query = ThreeAux.q() * RoundBasedFactory.q(
        use_cache=True,
        code_query=ThreeAux.q(),
    )
    results = estimate(
        application,
        architecture,
        isa_query,
        max_error=0.01,
        name="Heisenberg 4x4",
    )
    results.add_factory_summary_column()

    print("QRE Pareto-optimal estimates")
    print(results.as_frame().to_string(index=False))


if __name__ == "__main__":
    main()
