"""Estimate resources for time evolution of an open 4x4 Kitaev patch."""

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
from qdk_chemistry.utils.model_hamiltonians import create_kitaev_hamiltonian


def create_lattice() -> LatticeGraph:
    """Create the open 4x4 complete-plaquette patch in the default orientation."""
    return LatticeGraph.honeycomb_plaquettes(
        4,
        4,
        periodic_x=False,
        periodic_y=False,
    )


def create_hamiltonian(graph: LatticeGraph) -> QubitOperator:
    """Create the extended Kitaev Hamiltonian used by the smaller benchmarks."""
    crystallographic_transform = np.array(
        [
            [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
            [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        ]
    )
    kitaev_couplings = {1: -13.3, 2: -0.67, 3: 0.1}  # meV
    return create_kitaev_hamiltonian(
        graph,
        kx=kitaev_couplings,
        ky=kitaev_couplings,
        kz=kitaev_couplings,
        j={1: -1.3, 3: 1.0},  # meV; omitted J2 is zero
        gamma=9.4,  # meV
        gamma_prime=-2.3,  # meV
        magnetic_field_abc=(0.0, 10.0, 0.0),  # tesla; H_b = 10 T
        g_factors_abc=(2.3, 2.3, 1.3),
        bohr_magneton=5.988e-2,  # meV/T
        crystallographic_transform=crystallographic_transform,
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
    """Build and estimate the 4x4 time-evolution circuit with QRE."""
    Logger.set_global_level(Logger.LogLevel.off)

    dt = 1.0
    total_time = 100.0
    graph = create_lattice()
    hamiltonian = create_hamiltonian(graph)
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

    print("Open 4x4 complete-plaquette Kitaev benchmark")
    print(f"Sites: {graph.num_sites}")
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
        name="Kitaev 4x4",
    )
    results.add_factory_summary_column()

    print("QRE Pareto-optimal estimates")
    print(results.as_frame().to_string(index=False))


if __name__ == "__main__":
    main()
