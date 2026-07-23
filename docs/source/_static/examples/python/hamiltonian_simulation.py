"""HamiltonianSimulation usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import registry

sim = registry.create("hamiltonian_simulation", "euler_integrator")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
from qdk_chemistry.data import AlgorithmRef

# Configure the nested evolution circuit builder
sim.settings().set(
    "evolution_circuit_builder",
    AlgorithmRef(
        "evolution_circuit_builder",
        "euler",
        total_time=1.0,
        dt=0.1,
        propagator=AlgorithmRef("propagator", "magnus", order=1),
        evolution_builder=AlgorithmRef(
            "hamiltonian_unitary_builder", "trotter", order=4, num_divisions=2
        ),
    ),
)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, DrivenQubitHamiltonian, LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian

# 1. Build a driven Hamiltonian: H(t) = H0 + sin(2πt)·H1
lattice = LatticeGraph.chain(4)
h0 = create_ising_hamiltonian(lattice, j=1.0, h=0.0)
h1 = create_ising_hamiltonian(lattice, j=0.0, h=0.5)
td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=lambda t: np.sin(2 * np.pi * t))

# 2. Configure simulation
sim = registry.create("hamiltonian_simulation", "euler_integrator")
sim.settings().set(
    "evolution_circuit_builder",
    AlgorithmRef(
        "evolution_circuit_builder",
        "euler",
        total_time=1.0,
        dt=0.1,
        propagator=AlgorithmRef("propagator", "magnus", order=1),
    ),
)

# 3. Run: evolve and measure observables
state_prep = identity_state_prep(num_qubits=td_hamiltonian.num_qubits)
observables = [h0]  # Measure ZZ energy after evolution

results = sim.run(td_hamiltonian, observables, state_prep, shots=1000)

for energy_result, measurement_data in results:
    print(f"Energy: {energy_result.energy_expectation_value:.4f}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

registry.available("hamiltonian_simulation")
# end-cell-list-implementations
################################################################################
