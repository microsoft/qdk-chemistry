"""EulerEvolutionCircuitBuilder usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-configure-euler
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create an Euler evolution circuit builder with 4th-order Trotter
evolution_builder = AlgorithmRef(
    "hamiltonian_unitary_builder", "trotter", order=4, num_divisions=2
)
propagator = AlgorithmRef("propagator", "magnus", order=1)

circuit_builder = create(
    "evolution_circuit_builder",
    "euler",
    evolution_builder=evolution_builder,
    propagator=propagator,
    total_time=1.0,
    dt=1.0,  # single Euler step = full time
)
# end-cell-configure-euler
################################################################################

################################################################################
# start-cell-run

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, DrivenQubitHamiltonian, LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian

# 1. Define the Ising model on a small lattice
lattice = LatticeGraph.chain(4)
h0 = create_ising_hamiltonian(lattice, j=1.0, h=0.0)  # ZZ coupling
h1 = create_ising_hamiltonian(lattice, j=0.0, h=0.5)  # Transverse X field

# Constant drive → static Hamiltonian: H(t) = H0 + 1·H1
td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=lambda t: 1.0)

# 2. Configure the builder
evolution_builder = AlgorithmRef(
    "hamiltonian_unitary_builder", "trotter", order=4, num_divisions=2
)
propagator = AlgorithmRef("propagator", "magnus", order=1)

circuit_builder = create(
    "evolution_circuit_builder",
    "euler",
    evolution_builder=evolution_builder,
    propagator=propagator,
    total_time=1.0,
    dt=1.0,
)

# 3. Build the circuit (state-prep + evolution) without executing
state_prep = identity_state_prep(num_qubits=td_hamiltonian.num_qubits)
circuit = circuit_builder.run(td_hamiltonian, state_prep)

print(f"Circuit has QIR: {circuit.get_qir() is not None}")

# 4. Use for quantum resource estimation
app = circuit.get_qre_application()
print(f"QRE application created: {app is not None}")
# end-cell-run
################################################################################
