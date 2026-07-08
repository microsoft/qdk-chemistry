"""EulerEvolutionCircuitBuilder usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-configure-euler
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create an Euler evolution circuit builder.
# The propagator evaluates the effective Hamiltonian over each dt interval,
# and the evolution builder constructs the unitary with `num_divisions`
# subdivisions within that interval.
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

# 1. Define the Ising model on a small lattice with a sinusoidal drive
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, DrivenQubitHamiltonian, LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian

lattice = LatticeGraph.chain(4)
h0 = create_ising_hamiltonian(lattice, j=1.0, h=0.0)  # ZZ coupling
h1 = create_ising_hamiltonian(lattice, j=0.0, h=0.5)  # Transverse X field

# Sinusoidal drive → time-dependent Hamiltonian: H(t) = H0 + sin(2πt)·H1
td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=lambda t: np.sin(2 * np.pi * t))

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
    dt=0.1,
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
