"""Propagator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import registry

propagator = registry.create("propagator", "magnus")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
propagator.settings().set("order", 1)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.algorithms import registry
from qdk_chemistry.data import DrivenQubitHamiltonian, LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian

# 1. Build a driven Hamiltonian: H(t) = H0 + sin(2πt)·H1
lattice = LatticeGraph.chain(4)
h0 = create_ising_hamiltonian(lattice, j=1.0, h=0.0)  # ZZ coupling
h1 = create_ising_hamiltonian(lattice, j=0.0, h=0.5)  # Transverse X field
td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=lambda t: np.sin(2 * np.pi * t))

# 2. Create the propagator and compute the effective Hamiltonian
propagator = registry.create("propagator", "magnus")
h_eff = propagator.run(td_hamiltonian, t_start=0.0, t_end=0.1)

print(f"Effective Hamiltonian has {len(h_eff.pauli_strings)} Pauli terms")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

registry.available("propagator")
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-nested
from qdk_chemistry.data import AlgorithmRef

# Configure propagator as a nested algorithm inside an evolution circuit builder
propagator_ref = AlgorithmRef("propagator", "magnus", order=1)

euler_builder = registry.create(
    "evolution_circuit_builder",
    "euler",
    propagator=propagator_ref,
    total_time=1.0,
    dt=0.1,
)
# end-cell-nested
################################################################################
