"""Standard QPE workflow starting from an active Hamiltonian HDF5 file.

This example demonstrates:
1. Loading an active Hamiltonian from an HDF5 file
2. Preparing a simple Hartree-Fock (HF) state
3. Running state preparation to get a circuit
4. Mapping the active Hamiltonian to qubit operators (Jordan-Wigner)
5. Defining Trotter unitary builder
6. Defining controlled circuit mapper
7. Running the standard QPE circuit builder to get a full QPE circuit
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    Hamiltonian,
    MajoranaMapping,
    SlaterDeterminantContainer,
    Wavefunction,
)
from qdk_chemistry.utils import Logger

# =============================================================================
# 1. Load the active Hamiltonian from an HDF5 file
# =============================================================================
# The file must follow the naming convention: <name>.hamiltonian.h5
hamiltonian_path = "benzene_diradical.hamiltonian.h5"
active_hamiltonian = Hamiltonian.from_hdf5_file(hamiltonian_path)

orbitals = active_hamiltonian.get_orbitals()
n_orbitals = orbitals.get_num_molecular_orbitals()
active_alpha, active_beta = orbitals.get_active_space_indices()
n_active_orbitals = len(active_alpha)

Logger.info(
    f"Loaded Hamiltonian: {n_orbitals} MOs, {n_active_orbitals} active orbitals"
)
Logger.info(f"Core energy: {active_hamiltonian.get_core_energy():.6f} Hartree")

# =============================================================================
# 2. Prepare a simple Hartree-Fock (HF) state
# =============================================================================
# Define the number of alpha and beta electrons in the active space.
# Adjust these for your system.
n_alpha = len(active_alpha)  # e.g., number of occupied alpha orbitals in active space
n_beta = len(active_beta)  # e.g., number of occupied beta orbitals in active space

# Create a canonical HF configuration (lowest orbitals occupied)
hf_config = Configuration.canonical_hf_configuration(n_alpha, n_beta, n_active_orbitals)

# Build a single-determinant (HF) wavefunction
hf_wavefunction = Wavefunction(SlaterDeterminantContainer(hf_config, orbitals))
Logger.info(
    f"HF state: {n_alpha} alpha, {n_beta} beta electrons in {n_active_orbitals} orbitals"
)

# =============================================================================
# 3. Run state preparation to get a circuit
# =============================================================================
state_prep = create("state_prep", "sparse_isometry_gf2x")
state_prep_circuit = state_prep.run(hf_wavefunction)
Logger.info("State preparation circuit created")

# =============================================================================
# 4. Map the active Hamiltonian to qubit operators (Jordan-Wigner)
# =============================================================================
n_spin_orbitals = 2 * n_active_orbitals
qubit_mapper = create("qubit_mapper")
qubit_hamiltonian = qubit_mapper.run(
    active_hamiltonian,
    MajoranaMapping.jordan_wigner(n_spin_orbitals),
)
Logger.info(f"Qubit Hamiltonian: {qubit_hamiltonian.num_qubits} qubits")
Logger.info(qubit_hamiltonian.get_summary())

# =============================================================================
# 5. Define Trotter unitary builder
# =============================================================================
# evolution_time sets the overall time for the unitary e^{-iHt}
target_energy_accuracy = 0.01
evolution_time = np.pi / 4
num_bits = int(np.ceil(np.log2(2 * np.pi / (evolution_time * target_energy_accuracy))))
unitary_builder = AlgorithmRef(
    "hamiltonian_unitary_builder",
    "trotter",
    time=evolution_time,
    order=1,
)

# =============================================================================
# 6. Define controlled circuit mapper
# =============================================================================
controlled_circuit_mapper = AlgorithmRef(
    "controlled_circuit_mapper",
    "pauli_sequence",
)

# =============================================================================
# 7. Run standard QPE circuit builder to get a full QPE circuit
# This builder uses RepeatEstimates to tell the resource estimator to analyze
# a single controlled Trotter step and multiply by (2^numBits - 1), rather than
# tracing through all steps individually. Much faster for large circuits.
# =============================================================================
estimate_circuit_builder = create(
    "qpe_circuit_builder",
    "qdk_standard_estimate",
    num_bits=num_bits,
    unitary_builder=unitary_builder,
    controlled_circuit_mapper=controlled_circuit_mapper,
)

estimate_circuits = estimate_circuit_builder.run(
    state_preparation=state_prep_circuit,
    qubit_hamiltonian=qubit_hamiltonian,
)
estimate_circuit = estimate_circuits[0]
Logger.info("Fast estimation QPE circuit built successfully!")

from qdk._interpreter import logical_counts

lc = logical_counts(
    estimate_circuit._qsharp_factory.program,
    estimate_circuit._qsharp_factory.parameter.values(),
)
print(f"Logical counts: {lc}")
# =============================================================================
# 8. Resource estimation using the fast circuit
# =============================================================================
from qdk.qre import estimate
from qdk.qre.application import QSharpApplication
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux

Logger.info("Resource estimation for the QPE circuit (using RepeatEstimates)...")
# Use the fast estimation circuit — it wraps a single Trotter step with RepeatEstimates
qpe_app = QSharpApplication(
    estimate_circuit._qsharp_factory.program,
    estimate_circuit._qsharp_factory.parameter.values(),
)
Logger.info(
    "Estimating resources for a Majorana-based architecture with ThreeAux QEC code..."
)
architecture = Majorana(error_rate=1e-4)
isa_query = ThreeAux.q() * RoundBasedFactory.q(use_cache=True, code_query=ThreeAux.q())

# Run resource estimation with 10% total error budget
results = estimate(qpe_app, architecture, isa_query, max_error=0.5, name="QPE")

# Display the Pareto-optimal results table
results.add_factory_summary_column()
Logger.info("\nPareto-optimal resource estimates for the QPE circuit:")
print(results.as_frame())
