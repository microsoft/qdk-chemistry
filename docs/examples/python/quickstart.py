"""Minimal end-to-end example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# start-cell-1
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure
from qdk_chemistry.data.qubit_hamiltonian import (
    filter_and_group_pauli_ops_from_wavefunction,
)
from qdk_chemistry.utils.wavefunction import get_top_determinants

# Define benzene diradical structure directly using numpy arrays
coords = np.array(
    [
        [0.000000, 1.396000, 0.000000],
        [1.209077, 0.698000, 0.000000],
        [1.209077, -0.698000, 0.000000],
        [0.000000, -1.396000, 0.000000],
        [-1.209077, -0.698000, 0.000000],
        [-1.209077, 0.698000, 0.000000],
        [2.151000, 1.242000, 0.000000],
        [2.151000, -1.242000, 0.000000],
        [-2.151000, -1.242000, 0.000000],
        [-2.151000, 1.242000, 0.000000],
    ]
)
elements = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H"]
structure = Structure(coords, elements)

# Alternative: load from XYZ file
# from pathlib import Path
# structure = Structure.from_xyz_file(Path("benzene_diradical.structure.xyz"))

print(f"Created structure with {structure.get_num_atoms()} atoms")
print(f"Elements: {structure.get_elements()}")
# end-cell-1
# start-cell-2
# Perform an SCF calculation, returning the energy and wavefunction
scf_solver = create("scf_solver", basis_set="cc-pvdz")
E_hf, wfn_hf = scf_solver.run(structure, charge=0, spin_multiplicity=1)
print(f"SCF energy is {E_hf:.3f} Hartree")

# Display a summary of the molecular orbitals obtained from the SCF calculation
print("SCF Orbitals:\n", wfn_hf.get_orbitals().get_summary())
# end-cell-2
# start-cell-3
# Select active space (6 electrons in 6 orbitals for the benzene diradical)
#   to choose most chemically relevant orbitals
active_space_selector = create(
    "active_space_selector",
    algorithm_name="qdk_valence",
    num_active_electrons=6,
    num_active_orbitals=6,
)
active_wfn = active_space_selector.run(wfn_hf)
active_orbitals = active_wfn.get_orbitals()

# Print a summary of the active space orbitals
print("Active Space Orbitals:\n", active_orbitals.get_summary())
# end-cell-3
# start-cell-4
# Construct Hamiltonian in the active space and print its summary
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(active_orbitals)
print("Active Space Hamiltonian:\n", hamiltonian.get_summary())
# end-cell-4
# start-cell-5
# Perform CASCI calculation to get the wavefunction and exact energy for the active space
mc = create("multi_configuration_calculator")
E_cas, wfn_cas = mc.run(
    hamiltonian, n_active_alpha_electrons=3, n_active_beta_electrons=3
)
print(
    f"CASCI energy is {E_cas:.3f} Hartree, and the electron correlation energy is {E_cas - E_hf:.3f} Hartree"
)
# end-cell-5
# start-cell-6
# Get top 2 determinants from the CASCI wavefunction to form a sparse wavefunction
top_configurations = get_top_determinants(wfn_cas, max_determinants=2)

# Compute the reference energy of the sparse wavefunction
pmc_calculator = create("projected_multi_configuration_calculator")
E_sparse, wfn_sparse = pmc_calculator.run(hamiltonian, list(top_configurations.keys()))

print(f"Reference energy for top 2 determinants is {E_sparse:.6f} Hartree")

# Generate state preparation circuit for the sparse state via sparse isometry (GF2 + X)
state_prep = create("state_prep", "sparse_isometry_gf2x")
sparse_isometry_circuit = state_prep.run(wfn_sparse)
# end-cell-6
# start-cell-7
# Prepare qubit Hamiltonian
qubit_mapper = create("qubit_mapper", algorithm_name="qiskit", encoding="jordan-wigner")
qubit_hamiltonian = qubit_mapper.run(hamiltonian)

# Print the number of Pauli strings in the full Hamiltonian
print(
    f"Number of Pauli strings in the Hamiltonian: {len(qubit_hamiltonian.pauli_strings)}"
)

# Filter and group Pauli operators based on the wavefunction
filtered_hamiltonian_ops, classical_coeffs = (
    filter_and_group_pauli_ops_from_wavefunction(qubit_hamiltonian, wfn_sparse)
)
print(
    f"Filtered and grouped qubit Hamiltonian contains {len(filtered_hamiltonian_ops)} groups:"
)
for igroup, group in enumerate(filtered_hamiltonian_ops):
    print(f"Group {igroup + 1}: {[group.pauli_strings]}")
print(f"Number of classical coefficients: {len(classical_coeffs)}")
# end-cell-7
# start-cell-8
# Estimate energy using the optimized circuit and filtered Hamiltonian operators
estimator = create("energy_estimator", algorithm_name="qdk_base_simulator")
energy_results, simulation_data = estimator.run(
    circuit_qasm=sparse_isometry_circuit,
    qubit_hamiltonians=filtered_hamiltonian_ops,
    total_shots=250000,
    classical_coeffs=classical_coeffs,
)

for i, results in enumerate(simulation_data.bitstring_counts):
    print(
        f"Measurement Results for Hamiltonian Group {i + 1}: {simulation_data.hamiltonians[i].pauli_strings}"
    )

# Print statistic for measured energy
energy_mean = energy_results.energy_expectation_value + hamiltonian.get_core_energy()
energy_stddev = np.sqrt(energy_results.energy_variance)
print(
    f"Estimated energy from quantum circuit: {energy_mean:.3f} ± {energy_stddev:.3f} Hartree"
)

# Print comparison with reference energy
print(f"Difference from reference energy: {energy_mean - E_sparse} Hartree")
# end-cell-8
