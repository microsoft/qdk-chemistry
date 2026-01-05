"""Dynamical correlation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a DynamicalCorrelationCalculator instance
mp2_calculator = create("dynamical_correlation_calculator")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure settings (for implementations that support them)
# mp2_calculator.settings().set("conv_tol", 1e-8)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
from pathlib import Path  # noqa: E402
from qdk_chemistry.data import Ansatz, Structure  # noqa: E402

# Load H2 structure from XYZ file
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# Run initial SCF to get reference wavefunction
scf_solver = create("scf_solver")
E_hf, wfn_hf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="def2-svp"
)

# Create Hamiltonian from orbitals
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_hf.get_orbitals())

# Create ansatz combining wavefunction and Hamiltonian
ansatz = Ansatz(hamiltonian, wfn_hf)

# Run the correlation calculation
mp2_total_energy, final_wavefunction = mp2_calculator.run(ansatz)

# Extract correlation energy
mp2_corr_energy = mp2_total_energy - E_hf
print(f"MP2 Correlation Energy: {mp2_corr_energy:.10f} Hartree")
print(f"MP2 Total Energy: {mp2_total_energy:.10f} Hartree")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("dynamical_correlation_calculator"))
# ['pyscf_coupled_cluster', 'qdk_mp2_calculator']
# end-cell-list-implementations
################################################################################
