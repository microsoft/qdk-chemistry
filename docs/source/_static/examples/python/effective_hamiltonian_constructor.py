"""Effective Hamiltonian construction examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the DUCC effective Hamiltonian constructor
ducc = create("effective_hamiltonian_constructor", "qdk_ducc")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Select the commutator level (0 to 2)
ducc.settings().set("level", 2)
# end-cell-configure
################################################################################

################################################################################
# docs:xyz ../data/water.structure.xyz
# start-cell-run
from qdk_chemistry.data import Ansatz, Structure

structure = Structure.from_xyz("""\
3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
""")

# Reference determinant and the Hamiltonian over the full orbital window W
scf_solver = create("scf_solver")
scf_energy, scf_wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
hamiltonian = create("hamiltonian_constructor").run(scf_wavefunction.get_orbitals())

# T1/T2 amplitudes over W; pyscf_coupled_cluster with store_amplitudes=True also works
mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
_, amplitudes, _ = mp2_calculator.run(Ansatz(hamiltonian, scf_wavefunction))

# Target space P: a four-electron, four-orbital valence space
active_space_selector = create("active_space_selector", "qdk_valence")
active_space_selector.settings().set("num_active_electrons", 4)
active_space_selector.settings().set("num_active_orbitals", 4)
p_indices = active_space_selector.run(scf_wavefunction).get_orbitals().active_indices()

# Downfold W into P, then diagonalize the effective Hamiltonian in P
effective_hamiltonian = ducc.run(amplitudes, hamiltonian, p_indices)
mc_calculator = create("multi_configuration_calculator", "macis_cas")
ducc_energy, _ = mc_calculator.run(effective_hamiltonian, 2, 2)
print(f"SCF Energy: {scf_energy:.10f} Hartree")
print(f"DUCC CASCI Energy: {ducc_energy:.10f} Hartree")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(sorted(registry.available("effective_hamiltonian_constructor")))
# ['qdk_dsrg_pt2', 'qdk_ducc']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-dsrg-pt2-example
# DSRG-PT2 takes the SCF determinant as its reference; s is in Eh^-2
dsrg = create("effective_hamiltonian_constructor", "qdk_dsrg_pt2")
dsrg.settings().set("flow_parameter", 2.0)
dsrg_hamiltonian = dsrg.run(scf_wavefunction, hamiltonian, p_indices)
dsrg_energy, _ = mc_calculator.run(dsrg_hamiltonian, 2, 2)
print(f"DSRG-PT2 CASCI Energy: {dsrg_energy:.10f} Hartree")
# end-cell-dsrg-pt2-example
################################################################################
