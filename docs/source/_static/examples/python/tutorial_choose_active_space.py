"""Choose an active space for stretched N2."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import comb

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure
from qdk_chemistry.data.symmetry import SymmetryLabel, axes
from qdk_chemistry.utils import Logger, compute_valence_space_parameters

Logger.set_global_level(Logger.LogLevel.off)


def spatial_indices(index_set):
    """Return the alpha-channel indices for a restricted orbital space."""
    return list(index_set.indices(SymmetryLabel([axes.alpha()])))


################################################################################
# docs:xyz ../data/tutorial_stretched_n2.structure.xyz
# start-cell-hartree-fock
structure = Structure.from_xyz("""\
2
Stretched N2 molecule for the ground-state QPE tutorial
N    0.000000    0.000000    0.000000
N    0.000000    0.000000    1.270025
""")
charge = 0
spin_multiplicity = 1
basis_set = "cc-pvdz"

scf_solver = create("scf_solver", "qdk")
hartree_fock_energy, hartree_fock_wavefunction = scf_solver.run(
    structure,
    charge=charge,
    spin_multiplicity=spin_multiplicity,
    basis_or_guess=basis_set,
)
print(f"Hartree-Fock energy: {hartree_fock_energy:.12f} Hartree")
# end-cell-hartree-fock
################################################################################

################################################################################
# start-cell-valence-space
num_valence_electrons, num_valence_orbitals = compute_valence_space_parameters(
    hartree_fock_wavefunction, charge
)
valence_selector = create(
    "active_space_selector",
    "qdk_valence",
    num_active_electrons=num_valence_electrons,
    num_active_orbitals=num_valence_orbitals,
)
valence_wavefunction = valence_selector.run(hartree_fock_wavefunction)
valence_indices = spatial_indices(valence_wavefunction.get_orbitals().active_indices())
num_valence_alpha, num_valence_beta = valence_wavefunction.get_active_num_electrons()

print(f"Initial valence space: CAS({num_valence_electrons}e, {num_valence_orbitals}o)")
print(f"Initial active orbital indices: {valence_indices}")
# end-cell-valence-space
################################################################################

################################################################################
# start-cell-initial-casci
hamiltonian_constructor = create("hamiltonian_constructor")
casci_solver = create(
    "multi_configuration_calculator",
    "macis_cas",
    calculate_one_rdm=True,
    calculate_two_rdm=True,
)

valence_hamiltonian = hamiltonian_constructor.run(valence_wavefunction.get_orbitals())
valence_energy, valence_casci_wavefunction = casci_solver.run(
    valence_hamiltonian,
    num_valence_alpha,
    num_valence_beta,
)
num_valence_determinants = len(valence_casci_wavefunction.get_coefficients())
print(f"Initial CASCI energy: {valence_energy:.12f} Hartree")
print(f"Initial CASCI determinants: {num_valence_determinants}")
# end-cell-initial-casci
################################################################################

################################################################################
# start-cell-natural-orbitals
natural_orbital_transformer = create("orbital_localizer", "qdk_natural_orbitals")
natural_orbital_wavefunction = natural_orbital_transformer.run(
    valence_casci_wavefunction,
    valence_indices,
    valence_indices,
)

natural_orbital_hamiltonian = hamiltonian_constructor.run(natural_orbital_wavefunction.get_orbitals())
natural_orbital_energy, natural_orbital_casci_wavefunction = casci_solver.run(
    natural_orbital_hamiltonian,
    num_valence_alpha,
    num_valence_beta,
)
orbital_entropies = [float(value) for value in natural_orbital_casci_wavefunction.get_single_orbital_entropies()]
print(f"Natural-orbital CASCI energy: {natural_orbital_energy:.12f} Hartree")
# end-cell-natural-orbitals
################################################################################

################################################################################
# start-cell-refine
autocas_selector = create("active_space_selector", "qdk_autocas_eos")
refined_wavefunction = autocas_selector.run(natural_orbital_casci_wavefunction)
refined_orbitals = refined_wavefunction.get_orbitals()
refined_indices = spatial_indices(refined_orbitals.active_indices())
inactive_indices = spatial_indices(refined_orbitals.inactive_indices())
num_refined_alpha, num_refined_beta = refined_wavefunction.get_active_num_electrons()
num_refined_electrons = num_refined_alpha + num_refined_beta
num_refined_orbitals = len(refined_indices)
num_virtual_orbitals = refined_orbitals.get_num_molecular_orbitals() - len(inactive_indices) - num_refined_orbitals

print("Single-orbital entropies:")
for orbital_index, entropy in zip(valence_indices, orbital_entropies, strict=True):
    selection_marker = "*" if orbital_index in refined_indices else " "
    print(f" {selection_marker} orbital {orbital_index}: {entropy:.9f}")
print(f"Refined active space: CAS({num_refined_electrons}e, {num_refined_orbitals}o)")
print(f"Inactive orbital indices: {inactive_indices}")
print(f"Active orbital indices: {refined_indices}")
print(f"Virtual orbitals: {num_virtual_orbitals}")
# end-cell-refine
################################################################################

################################################################################
# start-cell-final-casci
refined_hamiltonian = hamiltonian_constructor.run(refined_orbitals)
refined_energy, refined_casci_wavefunction = casci_solver.run(
    refined_hamiltonian,
    num_refined_alpha,
    num_refined_beta,
)
num_refined_determinants = len(refined_casci_wavefunction.get_coefficients())
energy_increase = refined_energy - natural_orbital_energy

print(f"Final CASCI energy: {refined_energy:.12f} Hartree")
print(f"Final CASCI determinants: {num_refined_determinants}")
print(f"Energy increase from reducing the active space: {energy_increase:.12f} Hartree")
# end-cell-final-casci
################################################################################

# Numerical guards run in the documentation example test but are not displayed.
assert abs(hartree_fock_energy - (-108.866810916955)) < 1e-8
assert abs(valence_energy - (-108.997239708567)) < 1e-8
assert abs(natural_orbital_energy - valence_energy) < 1e-10
assert abs(refined_energy - (-108.964632065071)) < 1e-8
assert valence_indices == list(range(2, 10))
assert num_valence_determinants == comb(8, 5) ** 2 == 3136
assert inactive_indices == list(range(5))
assert refined_indices == list(range(5, 9))
assert num_refined_electrons == 4
assert num_virtual_orbitals == 19
assert num_refined_determinants == comb(4, 2) ** 2 == 36
assert valence_energy < refined_energy < hartree_fock_energy