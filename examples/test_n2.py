"""Test file."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Reduce logging output for demo
from qdk_chemistry.utils import Logger, compute_valence_space_parameters
from qsharp._simulation import run_qir
from utils.qpe_utils import compute_evolution_time, prepare_2_dets_trial_state

Logger.set_global_level(Logger.LogLevel.debug)

# Stretched N2 structure at 1.270025 Å bond length
structure = Structure.from_xyz_file(Path("data/stretched_n2.structure.xyz"))

# Perform an SCF calculation, returning the energy and wavefunction
scf_solver = create("scf_solver")
E_hf, wfn_hf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)

# Reduce the number of orbitals
num_val_e, num_val_o = compute_valence_space_parameters(wfn_hf, charge=0)
active_space_selector = create(
    "active_space_selector",
    "qdk_valence",
    num_active_electrons=num_val_e,
    num_active_orbitals=num_val_o,
)
valence_wf = active_space_selector.run(wfn_hf)

# Localize the orbitals
localizer = create("orbital_localizer", "qdk_mp2_natural_orbitals")
valence_indices = valence_wf.get_orbitals().get_active_space_indices()
loc_wfn = localizer.run(valence_wf, *valence_indices)
print("Localized orbitals:\n", loc_wfn.get_orbitals().get_summary())

# Construct a Hamiltonian from the localized orbitals
hamiltonian_constructor = create("hamiltonian_constructor")
loc_orbitals = loc_wfn.get_orbitals()
loc_hamiltonian = hamiltonian_constructor.run(loc_orbitals)
num_alpha_electrons, num_beta_electrons = loc_wfn.get_active_num_electrons()

# Compute the selected configuration interaction wavefunction
macis_mc = create(
    "multi_configuration_calculator",
    "macis_asci",
    calculate_one_rdm=True,
    calculate_two_rdm=True,
)
_, wfn_sci = macis_mc.run(loc_hamiltonian, num_alpha_electrons, num_beta_electrons)

# Optimize the problem with autoCAS-EOS active space selection
autocas = create("active_space_selector", "qdk_autocas_eos")
autocas_wfn = autocas.run(wfn_sci)
indices, _ = autocas_wfn.get_orbitals().get_active_space_indices()
print(
    f"autoCAS-EOS selected {len(indices)} of {num_val_o} orbitals for the active space: indices={list(indices)}"
)

# Construct the active space Hamiltonian
hamiltonian_constructor = create("hamiltonian_constructor")
refined_orbitals = autocas_wfn.get_orbitals()
active_hamiltonian = hamiltonian_constructor.run(refined_orbitals)

# Calculate the exact wavefunction and energy with CASCI
alpha_electrons, beta_electrons = autocas_wfn.get_active_num_electrons()
mc = create("multi_configuration_calculator", "macis_cas")
e_cas, wfn_cas = mc.run(active_hamiltonian, alpha_electrons, beta_electrons)
print(f"Active space system energy: {e_cas:.6f} Hartree")


# Prepare a trial state with two determinants (and noise). Compute its overlap with the CASCI wavefunction.
wfn_trial, fidelity = prepare_2_dets_trial_state(wfn_cas)


# Generate state preparation circuit for the sparse state via GF2+X sparse isometry
state_prep = create("state_prep", "sparse_isometry_gf2x")
sparse_isometry_circuit = state_prep.run(wfn_trial)

# Prepare the qubit-mapped Hamiltonian
qubit_mapper = create("qubit_mapper", algorithm_name="qiskit", encoding="jordan-wigner")
qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)
print("Qubit Hamiltonian:\n", qubit_hamiltonian.get_summary())

# Prepare the qubit-mapped Hamiltonian
qubit_mapper = create("qubit_mapper", algorithm_name="qiskit", encoding="jordan-wigner")
qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)
print("Qubit Hamiltonian:\n", qubit_hamiltonian.get_summary())

# Set up parameters for iQPE
M_PRECISION = 6
SHOTS_PER_BIT = 3
SIMULATOR_SEED = 42

# Propose evolution time given the qubit Hamiltonian and number of precision bits
evolution_time = compute_evolution_time(qubit_hamiltonian, num_bits=M_PRECISION)
print(f"Proposed evolution time: {evolution_time:.4f} Hartree^-1")


# Use factory methods to create the iQPE algorithm components
evolution_builder = create("time_evolution_builder", "trotter")
circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
iqpe = create(
    "phase_estimation",
    "iterative",
    num_bits=M_PRECISION,
    evolution_time=evolution_time,
    shots_per_bit=SHOTS_PER_BIT,
)

# Generate the iQPE iteration circuit for a specific iteration (3rd from last)
iqpe_iter_circuit = iqpe.create_iteration_circuit(
    state_preparation=sparse_isometry_circuit,
    qubit_hamiltonian=qubit_hamiltonian,
    evolution_builder=evolution_builder,
    circuit_mapper=circuit_mapper,
    iteration=0,
    total_iterations=M_PRECISION,
)


qir = iqpe_iter_circuit.get_qir()
results = run_qir(qir, type="cpu")
print(results)
