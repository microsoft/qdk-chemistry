"""Phase estimation usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default (iterative) phase estimation algorithm
iqpe = create("phase_estimation", "iterative")

# Or create the standard QFT-based variant (requires Qiskit)
qpe = create("phase_estimation", "qiskit_standard")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure-iqpe
# Configure iterative phase estimation
iqpe = create("phase_estimation", "iterative")
iqpe.settings().set("num_bits", 10)
iqpe.settings().set("evolution_time", 0.1)
iqpe.settings().set("shots_per_bit", 10)
# end-cell-configure-iqpe
################################################################################

################################################################################
# start-cell-configure-standard
# Configure standard QFT-based phase estimation
qpe = create("phase_estimation", "qiskit_standard")
qpe.settings().set("num_bits", 10)
qpe.settings().set("evolution_time", 0.1)
qpe.settings().set("shots", 100)
qpe.settings().set("qft_do_swaps", True)
# end-cell-configure-standard
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# 1. Setup molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# 2. SCF
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# 3. Hamiltonian construction
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# 4. Multi-configuration calculation (reference state)
cas_solver = create("multi_configuration_calculator")
E_cas, wfn_cas = cas_solver.run(hamiltonian, 1, 1)

# 5. Qubit mapping
qubit_mapper = create("qubit_mapper", encoding="jordan-wigner")
qubit_ham = qubit_mapper.run(hamiltonian)

# 6. State preparation
state_prep = create("state_prep", "sparse_isometry_gf2x")
circuit = state_prep.run(wfn_cas)

# 7. Create QPE dependencies
unitary_builder = create("unitary_builder", "trotter", order=2)
circuit_mapper = create("controlled_circuit_mapper", "pauli_sequence")
circuit_executor = create("circuit_executor", "qiskit_aer_simulator", seed=42)

# 8. Create and run IQPE
iqpe = create(
    "phase_estimation", "iterative", num_bits=10, evolution_time=0.1, shots_per_bit=10
)
result = iqpe.run(
    state_preparation=circuit,
    qubit_hamiltonian=qubit_ham,
    unitary_builder=unitary_builder,
    circuit_mapper=circuit_mapper,
    circuit_executor=circuit_executor,
)

# 9. Inspect results
print(result.get_summary())
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered phase estimation implementations
implementations = registry.available("phase_estimation")
print(implementations)  # e.g. ['iterative', 'qiskit_standard']
# end-cell-list-implementations
################################################################################
