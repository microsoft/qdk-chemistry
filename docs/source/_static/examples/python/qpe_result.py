"""QpeResult usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create-from-time-evolution

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import QpeResult, QubitHamiltonian

# Time-evolution QPE: U = e^{-iHt}, eigenvalue_from_phase wraps the angle.
hamiltonian = QubitHamiltonian(
    pauli_strings=["ZI", "IZ", "XX"], coefficients=[0.5, -0.3, 0.2]
)
evolution_time = 0.1

builder = create("hamiltonian_unitary_builder", "trotter", time=evolution_time)
unitary = builder.run(hamiltonian)
container = unitary.get_container()

result = QpeResult.from_phase_fraction(
    method="iterative",
    phase_fraction=0.423828125,
    eigenvalue_from_phase=container.eigenvalue_from_phase,
    bits_msb_first=(0, 1, 1, 0, 1, 1, 0, 0, 1, 0),
    bitstring_msb_first="0110110010",
)
# end-cell-create-from-time-evolution
################################################################################

################################################################################
# start-cell-create-from-qubitization

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import QpeResult, QubitHamiltonian

# Qubitization QPE: W = walk operator, E = lambda * cos(2*pi*phi).
hamiltonian = QubitHamiltonian(
    pauli_strings=["ZI", "IZ", "XX"], coefficients=[0.5, -0.3, 0.2]
)

builder = create("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
unitary = builder.run(hamiltonian)
walk_container = unitary.get_container()

result_qubitization = QpeResult.from_phase_fraction(
    method="qubitization_qpe",
    phase_fraction=0.25,
    eigenvalue_from_phase=walk_container.eigenvalue_from_phase,
    bits_msb_first=(0, 1, 0, 0),
    bitstring_msb_first="0100",
)
# end-cell-create-from-qubitization
################################################################################

################################################################################
# start-cell-inspect
# Inspect the result
print(f"Method: {result.method}")
print(f"Phase fraction: {result.phase_fraction:.6f}")
print(f"Phase angle: {result.phase_angle:.6f} rad")
print(f"Raw energy: {result.raw_energy:.8f} Ha")
print(f"Measured bits: {result.bits_msb_first}")

# Full summary
print(result.get_summary())
# end-cell-inspect
################################################################################

################################################################################
# start-cell-serialization
import os
import tempfile

tmpdir = tempfile.mkdtemp()

# Save to JSON
result.to_json_file(os.path.join(tmpdir, "result.qpe_result.json"))

# Load from JSON
loaded = QpeResult.from_json_file(os.path.join(tmpdir, "result.qpe_result.json"))

# Save to HDF5
result.to_hdf5_file(os.path.join(tmpdir, "result.qpe_result.h5"))

# Load from HDF5
loaded_h5 = QpeResult.from_hdf5_file(os.path.join(tmpdir, "result.qpe_result.h5"))
# end-cell-serialization
################################################################################
