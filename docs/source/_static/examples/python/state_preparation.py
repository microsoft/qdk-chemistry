"""State preparation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a StatePreparation instance
sparse_prep = create("state_prep", "sparse_isometry_gf2x")
regular_prep = create("state_prep", "regular_isometry")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure transpilation settings
sparse_prep.settings().set("transpile", True)
sparse_prep.settings().set("basis_gates", ["rz", "cz", "sdg", "h"])
sparse_prep.settings().set("transpile_optimization_level", 3)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np  # noqa: E402
from qdk_chemistry.data import (  # noqa: E402
    BasisSet,
    CasWavefunctionContainer,
    Configuration,
    Orbitals,
    OrbitalType,
    Shell,
    Wavefunction,
)

# Create a basis set
shells = []
for _ in range(3):
    exps = np.array([1.0])
    coefs = np.array([1.0])
    shell = Shell(0, OrbitalType.S, exps, coefs)
    shells.append(shell)
basis_set = BasisSet("foo", shells)

# Create orbitals
coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
orbitals = Orbitals(coeffs, None, None, basis_set)

# Construct a wavefunction
dets = [Configuration("200"), Configuration("ud0")]
coeffs = np.array([0.9, 0.1])
container = CasWavefunctionContainer(coeffs, dets, orbitals)
wavefunction = Wavefunction(container)

# Construct the circuit
regular_circuit = regular_prep.run(wavefunction)
sparse_circuit = sparse_prep.run(wavefunction)
print(f"Regular isometry QASM:\n{regular_circuit.get_qasm()}")
print(f"Sparse isometry QASM:\n{sparse_circuit.get_qasm()}")
# end-cell-run
################################################################################
