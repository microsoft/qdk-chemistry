"""State preparation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a StatePreparation instance
state_prep = create("state_prep", "sparse_isometry_gf2x")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure transpilation settings
state_prep.settings().set("transpile", True)
state_prep.settings().set("basis_gates", ["rz", "cz", "sdg", "h"])
state_prep.settings().set("transpile_optimization_level", 3)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.data import (
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

# Construct the quantum circuit
qasm_circuit = state_prep.run(wavefunction)
print(f"OpenQASM circuit:\n{qasm_circuit}")
# end-cell-run
################################################################################
