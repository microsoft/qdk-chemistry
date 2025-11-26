"""State preparation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

state_prep = create("state_prep", "sparse_isometry_gf2x")
state_prep.settings().set("transpile", True)
state_prep.settings().set("basis_gates", ["rz", "cz", "sdg"])
state_prep.settings().set("transpile_optimization_level", 3)

# Example usage with a Wavefunction object
# wavefunction = Wavefunction.from_json_file("molecule.wavefunction.json")
# circuit_qasm = state_prep.run(wavefunction)
