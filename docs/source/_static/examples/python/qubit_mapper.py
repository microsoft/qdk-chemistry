"""Qubit mapper usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")

# Example usage with a Hamiltonian object
# hamiltonian = Hamiltonian.from_json_file("molecule.hamiltonian.json")
# qubit_hamiltonian = mapper.run(hamiltonian)
