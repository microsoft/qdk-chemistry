"""Pauli operator arithmetic examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-creation
from qdk_chemistry.data import PauliOperator

# Create Pauli operators on specific qubits
X0 = PauliOperator.X(0)  # Pauli X on qubit 0
Y1 = PauliOperator.Y(1)  # Pauli Y on qubit 1
Z2 = PauliOperator.Z(2)  # Pauli Z on qubit 2
I0 = PauliOperator.I(0)  # Identity on qubit 0
# end-cell-creation
################################################################################

################################################################################
# start-cell-expressions
# Scalar multiplication
scaled = 0.5 * PauliOperator.X(0)
scaled_complex = (1 + 2j) * PauliOperator.Z(1)

# Products of operators
product = PauliOperator.X(0) * PauliOperator.Z(1)

# Sums of operators
sum_expr = PauliOperator.X(0) + PauliOperator.Y(1)

# Building a Hamiltonian-like expression
H = (
    0.5 * PauliOperator.X(0) * PauliOperator.X(1)
    + 0.5 * PauliOperator.Y(0) * PauliOperator.Y(1)
    + 1.0 * PauliOperator.Z(0) * PauliOperator.Z(1)
)

print(H)  # "0.5 * X(0) * X(1) + 0.5 * Y(0) * Y(1) + Z(0) * Z(1)"
# end-cell-expressions
################################################################################

################################################################################
# start-cell-simplify
# Pauli algebra: X * X = I (identity)
xx = PauliOperator.X(0) * PauliOperator.X(0)
result = xx.simplify()
print(result)  # "1"

# Pauli algebra: X * Y = iZ
xy = PauliOperator.X(0) * PauliOperator.Y(0)
result = xy.simplify()
print(result)  # "1j * Z(0)"

# Combining like terms
duplicate = PauliOperator.X(0) + PauliOperator.X(0)
result = duplicate.simplify()
print(result)  # "2 * X(0)"

# Distributing products over sums
nested = PauliOperator.X(0) * (PauliOperator.Y(1) + PauliOperator.Z(1))
distributed = nested.distribute()
print(distributed)  # "X(0) * Y(1) + X(0) * Z(1)"
# end-cell-simplify
################################################################################

################################################################################
# start-cell-canonical
# Get canonical string representation (little-endian: qubit 0 leftmost)
expr = PauliOperator.X(0) * PauliOperator.Z(2)
simplified = expr.simplify()

canonical = simplified.to_canonical_string(4)  # 4 qubits total
print(canonical)  # "XIZI"

# Get coefficient and string pairs for each term
terms = simplified.to_canonical_terms(4)
print(terms)  # [((1+0j), 'XIZI')]

# For sums, get all terms
H = PauliOperator.X(0) + PauliOperator.Z(1)
terms = H.to_canonical_terms(2)
print(terms)  # [((1+0j), 'XI'), ((1+0j), 'IZ')]
# end-cell-canonical
################################################################################
