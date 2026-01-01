// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Pauli operator arithmetic examples.
// --------------------------------------------------------------------------------------------
// start-cell-creation
#include <iostream>
#include <qdk/chemistry/data/pauli_operator.hpp>

using namespace qdk::chemistry::data;

int main() {
  // Create Pauli operators on specific qubits
  auto X0 = PauliOperator::X(0);  // Pauli X on qubit 0
  auto Y1 = PauliOperator::Y(1);  // Pauli Y on qubit 1
  auto Z2 = PauliOperator::Z(2);  // Pauli Z on qubit 2
  auto I0 = PauliOperator::I(0);  // Identity on qubit 0
  // end-cell-creation
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-expressions
  // Scalar multiplication
  auto scaled = 0.5 * PauliOperator::X(0);
  auto scaled_complex = std::complex<double>(1, 2) * PauliOperator::Z(1);

  // Products of operators
  auto product = PauliOperator::X(0) * PauliOperator::Z(1);

  // Sums of operators
  auto sum_expr = PauliOperator::X(0) + PauliOperator::Y(1);

  // Building a Hamiltonian-like expression
  auto H = 0.5 * PauliOperator::X(0) * PauliOperator::X(1) +
           0.5 * PauliOperator::Y(0) * PauliOperator::Y(1) +
           1.0 * PauliOperator::Z(0) * PauliOperator::Z(1);

  std::cout << H.to_string() << std::endl;
  // "0.5 * X(0) * X(1) + 0.5 * Y(0) * Y(1) + Z(0) * Z(1)"
  // end-cell-expressions
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-simplify
  // Pauli algebra: X * X = I (identity)
  auto xx = PauliOperator::X(0) * PauliOperator::X(0);
  auto result = xx.simplify();
  std::cout << result->to_string() << std::endl;  // "1"

  // Pauli algebra: X * Y = iZ
  auto xy = PauliOperator::X(0) * PauliOperator::Y(0);
  result = xy.simplify();
  std::cout << result->to_string() << std::endl;  // "i * Z(0)"

  // Combining like terms
  auto duplicate = PauliOperator::X(0) + PauliOperator::X(0);
  result = duplicate.simplify();
  std::cout << result->to_string() << std::endl;  // "2 * X(0)"

  // Distributing products over sums
  auto nested =
      PauliOperator::X(0) * (PauliOperator::Y(1) + PauliOperator::Z(1));
  auto distributed = nested.distribute();
  std::cout << distributed->to_string() << std::endl;
  // "X(0) * Y(1) + X(0) * Z(1)"
  // end-cell-simplify
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-canonical
  // Get canonical string representation (little-endian: qubit 0 leftmost)
  auto expr = PauliOperator::X(0) * PauliOperator::Z(2);
  auto simplified = expr.simplify();

  auto canonical = simplified->to_canonical_string(4);  // 4 qubits total
  std::cout << canonical << std::endl;                  // "XIZI"

  // Get coefficient and string pairs for each term
  auto terms = simplified->to_canonical_terms(4);
  // terms[0] = {(1+0i), "XIZI"}

  // For sums, get all terms
  auto H2 = PauliOperator::X(0) + PauliOperator::Z(1);
  auto sum_terms = H2.to_canonical_terms(2);
  // sum_terms = [{(1+0i), "XI"}, {(1+0i), "IZ"}]
  // end-cell-canonical
  // --------------------------------------------------------------------------------------------

  return 0;
}
