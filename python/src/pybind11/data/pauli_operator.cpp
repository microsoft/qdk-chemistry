// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/pauli_operator.hpp>

namespace py = pybind11;

void bind_pauli_operator(pybind11::module& data) {
  using namespace qdk::chemistry::data;

  // Base class PauliOperatorExpression
  py::class_<PauliOperatorExpression, std::unique_ptr<PauliOperatorExpression>>
      pauli_expr(data, "PauliOperatorExpression", R"(
Base class for Pauli operator expressions.

This abstract class serves as the base for all Pauli operator expressions,
including single Pauli operators, products of operators, and sums of operators.
)");

  pauli_expr.def("__str__", &PauliOperatorExpression::to_string)
      .def("__repr__", &PauliOperatorExpression::to_string)
      .def(
          "distribute",
          [](const PauliOperatorExpression& self)
              -> std::unique_ptr<SumPauliOperatorExpression> {
            return self.distribute();
          },
          R"(
Distribute products over sums in the expression.

Returns:
    SumPauliOperatorExpression: The distributed expression as a sum of products.
)")
      .def(
          "simplify",
          [](const PauliOperatorExpression& self)
              -> std::unique_ptr<PauliOperatorExpression> {
            return self.simplify();
          },
          R"(
Simplify the expression by combining like terms and applying Pauli algebra rules.

Returns:
    PauliOperatorExpression: The simplified expression.
)")
      .def("is_pauli_operator", &PauliOperatorExpression::is_pauli_operator,
           "Check if this expression is a single Pauli operator.")
      .def("is_product_expression",
           &PauliOperatorExpression::is_product_expression,
           "Check if this expression is a product of operators.")
      .def("is_sum_expression", &PauliOperatorExpression::is_sum_expression,
           "Check if this expression is a sum of operators.")
      .def("is_distributed", &PauliOperatorExpression::is_distributed,
           "Check if this expression is in distributed form (sum of products).")
      .def("min_qubit_index", &PauliOperatorExpression::min_qubit_index,
           R"(
Return the minimum qubit index referenced in this expression.

Returns:
    int: The minimum qubit index.

Raises:
    RuntimeError: If the expression is empty.
)")
      .def("max_qubit_index", &PauliOperatorExpression::max_qubit_index,
           R"(
Return the maximum qubit index referenced in this expression.

Returns:
    int: The maximum qubit index.

Raises:
    RuntimeError: If the expression is empty.
)")
      .def("num_qubits", &PauliOperatorExpression::num_qubits,
           R"(
Return the number of qubits spanned by this expression.

Returns:
    int: max_qubit_index() - min_qubit_index() + 1, or 0 if empty.
)")
      .def(
          "to_canonical_string",
          [](const PauliOperatorExpression& self, std::uint64_t num_qubits) {
            return self.to_canonical_string(num_qubits);
          },
          py::arg("num_qubits"),
          R"(
Return the canonical string representation of this expression.

The canonical string is a sequence of characters representing the Pauli
operators on each qubit, in little-endian order (qubit 0 is leftmost).
Identity operators are represented as 'I'.

Args:
    num_qubits: The total number of qubits to represent.

Returns:
    str: A string of length num_qubits, e.g., "XIZI" for X(0)*Z(2) on 4 qubits.
)")
      .def(
          "to_canonical_string",
          [](const PauliOperatorExpression& self, std::uint64_t min_qubit,
             std::uint64_t max_qubit) {
            return self.to_canonical_string(min_qubit, max_qubit);
          },
          py::arg("min_qubit"), py::arg("max_qubit"),
          R"(
Return the canonical string representation for a qubit range.

Args:
    min_qubit: The minimum qubit index to include.
    max_qubit: The maximum qubit index to include (inclusive).

Returns:
    str: A string of length (max_qubit - min_qubit + 1).
)")
      .def(
          "to_canonical_terms",
          [](const PauliOperatorExpression& self, std::uint64_t num_qubits) {
            return self.to_canonical_terms(num_qubits);
          },
          py::arg("num_qubits"),
          R"(
Return a list of (coefficient, canonical_string) tuples.

Args:
    num_qubits: The total number of qubits to represent.

Returns:
    list[tuple[complex, str]]: A list of tuples where each tuple contains
        the coefficient and canonical string for each term.

Examples:
    >>> X0 = PauliOperator.X(0)
    >>> X0.to_canonical_terms(2)
    [((1+0j), 'XI')]
)")
      .def(
          "to_canonical_terms",
          [](const PauliOperatorExpression& self) {
            return self.to_canonical_terms();
          },
          R"(
Return a list of (coefficient, canonical_string) tuples.

Uses auto-detected qubit range based on min_qubit_index() and max_qubit_index().

Returns:
    list[tuple[complex, str]]: A list of tuples where each tuple contains
        the coefficient and canonical string for each term.

Examples:
    >>> expr = PauliOperator.X(0) + PauliOperator.Z(1)
    >>> expr.to_canonical_terms()
    [((1+0j), 'XI'), ((1+0j), 'IZ')]
)");

  // PauliOperator class
  py::class_<PauliOperator, PauliOperatorExpression,
             std::unique_ptr<PauliOperator>>
      pauli_op(data, "PauliOperator", R"(
A single Pauli operator (I, X, Y, or Z) acting on a specific qubit.

This class represents one of the four Pauli matrices acting on a single qubit.
Pauli operators can be combined using arithmetic operators to form expressions:

- Multiplication (*): Creates a product of operators
- Addition (+): Creates a sum of operators
- Subtraction (-): Creates a difference of operators
- Scalar multiplication: Multiplies by a complex coefficient

Examples:
    Create Pauli operators:

    >>> X0 = PauliOperator.X(0)  # X operator on qubit 0
    >>> Z1 = PauliOperator.Z(1)  # Z operator on qubit 1

    Form expressions:

    >>> expr = X0 * Z1           # Product X_0 * Z_1
    >>> expr = 0.5 * X0 + Z1     # Sum 0.5*X_0 + Z_1
)");

  pauli_op
      .def_static("I", &PauliOperator::I, py::arg("qubit_index"),
                  R"(
Create an identity operator on the specified qubit.

Args:
    qubit_index: The index of the qubit.

Returns:
    PauliOperator: The identity operator I on the given qubit.
)")
      .def_static("X", &PauliOperator::X, py::arg("qubit_index"),
                  R"(
Create a Pauli X operator on the specified qubit.

Args:
    qubit_index: The index of the qubit.

Returns:
    PauliOperator: The Pauli X operator on the given qubit.
)")
      .def_static("Y", &PauliOperator::Y, py::arg("qubit_index"),
                  R"(
Create a Pauli Y operator on the specified qubit.

Args:
    qubit_index: The index of the qubit.

Returns:
    PauliOperator: The Pauli Y operator on the given qubit.
)")
      .def_static("Z", &PauliOperator::Z, py::arg("qubit_index"),
                  R"(
Create a Pauli Z operator on the specified qubit.

Args:
    qubit_index: The index of the qubit.

Returns:
    PauliOperator: The Pauli Z operator on the given qubit.
)")
      .def_property_readonly("qubit_index", &PauliOperator::get_qubit_index,
                             "The index of the qubit this operator acts on.")
      .def("to_char", &PauliOperator::to_char,
           R"(
Return the character representation of this Pauli operator.

Returns:
    str: 'I', 'X', 'Y', or 'Z'.

Examples:
    >>> PauliOperator.X(0).to_char()
    'X'
    >>> PauliOperator.Z(1).to_char()
    'Z'
)")
      // Arithmetic operators
      .def(
          "__mul__",
          [](const PauliOperator& self, const PauliOperator& other) {
            return self * other;
          },
          py::arg("other"), "Multiply two Pauli operators.")
      .def(
          "__mul__",
          [](const PauliOperator& self,
             const ProductPauliOperatorExpression& other) {
            return self * other;
          },
          py::arg("other"), "Multiply with a product expression.")
      .def(
          "__mul__",
          [](const PauliOperator& self,
             const SumPauliOperatorExpression& other) { return self * other; },
          py::arg("other"), "Multiply with a sum expression.")
      .def(
          "__mul__",
          [](const PauliOperator& self, std::complex<double> scalar) {
            return self * scalar;
          },
          py::arg("scalar"), "Multiply by a scalar.")
      .def(
          "__rmul__",
          [](const PauliOperator& self, std::complex<double> scalar) {
            return scalar * self;
          },
          py::arg("scalar"), "Right multiply by a scalar.")
      .def(
          "__add__",
          [](const PauliOperator& self, const PauliOperator& other) {
            return self + other;
          },
          py::arg("other"), "Add two Pauli operators.")
      .def(
          "__add__",
          [](const PauliOperator& self,
             const ProductPauliOperatorExpression& other) {
            return self + other;
          },
          py::arg("other"), "Add with a product expression.")
      .def(
          "__add__",
          [](const PauliOperator& self,
             const SumPauliOperatorExpression& other) { return self + other; },
          py::arg("other"), "Add with a sum expression.")
      .def(
          "__sub__",
          [](const PauliOperator& self, const PauliOperator& other) {
            return self - other;
          },
          py::arg("other"), "Subtract two Pauli operators.")
      .def(
          "__sub__",
          [](const PauliOperator& self,
             const ProductPauliOperatorExpression& other) {
            return self - other;
          },
          py::arg("other"), "Subtract a product expression.")
      .def(
          "__sub__",
          [](const PauliOperator& self,
             const SumPauliOperatorExpression& other) { return self - other; },
          py::arg("other"), "Subtract a sum expression.")
      .def(
          "__neg__", [](const PauliOperator& self) { return -self; },
          "Negate the Pauli operator.")
      .def(
          "prune_threshold",
          [](const PauliOperator& self,
             double epsilon) -> std::unique_ptr<SumPauliOperatorExpression> {
            return self.prune_threshold(epsilon);
          },
          py::arg("epsilon"),
          R"(
Remove terms with coefficient magnitude below the threshold.

Args:
    epsilon: The threshold below which terms are removed.

Returns:
    SumPauliOperatorExpression: A new expression with small terms filtered out.
)");

  // ProductPauliOperatorExpression class
  py::class_<ProductPauliOperatorExpression, PauliOperatorExpression,
             std::unique_ptr<ProductPauliOperatorExpression>>
      product_expr(data, "ProductPauliOperatorExpression", R"(
A product of Pauli operator expressions with an optional coefficient.

This class represents a product of Pauli operators, such as X_0 * Z_1 * Y_2,
optionally multiplied by a complex coefficient. Products are created by
multiplying Pauli operators together.

Examples:
    >>> X0 = PauliOperator.X(0)
    >>> Z1 = PauliOperator.Z(1)
    >>> product = X0 * Z1       # X_0 * Z_1
    >>> scaled = 0.5j * product # (0.5j) * X_0 * Z_1
)");

  product_expr.def(py::init<>())
      .def(py::init<std::complex<double>>(), py::arg("coefficient"))
      .def_property_readonly(
          "coefficient", &ProductPauliOperatorExpression::get_coefficient,
          "The complex coefficient of this product expression.")
      // Arithmetic operators
      .def(
          "__mul__",
          [](const ProductPauliOperatorExpression& self,
             const PauliOperator& other) { return self * other; },
          py::arg("other"), "Multiply with a Pauli operator.")
      .def(
          "__mul__",
          [](const ProductPauliOperatorExpression& self,
             const ProductPauliOperatorExpression& other) {
            return self * other;
          },
          py::arg("other"), "Multiply two product expressions.")
      .def(
          "__mul__",
          [](const ProductPauliOperatorExpression& self,
             const SumPauliOperatorExpression& other) { return self * other; },
          py::arg("other"), "Multiply with a sum expression.")
      .def(
          "__mul__",
          [](const ProductPauliOperatorExpression& self,
             std::complex<double> scalar) { return self * scalar; },
          py::arg("scalar"), "Multiply by a scalar.")
      .def(
          "__rmul__",
          [](const ProductPauliOperatorExpression& self,
             std::complex<double> scalar) { return scalar * self; },
          py::arg("scalar"), "Right multiply by a scalar.")
      .def(
          "__add__",
          [](const ProductPauliOperatorExpression& self,
             const PauliOperator& other) { return self + other; },
          py::arg("other"), "Add with a Pauli operator.")
      .def(
          "__add__",
          [](const ProductPauliOperatorExpression& self,
             const ProductPauliOperatorExpression& other) {
            return self + other;
          },
          py::arg("other"), "Add two product expressions.")
      .def(
          "__add__",
          [](const ProductPauliOperatorExpression& self,
             const SumPauliOperatorExpression& other) { return self + other; },
          py::arg("other"), "Add with a sum expression.")
      .def(
          "__sub__",
          [](const ProductPauliOperatorExpression& self,
             const PauliOperator& other) { return self - other; },
          py::arg("other"), "Subtract a Pauli operator.")
      .def(
          "__sub__",
          [](const ProductPauliOperatorExpression& self,
             const ProductPauliOperatorExpression& other) {
            return self - other;
          },
          py::arg("other"), "Subtract a product expression.")
      .def(
          "__sub__",
          [](const ProductPauliOperatorExpression& self,
             const SumPauliOperatorExpression& other) { return self - other; },
          py::arg("other"), "Subtract a sum expression.")
      .def(
          "__neg__",
          [](const ProductPauliOperatorExpression& self) { return -self; },
          "Negate the product expression.")
      .def(
          "prune_threshold",
          [](const ProductPauliOperatorExpression& self,
             double epsilon) -> std::unique_ptr<SumPauliOperatorExpression> {
            return self.prune_threshold(epsilon);
          },
          py::arg("epsilon"),
          R"(
Remove terms with coefficient magnitude below the threshold.

Args:
    epsilon: The threshold below which terms are removed.

Returns:
    SumPauliOperatorExpression: A new expression with small terms filtered out.
)");

  // SumPauliOperatorExpression class
  py::class_<SumPauliOperatorExpression, PauliOperatorExpression,
             std::unique_ptr<SumPauliOperatorExpression>>
      sum_expr(data, "SumPauliOperatorExpression", R"(
A sum of Pauli operator expressions.

This class represents a sum of Pauli operators or products, such as
X_0 + Z_1 or 0.5*X_0*Z_1 + 0.3*Y_2. Sums are created by adding Pauli
operators or expressions together.

Examples:
    >>> X0 = PauliOperator.X(0)
    >>> Z1 = PauliOperator.Z(1)
    >>> sum_expr = X0 + Z1              # X_0 + Z_1
    >>> sum_expr = 0.5*X0 + 0.3*Z1      # 0.5*X_0 + 0.3*Z_1
)");

  sum_expr
      .def(py::init<>())
      // Arithmetic operators
      .def(
          "__mul__",
          [](const SumPauliOperatorExpression& self,
             const PauliOperator& other) { return self * other; },
          py::arg("other"), "Multiply with a Pauli operator.")
      .def(
          "__mul__",
          [](const SumPauliOperatorExpression& self,
             const ProductPauliOperatorExpression& other) {
            return self * other;
          },
          py::arg("other"), "Multiply with a product expression.")
      .def(
          "__mul__",
          [](const SumPauliOperatorExpression& self,
             const SumPauliOperatorExpression& other) { return self * other; },
          py::arg("other"), "Multiply two sum expressions.")
      .def(
          "__mul__",
          [](const SumPauliOperatorExpression& self,
             std::complex<double> scalar) { return self * scalar; },
          py::arg("scalar"), "Multiply by a scalar.")
      .def(
          "__rmul__",
          [](const SumPauliOperatorExpression& self,
             std::complex<double> scalar) { return scalar * self; },
          py::arg("scalar"), "Right multiply by a scalar.")
      .def(
          "__add__",
          [](const SumPauliOperatorExpression& self,
             const PauliOperator& other) { return self + other; },
          py::arg("other"), "Add with a Pauli operator.")
      .def(
          "__add__",
          [](const SumPauliOperatorExpression& self,
             const ProductPauliOperatorExpression& other) {
            return self + other;
          },
          py::arg("other"), "Add with a product expression.")
      .def(
          "__add__",
          [](const SumPauliOperatorExpression& self,
             const SumPauliOperatorExpression& other) { return self + other; },
          py::arg("other"), "Add two sum expressions.")
      .def(
          "__sub__",
          [](const SumPauliOperatorExpression& self,
             const PauliOperator& other) { return self - other; },
          py::arg("other"), "Subtract a Pauli operator.")
      .def(
          "__sub__",
          [](const SumPauliOperatorExpression& self,
             const ProductPauliOperatorExpression& other) {
            return self - other;
          },
          py::arg("other"), "Subtract a product expression.")
      .def(
          "__sub__",
          [](const SumPauliOperatorExpression& self,
             const SumPauliOperatorExpression& other) { return self - other; },
          py::arg("other"), "Subtract a sum expression.")
      .def(
          "__neg__",
          [](const SumPauliOperatorExpression& self) { return -self; },
          "Negate the sum expression.")
      .def(
          "prune_threshold",
          [](const SumPauliOperatorExpression& self,
             double epsilon) -> std::unique_ptr<SumPauliOperatorExpression> {
            return self.prune_threshold(epsilon);
          },
          py::arg("epsilon"),
          R"(
Remove terms with coefficient magnitude below the threshold.

Args:
    epsilon: The threshold below which terms are removed.

Returns:
    SumPauliOperatorExpression: A new expression with small terms filtered out.
)");
}
