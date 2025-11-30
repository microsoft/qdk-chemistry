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
      .def(
          "is_distributed", &PauliOperatorExpression::is_distributed,
          "Check if this expression is in distributed form (sum of products).");

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
          py::arg("other"), "Subtract a sum expression.");

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
          py::arg("other"), "Subtract a sum expression.");

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
          py::arg("other"), "Subtract a sum expression.");
}
