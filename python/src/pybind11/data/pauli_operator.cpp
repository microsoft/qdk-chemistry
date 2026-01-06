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

  // PauliTermAccumulator class - high-performance accumulator for qubit
  // mappings
  py::class_<PauliTermAccumulator> accumulator(data, "PauliTermAccumulator", R"(
High-performance accumulator for Pauli terms with coefficient combining.

This class provides efficient accumulation of Pauli terms in sparse format,
with automatic coefficient combination for identical terms. It is optimized
for use cases like fermion-to-qubit mappings where many term products are
accumulated.

Sparse Pauli words are represented as lists of (qubit_index, operator_type)
tuples where operator_type is: 0=I (identity), 1=X, 2=Y, 3=Z. The list must
be sorted by qubit index, and identity operators are typically omitted
(an empty list represents the identity operator).

Example:
    >>> acc = PauliTermAccumulator()
    >>> x0 = [(0, 1)]  # X on qubit 0
    >>> z1 = [(1, 3)]  # Z on qubit 1
    >>> acc.accumulate(x0, 0.5)  # Add 0.5 * X(0)
    >>> acc.accumulate_product(x0, z1, 1.0)  # Add 1.0 * X(0) * Z(1)
    >>> terms = acc.get_terms_as_strings(4, 1e-12)
)");

  accumulator.def(py::init<>())
      .def(
          "accumulate",
          [](PauliTermAccumulator& self,
             const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word,
             std::complex<double> coeff) { self.accumulate(word, coeff); },
          py::arg("word"), py::arg("coeff"),
          R"(
Accumulate a single term with the given coefficient.

If a term with the same sparse Pauli word already exists, the coefficients
are added together.

Args:
    word: A sparse Pauli word as a list of (qubit_index, operator_type) tuples.
    coeff: The coefficient to add.
)")
      .def(
          "accumulate_product",
          [](PauliTermAccumulator& self,
             const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word1,
             const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word2,
             std::complex<double> scale) {
            self.accumulate_product(word1, word2, scale);
          },
          py::arg("word1"), py::arg("word2"), py::arg("scale"),
          R"(
Accumulate the product of two terms with a scale factor.

Computes word1 * word2 using Pauli algebra (with cached multiplication),
then accumulates the result scaled by the given factor.

Args:
    word1: The first sparse Pauli word.
    word2: The second sparse Pauli word.
    scale: The scale factor to apply to the product.
)")
      .def(
          "get_terms",
          [](const PauliTermAccumulator& self, double threshold) {
            return self.get_terms(threshold);
          },
          py::arg("threshold") = 0.0,
          R"(
Get all accumulated terms as sparse Pauli words.

Args:
    threshold: Terms with abs(coefficient) < threshold are excluded.

Returns:
    List of (coefficient, sparse_word) tuples.
)")
      .def(
          "get_terms_as_strings",
          [](const PauliTermAccumulator& self, std::uint64_t num_qubits,
             double threshold) {
            return self.get_terms_as_strings(num_qubits, threshold);
          },
          py::arg("num_qubits"), py::arg("threshold") = 0.0,
          R"(
Get all accumulated terms as canonical Pauli strings.

Args:
    num_qubits: Total number of qubits for string representation.
    threshold: Terms with abs(coefficient) < threshold are excluded.

Returns:
    List of (coefficient, canonical_string) tuples.
)")
      .def("clear", &PauliTermAccumulator::clear,
           "Clear all accumulated terms.")
      .def_property_readonly(
          "size", [](const PauliTermAccumulator& self) { return self.size(); },
          "Number of unique terms currently accumulated.")
      .def(
          "set_cache_capacity",
          [](PauliTermAccumulator& self, std::size_t capacity) {
            self.set_cache_capacity(capacity);
          },
          py::arg("capacity"),
          R"(
Set the capacity of this accumulator's multiplication cache.

The cache stores results of Pauli word multiplications to avoid redundant
computation. When the cache exceeds capacity, oldest entries are evicted
(LRU policy). Default capacity is 10000.

Args:
    capacity: Maximum number of entries in the cache.
)")
      .def(
          "clear_cache", [](PauliTermAccumulator& self) { self.clear_cache(); },
          "Clear this accumulator's multiplication cache.")
      .def_property_readonly(
          "cache_size",
          [](const PauliTermAccumulator& self) { return self.cache_size(); },
          "Get the current number of entries in this accumulator's "
          "multiplication cache.")
      .def(
          "multiply",
          [](PauliTermAccumulator& self,
             const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word1,
             const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word2) {
            return self.multiply(word1, word2);
          },
          py::arg("word1"), py::arg("word2"),
          R"(
Multiply two sparse Pauli words using Pauli algebra.

Uses this accumulator's cache for efficiency.

Args:
    word1: The first sparse Pauli word.
    word2: The second sparse Pauli word.

Returns:
    Tuple of (phase, result_word) where phase is the complex phase factor
    from Pauli multiplication rules.
)")
      .def_static(
          "multiply_uncached",
          [](const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word1,
             const std::vector<std::pair<std::uint64_t, std::uint8_t>>& word2) {
            return PauliTermAccumulator::multiply_uncached(word1, word2);
          },
          py::arg("word1"), py::arg("word2"),
          R"(
Multiply two sparse Pauli words using Pauli algebra without caching.

Args:
    word1: The first sparse Pauli word.
    word2: The second sparse Pauli word.

Returns:
    Tuple of (phase, result_word) where phase is the complex phase factor
    from Pauli multiplication rules.
)")
      .def_static(
          "compute_all_jw_excitation_terms",
          [](std::uint64_t n_spin_orbitals) {
            auto result = PauliTermAccumulator::compute_all_jw_excitation_terms(
                n_spin_orbitals);
            // Convert to Python-friendly format: dict[(p,q)] -> list[(coeff,
            // word)]
            py::dict py_result;
            for (auto& [key, terms] : result) {
              py::tuple py_key = py::make_tuple(key.first, key.second);
              py::list py_terms;
              for (auto& [coeff, word] : terms) {
                py_terms.append(py::make_tuple(coeff, word));
              }
              py_result[py_key] = py_terms;
            }
            return py_result;
          },
          py::arg("n_spin_orbitals"),
          R"(
Compute all Jordan-Wigner excitation terms E_pq = a†_p a_q for all (p, q) pairs.

This computes all N² one-body excitation terms in a single C++ call,
avoiding the overhead of many pybind11 boundary crossings.

For p == q: E_pp = (1/2)(I - Z_p)  [number operator]
For p != q: E_pq = (1/4)(XX + YY + iXY - iYX) with Z-strings between min(p,q) and max(p,q)

Args:
    n_spin_orbitals: Number of spin orbitals (qubits).

Returns:
    Dict mapping (p, q) tuples to lists of (coefficient, sparse_word) tuples.
)")
      .def_static(
          "compute_all_bk_excitation_terms",
          [](std::uint64_t n_spin_orbitals,
             const std::unordered_map<std::uint64_t,
                                      std::vector<std::uint64_t>>& parity_sets,
             const std::unordered_map<std::uint64_t,
                                      std::vector<std::uint64_t>>& update_sets,
             const std::unordered_map<
                 std::uint64_t, std::vector<std::uint64_t>>& remainder_sets) {
            auto result = PauliTermAccumulator::compute_all_bk_excitation_terms(
                n_spin_orbitals, parity_sets, update_sets, remainder_sets);
            // Convert to Python-friendly format
            py::dict py_result;
            for (auto& [key, terms] : result) {
              py::tuple py_key = py::make_tuple(key.first, key.second);
              py::list py_terms;
              for (auto& [coeff, word] : terms) {
                py_terms.append(py::make_tuple(coeff, word));
              }
              py_result[py_key] = py_terms;
            }
            return py_result;
          },
          py::arg("n_spin_orbitals"), py::arg("parity_sets"),
          py::arg("update_sets"), py::arg("remainder_sets"),
          R"(
Compute all Bravyi-Kitaev excitation terms E_pq = a†_p a_q for all (p, q) pairs.

This computes all N² one-body excitation terms in a single C++ call,
avoiding the overhead of many pybind11 boundary crossings.

The Bravyi-Kitaev transformation uses parity, update, and remainder sets
to encode fermionic operators efficiently. The ladder operators are::

    a†_j = (1/2)(Z_{P(j)} X_j X_{U(j)} - i Z_{R(j)} Y_j X_{U(j)})
    a_j  = (1/2)(Z_{P(j)} X_j X_{U(j)} + i Z_{R(j)} Y_j X_{U(j)})

Args:
    n_spin_orbitals: Number of spin orbitals (qubits).
    parity_sets: Dict mapping qubit index j to its parity set P(j).
    update_sets: Dict mapping qubit index j to its update set U(j).
    remainder_sets: Dict mapping qubit index j to its remainder set R(j).

Returns:
    Dict mapping (p, q) tuples to lists of (coefficient, sparse_word) tuples.
)");
}
