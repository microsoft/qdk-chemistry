// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <complex>
#include <concepts>
#include <cstdint>
#include <list>
#include <memory>
#include <qdk/chemistry/utils/hash.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief A sparse representation of a Pauli string (tensor product of Pauli
 * operators).
 *
 * Each element is a (qubit_index, operator_type) pair where operator_type is:
 * - 0: Identity (I) - typically not stored in sparse representation
 * - 1: Pauli X
 * - 2: Pauli Y
 * - 3: Pauli Z
 *
 * The vector is kept sorted by qubit_index for efficient comparison and
 * hashing. An empty SparsePauliWord represents the identity operator.
 *
 * Example: X(0) * Z(2) * Y(5) would be represented as:
 *   [(0, 1), (2, 3), (5, 2)]
 */
using SparsePauliWord = std::vector<std::pair<std::uint64_t, std::uint8_t>>;

/**
 * @brief Hash function for SparsePauliWord.
 *
 * Uses boost-style hash_combine for portable and efficient hash computation.
 */
struct SparsePauliWordHash {
  std::size_t operator()(const SparsePauliWord& word) const noexcept {
    std::size_t seed = 0;
    for (const auto& [qubit, op_type] : word) {
      seed = utils::hash_combine(seed, qubit, op_type);
    }
    return seed;
  }
};

/**
 * @brief Hash function for pairs of SparsePauliWord (used for multiplication
 * caching).
 *
 * Combines two SparsePauliWordHash results using hash_combine.
 */
struct SparsePauliWordPairHash {
  std::size_t operator()(
      const std::pair<SparsePauliWord, SparsePauliWord>& pair) const noexcept {
    SparsePauliWordHash hasher;
    std::size_t h1 = hasher(pair.first);
    std::size_t h2 = hasher(pair.second);
    return utils::hash_combine(h1, h2);
  }
};

// Forward declarations
class PauliOperator;
class ProductPauliOperatorExpression;
class SumPauliOperatorExpression;

/**
 * @brief Base interface for Pauli operator expressions.
 */
class PauliOperatorExpression {
 public:
  virtual ~PauliOperatorExpression() = default;

  /**
   * @brief Returns a string representation of this expression.
   *
   * Returns a human-readable string showing the structure of the arithmetic
   * expression. For example, a product of two Pauli operators might be
   * represented as "(X(0) * Z(1))".
   *
   * For the canonical string representation consistent with other frameworks,
   * use to_canonical_string().
   *
   * @return A string representing this expression.
   * @see to_canonical_string()
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Creates a deep copy of this expression.
   * @return A unique_ptr to the cloned expression.
   */
  virtual std::unique_ptr<PauliOperatorExpression> clone() const = 0;

  /**
   * @brief Distributes nested expressions to create a flat sum of products.
   *
   * For example, it transforms expressions like
   * (A + B) * (C - D) into A*C - A*D + B*C - B*D.
   *
   * @return A new SumPauliOperatorExpression in distributed form.
   */
  virtual std::unique_ptr<SumPauliOperatorExpression> distribute() const = 0;

  /**
   * @brief Simplifies the expression by combining like terms and carrying out
   *        qubit-wise multiplications.
   *
   * For example, it combines terms like 2*X(0)*Y(1) + 3*X(0)*Y(1) into
   * 5*X(0)*Y(1), and simplifies products like X(0)*X(0) into I(0).
   *
   * This function will also reorder terms into a canonical form. e.g.
   * X(1)*Y(0) will be reordered to Y(0)*X(1).
   *
   * By convention, distribute will be called internally before simplify to
   * ensure the expression is in a suitable form for simplification.
   *
   * @return A new simplified PauliOperatorExpression.
   */
  virtual std::unique_ptr<PauliOperatorExpression> simplify() const = 0;

  /**
   * @brief Returns a new expression with small-magnitude terms removed.
   * @param epsilon Terms with |coefficient| < epsilon are excluded.
   * @return A new SumPauliOperatorExpression with small terms filtered out.
   */
  virtual std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const = 0;

  /**
   * @brief Returns the minimum qubit index referenced in this expression.
   * @return The minimum qubit index.
   * @throws std::logic_error If the expression is empty.
   */
  virtual std::uint64_t min_qubit_index() const = 0;

  /**
   * @brief Returns the maximum qubit index referenced in this expression.
   * @return The maximum qubit index.
   * @throws std::logic_error If the expression is empty.
   */
  virtual std::uint64_t max_qubit_index() const = 0;

  /**
   * @brief Returns the number of qubits spanned by this expression.
   * @return max_qubit_index() - min_qubit_index() + 1, or 0 if empty.
   */
  virtual std::uint64_t num_qubits() const = 0;

  /**
   * @brief Returns the canonical string representation for a qubit range.
   *
   * The canonical string is a sequence of characters representing the Pauli
   * operators on each qubit, in little-endian order (qubit 0 is leftmost).
   * Identity operators are represented as 'I'.
   *
   * For example, for min_qubit=1 and max_qubit=3 for the expression
   *
   *  X(0) * Y(1) * Z(3) * X(4)
   *
   * the returned string would be "YIZ".
   *
   * @param min_qubit The minimum qubit index to include.
   * @param max_qubit The maximum qubit index to include (inclusive).
   * @return A canonical string representation.
   */
  virtual std::string to_canonical_string(std::uint64_t min_qubit,
                                          std::uint64_t max_qubit) const = 0;

  /**
   * @brief Returns the canonical string representation of this expression.
   *
   * Wraps to_canonical_string(0, max_qubit_index()+1).
   *
   * @param num_qubits The total number of qubits to represent.
   * @return A canonical string representation.
   */
  virtual std::string to_canonical_string(std::uint64_t num_qubits) const = 0;

  /**
   * @brief Returns a vector of (coefficient, canonical_string) pairs.
   *
   * For example, the expression 2*X(0)*Z(2) + 3i*Y(1) on 4 qubits would return:
   * [ (2, "XIZI"), (3i, "IYII") ]
   *
   * @param num_qubits The total number of qubits to represent.
   * @return Vector of pairs where each pair contains the coefficient and
   *         canonical string for each term.
   */
  virtual std::vector<std::pair<std::complex<double>, std::string>>
  to_canonical_terms(std::uint64_t num_qubits) const = 0;

  /**
   * @brief Returns a vector of (coefficient, canonical_string) pairs.
   *
   * Wraps to_canonical_terms(max_qubit_index()+1).
   *
   * @return Vector of pairs where each pair contains the coefficient and
   *         canonical string for each term.
   */
  virtual std::vector<std::pair<std::complex<double>, std::string>>
  to_canonical_terms() const = 0;

  /**
   * @brief Attempts to dynamically cast this expression to a PauliOperator.
   * @return Pointer to PauliOperator if successful, nullptr otherwise.
   */
  PauliOperator* as_pauli_operator();

  /**
   * @brief Attempts to dynamically cast this expression to a PauliOperator.
   * @return Pointer to PauliOperator if successful, nullptr otherwise.
   */
  const PauliOperator* as_pauli_operator() const;

  /**
   * @brief Attempts to dynamically cast this expression to a
   * ProductPauliOperatorExpression.
   * @return Pointer to ProductPauliOperatorExpression if successful, nullptr
   * otherwise.
   */
  ProductPauliOperatorExpression* as_product_expression();

  /**
   * @brief Attempts to dynamically cast this expression to a
   * ProductPauliOperatorExpression.
   * @return Pointer to ProductPauliOperatorExpression if successful, nullptr
   * otherwise.
   */
  const ProductPauliOperatorExpression* as_product_expression() const;

  /**
   * @brief Attempts to dynamically cast this expression to a
   * SumPauliOperatorExpression.
   * @return Pointer to SumPauliOperatorExpression if successful, nullptr
   * otherwise.
   */
  SumPauliOperatorExpression* as_sum_expression();

  /**
   * @brief Attempts to dynamically cast this expression to a
   * SumPauliOperatorExpression.
   * @return Pointer to SumPauliOperatorExpression if successful, nullptr
   * otherwise.
   */
  const SumPauliOperatorExpression* as_sum_expression() const;

  /**
   * @brief Returns whether this expression is a Pauli operator.
   * @return true if this is a PauliOperator, false otherwise.
   */
  inline bool is_pauli_operator() const {
    return as_pauli_operator() != nullptr;
  }

  /**
   * @brief Returns whether this expression is a product expression.
   * @return true if this is a ProductPauliOperatorExpression, false otherwise.
   */
  inline bool is_product_expression() const {
    return as_product_expression() != nullptr;
  }

  /**
   * @brief Returns whether this expression is a sum expression.
   * @return true if this is a SumPauliOperatorExpression, false otherwise.
   */
  inline bool is_sum_expression() const {
    return as_sum_expression() != nullptr;
  }

  /**
   * @brief Returns whether this expression is in distributed form.
   *
   * An expression is in distributed form when it contains no nested sums
   * inside products. Specifically:
   * - A PauliOperator is always distributed
   * - A ProductPauliOperatorExpression is distributed if all its factors
   *   are distributed and none are SumPauliOperatorExpressions
   * - A SumPauliOperatorExpression is distributed if all its terms are
   *   distributed
   *
   * Distributed form is required by to_canonical_string() and
   * to_canonical_terms(). Use distribute() to convert an expression to
   * distributed form.
   *
   * @return true if this expression is in distributed form, false otherwise.
   * @see distribute()
   */
  bool is_distributed() const;
};

/**
 * @brief Concept to check if a type is derived from PauliOperatorExpression.
 * @tparam T The type to check.
 */
template <typename T>
concept IsPauliOperatorExpression =
    std::derived_from<T, PauliOperatorExpression>;

/**
 * @brief A PauliOperatorExpression representing a single Pauli operator
 * acting on a qubit.
 *
 * This class serves as the leaf node in the expression tree for
 * PauliOperatorExpression trees. It represents one of the four Pauli operators:
 *
 * - Identity (I)
 * - Pauli-X (X)
 * - Pauli-Y (Y)
 * - Pauli-Z (Z)
 */
class PauliOperator : public PauliOperatorExpression {
 public:
  /**
   * @brief Constructs a PauliOperator with the specified type and qubit index.
   * @param operator_type The type of Pauli operator (0=I, 1=X, 2=Y, 3=Z).
   * @param qubit_index The index of the qubit this operator acts on.
   */
  PauliOperator(std::uint8_t operator_type, std::uint64_t qubit_index);

  /**
   * @brief Returns a string representation of this Pauli operator.
   *
   * For example, "X(0)" for a Pauli-X operator on qubit 0.
   *
   * See PauliOperatorExpression::to_string() for more details.
   * @return A string representing this Pauli operator.
   * @see PauliOperatorExpression::to_string()
   */
  std::string to_string() const override;

  /**
   * @brief Creates a deep copy of this Pauli operator.
   * @return A unique_ptr to the cloned Pauli operator.
   * @see PauliOperatorExpression::clone()
   */
  std::unique_ptr<PauliOperatorExpression> clone() const override;

  /**
   * @brief Distributes this Pauli operator.
   *
   * Since a single Pauli operator is already in simplest form, this method
   * simply returns a new SumPauliOperatorExpression containing this operator.
   *
   * @return A new SumPauliOperatorExpression containing this operator.
   * @see PauliOperatorExpression::distribute()
   */
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;

  /**
   * @brief Simplifies this Pauli operator.
   *
   * Since a single Pauli operator is already in simplest form, this method
   * simply returns a clone of this operator.
   *
   * @return A clone of this Pauli operator.
   * @see PauliOperatorExpression::simplify()
   */
  std::unique_ptr<PauliOperatorExpression> simplify() const override;

  /**
   * @brief Returns a sum containing this operator if it meets the threshold.
   *
   * Single Pauli operators have an implicit coefficient of 1.0.
   *
   * @param epsilon Terms with |coefficient| < epsilon are excluded.
   * @return A SumPauliOperatorExpression containing this operator if
   *         epsilon <= 1.0, or an empty sum otherwise.
   * @see PauliOperatorExpression::prune_threshold()
   */
  std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const override;

  /**
   * @brief Returns the minimum qubit index referenced in this operator.
   *
   * Since this operator acts on a single qubit, it simply returns that index.
   *
   * @return The qubit index this operator acts on.
   */
  std::uint64_t min_qubit_index() const override;

  /**
   * @brief Returns the maximum qubit index referenced in this operator.
   *
   * Since this operator acts on a single qubit, it simply returns that index.
   *
   * @return The qubit index this operator acts on.
   */
  std::uint64_t max_qubit_index() const override;

  /**
   * @brief Returns the number of qubits spanned by this operator.
   *
   * Since this operator acts on a single qubit, it always returns 1.
   *
   * @return 1
   */
  std::uint64_t num_qubits() const override;

  /**
   * @brief Returns the canonical string representation for a qubit range.
   *
   * See PauliOperatorExpression::to_canonical_string() for more details.
   *
   * @param min_qubit The minimum qubit index to include.
   * @param max_qubit The maximum qubit index to include (inclusive).
   * @return A string of length (max_qubit - min_qubit + 1).
   * @see PauliOperatorExpression::to_canonical_string()
   */
  std::string to_canonical_string(std::uint64_t min_qubit,
                                  std::uint64_t max_qubit) const override;

  /**
   * @brief Returns the canonical string representation of this Pauli operator.
   *
   * Wraps to_canonical_string(0, max_qubit_index()+1).
   *
   * @param num_qubits The total number of qubits to represent.
   * @return A string of length num_qubits.
   * @see PauliOperatorExpression::to_canonical_string()
   */
  std::string to_canonical_string(std::uint64_t num_qubits) const override;

  /**
   * @brief Returns a vector of (coefficient, canonical_string) pairs.
   *
   * For a single Pauli operator, this returns a single pair with
   * coefficient 1.0 and the canonical string representation if the requested
   * num_qubits includes the qubit this operator acts on. Otherwise, it returns
   * a single pair with coefficient 1.0 and a string of all identities.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return Vector of pairs where each pair contains the coefficient and
   *         canonical string for each term.
   * @see PauliOperatorExpression::to_canonical_terms()
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms(
      std::uint64_t num_qubits) const override;

  /**
   * @brief Returns a vector of (coefficient, canonical_string) pairs.
   *
   * Wraps to_canonical_terms(max_qubit_index()+1).
   *
   * @return Vector of pairs where each pair contains the coefficient and
   *         canonical string for each term.
   * @see PauliOperatorExpression::to_canonical_terms()
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms()
      const override;

  /**
   * @brief Returns the type of this Pauli operator.
   * @return The operator type (0=I, 1=X, 2=Y, 3=Z).
   */
  inline std::uint8_t get_operator_type() const { return operator_type_; }

  /**
   * @brief Returns the qubit index this Pauli operator acts on.
   * @return The qubit index.
   */
  inline std::uint64_t get_qubit_index() const { return qubit_index_; }

  /**
   * @brief Factory method to create an Identity operator on a specified qubit.
   *
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Identity operator on the specified
   * qubit.
   */
  inline static PauliOperator I(std::uint64_t qubit_index) {
    return PauliOperator(0, qubit_index);
  }

  /**
   * @brief Factory method to create a Pauli-X operator on a specified qubit.
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Pauli-X operator on the specified
   * qubit.
   */
  inline static PauliOperator X(std::uint64_t qubit_index) {
    return PauliOperator(1, qubit_index);
  }

  /**
   * @brief Factory method to create a Pauli-Y operator on a specified qubit.
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Pauli-Y operator on the specified
   * qubit.
   */
  inline static PauliOperator Y(std::uint64_t qubit_index) {
    return PauliOperator(2, qubit_index);
  }

  /**
   * @brief Factory method to create a Pauli-Z operator on a specified qubit.
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Pauli-Z operator on the specified
   * qubit.
   */
  inline static PauliOperator Z(std::uint64_t qubit_index) {
    return PauliOperator(3, qubit_index);
  }

  /**
   * @brief Returns the character representation of this Pauli operator.
   * @return 'I', 'X', 'Y', or 'Z'.
   * @throws std::runtime_error If the operator type is invalid (not 0-3).
   */
  char to_char() const;

 private:
  std::uint8_t operator_type_;  ///< e.g., 0 for I, 1 for X, 2 for Y, 3 for Z
  std::uint64_t qubit_index_;   ///< Index of the qubit this operator acts on
};

/**
 * @brief A PauliOperatorExpression representing Kronecker products of multiple
 * PauliOperatorExpression instances.
 *
 * For example, the expression X(0) * Y(1) represents the Pauli-X operator on
 * qubit 0 tensor product with the Pauli-Y operator on qubit 1, with an implicit
 * coefficient of 1.0.
 *
 * The class also supports nesting of expressions, e.g.,
 * 2.0 * (X(0) + Z(2)) * Y(1)
 *
 * where the left factor is SumPauliOperatorExpression and the right factor is
 * a PauliOperator.
 *
 * The product expression follows standard arithmetic rules for Kronecker
 * products:
 * - Distributive: A*(B + C) = A*B + A*C
 * - Associative: (A*B)*C = A*(B*C)
 * - Non-commutative: A*B != B*A in general for expressions acting on
 *   overlapping qubits.
 */
class ProductPauliOperatorExpression : public PauliOperatorExpression {
 public:
  /**
   * @brief Constructs an empty ProductPauliOperatorExpression with
   * coefficient 1.0.
   */
  ProductPauliOperatorExpression();

  /**
   * @brief Constructs a ProductPauliOperatorExpression with the specified
   * coefficient and no expression factors.
   * @param coefficient The scalar coefficient for this product expression.
   */
  ProductPauliOperatorExpression(std::complex<double> coefficient);

  /**
   * @brief Constructs a ProductPauliOperatorExpression representing the product
   * of two PauliOperatorExpression instances.
   *
   * For example:
   *  auto left = PauliOperator::X(0);
   *  auto right = SumPauliOperatorExpression(PauliOperator::Y(1),
   *    PauliOperator::Z(2));
   *  auto product = ProductPauliOperatorExpression(left, right);
   *
   * @param left The left PauliOperatorExpression factor.
   * @param right The right PauliOperatorExpression factor.
   */
  ProductPauliOperatorExpression(const PauliOperatorExpression& left,
                                 const PauliOperatorExpression& right);

  /**
   * @brief Constructs a ProductPauliOperatorExpression with the specified
   * coefficient and a single PauliOperatorExpression factor.
   * @param coefficient The scalar coefficient for this product expression.
   * @param expr The PauliOperatorExpression factor.
   */
  ProductPauliOperatorExpression(std::complex<double> coefficient,
                                 const PauliOperatorExpression& expr);

  /**
   * @brief Copy constructor. Deep copies all factors.
   * @param other The ProductPauliOperatorExpression to copy.
   */
  ProductPauliOperatorExpression(const ProductPauliOperatorExpression& other);

  /**
   * @brief Returns a human-readable string representation of this product.
   *
   * Renders the coefficient (if not 1) followed by factors joined with " * ".
   * Sum factors are wrapped in parentheses. An empty product returns "1" or
   * the coefficient string.
   *
   * To improve readability, coefficients sufficiently close (within fp64
   * epsilon: ~2.22e-16) to {-1, 1, i, -i} are rendered as "-", "", "i", or
   * "-i" respectively.
   *
   * @return A string like "2 * X(0) * Y(1)" or "(X(0) + Z(1)) * Y(2)".
   */
  std::string to_string() const override;

  /**
   * @brief Creates a deep copy of this product expression.
   * @return A unique_ptr to a new ProductPauliOperatorExpression.
   */
  std::unique_ptr<PauliOperatorExpression> clone() const override;

  /**
   * @brief Expands this product into a flat sum of products.
   *
   * Recursively distributes multiplication over addition for all factors.
   * For example, (X(0) + Y(0)) * Z(1) becomes X(0)*Z(1) + Y(0)*Z(1).
   *
   * The result is always a SumPauliOperatorExpression where each term is a
   * ProductPauliOperatorExpression containing only PauliOperator factors
   * (no nested sums).
   *
   * @return A new SumPauliOperatorExpression in distributed form.
   */
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;

  /**
   * @brief Simplifies this product by applying Pauli algebra rules.
   *
   * If the expression is not already distributed, distribute() is called
   * first.
   *
   * Simplification performs the following steps:
   * 1. Unrolls nested products into a flat list of PauliOperators
   * 2. Sorts factors by qubit index (stable sort)
   * 3. Combines operators on the same qubit using Pauli multiplication:
   *    - P * P = I for any Pauli P
   *    - X * Y = iZ, Y * Z = iX, Z * X = iY (cyclic)
   *    - Y * X = -iZ, Z * Y = -iX, X * Z = -iY (anti-cyclic)
   * 4. Strips identity operators from the result
   * 5. Accumulates phase factors into the coefficient
   *
   * @return A simplified ProductPauliOperatorExpression with sorted,
   *         non-identity factors and updated coefficient.
   */
  std::unique_ptr<PauliOperatorExpression> simplify() const override;

  /**
   * @brief Returns a sum containing this product if it meets the threshold.
   * @param epsilon Terms with |coefficient| < epsilon are excluded.
   * @return A SumPauliOperatorExpression containing this product, or an
   *         empty sum if excluded.
   */
  std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const override;

  /**
   * @brief Multiplies the coefficient by the given scalar.
   * @param c The scalar to multiply by.
   */
  void multiply_coefficient(std::complex<double> c);

  /**
   * @brief Appends a factor to this product.
   *
   * The factor is added to the end of the factor list. Ownership is
   * transferred to this expression.
   *
   * @param factor The expression to append.
   */
  void add_factor(std::unique_ptr<PauliOperatorExpression> factor);

  /**
   * @brief Pre-allocates capacity for the factor list.
   * @param capacity The number of factors to reserve space for.
   */
  void reserve_capacity(std::size_t capacity);

  /**
   * @brief Returns a const reference to the internal factor list.
   * @return Vector of expression factors in multiplication order.
   */
  const std::vector<std::unique_ptr<PauliOperatorExpression>>& get_factors()
      const;

  /**
   * @brief Returns the scalar coefficient of this product.
   * @return The complex coefficient.
   */
  std::complex<double> get_coefficient() const;

  /**
   * @brief Sets the scalar coefficient of this product.
   * @param c The new coefficient value.
   */
  void set_coefficient(std::complex<double> c);

  /**
   * @brief Returns the minimum qubit index referenced in this expression.
   * @return The minimum qubit index.
   * @throws std::logic_error If the expression has no factors.
   */
  std::uint64_t min_qubit_index() const override;

  /**
   * @brief Returns the maximum qubit index referenced in this expression.
   * @return The maximum qubit index.
   * @throws std::logic_error If the expression has no factors.
   */
  std::uint64_t max_qubit_index() const override;

  /**
   * @brief Returns the number of qubits spanned by this expression.
   * @return max_qubit_index() - min_qubit_index() + 1, or 0 if empty.
   */
  std::uint64_t num_qubits() const override;

  /**
   * @brief Returns the canonical string representation of this product.
   *
   * The canonical string is a sequence of characters representing the Pauli
   * operators on each qubit, in little-endian order (qubit 0 is leftmost).
   * Identity operators are represented as 'I'. The expression is simplified
   * internally before generating the string.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return A string of length num_qubits, e.g., "XIZI" for X(0)*Z(2) on 4
   * qubits.
   * @throws std::logic_error If the expression is not in distributed form.
   *         Call distribute() first.
   */
  std::string to_canonical_string(std::uint64_t num_qubits) const override;

  /**
   * @brief Returns the canonical string representation for a qubit range.
   *
   * @param min_qubit The minimum qubit index to include.
   * @param max_qubit The maximum qubit index to include (inclusive).
   * @return A string of length (max_qubit - min_qubit + 1).
   * @throws std::logic_error If the expression is not in distributed form.
   */
  std::string to_canonical_string(std::uint64_t min_qubit,
                                  std::uint64_t max_qubit) const override;

  /**
   * @brief Returns this product as a single (coefficient, canonical_string)
   * pair.
   * @param num_qubits The total number of qubits to represent.
   * @return A vector containing one pair.
   * @throws std::logic_error If the expression is not in distributed form.
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms(
      std::uint64_t num_qubits) const override;

  /**
   * @brief Returns this product as a single (coefficient, canonical_string)
   * pair.
   *
   * Uses max_qubit_index() + 1 as the qubit count. An empty product returns
   * a single term with coefficient and "I".
   *
   * @return A vector containing one pair.
   * @throws std::logic_error If the expression is not in distributed form.
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms()
      const override;

 private:
  std::complex<double> coefficient_;
  std::vector<std::unique_ptr<PauliOperatorExpression>> factors_;
};

/**
 * @brief A PauliOperatorExpression representing a sum of expressions.
 *
 * This class represents linear combinations of Pauli operator expressions.
 * For example, 2*X(0) + 3*Y(1)*Z(2) represents a sum of two terms.
 *
 * Terms can be any PauliOperatorExpression type, including nested sums
 * and products. The distribute() and simplify() methods can be used to
 * flatten and combine terms.
 */
class SumPauliOperatorExpression : public PauliOperatorExpression {
 public:
  /**
   * @brief Constructs an empty sum (represents zero).
   */
  SumPauliOperatorExpression();

  /**
   * @brief Constructs a sum of two expressions.
   * @param left The first term.
   * @param right The second term.
   */
  SumPauliOperatorExpression(const PauliOperatorExpression& left,
                             const PauliOperatorExpression& right);

  /**
   * @brief Copy constructor. Deep copies all terms.
   * @param other The SumPauliOperatorExpression to copy.
   */
  SumPauliOperatorExpression(const SumPauliOperatorExpression& other);

  /**
   * @brief Returns a human-readable string representation of this sum.
   *
   * Terms are joined with " + " or " - " depending on sign. An empty sum
   * returns "0".
   *
   * @return A string like "X(0) + 2 * Y(1) - Z(2)".
   */
  std::string to_string() const override;

  /**
   * @brief Creates a deep copy of this sum expression.
   * @return A unique_ptr to a new SumPauliOperatorExpression.
   */
  std::unique_ptr<PauliOperatorExpression> clone() const override;

  /**
   * @brief Distributes all terms and returns a flat sum of products.
   *
   * Calls distribute() on each term and collects all resulting products.
   * The result contains only ProductPauliOperatorExpression terms, each
   * containing only PauliOperator factors.
   *
   * @return A new SumPauliOperatorExpression in distributed form.
   */
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;

  /**
   * @brief Simplifies this sum by combining like terms.
   *
   * Simplification performs the following steps:
   * 1. Distributes the expression to flatten nested sums/products
   * 2. Simplifies each term individually (applying Pauli algebra)
   * 3. Collects like terms: terms with identical Pauli strings have their
   *    coefficients added together
   * 4. Removes terms with exactly zero coefficients
   *
   * Term ordering is preserved for non-duplicate terms.
   *
   * @return A simplified SumPauliOperatorExpression.
   */
  std::unique_ptr<PauliOperatorExpression> simplify() const override;

  /**
   * @brief Returns a sum with small-magnitude terms removed.
   *
   * Recursively processes nested sums. Bare PauliOperators have an implicit
   * coefficient of 1.0.
   *
   * @param epsilon Terms with |coefficient| < epsilon are excluded.
   * @return A new SumPauliOperatorExpression with small terms filtered out.
   */
  std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const override;

  /**
   * @brief Appends a term to this sum. Ownership is transferred.
   * @param term The expression to add.
   */
  void add_term(std::unique_ptr<PauliOperatorExpression> term);

  /**
   * @brief Pre-allocates capacity for the term list.
   * @param capacity The number of terms to reserve space for.
   */
  void reserve_capacity(std::size_t capacity);

  /**
   * @brief Returns a const reference to the internal term list.
   * @return Vector of expression terms in addition order.
   */
  const std::vector<std::unique_ptr<PauliOperatorExpression>>& get_terms()
      const;

  /**
   * @brief Returns the minimum qubit index referenced in this expression.
   * @return The minimum qubit index.
   * @throws std::logic_error If the expression has no terms.
   */
  std::uint64_t min_qubit_index() const override;

  /**
   * @brief Returns the maximum qubit index referenced in this expression.
   * @return The maximum qubit index.
   * @throws std::logic_error If the expression has no terms.
   */
  std::uint64_t max_qubit_index() const override;

  /**
   * @brief Returns the number of qubits spanned by this expression.
   * @return max_qubit_index() - min_qubit_index() + 1, or 0 if empty.
   */
  std::uint64_t num_qubits() const override;

  /**
   * @brief Returns the canonical string for a single-term sum.
   *
   * This method simplifies the sum first. It is intended for sums that
   * reduce to a single term after simplification.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return The canonical string of the single term, or "0" if empty.
   * @throws std::logic_error If the sum has more than one term after
   *         simplification. Use to_canonical_terms() for multi-term sums.
   */
  std::string to_canonical_string(std::uint64_t num_qubits) const override;

  /**
   * @brief Returns the canonical string for a single-term sum.
   *
   * @param min_qubit The minimum qubit index to include.
   * @param max_qubit The maximum qubit index to include (inclusive).
   * @return The canonical string, or "0" if empty.
   * @throws std::logic_error If the sum has more than one term after
   *         simplification.
   */
  std::string to_canonical_string(std::uint64_t min_qubit,
                                  std::uint64_t max_qubit) const override;

  /**
   * @brief Returns each term as a (coefficient, canonical_string) pair.
   *
   * Each term in the sum produces one entry. This method does not simplify
   * or combine like terms; call simplify() first if that is desired.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return Vector of (coefficient, canonical_string) pairs.
   * @throws std::logic_error If any term is not in distributed form.
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms(
      std::uint64_t num_qubits) const override;

  /**
   * @brief Returns each term as a (coefficient, canonical_string) pair.
   *
   * Uses the range [0, max_qubit_index()] for the canonical strings.
   * Returns an empty vector for an empty sum.
   *
   * @return Vector of (coefficient, canonical_string) pairs.
   * @throws std::logic_error If any term is not in distributed form.
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms()
      const override;

 private:
  std::vector<std::unique_ptr<PauliOperatorExpression>> terms_;
};

/**
 * @brief High-performance accumulator for Pauli terms with coefficient
 * combining.
 *
 * This class provides efficient accumulation of Pauli terms represented in
 * sparse format, with automatic coefficient combination for identical terms.
 * It is optimized for use cases like fermion-to-qubit mappings where many
 * term products are accumulated.
 *
 * Example usage:
 * @code
 *   PauliTermAccumulator acc;
 *   SparsePauliWord x0 = {{0, 1}};  // X(0)
 *   SparsePauliWord z1 = {{1, 3}};  // Z(1)
 *   acc.accumulate(x0, 0.5);         // Add 0.5 * X(0)
 *   acc.accumulate_product(x0, z1, 1.0);  // Add 1.0 * X(0) * Z(1)
 *   auto terms = acc.get_terms(1e-12);
 * @endcode
 */
class PauliTermAccumulator {
 public:
  /**
   * @brief Constructs an empty accumulator.
   */
  PauliTermAccumulator() = default;

  /**
   * @brief Accumulate a single term with the given coefficient.
   *
   * If a term with the same SparsePauliWord already exists, the coefficients
   * are added together.
   *
   * @param word The sparse Pauli word to accumulate.
   * @param coeff The coefficient to add.
   */
  void accumulate(const SparsePauliWord& word, std::complex<double> coeff);

  /**
   * @brief Accumulate the product of two terms with a scale factor.
   *
   * Computes word1 * word2 using Pauli algebra (with cached multiplication),
   * then accumulates the result scaled by the given factor.
   *
   * @param word1 The first sparse Pauli word.
   * @param word2 The second sparse Pauli word.
   * @param scale The scale factor to apply to the product.
   */
  void accumulate_product(const SparsePauliWord& word1,
                          const SparsePauliWord& word2,
                          std::complex<double> scale);

  /**
   * @brief Get all accumulated terms with coefficients above threshold.
   *
   * @param threshold Terms with |coefficient| < threshold are excluded.
   * @return Vector of (coefficient, SparsePauliWord) pairs.
   */
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> get_terms(
      double threshold = 0.0) const;

  /**
   * @brief Get accumulated terms as canonical strings.
   *
   * @param num_qubits The total number of qubits for string representation.
   * @param threshold Terms with |coefficient| < threshold are excluded.
   * @return Vector of (coefficient, canonical_string) pairs.
   */
  std::vector<std::pair<std::complex<double>, std::string>>
  get_terms_as_strings(std::uint64_t num_qubits, double threshold = 0.0) const;

  /**
   * @brief Clear all accumulated terms.
   */
  void clear();

  /**
   * @brief Get the number of unique terms currently accumulated.
   * @return The number of terms in the accumulator.
   */
  std::size_t size() const { return terms_.size(); }

  /// Default cache capacity
  static constexpr std::size_t kDefaultCacheCapacity = 10000;

  /**
   * @brief Set the capacity of the multiplication cache.
   *
   * The cache stores results of SparsePauliWord multiplications to avoid
   * redundant computation. When the cache exceeds capacity, oldest entries
   * are evicted (LRU policy).
   *
   * @param capacity Maximum number of entries in the cache.
   */
  void set_cache_capacity(std::size_t capacity);

  /**
   * @brief Clear the multiplication cache.
   */
  void clear_cache();

  /**
   * @brief Get the current number of entries in the multiplication cache.
   * @return The number of cached multiplications.
   */
  std::size_t cache_size() const;

  /**
   * @brief Multiply two SparsePauliWords using Pauli algebra.
   *
   * Computes word1 * word2, returning the phase factor and resulting word.
   * This method uses the instance cache for efficiency.
   *
   * @param word1 The first sparse Pauli word.
   * @param word2 The second sparse Pauli word.
   * @return A pair of (phase, result_word) where phase is the complex phase
   *         from Pauli multiplication rules.
   */
  std::pair<std::complex<double>, SparsePauliWord> multiply(
      const SparsePauliWord& word1, const SparsePauliWord& word2);

  /**
   * @brief Multiply two SparsePauliWords without caching.
   *
   * This is the core Pauli algebra implementation.
   *
   * @param word1 The first sparse Pauli word.
   * @param word2 The second sparse Pauli word.
   * @return A pair of (phase, result_word).
   */
  static std::pair<std::complex<double>, SparsePauliWord> multiply_uncached(
      const SparsePauliWord& word1, const SparsePauliWord& word2);

  /**
   * @brief Compute all Jordan-Wigner excitation operator terms.
   *
   * Computes E_pq = a†_p a_q for all (p, q) pairs in sparse format.
   * This is optimized to compute all N² terms in one call, avoiding
   * repeated pybind11 boundary crossings when invoked from Python.
   *
   * @param n_spin_orbitals Total number of spin orbitals (N).
   * @return Map from (p, q) indices to list of (coefficient, SparsePauliWord)
   * terms.
   */
  static std::unordered_map<
      std::pair<std::uint64_t, std::uint64_t>,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>,
      std::function<
          std::size_t(const std::pair<std::uint64_t, std::uint64_t>&)>>
  compute_all_jw_excitation_terms(std::uint64_t n_spin_orbitals);

  /**
   * @brief Compute all Bravyi-Kitaev excitation operator terms.
   *
   * Computes E_pq = a†_p a_q for all (p, q) pairs in sparse format.
   * Uses the provided BK index sets (parity, update, remainder) for each
   * orbital.
   *
   * @param n_spin_orbitals Total number of spin orbitals (N).
   * @param parity_sets Map from orbital index to parity set P(j).
   * @param update_sets Map from orbital index to update set U(j).
   * @param remainder_sets Map from orbital index to remainder set R(j).
   * @return Map from (p, q) indices to list of (coefficient, SparsePauliWord)
   * terms.
   */
  static std::unordered_map<
      std::pair<std::uint64_t, std::uint64_t>,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>,
      std::function<
          std::size_t(const std::pair<std::uint64_t, std::uint64_t>&)>>
  compute_all_bk_excitation_terms(
      std::uint64_t n_spin_orbitals,
      const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>&
          parity_sets,
      const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>&
          update_sets,
      const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>&
          remainder_sets);

 private:
  /// Accumulated terms: SparsePauliWord -> coefficient
  std::unordered_map<SparsePauliWord, std::complex<double>, SparsePauliWordHash>
      terms_;

  /// Cache types for LRU multiplication cache
  using CacheKey = std::pair<SparsePauliWord, SparsePauliWord>;
  using CacheValue = std::pair<std::complex<double>, SparsePauliWord>;
  using LRUIterator = std::list<CacheKey>::iterator;

  /// LRU cache for Pauli multiplication results
  std::unordered_map<CacheKey, std::pair<CacheValue, LRUIterator>,
                     SparsePauliWordPairHash>
      cache_map_;
  std::list<CacheKey> lru_list_;
  std::size_t cache_capacity_ = kDefaultCacheCapacity;
};

// --- Inline member function definitions moved out of class body ---
inline PauliOperator* PauliOperatorExpression::as_pauli_operator() {
  return dynamic_cast<PauliOperator*>(this);
}

inline const PauliOperator* PauliOperatorExpression::as_pauli_operator() const {
  return dynamic_cast<const PauliOperator*>(this);
}

inline ProductPauliOperatorExpression*
PauliOperatorExpression::as_product_expression() {
  return dynamic_cast<ProductPauliOperatorExpression*>(this);
}

inline const ProductPauliOperatorExpression*
PauliOperatorExpression::as_product_expression() const {
  return dynamic_cast<const ProductPauliOperatorExpression*>(this);
}

inline SumPauliOperatorExpression*
PauliOperatorExpression::as_sum_expression() {
  return dynamic_cast<SumPauliOperatorExpression*>(this);
}

inline const SumPauliOperatorExpression*
PauliOperatorExpression::as_sum_expression() const {
  return dynamic_cast<const SumPauliOperatorExpression*>(this);
}

/**
 * @defgroup PauliArithmetic Pauli Expression Arithmetic Operators
 * @brief Operators for building Pauli expressions using natural notation.
 *
 * Example:
 * @code
 * auto H = 0.5 * (X(0)*X(1) + Y(0)*Y(1) + Z(0)*Z(1));
 * auto simplified = H.simplify();
 * @endcode
 *
 * @note Results are unevaluated expression trees. Call simplify() to apply
 *       Pauli algebra and combine like terms.
 * @{
 */

/**
 * @brief Multiplies a product expression by a scalar.
 * @param s The scalar multiplier.
 * @param op The product expression.
 * @return A product with coefficient scaled by s.
 */
inline ProductPauliOperatorExpression operator*(
    std::complex<double> s, const ProductPauliOperatorExpression& op) {
  ProductPauliOperatorExpression result(op);
  result.set_coefficient(s * op.get_coefficient());
  return result;
}

/**
 * @brief Multiplies a product expression by a scalar.
 * @param op The product expression.
 * @param s The scalar multiplier.
 * @return A product with coefficient scaled by s.
 */
inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& op, std::complex<double> s) {
  return s * op;
}

/**
 * @brief Multiplies a product expression by a Pauli operator.
 * @param prod The product expression.
 * @param op The Pauli operator.
 * @return A product with op appended.
 */
inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& prod, const PauliOperator& op) {
  ProductPauliOperatorExpression result(prod);
  result.add_factor(op.clone());
  return result;
}

/**
 * @brief Multiplies a Pauli operator by a product expression.
 * @param op The Pauli operator.
 * @param prod The product expression.
 * @return A product with op prepended.
 */
inline ProductPauliOperatorExpression operator*(
    const PauliOperator& op, const ProductPauliOperatorExpression& prod) {
  ProductPauliOperatorExpression result(prod.get_coefficient(), op);
  for (const auto& factor : prod.get_factors()) {
    result.add_factor(factor->clone());
  }
  return result;
}

/**
 * @brief Multiplies two product expressions.
 * @param left The left operand.
 * @param right The right operand.
 * @return A product with combined factors and multiplied coefficients.
 */
inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& left,
    const ProductPauliOperatorExpression& right) {
  ProductPauliOperatorExpression result(left.get_coefficient() *
                                        right.get_coefficient());
  for (const auto& factor : left.get_factors()) {
    result.add_factor(factor->clone());
  }
  for (const auto& factor : right.get_factors()) {
    result.add_factor(factor->clone());
  }
  return result;
}

/**
 * @brief Negates a product expression.
 * @param op The product expression.
 * @return A product with negated coefficient.
 */
inline ProductPauliOperatorExpression operator-(
    const ProductPauliOperatorExpression& op) {
  ProductPauliOperatorExpression result(op);
  result.set_coefficient(-op.get_coefficient());
  return result;
}

/**
 * @brief Multiplies a product expression by a sum expression.
 * @param prod The product expression.
 * @param sum The sum expression.
 * @return A product containing sum as a factor (not distributed).
 */
inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& prod,
    const SumPauliOperatorExpression& sum) {
  ProductPauliOperatorExpression result(prod);
  result.add_factor(sum.clone());
  return result;
}

/**
 * @brief Multiplies a sum expression by a product expression.
 * @param sum The sum expression.
 * @param prod The product expression.
 * @return A product with sum as first factor (not distributed).
 */
inline ProductPauliOperatorExpression operator*(
    const SumPauliOperatorExpression& sum,
    const ProductPauliOperatorExpression& prod) {
  ProductPauliOperatorExpression result(prod.get_coefficient(), sum);
  for (const auto& factor : prod.get_factors()) {
    result.add_factor(factor->clone());
  }
  return result;
}

/**
 * @brief Multiplies an expression by a scalar.
 * @tparam Ex Expression type.
 * @param s The scalar coefficient.
 * @param op The expression.
 * @return A ProductPauliOperatorExpression with coefficient s.
 */
template <IsPauliOperatorExpression Ex>
  requires(!std::same_as<Ex, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator*(std::complex<double> s, const Ex& op) {
  return ProductPauliOperatorExpression(s, op);
}

/**
 * @brief Multiplies an expression by a scalar.
 * @tparam Ex Expression type.
 * @param op The expression.
 * @param s The scalar coefficient.
 * @return A ProductPauliOperatorExpression with coefficient s.
 */
template <IsPauliOperatorExpression Ex>
  requires(!std::same_as<Ex, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator*(const Ex& op, std::complex<double> s) {
  return s * op;
}

/**
 * @brief Multiplies two expressions.
 * @tparam Lhs Left expression type.
 * @tparam Rhs Right expression type.
 * @param left The left operand.
 * @param right The right operand.
 * @return A ProductPauliOperatorExpression containing both as factors.
 */
template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
  requires(!std::same_as<Lhs, ProductPauliOperatorExpression> &&
           !std::same_as<Rhs, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator*(const Lhs& left, const Rhs& right) {
  return ProductPauliOperatorExpression(left, right);
}

/**
 * @brief Adds two expressions.
 * @tparam Lhs Left expression type.
 * @tparam Rhs Right expression type.
 * @param left The left operand.
 * @param right The right operand.
 * @return A SumPauliOperatorExpression containing both as terms.
 */
template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
SumPauliOperatorExpression operator+(const Lhs& left, const Rhs& right) {
  return SumPauliOperatorExpression(left, right);
}

/**
 * @brief Subtracts two expressions.
 * @tparam Lhs Left expression type.
 * @tparam Rhs Right expression type.
 * @param left The left operand.
 * @param right The right operand.
 * @return A SumPauliOperatorExpression representing left + (-right).
 */
template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
SumPauliOperatorExpression operator-(const Lhs& left, const Rhs& right) {
  return left + (-1 * right);
}

/**
 * @brief Negates an expression.
 * @tparam Ex Expression type.
 * @param expr The expression to negate.
 * @return A ProductPauliOperatorExpression with coefficient -1.
 */
template <IsPauliOperatorExpression Ex>
  requires(!std::same_as<Ex, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator-(const Ex& expr) {
  return -1 * expr;
}

/** @} */  // end of PauliArithmetic group

}  // namespace qdk::chemistry::data
