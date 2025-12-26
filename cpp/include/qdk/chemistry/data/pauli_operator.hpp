// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <complex>
#include <concepts>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace qdk::chemistry::data {

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
   * @brief Returns a new expression with terms having coefficient magnitude
   *        below the threshold removed.
   * @param epsilon The threshold below which terms are removed.
   * @return A new SumPauliOperatorExpression with small terms filtered out.
   */
  virtual std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const = 0;

  /**
   * @brief Returns the minimum qubit index referenced in this expression.
   * @return The minimum qubit index, or throws if the expression is empty.
   */
  virtual std::uint64_t min_qubit_index() const = 0;

  /**
   * @brief Returns the maximum qubit index referenced in this expression.
   * @return The maximum qubit index, or throws if the expression is empty.
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
   * @return true if this is in distributed form, false otherwise.
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
   * @brief Prunes this Pauli operator based on the threshold.
   *
   * Singple Puali operators are interpreted as having coefficient 1.0.
   * If the threshold epsilon is >= 1.0, this operator is pruned away.
   * Otherwise, it is retained.
   *
   * @param epsilon The threshold below which terms are removed.
   * @return A new SumPauliOperatorExpression containing this operator if
   * epsilon < 1.0, or an empty SumPauliOperatorExpression otherwise.
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
   * @brief Factory method to crete an Identity Pauli operator acting on a
   * specified qubit.
   *
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Identity operator on the specified
   * qubit.
   */
  inline static PauliOperator I(std::uint64_t qubit_index) {
    return PauliOperator(0, qubit_index);
  }

  /**
   * @brief Factory method to crete a Pauli-X operator acting on a specified
   * qubit.
   *
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Pauli-X operator on the specified
   * qubit.
   */
  inline static PauliOperator X(std::uint64_t qubit_index) {
    return PauliOperator(1, qubit_index);
  }

  /**
   * @brief Factory method to crete a Pauli-Y operator acting on a specified
   * qubit.
   *
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Pauli-Y operator on the specified
   * qubit.
   */
  inline static PauliOperator Y(std::uint64_t qubit_index) {
    return PauliOperator(2, qubit_index);
  }

  /**
   * @brief Factory method to crete a Pauli-Z operator acting on a specified
   * qubit.
   *
   * @param qubit_index The index of the qubit.
   * @return PauliOperator representing the Pauli-Z operator on the specified
   * qubit.
   */
  inline static PauliOperator Z(std::uint64_t qubit_index) {
    return PauliOperator(3, qubit_index);
  }

  /**
   * @brief Returns the character representation of this Pauli operator.
   * @return 'I', 'X', 'Y', or 'Z'
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
   * @brief Copy constructor.
   * @param other The ProductPauliOperatorExpression to copy.
   */
  ProductPauliOperatorExpression(const ProductPauliOperatorExpression& other);

  std::string to_string() const override;
  std::unique_ptr<PauliOperatorExpression> clone() const override;
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;
  std::unique_ptr<PauliOperatorExpression> simplify() const override;
  std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const override;

  void multiply_coefficient(std::complex<double> c);
  void add_factor(std::unique_ptr<PauliOperatorExpression> factor);

  const std::vector<std::unique_ptr<PauliOperatorExpression>>& get_factors()
      const;

  std::complex<double> get_coefficient() const;
  void set_coefficient(std::complex<double> c);

  /**
   * @brief Returns the minimum qubit index referenced in this expression.
   * @return The minimum qubit index, or throws if the expression is empty.
   */
  std::uint64_t min_qubit_index() const override;

  /**
   * @brief Returns the maximum qubit index referenced in this expression.
   * @return The maximum qubit index, or throws if the expression is empty.
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
   * Identity operators are represented as 'I'.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return A string of length num_qubits, e.g., "XIZI" for X(0)*Z(2) on 4
   * qubits.
   */
  std::string to_canonical_string(std::uint64_t num_qubits) const override;

  /**
   * @brief Returns the canonical string representation for a qubit range.
   *
   * @param min_qubit The minimum qubit index to include.
   * @param max_qubit The maximum qubit index to include (inclusive).
   * @return A string of length (max_qubit - min_qubit + 1).
   */
  std::string to_canonical_string(std::uint64_t min_qubit,
                                  std::uint64_t max_qubit) const override;

  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms(
      std::uint64_t num_qubits) const override;
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms()
      const override;

 private:
  std::complex<double> coefficient_;
  std::vector<std::unique_ptr<PauliOperatorExpression>> factors_;
};

class SumPauliOperatorExpression : public PauliOperatorExpression {
 public:
  SumPauliOperatorExpression();
  SumPauliOperatorExpression(const PauliOperatorExpression& left,
                             const PauliOperatorExpression& right);

  SumPauliOperatorExpression(const SumPauliOperatorExpression& other);

  std::string to_string() const override;
  std::unique_ptr<PauliOperatorExpression> clone() const override;
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;
  std::unique_ptr<PauliOperatorExpression> simplify() const override;
  std::unique_ptr<SumPauliOperatorExpression> prune_threshold(
      double epsilon) const override;

  void add_term(std::unique_ptr<PauliOperatorExpression> term);

  const std::vector<std::unique_ptr<PauliOperatorExpression>>& get_terms()
      const;

  /**
   * @brief Returns the minimum qubit index referenced in this expression.
   * @return The minimum qubit index, or throws if the expression is empty.
   */
  std::uint64_t min_qubit_index() const override;

  /**
   * @brief Returns the maximum qubit index referenced in this expression.
   * @return The maximum qubit index, or throws if the expression is empty.
   */
  std::uint64_t max_qubit_index() const override;

  /**
   * @brief Returns the number of qubits spanned by this expression.
   * @return max_qubit_index() - min_qubit_index() + 1, or 0 if empty.
   */
  std::uint64_t num_qubits() const override;

  /**
   * @brief Returns the canonical string representation of this sum.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return A string representation showing all terms in canonical form.
   */
  std::string to_canonical_string(std::uint64_t num_qubits) const override;

  /**
   * @brief Returns the canonical string representation for a qubit range.
   *
   * @param min_qubit The minimum qubit index to include.
   * @param max_qubit The maximum qubit index to include (inclusive).
   * @return A string representation showing all terms in canonical form.
   */
  std::string to_canonical_string(std::uint64_t min_qubit,
                                  std::uint64_t max_qubit) const override;

  /**
   * @brief Returns a vector of (coefficient, canonical_string) pairs.
   *
   * @param num_qubits The total number of qubits to represent.
   * @return Vector of pairs where each pair contains the coefficient and
   *         canonical string for each term.
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms(
      std::uint64_t num_qubits) const override;

  /**
   * @brief Returns a vector of (coefficient, canonical_string) pairs.
   *
   * Uses auto-detected qubit range based on min_qubit_index() and
   * max_qubit_index().
   *
   * @return Vector of pairs where each pair contains the coefficient and
   *         canonical string for each term.
   */
  std::vector<std::pair<std::complex<double>, std::string>> to_canonical_terms()
      const override;

 private:
  std::vector<std::unique_ptr<PauliOperatorExpression>> terms_;
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

// Operator Overloads

// ProductPauliOperatorExpression specializations keep products flat.

// Specialization for ProductPauliOperatorExpression: multiply coefficient
// directly
inline ProductPauliOperatorExpression operator*(
    std::complex<double> s, const ProductPauliOperatorExpression& op) {
  ProductPauliOperatorExpression result(op);
  result.set_coefficient(s * op.get_coefficient());
  return result;
}

inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& op, std::complex<double> s) {
  return s * op;
}

inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& prod, const PauliOperator& op) {
  ProductPauliOperatorExpression result(prod);
  result.add_factor(op.clone());
  return result;
}

inline ProductPauliOperatorExpression operator*(
    const PauliOperator& op, const ProductPauliOperatorExpression& prod) {
  ProductPauliOperatorExpression result(prod.get_coefficient(), op);
  for (const auto& factor : prod.get_factors()) {
    result.add_factor(factor->clone());
  }
  return result;
}

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

inline ProductPauliOperatorExpression operator-(
    const ProductPauliOperatorExpression& op) {
  ProductPauliOperatorExpression result(op);
  result.set_coefficient(-op.get_coefficient());
  return result;
}

inline ProductPauliOperatorExpression operator*(
    const ProductPauliOperatorExpression& prod,
    const SumPauliOperatorExpression& sum) {
  ProductPauliOperatorExpression result(prod);
  result.add_factor(sum.clone());
  return result;
}

inline ProductPauliOperatorExpression operator*(
    const SumPauliOperatorExpression& sum,
    const ProductPauliOperatorExpression& prod) {
  ProductPauliOperatorExpression result(prod.get_coefficient(), sum);
  for (const auto& factor : prod.get_factors()) {
    result.add_factor(factor->clone());
  }
  return result;
}

// Generic templates (excluded for ProductPauliOperatorExpression).

template <IsPauliOperatorExpression Ex>
  requires(!std::same_as<Ex, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator*(std::complex<double> s, const Ex& op) {
  return ProductPauliOperatorExpression(s, op);
}

template <IsPauliOperatorExpression Ex>
  requires(!std::same_as<Ex, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator*(const Ex& op, std::complex<double> s) {
  return s * op;
}

template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
  requires(!std::same_as<Lhs, ProductPauliOperatorExpression> &&
           !std::same_as<Rhs, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator*(const Lhs& left, const Rhs& right) {
  return ProductPauliOperatorExpression(left, right);
}

template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
SumPauliOperatorExpression operator+(const Lhs& left, const Rhs& right) {
  return SumPauliOperatorExpression(left, right);
}

template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
SumPauliOperatorExpression operator-(const Lhs& left, const Rhs& right) {
  return left + (-1 * right);
}

/**
 * @brief Unary negation operator for Pauli operator expressions.
 * @return A ProductPauliOperatorExpression representing -1 * expr.
 */
template <IsPauliOperatorExpression Ex>
  requires(!std::same_as<Ex, ProductPauliOperatorExpression>)
ProductPauliOperatorExpression operator-(const Ex& expr) {
  return -1 * expr;
}

}  // namespace qdk::chemistry::data
