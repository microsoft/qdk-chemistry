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
  virtual std::string to_string() const = 0;
  virtual std::unique_ptr<PauliOperatorExpression> clone() const = 0;
  virtual std::unique_ptr<SumPauliOperatorExpression> distribute() const = 0;
  virtual std::unique_ptr<PauliOperatorExpression> simplify() const = 0;

  bool is_pauli_operator() const;

  bool is_product_expression() const;

  bool is_sum_expression() const;

  bool is_distributed() const;
};

template <typename T>
concept IsPauliOperatorExpression =
    std::derived_from<T, PauliOperatorExpression>;

class PauliOperator : public PauliOperatorExpression {
 public:
  PauliOperator(std::uint8_t operator_type, std::uint64_t qubit_index);

  std::string to_string() const override;
  std::unique_ptr<PauliOperatorExpression> clone() const override;
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;
  std::unique_ptr<PauliOperatorExpression> simplify() const override;
  inline std::uint8_t get_operator_type() const { return operator_type_; }
  inline std::uint64_t get_qubit_index() const { return qubit_index_; }

  inline static PauliOperator I(std::uint64_t qubit_index) {
    return PauliOperator(0, qubit_index);
  }

  inline static PauliOperator X(std::uint64_t qubit_index) {
    return PauliOperator(1, qubit_index);
  }

  inline static PauliOperator Y(std::uint64_t qubit_index) {
    return PauliOperator(2, qubit_index);
  }

  inline static PauliOperator Z(std::uint64_t qubit_index) {
    return PauliOperator(3, qubit_index);
  }

 private:
  std::uint8_t operator_type_;  ///< e.g., 0 for I, 1 for X, 2 for Y, 3 for Z
  std::uint64_t qubit_index_;   ///< Index of the qubit this operator acts on
};

class ProductPauliOperatorExpression : public PauliOperatorExpression {
 public:
  ProductPauliOperatorExpression();
  ProductPauliOperatorExpression(std::complex<double> coefficient);
  ProductPauliOperatorExpression(const PauliOperatorExpression& left,
                                 const PauliOperatorExpression& right);
  ProductPauliOperatorExpression(std::complex<double> coefficient,
                                 const PauliOperatorExpression& expr);

  ProductPauliOperatorExpression(const ProductPauliOperatorExpression& other);

  std::string to_string() const override;
  std::unique_ptr<PauliOperatorExpression> clone() const override;
  std::unique_ptr<SumPauliOperatorExpression> distribute() const override;
  std::unique_ptr<PauliOperatorExpression> simplify() const override;

  void multiply_coefficient(std::complex<double> c);
  void add_factor(std::unique_ptr<PauliOperatorExpression> factor);

  const std::vector<std::unique_ptr<PauliOperatorExpression>>& get_factors()
      const;

  std::complex<double> get_coefficient() const;
  void set_coefficient(std::complex<double> c);

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

  void add_term(std::unique_ptr<PauliOperatorExpression> term);

  const std::vector<std::unique_ptr<PauliOperatorExpression>>& get_terms()
      const;

 private:
  std::vector<std::unique_ptr<PauliOperatorExpression>> terms_;
};

// Operator Overloads

template <IsPauliOperatorExpression Ex>
ProductPauliOperatorExpression operator*(std::complex<double> s, const Ex& op) {
  return ProductPauliOperatorExpression(s, op);
}

template <IsPauliOperatorExpression Ex>
ProductPauliOperatorExpression operator*(const Ex& op, std::complex<double> s) {
  return s * op;
}

template <IsPauliOperatorExpression Lhs, IsPauliOperatorExpression Rhs>
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

}  // namespace qdk::chemistry::data
