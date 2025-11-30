// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <functional>
#include <qdk/chemistry/data/pauli_operator.hpp>

namespace qdk::chemistry::data {

namespace detail {

std::string pauli_operator_scalar_to_string(std::complex<double> coefficient) {
  std::ostringstream oss;
  if (coefficient.imag() == 0.0) {
    if (std::abs(coefficient.real() - 1.0) < 1e-10) {
      return "";
    } else if (std::abs(coefficient.real() + 1.0) < 1e-10) {
      return "-";
    }
    oss << coefficient.real();
  } else if (coefficient.real() == 0.0) {
    if (std::abs(coefficient.imag() - 1.0) < 1e-10) {
      return "i";
    } else if (std::abs(coefficient.imag() + 1.0) < 1e-10) {
      return "-i";
    }
    oss << coefficient.imag() << "i";
  } else {
    oss << "(" << coefficient.real();
    if (coefficient.imag() >= 0) oss << "+";
    oss << coefficient.imag() << "i)";
  }
  return oss.str();
}

}  // namespace detail

// ABC methods

bool PauliOperatorExpression::is_pauli_operator() const {
  return dynamic_cast<const PauliOperator*>(this) != nullptr;
}

bool PauliOperatorExpression::is_product_expression() const {
  return dynamic_cast<const ProductPauliOperatorExpression*>(this) != nullptr;
}

bool PauliOperatorExpression::is_sum_expression() const {
  return dynamic_cast<const SumPauliOperatorExpression*>(this) != nullptr;
}

bool PauliOperatorExpression::is_distributed() const {
  if (is_pauli_operator()) {
    return true;
  } else if (is_product_expression()) {
    const auto* prod =
        dynamic_cast<const ProductPauliOperatorExpression*>(this);
    for (const auto& factor : prod->get_factors()) {
      if (factor->is_sum_expression() || !factor->is_distributed()) {
        return false;
      }
    }
    return true;
  } else if (is_sum_expression()) {
    const auto* sum = dynamic_cast<const SumPauliOperatorExpression*>(this);
    for (const auto& term : sum->get_terms()) {
      if (!term->is_distributed()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// PauliOperator methods

PauliOperator::PauliOperator(std::uint8_t operator_type,
                             std::uint64_t qubit_index)
    : operator_type_(operator_type), qubit_index_(qubit_index) {}

std::string PauliOperator::to_string() const {
  switch (operator_type_) {
    case 0:
      return "I(" + std::to_string(qubit_index_) + ")";
    case 1:
      return "X(" + std::to_string(qubit_index_) + ")";
    case 2:
      return "Y(" + std::to_string(qubit_index_) + ")";
    case 3:
      return "Z(" + std::to_string(qubit_index_) + ")";
    default:
      throw std::runtime_error("Invalid Pauli operator type");
  }
}

std::unique_ptr<PauliOperatorExpression> PauliOperator::clone() const {
  return std::make_unique<PauliOperator>(*this);
}

std::unique_ptr<SumPauliOperatorExpression> PauliOperator::distribute() const {
  auto sum = std::make_unique<SumPauliOperatorExpression>();
  auto prod = std::make_unique<ProductPauliOperatorExpression>();
  prod->add_factor(this->clone());
  sum->add_term(std::move(prod));
  return sum;
}

std::unique_ptr<PauliOperatorExpression> PauliOperator::simplify() const {
  return this->clone();
}

// ProductPauliOperatorExpression methods
ProductPauliOperatorExpression::ProductPauliOperatorExpression()
    : coefficient_(1.0) {}

ProductPauliOperatorExpression::ProductPauliOperatorExpression(
    std::complex<double> coefficient)
    : coefficient_(coefficient) {}

ProductPauliOperatorExpression::ProductPauliOperatorExpression(
    const PauliOperatorExpression& left, const PauliOperatorExpression& right)
    : coefficient_(1.0) {
  factors_.push_back(left.clone());
  factors_.push_back(right.clone());
}

ProductPauliOperatorExpression::ProductPauliOperatorExpression(
    std::complex<double> coefficient, const PauliOperatorExpression& expr)
    : coefficient_(coefficient) {
  factors_.push_back(expr.clone());
}

ProductPauliOperatorExpression::ProductPauliOperatorExpression(
    const ProductPauliOperatorExpression& other)
    : coefficient_(other.coefficient_) {
  for (const auto& f : other.factors_) {
    factors_.push_back(f->clone());
  }
}

std::string ProductPauliOperatorExpression::to_string() const {
  std::string coeff_str = detail::pauli_operator_scalar_to_string(coefficient_);
  std::string factors_str;

  if (factors_.empty()) {
    return coeff_str.empty() ? "1" : coeff_str;
  }

  factors_str = factors_[0]->to_string();
  if (auto _ =
          dynamic_cast<const SumPauliOperatorExpression*>(factors_[0].get())) {
    factors_str = "(" + factors_str + ")";
  }
  for (size_t i = 1; i < factors_.size(); ++i) {
    auto _term_str = factors_[i]->to_string();
    if (auto _ = dynamic_cast<const SumPauliOperatorExpression*>(
            factors_[i].get())) {
      _term_str = "(" + _term_str + ")";
    }
    factors_str += " * " + _term_str;
  }

  if (coeff_str.empty()) {
    return factors_str;
  } else if (coeff_str == "-") {
    return "-" + factors_str;
  } else {
    return coeff_str + " * " + factors_str;
  }
}

std::unique_ptr<PauliOperatorExpression> ProductPauliOperatorExpression::clone()
    const {
  return std::make_unique<ProductPauliOperatorExpression>(*this);
}

std::unique_ptr<SumPauliOperatorExpression>
ProductPauliOperatorExpression::distribute() const {
  // Start with a sum containing a single empty product
  auto result = std::make_unique<SumPauliOperatorExpression>();
  auto initial = std::make_unique<ProductPauliOperatorExpression>(coefficient_);
  result->add_term(std::move(initial));

  // For each factor, distribute it across existing terms
  for (const auto& factor : factors_) {
    auto factor_dist = factor->distribute();
    auto new_result = std::make_unique<SumPauliOperatorExpression>();

    // Multiply each existing term with each term from factor distribution
    for (const auto& existing_term : result->get_terms()) {
      for (const auto& factor_term : factor_dist->get_terms()) {
        auto new_product = std::make_unique<ProductPauliOperatorExpression>();

        // Get coefficient from existing term
        std::complex<double> coeff(1.0);
        if (auto* prod = dynamic_cast<const ProductPauliOperatorExpression*>(
                existing_term.get())) {
          coeff = prod->get_coefficient();
          // Copy factors from existing term
          for (const auto& f : prod->get_factors()) {
            new_product->add_factor(f->clone());
          }
        } else {
          new_product->add_factor(existing_term->clone());
        }

        // Multiply coefficient from factor term
        if (auto* prod = dynamic_cast<const ProductPauliOperatorExpression*>(
                factor_term.get())) {
          coeff *= prod->get_coefficient();
          // Add factors from factor term
          for (const auto& f : prod->get_factors()) {
            new_product->add_factor(f->clone());
          }
        } else {
          new_product->add_factor(factor_term->clone());
        }

        new_product->set_coefficient(coeff);
        new_result->add_term(std::move(new_product));
      }
    }

    result = std::move(new_result);
  }

  return result;
}

std::unique_ptr<PauliOperatorExpression>
ProductPauliOperatorExpression::simplify() const {
  // If the product contains sums, distribute first and simplify the result
  if (!this->is_distributed()) {
    auto distributed = this->distribute();
    return distributed->simplify();
  }

  std::complex<double> new_coefficient = coefficient_;
  for (const auto& factor : factors_) {
    auto simplified_factor = factor->simplify();
    if (auto* prod = dynamic_cast<const ProductPauliOperatorExpression*>(
            simplified_factor.get())) {
      new_coefficient *= prod->get_coefficient();
    }
  }

  // Create new ProductPauliOperatorExpression with combined factor
  auto simplified_product =
      std::make_unique<ProductPauliOperatorExpression>(new_coefficient);
  for (const auto& factor : factors_) {
    auto simplified_factor = factor->simplify();
    if (auto* prod = dynamic_cast<ProductPauliOperatorExpression*>(
            simplified_factor.get())) {
      prod->set_coefficient(1.0);  // Remove coefficient from factor
    }
    simplified_product->add_factor(std::move(simplified_factor));
  }

  // stable sort factors and perform qubit-wise product simplification
  if (simplified_product->factors_.size() > 1) {
    // First unroll the products into a single product with all factors
    // Since is_distributed() is true, all factors are either PauliOperators
    // or ProductPauliOperatorExpressions containing only PauliOperators
    std::vector<std::unique_ptr<PauliOperatorExpression>> unrolled_factors;
    std::function<void(const std::unique_ptr<PauliOperatorExpression>&)>
        unroll = [&unrolled_factors, &unroll](
                     const std::unique_ptr<PauliOperatorExpression>& expr) {
          if (auto* pauli = dynamic_cast<const PauliOperator*>(expr.get())) {
            unrolled_factors.push_back(expr->clone());
          } else if (auto* prod =
                         dynamic_cast<const ProductPauliOperatorExpression*>(
                             expr.get())) {
            for (const auto& factor : prod->get_factors()) {
              unroll(factor);
            }
          }
        };

    for (const auto& factor : simplified_product->factors_) {
      unroll(factor);
    }

    simplified_product->factors_ = std::move(unrolled_factors);

    auto& factors = simplified_product->factors_;
    std::stable_sort(factors.begin(), factors.end(),
                     [](const auto& a, const auto& b) {
                       auto* pa = dynamic_cast<const PauliOperator*>(a.get());
                       auto* pb = dynamic_cast<const PauliOperator*>(b.get());
                       return pa->get_qubit_index() < pb->get_qubit_index();
                     });

    // Combine factors acting on the same qubit
    // Pauli multiplication rules:
    // I * P = P * I = P
    // P * P = I
    // X * Y = iZ, Y * X = -iZ
    // Y * Z = iX, Z * Y = -iX
    // Z * X = iY, X * Z = -iY
    std::vector<std::unique_ptr<PauliOperatorExpression>> combined_factors;
    std::complex<double> phase_factor(1.0, 0.0);
    const std::complex<double> imag_unit(0.0, 1.0);

    size_t i = 0;
    while (i < factors.size()) {
      auto* current = dynamic_cast<const PauliOperator*>(factors[i].get());
      std::uint64_t qubit = current->get_qubit_index();
      std::uint8_t result_type = current->get_operator_type();

      // Combine all operators on the same qubit
      size_t j = i + 1;
      while (j < factors.size()) {
        auto* next = dynamic_cast<const PauliOperator*>(factors[j].get());
        if (next->get_qubit_index() != qubit) break;

        std::uint8_t next_type = next->get_operator_type();

        // Multiply result_type with next_type
        if (next_type == 0) {
          // I * anything = anything
        } else if (result_type == 0) {
          // anything * I = anything
          result_type = next_type;
        } else if (result_type == next_type) {
          // P * P = I
          result_type = 0;
        } else {
          // Different non-identity Paulis
          // Use the Levi-Civita symbol for the phase
          // X=1, Y=2, Z=3
          // X*Y = iZ, Y*Z = iX, Z*X = iY (cyclic)
          // Y*X = -iZ, Z*Y = -iX, X*Z = -iY (anti-cyclic)
          int a = result_type;
          int b = next_type;
          // Compute the third Pauli type: {1,2,3} \ {a,b}
          int c = 6 - a - b;  // 1+2+3 = 6
          // Determine sign: cyclic (1->2->3->1) gives +i
          if ((a == 1 && b == 2) || (a == 2 && b == 3) || (a == 3 && b == 1)) {
            phase_factor *= imag_unit;
          } else {
            phase_factor *= -imag_unit;
          }
          result_type = static_cast<std::uint8_t>(c);
        }
        ++j;
      }

      // Only add non-identity operators to the result

      combined_factors.push_back(
          std::make_unique<PauliOperator>(result_type, qubit));

      i = j;
    }

    simplified_product->factors_ = std::move(combined_factors);
    simplified_product->coefficient_ *= phase_factor;
  }
  return simplified_product;
}

void ProductPauliOperatorExpression::multiply_coefficient(
    std::complex<double> c) {
  coefficient_ *= c;
}

void ProductPauliOperatorExpression::add_factor(
    std::unique_ptr<PauliOperatorExpression> factor) {
  factors_.push_back(std::move(factor));
}

const std::vector<std::unique_ptr<PauliOperatorExpression>>&
ProductPauliOperatorExpression::get_factors() const {
  return factors_;
}

std::complex<double> ProductPauliOperatorExpression::get_coefficient() const {
  return coefficient_;
}

void ProductPauliOperatorExpression::set_coefficient(std::complex<double> c) {
  coefficient_ = c;
}

// SumPauliOperatorExpression methods
SumPauliOperatorExpression::SumPauliOperatorExpression() = default;

SumPauliOperatorExpression::SumPauliOperatorExpression(
    const PauliOperatorExpression& left, const PauliOperatorExpression& right) {
  terms_.push_back(left.clone());
  terms_.push_back(right.clone());
}

SumPauliOperatorExpression::SumPauliOperatorExpression(
    const SumPauliOperatorExpression& other) {
  for (const auto& t : other.terms_) {
    terms_.push_back(t->clone());
  }
}

std::string SumPauliOperatorExpression::to_string() const {
  if (terms_.empty()) return "0";
  std::string result = terms_[0]->to_string();
  for (size_t i = 1; i < terms_.size(); ++i) {
    std::string term_str = terms_[i]->to_string();
    if (term_str[0] == '-') {
      result += " - " + term_str.substr(1);
    } else {
      result += " + " + term_str;
    }
  }
  return result;
}

std::unique_ptr<PauliOperatorExpression> SumPauliOperatorExpression::clone()
    const {
  return std::make_unique<SumPauliOperatorExpression>(*this);
}

std::unique_ptr<SumPauliOperatorExpression>
SumPauliOperatorExpression::distribute() const {
  auto result = std::make_unique<SumPauliOperatorExpression>();
  for (const auto& term : terms_) {
    auto distributed_term = term->distribute();
    // Add all terms from the distributed result to our result
    for (const auto& dist_term : distributed_term->get_terms()) {
      result->add_term(dist_term->clone());
    }
  }
  return result;
}

std::unique_ptr<PauliOperatorExpression> SumPauliOperatorExpression::simplify()
    const {
  // First distribute to get a sum of products
  auto distributed = this->distribute();

  auto simplified_sum = std::make_unique<SumPauliOperatorExpression>();
  for (const auto& term : distributed->get_terms()) {
    auto simplified_term = term->simplify();
    simplified_sum->add_term(std::move(simplified_term));
  }
  return simplified_sum;
}

void SumPauliOperatorExpression::add_term(
    std::unique_ptr<PauliOperatorExpression> term) {
  terms_.push_back(std::move(term));
}

const std::vector<std::unique_ptr<PauliOperatorExpression>>&
SumPauliOperatorExpression::get_terms() const {
  return terms_;
}

}  // namespace qdk::chemistry::data
