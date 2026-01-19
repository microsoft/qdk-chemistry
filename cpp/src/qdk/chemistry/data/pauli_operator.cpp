// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <list>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace qdk::chemistry::data {

namespace detail {

/**
 * @brief Internal implementation of Pauli algebra.
 *
 * This class provides the core Pauli multiplication logic shared between
 * PauliTermAccumulator and excitation term computation.
 */
class PauliAlgebraImpl {
 public:
  /**
   * @brief Multiply two SparsePauliWords, returning phase and result.
   *
   * This is the core Pauli algebra implementation using merge-style algorithm.
   */
  static std::pair<std::complex<double>, SparsePauliWord> multiply(
      const SparsePauliWord& word1, const SparsePauliWord& word2) {
    const std::complex<double> imag_unit(0.0, 1.0);
    std::complex<double> phase(1.0, 0.0);
    SparsePauliWord result;
    result.reserve(word1.size() + word2.size());

    // Merge-style algorithm: both inputs are sorted by qubit index
    auto it1 = word1.begin();
    auto it2 = word2.begin();

    while (it1 != word1.end() && it2 != word2.end()) {
      if (it1->first < it2->first) {
        // Qubit only in word1
        result.push_back(*it1);
        ++it1;
      } else if (it2->first < it1->first) {
        // Qubit only in word2
        result.push_back(*it2);
        ++it2;
      } else {
        // Same qubit: multiply operators
        std::uint8_t op1 = it1->second;
        std::uint8_t op2 = it2->second;
        std::uint64_t qubit = it1->first;

        if (op1 == 0) {
          // I * P = P
          if (op2 != 0) result.emplace_back(qubit, op2);
        } else if (op2 == 0) {
          // P * I = P
          result.emplace_back(qubit, op1);
        } else if (op1 == op2) {
          // P * P = I (don't add to result, identity)
        } else {
          // Different non-identity Paulis: use Levi-Civita
          int a = op1;
          int b = op2;
          int c = 6 - a - b;  // 1+2+3 = 6, so third Pauli type
          // Cyclic: X*Y=iZ, Y*Z=iX, Z*X=iY
          // Anti-cyclic: Y*X=-iZ, Z*Y=-iX, X*Z=-iY
          if ((a == 1 && b == 2) || (a == 2 && b == 3) || (a == 3 && b == 1)) {
            phase *= imag_unit;
          } else {
            phase *= -imag_unit;
          }
          result.emplace_back(qubit, static_cast<std::uint8_t>(c));
        }
        ++it1;
        ++it2;
      }
    }

    // Append remaining elements
    while (it1 != word1.end()) {
      result.push_back(*it1);
      ++it1;
    }
    while (it2 != word2.end()) {
      result.push_back(*it2);
      ++it2;
    }

    return {phase, result};
  }
};

std::string pauli_operator_scalar_to_string(std::complex<double> coefficient) {
  constexpr double zero_tolerance = std::numeric_limits<double>::epsilon();
  std::ostringstream oss;
  if (coefficient.imag() == 0.0) {
    if (std::abs(coefficient.real() - 1.0) < zero_tolerance) {
      return "";
    } else if (std::abs(coefficient.real() + 1.0) < zero_tolerance) {
      return "-";
    }
    oss << coefficient.real();
  } else if (coefficient.real() == 0.0) {
    if (std::abs(coefficient.imag() - 1.0) < zero_tolerance) {
      return "i";
    } else if (std::abs(coefficient.imag() + 1.0) < zero_tolerance) {
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

bool PauliOperatorExpression::is_distributed() const {
  if (is_pauli_operator()) {
    return true;
  } else if (is_product_expression()) {
    const auto* prod = as_product_expression();
    for (const auto& factor : prod->get_factors()) {
      if (factor->is_sum_expression() || !factor->is_distributed()) {
        return false;
      }
    }
    return true;
  } else if (is_sum_expression()) {
    const auto* sum = as_sum_expression();
    for (const auto& term : sum->get_terms()) {
      if (!term->is_distributed()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

/*************************
 * PauliOperator methods *
 *************************/

PauliOperator::PauliOperator(std::uint8_t operator_type,
                             std::uint64_t qubit_index)
    : operator_type_(operator_type), qubit_index_(qubit_index) {}

std::string PauliOperator::to_string() const {
  return std::string(1, this->to_char()) + "(" + std::to_string(qubit_index_) +
         ")";
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

std::unique_ptr<SumPauliOperatorExpression> PauliOperator::prune_threshold(
    double epsilon) const {
  // A bare PauliOperator has implicit coefficient 1.0
  auto result = std::make_unique<SumPauliOperatorExpression>();
  if (1.0 >= epsilon) {
    auto prod = std::make_unique<ProductPauliOperatorExpression>();
    prod->add_factor(this->clone());
    result->add_term(std::move(prod));
  }
  return result;
}

char PauliOperator::to_char() const {
  switch (operator_type_) {
    case 0:
      return 'I';
    case 1:
      return 'X';
    case 2:
      return 'Y';
    case 3:
      return 'Z';
    default:
      throw std::runtime_error("Invalid Pauli operator type");
  }
}

std::uint64_t PauliOperator::min_qubit_index() const { return qubit_index_; }

std::uint64_t PauliOperator::max_qubit_index() const { return qubit_index_; }

std::uint64_t PauliOperator::num_qubits() const { return 1; }

std::string PauliOperator::to_canonical_string(std::uint64_t num_qubits) const {
  return to_canonical_string(0, num_qubits - 1);
}

std::string PauliOperator::to_canonical_string(std::uint64_t min_qubit,
                                               std::uint64_t max_qubit) const {
  std::string result(max_qubit - min_qubit + 1, 'I');
  if (qubit_index_ >= min_qubit && qubit_index_ <= max_qubit) {
    result[qubit_index_ - min_qubit] = to_char();
  }
  return result;
}

std::vector<std::pair<std::complex<double>, std::string>>
PauliOperator::to_canonical_terms(std::uint64_t num_qubits) const {
  return {{std::complex<double>(1.0, 0.0), to_canonical_string(num_qubits)}};
}

std::vector<std::pair<std::complex<double>, std::string>>
PauliOperator::to_canonical_terms() const {
  // Single Pauli operator spans 1 qubit, but we include from 0 to qubit_index_
  return to_canonical_terms(qubit_index_ + 1);
}

/******************************************
 * ProductPauliOperatorExpression methods *
 ******************************************/

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
  if (factors_[0]->is_sum_expression()) {
    factors_str = "(" + factors_str + ")";
  }
  for (size_t i = 1; i < factors_.size(); ++i) {
    auto _term_str = factors_[i]->to_string();
    if (factors_[i]->is_sum_expression()) {
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
  // Short-circuit: if already distributed (all factors are Pauli or flat
  // products), just wrap in a sum
  if (this->is_distributed()) {
    auto result = std::make_unique<SumPauliOperatorExpression>();
    result->add_term(this->clone());
    return result;
  }

  using FlatTerm =
      std::pair<std::complex<double>,
                std::vector<std::pair<std::uint64_t, std::uint8_t>>>;
  std::vector<FlatTerm> flat_terms;
  flat_terms.emplace_back(
      coefficient_, std::vector<std::pair<std::uint64_t, std::uint8_t>>{});

  // Helper to extract flat Pauli operators from an expression
  std::function<void(const PauliOperatorExpression*,
                     std::vector<std::pair<std::uint64_t, std::uint8_t>>&)>
      extract_paulis;
  extract_paulis =
      [&extract_paulis](
          const PauliOperatorExpression* expr,
          std::vector<std::pair<std::uint64_t, std::uint8_t>>& out) {
        if (const auto* pauli = expr->as_pauli_operator()) {
          out.emplace_back(pauli->get_qubit_index(),
                           pauli->get_operator_type());
        } else if (const auto* prod = expr->as_product_expression()) {
          for (const auto& f : prod->get_factors()) {
            extract_paulis(f.get(), out);
          }
        }
      };

  for (const auto& factor : factors_) {
    auto factor_dist = factor->distribute();
    const auto& factor_terms = factor_dist->get_terms();

    std::vector<FlatTerm> new_flat_terms;
    new_flat_terms.reserve(flat_terms.size() * factor_terms.size());

    for (const auto& existing : flat_terms) {
      for (const auto& factor_term : factor_terms) {
        FlatTerm new_term;
        new_term.first = existing.first;
        new_term.second = existing.second;

        if (const auto* prod = factor_term->as_product_expression()) {
          new_term.first *= prod->get_coefficient();
          new_term.second.reserve(new_term.second.size() +
                                  prod->get_factors().size());
          extract_paulis(factor_term.get(), new_term.second);
        } else if (const auto* pauli = factor_term->as_pauli_operator()) {
          new_term.second.emplace_back(pauli->get_qubit_index(),
                                       pauli->get_operator_type());
        }

        new_flat_terms.push_back(std::move(new_term));
      }
    }

    flat_terms = std::move(new_flat_terms);
  }

  // Convert flat representation back to expression tree
  auto result = std::make_unique<SumPauliOperatorExpression>();
  result->reserve_capacity(flat_terms.size());

  for (const auto& flat_term : flat_terms) {
    auto prod =
        std::make_unique<ProductPauliOperatorExpression>(flat_term.first);
    prod->reserve_capacity(flat_term.second.size());
    for (const auto& [qubit, op_type] : flat_term.second) {
      prod->add_factor(std::make_unique<PauliOperator>(op_type, qubit));
    }
    result->add_term(std::move(prod));
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
    if (auto* prod = simplified_factor->as_product_expression()) {
      new_coefficient *= prod->get_coefficient();
    }
  }

  // Create new ProductPauliOperatorExpression with combined factor
  auto simplified_product =
      std::make_unique<ProductPauliOperatorExpression>(new_coefficient);
  for (const auto& factor : factors_) {
    auto simplified_factor = factor->simplify();
    if (auto* prod = simplified_factor->as_product_expression()) {
      prod->set_coefficient(1.0);  // Remove coefficient from factor
    }
    simplified_product->add_factor(std::move(simplified_factor));
  }

  // Always unroll nested products into a flat list of PauliOperators
  // Since is_distributed() is true, all factors are either PauliOperators
  // or ProductPauliOperatorExpressions containing only PauliOperators
  std::vector<std::unique_ptr<PauliOperatorExpression>> unrolled_factors;
  std::function<void(const std::unique_ptr<PauliOperatorExpression>&)> unroll =
      [&unrolled_factors,
       &unroll](const std::unique_ptr<PauliOperatorExpression>& expr) {
        if (auto* pauli = expr->as_pauli_operator()) {
          unrolled_factors.push_back(expr->clone());
        } else if (auto* prod = expr->as_product_expression()) {
          for (const auto& factor : prod->get_factors()) {
            unroll(factor);
          }
        }
      };

  for (const auto& factor : simplified_product->factors_) {
    unroll(factor);
  }

  simplified_product->factors_ = std::move(unrolled_factors);

  // If we have factors, sort and combine them
  if (!simplified_product->factors_.empty()) {
    auto& factors = simplified_product->factors_;
    std::ranges::stable_sort(factors, [](const auto& a, const auto& b) {
      auto* pa = a->as_pauli_operator();
      auto* pb = b->as_pauli_operator();
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
      auto* current = factors[i]->as_pauli_operator();
      std::uint64_t qubit = current->get_qubit_index();
      std::uint8_t result_type = current->get_operator_type();

      // Combine all operators on the same qubit
      size_t j = i + 1;
      while (j < factors.size()) {
        auto* next = factors[j]->as_pauli_operator();
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
      if (result_type != 0) {
        combined_factors.push_back(
            std::make_unique<PauliOperator>(result_type, qubit));
      }

      i = j;
    }

    simplified_product->factors_ = std::move(combined_factors);
    simplified_product->coefficient_ *= phase_factor;
  }
  return simplified_product;
}

std::unique_ptr<SumPauliOperatorExpression>
ProductPauliOperatorExpression::prune_threshold(double epsilon) const {
  auto result = std::make_unique<SumPauliOperatorExpression>();
  if (std::abs(coefficient_) >= epsilon) {
    result->add_term(this->clone());
  }
  return result;
}

void ProductPauliOperatorExpression::multiply_coefficient(
    std::complex<double> c) {
  coefficient_ *= c;
}

void ProductPauliOperatorExpression::add_factor(
    std::unique_ptr<PauliOperatorExpression> factor) {
  factors_.push_back(std::move(factor));
}

void ProductPauliOperatorExpression::reserve_capacity(std::size_t capacity) {
  factors_.reserve(capacity);
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

std::uint64_t ProductPauliOperatorExpression::min_qubit_index() const {
  if (factors_.empty()) {
    throw std::logic_error(
        "min_qubit_index() called on empty ProductPauliOperatorExpression");
  }
  std::uint64_t min_idx = std::numeric_limits<std::uint64_t>::max();
  for (const auto& factor : factors_) {
    if (auto* pauli = factor->as_pauli_operator()) {
      min_idx = std::min(min_idx, pauli->get_qubit_index());
    } else if (auto* prod = factor->as_product_expression()) {
      if (!prod->get_factors().empty()) {
        min_idx = std::min(min_idx, prod->min_qubit_index());
      }
    } else if (auto* sum = factor->as_sum_expression()) {
      if (!sum->get_terms().empty()) {
        min_idx = std::min(min_idx, sum->min_qubit_index());
      }
    }
  }
  return min_idx;
}

std::uint64_t ProductPauliOperatorExpression::max_qubit_index() const {
  if (factors_.empty()) {
    throw std::logic_error(
        "max_qubit_index() called on empty ProductPauliOperatorExpression");
  }
  std::uint64_t max_idx = 0;
  for (const auto& factor : factors_) {
    if (auto* pauli = factor->as_pauli_operator()) {
      max_idx = std::max(max_idx, pauli->get_qubit_index());
    } else if (auto* prod = factor->as_product_expression()) {
      if (!prod->get_factors().empty()) {
        max_idx = std::max(max_idx, prod->max_qubit_index());
      }
    } else if (auto* sum = factor->as_sum_expression()) {
      if (!sum->get_terms().empty()) {
        max_idx = std::max(max_idx, sum->max_qubit_index());
      }
    }
  }
  return max_idx;
}

std::uint64_t ProductPauliOperatorExpression::num_qubits() const {
  if (factors_.empty()) {
    return 0;
  }
  return max_qubit_index() - min_qubit_index() + 1;
}

std::string ProductPauliOperatorExpression::to_canonical_string(
    std::uint64_t num_qubits) const {
  return to_canonical_string(0, num_qubits - 1);
}

std::string ProductPauliOperatorExpression::to_canonical_string(
    std::uint64_t min_qubit, std::uint64_t max_qubit) const {
  // Check if the term is distributed
  if (!this->is_distributed()) {
    throw std::logic_error(
        "to_canonical_string() requires a distributed "
        "ProductPauliOperatorExpression. "
        "Call distribute() first.");
  }

  // Build a map from qubit index to operator type
  std::vector<char> result(max_qubit - min_qubit + 1, 'I');

  auto simplified = this->simplify();
  auto* simplified_product = simplified->as_product_expression();
  for (const auto& factor : simplified_product->get_factors()) {
    if (auto* pauli = factor->as_pauli_operator()) {
      std::uint64_t idx = pauli->get_qubit_index();
      if (idx >= min_qubit && idx <= max_qubit) {
        result[idx - min_qubit] = pauli->to_char();
      }
    }
  }

  return std::string(result.begin(), result.end());
}

std::vector<std::pair<std::complex<double>, std::string>>
ProductPauliOperatorExpression::to_canonical_terms(
    std::uint64_t num_qubits) const {
  // Simplify first to compute phase factors from Pauli multiplication
  // simplify() already includes the original coefficient_ in the result
  auto simplified = this->simplify();
  auto* simplified_product = simplified->as_product_expression();
  return {{simplified_product->get_coefficient(),
           simplified_product->to_canonical_string(num_qubits)}};
}

std::vector<std::pair<std::complex<double>, std::string>>
ProductPauliOperatorExpression::to_canonical_terms() const {
  if (factors_.empty()) {
    // Pure scalar - return single term with all identities
    return {{coefficient_, "I"}};
  }
  std::uint64_t effective_num_qubits = max_qubit_index() + 1;
  return to_canonical_terms(effective_num_qubits);
}

/***************************************
 *  SumPauliOperatorExpression methods *
 ***************************************/

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
  // Short-circuit: if already distributed, just clone
  if (this->is_distributed()) {
    return std::make_unique<SumPauliOperatorExpression>(*this);
  }

  auto result = std::make_unique<SumPauliOperatorExpression>();
  result->reserve_capacity(terms_.size() * 2);

  for (const auto& term : terms_) {
    auto distributed_term = term->distribute();
    for (const auto& dist_term : distributed_term->get_terms()) {
      result->add_term(dist_term->clone());
    }
  }
  return result;
}

std::unique_ptr<PauliOperatorExpression> SumPauliOperatorExpression::simplify()
    const {
  // Helper function to create a term key from a simplified product
  // The key is a sorted vector of (qubit_index, operator_type) pairs
  auto make_term_key =
      [](const ProductPauliOperatorExpression* prod) -> SparsePauliWord {
    SparsePauliWord key;
    key.reserve(prod->get_factors().size());
    for (const auto& factor : prod->get_factors()) {
      if (auto* pauli = factor->as_pauli_operator()) {
        key.emplace_back(pauli->get_qubit_index(), pauli->get_operator_type());
      }
    }
    return key;
  };

  // First simplify all terms individually
  std::vector<std::unique_ptr<ProductPauliOperatorExpression>> simplified_terms;
  simplified_terms.reserve(terms_.size());

  // Helper function to add a simplified expression to simplified_terms
  std::function<void(std::unique_ptr<PauliOperatorExpression>)>
      add_simplified_term;
  add_simplified_term = [&simplified_terms, &add_simplified_term](
                            std::unique_ptr<PauliOperatorExpression> expr) {
    if (auto* prod = expr->as_product_expression()) {
      simplified_terms.push_back(
          std::make_unique<ProductPauliOperatorExpression>(*prod));
    } else if (auto* pauli = expr->as_pauli_operator()) {
      auto wrapped = std::make_unique<ProductPauliOperatorExpression>();
      wrapped->add_factor(std::make_unique<PauliOperator>(*pauli));
      simplified_terms.push_back(std::move(wrapped));
    } else if (auto* sum = expr->as_sum_expression()) {
      for (const auto& term : sum->get_terms()) {
        add_simplified_term(term->clone());
      }
    }
  };

  // Check if already distributed - if so, we can skip the expensive
  // distribute() call
  const bool already_distributed = this->is_distributed();

  if (already_distributed) {
    // Skip distribute(), simplify terms directly from this sum
    for (const auto& term : terms_) {
      add_simplified_term(term->simplify());
    }
  } else {
    auto distributed = this->distribute();
    for (const auto& term : distributed->get_terms()) {
      auto simplified_term = term->simplify();
      add_simplified_term(std::move(simplified_term));
    }
  }

  // Use SparsePauliWord (same as TermKey) with SparsePauliWordHash from header
  std::unordered_map<SparsePauliWord, std::size_t, SparsePauliWordHash>
      term_index_map;
  std::vector<std::tuple<SparsePauliWord, std::complex<double>,
                         std::unique_ptr<ProductPauliOperatorExpression>>>
      collected_terms;

  for (auto& term : simplified_terms) {
    auto key = make_term_key(term.get());
    auto it = term_index_map.find(key);

    if (it == term_index_map.end()) {
      auto coeff = term->get_coefficient();
      term->set_coefficient(1.0);
      std::size_t idx = collected_terms.size();
      term_index_map.emplace(key, idx);
      collected_terms.emplace_back(std::move(key), coeff, std::move(term));
    } else {
      std::get<1>(collected_terms[it->second]) += term->get_coefficient();
    }
  }

  // Build the simplified sum, excluding exactly-zero coefficient terms
  // Only remove terms where coefficient is exactly zero
  auto simplified_sum = std::make_unique<SumPauliOperatorExpression>();
  for (auto& [key, coeff, term] : collected_terms) {
    // Skip only if coefficient is exactly zero
    if (coeff == std::complex<double>(0.0, 0.0)) {
      continue;
    }
    term->set_coefficient(coeff);
    simplified_sum->add_term(std::move(term));
  }

  return simplified_sum;
}

std::unique_ptr<SumPauliOperatorExpression>
SumPauliOperatorExpression::prune_threshold(double epsilon) const {
  auto result = std::make_unique<SumPauliOperatorExpression>();

  // Helper function to recursively process terms
  std::function<void(const PauliOperatorExpression*)> process_term;
  process_term = [&result, epsilon,
                  &process_term](const PauliOperatorExpression* term) {
    if (auto* sum = term->as_sum_expression()) {
      // Recursively process nested sums
      for (const auto& nested_term : sum->get_terms()) {
        process_term(nested_term.get());
      }
    } else if (auto* prod = term->as_product_expression()) {
      // Keep the term only if its coefficient magnitude is >= epsilon
      if (std::abs(prod->get_coefficient()) >= epsilon) {
        result->add_term(term->clone());
      }
    } else if (auto* pauli = term->as_pauli_operator()) {
      // Bare PauliOperator has implicit coefficient of 1.0
      if (1.0 >= epsilon) {
        result->add_term(term->clone());
      }
    }
  };

  for (const auto& term : terms_) {
    process_term(term.get());
  }

  return result;
}

void SumPauliOperatorExpression::add_term(
    std::unique_ptr<PauliOperatorExpression> term) {
  terms_.push_back(std::move(term));
}

void SumPauliOperatorExpression::reserve_capacity(std::size_t capacity) {
  terms_.reserve(capacity);
}

const std::vector<std::unique_ptr<PauliOperatorExpression>>&
SumPauliOperatorExpression::get_terms() const {
  return terms_;
}

std::uint64_t SumPauliOperatorExpression::min_qubit_index() const {
  if (terms_.empty()) {
    throw std::logic_error(
        "min_qubit_index() called on empty SumPauliOperatorExpression");
  }
  std::uint64_t min_idx = std::numeric_limits<std::uint64_t>::max();
  for (const auto& term : terms_) {
    if (auto* pauli = term->as_pauli_operator()) {
      min_idx = std::min(min_idx, pauli->get_qubit_index());
    } else if (auto* prod = term->as_product_expression()) {
      if (!prod->get_factors().empty()) {
        min_idx = std::min(min_idx, prod->min_qubit_index());
      }
    } else if (auto* sum = term->as_sum_expression()) {
      if (!sum->get_terms().empty()) {
        min_idx = std::min(min_idx, sum->min_qubit_index());
      }
    }
  }
  return min_idx;
}

std::uint64_t SumPauliOperatorExpression::max_qubit_index() const {
  if (terms_.empty()) {
    throw std::logic_error(
        "max_qubit_index() called on empty SumPauliOperatorExpression");
  }
  std::uint64_t max_idx = 0;
  for (const auto& term : terms_) {
    if (auto* pauli = term->as_pauli_operator()) {
      max_idx = std::max(max_idx, pauli->get_qubit_index());
    } else if (auto* prod = term->as_product_expression()) {
      if (!prod->get_factors().empty()) {
        max_idx = std::max(max_idx, prod->max_qubit_index());
      }
    } else if (auto* sum = term->as_sum_expression()) {
      if (!sum->get_terms().empty()) {
        max_idx = std::max(max_idx, sum->max_qubit_index());
      }
    }
  }
  return max_idx;
}

std::uint64_t SumPauliOperatorExpression::num_qubits() const {
  if (terms_.empty()) {
    return 0;
  }
  return max_qubit_index() - min_qubit_index() + 1;
}

std::string SumPauliOperatorExpression::to_canonical_string(
    std::uint64_t num_qubits) const {
  return to_canonical_string(0, num_qubits - 1);
}

std::string SumPauliOperatorExpression::to_canonical_string(
    std::uint64_t min_qubit, std::uint64_t max_qubit) const {
  if (terms_.empty()) {
    return "0";
  }

  auto simplified = this->simplify();
  auto* simplified_sum = simplified->as_sum_expression();
  if (simplified_sum->get_terms().empty()) {
    return "0";
  }

  if (simplified_sum->get_terms().size() != 1) {
    throw std::logic_error(
        "to_canonical_string() requires a SumPauliOperatorExpression with a "
        "single term after simplification.");
  }

  return simplified_sum->get_terms()[0]->to_canonical_string(min_qubit,
                                                             max_qubit);
}

std::vector<std::pair<std::complex<double>, std::string>>
SumPauliOperatorExpression::to_canonical_terms(std::uint64_t num_qubits) const {
  std::vector<std::pair<std::complex<double>, std::string>> result;

  for (const auto& term : terms_) {
    if (auto* prod = term->as_product_expression()) {
      // Use the product's to_canonical_terms which handles phase computation
      auto terms = prod->to_canonical_terms(num_qubits);
      for (auto& t : terms) {
        result.push_back(std::move(t));
      }
    } else if (auto* pauli = term->as_pauli_operator()) {
      // Wrap in a product to get canonical string
      ProductPauliOperatorExpression temp_prod;
      temp_prod.add_factor(pauli->clone());
      auto terms = temp_prod.to_canonical_terms(num_qubits);
      for (auto& t : terms) {
        result.push_back(std::move(t));
      }
    } else {
      // Fallback for other expression types
      result.emplace_back(std::complex<double>(1.0, 0.0), term->to_string());
    }
  }

  return result;
}

std::vector<std::pair<std::complex<double>, std::string>>
SumPauliOperatorExpression::to_canonical_terms() const {
  if (terms_.empty()) {
    return {};
  }
  std::uint64_t n = num_qubits();
  std::uint64_t min_q = min_qubit_index();
  // Adjust num_qubits to cover from 0 to max_qubit
  std::uint64_t effective_num_qubits = min_q + n;
  return to_canonical_terms(effective_num_qubits);
}

// ============================================================================
// PauliTermAccumulator implementation
// ============================================================================

void PauliTermAccumulator::accumulate(const SparsePauliWord& word,
                                      std::complex<double> coeff) {
  auto it = terms_.find(word);
  if (it != terms_.end()) {
    it->second += coeff;
  } else {
    terms_[word] = coeff;
  }
}

void PauliTermAccumulator::accumulate_product(const SparsePauliWord& word1,
                                              const SparsePauliWord& word2,
                                              std::complex<double> scale) {
  auto [phase, result_word] = multiply(word1, word2);
  accumulate(result_word, scale * phase);
}

std::vector<std::pair<std::complex<double>, SparsePauliWord>>
PauliTermAccumulator::get_terms(double threshold) const {
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> result;
  result.reserve(terms_.size());
  for (const auto& [word, coeff] : terms_) {
    if (std::abs(coeff) >= threshold) {
      result.emplace_back(coeff, word);
    }
  }
  return result;
}

std::vector<std::pair<std::complex<double>, std::string>>
PauliTermAccumulator::get_terms_as_strings(std::uint64_t num_qubits,
                                           double threshold) const {
  std::vector<std::pair<std::complex<double>, std::string>> result;
  result.reserve(terms_.size());

  for (const auto& [word, coeff] : terms_) {
    if (std::abs(coeff) >= threshold) {
      // Convert SparsePauliWord to canonical string
      std::string pauli_str(num_qubits, 'I');
      for (const auto& [qubit, op_type] : word) {
        if (qubit < num_qubits) {
          switch (op_type) {
            case 1:
              pauli_str[qubit] = 'X';
              break;
            case 2:
              pauli_str[qubit] = 'Y';
              break;
            case 3:
              pauli_str[qubit] = 'Z';
              break;
            default:
              break;  // Identity, already 'I'
          }
        }
      }
      result.emplace_back(coeff, std::move(pauli_str));
    }
  }
  return result;
}

void PauliTermAccumulator::clear() {
  terms_.clear();
  clear_cache();
}

void PauliTermAccumulator::set_cache_capacity(std::size_t capacity) {
  cache_capacity_ = capacity;
  // Evict excess entries
  while (cache_map_.size() > capacity && !lru_list_.empty()) {
    auto oldest = lru_list_.back();
    cache_map_.erase(oldest);
    lru_list_.pop_back();
  }
}

void PauliTermAccumulator::clear_cache() {
  cache_map_.clear();
  lru_list_.clear();
}

std::size_t PauliTermAccumulator::cache_size() const {
  return cache_map_.size();
}

std::pair<std::complex<double>, SparsePauliWord> PauliTermAccumulator::multiply(
    const SparsePauliWord& word1, const SparsePauliWord& word2) {
  // Check cache first
  auto key = std::make_pair(word1, word2);
  auto it = cache_map_.find(key);
  if (it != cache_map_.end()) {
    // Move to front of LRU list (most recently used)
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.second);
    return it->second.first;
  }

  // Compute multiplication
  auto result = multiply_uncached(word1, word2);

  // Add to cache with LRU eviction
  if (cache_capacity_ > 0) {
    // Evict if at capacity
    while (cache_map_.size() >= cache_capacity_ && !lru_list_.empty()) {
      auto oldest = lru_list_.back();
      cache_map_.erase(oldest);
      lru_list_.pop_back();
    }

    // Insert new entry
    lru_list_.push_front(key);
    cache_map_[key] = {result, lru_list_.begin()};
  }

  return result;
}

std::pair<std::complex<double>, SparsePauliWord>
PauliTermAccumulator::multiply_uncached(const SparsePauliWord& word1,
                                        const SparsePauliWord& word2) {
  return detail::PauliAlgebraImpl::multiply(word1, word2);
}

// ============================================================================
// Excitation Term Computation
// ============================================================================

namespace {

// Pauli operator type constants
constexpr std::uint8_t OP_X = 1;
constexpr std::uint8_t OP_Y = 2;
constexpr std::uint8_t OP_Z = 3;

// Hash function for (p, q) pairs
struct PairHash {
  std::size_t operator()(
      const std::pair<std::uint64_t, std::uint64_t>& p) const noexcept {
    return p.first * 0x9e3779b97f4a7c15ULL + p.second;
  }
};

/**
 * @brief Compute JW excitation terms E_pq = a†_p a_q for a single (p, q) pair.
 */
std::vector<std::pair<std::complex<double>, SparsePauliWord>>
compute_jw_excitation_terms_single(std::uint64_t p, std::uint64_t q) {
  const std::complex<double> imag_unit(0.0, 1.0);

  if (p == q) {
    // Number operator: a†_p a_p = (1/2)(I - Z_p)
    return {
        {std::complex<double>(0.5, 0.0), {}},           // 0.5 * I
        {std::complex<double>(-0.5, 0.0), {{p, OP_Z}}}  // -0.5 * Z_p
    };
  }

  // For p != q, the Z strings partially cancel
  std::uint64_t lo = std::min(p, q);
  std::uint64_t hi = std::max(p, q);

  // Build Z-string for qubits between lo+1 and hi-1
  SparsePauliWord z_middle;
  for (std::uint64_t j = lo + 1; j < hi; ++j) {
    z_middle.emplace_back(j, OP_Z);
  }

  // Construct words directly in sorted order
  // Format: [(lo, op_lo), z_middle..., (hi, op_hi)]
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> terms;
  terms.reserve(4);

  // XX and YY terms always have the same structure
  SparsePauliWord xx_word, yy_word, xy_word, yx_word;
  xx_word.reserve(2 + z_middle.size());
  yy_word.reserve(2 + z_middle.size());
  xy_word.reserve(2 + z_middle.size());
  yx_word.reserve(2 + z_middle.size());

  xx_word.emplace_back(lo, OP_X);
  xx_word.insert(xx_word.end(), z_middle.begin(), z_middle.end());
  xx_word.emplace_back(hi, OP_X);

  yy_word.emplace_back(lo, OP_Y);
  yy_word.insert(yy_word.end(), z_middle.begin(), z_middle.end());
  yy_word.emplace_back(hi, OP_Y);

  if (p < q) {
    // X_p at lo, Y_q at hi for xy; Y_p at lo, X_q at hi for yx
    xy_word.emplace_back(lo, OP_X);
    xy_word.insert(xy_word.end(), z_middle.begin(), z_middle.end());
    xy_word.emplace_back(hi, OP_Y);

    yx_word.emplace_back(lo, OP_Y);
    yx_word.insert(yx_word.end(), z_middle.begin(), z_middle.end());
    yx_word.emplace_back(hi, OP_X);
  } else {
    // p > q: X_p at hi, Y_q at lo for xy -> means Y at lo, X at hi
    xy_word.emplace_back(lo, OP_Y);
    xy_word.insert(xy_word.end(), z_middle.begin(), z_middle.end());
    xy_word.emplace_back(hi, OP_X);

    yx_word.emplace_back(lo, OP_X);
    yx_word.insert(yx_word.end(), z_middle.begin(), z_middle.end());
    yx_word.emplace_back(hi, OP_Y);
  }

  terms.emplace_back(std::complex<double>(0.25, 0.0), std::move(xx_word));
  terms.emplace_back(std::complex<double>(0.25, 0.0), std::move(yy_word));
  terms.emplace_back(std::complex<double>(0.0, 0.25), std::move(xy_word));
  terms.emplace_back(std::complex<double>(0.0, -0.25), std::move(yx_word));

  return terms;
}

/**
 * @brief Build X-component of BK ladder operator: Z_{P(j)} * X_j * X_{U(j)}
 */
SparsePauliWord build_bk_x_component(
    std::uint64_t j, const std::vector<std::uint64_t>& parity_set,
    const std::vector<std::uint64_t>& update_set) {
  SparsePauliWord word;
  word.reserve(1 + parity_set.size() + update_set.size());

  word.emplace_back(j, OP_X);
  for (auto q : parity_set) {
    word.emplace_back(q, OP_Z);
  }
  for (auto q : update_set) {
    word.emplace_back(q, OP_X);
  }
  std::sort(word.begin(), word.end());
  return word;
}

/**
 * @brief Build Y-component of BK ladder operator: Z_{R(j)} * Y_j * X_{U(j)}
 */
SparsePauliWord build_bk_y_component(
    std::uint64_t j, const std::vector<std::uint64_t>& remainder_set,
    const std::vector<std::uint64_t>& update_set) {
  SparsePauliWord word;
  word.reserve(1 + remainder_set.size() + update_set.size());

  word.emplace_back(j, OP_Y);
  for (auto q : remainder_set) {
    word.emplace_back(q, OP_Z);
  }
  for (auto q : update_set) {
    word.emplace_back(q, OP_X);
  }
  std::sort(word.begin(), word.end());
  return word;
}

/**
 * @brief Compute BK excitation terms E_pq = a†_p a_q for a single (p, q) pair.
 */
std::vector<std::pair<std::complex<double>, SparsePauliWord>>
compute_bk_excitation_terms_single(
    std::uint64_t p, std::uint64_t q,
    const std::vector<std::uint64_t>& parity_p,
    const std::vector<std::uint64_t>& update_p,
    const std::vector<std::uint64_t>& remainder_p,
    const std::vector<std::uint64_t>& parity_q,
    const std::vector<std::uint64_t>& update_q,
    const std::vector<std::uint64_t>& remainder_q) {
  const std::complex<double> imag_unit(0.0, 1.0);

  // Build X and Y components for both p and q
  SparsePauliWord x_p = build_bk_x_component(p, parity_p, update_p);
  SparsePauliWord y_p = build_bk_y_component(p, remainder_p, update_p);
  SparsePauliWord x_q = build_bk_x_component(q, parity_q, update_q);
  SparsePauliWord y_q = build_bk_y_component(q, remainder_q, update_q);

  // a†_p = (1/2)(X_p - i*Y_p), a_q = (1/2)(X_q + i*Y_q)
  // a†_p * a_q = (1/4)(X_p*X_q + i*X_p*Y_q - i*Y_p*X_q + Y_p*Y_q)

  auto [phase_xx, word_xx] = detail::PauliAlgebraImpl::multiply(x_p, x_q);
  auto [phase_xy, word_xy] = detail::PauliAlgebraImpl::multiply(x_p, y_q);
  auto [phase_yx, word_yx] = detail::PauliAlgebraImpl::multiply(y_p, x_q);
  auto [phase_yy, word_yy] = detail::PauliAlgebraImpl::multiply(y_p, y_q);

  // Combine terms with same word
  std::unordered_map<SparsePauliWord, std::complex<double>, SparsePauliWordHash>
      combined;

  combined[word_xx] += std::complex<double>(0.25, 0.0) * phase_xx;
  combined[word_xy] += std::complex<double>(0.0, 0.25) * phase_xy;
  combined[word_yx] += std::complex<double>(0.0, -0.25) * phase_yx;
  combined[word_yy] += std::complex<double>(0.25, 0.0) * phase_yy;

  // Filter out zero terms
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> result;
  result.reserve(combined.size());
  for (auto& [word, coeff] : combined) {
    if (std::abs(coeff) > std::numeric_limits<double>::epsilon()) {
      result.emplace_back(coeff, word);
    }
  }

  return result;
}

}  // anonymous namespace

std::unordered_map<
    std::pair<std::uint64_t, std::uint64_t>,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>>,
    std::function<std::size_t(const std::pair<std::uint64_t, std::uint64_t>&)>>
PauliTermAccumulator::compute_all_jw_excitation_terms(
    std::uint64_t n_spin_orbitals) {
  // Use a proper hash function
  auto hash_fn = [](const std::pair<std::uint64_t, std::uint64_t>& p) {
    return p.first * 0x9e3779b97f4a7c15ULL + p.second;
  };

  std::unordered_map<
      std::pair<std::uint64_t, std::uint64_t>,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>,
      std::function<std::size_t(
          const std::pair<std::uint64_t, std::uint64_t>&)>>
      result(n_spin_orbitals * n_spin_orbitals, hash_fn);

  for (std::uint64_t p = 0; p < n_spin_orbitals; ++p) {
    for (std::uint64_t q = 0; q < n_spin_orbitals; ++q) {
      result[{p, q}] = compute_jw_excitation_terms_single(p, q);
    }
  }

  return result;
}

std::unordered_map<
    std::pair<std::uint64_t, std::uint64_t>,
    std::vector<std::pair<std::complex<double>, SparsePauliWord>>,
    std::function<std::size_t(const std::pair<std::uint64_t, std::uint64_t>&)>>
PauliTermAccumulator::compute_all_bk_excitation_terms(
    std::uint64_t n_spin_orbitals,
    const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>&
        parity_sets,
    const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>&
        update_sets,
    const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>&
        remainder_sets) {
  // Use a proper hash function
  auto hash_fn = [](const std::pair<std::uint64_t, std::uint64_t>& p) {
    return p.first * 0x9e3779b97f4a7c15ULL + p.second;
  };

  std::unordered_map<
      std::pair<std::uint64_t, std::uint64_t>,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>,
      std::function<std::size_t(
          const std::pair<std::uint64_t, std::uint64_t>&)>>
      result(n_spin_orbitals * n_spin_orbitals, hash_fn);

  for (std::uint64_t p = 0; p < n_spin_orbitals; ++p) {
    const auto& parity_p = parity_sets.at(p);
    const auto& update_p = update_sets.at(p);
    const auto& remainder_p = remainder_sets.at(p);

    for (std::uint64_t q = 0; q < n_spin_orbitals; ++q) {
      const auto& parity_q = parity_sets.at(q);
      const auto& update_q = update_sets.at(q);
      const auto& remainder_q = remainder_sets.at(q);

      result[{p, q}] = compute_bk_excitation_terms_single(
          p, q, parity_p, update_p, remainder_p, parity_q, update_q,
          remainder_q);
    }
  }

  return result;
}

}  // namespace qdk::chemistry::data
