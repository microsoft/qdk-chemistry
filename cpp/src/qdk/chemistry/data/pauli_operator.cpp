// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <stdexcept>
#include <tuple>

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
        if (auto* prod = existing_term->as_product_expression()) {
          coeff = prod->get_coefficient();
          // Copy factors from existing term
          for (const auto& f : prod->get_factors()) {
            new_product->add_factor(f->clone());
          }
        } else {
          new_product->add_factor(existing_term->clone());
        }

        // Multiply coefficient from factor term
        if (auto* prod = factor_term->as_product_expression()) {
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
  return {{coefficient_, to_canonical_string(num_qubits)}};
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
  // Helper function to create a term key from a simplified product
  // The key is a sorted vector of (qubit_index, operator_type) pairs
  auto make_term_key = [](const ProductPauliOperatorExpression* prod)
      -> std::vector<std::pair<std::uint64_t, std::uint8_t>> {
    std::vector<std::pair<std::uint64_t, std::uint8_t>> key;
    for (const auto& factor : prod->get_factors()) {
      if (auto* pauli = factor->as_pauli_operator()) {
        key.emplace_back(pauli->get_qubit_index(), pauli->get_operator_type());
      }
    }
    // Key should already be sorted since simplify() sorts factors by qubit
    return key;
  };

  // First simplify all terms individually
  std::vector<std::unique_ptr<ProductPauliOperatorExpression>> simplified_terms;

  // Helper function to add a simplified expression to simplified_terms
  std::function<void(std::unique_ptr<PauliOperatorExpression>)>
      add_simplified_term;
  add_simplified_term = [&simplified_terms, &add_simplified_term](
                            std::unique_ptr<PauliOperatorExpression> expr) {
    if (auto* prod = expr->as_product_expression()) {
      simplified_terms.push_back(
          std::make_unique<ProductPauliOperatorExpression>(*prod));
    } else if (auto* pauli = expr->as_pauli_operator()) {
      // Wrap single PauliOperator in a ProductPauliOperatorExpression
      auto wrapped = std::make_unique<ProductPauliOperatorExpression>();
      wrapped->add_factor(std::make_unique<PauliOperator>(*pauli));
      simplified_terms.push_back(std::move(wrapped));
    } else if (auto* sum = expr->as_sum_expression()) {
      // Recursively add terms from the sum
      for (const auto& term : sum->get_terms()) {
        add_simplified_term(term->clone());
      }
    }
  };

  // Distribute first to ensure all terms are products
  auto distributed = this->distribute();

  for (const auto& term : distributed->get_terms()) {
    auto simplified_term = term->simplify();
    add_simplified_term(std::move(simplified_term));
  }

  // Collect like terms using a vector to preserve insertion order
  // Each entry: (key, coefficient, term)
  using TermKey = std::vector<std::pair<std::uint64_t, std::uint8_t>>;
  std::vector<std::tuple<TermKey, std::complex<double>,
                         std::unique_ptr<ProductPauliOperatorExpression>>>
      collected_terms;

  for (auto& term : simplified_terms) {
    auto key = make_term_key(term.get());
    // Linear search for existing term with same key
    auto it = std::ranges::find_if(collected_terms, [&key](const auto& entry) {
      return std::get<0>(entry) == key;
    });

    if (it == collected_terms.end()) {
      // First occurrence of this Pauli string
      auto coeff = term->get_coefficient();
      term->set_coefficient(1.0);  // Store with unit coefficient
      collected_terms.emplace_back(std::move(key), coeff, std::move(term));
    } else {
      // Add coefficient to existing term
      std::get<1>(*it) += term->get_coefficient();
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
    std::complex<double> coeff(1.0, 0.0);
    std::string term_str;

    if (auto* prod = term->as_product_expression()) {
      coeff = prod->get_coefficient();
      term_str = prod->to_canonical_string(num_qubits);
    } else if (auto* pauli = term->as_pauli_operator()) {
      // Wrap in a product to get canonical string
      ProductPauliOperatorExpression temp_prod;
      temp_prod.add_factor(pauli->clone());
      term_str = temp_prod.to_canonical_string(num_qubits);
    } else {
      term_str = term->to_string();
    }

    result.emplace_back(coeff, term_str);
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

}  // namespace qdk::chemistry::data
