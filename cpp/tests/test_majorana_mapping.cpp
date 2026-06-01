// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <complex>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace qdk::chemistry::data;

namespace {

std::string sparse_to_dense_le(const SparsePauliWord& word,
                               std::size_t num_qubits) {
  std::string label(num_qubits, 'I');
  for (const auto& [qubit, op] : word) {
    char pauli = 'I';
    switch (op) {
      case 1:
        pauli = 'X';
        break;
      case 2:
        pauli = 'Y';
        break;
      case 3:
        pauli = 'Z';
        break;
      default:
        pauli = 'I';
        break;
    }
    label[num_qubits - 1 - qubit] = pauli;
  }
  return label;
}

std::unordered_map<std::string, std::complex<double>> collect_terms(
    const MajoranaMapResult& result, std::size_t num_qubits) {
  std::unordered_map<std::string, std::complex<double>> terms;
  for (std::size_t i = 0; i < result.words.size(); ++i) {
    terms[sparse_to_dense_le(result.words[i], num_qubits)] +=
        result.coefficients[i];
  }
  return terms;
}

void expect_real_term(
    const std::unordered_map<std::string, std::complex<double>>& terms,
    const std::string& label, double expected) {
  auto it = terms.find(label);
  ASSERT_NE(it, terms.end()) << "Missing term " << label;
  EXPECT_NEAR(it->second.real(), expected, 1e-12);
  EXPECT_NEAR(it->second.imag(), 0.0, 1e-12);
}

}  // namespace

TEST(MajoranaMapEngineTest, MapsOneBodyUnrestrictedJordanWignerHamiltonian) {
  auto mapping = MajoranaMapping::jordan_wigner(2);
  const double h1_alpha[1] = {1.0};
  const double h1_beta[1] = {2.0};
  const double eri_zero[1] = {0.0};

  auto result = majorana_map_hamiltonian(
      mapping, 0.5, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/1, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto terms = collect_terms(result, mapping.num_qubits());
  ASSERT_EQ(terms.size(), 3);
  expect_real_term(terms, "II", 2.0);
  expect_real_term(terms, "IZ", -0.5);
  expect_real_term(terms, "ZI", -1.0);
}

TEST(MajoranaMapEngineTest, RejectsZeroQubitMappings) {
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears = {
      {{1.0, 0.0}, {}}};
  auto mapping = MajoranaMapping::from_bilinears(1, std::move(bilinears));
  const double zero[1] = {0.0};

  EXPECT_THROW(
      majorana_map_hamiltonian(mapping, 0.0, zero, zero, zero, zero, zero,
                               /*n_spatial=*/1, /*spin_symmetric=*/false,
                               /*threshold=*/1e-12,
                               /*integral_threshold=*/1e-12),
      std::invalid_argument);
}
