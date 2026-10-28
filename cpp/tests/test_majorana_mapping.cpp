// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/qdk/chemistry/data/two_body_symmetry.hpp"

using namespace qdk::chemistry::data;

namespace qdk::chemistry::tests::test_support {

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

}  // namespace qdk::chemistry::tests::test_support

namespace test_support = qdk::chemistry::tests::test_support;
using test_support::collect_terms;
using test_support::expect_real_term;

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

TEST(MajoranaMapEngineTest, BravyiKitaevProducesCorrectIdentityCoefficient) {
  auto jw = MajoranaMapping::jordan_wigner(4);
  auto bk = MajoranaMapping::bravyi_kitaev(4);
  // 2 spatial orbitals, diagonal one-body, no two-body.
  const double h1_alpha[4] = {1.0, 0.0, 0.0, 0.5};
  const double h1_beta[4] = {2.0, 0.0, 0.0, 1.5};
  const double eri_zero[16] = {};

  auto jw_result = majorana_map_hamiltonian(
      jw, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);
  auto bk_result = majorana_map_hamiltonian(
      bk, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto jw_terms = collect_terms(jw_result, jw.num_qubits());
  auto bk_terms = collect_terms(bk_result, bk.num_qubits());

  // Identity coefficient = sum(h_diag)/2 is encoding-independent.
  expect_real_term(jw_terms, "IIII", 2.5);
  expect_real_term(bk_terms, "IIII", 2.5);
}

TEST(MajoranaMapEngineTest, ParityProducesCorrectIdentityCoefficient) {
  auto jw = MajoranaMapping::jordan_wigner(4);
  auto par = MajoranaMapping::parity(4);
  const double h1_alpha[4] = {1.0, 0.0, 0.0, 0.5};
  const double h1_beta[4] = {2.0, 0.0, 0.0, 1.5};
  const double eri_zero[16] = {};

  auto jw_result = majorana_map_hamiltonian(
      jw, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);
  auto par_result = majorana_map_hamiltonian(
      par, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto jw_terms = collect_terms(jw_result, jw.num_qubits());
  auto par_terms = collect_terms(par_result, par.num_qubits());

  expect_real_term(jw_terms, "IIII", 2.5);
  expect_real_term(par_terms, "IIII", 2.5);
}

TEST(MajoranaMapEngineTest, SpinSymmetricMatchesUnrestricted) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  // (00|00)=0.6, (11|11)=0.4, (00|11)=(11|00)=0.1
  const double eri[16] = {0.6, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.4};

  auto restricted = majorana_map_hamiltonian(
      mapping, 0.5, h1, h1, eri, eri, eri,
      /*n_spatial=*/2, /*spin_symmetric=*/true, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);
  auto unrestricted = majorana_map_hamiltonian(
      mapping, 0.5, h1, h1, eri, eri, eri,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto r_terms = collect_terms(restricted, mapping.num_qubits());
  auto u_terms = collect_terms(unrestricted, mapping.num_qubits());

  EXPECT_EQ(r_terms.size(), u_terms.size());
  for (const auto& [label, coeff] : r_terms) {
    auto it = u_terms.find(label);
    ASSERT_NE(it, u_terms.end()) << "Missing term " << label;
    EXPECT_NEAR(coeff.real(), it->second.real(), 1e-10)
        << "Real mismatch at " << label;
    EXPECT_NEAR(coeff.imag(), it->second.imag(), 1e-10)
        << "Imag mismatch at " << label;
  }
}

TEST(MajoranaMapEngineTest, SymmetryToleranceScalesWithIntegralMagnitude) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  // Round-off from an integral transformation in a nearly linearly dependent
  // basis reaches ~1e-6 relative to the largest integral, and must not be
  // mistaken for genuinely reduced symmetry.
  const double magnitude = 1.0e6;
  double eri[16] = {};
  for (double& value : eri) value = magnitude;
  eri[1] += 1.0e-6 * magnitude;

  EXPECT_NO_THROW(majorana_map_hamiltonian(mapping, 0.0, h1, h1, eri, eri, eri,
                                           /*n_spatial=*/2,
                                           /*spin_symmetric=*/true,
                                           /*threshold=*/0.0,
                                           /*integral_threshold=*/0.0));
}

TEST(MajoranaMapEngineTest, SparseEntriesAreReadLiterally) {
  const auto mapping = MajoranaMapping::jordan_wigner(4);
  const double one_body[4] = {1.0, 0.3, 0.3, 0.5};
  // Each entry is the only stored member of its 8-fold class; only the
  // container constructor expands that deprecated shorthand.
  const int indices[8] = {0, 1, 0, 1, 1, 1, 0, 0};
  const double values[2] = {0.7, 0.4};

  EXPECT_THROW(majorana_map_hamiltonian_sparse(
                   mapping, 0.5, one_body, one_body, indices, values,
                   /*num_entries=*/2, /*n_spatial=*/2, /*spin_symmetric=*/true,
                   /*threshold=*/0.0, /*integral_threshold=*/0.0),
               std::invalid_argument);
}

TEST(MajoranaMapEngineTest, TwoBodyIntegralsProduceAdditionalTerms) {
  auto mapping = MajoranaMapping::jordan_wigner(2);
  const double h1_alpha[1] = {1.0};
  const double h1_beta[1] = {2.0};
  const double eri_zero[1] = {0.0};
  const double eri_nonzero[1] = {0.8};

  auto one_body_only = majorana_map_hamiltonian(
      mapping, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/1, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);
  auto with_two_body = majorana_map_hamiltonian(
      mapping, 0.0, h1_alpha, h1_beta, eri_nonzero, eri_nonzero, eri_nonzero,
      /*n_spatial=*/1, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto ob_terms = collect_terms(one_body_only, mapping.num_qubits());
  auto tb_terms = collect_terms(with_two_body, mapping.num_qubits());

  // Two-body integrals should produce additional or modified terms.
  EXPECT_NE(ob_terms, tb_terms);
  // Identity coefficient should differ due to two-body contributions.
  EXPECT_NE(ob_terms.at("II").real(), tb_terms.at("II").real());
}

TEST(MajoranaMapEngineTest, MultiWordDispatchDoesNotCrash) {
  // 33 spatial orbitals → 66 spin-orbitals (modes) → 66 qubits → NW=2,
  // exercising multi-word packed-Pauli dispatch.
  constexpr std::size_t n_spatial = 33;
  auto mapping = MajoranaMapping::jordan_wigner(2 * n_spatial);
  std::vector<double> h1(n_spatial * n_spatial, 0.0);
  h1[0] = 1.0;  // single non-zero diagonal element
  std::vector<double> eri(n_spatial * n_spatial * n_spatial * n_spatial, 0.0);

  auto result = majorana_map_hamiltonian(
      mapping, 0.5, h1.data(), h1.data(), eri.data(), eri.data(), eri.data(),
      n_spatial, /*spin_symmetric=*/true, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto terms = collect_terms(result, mapping.num_qubits());
  // Should have at least the identity term.
  EXPECT_GE(terms.size(), 1u);
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

namespace {

void expect_same_coefficients(const MajoranaMapResult& actual,
                              const MajoranaMapResult& expected,
                              std::size_t num_qubits) {
  auto difference = collect_terms(actual, num_qubits);
  for (const auto& [label, coefficient] : collect_terms(expected, num_qubits))
    difference[label] -= coefficient;
  for (const auto& [label, coefficient] : difference)
    EXPECT_NEAR(std::abs(coefficient), 0.0, 1e-11) << label;
}

std::vector<double> fourfold_test_tensor(std::size_t extent) {
  std::vector<double> tensor(extent * extent * extent * extent);
  for (std::size_t bra = 0; bra < extent; ++bra)
    for (std::size_t ket = 0; ket < extent; ++ket)
      for (std::size_t other_bra = 0; other_bra < extent; ++other_bra)
        for (std::size_t other_ket = 0; other_ket < extent; ++other_ket) {
          using Key = std::array<std::size_t, 4>;
          const auto canonical =
              std::min({Key{bra, ket, other_bra, other_ket},
                        Key{other_bra, other_ket, bra, ket},
                        Key{ket, bra, other_ket, other_bra},
                        Key{other_ket, other_bra, ket, bra}});
          const auto flattened = [extent](const Key& key) {
            return ((key[0] * extent + key[1]) * extent + key[2]) * extent +
                   key[3];
          };
          tensor[flattened({bra, ket, other_bra, other_ket})] =
              0.3 * std::sin(1.0 + static_cast<double>(flattened(canonical)));
        }
  return tensor;
}

}  // namespace

TEST(MajoranaMapEngineTest, FourFoldSplitMatchesGeneralAcrossMappings) {
  const auto tensor = fourfold_test_tensor(3);
  // Bra-ket antisymmetric: same operator, but aaaa no longer allows the split.
  auto general_tensor = tensor;
  general_tensor[4] += 0.5;
  general_tensor[36] -= 0.5;
  const double one_alpha[9] = {1.0, 0.2, -0.1, 0.2, 0.5, 0.3, -0.1, 0.3, 0.7};
  const double one_beta[9] = {0.3, -0.2, 0.1, -0.2, 0.4, 0.2, 0.1, 0.2, -0.5};
  // Breaks ab bra-ket symmetry but keeps reversal, which is all the split
  // needs.
  auto mixed = tensor;
  mixed[1] += 0.25;
  mixed[3] += 0.25;
  std::vector<MajoranaMapping> mappings{
      MajoranaMapping::jordan_wigner(6), MajoranaMapping::bravyi_kitaev(6),
      MajoranaMapping::parity(6), MajoranaMapping::bravyi_kitaev_tree(6)};
  const LatticeGraph lattice({{{0, 1}, 1.0}, {{1, 2}, 1.0}, {{0, 2}, 1.0}}, 3);
  mappings.push_back(MajoranaMapping::verstraete_cirac(lattice));
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears;
  for (std::size_t first = 0; first < 12; ++first)
    for (std::size_t second = first + 1; second < 12; ++second) {
      const auto [coefficient, word] = mappings.front().bilinear(first, second);
      bilinears.emplace_back(coefficient, word);
    }
  mappings.push_back(MajoranaMapping::from_bilinears(6, std::move(bilinears)));
  auto shifted_table = mappings.front().table();
  for (auto& word : shifted_table)
    for (auto& [qubit, pauli] : word) qubit += 63;
  mappings.push_back(MajoranaMapping::from_table(std::move(shifted_table)));
  for (const auto& mapping : mappings) {
    for (bool restricted : {true, false}) {
      const auto* beta = restricted ? one_alpha : one_beta;
      const auto* cross = restricted ? tensor.data() : mixed.data();
      const auto fast = majorana_map_hamiltonian(
          mapping, 0.375, one_alpha, beta, tensor.data(), cross, tensor.data(),
          3, restricted, 1e-12, 0.0);
      const auto general = majorana_map_hamiltonian(
          mapping, 0.375, one_alpha, beta, general_tensor.data(), cross,
          tensor.data(), 3, false, 1e-12, 0.0);
      expect_same_coefficients(fast, general, mapping.num_qubits());
    }
  }
}

TEST(MajoranaMapEngineTest, SparseFourFoldInputOrderIsDeterministic) {
  const auto mapping = MajoranaMapping::jordan_wigner(4);
  const double one_body[4] = {1.0, 0.3, 0.3, 0.5};
  std::vector<std::array<int, 4>> keys{
      {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};
  std::vector<double> values{0.7, 0.7, 0.0, 0.0};
  auto map_entries = [&]() {
    std::vector<int> indices;
    for (const auto& key : keys)
      indices.insert(indices.end(), key.begin(), key.end());
    return majorana_map_hamiltonian_sparse(mapping, 0.0, one_body, one_body,
                                           indices.data(), values.data(),
                                           values.size(), 2, true, 1e-12, 0.0);
  };
  const auto forward = map_entries();
  std::reverse(keys.begin(), keys.end());
  std::reverse(values.begin(), values.end());
  const auto reverse = map_entries();
  EXPECT_EQ(forward.words, reverse.words);
  EXPECT_EQ(forward.coefficients, reverse.coefficients);
}

TEST(MajoranaMapEngineTest, CholeskyFourFoldValidationIsScaleInvariant) {
  namespace detail = qdk::chemistry::data::detail;
  for (double scale : {1e-150, 1.0, 1e150}) {
    Eigen::MatrixXd factors(4, 2);
    factors << 0.8, -0.8, 0.7, 0.3, -0.3, -0.7, 0.3, -0.3;
    factors *= scale;
    EXPECT_TRUE(detail::cholesky_has_reversal_symmetry(
        factors.data(), factors.data(), 2, 2, 1e-8));
    factors(0, 1) += 0.2 * scale;
    EXPECT_FALSE(detail::cholesky_has_reversal_symmetry(
        factors.data(), factors.data(), 2, 2, 1e-8));
  }
}

TEST(MajoranaMapEngineTest, CholeskyMixedSpinValidationChecksCrossParity) {
  namespace detail = qdk::chemistry::data::detail;
  Eigen::MatrixXd alpha(4, 2);
  alpha << 0.8, 0.0, 0.2, 0.5, 0.2, -0.5, 0.3, 0.0;
  Eigen::MatrixXd beta = alpha;
  EXPECT_TRUE(detail::cholesky_has_reversal_symmetry(alpha.data(), beta.data(),
                                                     2, 2, 1e-8));
  beta.col(0).swap(beta.col(1));
  EXPECT_TRUE(detail::cholesky_has_reversal_symmetry(beta.data(), beta.data(),
                                                     2, 2, 1e-8));
  EXPECT_FALSE(detail::cholesky_has_reversal_symmetry(alpha.data(), beta.data(),
                                                      2, 2, 1e-8));
}

namespace {

// (pq|rs) = sum_Q L^Q_pq L^Q_rs for column-major [n^2 x naux] factors; the
// [n^2 x n^2] result is symmetric, so its storage is also row-major.
std::vector<double> reconstruct_dense(const Eigen::MatrixXd& factors) {
  const Eigen::MatrixXd tensor = factors * factors.transpose();
  return {tensor.data(), tensor.data() + tensor.size()};
}

std::vector<double> diagonal_one_body(std::size_t extent) {
  std::vector<double> one_body(extent * extent, 0.0);
  for (std::size_t p = 0; p < extent; ++p)
    one_body[p * extent + p] = 0.1 * static_cast<double>(p);
  return one_body;
}

}  // namespace

TEST(MajoranaMapEngineTest, CholeskyRejectsNonFiniteFactors) {
  const auto mapping = MajoranaMapping::jordan_wigner(4);
  const auto one_body = diagonal_one_body(2);
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    Eigen::MatrixXd factors(4, 1);
    factors << 0.8, 0.2, 0.2, invalid;
    for (bool restricted : {true, false}) {
      EXPECT_THROW(
          majorana_map_hamiltonian_cholesky(
              mapping, 0.0, one_body.data(), one_body.data(), factors.data(),
              factors.data(), 2, 1, restricted, 1e-12, 1e-12),
          std::invalid_argument);
    }
  }
}

TEST(MajoranaMapEngineTest, CholeskyValidationIsElementwiseLikeDense) {
  // A violation at one pair is not diluted by the size of the tensor.
  {
    constexpr std::size_t n = 10;
    const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
    const auto one_body = diagonal_one_body(n);
    Eigen::MatrixXd factors = Eigen::MatrixXd::Ones(n * n, 1);
    factors(0 * n + 1, 0) = 1.0 + 3e-5;
    factors(1 * n + 0, 0) = 1.0 - 3e-5;
    const auto dense = reconstruct_dense(factors);
    EXPECT_THROW(
        majorana_map_hamiltonian(mapping, 0.0, one_body.data(), one_body.data(),
                                 dense.data(), dense.data(), dense.data(), n,
                                 true, 1e-12, 1e-12),
        std::invalid_argument);
    EXPECT_THROW(majorana_map_hamiltonian_cholesky(
                     mapping, 0.0, one_body.data(), one_body.data(),
                     factors.data(), factors.data(), n, 1, true, 1e-12, 1e-12),
                 std::invalid_argument);
  }
  // A violation spread thinly over every element is accepted by both, even
  // though its Frobenius norm exceeds the tolerance.
  constexpr std::size_t n = 6;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  Eigen::MatrixXd factors = Eigen::MatrixXd::Zero(n * n, 2);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      factors(p * n + q, 0) = 1.0 + (p < q ? 1e-6 : p > q ? -1e-6 : 0.0);
  factors(0 * n + 1, 1) = 0.5;
  factors(1 * n + 0, 1) = -0.5;
  const auto dense = reconstruct_dense(factors);
  const auto from_dense = majorana_map_hamiltonian(
      mapping, 0.0, one_body.data(), one_body.data(), dense.data(),
      dense.data(), dense.data(), n, true, 1e-12, 0.0);
  const auto from_factors = majorana_map_hamiltonian_cholesky(
      mapping, 0.0, one_body.data(), one_body.data(), factors.data(),
      factors.data(), n, 2, true, 1e-12, 0.0);
  expect_same_coefficients(from_factors, from_dense, mapping.num_qubits());
}

TEST(MajoranaMapEngineTest, CholeskyKeepsAccumulatedFourFoldPart) {
  // Each antisymmetric column is small, but their contribution to the
  // integrals accumulates over the auxiliary index and must be kept.
  constexpr std::size_t n = 3;
  constexpr std::size_t antisymmetric_columns = 100;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  Eigen::MatrixXd factors =
      Eigen::MatrixXd::Zero(n * n, 1 + antisymmetric_columns);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = p; q < n; ++q)
      factors(p * n + q, 0) = factors(q * n + p, 0) =
          std::cos(static_cast<double>(p * n + q));
  for (std::size_t column = 1; column <= antisymmetric_columns; ++column)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = p + 1; q < n; ++q) {
        factors(p * n + q, column) = 4e-6;
        factors(q * n + p, column) = -4e-6;
      }
  const auto dense = reconstruct_dense(factors);
  const auto from_dense = majorana_map_hamiltonian(
      mapping, 0.0, one_body.data(), one_body.data(), dense.data(),
      dense.data(), dense.data(), n, true, 1e-14, 1e-12);
  const auto from_factors = majorana_map_hamiltonian_cholesky(
      mapping, 0.0, one_body.data(), one_body.data(), factors.data(),
      factors.data(), n, static_cast<std::size_t>(factors.cols()), true, 1e-14,
      1e-12);
  expect_same_coefficients(from_factors, from_dense, mapping.num_qubits());

  Eigen::MatrixXd symmetric_part = factors.leftCols(1);
  const auto eightfold = reconstruct_dense(symmetric_part);
  const auto without_part = majorana_map_hamiltonian(
      mapping, 0.0, one_body.data(), one_body.data(), eightfold.data(),
      eightfold.data(), eightfold.data(), n, true, 1e-14, 1e-12);
  double largest_difference = 0.0;
  auto difference = test_support::collect_terms(from_factors, 2 * n);
  for (const auto& [label, coefficient] :
       test_support::collect_terms(without_part, 2 * n))
    difference[label] -= coefficient;
  for (const auto& [label, coefficient] : difference)
    largest_difference = std::max(largest_difference, std::abs(coefficient));
  // Well above the 1e-11 tolerance of expect_same_coefficients.
  EXPECT_GT(largest_difference, 5e-11);
}

TEST(MajoranaMapEngineTest, FourFoldPartBelowIntegralThresholdAddsNoTerms) {
  // Round-off well above the old 1e-10 relative cutoff but below
  // integral_threshold in g^A maps like the exactly eightfold tensor.
  namespace detail = qdk::chemistry::data::detail;
  constexpr std::size_t n = 3;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  const auto fourfold = fourfold_test_tensor(n);
  std::vector<double> eightfold(fourfold.size());
  std::vector<double> perturbed(fourfold.size());
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const auto canonical = detail::eightfold_canonical({p, q, r, s});
          const std::size_t index = ((p * n + q) * n + r) * n + s;
          eightfold[index] =
              1e-3 * std::sin(1.0 + static_cast<double>(
                                        ((canonical[0] * n + canonical[1]) * n +
                                         canonical[2]) *
                                            n +
                                        canonical[3]));
          perturbed[index] = eightfold[index] +
                             5e-13 * (fourfold[index] -
                                      fourfold[((p * n + q) * n + s) * n + r]);
        }
  const auto exact = majorana_map_hamiltonian(
      mapping, 0.0, one_body.data(), one_body.data(), eightfold.data(),
      eightfold.data(), eightfold.data(), n, true, 0.0, 1e-12);
  const auto mapped = majorana_map_hamiltonian(
      mapping, 0.0, one_body.data(), one_body.data(), perturbed.data(),
      perturbed.data(), perturbed.data(), n, true, 0.0, 1e-12);
  EXPECT_EQ(test_support::collect_terms(mapped, 2 * n).size(),
            test_support::collect_terms(exact, 2 * n).size());
  expect_same_coefficients(mapped, exact, mapping.num_qubits());
}

TEST(MajoranaMappingHashTest, EqualMappingsHaveStableHash) {
  auto first = MajoranaMapping::jordan_wigner(4);
  auto second = MajoranaMapping::jordan_wigner(4);

  EXPECT_EQ(first.content_hash(), second.content_hash());
  EXPECT_EQ(first.content_hash(32), second.content_hash(32));
}

TEST(MajoranaMappingHashTest, HashIncludesMappingAndTaperingData) {
  auto jw = MajoranaMapping::jordan_wigner(4);
  auto parity = MajoranaMapping::parity(4);
  auto reduced = MajoranaMapping::parity(4, 1, 1);

  EXPECT_NE(jw.content_hash(), parity.content_hash());
  EXPECT_NE(parity.content_hash(), reduced.content_hash());
  EXPECT_NE(reduced.content_hash(), reduced.without_tapering().content_hash());
}

TEST(MajoranaMappingHashTest, BilinearOnlyMappingsHashTheirCoefficients) {
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears = {
      {{1.0, 0.0}, {{0, 3}}}};
  auto first = MajoranaMapping::from_bilinears(1, bilinears, "custom");

  bilinears[0].first = {0.0, 1.0};
  auto second = MajoranaMapping::from_bilinears(1, bilinears, "custom");

  EXPECT_NE(first.content_hash(), second.content_hash());
}

TEST(TaperingSpecificationHashTest, HashIncludesIndicesAndEigenvalues) {
  TaperingSpecification first({0, 3}, {1, -1});
  TaperingSpecification same({0, 3}, {1, -1});
  TaperingSpecification different_index({1, 3}, {1, -1});
  TaperingSpecification different_eigenvalue({0, 3}, {-1, -1});

  EXPECT_EQ(first.content_hash(), same.content_hash());
  EXPECT_NE(first.content_hash(), different_index.content_hash());
  EXPECT_NE(first.content_hash(), different_eigenvalue.content_hash());
}
