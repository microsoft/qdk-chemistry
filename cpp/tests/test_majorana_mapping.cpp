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
#include <map>
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/qdk/chemistry/data/two_body_symmetry.hpp"
#include "spdlog/sinks/ostream_sink.h"

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

std::size_t flat_index(std::size_t n, std::size_t p, std::size_t q,
                       std::size_t r, std::size_t s) {
  return ((p * n + q) * n + r) * n + s;
}

// A tensor without permutation symmetry.
std::vector<double> asymmetric_tensor(std::size_t n, double phase) {
  std::vector<double> tensor(n * n * n * n);
  for (std::size_t index = 0; index < tensor.size(); ++index)
    tensor[index] = 0.3 * std::sin(phase + 1.7 * static_cast<double>(index));
  return tensor;
}

// Average over (pq|rs), (rs|pq), (qp|sr) and (sr|qp).
std::vector<double> fourfold_average(const std::vector<double>& tensor,
                                     std::size_t n) {
  std::vector<double> average(tensor.size());
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          average[flat_index(n, p, q, r, s)] =
              0.25 * (tensor[flat_index(n, p, q, r, s)] +
                      tensor[flat_index(n, r, s, p, q)] +
                      tensor[flat_index(n, q, p, s, r)] +
                      tensor[flat_index(n, s, r, q, p)]);
  return average;
}

// Average over (pq|rs) and (qp|sr).
std::vector<double> reversal_average(const std::vector<double>& tensor,
                                     std::size_t n) {
  std::vector<double> average(tensor.size());
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          average[flat_index(n, p, q, r, s)] =
              0.5 * (tensor[flat_index(n, p, q, r, s)] +
                     tensor[flat_index(n, q, p, s, r)]);
  return average;
}

std::shared_ptr<ModelOrbitals> model_orbitals(std::size_t n, bool restricted) {
  return std::make_shared<ModelOrbitals>(
      n, std::make_shared<const SymmetryProduct>(
             SymmetryProduct({axes::spin(1, restricted)})));
}

Eigen::MatrixXd square_matrix(const double* values, std::size_t n) {
  return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>(
      values, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
}

Eigen::VectorXd as_vector(const std::vector<double>& values) {
  return Eigen::Map<const Eigen::VectorXd>(
      values.data(), static_cast<Eigen::Index>(values.size()));
}

Hamiltonian model_hamiltonian(const double* one_alpha, const double* one_beta,
                              const std::vector<double>& aaaa,
                              const std::vector<double>& aabb,
                              const std::vector<double>& bbbb, std::size_t n,
                              bool restricted, HamiltonianType type) {
  const Eigen::MatrixXd no_fock = Eigen::MatrixXd::Zero(0, 0);
  if (restricted) {
    return Hamiltonian(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            square_matrix(one_alpha, n), as_vector(aaaa),
            model_orbitals(n, true), 0.0, no_fock, type));
  }
  return Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      square_matrix(one_alpha, n), square_matrix(one_beta, n), as_vector(aaaa),
      as_vector(aabb), as_vector(bbbb), model_orbitals(n, false), 0.0, no_fock,
      no_fock, type));
}

}  // namespace

TEST(MajoranaMapEngineTest, FourFoldSplitMatchesGeneralAcrossMappings) {
  // Exactly fourfold input maps the same through the split and literally.
  const auto tensor = fourfold_test_tensor(3);
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
      const auto& cross = restricted ? tensor : mixed;
      const auto split = majorana_map_hamiltonian(
          mapping,
          model_hamiltonian(one_alpha, beta, tensor, cross, tensor, 3,
                            restricted, HamiltonianType::Hermitian),
          restricted, 1e-12, 0.0);
      const auto literal = majorana_map_hamiltonian(
          mapping,
          model_hamiltonian(one_alpha, beta, tensor, cross, tensor, 3,
                            restricted, HamiltonianType::NonHermitian),
          restricted, 1e-12, 0.0);
      expect_same_coefficients(split, literal, mapping.num_qubits());
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

TEST(MajoranaMapEngineTest, RejectsNonFiniteOneBodyIntegrals) {
  // Screening alone would drop a NaN one-body integral silently.
  const auto mapping = MajoranaMapping::jordan_wigner(4);
  const auto finite = diagonal_one_body(2);
  auto invalid = finite;
  invalid[1] = invalid[2] = std::numeric_limits<double>::quiet_NaN();
  const double eri[16] = {};
  for (bool restricted : {true, false}) {
    EXPECT_THROW(
        majorana_map_hamiltonian(mapping, 0.0, invalid.data(), finite.data(),
                                 eri, eri, eri, 2, restricted, 1e-12, 1e-12),
        std::invalid_argument);
  }
  // The beta integrals are read, and so checked, only when unrestricted.
  EXPECT_NO_THROW(majorana_map_hamiltonian(mapping, 0.0, finite.data(),
                                           invalid.data(), eri, eri, eri, 2,
                                           true, 1e-12, 1e-12));
  EXPECT_THROW(
      majorana_map_hamiltonian(mapping, 0.0, finite.data(), invalid.data(), eri,
                               eri, eri, 2, false, 1e-12, 1e-12),
      std::invalid_argument);
  // The sparse path beyond 1024 qubits shares the check.
  constexpr std::size_t wide = 520;
  std::vector<double> wide_invalid(wide * wide, 0.0);
  wide_invalid[1] = std::numeric_limits<double>::quiet_NaN();
  const int index[4] = {0, 0, 0, 0};
  const double value[1] = {0.5};
  EXPECT_THROW(
      majorana_map_hamiltonian_sparse(
          MajoranaMapping::jordan_wigner(2 * wide), 0.0, wide_invalid.data(),
          wide_invalid.data(), index, value, 1, wide, true, 1e-12, 1e-12),
      std::invalid_argument);
}

TEST(MajoranaMapEngineTest, RejectsNonFiniteCoreEnergy) {
  // Screening alone would drop a NaN core energy from the identity term.
  const auto mapping = MajoranaMapping::jordan_wigner(4);
  const auto one_body = diagonal_one_body(2);
  const double eri[16] = {};
  Eigen::MatrixXd factors = Eigen::MatrixXd::Zero(4, 1);
  const int index[4] = {0, 0, 0, 0};
  const double value[1] = {0.5};
  // The sparse path beyond 1024 qubits shares the check.
  constexpr std::size_t wide = 520;
  const auto wide_mapping = MajoranaMapping::jordan_wigner(2 * wide);
  const std::vector<double> wide_one_body(wide * wide, 0.0);
  for (double core_energy : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
    EXPECT_THROW(
        majorana_map_hamiltonian_sparse(
            wide_mapping, core_energy, wide_one_body.data(),
            wide_one_body.data(), index, value, 1, wide, true, 1e-12, 1e-12),
        std::invalid_argument);
    EXPECT_THROW(majorana_map_hamiltonian(mapping, core_energy, one_body.data(),
                                          one_body.data(), eri, eri, eri, 2,
                                          true, 1e-12, 1e-12),
                 std::invalid_argument);
    EXPECT_THROW(majorana_map_hamiltonian_cholesky(
                     mapping, core_energy, one_body.data(), one_body.data(),
                     factors.data(), factors.data(), 2, 1, true, 1e-12, 1e-12),
                 std::invalid_argument);
    EXPECT_THROW(majorana_map_hamiltonian_sparse(
                     mapping, core_energy, one_body.data(), one_body.data(),
                     index, value, 1, 2, true, 1e-12, 1e-12),
                 std::invalid_argument);
  }
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

namespace {

// Real part of every coefficient: the Hermitian part of a qubit operator.
MajoranaMapResult real_part(MajoranaMapResult result) {
  for (auto& coefficient : result.coefficients)
    coefficient = coefficient.real();
  return result;
}

double largest_imaginary_part(const MajoranaMapResult& result) {
  double largest = 0.0;
  for (const auto& coefficient : result.coefficients)
    largest = std::max(largest, std::abs(coefficient.imag()));
  return largest;
}

// Collects warnings logged while it is alive.
class WarningCapture {
 public:
  WarningCapture()
      : sink_(std::make_shared<spdlog::sinks::ostream_sink_mt>(stream_)),
        level_(qdk::chemistry::utils::Logger::get_global_level()) {
    sink_->set_level(spdlog::level::warn);
    QDK_RAW_LOGGER()->sinks().push_back(sink_);
    qdk::chemistry::utils::Logger::set_global_level(
        qdk::chemistry::utils::LogLevel::warn);
  }
  ~WarningCapture() {
    auto& sinks = QDK_RAW_LOGGER()->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
    qdk::chemistry::utils::Logger::set_global_level(level_);
  }
  WarningCapture(const WarningCapture&) = delete;
  WarningCapture& operator=(const WarningCapture&) = delete;

  std::string text() const { return stream_.str(); }

 private:
  std::ostringstream stream_;
  std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink_;
  qdk::chemistry::utils::LogLevel level_;
};

}  // namespace

TEST(MajoranaMapEngineTest, MapsHermitianPartOfTwoBodyIntegrals) {
  constexpr std::size_t n = 3;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  auto map = [&](const std::vector<double>& aaaa,
                 const std::vector<double>& aabb,
                 const std::vector<double>& bbbb, bool restricted) {
    return majorana_map_hamiltonian(mapping, 0.0, one_body.data(),
                                    one_body.data(), aaaa.data(), aabb.data(),
                                    bbbb.data(), n, restricted, 1e-12, 0.0);
  };
  const auto alpha = asymmetric_tensor(n, 0.0);
  const auto alpha_beta = asymmetric_tensor(n, 1.0);
  const auto beta = asymmetric_tensor(n, 2.0);
  const auto alpha_part = fourfold_average(alpha, n);
  const auto restricted = map(alpha_part, alpha_part, alpha_part, true);
  expect_same_coefficients(map(alpha, alpha, alpha, true), restricted, 2 * n);
  // Without spin_symmetric the blocks are unrestricted, even in one buffer.
  expect_same_coefficients(
      map(alpha, alpha, alpha, false),
      map(alpha_part, reversal_average(alpha, n), alpha_part, false), 2 * n);
  expect_same_coefficients(map(alpha, alpha_beta, beta, false),
                           map(alpha_part, reversal_average(alpha_beta, n),
                               fourfold_average(beta, n), false),
                           2 * n);
}

TEST(MajoranaMapEngineTest, HermitianPartMatchesLiteralOperator) {
  // The Hermitian mapping is the Hermitian part of the literal one.
  constexpr std::size_t n = 3;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  auto expect_hermitian_part = [&](const Hamiltonian& hermitian,
                                   const Hamiltonian& non_hermitian,
                                   bool restricted) {
    const auto literal = majorana_map_hamiltonian(mapping, non_hermitian,
                                                  restricted, 1e-12, 0.0);
    EXPECT_GT(largest_imaginary_part(literal), 1e-3);
    expect_same_coefficients(
        majorana_map_hamiltonian(mapping, hermitian, restricted, 1e-12, 0.0),
        real_part(literal), 2 * n);
  };
  auto dense = [&](const std::vector<double>& aaaa,
                   const std::vector<double>& aabb,
                   const std::vector<double>& bbbb, bool restricted,
                   HamiltonianType type) {
    return model_hamiltonian(one_body.data(), one_body.data(), aaaa, aabb, bbbb,
                             n, restricted, type);
  };
  const auto restricted = asymmetric_tensor(n, 0.0);
  for (const auto type :
       {HamiltonianType::Hermitian, HamiltonianType::NonHermitian}) {
    // A restricted Hamiltonian is mapped spin-summed even without
    // spin_symmetric.
    const auto hamiltonian =
        dense(restricted, restricted, restricted, true, type);
    expect_same_coefficients(
        majorana_map_hamiltonian(mapping, hamiltonian, false, 1e-12, 0.0),
        majorana_map_hamiltonian(mapping, hamiltonian, true, 1e-12, 0.0),
        2 * n);
  }
  expect_hermitian_part(dense(restricted, restricted, restricted, true,
                              HamiltonianType::Hermitian),
                        dense(restricted, restricted, restricted, true,
                              HamiltonianType::NonHermitian),
                        true);
  const auto alpha = asymmetric_tensor(n, 0.0);
  const auto alpha_beta = asymmetric_tensor(n, 1.0);
  const auto beta = asymmetric_tensor(n, 2.0);
  expect_hermitian_part(
      dense(alpha, alpha_beta, beta, false, HamiltonianType::Hermitian),
      dense(alpha, alpha_beta, beta, false, HamiltonianType::NonHermitian),
      false);

  Eigen::MatrixXd factors(n * n, 2);
  for (Eigen::Index index = 0; index < factors.size(); ++index)
    factors.data()[index] = std::cos(0.9 * static_cast<double>(index));
  auto cholesky = [&](HamiltonianType type) {
    return Hamiltonian(std::make_unique<CholeskyHamiltonianContainer>(
        square_matrix(one_body.data(), n), factors, model_orbitals(n, true),
        0.0, Eigen::MatrixXd::Zero(0, 0), std::nullopt, type));
  };
  expect_hermitian_part(cholesky(HamiltonianType::Hermitian),
                        cholesky(HamiltonianType::NonHermitian), true);

  // (01|00) and its bra-ket partner without their Hermitian conjugates.
  Eigen::SparseMatrix<double> sparse_one_body(n, n);
  for (std::size_t p = 0; p < n; ++p)
    sparse_one_body.insert(static_cast<Eigen::Index>(p),
                           static_cast<Eigen::Index>(p)) =
        one_body[p * n + p] + 1.0;
  sparse_one_body.makeCompressed();
  const SparseHamiltonianContainer::TwoBodyMap entries{{{0, 1, 0, 0}, 0.3},
                                                       {{0, 0, 0, 1}, 0.3},
                                                       {{1, 2, 1, 2}, 0.2},
                                                       {{2, 1, 2, 1}, 0.2}};
  auto sparse = [&](HamiltonianType type) {
    return Hamiltonian(std::make_unique<SparseHamiltonianContainer>(
        sparse_one_body, entries, 0.0, type));
  };
  expect_hermitian_part(sparse(HamiltonianType::Hermitian),
                        sparse(HamiltonianType::NonHermitian), true);
}

TEST(MajoranaMapEngineTest, SparseEntriesAreReadLiterallyAndAveraged) {
  namespace detail = qdk::chemistry::data::detail;
  constexpr std::size_t n = 2;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const double one_body[4] = {1.0, 0.3, 0.3, 0.5};
  // (01|01) and (11|00) are the only stored members of their 8-fold classes,
  // the deprecated shorthand for the whole class; (00|01) and (01|00) are
  // part of a class read literally.
  const int indices[16] = {0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0};
  const double values[4] = {0.7, 0.4, 0.3, 0.3};
  std::vector<double> stored(n * n * n * n, 0.0);
  for (const auto& [key, value] :
       {std::pair{detail::TwoBodyKey{0, 1, 0, 1}, 0.7},
        std::pair{detail::TwoBodyKey{1, 1, 0, 0}, 0.4}})
    for (const auto& k : detail::eightfold_orbit(key))
      stored[flat_index(n, k[0], k[1], k[2], k[3])] = value;
  stored[flat_index(n, 0, 0, 0, 1)] = stored[flat_index(n, 0, 1, 0, 0)] = 0.3;
  const auto average = fourfold_average(stored, n);

  WarningCapture warnings;
  const auto sparse = majorana_map_hamiltonian_sparse(
      mapping, 0.5, one_body, one_body, indices, values, 4, n, true, 0.0, 0.0);
  EXPECT_NE(warnings.text().find("deprecated"), std::string::npos)
      << warnings.text();
  EXPECT_NE(warnings.text().find("Hermitian part"), std::string::npos)
      << warnings.text();
  expect_same_coefficients(
      sparse,
      majorana_map_hamiltonian(mapping, 0.5, one_body, one_body, average.data(),
                               average.data(), average.data(), n, true, 0.0,
                               0.0),
      2 * n);
}

TEST(MajoranaMapEngineTest, CholeskyAveragesLikeDense) {
  constexpr std::size_t n = 3;
  constexpr std::size_t naux = 2;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  Eigen::MatrixXd alpha(n * n, naux);
  Eigen::MatrixXd beta(n * n, naux);
  for (Eigen::Index index = 0; index < alpha.size(); ++index) {
    alpha.data()[index] = std::cos(0.9 * static_cast<double>(index));
    beta.data()[index] = std::sin(1.3 * static_cast<double>(index));
  }
  // Row-major (pq|rs) = sum_Q L^Q_pq R^Q_rs.
  auto reconstruct = [](const Eigen::MatrixXd& left,
                        const Eigen::MatrixXd& right) {
    const Eigen::MatrixXd tensor = right * left.transpose();
    return std::vector<double>(tensor.data(), tensor.data() + tensor.size());
  };
  const auto aaaa = reconstruct(alpha, alpha);
  const auto aabb = reconstruct(alpha, beta);
  const auto bbbb = reconstruct(beta, beta);
  expect_same_coefficients(
      majorana_map_hamiltonian_cholesky(
          mapping, 0.0, one_body.data(), one_body.data(), alpha.data(),
          alpha.data(), n, naux, true, 1e-12, 0.0),
      majorana_map_hamiltonian(mapping, 0.0, one_body.data(), one_body.data(),
                               aaaa.data(), aaaa.data(), aaaa.data(), n, true,
                               1e-12, 0.0),
      2 * n);
  expect_same_coefficients(
      majorana_map_hamiltonian_cholesky(
          mapping, 0.0, one_body.data(), one_body.data(), alpha.data(),
          beta.data(), n, naux, false, 1e-12, 0.0),
      majorana_map_hamiltonian(mapping, 0.0, one_body.data(), one_body.data(),
                               aaaa.data(), aabb.data(), bbbb.data(), n, false,
                               1e-12, 0.0),
      2 * n);
}

TEST(MajoranaMapEngineTest, WarnsOnlyWhenAveragingChangesTheOperator) {
  constexpr std::size_t n = 3;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const auto one_body = diagonal_one_body(n);
  auto dense_warnings = [&](const std::vector<double>& aaaa,
                            const std::vector<double>& aabb,
                            const std::vector<double>& bbbb, bool restricted) {
    WarningCapture warnings;
    majorana_map_hamiltonian(mapping, 0.0, one_body.data(), one_body.data(),
                             aaaa.data(), aabb.data(), bbbb.data(), n,
                             restricted, 1e-12, 0.0);
    return warnings.text();
  };
  auto cholesky_warnings = [&](const Eigen::MatrixXd& alpha,
                               const Eigen::MatrixXd& beta, bool restricted) {
    WarningCapture warnings;
    majorana_map_hamiltonian_cholesky(
        mapping, 0.0, one_body.data(), one_body.data(), alpha.data(),
        beta.data(), n, static_cast<std::size_t>(alpha.cols()), restricted,
        1e-12, 0.0);
    return warnings.text();
  };

  // Exact, or Hermitian in another bra-ket gauge: the operator is unchanged.
  const auto fourfold = fourfold_test_tensor(n);
  EXPECT_EQ(dense_warnings(fourfold, fourfold, fourfold, true), "");
  auto gauge = fourfold;
  gauge[flat_index(n, 0, 1, 2, 2)] += 0.5;
  gauge[flat_index(n, 2, 2, 0, 1)] -= 0.5;
  EXPECT_EQ(dense_warnings(gauge, gauge, gauge, true), "");

  const auto asymmetric = asymmetric_tensor(n, 0.0);
  EXPECT_NE(dense_warnings(asymmetric, asymmetric, asymmetric, true)
                .find("Hermitian part"),
            std::string::npos);
  EXPECT_NE(dense_warnings(fourfold, asymmetric, fourfold, false)
                .find("majorana_map_hamiltonian (alpha-beta block)"),
            std::string::npos);

  // Separate symmetric and antisymmetric pair parts give fourfold integrals.
  Eigen::MatrixXd fourfold_factors = Eigen::MatrixXd::Zero(n * n, 2);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      const double sign = p < q ? 1.0 : p > q ? -1.0 : 0.0;
      fourfold_factors(p * n + q, 0) = 1.0 + 0.1 * static_cast<double>(p + q);
      fourfold_factors(p * n + q, 1) =
          0.4 * sign * static_cast<double>(1 + p * q);
    }
  EXPECT_EQ(cholesky_warnings(fourfold_factors, fourfold_factors, true), "");
  EXPECT_EQ(cholesky_warnings(fourfold_factors, fourfold_factors, false), "");

  Eigen::MatrixXd factors(n * n, 2);
  for (Eigen::Index index = 0; index < factors.size(); ++index)
    factors.data()[index] = std::cos(0.9 * static_cast<double>(index));
  EXPECT_NE(cholesky_warnings(factors, factors, true).find("Hermitian part"),
            std::string::npos);
  const auto unrestricted = cholesky_warnings(fourfold_factors, factors, false);
  EXPECT_EQ(unrestricted.find("(alpha-alpha block)"), std::string::npos)
      << unrestricted;
  EXPECT_NE(unrestricted.find("(alpha-beta block)"), std::string::npos)
      << unrestricted;
  EXPECT_NE(unrestricted.find("(beta-beta block)"), std::string::npos)
      << unrestricted;
}

namespace {

// Terms keyed by spin block and in-block qubit, so that Jordan-Wigner
// mappings of the same orbitals embedded in different spaces compare directly.
std::map<std::string, std::complex<double>> block_terms(
    const MajoranaMapResult& result, std::size_t n_spatial) {
  std::map<std::string, std::complex<double>> terms;
  for (std::size_t i = 0; i < result.words.size(); ++i) {
    std::string label;
    for (const auto& [qubit, op] : result.words[i])
      label += (qubit < n_spatial ? "a" : "b") +
               std::to_string(qubit % n_spatial) + "XYZ"[op - 1];
    terms[label] += result.coefficients[i];
  }
  return terms;
}

void expect_same_block_terms(const MajoranaMapResult& actual,
                             std::size_t actual_n,
                             const MajoranaMapResult& expected,
                             std::size_t expected_n) {
  const auto actual_terms = block_terms(actual, actual_n);
  const auto expected_terms = block_terms(expected, expected_n);
  ASSERT_EQ(actual_terms.size(), expected_terms.size());
  for (const auto& [label, coefficient] : expected_terms) {
    const auto it = actual_terms.find(label);
    ASSERT_NE(it, actual_terms.end()) << label;
    EXPECT_NEAR(std::abs(it->second - coefficient), 0.0, 1e-12) << label;
  }
}

}  // namespace

TEST(MajoranaMapEngineTest, SparseMappingBeyondTheDenseWidthMatchesSmallOne) {
  // Sparse mappings beyond 1024 qubits match the same terms mapped within it.
  constexpr std::size_t active = 3;
  const std::vector<double> one_body{0.3,  0.05, 0.0,  0.05, -0.2,
                                     0.08, 0.0,  0.08, 0.1};
  std::vector<double> eightfold(active * active * active * active);
  for (std::size_t p = 0; p < active; ++p)
    for (std::size_t q = 0; q < active; ++q)
      for (std::size_t r = 0; r < active; ++r)
        for (std::size_t s = 0; s < active; ++s) {
          const auto c =
              qdk::chemistry::data::detail::eightfold_canonical({p, q, r, s});
          eightfold[flat_index(active, p, q, r, s)] =
              0.3 * std::cos(1.0 + static_cast<double>(flat_index(
                                       active, c[0], c[1], c[2], c[3])));
        }
  const auto fourfold = fourfold_test_tensor(active);
  const auto asymmetric = asymmetric_tensor(active, 0.4);
  auto map_embedded = [&](std::size_t n, const std::vector<double>& two_body,
                          HamiltonianType type, double integral_threshold) {
    Eigen::SparseMatrix<double> h(n, n);
    SparseHamiltonianContainer::TwoBodyMap entries;
    for (std::size_t p = 0; p < active; ++p) {
      for (std::size_t q = 0; q < active; ++q) {
        if (one_body[p * active + q] != 0.0)
          h.insert(p, q) = one_body[p * active + q];
        for (std::size_t r = 0; r < active; ++r)
          for (std::size_t s = 0; s < active; ++s)
            entries[{static_cast<int>(p), static_cast<int>(q),
                     static_cast<int>(r), static_cast<int>(s)}] =
                two_body[flat_index(active, p, q, r, s)];
      }
    }
    return majorana_map_hamiltonian(
        MajoranaMapping::jordan_wigner(2 * n),
        Hamiltonian(std::make_unique<SparseHamiltonianContainer>(h, entries,
                                                                 0.0, type)),
        true, 1e-12, integral_threshold);
  };
  using Case = std::tuple<const std::vector<double>*, HamiltonianType, double>;
  for (const auto& [two_body, type, integral_threshold] :
       {Case{&eightfold, HamiltonianType::Hermitian, 1e-12},
        Case{&fourfold, HamiltonianType::Hermitian, 1e-12},
        Case{&fourfold, HamiltonianType::Hermitian, 0.1},
        Case{&asymmetric, HamiltonianType::NonHermitian, 1e-12}}) {
    expect_same_block_terms(
        map_embedded(520, *two_body, type, integral_threshold), 520,
        map_embedded(4, *two_body, type, integral_threshold), 4);
  }

  // A beta qubit at 65536 needs the widest word count; one at 131072 exceeds
  // it.
  const double h1[1] = {1.0};
  const int index[4] = {0, 0, 0, 0};
  const double value[1] = {0.75};
  auto map_gapped = [&](std::uint64_t beta) {
    return majorana_map_hamiltonian_sparse(
        MajoranaMapping::from_table(
            {{{0, 1}}, {{0, 2}}, {{0, 3}, {beta, 1}}, {{0, 3}, {beta, 2}}}),
        0.5, h1, h1, index, value, 1, 1, true, 1e-12, 1e-12);
  };
  expect_same_block_terms(map_gapped(65536), 1, map_gapped(1), 1);
  EXPECT_THROW(map_gapped(131072), std::invalid_argument);

  // Verstraete-Cirac stabilizer penalties map alike with every qubit shifted
  // past 1024.
  const auto vc = MajoranaMapping::verstraete_cirac(
      LatticeGraph({{{0, 1}, 1.0}, {{1, 2}, 1.0}, {{0, 2}, 1.0}}, 3));
  auto shifted_json = vc.to_json();
  auto shift = [](nlohmann::json& word) {
    for (auto& entry : word) entry[0] = entry[0].get<std::uint64_t>() + 1024;
  };
  for (auto& bilinear : shifted_json["bilinears"]) shift(bilinear["word"]);
  for (auto& stabilizer : shifted_json["stabilizers"])
    shift(stabilizer["word"]);
  const auto shifted_vc = MajoranaMapping::from_json(shifted_json);
  ASSERT_GT(shifted_vc.num_qubits(), 1024u);
  std::vector<int> indices;
  std::vector<double> values;
  for (std::size_t p = 0; p < active; ++p)
    for (std::size_t q = 0; q < active; ++q)
      for (std::size_t r = 0; r < active; ++r)
        for (std::size_t s = 0; s < active; ++s) {
          indices.insert(indices.end(), {int(p), int(q), int(r), int(s)});
          values.push_back(eightfold[flat_index(active, p, q, r, s)]);
        }
  auto map_vc = [&](const MajoranaMapping& mapping) {
    return majorana_map_hamiltonian_sparse(
        mapping, 0.5, one_body.data(), one_body.data(), indices.data(),
        values.data(), values.size(), active, true, 1e-12, 1e-12);
  };
  auto shifted = map_vc(shifted_vc);
  for (auto& word : shifted.words)
    for (auto& [qubit, op] : word) qubit -= 1024;
  expect_same_coefficients(shifted, map_vc(vc), vc.num_qubits());
}

TEST(MajoranaMapEngineTest, DenseAndCholeskyRejectMappingsBeyondTheirWidth) {
  // Dense and Cholesky reject the width before reading their N^4 integrals.
  constexpr std::size_t n = 520;
  const auto mapping = MajoranaMapping::jordan_wigner(2 * n);
  const std::vector<double> one_body(n * n, 0.0);
  const double placeholder[1] = {0.0};
  EXPECT_THROW(majorana_map_hamiltonian(
                   mapping, 0.0, one_body.data(), one_body.data(), placeholder,
                   placeholder, placeholder, n, true, 1e-12, 1e-12),
               std::invalid_argument);
  EXPECT_THROW(majorana_map_hamiltonian_cholesky(
                   mapping, 0.0, one_body.data(), one_body.data(), placeholder,
                   placeholder, n, 1, true, 1e-12, 1e-12),
               std::invalid_argument);
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
