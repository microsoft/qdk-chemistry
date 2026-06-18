// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <complex>
#include <map>
#include <qdk/chemistry/data/lattice_graph.hpp>
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

TEST(VerstraeteCiracFactoryTest, RejectsTooFewSites) {
  auto lattice = LatticeGraph::square(1, 2);
  EXPECT_THROW(MajoranaMapping::verstraete_cirac(lattice),
               std::invalid_argument);
}

TEST(VerstraeteCiracFactoryTest, SquareLatticeDimensions) {
  auto lattice = LatticeGraph::square(2, 2);
  const std::size_t n_sites = lattice.num_sites();
  auto mapping = MajoranaMapping::verstraete_cirac(lattice);

  EXPECT_EQ(mapping.name(), "verstraete-cirac");
  EXPECT_EQ(mapping.base_encoding(), "verstraete-cirac");
  EXPECT_EQ(mapping.num_modes(), 2 * n_sites);
  EXPECT_GE(mapping.num_qubits(), mapping.num_modes());
  EXPECT_FALSE(mapping.is_majorana_atomic());
  EXPECT_FALSE(mapping.stabilizers().empty());
  EXPECT_EQ(mapping.num_qubits() - mapping.stabilizers().size(), 2 * n_sites);
}

TEST(VerstraeteCiracFactoryTest, ChainGraphNeedsNoLatticePresets) {
  auto chain = LatticeGraph::chain(5);
  auto mapping = MajoranaMapping::verstraete_cirac(chain);

  EXPECT_EQ(mapping.num_modes(), 2 * chain.num_sites());
  EXPECT_EQ(mapping.num_qubits(), mapping.num_modes());
  EXPECT_TRUE(mapping.stabilizers().empty());
}

TEST(VerstraeteCiracFactoryTest, CustomEdgeListGraphAccepted) {
  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  // 5-cycle: topology is not square/triangular/honeycomb/kagome.
  std::map<Edge, double> edges = {
      {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 3}, 1.0}, {{3, 4}, 1.0}, {{0, 4}, 1.0},
  };
  auto graph = LatticeGraph::make_bidirectional(LatticeGraph(edges, 5));
  auto mapping = MajoranaMapping::verstraete_cirac(graph);

  EXPECT_EQ(mapping.num_modes(), 10);
  EXPECT_GE(mapping.num_qubits(), mapping.num_modes());
  EXPECT_FALSE(mapping.stabilizers().empty());
}

TEST(VerstraeteCiracFactoryTest, StabilizersAreHermitianPauliOperators) {
  auto mapping = MajoranaMapping::verstraete_cirac(LatticeGraph::square(3, 3));
  for (const auto& [coeff, word] : mapping.stabilizers()) {
    EXPECT_NEAR(coeff.imag(), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(coeff.real()), 1.0, 1e-12);
    for (const auto& [qubit, op] : word) {
      EXPECT_TRUE(op == 1 || op == 2 || op == 3);
      (void)qubit;
    }
  }
}

TEST(VerstraeteCiracFactoryTest, MajoranaAccessorRaises) {
  auto mapping = MajoranaMapping::verstraete_cirac(LatticeGraph::square(2, 2));
  EXPECT_THROW(mapping.majorana(0), std::logic_error);
}

TEST(VerstraeteCiracFactoryTest, HashIncludesStabilizers) {
  auto first = MajoranaMapping::verstraete_cirac(LatticeGraph::square(2, 2));
  auto second = MajoranaMapping::verstraete_cirac(LatticeGraph::square(2, 3));
  EXPECT_NE(first.content_hash(), second.content_hash());
}

TEST(VerstraeteCiracEngineTest, IdentityCoefficientMatchesJordanWigner) {
  auto lattice = LatticeGraph::square(2, 2);
  auto vc = MajoranaMapping::verstraete_cirac(lattice);
  auto jw = MajoranaMapping::jordan_wigner(vc.num_modes());
  const double h1_alpha[4] = {1.0, 0.0, 0.0, 0.5};
  const double h1_beta[4] = {2.0, 0.0, 0.0, 1.5};
  const double eri_zero[16] = {};

  auto jw_result = majorana_map_hamiltonian(
      jw, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);
  auto vc_result = majorana_map_hamiltonian(
      vc, 0.0, h1_alpha, h1_beta, eri_zero, eri_zero, eri_zero,
      /*n_spatial=*/2, /*spin_symmetric=*/false, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12);

  auto jw_terms = collect_terms(jw_result, jw.num_qubits());
  auto vc_terms = collect_terms(vc_result, vc.num_qubits());

  expect_real_term(jw_terms, "IIIIIIII", 2.5);
  expect_real_term(vc_terms, std::string(vc.num_qubits(), 'I'), 2.5);
}
