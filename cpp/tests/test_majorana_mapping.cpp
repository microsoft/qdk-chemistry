// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

// ── Ground-truth reference for the mapping ─────────────────────────
//
// Comparing sorted eigenvalues makes the check independent of qubit and mode
// ordering: relabeling either is a unitary conjugation, so the spectrum is
// invariant.  That lets the reference below be built straight from the second
// quantized Hamiltonian without reproducing the engine's mode layout.

std::vector<double> mapped_spectrum(const MajoranaMapResult& result,
                                    std::size_t num_qubits) {
  const std::size_t dim = std::size_t{1} << num_qubits;
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(dim, dim);
  for (std::size_t t = 0; t < result.words.size(); ++t) {
    const std::string label = sparse_to_dense_le(result.words[t], num_qubits);
    // Every Pauli word maps each basis state to exactly one other state, so
    // the term is applied column by column instead of built by Kronecker
    // products.
    for (std::size_t column = 0; column < dim; ++column) {
      std::size_t row = column;
      std::complex<double> phase = result.coefficients[t];
      for (std::size_t qubit = 0; qubit < num_qubits; ++qubit) {
        const bool set = ((column >> qubit) & std::size_t{1}) != 0;
        switch (label[num_qubits - 1 - qubit]) {
          case 'X':
            row ^= std::size_t{1} << qubit;
            break;
          case 'Y':
            row ^= std::size_t{1} << qubit;
            phase *= std::complex<double>(0.0, set ? -1.0 : 1.0);
            break;
          case 'Z':
            if (set) phase = -phase;
            break;
          default:
            break;
        }
      }
      matrix(static_cast<Eigen::Index>(row),
             static_cast<Eigen::Index>(column)) += phase;
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(matrix);
  const auto& values = solver.eigenvalues();
  return std::vector<double>(values.data(), values.data() + values.size());
}

// Spectrum of
//   H = core + sum_{pq,sigma} h_pq a+_{p sigma} a_{q sigma}
//     + 1/2 sum_{pqrs,sigma tau} (pq|rs) a+_{p sigma} a+_{r tau}
//                                        a_{s tau} a_{q sigma}
// built directly in the occupation-number basis, sharing no code with the
// mapping engine.  Mode 2*p + spin, bit `mode` of the basis index.
std::vector<double> fock_spectrum(const double* h1, const double* eri,
                                  std::size_t n, double core_energy) {
  const std::size_t modes = 2 * n;
  const std::size_t dim = std::size_t{1} << modes;

  // Returns false when the operator annihilates the state.
  auto apply = [](std::size_t& state, double& sign, std::size_t mode,
                  bool dagger) {
    const bool occupied = ((state >> mode) & std::size_t{1}) != 0;
    if (dagger == occupied) return false;
    int below = 0;
    for (std::size_t m = 0; m < mode; ++m) {
      below += static_cast<int>((state >> m) & std::size_t{1});
    }
    if (below % 2 != 0) sign = -sign;
    state ^= std::size_t{1} << mode;
    return true;
  };

  Eigen::MatrixXd matrix = core_energy * Eigen::MatrixXd::Identity(dim, dim);
  for (std::size_t column = 0; column < dim; ++column) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = 0; q < n; ++q) {
        for (std::size_t spin = 0; spin < 2; ++spin) {
          std::size_t state = column;
          double sign = 1.0;
          if (!apply(state, sign, 2 * q + spin, false)) continue;
          if (!apply(state, sign, 2 * p + spin, true)) continue;
          matrix(static_cast<Eigen::Index>(state),
                 static_cast<Eigen::Index>(column)) += sign * h1[p * n + q];
        }
      }
    }
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = 0; q < n; ++q) {
        for (std::size_t r = 0; r < n; ++r) {
          for (std::size_t s = 0; s < n; ++s) {
            const double value = eri[((p * n + q) * n + r) * n + s];
            if (value == 0.0) continue;
            for (std::size_t sigma = 0; sigma < 2; ++sigma) {
              for (std::size_t tau = 0; tau < 2; ++tau) {
                std::size_t state = column;
                double sign = 1.0;
                if (!apply(state, sign, 2 * q + sigma, false)) continue;
                if (!apply(state, sign, 2 * s + tau, false)) continue;
                if (!apply(state, sign, 2 * r + tau, true)) continue;
                if (!apply(state, sign, 2 * p + sigma, true)) continue;
                matrix(static_cast<Eigen::Index>(state),
                       static_cast<Eigen::Index>(column)) += 0.5 * sign * value;
              }
            }
          }
        }
      }
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
  const auto& values = solver.eigenvalues();
  return std::vector<double>(values.data(), values.data() + values.size());
}

void expect_matches_fock_reference(const MajoranaMapResult& result,
                                   std::size_t num_qubits, const double* h1,
                                   const double* eri, std::size_t n_spatial,
                                   double core_energy) {
  const auto mapped = mapped_spectrum(result, num_qubits);
  const auto reference = fock_spectrum(h1, eri, n_spatial, core_energy);
  ASSERT_EQ(mapped.size(), reference.size());
  for (std::size_t i = 0; i < mapped.size(); ++i) {
    EXPECT_NEAR(mapped[i], reference[i], 1e-10)
        << "Eigenvalue " << i << " disagrees with the Fock-space reference";
  }
}

}  // namespace qdk::chemistry::tests::test_support

namespace test_support = qdk::chemistry::tests::test_support;
using test_support::collect_terms;
using test_support::expect_matches_fock_reference;
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

TEST(MajoranaMapEngineTest, SpinSummedPathMatchesFockSpaceReference) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  double eri[16] = {};
  auto idx = [](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * 2 + q) * 2 + r) * 2 + s;
  };
  // Fully 8-fold symmetric, so the spin-summed fast path is selected.
  for (std::size_t p = 0; p < 2; ++p) {
    for (std::size_t q = 0; q < 2; ++q) {
      for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t s = 0; s < 2; ++s) {
          eri[idx(p, q, r, s)] =
              0.3 + 0.1 * static_cast<double>((p + q) * (r + s));
        }
      }
    }
  }

  auto result = majorana_map_hamiltonian(mapping, 0.5, h1, h1, eri, eri, eri,
                                         /*n_spatial=*/2,
                                         /*spin_symmetric=*/true,
                                         /*threshold=*/0.0,
                                         /*integral_threshold=*/0.0);

  expect_matches_fock_reference(result, mapping.num_qubits(), h1, eri, 2, 0.5);
}

TEST(MajoranaMapEngineTest, FourFoldTwoBodyMatchesFockSpaceReference) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  double eri[16] = {};
  auto idx = [](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * 2 + q) * 2 + r) * 2 + s;
  };

  // Four-fold symmetry: (pq|rs) = (qp|sr) = (rs|pq) = (sr|qp), but
  // (01|01) != (10|01), so independent bra/ket swaps are not valid.
  eri[idx(0, 1, 0, 1)] = 0.7;
  eri[idx(1, 0, 1, 0)] = 0.7;
  eri[idx(1, 0, 0, 1)] = -0.2;
  eri[idx(0, 1, 1, 0)] = -0.2;

  auto result = majorana_map_hamiltonian(mapping, 0.5, h1, h1, eri, eri, eri,
                                         /*n_spatial=*/2,
                                         /*spin_symmetric=*/true,
                                         /*threshold=*/0.0,
                                         /*integral_threshold=*/0.0);

  expect_matches_fock_reference(result, mapping.num_qubits(), h1, eri, 2, 0.5);
}

TEST(MajoranaMapEngineTest, RejectsBraKetAsymmetricStorage) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  double eri[16] = {};
  auto idx = [](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * 2 + q) * 2 + r) * 2 + s;
  };

  // Hermitian, and its bra-ket average is fully 8-fold, but the cross-spin
  // channel reads the stored tensor directly, so the caller has to average.
  eri[idx(0, 0, 1, 1)] = 0.4;
  eri[idx(1, 1, 0, 0)] = 0.9;

  try {
    majorana_map_hamiltonian(mapping, 0.5, h1, h1, eri, eri, eri,
                             /*n_spatial=*/2, /*spin_symmetric=*/true,
                             /*threshold=*/0.0, /*integral_threshold=*/0.0);
    FAIL() << "Expected bra-ket-asymmetric storage to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("bra-ket average"),
              std::string::npos);
  }
}

TEST(MajoranaMapEngineTest, BraKetAverageOfRejectedStorageMapsExactly) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  double eri[16] = {};
  auto idx = [](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * 2 + q) * 2 + r) * 2 + s;
  };
  // The remedy the rejection above asks for: same operator, mapped exactly.
  eri[idx(0, 0, 1, 1)] = 0.65;
  eri[idx(1, 1, 0, 0)] = 0.65;

  auto result = majorana_map_hamiltonian(mapping, 0.5, h1, h1, eri, eri, eri,
                                         /*n_spatial=*/2,
                                         /*spin_symmetric=*/true,
                                         /*threshold=*/0.0,
                                         /*integral_threshold=*/0.0);

  expect_matches_fock_reference(result, mapping.num_qubits(), h1, eri, 2, 0.5);
}

TEST(MajoranaMapEngineTest, RejectsNonHermitianTwoBody) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  double eri[16] = {};
  auto idx = [](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * 2 + q) * 2 + r) * 2 + s;
  };

  // Bra-ket symmetric, so no gauge freedom is left, but (pq|rs) != (qp|sr):
  // the operator it denotes is not Hermitian.
  eri[idx(0, 1, 0, 0)] = 0.25;
  eri[idx(0, 0, 0, 1)] = 0.25;
  eri[idx(1, 0, 0, 0)] = -0.25;
  eri[idx(0, 0, 1, 0)] = -0.25;

  try {
    majorana_map_hamiltonian(mapping, 0.5, h1, h1, eri, eri, eri,
                             /*n_spatial=*/2, /*spin_symmetric=*/true,
                             /*threshold=*/0.0, /*integral_threshold=*/0.0);
    FAIL() << "Expected a non-Hermitian two-body tensor to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("non-Hermitian"),
              std::string::npos);
  }
}

TEST(MajoranaMapEngineTest, SymmetryToleranceScalesWithIntegralMagnitude) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};
  // Round-off left by an integral transformation is relative to the integral
  // magnitude, so a large tensor perturbed far below that relative scale must
  // still be accepted.  An absolute tolerance rejects this, and with it the
  // ordinary RHF Hamiltonians whose asymmetry grows with the basis.
  const double magnitude = 1.0e6;
  double eri[16] = {};
  for (double& value : eri) value = magnitude;
  eri[1] += 1.0e-9 * magnitude;

  EXPECT_NO_THROW(majorana_map_hamiltonian(mapping, 0.0, h1, h1, eri, eri, eri,
                                           /*n_spatial=*/2,
                                           /*spin_symmetric=*/true,
                                           /*threshold=*/0.0,
                                           /*integral_threshold=*/0.0));
}

TEST(MajoranaMapEngineTest, CholeskyRejectsPairAsymmetricFactors) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};

  // One antisymmetric factor: sum_Q L_pq L_rs stays bra-ket symmetric but
  // loses the p<->q symmetry the spin-summed path needs, and confirming the
  // weaker 4-fold symmetry would mean materializing the dense tensor.
  const double three_center[4] = {0.0, 0.6, -0.6, 0.0};

  try {
    majorana_map_hamiltonian_cholesky(mapping, 0.0, h1, h1, three_center,
                                      three_center, /*n_spatial=*/2,
                                      /*naux=*/1, /*spin_symmetric=*/true,
                                      /*threshold=*/0.0,
                                      /*integral_threshold=*/0.0);
    FAIL() << "Expected pair-asymmetric three-center factors to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("pair"), std::string::npos);
  }
}

TEST(MajoranaMapEngineTest, SparseRejectsDisagreeingPermutationClass) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};

  // Same permutation class as (0,1,0,1), but a different value: the sparse
  // path keeps one representative per class, so this cannot be represented.
  const int indices[8] = {0, 1, 0, 1, 1, 0, 0, 1};
  const double values[2] = {0.7, -0.2};

  try {
    majorana_map_hamiltonian_sparse(mapping, 0.5, h1, h1, indices, values,
                                    /*num_entries=*/2, /*n_spatial=*/2,
                                    /*spin_symmetric=*/true,
                                    /*threshold=*/1e-12,
                                    /*integral_threshold=*/1e-12);
    FAIL() << "Expected reduced-symmetry sparse input to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("8-fold"), std::string::npos);
  }
}

TEST(MajoranaMapEngineTest, SparseRejectsPartiallyStoredPermutationClass) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};

  // (01|01) and (10|10) are two of the four distinct members of one class;
  // the caller means the other two to be zero, but symmetry expansion would
  // overwrite them, so the intent is ambiguous rather than representable.
  const int indices[8] = {0, 1, 0, 1, 1, 0, 1, 0};
  const double values[2] = {0.7, 0.7};

  try {
    majorana_map_hamiltonian_sparse(mapping, 0.5, h1, h1, indices, values,
                                    /*num_entries=*/2, /*n_spatial=*/2,
                                    /*spin_symmetric=*/true,
                                    /*threshold=*/1e-12,
                                    /*integral_threshold=*/1e-12);
    FAIL() << "Expected a partially stored permutation class to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("permutation class"),
              std::string::npos);
  }
}

TEST(MajoranaMapEngineTest, SparseAcceptsRedundantSymmetryRelatedStorage) {
  auto mapping = MajoranaMapping::jordan_wigner(4);
  const double h1[4] = {1.0, 0.3, 0.3, 0.5};

  const int indices[8] = {0, 0, 1, 1, 1, 1, 0, 0};
  const double values[2] = {0.7, 0.7};

  EXPECT_NO_THROW(majorana_map_hamiltonian_sparse(
      mapping, 0.5, h1, h1, indices, values, /*num_entries=*/2,
      /*n_spatial=*/2, /*spin_symmetric=*/true, /*threshold=*/1e-12,
      /*integral_threshold=*/1e-12));
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
