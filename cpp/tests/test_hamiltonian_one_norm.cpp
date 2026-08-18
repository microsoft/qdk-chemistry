// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>

#include "ut_common.hpp"

using qdk::chemistry::utils::double_factorize;
using qdk::chemistry::utils::hamiltonian_one_norm;
using qdk::chemistry::utils::TwoBodyFragment;

class HamiltonianOneNormTest : public ::testing::Test {};

TEST_F(HamiltonianOneNormTest, WaterSTO3GIsPositiveAndConsistent) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto norm = hamiltonian_one_norm(*ham, 0.0);
  EXPECT_GT(norm.one_body, 0.0);
  EXPECT_GT(norm.two_body, 0.0);
  EXPECT_NEAR(norm.total, norm.one_body + norm.two_body,
              testing::numerical_zero_tolerance);

  // No truncation is the default: calling without an explicit threshold
  // should give the same result as threshold=0.0.
  auto norm_default = hamiltonian_one_norm(*ham);
  EXPECT_NEAR(norm_default.total, norm.total,
              testing::numerical_zero_tolerance);
}

TEST_F(HamiltonianOneNormTest, TruncationNeverIncreasesTwoBodyNorm) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto norm_exact = hamiltonian_one_norm(*ham, 0.0);
  auto norm_truncated = hamiltonian_one_norm(*ham, 1e-2);
  // Truncating fragments removes contributions to the low-rank
  // reconstruction, so the reported two-body 1-norm (computed from the
  // retained fragments only) should not exceed the exact value.
  EXPECT_LE(norm_truncated.two_body,
            norm_exact.two_body + testing::numerical_zero_tolerance);
}

namespace {

/// Reconstruct the flattened g_ijkl tensor from a set of DF fragments:
///   g_ijkl = sum_alpha sign_alpha * sum_pq U_ip U_jp eps_p eps_q U_kq U_lq
Eigen::VectorXd reconstruct(const std::vector<TwoBodyFragment>& fragments,
                            size_t norb) {
  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (const auto& fragment : fragments) {
    // M_ij = sum_p U_ip eps_p U_jp  (i.e. U * diag(eps) * U^T)
    Eigen::MatrixXd M =
        fragment.U * fragment.eps.asDiagonal() * fragment.U.transpose();
    for (size_t i = 0; i < norb; ++i) {
      for (size_t j = 0; j < norb; ++j) {
        for (size_t k = 0; k < norb; ++k) {
          for (size_t l = 0; l < norb; ++l) {
            g[i * norb * norb * norb + j * norb * norb + k * norb + l] +=
                fragment.sign * M(i, j) * M(k, l);
          }
        }
      }
    }
  }
  return g;
}

}  // namespace

// double_factorize() is the fragment decomposition that hamiltonian_one_norm()
// (tested above) uses internally to compute the two-body 1-norm; both are
// tested in this file since they share the same water/STO-3G test fixture.
class DoubleFactorizationTest : public ::testing::Test {};

TEST_F(DoubleFactorizationTest, ExactReconstructionNoTruncation) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  auto [g_aaaa, g_aabb, g_bbbb] = ham->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;
  const size_t n =
      static_cast<size_t>(ham->get_orbitals()->get_num_molecular_orbitals());

  // Double-factorize with threshold=0.0 (no truncation): reconstruction
  // should reproduce the original tensor to machine precision.
  auto fragments = double_factorize(g_aaaa, n, 0.0);
  ASSERT_FALSE(fragments.empty());
  Eigen::VectorXd g_reconstructed = reconstruct(fragments, n);
  EXPECT_TRUE(
      g_reconstructed.isApprox(g_aaaa, testing::numerical_zero_tolerance * 100))
      << "Reconstruction max abs diff: "
      << (g_reconstructed - g_aaaa).cwiseAbs().maxCoeff();
}

TEST_F(DoubleFactorizationTest,
       TruncationReducesFragmentCountAndDefaultsToZero) {
  auto water = testing::create_water_structure();
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());
  auto [g_aaaa, g_aabb, g_bbbb] = ham->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;
  const size_t n =
      static_cast<size_t>(ham->get_orbitals()->get_num_molecular_orbitals());

  // The default threshold argument is 0.0 (no truncation): calling without
  // specifying it should give the same fragment count as threshold=0.0.
  auto fragments_default = double_factorize(g_aaaa, n);
  auto fragments_explicit_zero = double_factorize(g_aaaa, n, 0.0);
  EXPECT_EQ(fragments_default.size(), fragments_explicit_zero.size());

  // A larger threshold should never retain more fragments than a smaller one.
  auto fragments_loose = double_factorize(g_aaaa, n, 1e-2);
  EXPECT_LE(fragments_loose.size(), fragments_explicit_zero.size());
}
