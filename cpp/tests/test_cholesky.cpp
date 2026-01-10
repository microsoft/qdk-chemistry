// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>

#include "../src/qdk/chemistry/algorithms/microsoft/cholesky.hpp"
#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class CholeskyTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// TEST_F(CholeskyTest, CompareCholeskyWithEigenCholesky) {
//   // Test the pivoted Cholesky decomposition against Eigen's standard
//   Cholesky const int n = 100; Eigen::MatrixXd A = Eigen::MatrixXd::Random(n,
//   n); Eigen::MatrixXd M = A * A.transpose();

//   // Perform pivoted Cholesky with tight tolerance
//   double tolerance = testing::integral_tolerance;
//   Eigen::MatrixXd L = microsoft::pivoted_cholesky_decomposition(M,
//   tolerance); Eigen::MatrixXd M_approx = L * L.transpose();

//   // Compare with original
//   double max_error = (M - M_approx).cwiseAbs().maxCoeff();
//   EXPECT_LT(max_error, testing::numerical_zero_tolerance)
//       << "Cholesky reconstruction error too large";

//   // Verify against Eigen
//   Eigen::LLT<Eigen::MatrixXd> llt(M);
//   EXPECT_EQ(llt.info(), Eigen::Success) << "Eigen's Cholesky failed";
//   Eigen::MatrixXd L_eigen = llt.matrixL();
//   Eigen::MatrixXd M_eigen = L_eigen * L_eigen.transpose();

//   // Both should reconstruct M equally well
//   double eigen_error = (M - M_eigen).cwiseAbs().maxCoeff();
//   EXPECT_LT(eigen_error, testing::numerical_zero_tolerance);

//   // Errors should be comparable
//   EXPECT_LT(abs(max_error - eigen_error), testing::numerical_zero_tolerance)
//       << "Pivoted Cholesky error different than standard Cholesky";
// }

TEST_F(CholeskyTest, N2_Restricted_Comparison) {
  // 1. Setup N2
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(0.0, 0.0, 3.0)};
  std::vector<std::string> symbols = {"N", "N"};
  Structure structure(coordinates, symbols);
  auto structure_ptr = std::make_shared<Structure>(structure);

  // 2. Run SCF (RHF)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");
  auto [energy, wavefunction] =
      scf_factory->run(structure_ptr, 0, 1, "cc-pvdz");
  auto orbitals_scf = wavefunction->get_orbitals();

  // Create new Orbitals with active space
  auto coeffs = orbitals_scf->get_coefficients();
  auto energies = orbitals_scf->get_energies();
  auto overlap = orbitals_scf->get_overlap_matrix();
  auto basis = orbitals_scf->get_basis_set();

  // cholesky tolerance
  double tolerance = testing::integral_tolerance;

  // full space
  {
    auto orbitals = wavefunction->get_orbitals();

    // 3. Run Hamiltonian with Incore (Exact)
    auto ham_incore_factory = HamiltonianConstructorFactory::create("qdk");
    ham_incore_factory->settings().set("eri_method", "incore");
    auto ham_incore = ham_incore_factory->run(orbitals);

    // 4. Run Hamiltonian with Cholesky
    auto ham_chol_factory = HamiltonianConstructorFactory::create("qdk");
    ham_chol_factory->settings().set("eri_method", "cholesky");
    ham_chol_factory->settings().set("cholesky_tolerance", tolerance);
    auto ham_chol = ham_chol_factory->run(orbitals);

    // 5. Compare
    // One-body
    auto [aa_incore, bb_incore] = ham_incore->get_one_body_integrals();
    auto [aa_chol, bb_chol] = ham_chol->get_one_body_integrals();

    EXPECT_TRUE(aa_incore.isApprox(aa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(bb_incore.isApprox(bb_chol, testing::numerical_zero_tolerance));

    // Two-body
    auto [aaaa_incore, aabb_incore, bbbb_incore] =
        ham_incore->get_two_body_integrals();
    auto [aaaa_chol, aabb_chol, bbbb_chol] = ham_chol->get_two_body_integrals();

    EXPECT_EQ(aaaa_incore.size(), aaaa_chol.size());
    double max_diff = (aaaa_incore - aaaa_chol).cwiseAbs().maxCoeff();
    EXPECT_LT(max_diff, testing::numerical_zero_tolerance);

    EXPECT_EQ(aabb_incore.size(), aabb_chol.size());
    max_diff = (aabb_incore - aabb_chol).cwiseAbs().maxCoeff();
    EXPECT_LT(max_diff, testing::numerical_zero_tolerance);

    EXPECT_EQ(bbbb_incore.size(), bbbb_chol.size());
    max_diff = (bbbb_incore - bbbb_chol).cwiseAbs().maxCoeff();
    EXPECT_LT(max_diff, testing::numerical_zero_tolerance);
  }

  // continuous active space
  {
    auto active_space_selector =
        ActiveSpaceSelectorFactory::create("qdk_valence");
    active_space_selector->settings().set("num_active_electrons", 6);
    active_space_selector->settings().set("num_active_orbitals", 6);
    auto wavefunction_active = active_space_selector->run(wavefunction);
    auto orbitals = wavefunction_active->get_orbitals();

    // 3. Run Hamiltonian with Incore (Exact)
    auto ham_incore_factory = HamiltonianConstructorFactory::create("qdk");
    ham_incore_factory->settings().set("eri_method", "incore");
    auto ham_incore = ham_incore_factory->run(orbitals);

    // 4. Run Hamiltonian with Cholesky
    auto ham_chol_factory = HamiltonianConstructorFactory::create("qdk");
    ham_chol_factory->settings().set("eri_method", "cholesky");
    ham_chol_factory->settings().set("cholesky_tolerance", tolerance);
    auto ham_chol = ham_chol_factory->run(orbitals);

    // 5. Compare
    // One-body
    auto [aa_incore, bb_incore] = ham_incore->get_one_body_integrals();
    auto [aa_chol, bb_chol] = ham_chol->get_one_body_integrals();

    EXPECT_TRUE(aa_incore.isApprox(aa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(bb_incore.isApprox(bb_chol, testing::numerical_zero_tolerance));

    // Two-body
    auto [aaaa_incore, aabb_incore, bbbb_incore] =
        ham_incore->get_two_body_integrals();
    auto [aaaa_chol, aabb_chol, bbbb_chol] = ham_chol->get_two_body_integrals();

    EXPECT_TRUE(
        aaaa_incore.isApprox(aaaa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        aabb_incore.isApprox(aabb_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        bbbb_incore.isApprox(bbbb_chol, testing::numerical_zero_tolerance));

    // inactive fock matrix
    auto fock_incore = ham_incore->get_inactive_fock_matrix();
    auto fock_chol = ham_chol->get_inactive_fock_matrix();
    EXPECT_TRUE(fock_incore.first.isApprox(fock_chol.first,
                                           testing::numerical_zero_tolerance));
    EXPECT_TRUE(fock_incore.second.isApprox(fock_chol.second,
                                            testing::numerical_zero_tolerance));

    // core energy
    auto core_incore = ham_incore->get_core_energy();
    auto core_chol = ham_chol->get_core_energy();
    EXPECT_NEAR(core_incore, core_chol, testing::numerical_zero_tolerance);
  }

  // discontinuous active space
  {
    auto full_orbitals = wavefunction->get_orbitals();
    // manual active space selection
    std::vector<size_t> active_alpha = {2, 3, 5, 6, 7, 9};
    std::vector<size_t> inactive_alpha = {0, 1, 4};
    auto orbitals = std::make_shared<Orbitals>(
        full_orbitals->get_coefficients().first,
        full_orbitals->get_energies().first,
        full_orbitals->get_overlap_matrix(), full_orbitals->get_basis_set(),
        std::make_tuple(active_alpha, inactive_alpha));

    // 3. Run Hamiltonian with Incore (Exact)
    auto ham_incore_factory = HamiltonianConstructorFactory::create("qdk");
    ham_incore_factory->settings().set("eri_method", "incore");
    auto ham_incore = ham_incore_factory->run(orbitals);

    // 4. Run Hamiltonian with Cholesky
    auto ham_chol_factory = HamiltonianConstructorFactory::create("qdk");
    ham_chol_factory->settings().set("eri_method", "cholesky");
    ham_chol_factory->settings().set("cholesky_tolerance", tolerance);
    auto ham_chol = ham_chol_factory->run(orbitals);

    // 5. Compare
    // One-body
    auto [aa_incore, bb_incore] = ham_incore->get_one_body_integrals();
    auto [aa_chol, bb_chol] = ham_chol->get_one_body_integrals();

    EXPECT_TRUE(aa_incore.isApprox(aa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(bb_incore.isApprox(bb_chol, testing::numerical_zero_tolerance));

    // Two-body
    auto [aaaa_incore, aabb_incore, bbbb_incore] =
        ham_incore->get_two_body_integrals();
    auto [aaaa_chol, aabb_chol, bbbb_chol] = ham_chol->get_two_body_integrals();

    EXPECT_TRUE(
        aaaa_incore.isApprox(aaaa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        aabb_incore.isApprox(aabb_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        bbbb_incore.isApprox(bbbb_chol, testing::numerical_zero_tolerance));

    // inactive fock matrix
    auto fock_incore = ham_incore->get_inactive_fock_matrix();
    auto fock_chol = ham_chol->get_inactive_fock_matrix();
    EXPECT_TRUE(fock_incore.first.isApprox(fock_chol.first,
                                           testing::numerical_zero_tolerance));
    EXPECT_TRUE(fock_incore.second.isApprox(fock_chol.second,
                                            testing::numerical_zero_tolerance));

    // core energy
    auto core_incore = ham_incore->get_core_energy();
    auto core_chol = ham_chol->get_core_energy();
    EXPECT_NEAR(core_incore, core_chol, testing::numerical_zero_tolerance);
  }
}

TEST_F(CholeskyTest, O2_Unrestricted_Comparison) {
  // 1. Setup O2 (Triplet)
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(0.0, 0.0, 3.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure structure(coordinates, symbols);
  auto structure_ptr = std::make_shared<Structure>(structure);

  // 2. Run SCF (UHF)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");
  auto [energy, wavefunction] =
      scf_factory->run(structure_ptr, 0, 3, "def2-svp");
  auto orbitals_scf = wavefunction->get_orbitals();

  // Create new Orbitals with active space
  auto coeffs = orbitals_scf->get_coefficients();
  auto energies = orbitals_scf->get_energies();
  auto overlap = orbitals_scf->get_overlap_matrix();
  auto basis = orbitals_scf->get_basis_set();

  // cholesky tolerance
  double tolerance = testing::integral_tolerance;

  // full space
  {
    auto orbitals = wavefunction->get_orbitals();

    // 3. Run Hamiltonian with Incore (Exact)
    auto ham_incore_factory = HamiltonianConstructorFactory::create("qdk");
    ham_incore_factory->settings().set("eri_method", "incore");
    auto ham_incore = ham_incore_factory->run(orbitals);

    // 4. Run Hamiltonian with Cholesky
    auto ham_chol_factory = HamiltonianConstructorFactory::create("qdk");
    ham_chol_factory->settings().set("eri_method", "cholesky");
    ham_chol_factory->settings().set("cholesky_tolerance", tolerance);
    auto ham_chol = ham_chol_factory->run(orbitals);

    // 5. Compare
    // One-body
    auto [aa_incore, bb_incore] = ham_incore->get_one_body_integrals();
    auto [aa_chol, bb_chol] = ham_chol->get_one_body_integrals();

    EXPECT_TRUE(aa_incore.isApprox(aa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(bb_incore.isApprox(bb_chol, testing::numerical_zero_tolerance));

    // Two-body
    auto [aaaa_incore, aabb_incore, bbbb_incore] =
        ham_incore->get_two_body_integrals();
    auto [aaaa_chol, aabb_chol, bbbb_chol] = ham_chol->get_two_body_integrals();

    EXPECT_TRUE(
        aaaa_incore.isApprox(aaaa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        aabb_incore.isApprox(aabb_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        bbbb_incore.isApprox(bbbb_chol, testing::numerical_zero_tolerance));
  }

  // continuous active space
  {
    auto full_orbitals = wavefunction->get_orbitals();
    // manual active space selection
    std::vector<size_t> active_alpha = {2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<size_t> inactive_alpha = {0, 1};
    std::vector<size_t> active_beta = {2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<size_t> inactive_beta = {0, 1};
    auto orbitals = std::make_shared<Orbitals>(
        full_orbitals->get_coefficients().first,
        full_orbitals->get_coefficients().second,
        full_orbitals->get_energies().first,
        full_orbitals->get_energies().second,
        full_orbitals->get_overlap_matrix(), full_orbitals->get_basis_set(),
        std::make_tuple(active_alpha, active_beta, inactive_alpha,
                        inactive_beta));

    // 3. Run Hamiltonian with Incore (Exact)
    auto ham_incore_factory = HamiltonianConstructorFactory::create("qdk");
    ham_incore_factory->settings().set("eri_method", "incore");
    auto ham_incore = ham_incore_factory->run(orbitals);

    // 4. Run Hamiltonian with Cholesky
    auto ham_chol_factory = HamiltonianConstructorFactory::create("qdk");
    ham_chol_factory->settings().set("eri_method", "cholesky");
    ham_chol_factory->settings().set("cholesky_tolerance", tolerance);
    auto ham_chol = ham_chol_factory->run(orbitals);

    // 5. Compare
    // One-body
    auto [aa_incore, bb_incore] = ham_incore->get_one_body_integrals();
    auto [aa_chol, bb_chol] = ham_chol->get_one_body_integrals();

    EXPECT_TRUE(aa_incore.isApprox(aa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(bb_incore.isApprox(bb_chol, testing::numerical_zero_tolerance));

    // Two-body
    auto [aaaa_incore, aabb_incore, bbbb_incore] =
        ham_incore->get_two_body_integrals();
    auto [aaaa_chol, aabb_chol, bbbb_chol] = ham_chol->get_two_body_integrals();

    EXPECT_TRUE(
        aaaa_incore.isApprox(aaaa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        aabb_incore.isApprox(aabb_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        bbbb_incore.isApprox(bbbb_chol, testing::numerical_zero_tolerance));

    // inactive fock matrix
    auto fock_incore = ham_incore->get_inactive_fock_matrix();
    auto fock_chol = ham_chol->get_inactive_fock_matrix();
    EXPECT_TRUE(fock_incore.first.isApprox(fock_chol.first,
                                           testing::numerical_zero_tolerance));
    EXPECT_TRUE(fock_incore.second.isApprox(fock_chol.second,
                                            testing::numerical_zero_tolerance));

    // core energy
    auto core_incore = ham_incore->get_core_energy();
    auto core_chol = ham_chol->get_core_energy();
    EXPECT_NEAR(core_incore, core_chol, testing::numerical_zero_tolerance);
  }

  // discontinuous active space
  {
    auto full_orbitals = wavefunction->get_orbitals();
    // manual active space selection
    std::vector<size_t> active_alpha = {2, 3, 5, 6, 7, 9};
    std::vector<size_t> inactive_alpha = {0, 1, 4};
    std::vector<size_t> active_beta = {2, 3, 5, 6, 7, 9};
    std::vector<size_t> inactive_beta = {0, 1, 4};
    auto orbitals = std::make_shared<Orbitals>(
        full_orbitals->get_coefficients().first,
        full_orbitals->get_coefficients().second,
        full_orbitals->get_energies().first,
        full_orbitals->get_energies().second,
        full_orbitals->get_overlap_matrix(), full_orbitals->get_basis_set(),
        std::make_tuple(active_alpha, active_beta, inactive_alpha,
                        inactive_beta));

    // 3. Run Hamiltonian with Incore (Exact)
    auto ham_incore_factory = HamiltonianConstructorFactory::create("qdk");
    ham_incore_factory->settings().set("eri_method", "incore");
    auto ham_incore = ham_incore_factory->run(orbitals);

    // 4. Run Hamiltonian with Cholesky
    auto ham_chol_factory = HamiltonianConstructorFactory::create("qdk");
    ham_chol_factory->settings().set("eri_method", "cholesky");
    ham_chol_factory->settings().set("cholesky_tolerance", tolerance);
    auto ham_chol = ham_chol_factory->run(orbitals);

    // 5. Compare
    // One-body
    auto [aa_incore, bb_incore] = ham_incore->get_one_body_integrals();
    auto [aa_chol, bb_chol] = ham_chol->get_one_body_integrals();

    EXPECT_TRUE(aa_incore.isApprox(aa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(bb_incore.isApprox(bb_chol, testing::numerical_zero_tolerance));

    // Two-body
    auto [aaaa_incore, aabb_incore, bbbb_incore] =
        ham_incore->get_two_body_integrals();
    auto [aaaa_chol, aabb_chol, bbbb_chol] = ham_chol->get_two_body_integrals();

    EXPECT_TRUE(
        aaaa_incore.isApprox(aaaa_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        aabb_incore.isApprox(aabb_chol, testing::numerical_zero_tolerance));
    EXPECT_TRUE(
        bbbb_incore.isApprox(bbbb_chol, testing::numerical_zero_tolerance));

    // inactive fock matrix
    auto fock_incore = ham_incore->get_inactive_fock_matrix();
    auto fock_chol = ham_chol->get_inactive_fock_matrix();
    EXPECT_TRUE(fock_incore.first.isApprox(fock_chol.first,
                                           testing::numerical_zero_tolerance));
    EXPECT_TRUE(fock_incore.second.isApprox(fock_chol.second,
                                            testing::numerical_zero_tolerance));

    // core energy
    auto core_incore = ham_incore->get_core_energy();
    auto core_chol = ham_chol->get_core_energy();
    EXPECT_NEAR(core_incore, core_chol, testing::numerical_zero_tolerance);
  }
}
