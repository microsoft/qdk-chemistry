// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

static Wavefunction wavefunction_with_spin_rdms(
    const Eigen::VectorXd& coefficients,
    const Wavefunction::DeterminantVector& determinants,
    std::shared_ptr<Orbitals> orbitals, const Eigen::MatrixXd& one_rdm_aa,
    const Eigen::MatrixXd& one_rdm_bb, const Eigen::VectorXd& two_rdm_aaaa,
    const Eigen::VectorXd& two_rdm_aabb, const Eigen::VectorXd& two_rdm_bbbb) {
  using Sbt2 = SymmetryBlockedTensor<2, double>;
  using Sbt4 = SymmetryBlockedTensor<4, double>;
  auto one_rdm = std::make_shared<const SymmetryBlockedTensorVariant<2>>(
      std::in_place_type<Sbt2>,
      make_spin_diagonal_rank2_sbt(one_rdm_aa, one_rdm_bb, false));
  auto two_rdm = std::make_shared<const SymmetryBlockedTensorVariant<4>>(
      std::in_place_type<Sbt4>,
      make_spin_diagonal_rank4_sbt(two_rdm_aaaa, two_rdm_aabb, two_rdm_bbbb,
                                   false));
  return Wavefunction(std::make_unique<StateVectorContainer>(
      coefficients, determinants, std::move(orbitals), nullptr, nullptr,
      std::move(one_rdm), std::move(two_rdm)));
}

// Test compute_s_squared for a 2-electron, 2-orbital RHF singlet wavefunction.
//
// The state is the single closed-shell determinant |20> (orbital 0 doubly
// occupied). Its spin-dependent RDMs are generated automatically by the
// StateVectorContainer from the determinant occupations, so there is no need
// to hardcode them here.
//
// Expected <S^2> = 0 for a singlet.
TEST(SSquared, RHFSinglet) {
  const int norbs = 2;
  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("20"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for a single alpha electron (doublet, S=1/2).
//
// The single determinant |u> (one alpha electron in orbital 0) has its
// spin-dependent RDMs generated automatically from the occupations.
//
// Expected <S^2> = S(S+1) = 0.75
TEST(SSquared, SingleElectronDoublet) {
  const int norbs = 1;
  auto orbitals = testing::create_test_orbitals(2, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("u"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.75, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for a triplet state (S=1, M_S=0).
//
// This is a genuine two-determinant spin eigenstate, so its RDMs are hardcoded:
// StateVectorContainer only auto-generates RDMs for single-determinant
// expansions, and there is no CI-vector -> RDM path in the data layer without
// the full SCF/MACIS pipeline. The singlet/triplet distinction lives entirely
// in the sign of the mixed-spin (aabb) exchange terms below.
//
// 2 orbitals, 2 electrons. In QDK's blocked spin-orbital ordering (all alpha
// creators before all beta), the string determinant |du> canonicalizes with a
// sign: |du> = a†_{1α} a†_{0β}|vac> = -a†_{0β} a†_{1α}|vac>.
// The triplet M_S=0 (symmetric spin) state is, in raw operators,
//   |T,0> = (1/sqrt(2)) (a†_{0α} a†_{1β} + a†_{0β} a†_{1α}) |vac>
// which, in string determinants, becomes
//   |T,0> = (1/sqrt(2)) (|ud> - |du>).
//
// 1-RDMs: gamma^a_{00} = gamma^a_{11} = 0.5, gamma^b_{00} = gamma^b_{11} = 0.5
//
// Expected <S^2> = S(S+1) = 2.0
TEST(SSquared, TripletMSZero) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  // 1-RDMs for triplet M_S=0 (each orbital half-occupied in each spin)
  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 0.5;
  one_rdm_aa(1, 1) = 0.5;
  one_rdm_bb(0, 0) = 0.5;
  one_rdm_bb(1, 1) = 0.5;

  // 2-RDMs for the triplet M_S=0 state (QDK convention)
  // |T,0> = (1/sqrt(2)) (|0α,1β> - |1α,0β>)
  //
  // Gamma^{aabb}(p,q,r,s) = <a†_{pα} a†_{rβ} a_{sβ} a_{qα}>
  //
  // For the triplet M_S=0 state, same-spin 2-RDMs are zero (one electron per
  // spin).
  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  // Non-zero aabb elements for |T,0> = (1/sqrt(2))(|0α,1β> - |1α,0β>):
  //
  // Gamma^{aabb}(0,0,1,1) = <a†_{0α} a†_{1β} a_{1β} a_{0α}> =  0.5 (Coulomb)
  // Gamma^{aabb}(1,1,0,0) = <a†_{1α} a†_{0β} a_{0β} a_{1α}> =  0.5 (Coulomb)
  // Gamma^{aabb}(0,1,1,0) = <a†_{0α} a†_{1β} a_{0β} a_{1α}> = -0.5 (exchange)
  // Gamma^{aabb}(1,0,0,1) = <a†_{1α} a†_{0β} a_{1β} a_{0α}> = -0.5 (exchange)
  auto idx = [norbs](int p, int q, int r, int s) {
    return p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
  };

  two_rdm_aabb[idx(0, 0, 1, 1)] = 0.5;   // Coulomb
  two_rdm_aabb[idx(1, 1, 0, 0)] = 0.5;   // Coulomb
  two_rdm_aabb[idx(0, 1, 1, 0)] = -0.5;  // exchange
  two_rdm_aabb[idx(1, 0, 0, 1)] = -0.5;  // exchange

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(2);
  coeffs(0) = 1.0 / std::sqrt(2.0);
  coeffs(1) = -1.0 / std::sqrt(2.0);
  Wavefunction::DeterminantVector dets = {
      Configuration::from_spin_half_string("ud"),
      Configuration::from_spin_half_string("du")};

  // Note: We override the RDMs manually here to match the triplet state.
  // StateVectorContainer only auto-generates RDMs for single-determinant
  // expansions, so it won't recompute them from CI coefficients.
  auto wf = wavefunction_with_spin_rdms(coeffs, dets, orbitals, one_rdm_aa,
                                        one_rdm_bb, two_rdm_aaaa, two_rdm_aabb,
                                        two_rdm_bbbb);

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 2.0, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for triplet M_S=+1: two alpha electrons in 2 orbitals.
//
// The single determinant |uu> (two parallel alpha spins) is the M_S=+1
// component of a triplet; its RDMs are generated automatically.
//
// Expected <S^2> = S(S+1) = 2.0
TEST(SSquared, TripletMSPlusOne) {
  const int norbs = 2;
  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("uu"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 2.0, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for a 3-electron quartet (S=3/2, M_S=+3/2).
//
// The single determinant |uuu> (three parallel alpha spins) is the M_S=+3/2
// component of a quartet; its RDMs are generated automatically.
//
// Expected <S^2> = S(S+1) = 3.75
TEST(SSquared, QuartetMSPlusThreeHalf) {
  const int norbs = 3;
  auto orbitals = testing::create_test_orbitals(6, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("uuu"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 3.75, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for a 4-electron singlet: two doubly-occupied orbitals
// in a 3-orbital active space (orbital 2 unoccupied).
//
// The single closed-shell determinant |220> has its RDMs generated
// automatically.
//
// Expected <S^2> = 0.0
TEST(SSquared, FourElectronSinglet) {
  const int norbs = 3;
  auto orbitals = testing::create_test_orbitals(6, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("220"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for a single beta electron (doublet, S=1/2).
//
// The single determinant |d> (one beta electron) has its RDMs generated
// automatically. Same <S^2> as the single alpha electron by symmetry.
//
// Expected <S^2> = 0.75
TEST(SSquared, SingleBetaElectronDoublet) {
  const int norbs = 1;
  auto orbitals = testing::create_test_orbitals(2, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("d"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.75, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for an open-shell singlet (S=0, M_S=0).
//
// Like TripletMSZero, this is a genuine two-determinant spin eigenstate whose
// RDMs must be hardcoded (no auto-generation for multi-determinant expansions).
//
// 2 orbitals, 2 electrons. The open-shell singlet is:
//   |S,0> = (1/sqrt(2)) (|0α,1β> + |1α,0β>)
//
// 1-RDMs are the same as the triplet M_S=0: gamma^a = gamma^b = 0.5 * I
// But the 2-RDM cross terms have opposite sign.
//
// Expected <S^2> = 0.0
TEST(SSquared, OpenShellSinglet) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 0.5;
  one_rdm_aa(1, 1) = 0.5;
  one_rdm_bb(0, 0) = 0.5;
  one_rdm_bb(1, 1) = 0.5;

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  auto idx = [norbs](int p, int q, int r, int s) {
    return p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
  };

  // Coulomb terms are the same as the triplet
  two_rdm_aabb[idx(0, 0, 1, 1)] = 0.5;
  two_rdm_aabb[idx(1, 1, 0, 0)] = 0.5;
  // Exchange terms have OPPOSITE sign compared to triplet (+0.5 instead of
  // -0.5)
  two_rdm_aabb[idx(0, 1, 1, 0)] = 0.5;
  two_rdm_aabb[idx(1, 0, 0, 1)] = 0.5;

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(2);
  coeffs(0) = 1.0 / std::sqrt(2.0);
  coeffs(1) = 1.0 / std::sqrt(2.0);
  Wavefunction::DeterminantVector dets = {
      Configuration::from_spin_half_string("ud"),
      Configuration::from_spin_half_string("du")};

  auto wf = wavefunction_with_spin_rdms(coeffs, dets, orbitals, one_rdm_aa,
                                        one_rdm_bb, two_rdm_aaaa, two_rdm_aabb,
                                        two_rdm_bbbb);

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
}

// Test compute_s_squared for the vacuum (0 electrons).
//
// The single empty determinant |00> has zero RDMs (generated automatically).
//
// Expected <S^2> = 0.
TEST(SSquared, Vacuum) {
  const int norbs = 2;
  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("00"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
}

// Broken-symmetry determinant in a common spatial-orbital basis. The state
// |ud> is a 50/50 singlet-triplet mixture, and its RDMs are generated
// automatically from the determinant.
// Expected <S²> = 1.0
TEST(SSquared, SpinContaminatedCommonBasis) {
  const int norbs = 2;
  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("ud"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 1.0, testing::numerical_zero_tolerance);
}

TEST(SSquared, ThrowsForUnrestrictedOrbitals) {
  auto orbitals = testing::create_test_orbitals(2, 1, false);
  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("2"), orbitals));

  ASSERT_TRUE(orbitals->is_unrestricted());
  ASSERT_TRUE(wf.has_one_rdm_spin_dependent());
  ASSERT_TRUE(wf.has_two_rdm_spin_dependent());
  EXPECT_THROW(wf.compute_s_squared(), std::runtime_error);
}

TEST(SSquared, ThrowsForMismatchedRdmExtents) {
  auto orbitals = testing::create_test_orbitals(2, 1, true);
  Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(1);
  Wavefunction::DeterminantVector determinants = {
      Configuration::from_spin_half_string("2")};
  Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Ones(1, 1);
  Eigen::VectorXd two_rdm = Eigen::VectorXd::Zero(16);

  auto wf =
      wavefunction_with_spin_rdms(coefficients, determinants, orbitals, one_rdm,
                                  one_rdm, two_rdm, two_rdm, two_rdm);

  EXPECT_THROW(wf.compute_s_squared(), std::runtime_error);
}

// Test that compute_s_squared throws when RDMs are missing.
// Use a multi-determinant wavefunction (StateVectorContainer auto-generates
// RDMs only for single-determinant expansions).
TEST(SSquared, ThrowsWithoutRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 2, true);
  Eigen::VectorXd coeffs(2);
  coeffs(0) = 1.0 / std::sqrt(2.0);
  coeffs(1) = 1.0 / std::sqrt(2.0);
  Wavefunction::DeterminantVector dets = {
      Configuration::from_spin_half_string("20"),
      Configuration::from_spin_half_string("02")};

  auto wf = Wavefunction(
      std::make_unique<StateVectorContainer>(coeffs, dets, orbitals));

  try {
    (void)wf.compute_s_squared();
    FAIL() << "Expected missing RDMs to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("Cannot compute <S^2>"),
              std::string::npos)
        << error.what();
  }
}

// ---------------------------------------------------------------------------
// SCF + MACIS integration tests
// ---------------------------------------------------------------------------

static std::shared_ptr<Wavefunction> run_cas_with_rdms(
    std::shared_ptr<Structure> structure, int charge, int multiplicity,
    const std::string& basis, int nalpha, int nbeta,
    std::shared_ptr<Orbitals> custom_orbitals = nullptr) {
  std::shared_ptr<Orbitals> orbitals;
  if (custom_orbitals) {
    orbitals = custom_orbitals;
  } else {
    auto scf_solver = ScfSolverFactory::create();
    scf_solver->settings().set("scf_type", std::string("restricted"));
    scf_solver->settings().set("method", "hf");
    scf_solver->settings().set("enable_gdm", false);
    auto [energy, wavefunction] =
        scf_solver->run(structure, charge, multiplicity, basis);
    orbitals = wavefunction->get_orbitals();
  }

  auto hamiltonian = HamiltonianConstructorFactory::create()->run(orbitals);
  auto calculator = MultiConfigurationCalculatorFactory::create("macis_cas");
  calculator->settings().set("calculate_one_rdm", true);
  calculator->settings().set("calculate_two_rdm", true);
  return calculator->run(hamiltonian, nalpha, nbeta).second;
}

static std::shared_ptr<Wavefunction> run_sci_with_rdms(
    std::shared_ptr<Structure> structure, int charge, int multiplicity,
    const std::string& basis, int nalpha, int nbeta, int ntdets_max) {
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("restricted"));
  scf_solver->settings().set("method", "hf");
  scf_solver->settings().set("enable_gdm", false);
  auto [energy, wavefunction] =
      scf_solver->run(structure, charge, multiplicity, basis);
  auto hamiltonian = HamiltonianConstructorFactory::create()->run(
      wavefunction->get_orbitals());

  auto calculator = MultiConfigurationCalculatorFactory::create("macis_asci");
  calculator->settings().set("calculate_one_rdm", true);
  calculator->settings().set("calculate_two_rdm", true);
  calculator->settings().set("ntdets_max", ntdets_max);
  calculator->settings().set("max_refine_iter", 0);
  calculator->settings().set("grow_factor", 2);
  calculator->settings().set("core_selection_strategy", std::string("fixed"));
  return calculator->run(hamiltonian, nalpha, nbeta).second;
}

TEST(SSquaredCAS, H_Doublet) {
  auto wavefunction = run_cas_with_rdms(testing::create_hydrogen_structure(), 0,
                                        2, "sto-3g", 1, 0);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

TEST(SSquaredCAS, H2Plus_Doublet) {
  std::vector<Eigen::Vector3d> coordinates = {{0., 0., 0.}, {0., 0., 1.4}};
  auto structure = std::make_shared<Structure>(
      coordinates, std::vector<std::string>{"H", "H"});
  auto wavefunction = run_cas_with_rdms(structure, 1, 2, "sto-3g", 1, 0);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

TEST(SSquaredCAS, H2_Singlet) {
  std::vector<Eigen::Vector3d> coordinates = {{0., 0., 0.}, {0., 0., 1.4}};
  auto structure = std::make_shared<Structure>(
      coordinates, std::vector<std::string>{"H", "H"});
  auto wavefunction = run_cas_with_rdms(structure, 0, 1, "sto-3g", 1, 1);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

TEST(SSquaredCAS, Water_Singlet) {
  auto wavefunction = run_cas_with_rdms(testing::create_water_structure(), 0, 1,
                                        "sto-3g", 5, 5);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

TEST(SSquaredCAS, Li_Doublet) {
  auto wavefunction =
      run_cas_with_rdms(testing::create_li_structure(), 0, 2, "sto-3g", 2, 1);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

TEST(SSquaredCAS, O2_Triplet) {
  auto structure = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("restricted"));
  scf_solver->settings().set("method", "hf");
  scf_solver->settings().set("enable_gdm", false);
  auto [energy, wavefunction] = scf_solver->run(structure, 0, 3, "sto-3g");
  auto orbitals = testing::with_active_space(
      wavefunction->get_orbitals(), std::vector<size_t>{6, 7, 8, 9},
      std::vector<size_t>{0, 1, 2, 3, 4, 5});
  auto cas_wavefunction =
      run_cas_with_rdms(structure, 0, 3, "sto-3g", 3, 1, orbitals);
  EXPECT_NEAR(cas_wavefunction->compute_s_squared(), 2.0,
              testing::rdm_tolerance);
}

TEST(SSquaredCAS, N_Quartet) {
  auto structure = testing::create_nitrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("restricted"));
  scf_solver->settings().set("method", "hf");
  scf_solver->settings().set("enable_gdm", false);
  auto [energy, wavefunction] = scf_solver->run(structure, 0, 4, "sto-3g");
  ASSERT_TRUE(wavefunction->get_orbitals()->is_restricted());
  auto orbitals = testing::with_active_space(wavefunction->get_orbitals(),
                                             std::vector<size_t>{1, 2, 3, 4},
                                             std::vector<size_t>{0});
  ASSERT_TRUE(orbitals->is_restricted());
  auto cas_wavefunction =
      run_cas_with_rdms(structure, 0, 4, "sto-3g", 4, 1, orbitals);
  EXPECT_NEAR(cas_wavefunction->compute_s_squared(), 3.75,
              testing::rdm_tolerance);
}

TEST(SSquaredCAS, StretchedH2_Singlet) {
  std::vector<Eigen::Vector3d> coordinates = {{0., 0., 0.}, {0., 0., 5.0}};
  auto structure = std::make_shared<Structure>(
      coordinates, std::vector<std::string>{"H", "H"});
  auto wavefunction = run_cas_with_rdms(structure, 0, 1, "sto-3g", 1, 1);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

TEST(SSquaredCAS, StretchedH2_Triplet) {
  std::vector<Eigen::Vector3d> coordinates = {{0., 0., 0.}, {0., 0., 5.0}};
  auto structure = std::make_shared<Structure>(
      coordinates, std::vector<std::string>{"H", "H"});
  auto wavefunction = run_cas_with_rdms(structure, 0, 3, "sto-3g", 2, 0);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 2.0, testing::rdm_tolerance);
}

TEST(SSquaredSCI, TruncatedH3Quartet) {
  std::vector<Eigen::Vector3d> coordinates = {
      {0., 0., -1.4}, {0., 0., 0.}, {0., 0., 1.4}};
  auto structure = std::make_shared<Structure>(
      coordinates, std::vector<std::string>{"H", "H", "H"});
  auto wavefunction = run_sci_with_rdms(structure, 0, 4, "6-31g", 3, 0, 5);

  // Three alpha electrons force S=3/2 for every determinant. The 6-orbital
  // fixed-M_S space has C(6,3)=20 determinants, so this remains spin-pure
  // while proving that compute_s_squared works on a genuinely selected space.
  EXPECT_GT(wavefunction->size(), 1);
  EXPECT_LT(wavefunction->size(), 20);
  EXPECT_NEAR(wavefunction->compute_s_squared(), 3.75, testing::rdm_tolerance);
}
