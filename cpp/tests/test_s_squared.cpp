// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

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
  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aaaa), std::make_optional(two_rdm_aabb),
      std::make_optional(two_rdm_bbbb)));

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

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aaaa), std::make_optional(two_rdm_aabb),
      std::make_optional(two_rdm_bbbb)));

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

// Stretched H2 broken-symmetry UHF: textbook spin-contamination example.
// Single determinant |ud> with α in orbital 0, β in orbital 1 (orthogonal
// spatial orbitals). This models the dissociation limit where α localizes on
// atom A and β on atom B. The state is a 50/50 singlet-triplet mixture, and its
// RDMs are generated automatically from the determinant.
// Expected <S²> = 1.0
TEST(SSquared, SpinContaminatedUHF_StretchedH2) {
  const int norbs = 2;
  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  auto wf = Wavefunction(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("ud"), orbitals));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 1.0, testing::numerical_zero_tolerance);
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

  EXPECT_THROW(wf.compute_s_squared(), std::runtime_error);
}
