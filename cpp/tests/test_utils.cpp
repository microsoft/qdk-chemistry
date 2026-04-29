// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <qdk/chemistry/utils/orbital_rotation.hpp>
#include <qdk/chemistry/utils/valence_space.hpp>
#include <stdexcept>
#include <vector>

#include "../src/qdk/chemistry/algorithms/microsoft/utils.hpp"
#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::utils;

class ValenceActiveParametersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto scf_solver = ScfSolverFactory::create();

    std::vector<std::string> symbols = {"O", "H", "H"};
    Eigen::MatrixXd coords(3, 3);
    coords << 0.000000, 0.000000, 0.000000,  // O at origin
        0.757000, 0.586000, 0.000000,        // H1
        -0.757000, 0.586000, 0.000000;       // H2
    std::shared_ptr<Structure> water_structure =
        std::make_shared<Structure>(coords, symbols);
    auto [water_e, water_wf] = scf_solver->run(water_structure, 0, 1, "STO-3G");
    std::shared_ptr<Orbitals> water_orbitals = water_wf->get_orbitals();
    std::shared_ptr<BasisSet> basis_set = water_orbitals->get_basis_set();

    water_wavefunction = water_wf;

    Configuration config_truncated(
        "22222");  // the orbitals of this config needs to be built specifically
    Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(
        water_orbitals->get_num_molecular_orbitals(), 5);
    Orbitals water_orbitals_truncated(coeffs, std::nullopt, std::nullopt,
                                      basis_set, std::nullopt);
    std::shared_ptr<Orbitals> water_orbitals_truncated_ptr =
        std::make_shared<Orbitals>(water_orbitals_truncated);
    auto wfn_container_truncated = std::make_unique<SlaterDeterminantContainer>(
        config_truncated, water_orbitals_truncated_ptr);
    water_wavefunction_truncated =
        std::make_shared<Wavefunction>(std::move(wfn_container_truncated));

    std::vector<std::string> he_symbols = {"He"};
    Eigen::MatrixXd he_coords(1, 3);
    he_coords << 0.000000, 0.000000, 0.000000;  // He at origin
    std::shared_ptr<Structure> he_structure =
        std::make_shared<Structure>(he_coords, he_symbols);
    auto [he_e, he_wf] = scf_solver->run(he_structure, 0, 1, "STO-3G");
    he_wavefunction = he_wf;

    std::vector<std::string> symbols_oh = {"O", "H"};
    Eigen::MatrixXd coords_oh(2, 3);
    coords_oh << 0.000000, 0.000000, 0.000000,  // O at origin
        0.757000, 0.586000, 0.000000;           // H1
    std::shared_ptr<Structure> oh_structure =
        std::make_shared<Structure>(coords_oh, symbols_oh);

    auto [oh_e, oh_wf] =
        scf_solver->run(oh_structure, 0, 2, "STO-3G");  // doublet
    std::shared_ptr<Orbitals> base_orbitals_oh = oh_wf->get_orbitals();
    oh_wavefunction = oh_wf;

    // Create configuration for 8 electrons (4 doubly occupied orbitals)
    Configuration config_ohp("222200000");  // 4 doubly occupied orbitals
    auto ohp_wfn_container = std::make_unique<SlaterDeterminantContainer>(
        config_ohp, base_orbitals_oh);
    ohp_wavefunction =
        std::make_shared<Wavefunction>(std::move(ohp_wfn_container));

    // Create configuration for 10 electrons (5 doubly occupied orbitals)
    Configuration config_ohn("222220000");  // 4 doubly occupied orbitals
    auto ohn_wfn_container = std::make_unique<SlaterDeterminantContainer>(
        config_ohn, base_orbitals_oh);
    ohn_wavefunction =
        std::make_shared<Wavefunction>(std::move(ohn_wfn_container));
  }

  std::shared_ptr<Wavefunction> water_wavefunction;
  std::shared_ptr<Wavefunction> water_wavefunction_truncated;
  std::shared_ptr<Wavefunction> he_wavefunction;
  std::shared_ptr<Wavefunction> oh_wavefunction;
  std::shared_ptr<Wavefunction> ohp_wavefunction;
  std::shared_ptr<Wavefunction> ohn_wavefunction;
};

// Test basic functionality with water molecule
TEST_F(ValenceActiveParametersTest, WaterMoleculeBasicTest) {
  auto result = compute_valence_space_parameters(water_wavefunction, 0);

  // For water (O + 2H):
  // O has 6 valence electrons, 4 valence orbitals (2s, 3*2p)
  // Each H has 1 valence electron, 1 valence orbital (1s)
  // Total expected: 6 + 1 + 1 = 8 valence electrons
  //                 4 + 1 + 1 = 6 valence orbitals

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 8);
  EXPECT_EQ(num_active_orbitals, 6);
}

// Test basic functionality with truncated water molecule wavefunction
TEST_F(ValenceActiveParametersTest, WaterMoleculeTruncatedTest) {
  auto result =
      compute_valence_space_parameters(water_wavefunction_truncated, 0);

  // For water (O + 2H):
  // O has 6 valence electrons, 4 valence orbitals (2s, 3*2p)
  // Each H has 1 valence electron, 1 valence orbital (1s)
  // Total expected: (5 - (5 - 5)) * 2 + 0 = 10 valence electrons
  //                 6 > (5 - 1), so 4 valence orbitals

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 8);
  EXPECT_EQ(num_active_orbitals, 4);
}

// Test basic functionality with single Helium atom
TEST_F(ValenceActiveParametersTest, HeliumAtomBasicTest) {
  auto result = compute_valence_space_parameters(he_wavefunction, 0);

  // For Helium atom:
  // - num_active_orbitals: 1 (both lower bound and upper bound is 1)
  // - num_active_electrons: max(2, 2*min(1,1)) = max(2, 2) = 2

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 2);
  EXPECT_EQ(num_active_orbitals, 1);  // Lower bound applied
}

// Test basic functionality with single Oxygen-Hydrogen molecule
TEST_F(ValenceActiveParametersTest, OxygenHydrogenMoleculeBasicTest) {
  auto result = compute_valence_space_parameters(oh_wavefunction, 0);

  // For Oxygen-Hydrogen molecule:
  // - num_active_orbitals: 5 (4 from O, 1 from H)
  // - num_active_electrons: 7

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 7);
  EXPECT_EQ(num_active_orbitals, 5);
}

// Test basic functionality with positive charge Oxygen-Hydrogen molecule
TEST_F(ValenceActiveParametersTest, OxygenHydrogenMoleculePositiveChargeTest) {
  auto result = compute_valence_space_parameters(ohp_wavefunction, 1);

  // For Oxygen-Hydrogen molecule:
  // - num_active_orbitals: 5 (4 from O, 1 from H)
  // - num_active_electrons: 6

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 6);
  EXPECT_EQ(num_active_orbitals, 5);
}

// Test basic functionality with negative charge Oxygen-Hydrogen molecule
TEST_F(ValenceActiveParametersTest, OxygenHydrogenMoleculeNegativeChargeTest) {
  auto result = compute_valence_space_parameters(ohn_wavefunction, -1);

  // For Oxygen-Hydrogen molecule:
  // - num_active_orbitals: 5 (4 from O, 1 from H)
  // - num_active_electrons: 8

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 8);
  EXPECT_EQ(num_active_orbitals, 5);
}

// ========== Transition metal double-d-shell tests ==========
// Periods 4-5 include a correlating d' shell when ``include_double_d_shell``
// is enabled: 14 valence orbitals per atom
// (ns + 5*(n-1)d + 5*nd' + 3*np) instead of the default 9.
//
// These tests only validate the valence-space sizing logic in
// ``compute_valence_space_parameters``; that function only consults the
// structure, the total electron count, and ``num_molecular_orbitals``. We
// therefore build a minimal Wavefunction directly (single dummy s-shell per
// atom, identity orbital coefficients, Configuration string sized to the
// requested electron count) and skip SCF entirely. This keeps the suite fast
// and removes any dependence on a particular basis set being available.

namespace {

// Build a minimal valid Wavefunction without running SCF. Each atom gets a
// single dummy s-shell so the BasisSet validates; the orbital coefficient
// matrix is sized to ``num_molecular_orbitals`` and a Configuration string is
// constructed to yield the requested ``(n_alpha, n_beta)`` electron counts.
std::shared_ptr<Wavefunction> make_minimal_wavefunction(
    const std::vector<std::string>& symbols, const Eigen::MatrixXd& coords,
    size_t n_alpha, size_t n_beta, size_t num_molecular_orbitals) {
  auto structure = std::make_shared<Structure>(coords, symbols);

  std::vector<Shell> shells;
  Eigen::VectorXd exps(1);
  exps << 1.0;
  Eigen::VectorXd coefs(1);
  coefs << 1.0;
  for (size_t i = 0; i < symbols.size(); ++i) {
    shells.emplace_back(i, OrbitalType::S, exps, coefs);
  }
  auto basis_set = std::make_shared<BasisSet>("dummy", shells, structure);

  Eigen::MatrixXd coeffs =
      Eigen::MatrixXd::Identity(num_molecular_orbitals, num_molecular_orbitals);
  auto orbitals =
      std::make_shared<Orbitals>(coeffs, std::nullopt, std::nullopt, basis_set);

  // Build a Configuration that yields exactly (n_alpha, n_beta) electrons.
  std::string config_str(num_molecular_orbitals, '0');
  size_t pair_count = std::min(n_alpha, n_beta);
  for (size_t i = 0; i < pair_count; ++i) config_str[i] = '2';
  if (n_alpha > n_beta) {
    for (size_t i = 0; i < n_alpha - n_beta; ++i)
      config_str[pair_count + i] = 'u';
  } else if (n_beta > n_alpha) {
    for (size_t i = 0; i < n_beta - n_alpha; ++i)
      config_str[pair_count + i] = 'd';
  }
  Configuration config(config_str);
  auto container =
      std::make_unique<SlaterDeterminantContainer>(config, orbitals);
  return std::make_shared<Wavefunction>(std::move(container));
}

// Convenience overload for a single atom at the origin.
std::shared_ptr<Wavefunction> make_minimal_wavefunction(
    const std::string& symbol, size_t n_alpha, size_t n_beta,
    size_t num_molecular_orbitals) {
  Eigen::MatrixXd coords(1, 3);
  coords << 0.0, 0.0, 0.0;
  return make_minimal_wavefunction({symbol}, coords, n_alpha, n_beta,
                                   num_molecular_orbitals);
}

}  // namespace

// Cu atom (Z=29, period 4): 11 valence electrons, 14 valence orbitals.
TEST(TransitionMetalValenceTest, CopperAtomTest) {
  // 29 electrons: 14 alpha + 15 beta (or 15+14); doublet. Use 50 MOs (>>
  // num_core_mos + 14 = 9 + 14 = 23) so the cap doesn't bind.
  auto wavefunction = make_minimal_wavefunction("Cu", 15, 14, 50);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  EXPECT_EQ(result.first, 11);   // 29 - 18 = 11 valence electrons
  EXPECT_EQ(result.second, 14);  // 4s + 5*3d + 5*4d' + 3*4p
}

// Ni atom (Z=28, period 4): 10 valence electrons, 14 valence orbitals.
TEST(TransitionMetalValenceTest, NickelAtomTest) {
  // 28 electrons triplet: 15 alpha + 13 beta.
  auto wavefunction = make_minimal_wavefunction("Ni", 15, 13, 50);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  EXPECT_EQ(result.first, 10);   // 28 - 18
  EXPECT_EQ(result.second, 14);  // full double-d-shell
}

// Zn atom (Z=30, period 4, d10 closed-shell singlet).
TEST(TransitionMetalValenceTest, ZincAtomTest) {
  auto wavefunction = make_minimal_wavefunction("Zn", 15, 15, 50);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  EXPECT_EQ(result.first, 12);   // 30 - 18
  EXPECT_EQ(result.second, 14);  // full double-d-shell
}

// AgH molecule: period 5 transition metal.
TEST(TransitionMetalValenceTest, SilverHydrideTest) {
  // Ag (Z=47) + H (Z=1) = 48 electrons, singlet.
  std::vector<std::string> symbols = {"Ag", "H"};
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0, 0.0, 0.0, 1.617;
  auto wavefunction = make_minimal_wavefunction(symbols, coords, 24, 24, 60);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  // Ag (Z=47): 11 valence electrons (47 - 36 core from Kr)
  // Period 5: 14 valence orbitals (5s + 5*4d + 5*5d' + 3*5p)
  // H: 1 valence electron, 1 valence orbital
  EXPECT_EQ(result.first, 12);   // 11 + 1
  EXPECT_EQ(result.second, 15);  // 14 + 1
}

// Period 3 elements are NOT affected by the double-d-shell.
TEST(TransitionMetalValenceTest, Period3UnaffectedTest) {
  std::vector<std::string> symbols = {"Na", "Cl"};
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0, 0.0, 0.0, 2.361;
  // Na (11) + Cl (17) = 28 electrons, singlet.
  auto wavefunction = make_minimal_wavefunction(symbols, coords, 14, 14, 30);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  // Na: 1 valence electron, 4 valence orbitals (3s + 3*3p)
  // Cl: 7 valence electrons, 4 valence orbitals (3s + 3*3p)
  EXPECT_EQ(result.first, 8);
  EXPECT_EQ(result.second, 8);
}

// K atom (Z=19, period 4 main-group): 9 valence orbitals (no d' shell).
TEST(TransitionMetalValenceTest, PotassiumAtomTest) {
  // 19 electrons doublet: 10 alpha + 9 beta.
  auto wavefunction = make_minimal_wavefunction("K", 10, 9, 30);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  EXPECT_EQ(result.first, 1);   // 19 - 18
  EXPECT_EQ(result.second, 9);  // period 4 main-group: no d' shell
}

// Pt atom (Z=78, period 6 d-block): 24 valence electrons, 21 valence orbitals.
// Period 6 base (16) + 5 d' = 21.
TEST(TransitionMetalValenceTest, PlatinumAtomTest) {
  // 78 electrons singlet.
  auto wavefunction = make_minimal_wavefunction("Pt", 39, 39, 100);
  auto result = compute_valence_space_parameters(
      wavefunction, 0, /*include_double_d_shell=*/true);

  EXPECT_EQ(result.first, 24);   // 78 - 54 (Xe core)
  EXPECT_EQ(result.second, 21);  // 6s + 7*4f + 5*5d + 5*6d' + 3*6p
}

// Default behavior (include_double_d_shell=false) for d-block elements: the
// d' correlating shell is NOT added.
TEST(TransitionMetalValenceTest, CopperAtomDefaultNoDoubleDShellTest) {
  auto wavefunction = make_minimal_wavefunction("Cu", 15, 14, 50);
  // Default: include_double_d_shell=false.
  auto result = compute_valence_space_parameters(wavefunction, 0);

  EXPECT_EQ(result.first, 11);  // 29 - 18
  EXPECT_EQ(result.second, 9);  // 4s + 5*3d + 3*4p (no d' shell)
}

TEST(TransitionMetalValenceTest, PlatinumAtomDefaultNoDoubleDShellTest) {
  auto wavefunction = make_minimal_wavefunction("Pt", 39, 39, 100);
  // Default: include_double_d_shell=false.
  auto result = compute_valence_space_parameters(wavefunction, 0);

  EXPECT_EQ(result.first, 24);   // 78 - 54
  EXPECT_EQ(result.second, 16);  // 6s + 7*4f + 5*5d + 3*6p (no d' shell)
}

// Test fixture for orbital rotation
class OrbitalRotationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use a real molecular system - Water
    auto water = testing::create_water_structure();
    auto scf_solver = ScfSolverFactory::create();

    auto [water_e, water_wf] = scf_solver->run(water, 0, 1, "STO-3G");

    test_orbitals_restricted = water_wf->get_orbitals();

    // Get electron counts from wavefunction
    std::tie(num_alpha_occupied_orbitals, num_beta_occupied_orbitals) =
        water_wf->get_total_num_electrons();
    num_molecular_orbitals =
        test_orbitals_restricted->get_num_molecular_orbitals();
    num_atomic_orbitals = test_orbitals_restricted->get_num_atomic_orbitals();
  }

  size_t num_atomic_orbitals;
  size_t num_molecular_orbitals;
  size_t num_alpha_occupied_orbitals;
  size_t num_beta_occupied_orbitals;
  std::shared_ptr<Orbitals> test_orbitals_restricted;
};

// Test basic orbital rotation functionality
TEST_F(OrbitalRotationTest, BasicRotationTest) {
  using namespace qdk::chemistry::utils;

  // Create a small rotation vector
  // For n_occ occupied and n_vir virtual orbitals: rotation size = n_occ *
  // n_vir
  size_t num_virtual_orbitals =
      num_molecular_orbitals - num_alpha_occupied_orbitals;
  size_t rotation_size = num_alpha_occupied_orbitals * num_virtual_orbitals;
  Eigen::VectorXd rotation_vector =
      Eigen::VectorXd::Constant(rotation_size, 0.01);

  auto rotated_orbitals =
      rotate_orbitals(test_orbitals_restricted, rotation_vector,
                      num_alpha_occupied_orbitals, num_beta_occupied_orbitals);

  // Check that we got a valid Orbitals object
  ASSERT_NE(rotated_orbitals, nullptr);
  EXPECT_EQ(rotated_orbitals->get_num_molecular_orbitals(),
            num_molecular_orbitals);
  EXPECT_EQ(rotated_orbitals->get_num_atomic_orbitals(), num_atomic_orbitals);
  EXPECT_TRUE(rotated_orbitals->is_restricted());

  // Check that energies are invalidated (should be nullopt)
  EXPECT_FALSE(rotated_orbitals->has_energies());
}

// Test that identity rotation returns the same orbitals
TEST_F(OrbitalRotationTest, IdentityRotationTest) {
  using namespace qdk::chemistry::utils;

  // Zero rotation should leave orbitals essentially unchanged
  size_t num_virtual_orbitals =
      num_molecular_orbitals - num_alpha_occupied_orbitals;
  size_t rotation_size = num_alpha_occupied_orbitals * num_virtual_orbitals;
  Eigen::VectorXd rotation_vector = Eigen::VectorXd::Zero(rotation_size);

  auto rotated_orbitals =
      rotate_orbitals(test_orbitals_restricted, rotation_vector,
                      num_alpha_occupied_orbitals, num_beta_occupied_orbitals);

  const auto& original_coeffs =
      test_orbitals_restricted->get_coefficients_alpha();
  const auto& rotated_coeffs = rotated_orbitals->get_coefficients_alpha();

  // With zero rotation, coefficients should be the same
  EXPECT_TRUE(original_coeffs.isApprox(rotated_coeffs,
                                       testing::numerical_zero_tolerance));
}

// Test that the rotation produces unitary transformation
TEST_F(OrbitalRotationTest, UnitaryRotationTest) {
  using namespace qdk::chemistry::utils;

  // Create a small rotation vector
  size_t num_virtual_orbitals =
      num_molecular_orbitals - num_alpha_occupied_orbitals;
  size_t rotation_size = num_alpha_occupied_orbitals * num_virtual_orbitals;
  Eigen::VectorXd rotation_vector = Eigen::VectorXd::Constant(
      rotation_size, 0.1);  // Larger rotation for clear test

  auto rotated_orbitals =
      rotate_orbitals(test_orbitals_restricted, rotation_vector,
                      num_alpha_occupied_orbitals, num_beta_occupied_orbitals);

  const auto& S = test_orbitals_restricted->get_overlap_matrix();
  const auto& original_coeffs =
      test_orbitals_restricted->get_coefficients_alpha();
  const auto& rotated_coeffs = rotated_orbitals->get_coefficients_alpha();

  // Compute the transformation matrix U = C_original^T * S * C_rotated
  Eigen::MatrixXd U = original_coeffs.transpose() * S * rotated_coeffs;

  // Check that the transformation matrix is unitary (U^T * U = I)
  Eigen::MatrixXd should_be_identity = U.transpose() * U;
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(U.cols(), U.cols());

  EXPECT_NEAR(0.0, (should_be_identity - identity).norm(),
              testing::numerical_zero_tolerance);
}

// Test fixture for mathematical utility functions
class MathUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// ========== Tests for factorial function ==========

TEST_F(MathUtilsTest, FactorialEdgeCases) {
  EXPECT_EQ(qdk::chemistry::utils::microsoft::factorial(0),
            1);  // 0! = 1 by definition
  EXPECT_EQ(qdk::chemistry::utils::microsoft::factorial(1), 1);  // 1! = 1
}

TEST_F(MathUtilsTest, FactorialValues) {
  using qdk::chemistry::utils::microsoft::factorial;

  // Small values
  EXPECT_EQ(factorial(2), 2);
  EXPECT_EQ(factorial(3), 6);
  EXPECT_EQ(factorial(4), 24);
  EXPECT_EQ(factorial(5), 120);
  EXPECT_EQ(factorial(6), 720);
  EXPECT_EQ(factorial(7), 5040);

  // Medium values
  EXPECT_EQ(factorial(10), 3628800);
  EXPECT_EQ(factorial(12), 479001600);
  EXPECT_EQ(factorial(15), 1307674368000);

  // Large values - 20! is the maximum safe value for 64-bit size_t
  EXPECT_EQ(factorial(20), 2432902008176640000ULL);
}

TEST_F(MathUtilsTest, FactorialOverflow) {
  using qdk::chemistry::utils::microsoft::factorial;

  // Values > 20 should throw overflow_error
  EXPECT_THROW(factorial(21), std::overflow_error);
  EXPECT_THROW(factorial(25), std::overflow_error);
  EXPECT_THROW(factorial(100), std::overflow_error);
}

// ========== Tests for binomial_coefficient function ==========

TEST_F(MathUtilsTest, BinomialCoefficientEdgeCases) {
  using qdk::chemistry::utils::microsoft::binomial_coefficient;

  // C(n, 0) = 1 for any n
  EXPECT_EQ(binomial_coefficient(0, 0), 1);
  EXPECT_EQ(binomial_coefficient(5, 0), 1);
  EXPECT_EQ(binomial_coefficient(10, 0), 1);
  EXPECT_EQ(binomial_coefficient(100, 0), 1);

  // C(n, n) = 1 for any n
  EXPECT_EQ(binomial_coefficient(1, 1), 1);
  EXPECT_EQ(binomial_coefficient(5, 5), 1);
  EXPECT_EQ(binomial_coefficient(10, 10), 1);
  EXPECT_EQ(binomial_coefficient(50, 50), 1);

  // C(n, k) = 0 when k > n
  EXPECT_EQ(binomial_coefficient(5, 6), 0);
  EXPECT_EQ(binomial_coefficient(10, 15), 0);
  EXPECT_EQ(binomial_coefficient(0, 1), 0);
  EXPECT_EQ(binomial_coefficient(3, 10), 0);

  // C(1, 0) = 1, C(1, 1) = 1
  EXPECT_EQ(binomial_coefficient(1, 0), 1);
  EXPECT_EQ(binomial_coefficient(1, 1), 1);

  // C(n, 1) = n for any n >= 1
  EXPECT_EQ(binomial_coefficient(1, 1), 1);
  EXPECT_EQ(binomial_coefficient(5, 1), 5);
  EXPECT_EQ(binomial_coefficient(10, 1), 10);
  EXPECT_EQ(binomial_coefficient(100, 1), 100);
}

TEST_F(MathUtilsTest, BinomialCoefficientValues) {
  using qdk::chemistry::utils::microsoft::binomial_coefficient;

  // Small values
  EXPECT_EQ(binomial_coefficient(4, 2), 6);
  EXPECT_EQ(binomial_coefficient(5, 2), 10);
  EXPECT_EQ(binomial_coefficient(5, 3), 10);
  EXPECT_EQ(binomial_coefficient(6, 3), 20);
  EXPECT_EQ(binomial_coefficient(7, 3), 35);
  EXPECT_EQ(binomial_coefficient(8, 4), 70);

  // Medium values
  EXPECT_EQ(binomial_coefficient(10, 5), 252);
  EXPECT_EQ(binomial_coefficient(15, 7), 6435);
  EXPECT_EQ(binomial_coefficient(20, 10), 184756);

  // Large values
  EXPECT_EQ(binomial_coefficient(30, 15), 155117520);
  EXPECT_EQ(binomial_coefficient(40, 20), 137846528820ULL);
  EXPECT_EQ(binomial_coefficient(50, 25), 126410606437752ULL);
  EXPECT_EQ(binomial_coefficient(60, 30), 118264581564861424ULL);
}
