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
#include <qdk/chemistry/utils/valence_space.hpp>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class ValenceActiveParametersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto scf_solver = ScfSolverFactory::create();
    scf_solver->settings().set("basis_set", "STO-3G");

    std::vector<std::string> symbols = {"O", "H", "H"};
    Eigen::MatrixXd coords(3, 3);
    coords << 0.000000, 0.000000, 0.000000,  // O at origin
        0.757000, 0.586000, 0.000000,        // H1
        -0.757000, 0.586000, 0.000000;       // H2
    std::shared_ptr<Structure> water_structure =
        std::make_shared<Structure>(coords, symbols);
    auto [water_e, water_wf] = scf_solver->run(water_structure, 0, 1);
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
    auto [he_e, he_wf] = scf_solver->run(he_structure, 0, 1);
    he_wavefunction = he_wf;

    std::vector<std::string> symbols_oh = {"O", "H"};
    Eigen::MatrixXd coords_oh(2, 3);
    coords_oh << 0.000000, 0.000000, 0.000000,  // O at origin
        0.757000, 0.586000, 0.000000;           // H1
    std::shared_ptr<Structure> oh_structure =
        std::make_shared<Structure>(coords_oh, symbols_oh);

    auto [oh_e, oh_wf] = scf_solver->run(oh_structure, 0, 2);  // doublet
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
  auto result = compute_valence_space(water_wavefunction, 0);

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
  auto result = compute_valence_space(water_wavefunction_truncated, 0);

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
  auto result = compute_valence_space(he_wavefunction, 0);

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
  auto result = compute_valence_space(oh_wavefunction, 0);

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
  auto result = compute_valence_space(ohp_wavefunction, 1);

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
  auto result = compute_valence_space(ohn_wavefunction, -1);

  // For Oxygen-Hydrogen molecule:
  // - num_active_orbitals: 5 (4 from O, 1 from H)
  // - num_active_electrons: 8

  size_t num_active_electrons = result.first;
  size_t num_active_orbitals = result.second;

  EXPECT_EQ(num_active_electrons, 8);
  EXPECT_EQ(num_active_orbitals, 5);
}
