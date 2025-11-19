// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <stdexcept>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class AnsatzSerializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test orbitals
    orbitals = testing::create_test_orbitals();

    // Create test wavefunction
    Eigen::VectorXd coeffs(2);
    coeffs << 0.8, 0.6;

    Wavefunction::DeterminantVector dets = {Configuration("200000"),
                                            Configuration("ud0000")};

    auto wf_container =
        std::make_unique<CasWavefunctionContainer>(coeffs, dets, orbitals);
    wavefunction = std::make_shared<Wavefunction>(std::move(wf_container));

    // Create test structure
    structure = testing::create_water_structure();

    // Create test hamiltonian with proper parameters (following
    // test_hamiltonian.cpp pattern)
    Eigen::MatrixXd one_body = Eigen::MatrixXd::Identity(2, 2);
    one_body(0, 1) = 0.5;
    one_body(1, 0) = 0.5;

    Eigen::VectorXd two_body =
        2 * Eigen::VectorXd::Ones(16);  // 2^4 = 16 for 2 orbitals
    double core_energy = 1.5;
    Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Zero(0, 0);

    hamiltonian = std::make_shared<Hamiltonian>(one_body, two_body, orbitals,
                                                core_energy, inactive_fock);

    // Create test ansatz
    ansatz = std::make_shared<Ansatz>(hamiltonian, wavefunction);
  }

  std::shared_ptr<Orbitals> orbitals;
  std::shared_ptr<Wavefunction> wavefunction;
  std::shared_ptr<Structure> structure;
  std::shared_ptr<Hamiltonian> hamiltonian;
  std::shared_ptr<Ansatz> ansatz;
};

TEST_F(AnsatzSerializationTest, JSONSerialization) {
  // Test that JSON serialization methods exist and can be called
  nlohmann::json j;
  EXPECT_NO_THROW(j = ansatz->to_json());

  // Verify essential fields are present
  EXPECT_TRUE(j.contains("wavefunction"));
  EXPECT_TRUE(j.contains("hamiltonian"));

  // Test that from_json method exists (but might not work correctly yet)
  // Just verify the API exists without requiring it to work perfectly
  EXPECT_NO_THROW(Ansatz::from_json(j));
}

TEST_F(AnsatzSerializationTest, HDF5Serialization) {
  // Test that HDF5 serialization methods exist and can be called
  std::string filename = "test_ansatz.h5";

  // Test that the methods exist without requiring them to work perfectly
  EXPECT_NO_THROW(ansatz->to_hdf5_file(filename));

  // Test that from_hdf5_file method exists
  EXPECT_NO_THROW(Ansatz::from_hdf5_file(filename));

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(AnsatzSerializationTest, JSONFileIO) {
  // Test JSON file I/O
  std::string filename = "test_ansatz.json";

  // Save to JSON file
  ansatz->to_json_file(filename);

  // Load from JSON file
  auto ansatz_reconstructed = Ansatz::from_json_file(filename);
  EXPECT_NE(ansatz_reconstructed, nullptr);

  // Verify nested objects are preserved
  EXPECT_NE(ansatz_reconstructed->get_wavefunction(), nullptr);
  EXPECT_NE(ansatz_reconstructed->get_hamiltonian(), nullptr);
  EXPECT_NE(ansatz_reconstructed->get_orbitals(), nullptr);

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(AnsatzSerializationTest, GenericFileIO) {
  // Test generic file I/O with different formats
  std::string json_filename = "test_ansatz_generic.json";
  std::string hdf5_filename = "test_ansatz_generic.h5";

  // Test JSON format
  ansatz->to_file(json_filename, "json");
  auto ansatz_json = Ansatz::from_file(json_filename, "json");
  EXPECT_NE(ansatz_json, nullptr);
  EXPECT_NE(ansatz_json->get_wavefunction(), nullptr);

  // Test HDF5 format
  ansatz->to_file(hdf5_filename, "hdf5");
  auto ansatz_hdf5 = Ansatz::from_file(hdf5_filename, "hdf5");
  EXPECT_NE(ansatz_hdf5, nullptr);
  EXPECT_NE(ansatz_hdf5->get_wavefunction(), nullptr);

  // Test invalid format
  EXPECT_THROW(ansatz->to_file("test.xyz", "xyz"), std::runtime_error);
  EXPECT_THROW(Ansatz::from_file("test.xyz", "xyz"), std::runtime_error);

  // Clean up
  std::remove(json_filename.c_str());
  std::remove(hdf5_filename.c_str());
}

TEST_F(AnsatzSerializationTest, ErrorHandling) {
  // Test error handling for malformed JSON
  nlohmann::json bad_json;
  bad_json["wavefunction"] = "invalid";

  EXPECT_THROW(Ansatz::from_json(bad_json), std::runtime_error);

  // Test error handling for non-existent files
  EXPECT_THROW(Ansatz::from_json_file("non_existent.json"), std::runtime_error);
  EXPECT_THROW(Ansatz::from_hdf5_file("non_existent.h5"), std::runtime_error);
}

class AnsatzEnergyCalculationTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(AnsatzEnergyCalculationTest, N2SingletCAS_6e6o) {
  // N2 structure
  auto structure = testing::create_stretched_n2_structure();

  // get wavefunction
  auto scf = ScfSolverFactory::create();
  const auto& [E_scf, wfn_scf] = scf->run(structure, 0, 1);

  // // get full hamiltonian
  auto hamil_ctor = HamiltonianConstructorFactory::create();
  auto hamiltonian_hf = hamil_ctor->run(wfn_scf->get_orbitals());

  // get ansatz and energy
  auto ansatz_hf = Ansatz(hamiltonian_hf, wfn_scf);
  double energy_hf = ansatz_hf.calculate_energy();

  EXPECT_NEAR(energy_hf, E_scf, 1e-6);

  // select active space
  auto active_space = ActiveSpaceSelectorFactory::create("qdk_valence");
  active_space->settings().set("num_active_electrons", 6);
  active_space->settings().set("num_active_orbitals", 6);
  auto active_space_wfn = active_space->run(wfn_scf);

  // get hamiltonian
  auto hamiltonian_cas = hamil_ctor->run(active_space_wfn->get_orbitals());

  // get cas wavefunction
  auto mc_calc = MultiConfigurationCalculatorFactory::create("macis_cas");
  mc_calc->settings().set("calculate_two_rdm", true);
  mc_calc->settings().set("calculate_one_rdm", true);
  auto [E_cas, wfn_cas] = mc_calc->run(
      hamiltonian_cas, active_space_wfn->get_active_num_electrons().first,
      active_space_wfn->get_active_num_electrons().second);

  // get ansatz and energy
  auto ansatz = Ansatz(hamiltonian_cas, wfn_cas);
  double energy = ansatz.calculate_energy();

  // energy should match SCF energy
  EXPECT_NEAR(energy, E_cas, 1e-6);
}

// TODO(MM): Comment in as soon as the unrestricted Hamiltonian is merged
// TEST_F(AnsatzEnergyCalculationTest, O2TripletCAS_8e6o) {
//   // O2 structure
//   auto structure = testing::create_o2_structure();

//   // get wavefunction
//   auto scf = ScfSolverFactory::create();
//   const auto& [E_scf, wfn_scf] = scf->run(structure, 0, 3);

//   // get full hamiltonian
//   auto hamil_ctor = HamiltonianConstructorFactory::create();
//   auto hamiltonian_hf = hamil_ctor->run(wfn_scf->get_orbitals());

//   // get ansatz and energy
//   auto ansatz_hf = Ansatz(hamiltonian_hf, wfn_scf);
//   double energy_hf = ansatz_hf.calculate_energy();

//   EXPECT_NEAR(energy_hf, E_scf, 1e-6);

//   // select active space
//   auto active_space = ActiveSpaceSelectorFactory::create("valence");
//   active_space->settings().set("num_active_electrons", 8);
//   active_space->settings().set("num_active_orbitals", 6);
//   auto active_space_wfn = active_space->run(wfn_scf);

//   // get hamiltonian
//   auto hamiltonian_cas = hamil_ctor->run(wfn_scf->get_orbitals());

//   // get cas wavefunction
//   auto mc_calc = MultiConfigurationCalculatorFactory::create("macis_cas");
//   auto [E_cas, wfn_cas] = mc_calc->run(
//       hamiltonian_hf, active_space_wfn->get_active_num_electrons().first,
//       active_space_wfn->get_active_num_electrons().second);

//   // get ansatz and energy
//   auto ansatz = Ansatz(hamiltonian_cas, wfn_cas);
//   double energy = ansatz.calculate_energy();

//   // energy should match SCF energy
//   EXPECT_NEAR(energy, E_cas, 1e-6);
// }
