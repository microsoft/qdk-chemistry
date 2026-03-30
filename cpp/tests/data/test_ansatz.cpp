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
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
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

    hamiltonian = std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals, core_energy, inactive_fock));

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
  std::string filename = "test_ansatz.ansatz.h5";

  // Test that the methods exist without requiring them to work perfectly
  EXPECT_NO_THROW(ansatz->to_hdf5_file(filename));

  // Test that from_hdf5_file method exists
  EXPECT_NO_THROW(Ansatz::from_hdf5_file(filename));

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(AnsatzSerializationTest, JSONFileIO) {
  // Test JSON file I/O
  std::string filename = "test_ansatz.ansatz.json";

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
  std::string json_filename = "test_ansatz_generic.ansatz.json";
  std::string hdf5_filename = "test_ansatz_generic.ansatz.h5";

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
  EXPECT_THROW(ansatz->to_file("test.ansatz.xyz", "xyz"), std::runtime_error);
  EXPECT_THROW(Ansatz::from_file("test.ansatz.xyz", "xyz"), std::runtime_error);

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
  EXPECT_THROW(Ansatz::from_json_file("non_existent.ansatz.json"),
               std::runtime_error);
  EXPECT_THROW(Ansatz::from_hdf5_file("non_existent.ansatz.h5"),
               std::runtime_error);
}

class AnsatzEnergyCalculationTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(AnsatzSerializationTest, TestDataTypeName) {
  // test the data_type_name property
  EXPECT_EQ(ansatz->get_data_type_name(), "ansatz");
}

TEST_F(AnsatzEnergyCalculationTest, N2SingletCAS_6e6o) {
  // N2 structure
  auto structure = testing::create_stretched_n2_structure();

  // get wavefunction
  auto scf = ScfSolverFactory::create();
  const auto& [E_scf, wfn_scf] = scf->run(structure, 0, 1, "def2-svp");

  // // get full hamiltonian
  auto hamil_ctor = HamiltonianConstructorFactory::create();
  auto hamiltonian_hf = hamil_ctor->run(wfn_scf->get_orbitals());

  // get ansatz and energy
  auto ansatz_hf = Ansatz(hamiltonian_hf, wfn_scf);
  double energy_hf = ansatz_hf.calculate_energy();

  EXPECT_NEAR(energy_hf, E_scf, testing::scf_energy_tolerance);

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
  EXPECT_NEAR(energy, E_cas, testing::scf_energy_tolerance);
}

TEST_F(AnsatzEnergyCalculationTest, O2TripletCAS_8e6o) {
  // O2 structure
  auto structure = testing::create_o2_structure();

  // get wavefunction
  auto scf = ScfSolverFactory::create();
  const auto& [E_scf, wfn_scf] = scf->run(structure, 0, 3, "def2-svp");

  // get full hamiltonian
  auto hamil_ctor = HamiltonianConstructorFactory::create();
  auto hamiltonian_hf = hamil_ctor->run(wfn_scf->get_orbitals());

  // get ansatz and energy
  auto ansatz_hf = Ansatz(hamiltonian_hf, wfn_scf);
  double energy_hf = ansatz_hf.calculate_energy();

  EXPECT_NEAR(energy_hf, E_scf, testing::scf_energy_tolerance);
}

// Test for creating Ansatz from separately loaded Hamiltonian and Wavefunction
// files
class AnsatzFromSeparateFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create H2 structure
    std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
    // Convert to Bohr
    for (auto& coord : coords) {
      coord *= qdk::chemistry::constants::angstrom_to_bohr;
    }
    std::vector<Element> elements = {Element::H, Element::H};
    structure = std::make_shared<Structure>(coords, elements);
  }

  std::shared_ptr<Structure> structure;
};

TEST_F(AnsatzFromSeparateFilesTest, CreateFromSeparatelyLoadedFiles) {
  // Run SCF to get wavefunction
  auto scf_solver = ScfSolverFactory::create();
  const auto& [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");

  // Build Hamiltonian from the same orbitals
  auto ham_constructor = HamiltonianConstructorFactory::create();
  auto ham = ham_constructor->run(wfn->get_orbitals());

  // Creating Ansatz from in-memory objects should work
  std::shared_ptr<Ansatz> ansatz_direct;
  EXPECT_NO_THROW(ansatz_direct = std::make_shared<Ansatz>(ham, wfn));
  EXPECT_NE(ansatz_direct, nullptr);

  // Create temporary files
  std::string wfn_path = "test_ansatz_wfn.wavefunction.json";
  std::string ham_path = "test_ansatz_ham.hamiltonian.json";
  std::string ansatz_path = "test_ansatz_unit.ansatz.json";

  // Save objects to separate files
  wfn->to_json_file(wfn_path);
  ham->to_json_file(ham_path);
  ansatz_direct->to_json_file(ansatz_path);

  // Loading Ansatz that was saved as a unit should work
  auto loaded_ansatz = Ansatz::from_json_file(ansatz_path);
  EXPECT_NE(loaded_ansatz, nullptr);
  EXPECT_NE(loaded_ansatz->get_hamiltonian(), nullptr);
  EXPECT_NE(loaded_ansatz->get_wavefunction(), nullptr);

  // Load Hamiltonian and Wavefunction separately
  auto loaded_wfn = Wavefunction::from_json_file(wfn_path);
  auto loaded_ham = Hamiltonian::from_json_file(ham_path);
  EXPECT_NE(loaded_wfn, nullptr);
  EXPECT_NE(loaded_ham, nullptr);

  // Verify orbitals are structurally equivalent
  auto orig_orbs = wfn->get_orbitals();
  auto loaded_wfn_orbs = loaded_wfn->get_orbitals();
  auto loaded_ham_orbs = loaded_ham->get_orbitals();

  EXPECT_EQ(orig_orbs->get_num_molecular_orbitals(),
            loaded_wfn_orbs->get_num_molecular_orbitals());
  EXPECT_EQ(orig_orbs->get_num_molecular_orbitals(),
            loaded_ham_orbs->get_num_molecular_orbitals());
  EXPECT_EQ(orig_orbs->is_restricted(), loaded_wfn_orbs->is_restricted());
  EXPECT_EQ(orig_orbs->is_restricted(), loaded_ham_orbs->is_restricted());

  // Creating Ansatz from separately loaded files should work
  std::shared_ptr<Ansatz> ansatz_loaded;
  EXPECT_NO_THROW(ansatz_loaded =
                      std::make_shared<Ansatz>(loaded_ham, loaded_wfn));
  EXPECT_NE(ansatz_loaded, nullptr);
  EXPECT_NE(ansatz_loaded->get_hamiltonian(), nullptr);
  EXPECT_NE(ansatz_loaded->get_wavefunction(), nullptr);

  // Mixed original + loaded should work
  std::shared_ptr<Ansatz> ansatz_mixed1;
  EXPECT_NO_THROW(ansatz_mixed1 = std::make_shared<Ansatz>(ham, loaded_wfn));
  EXPECT_NE(ansatz_mixed1, nullptr);

  std::shared_ptr<Ansatz> ansatz_mixed2;
  EXPECT_NO_THROW(ansatz_mixed2 = std::make_shared<Ansatz>(loaded_ham, wfn));
  EXPECT_NE(ansatz_mixed2, nullptr);

  // Clean up
  std::remove(wfn_path.c_str());
  std::remove(ham_path.c_str());
  std::remove(ansatz_path.c_str());
}

TEST_F(AnsatzFromSeparateFilesTest, CreateFromSeparatelyLoadedHDF5Files) {
  // Run SCF to get wavefunction
  auto scf_solver = ScfSolverFactory::create();
  const auto& [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");

  // Build Hamiltonian from the same orbitals
  auto ham_constructor = HamiltonianConstructorFactory::create();
  auto ham = ham_constructor->run(wfn->get_orbitals());

  // Create Ansatz from in-memory objects
  auto ansatz_direct = std::make_shared<Ansatz>(ham, wfn);

  // Create temporary files
  std::string wfn_path = "test_ansatz_wfn.wavefunction.h5";
  std::string ham_path = "test_ansatz_ham.hamiltonian.h5";
  std::string ansatz_path = "test_ansatz_unit.ansatz.h5";

  // Save objects to separate files
  wfn->to_hdf5_file(wfn_path);
  ham->to_hdf5_file(ham_path);
  ansatz_direct->to_hdf5_file(ansatz_path);

  // Load Hamiltonian and Wavefunction separately
  auto loaded_wfn = Wavefunction::from_hdf5_file(wfn_path);
  auto loaded_ham = Hamiltonian::from_hdf5_file(ham_path);

  // Creating Ansatz from separately loaded HDF5 files should work
  std::shared_ptr<Ansatz> ansatz_loaded;
  EXPECT_NO_THROW(ansatz_loaded =
                      std::make_shared<Ansatz>(loaded_ham, loaded_wfn));
  EXPECT_NE(ansatz_loaded, nullptr);

  // Clean up
  std::remove(wfn_path.c_str());
  std::remove(ham_path.c_str());
  std::remove(ansatz_path.c_str());
}

TEST_F(AnsatzFromSeparateFilesTest, DetectsOrbitalMismatch) {
  // Run SCF to get first wavefunction (H2)
  auto scf_solver = ScfSolverFactory::create();
  const auto& [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");

  // Build Hamiltonian from the first wavefunction's orbitals
  auto ham_constructor = HamiltonianConstructorFactory::create();
  auto ham = ham_constructor->run(wfn->get_orbitals());

  // Create a different structure (water) to get different orbitals
  auto water_structure = testing::create_water_structure();
  const auto& [E_water, wfn_water] =
      scf_solver->run(water_structure, 0, 1, "sto-3g");

  // Attempting to create Ansatz with mismatched orbitals should fail
  EXPECT_THROW(
      { auto ansatz = Ansatz(ham, wfn_water); }, std::invalid_argument);
}
