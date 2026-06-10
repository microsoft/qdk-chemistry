/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

// Helper struct to hold common MP2 test setup results
struct O2TestSetup {
  std::shared_ptr<Structure> structure;
  std::shared_ptr<Wavefunction> hf_wavefunction;
  std::shared_ptr<Orbitals> hf_orbitals;
  std::shared_ptr<Hamiltonian> hf_hamiltonian;
  double hf_energy;
};

// Helper function to create O2 structure and run HF calculation
// bond_length: O-O bond length in Bohr (default 2.3)
// multiplicity: 1 for singlet (restricted), 3 for triplet (unrestricted)
// basis_set: basis set name (default "cc-pvdz")
inline O2TestSetup create_o2_hf_setup(
    int multiplicity = 1, const std::string& basis_set = "cc-pvdz",
    double bond_length = 2.3,
    std::optional<std::pair<int, int>> active_space = std::nullopt) {
  O2TestSetup setup;

  // O2 structure with specified bond length
  std::vector<Eigen::Vector3d> coordinates = {
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(bond_length, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  setup.structure = std::make_shared<Structure>(coordinates, symbols);

  // HF calculation
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");
  auto [hf_energy, hf_wavefunction] =
      scf_factory->run(setup.structure, 0, multiplicity, basis_set);

  setup.hf_energy = hf_energy;
  setup.hf_wavefunction = hf_wavefunction;
  setup.hf_orbitals = hf_wavefunction->get_orbitals();

  if (active_space.has_value()) {
    auto valence_active_space_selector =
        qdk::chemistry::algorithms::ActiveSpaceSelectorFactory::create(
            "qdk_valence");
    valence_active_space_selector->settings().set("num_active_electrons",
                                                  active_space->first);
    valence_active_space_selector->settings().set("num_active_orbitals",
                                                  active_space->second);
    auto wfn_active = valence_active_space_selector->run(hf_wavefunction);

    setup.hf_wavefunction = wfn_active;
    setup.hf_orbitals = wfn_active->get_orbitals();
  }
  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  setup.hf_hamiltonian = ham_factory->run(setup.hf_orbitals);
  return setup;
}

class MP2Test : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Runs MP2 on the given setup and returns the resulting wavefunction, which is
// backed by an AmplitudeContainer storing the computed T1/T2 amplitudes.
inline std::shared_ptr<Wavefunction> run_mp2(const O2TestSetup& setup) {
  auto ansatz =
      std::make_shared<Ansatz>(*setup.hf_hamiltonian, *setup.hf_wavefunction);
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");
  auto [energy, wavefunction, bra] = mp2_calculator->run(ansatz);
  return wavefunction;
}

TEST_F(MP2Test, UMP2Energies_CCPVDZ) {
  // Test the UMP2 energies against reference for cc-pvdz
  double ref = -0.35094710125187589;

  // Triplet O2 (unrestricted)
  auto setup = create_o2_hf_setup(3);
  auto [n_alpha_active, n_beta_active] =
      setup.hf_wavefunction->get_active_num_electrons();

  // Create ansatz from Hamiltonian and wavefunction
  auto ansatz =
      std::make_shared<Ansatz>(*setup.hf_hamiltonian, *setup.hf_wavefunction);

  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");

  auto [mp2_total_energy, final_wavefunction, _] = mp2_calculator->run(ansatz);

  double reference = ansatz->calculate_energy();
  double mp2_corr_energy = mp2_total_energy - reference;

  EXPECT_LT(std::abs(mp2_corr_energy - ref), testing::mp2_tolerance)
      << "UMP2 correlation energy mismatch (cc-pvdz). Calculated: "
      << mp2_corr_energy << ", Reference: " << ref
      << ", Difference: " << (mp2_corr_energy - ref);
}

TEST_F(MP2Test, RMP2Energies_CCPVDZ) {
  // Test the RMP2 energies against PySCF reference for singlet O2 with
  // cc-pvdz
  float pyscf_rmp2_corr_cc_pvdz = -0.38428662586339435;

  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);
  auto [n_alpha_active, n_beta_active] =
      setup.hf_wavefunction->get_active_num_electrons();

  // Verify closed shell
  EXPECT_EQ(n_alpha_active, n_beta_active)
      << "Alpha and beta electrons should be equal for restricted "
         "calculation";

  // Create ansatz from Hamiltonian and wavefunction
  auto ansatz =
      std::make_shared<Ansatz>(*setup.hf_hamiltonian, *setup.hf_wavefunction);

  // Use MP2 calculator
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");

  // MP2 returns total energy, subtract reference to get correlation energy
  auto [mp2_total_energy, final_wavefunction, _] = mp2_calculator->run(ansatz);
  double hf_reference_energy = ansatz->calculate_energy();
  double mp2_corr_energy = mp2_total_energy - hf_reference_energy;

  // Verify correlation energy matches PySCF reference
  EXPECT_LT(std::abs(mp2_corr_energy - pyscf_rmp2_corr_cc_pvdz),
            testing::mp2_tolerance)
      << "RMP2 correlation energy mismatch (cc-pvdz). Calculated: "
      << mp2_corr_energy << ", Reference: " << pyscf_rmp2_corr_cc_pvdz
      << ", Difference: " << (mp2_corr_energy - pyscf_rmp2_corr_cc_pvdz);
}

TEST_F(MP2Test, ActiveRMP2Energies_CCPVDZ) {
  // Test the RMP2 energies against Psi4 reference for singlet O2 with
  // cc-pvdz
  float psi4_act_rmp2_corr_cc_pvdz = -0.0779663051614509;

  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1, "cc-pvdz", 2.3, std::make_pair(12, 8));
  auto [n_alpha_active, n_beta_active] =
      setup.hf_wavefunction->get_active_num_electrons();

  // Verify closed shell
  EXPECT_EQ(n_alpha_active, n_beta_active)
      << "Alpha and beta electrons should be equal for restricted "
         "calculation";

  // Create ansatz from Hamiltonian and wavefunction
  auto ansatz =
      std::make_shared<Ansatz>(*setup.hf_hamiltonian, *setup.hf_wavefunction);

  // Use MP2 calculator
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");

  // MP2 returns total energy, subtract reference to get correlation energy
  auto [mp2_total_energy, final_wavefunction, _] = mp2_calculator->run(ansatz);
  double hf_reference_energy = ansatz->calculate_energy();
  double mp2_corr_energy = mp2_total_energy - hf_reference_energy;

  // Verify correlation energy matches PySCF reference
  EXPECT_LT(std::abs(mp2_corr_energy - psi4_act_rmp2_corr_cc_pvdz),
            testing::mp2_tolerance)
      << "Active RMP2 correlation energy mismatch (cc-pvdz). Calculated: "
      << mp2_corr_energy << ", Reference: " << psi4_act_rmp2_corr_cc_pvdz
      << ", Difference: " << (mp2_corr_energy - psi4_act_rmp2_corr_cc_pvdz);
}

TEST_F(MP2Test, MP2Container) {
  // Test that the MP2 algorithm produces an AmplitudeContainer with the
  // expected amplitudes. Singlet O2 (restricted).
  auto setup = create_o2_hf_setup(1);

  auto wavefunction = run_mp2(setup);
  EXPECT_EQ(wavefunction->get_container_type(), "amplitude");
  const auto& container = wavefunction->get_container<AmplitudeContainer>();

  EXPECT_EQ(container.get_amplitude_type(), AmplitudeType::MP2)
      << "The MP2 algorithm should tag its container as MP2";

  EXPECT_TRUE(container.has_t1_amplitudes())
      << "T1 amplitudes should be stored by the MP2 algorithm";
  EXPECT_TRUE(container.has_t2_amplitudes())
      << "T2 amplitudes should be stored by the MP2 algorithm";

  auto [t1_aa, t1_bb] = container.get_t1_amplitudes();
  auto [t2_abab, t2_aaaa, t2_bbbb] = container.get_t2_amplitudes();

  // Verify T1 amplitudes are zero for MP2
  auto check_t1_zero = [](const AmplitudeContainer::VectorVariant& t1) {
    return std::visit([](auto&& vec) { return vec.isZero(1e-10); }, t1);
  };

  EXPECT_TRUE(check_t1_zero(t1_aa))
      << "T1 alpha amplitudes should be zero for MP2";
  EXPECT_TRUE(check_t1_zero(t1_bb))
      << "T1 beta amplitudes should be zero for MP2";

  // Verify T2 amplitudes are non-zero
  auto check_t2_nonzero = [](const AmplitudeContainer::VectorVariant& t2) {
    return std::visit([](auto&& vec) { return vec.norm() > 1e-10; }, t2);
  };

  EXPECT_TRUE(check_t2_nonzero(t2_abab))
      << "T2 alpha-beta amplitudes should be non-zero for MP2";
  EXPECT_TRUE(check_t2_nonzero(t2_aaaa))
      << "T2 alpha-alpha amplitudes should be non-zero for MP2";
  EXPECT_TRUE(check_t2_nonzero(t2_bbbb))
      << "T2 beta-beta amplitudes should be non-zero for MP2";
}

TEST_F(MP2Test, JsonSerializationSpatial) {
  // Test JSON serialization/deserialization for spatial MP2
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Run MP2 to obtain an AmplitudeContainer with stored amplitudes
  auto wavefunction = run_mp2(setup);
  const auto& original = wavefunction->get_container<AmplitudeContainer>();
  auto [t1_aa_orig, t1_bb_orig] = original.get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original.get_t2_amplitudes();

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  auto restored = AmplitudeContainer::from_json(j);

  // Verify amplitudes match
  auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
  auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
      restored->get_t2_amplitudes();

  auto compare_amplitudes = [](const AmplitudeContainer::VectorVariant& orig,
                               const AmplitudeContainer::VectorVariant& rest) {
    return std::visit(
        [](auto&& orig_vec, auto&& rest_vec) {
          using T1 = std::decay_t<decltype(orig_vec)>;
          using T2 = std::decay_t<decltype(rest_vec)>;
          if constexpr (std::is_same_v<T1, T2>) {
            return orig_vec.isApprox(rest_vec, testing::wf_tolerance);
          } else {
            return false;
          }
        },
        orig, rest);
  };

  EXPECT_TRUE(compare_amplitudes(t1_aa_orig, t1_aa_rest))
      << "T1 alpha amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t1_bb_orig, t1_bb_rest))
      << "T1 beta amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t2_abab_orig, t2_abab_rest))
      << "T2 alpha-beta amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t2_aaaa_orig, t2_aaaa_rest))
      << "T2 alpha-alpha amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t2_bbbb_orig, t2_bbbb_rest))
      << "T2 beta-beta amplitudes should match after JSON serialization";
}

TEST_F(MP2Test, JsonSerializationSpin) {
  // Test JSON serialization/deserialization for unrestricted MP2
  // Triplet O2 (unrestricted)
  auto setup = create_o2_hf_setup(3);

  // Run MP2 to obtain an AmplitudeContainer with stored amplitudes
  auto wavefunction = run_mp2(setup);
  const auto& original = wavefunction->get_container<AmplitudeContainer>();
  auto [t1_aa_orig, t1_bb_orig] = original.get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original.get_t2_amplitudes();

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  auto restored = AmplitudeContainer::from_json(j);

  // Verify amplitudes match
  auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
  auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
      restored->get_t2_amplitudes();

  auto compare_amplitudes = [](const AmplitudeContainer::VectorVariant& orig,
                               const AmplitudeContainer::VectorVariant& rest) {
    return std::visit(
        [](auto&& orig_vec, auto&& rest_vec) {
          using T1 = std::decay_t<decltype(orig_vec)>;
          using T2 = std::decay_t<decltype(rest_vec)>;
          if constexpr (std::is_same_v<T1, T2>) {
            return orig_vec.isApprox(rest_vec, testing::wf_tolerance);
          } else {
            return false;
          }
        },
        orig, rest);
  };

  EXPECT_TRUE(compare_amplitudes(t1_aa_orig, t1_aa_rest))
      << "T1 alpha amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t1_bb_orig, t1_bb_rest))
      << "T1 beta amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t2_abab_orig, t2_abab_rest))
      << "T2 alpha-beta amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t2_aaaa_orig, t2_aaaa_rest))
      << "T2 alpha-alpha amplitudes should match after JSON serialization";
  EXPECT_TRUE(compare_amplitudes(t2_bbbb_orig, t2_bbbb_rest))
      << "T2 beta-beta amplitudes should match after JSON serialization";
}

TEST_F(MP2Test, Hdf5SerializationSpatial) {
  // Test HDF5 serialization/deserialization for spatial MP2
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Run MP2 to obtain an AmplitudeContainer with stored amplitudes
  auto wavefunction = run_mp2(setup);
  const auto& original = wavefunction->get_container<AmplitudeContainer>();
  auto [t1_aa_orig, t1_bb_orig] = original.get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original.get_t2_amplitudes();

  std::string filename = "test_mp2_spatial_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = AmplitudeContainer::from_hdf5(root);

    // Verify amplitudes match
    auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();

    auto compare_amplitudes =
        [](const AmplitudeContainer::VectorVariant& orig,
           const AmplitudeContainer::VectorVariant& rest) {
          return std::visit(
              [](auto&& orig_vec, auto&& rest_vec) {
                using T1 = std::decay_t<decltype(orig_vec)>;
                using T2 = std::decay_t<decltype(rest_vec)>;
                if constexpr (std::is_same_v<T1, T2>) {
                  return orig_vec.isApprox(rest_vec, testing::wf_tolerance);
                } else {
                  return false;
                }
              },
              orig, rest);
        };

    EXPECT_TRUE(compare_amplitudes(t1_aa_orig, t1_aa_rest))
        << "T1 alpha amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t1_bb_orig, t1_bb_rest))
        << "T1 beta amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t2_abab_orig, t2_abab_rest))
        << "T2 alpha-beta amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t2_aaaa_orig, t2_aaaa_rest))
        << "T2 alpha-alpha amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t2_bbbb_orig, t2_bbbb_rest))
        << "T2 beta-beta amplitudes should match after HDF5 serialization";

    file.close();
  }

  std::remove(filename.c_str());
}

TEST_F(MP2Test, Hdf5SerializationSpin) {
  // Test HDF5 serialization/deserialization for unrestricted MP2
  // Triplet O2 (unrestricted)
  auto setup = create_o2_hf_setup(3);

  // Run MP2 to obtain an AmplitudeContainer with stored amplitudes
  auto wavefunction = run_mp2(setup);
  const auto& original = wavefunction->get_container<AmplitudeContainer>();
  auto [t1_aa_orig, t1_bb_orig] = original.get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original.get_t2_amplitudes();

  std::string filename = "test_mp2_spin_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = AmplitudeContainer::from_hdf5(root);

    // Verify amplitudes match
    auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();

    auto compare_amplitudes =
        [](const AmplitudeContainer::VectorVariant& orig,
           const AmplitudeContainer::VectorVariant& rest) {
          return std::visit(
              [](auto&& orig_vec, auto&& rest_vec) {
                using T1 = std::decay_t<decltype(orig_vec)>;
                using T2 = std::decay_t<decltype(rest_vec)>;
                if constexpr (std::is_same_v<T1, T2>) {
                  return orig_vec.isApprox(rest_vec, testing::wf_tolerance);
                } else {
                  return false;
                }
              },
              orig, rest);
        };

    EXPECT_TRUE(compare_amplitudes(t1_aa_orig, t1_aa_rest))
        << "T1 alpha amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t1_bb_orig, t1_bb_rest))
        << "T1 beta amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t2_abab_orig, t2_abab_rest))
        << "T2 alpha-beta amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t2_aaaa_orig, t2_aaaa_rest))
        << "T2 alpha-alpha amplitudes should match after HDF5 serialization";
    EXPECT_TRUE(compare_amplitudes(t2_bbbb_orig, t2_bbbb_rest))
        << "T2 beta-beta amplitudes should match after HDF5 serialization";

    file.close();
  }

  std::remove(filename.c_str());
}

// Test (base) Wavefunction-level JSON serialization/deserialization
// The other tests exercise the AmplitudeContainer directly
TEST_F(MP2Test, WavefunctionJsonSerializationSpatial) {
  // Test JSON serialization/deserialization for MP2 via
  // Wavefunction::from_json()
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Run MP2 to obtain a wavefunction backed by an AmplitudeContainer.
  auto original_wavefunction = run_mp2(setup);

  // Verify container type
  EXPECT_EQ(original_wavefunction->get_container_type(), "amplitude");

  // Serialize to JSON using Wavefunction::to_json()
  nlohmann::json j = original_wavefunction->to_json();

  // Verify JSON contains container_type field
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_EQ(j["container_type"], "amplitude");

  // Deserialize from JSON using Wavefunction::from_json()
  auto restored_wavefunction = Wavefunction::from_json(j);

  // Verify restored wavefunction has correct container type
  EXPECT_EQ(restored_wavefunction->get_container_type(), "amplitude");
}

// Test (base-) Wavefunction-level HDF5 serialization/deserialization
TEST_F(MP2Test, WavefunctionHdf5SerializationSpatial) {
  // Test HDF5 serialization/deserialization for MP2 via
  // Wavefunction::from_hdf5()
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Run MP2 to obtain a wavefunction backed by an AmplitudeContainer.
  auto original_wavefunction = run_mp2(setup);

  EXPECT_EQ(original_wavefunction->get_container_type(), "amplitude");

  std::string filename = "test_mp2_wavefunction_hdf5_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    original_wavefunction->to_hdf5(root);
    auto restored_wavefunction = Wavefunction::from_hdf5(root);

    EXPECT_EQ(restored_wavefunction->get_container_type(), "amplitude");

    file.close();
  }

  std::remove(filename.c_str());
}
