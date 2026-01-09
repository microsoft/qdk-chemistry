/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <H5Cpp.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>

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
inline O2TestSetup create_o2_hf_setup(int multiplicity = 1,
                                      const std::string& basis_set = "cc-pvdz",
                                      double bond_length = 2.3) {
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

TEST_F(MP2Test, UMP2Energies_CCPVDZ) {
  // Test the UMP2 energies against reference for cc-pvdz
  float pyscf_mp2_corr_cc_pvdz = -0.3509470131940627;

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

  EXPECT_LT(std::abs(mp2_corr_energy - pyscf_mp2_corr_cc_pvdz),
            testing::mp2_tolerance)
      << "UMP2 correlation energy mismatch (cc-pvdz). Calculated: "
      << mp2_corr_energy << ", Reference: " << pyscf_mp2_corr_cc_pvdz
      << ", Difference: " << (mp2_corr_energy - pyscf_mp2_corr_cc_pvdz);
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

TEST_F(MP2Test, MP2Container) {
  // Test that MP2Container properly computes amplitudes
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Create MP2Container
  auto mp2_container_with_amplitudes = std::make_unique<MP2Container>(
      setup.hf_hamiltonian, setup.hf_wavefunction);

  // Verify Hamiltonian is stored
  EXPECT_NE(mp2_container_with_amplitudes->get_hamiltonian(), nullptr)
      << "MP2Container should store Hamiltonian reference";

  // Lazy evaluation: Amplitudes should not be available initially
  EXPECT_FALSE(mp2_container_with_amplitudes->has_t1_amplitudes())
      << "T1 amplitudes should NOT be computed until requested (lazy "
         "evaluation)";
  EXPECT_FALSE(mp2_container_with_amplitudes->has_t2_amplitudes())
      << "T2 amplitudes should NOT be computed until requested (lazy "
         "evaluation)";

  // Verify we can retrieve the amplitudes (this triggers lazy computation)
  auto [t1_aa, t1_bb] = mp2_container_with_amplitudes->get_t1_amplitudes();
  auto [t2_abab, t2_aaaa, t2_bbbb] =
      mp2_container_with_amplitudes->get_t2_amplitudes();

  // After calling getters, amplitudes should now be available
  EXPECT_TRUE(mp2_container_with_amplitudes->has_t1_amplitudes())
      << "T1 amplitudes should be cached after first access";
  EXPECT_TRUE(mp2_container_with_amplitudes->has_t2_amplitudes())
      << "T2 amplitudes should be cached after first access";

  // Verify T1 amplitudes are zero for MP2
  auto check_t1_zero = [](const MP2Container::VectorVariant& t1) {
    return std::visit([](auto&& vec) { return vec.isZero(1e-10); }, t1);
  };

  EXPECT_TRUE(check_t1_zero(t1_aa))
      << "T1 alpha amplitudes should be zero for MP2";
  EXPECT_TRUE(check_t1_zero(t1_bb))
      << "T1 beta amplitudes should be zero for MP2";

  // Verify T2 amplitudes are non-zero
  auto check_t2_nonzero = [](const MP2Container::VectorVariant& t2) {
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

  // Create MP2Container and compute amplitudes
  auto original = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                 setup.hf_wavefunction);

  // Trigger amplitude computation
  auto [t1_aa_orig, t1_bb_orig] = original->get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original->get_t2_amplitudes();

  // Serialize to JSON
  nlohmann::json j = original->to_json();

  // Deserialize from JSON
  auto restored = MP2Container::from_json(j);

  // Verify amplitudes match
  auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
  auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
      restored->get_t2_amplitudes();

  auto compare_amplitudes = [](const MP2Container::VectorVariant& orig,
                               const MP2Container::VectorVariant& rest) {
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

  // Create MP2Container and compute amplitudes
  auto original = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                 setup.hf_wavefunction);

  // Trigger amplitude computation
  auto [t1_aa_orig, t1_bb_orig] = original->get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original->get_t2_amplitudes();

  // Serialize to JSON
  nlohmann::json j = original->to_json();

  // Deserialize from JSON
  auto restored = MP2Container::from_json(j);

  // Verify amplitudes match
  auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
  auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
      restored->get_t2_amplitudes();

  auto compare_amplitudes = [](const MP2Container::VectorVariant& orig,
                               const MP2Container::VectorVariant& rest) {
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

  // Create MP2Container and compute amplitudes
  auto original = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                 setup.hf_wavefunction);

  // Trigger amplitude computation
  auto [t1_aa_orig, t1_bb_orig] = original->get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original->get_t2_amplitudes();

  std::string filename = "test_mp2_spatial_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original->to_hdf5(root);

    // Deserialize from HDF5
    auto restored = MP2Container::from_hdf5(root);

    // Verify amplitudes match
    auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();

    auto compare_amplitudes = [](const MP2Container::VectorVariant& orig,
                                 const MP2Container::VectorVariant& rest) {
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

  // Create MP2Container and compute amplitudes
  auto original = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                 setup.hf_wavefunction);

  // Trigger amplitude computation
  auto [t1_aa_orig, t1_bb_orig] = original->get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original->get_t2_amplitudes();

  std::string filename = "test_mp2_spin_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original->to_hdf5(root);

    // Deserialize from HDF5
    auto restored = MP2Container::from_hdf5(root);

    // Verify amplitudes match
    auto [t1_aa_rest, t1_bb_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();

    auto compare_amplitudes = [](const MP2Container::VectorVariant& orig,
                                 const MP2Container::VectorVariant& rest) {
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
// The other tests test the MP2Container directly
TEST_F(MP2Test, WavefunctionJsonSerializationSpatial) {
  // Test JSON serialization/deserialization for MP2 via
  // Wavefunction::from_json()
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Create MP2Container
  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);

  // Trigger amplitude computation before wrapping in Wavefunction
  mp2_container->get_t2_amplitudes();

  auto original_wavefunction =
      std::make_shared<Wavefunction>(std::move(mp2_container));

  // Verify container type
  EXPECT_EQ(original_wavefunction->get_container_type(), "mp2");

  // Serialize to JSON using Wavefunction::to_json()
  nlohmann::json j = original_wavefunction->to_json();

  // Verify JSON contains container_type field
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_EQ(j["container_type"], "mp2");

  // Deserialize from JSON using Wavefunction::from_json()
  auto restored_wavefunction = Wavefunction::from_json(j);

  // Verify restored wavefunction has correct container type
  EXPECT_EQ(restored_wavefunction->get_container_type(), "mp2");
}

// Test (base-) Wavefunction-level HDF5 serialization/deserialization
TEST_F(MP2Test, WavefunctionHdf5SerializationSpatial) {
  // Test HDF5 serialization/deserialization for MP2 via
  // Wavefunction::from_hdf5()
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);
  mp2_container->get_t2_amplitudes();

  auto original_wavefunction =
      std::make_shared<Wavefunction>(std::move(mp2_container));

  EXPECT_EQ(original_wavefunction->get_container_type(), "mp2");

  std::string filename = "test_mp2_wavefunction_hdf5_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    original_wavefunction->to_hdf5(root);
    auto restored_wavefunction = Wavefunction::from_hdf5(root);

    EXPECT_EQ(restored_wavefunction->get_container_type(), "mp2");

    file.close();
  }

  std::remove(filename.c_str());
}

// Test CI coefficients generation from MP2 amplitudes
TEST_F(MP2Test, CICoefficientsGeneration) {
  // Singlet O2 (restricted)
  // Set OMP_NUM_THREADS to 1 for reproducible test results
  int old_omp_threads = omp_get_max_threads();
  omp_set_num_threads(1);

  auto setup = create_o2_hf_setup(1);

  // Restore original OMP_NUM_THREADS value
  omp_set_num_threads(old_omp_threads);

  // Create MP2Container
  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);

  // Test that CI coefficients can be retrieved (lazy evaluation)
  const auto& coefficients = mp2_container->get_coefficients();

  // Verify number of coefficients matches expected for MP2 expansion
  std::visit(
      [](const auto& vec) {
        EXPECT_EQ(vec.size(), 6009)
            << "6009 coefficients should be generated for MP2 expansion";
      },
      coefficients);

  // Test that determinants can be retrieved and count matches
  const auto& determinants = mp2_container->get_active_determinants();
  EXPECT_EQ(determinants.size(), 6009)
      << "6009 determinants should be generated for MP2 expansion";

  // The number of coefficients should match the number of determinants
  std::visit(
      [&determinants](const auto& vec) {
        EXPECT_EQ(static_cast<size_t>(vec.size()), determinants.size())
            << "Number of coefficients should match number of determinants";
      },
      coefficients);

  // Test size() returns the number of determinants
  EXPECT_EQ(mp2_container->size(), determinants.size())
      << "size() should return the number of determinants";
}

// Test CI expansion consistency for MP2
TEST_F(MP2Test, CIExpansionConsistency) {
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Create MP2Container
  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);

  const auto& coefficients = mp2_container->get_coefficients();
  const auto& determinants = mp2_container->get_active_determinants();

  // Verify each determinant can be looked up individually
  for (size_t i = 0; i < std::min(determinants.size(), static_cast<size_t>(10));
       ++i) {
    auto coeff = mp2_container->get_coefficient(determinants[i]);
    std::visit(
        [i, &coefficients](const auto& individual_coeff) {
          using T = std::decay_t<decltype(individual_coeff)>;
          std::visit(
              [i, &individual_coeff](const auto& all_coeffs) {
                using U = std::decay_t<decltype(all_coeffs[0])>;
                if constexpr (std::is_same_v<T, U>) {
                  EXPECT_NEAR(std::abs(individual_coeff),
                              std::abs(all_coeffs[i]), testing::wf_tolerance)
                      << "Individual coefficient lookup should match vector at "
                         "index "
                      << i;
                }
              },
              coefficients);
        },
        coeff);
  }
}

// Test that reference determinant is in MP2 expansion with coefficient 1.0
TEST_F(MP2Test, ReferenceInExpansion) {
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Create MP2Container
  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);

  const auto& determinants = mp2_container->get_active_determinants();

  // The reference determinant should be the first one in the expansion
  const auto& ref_dets = setup.hf_wavefunction->get_total_determinants();
  ASSERT_FALSE(ref_dets.empty())
      << "Reference wavefunction should have determinants";

  // Check that the reference is in the MP2 expansion
  bool found_reference = false;
  for (const auto& det : determinants) {
    for (const auto& ref : ref_dets) {
      if (det.to_string() == ref.to_string()) {
        found_reference = true;
        // The reference should have coefficient 1.0 in MP2
        auto ref_coeff = mp2_container->get_coefficient(det);
        std::visit(
            [](const auto& coeff) {
              // The reference coefficient should be dominant in the MP2
              // expansion. We check that it is greater than 0.9, rather than
              // matching a hardcoded value.
              EXPECT_GT(std::abs(coeff), 0.9);
            },
            ref_coeff);
        break;
      }
    }
    if (found_reference) break;
  }
  EXPECT_TRUE(found_reference)
      << "Reference determinant should be in the MP2 expansion";
}

// Test RDM availability for MP2 container
TEST_F(MP2Test, LazyRDMComputationFromAmplitudes) {
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  // Create MP2Container
  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);

  // CI coefficients should be available (lazy evaluation from amplitudes)
  const auto& coefficients = mp2_container->get_coefficients();
  std::visit(
      [](const auto& vec) {
        EXPECT_GT(vec.size(), 0) << "CI coefficients should be available";
      },
      coefficients);

  // RDMs SHOULD be available via lazy computation from CI coefficients
  EXPECT_TRUE(mp2_container->has_one_rdm_spin_dependent())
      << "MP2 container should have spin-dependent 1-RDM via lazy computation";
  EXPECT_TRUE(mp2_container->has_one_rdm_spin_traced())
      << "MP2 container should have spin-traced 1-RDM via lazy computation";
  EXPECT_TRUE(mp2_container->has_two_rdm_spin_dependent())
      << "MP2 container should have spin-dependent 2-RDM via lazy computation";
  EXPECT_TRUE(mp2_container->has_two_rdm_spin_traced())
      << "MP2 container should have spin-traced 2-RDM via lazy computation";

  // Get RDMs and verify they have reasonable values
  auto [one_rdm_aa, one_rdm_bb] =
      mp2_container->get_active_one_rdm_spin_dependent();

  // Verify 1-RDM has non-zero values (visit each variant separately)
  std::visit(
      [](const auto& mat) {
        EXPECT_GT(mat.norm(), 0.0) << "1-RDM aa should not be zero";
      },
      one_rdm_aa);
  std::visit(
      [](const auto& mat) {
        EXPECT_GT(mat.norm(), 0.0) << "1-RDM bb should not be zero";
      },
      one_rdm_bb);

  // Get number of electrons
  auto [n_alpha, n_beta] = setup.hf_wavefunction->get_active_num_electrons();
  size_t n_electrons = n_alpha + n_beta;

  // Verify 1-RDM trace equals number of electrons
  // Tr(γ) = N
  const auto& rdm1 = std::get<Eigen::MatrixXd>(
      mp2_container->get_active_one_rdm_spin_traced());
  double rdm1_trace = rdm1.trace();
  EXPECT_NEAR(rdm1_trace, static_cast<double>(n_electrons), 1e-6)
      << "1-RDM trace should equal number of electrons. "
      << "Trace: " << rdm1_trace << ", N_electrons: " << n_electrons;

  // Verify 2-RDMs are available and have non-zero values
  auto [two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb] =
      mp2_container->get_active_two_rdm_spin_dependent();
  std::visit(
      [](const auto& rdm) {
        bool has_nonzero = false;
        for (size_t i = 0; i < rdm.size() && !has_nonzero; ++i) {
          if (std::abs(rdm.data()[i]) > 1e-12) {
            has_nonzero = true;
          }
        }
        EXPECT_TRUE(has_nonzero) << "2-RDM aaaa should have non-zero values";
      },
      two_rdm_aaaa);
}

// Test that both amplitudes and CI coefficients are available on MP2Container
TEST_F(MP2Test, AmplitudesAndCICoefficientsAvailable) {
  // Singlet O2 (restricted)
  auto setup = create_o2_hf_setup(1);

  auto mp2_container = std::make_unique<MP2Container>(setup.hf_hamiltonian,
                                                      setup.hf_wavefunction);

  // Test that both T amplitudes and CI coefficients are available
  // First, get CI coefficients (this triggers lazy evaluation)
  const auto& coefficients = mp2_container->get_coefficients();
  const auto& determinants = mp2_container->get_active_determinants();

  // Then, get amplitudes
  auto [t1_aa, t1_bb] = mp2_container->get_t1_amplitudes();
  auto [t2_abab, t2_aaaa, t2_bbbb] = mp2_container->get_t2_amplitudes();

  // Verify all are available
  EXPECT_TRUE(mp2_container->has_t1_amplitudes())
      << "T1 amplitudes should be available";
  EXPECT_TRUE(mp2_container->has_t2_amplitudes())
      << "T2 amplitudes should be available";

  std::visit(
      [](const auto& vec) {
        EXPECT_GT(vec.size(), 0) << "CI coefficients should be non-empty";
      },
      coefficients);

  EXPECT_GT(determinants.size(), 0) << "Determinants should be non-empty";

  // Verify T1 is zero for MP2
  std::visit(
      [](const auto& vec) {
        EXPECT_TRUE(vec.isZero(1e-10))
            << "T1 amplitudes should be zero for MP2";
      },
      t1_aa);
}
