// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class ActiveSpaceTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class TestActiveSpaceSelector : public ActiveSpaceSelector {
 public:
  std::string name() const override { return "_test_active_space_selector"; }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction) const override {
    auto orbitals = wavefunction->get_orbitals();
    // Dummy implementation for testing - create new Orbitals with active space
    // info
    if (orbitals->is_unrestricted()) {
      // Unrestricted case - get separate alpha and beta data
      auto coeffs = orbitals->get_coefficients();

      std::optional<Eigen::VectorXd> energies_alpha, energies_beta;
      if (orbitals->has_energies()) {
        auto energies = orbitals->get_energies();
        energies_alpha = energies.first;
        energies_beta = energies.second;
      }

      std::optional<Eigen::MatrixXd> ao_overlap;
      if (orbitals->has_overlap_matrix()) {
        ao_overlap = orbitals->get_overlap_matrix();
      }

      std::shared_ptr<BasisSet> basis_set;
      if (orbitals->has_basis_set()) {
        basis_set = orbitals->get_basis_set();
      }

      std::vector<size_t> active_alpha = {{0, 1, 2}};
      std::vector<size_t> active_beta = {{0, 1, 2}};

      auto new_orbitals = std::make_shared<Orbitals>(
          coeffs.first, coeffs.second, energies_alpha, energies_beta,
          ao_overlap, basis_set,
          std::make_tuple(std::move(active_alpha), std::move(active_beta),
                          std::vector<size_t>{}, std::vector<size_t>{}));
      return qdk::chemistry::algorithms::detail::new_wavefunction(wavefunction,
                                                                  new_orbitals);
    } else {
      // Restricted case - use first coefficient matrix only
      auto coeffs = orbitals->get_coefficients();

      std::optional<Eigen::VectorXd> energies;
      if (orbitals->has_energies()) {
        energies = orbitals->get_energies().first;
      }

      std::optional<Eigen::MatrixXd> ao_overlap;
      if (orbitals->has_overlap_matrix()) {
        ao_overlap = orbitals->get_overlap_matrix();
      }

      std::shared_ptr<BasisSet> basis_set;
      if (orbitals->has_basis_set()) {
        basis_set = orbitals->get_basis_set();
      }

      std::vector<size_t> active_indices = {0, 1, 2};

      auto new_orbitals = std::make_shared<Orbitals>(
          coeffs.first, energies, ao_overlap, basis_set,
          std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
      return qdk::chemistry::algorithms::detail::new_wavefunction(wavefunction,
                                                                  new_orbitals);
    }
  }
};

TEST_F(ActiveSpaceTest, ActiveSpaceSelector_MetaData) {
  auto selector = ActiveSpaceSelectorFactory::create();
  EXPECT_NO_THROW({ auto settings = selector->settings(); });
}

TEST_F(ActiveSpaceTest, Factory) {
  auto available_selectors = ActiveSpaceSelectorFactory::available();
  EXPECT_EQ(available_selectors.size(), 4);
  EXPECT_TRUE(std::find(available_selectors.begin(), available_selectors.end(),
                        "qdk_valence") != available_selectors.end());
  EXPECT_TRUE(std::find(available_selectors.begin(), available_selectors.end(),
                        "qdk_occupation") != available_selectors.end());
  EXPECT_TRUE(std::find(available_selectors.begin(), available_selectors.end(),
                        "qdk_autocas") != available_selectors.end());
  EXPECT_TRUE(std::find(available_selectors.begin(), available_selectors.end(),
                        "qdk_autocas_eos") != available_selectors.end());
  EXPECT_THROW(ActiveSpaceSelectorFactory::create("nonexistent_selector"),
               std::runtime_error);
  EXPECT_NO_THROW(ActiveSpaceSelectorFactory::register_instance(
      []() -> ActiveSpaceSelectorFactory::return_type {
        return std::make_unique<TestActiveSpaceSelector>();
      }));
  EXPECT_THROW(ActiveSpaceSelectorFactory::register_instance(
                   []() -> ActiveSpaceSelectorFactory::return_type {
                     return std::make_unique<TestActiveSpaceSelector>();
                   }),
               std::runtime_error);

  // Test unregister_instance
  // First test unregistering a non-existent key (should return false)
  EXPECT_FALSE(
      ActiveSpaceSelectorFactory::unregister_instance("nonexistent_key"));

  // Test unregistering an existing key (should return true)
  EXPECT_TRUE(ActiveSpaceSelectorFactory::unregister_instance(
      "_test_active_space_selector"));

  // Test unregistering the same key again (should return false since it's
  // already removed)
  EXPECT_FALSE(ActiveSpaceSelectorFactory::unregister_instance(
      "_test_active_space_selector"));
}

TEST_F(ActiveSpaceTest, Occupation) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_occupation");
  EXPECT_NO_THROW({ auto settings = selector->settings(); });

  // Create data for orbitals
  auto coeffs = Eigen::MatrixXd::Identity(4, 4);

  Eigen::VectorXd energies(4);
  energies << -0.5, -0.3, 0.1, 0.2;

  // Create orbitals with constructor
  auto basis = testing::create_random_basis_set(4);
  Orbitals orbitals(coeffs, std::make_optional(energies), std::nullopt, basis);

  // Create a wavefunction with the orbitals
  auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
  auto wfn = std::make_shared<Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(Configuration("2ud0"),
                                                   orbitals_ptr));

  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();

  EXPECT_EQ(indices_alpha, std::vector<size_t>({1, 2}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({1, 2}));
}

TEST_F(ActiveSpaceTest, Occupation_EdgeCase) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_occupation");

  // Unrestricted orbitals should throw an error
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setRandom();
        Eigen::MatrixXd coeffs2(4, 4);
        coeffs2.setIdentity();
        // Create unrestricted orbitals
        auto basis = testing::create_random_basis_set(4);
        Orbitals orbitals(coeffs, coeffs2, std::nullopt, std::nullopt,
                          std::nullopt, basis, std::nullopt);

        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2000"),
                                                         orbitals_ptr));
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);
}

TEST_F(ActiveSpaceTest, Valence) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  EXPECT_NO_THROW({ auto settings = selector->settings(); });

  // Create data for orbitals
  Eigen::MatrixXd coeffs(4, 4);
  coeffs.setIdentity();

  Eigen::VectorXd energies(4);
  energies << -0.5, -0.3, 0.1, 0.2;

  // Create orbitals with constructor
  auto basis = testing::create_random_basis_set(4);
  Orbitals orbitals(coeffs, std::make_optional(energies), std::nullopt, basis,
                    std::nullopt);

  auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
  auto wfn = std::make_shared<Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(Configuration("2ud0"),
                                                   orbitals_ptr));

  selector->settings().set("num_active_electrons", 2);
  selector->settings().set("num_active_orbitals", 2);
  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();

  EXPECT_EQ(indices_alpha, std::vector<size_t>({1, 2}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({1, 2}));
}

TEST_F(ActiveSpaceTest, Valence_EdgeCase) {
  // Unrestricted orbitals should throw an error
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setRandom();
        Eigen::MatrixXd coeffs2(4, 4);
        coeffs2.setIdentity();
        // Create unrestricted orbitals
        auto basis = testing::create_random_basis_set(4);
        Eigen::VectorXd energies(4);
        energies[0] = -0.5;
        energies[1] = -0.3;
        energies[2] = 0.1;
        energies[3] = 0.2;
        Orbitals orbitals(coeffs, coeffs2, energies, energies, std::nullopt,
                          basis, std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("dd00"),
                                                         orbitals_ptr));
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);

  // Invalid number of electrons
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        // Create restricted orbitals
        auto basis = testing::create_random_basis_set(4);
        Eigen::VectorXd energies(4);
        energies[0] = -0.5;
        energies[1] = -0.3;
        energies[2] = 0.1;
        energies[3] = 0.2;
        Orbitals orbitals(coeffs, energies, std::nullopt, basis, std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals_ptr));
        selector->settings().set("num_active_electrons", -1);
        selector->settings().set("num_active_orbitals", 2);
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);

  // Invalid number of orbitals
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        auto basis = testing::create_random_basis_set(4);
        Eigen::VectorXd energies(4);
        energies[0] = -0.5;
        energies[1] = -0.3;
        energies[2] = 0.1;
        energies[3] = 0.2;
        Orbitals orbitals(coeffs, energies, std::nullopt, basis, std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals_ptr));
        selector->settings().set("num_active_electrons", 2);
        selector->settings().set("num_active_orbitals", -1);
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);

  // Too many active orbitals
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        auto basis = testing::create_random_basis_set(4);
        Eigen::VectorXd energies(4);
        energies[0] = -0.5;
        energies[1] = -0.3;
        energies[2] = 0.1;
        energies[3] = 0.2;
        Orbitals orbitals(coeffs, energies, std::nullopt, basis, std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals_ptr));
        selector->settings().set("num_active_electrons", 2);
        selector->settings().set("num_active_orbitals", 5);
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);

  // Too many active electrons
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        auto basis = testing::create_random_basis_set(4);
        Eigen::VectorXd energies(4);
        energies[0] = -0.5;
        energies[1] = -0.3;
        energies[2] = 0.1;
        energies[3] = 0.2;
        Orbitals orbitals(coeffs, energies, std::nullopt, basis, std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2000"),
                                                         orbitals_ptr));
        selector->settings().set("num_active_electrons", 3);
        selector->settings().set("num_active_orbitals", 4);
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);

  // Odd number of inactive electrons
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        auto basis = testing::create_random_basis_set(4);
        Eigen::VectorXd energies(4);
        energies[0] = -0.5;
        energies[1] = -0.3;
        energies[2] = 0.1;
        energies[3] = 0.2;
        Orbitals orbitals(coeffs, energies, std::nullopt, basis, std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals_ptr));
        selector->settings().set("num_active_electrons", 3);
        selector->settings().set("num_active_orbitals", 4);
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);

  // No orbital energies
  EXPECT_THROW(
      {
        auto selector = ActiveSpaceSelectorFactory::create("valence");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        auto basis = testing::create_random_basis_set(4);
        Orbitals orbitals(coeffs, std::nullopt, std::nullopt, basis,
                          std::nullopt);
        auto orbitals_ptr = std::make_shared<Orbitals>(orbitals);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals_ptr));
        selector->settings().set("num_active_electrons", 2);
        selector->settings().set("num_active_orbitals", 2);
        auto result_wavefunction = selector->run(wfn);
      },
      std::runtime_error);
}

// Test for the presence of active space on input orbitals
TEST_F(ActiveSpaceTest, ActiveSpaceAlreadySet) {
  // Occupation selector - should work if active space already set
  EXPECT_NO_THROW({
    auto selector = ActiveSpaceSelectorFactory::create("qdk_occupation");
    Eigen::MatrixXd coeffs(4, 4);
    coeffs.setIdentity();
    std::vector<size_t> active_indices({0, 1});
    auto basis = testing::create_random_basis_set(4);
    auto orbitals = std::make_shared<Orbitals>(
        coeffs, std::nullopt, std::nullopt, basis,
        std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
    auto wfn = std::make_shared<Wavefunction>(
        std::make_unique<SlaterDeterminantContainer>(Configuration("dd00"),
                                                     orbitals));
    auto result_wavefunction = selector->run(wfn);
  });

  // Valence selector - should work even with active space already set
  EXPECT_NO_THROW({
    auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
    Eigen::MatrixXd coeffs(4, 4);
    coeffs.setIdentity();
    std::vector<size_t> active_indices({0, 1});
    auto basis = testing::create_random_basis_set(4);
    Eigen::VectorXd energies(4);
    energies[0] = -0.5;
    energies[1] = -0.3;
    energies[2] = 0.1;
    energies[3] = 0.2;
    auto orbitals = std::make_shared<Orbitals>(
        coeffs, energies, std::nullopt, basis,
        std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
    auto wfn = std::make_shared<Wavefunction>(
        std::make_unique<SlaterDeterminantContainer>(Configuration("dd00"),
                                                     orbitals));
    selector->settings().set("num_active_electrons", 2);
    selector->settings().set("num_active_orbitals", 2);
    auto result_wavefunction = selector->run(wfn);
  });
}

// Mock container for MockWavefunction
class MockWavefunctionContainer : public WavefunctionContainer {
 public:
  MockWavefunctionContainer(std::shared_ptr<Orbitals> orbitals,
                            const Eigen::VectorXd& entropies)
      : WavefunctionContainer(WavefunctionType::SelfDual),
        orbitals_(orbitals),
        entropies_(entropies) {
    // Create a single determinant with the correct length
    size_t num_orbitals = orbitals->get_num_molecular_orbitals();
    std::string det_string(num_orbitals, '0');
    // Set first few orbitals to doubly occupied (matching the 4 electrons: 2
    // alpha + 2 beta)
    for (size_t i = 0; i < (num_orbitals / 2); ++i) {
      det_string[i] = '2';
    }
    determinants_.push_back(Configuration(det_string));

    // Initialize coefficients with a single value of 1.0
    Eigen::VectorXd coeff_vec(1);
    coeff_vec(0) = 1.0;
    coefficients_ = coeff_vec;
  }

  std::unique_ptr<WavefunctionContainer> clone() const override {
    return std::make_unique<MockWavefunctionContainer>(orbitals_, entropies_);
  }

  std::shared_ptr<Orbitals> get_orbitals() const override { return orbitals_; }

  const VectorVariant& get_coefficients() const override {
    return coefficients_;
  }

  ScalarVariant get_coefficient(const Configuration& det) const override {
    return 1.0;
  }

  const DeterminantVector& get_active_determinants() const override {
    return determinants_;
  }

  size_t size() const override { return 1; }

  ScalarVariant overlap(const WavefunctionContainer& other) const override {
    return 1.0;
  }

  double norm() const override { return 1.0; }

  bool has_one_rdm_spin_dependent() const override { return false; }

  bool has_one_rdm_spin_traced() const override { return false; }

  bool has_two_rdm_spin_dependent() const override { return false; }

  bool has_two_rdm_spin_traced() const override { return false; }

  bool has_single_orbital_entropies() const override { return true; }

  std::tuple<const MatrixVariant&, const MatrixVariant&>
  get_active_one_rdm_spin_dependent() const override {
    throw std::runtime_error("Not implemented");
  }

  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_active_two_rdm_spin_dependent() const override {
    throw std::runtime_error("Not implemented");
  }

  const MatrixVariant& get_active_one_rdm_spin_traced() const override {
    throw std::runtime_error("Not implemented");
  }

  const VectorVariant& get_active_two_rdm_spin_traced() const override {
    throw std::runtime_error("Not implemented");
  }

  Eigen::VectorXd get_single_orbital_entropies() const override {
    return entropies_;
  }

  std::pair<size_t, size_t> get_total_num_electrons() const override {
    return std::make_pair(2, 2);
  }

  std::pair<size_t, size_t> get_active_num_electrons() const override {
    return std::make_pair(2, 2);
  }

  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_total_orbital_occupations()
      const override {
    throw std::runtime_error("Not implemented");
  }

  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_active_orbital_occupations()
      const override {
    throw std::runtime_error("Not implemented");
  }

  void clear_caches() const override {}

  nlohmann::json to_json() const override { return nlohmann::json{}; }

  void to_hdf5(H5::Group& group) const override {}

  std::string get_container_type() const override { return "mock"; }

  bool is_complex() const override { return false; }

 private:
  std::shared_ptr<Orbitals> orbitals_;
  Eigen::VectorXd entropies_;
  VectorVariant coefficients_;
  DeterminantVector determinants_;
};

// Mock for wavefunction that takes entropies, orbital indices, and includes
// orbitals
class MockWavefunction : public Wavefunction {
 public:
  // Constructor that takes entropies, orbital indices and creates default
  // orbitals
  MockWavefunction(const Eigen::VectorXd& entropies,
                   const std::vector<size_t>& orbital_indices)
      : Wavefunction(std::make_unique<MockWavefunctionContainer>(
            testing::with_active_space(
                testing::create_test_orbitals(
                    *std::max_element(orbital_indices.begin(),
                                      orbital_indices.end()) +
                        1,
                    *std::max_element(orbital_indices.begin(),
                                      orbital_indices.end()) +
                        1),
                orbital_indices, std::vector<size_t>{}),
            entropies)) {}
};

class WavefunctionActiveSpaceTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Autocas tests
TEST_F(WavefunctionActiveSpaceTest, Autocas) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas");
  EXPECT_NO_THROW({ auto settings = selector->settings(); });
  Eigen::VectorXd entropies(10);
  entropies << 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.02, 0.02, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  EXPECT_EQ(indices_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
}

TEST_F(WavefunctionActiveSpaceTest, AutocasSinglereference) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas");

  Eigen::VectorXd entropies(10);
  entropies << 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.02, 0.02, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  EXPECT_EQ(indices_alpha, std::vector<size_t>({}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({}));
}

TEST_F(WavefunctionActiveSpaceTest, AutocasOnlyHighEntropies) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas");

  // all entropies below threshold
  Eigen::VectorXd entropies(10);
  entropies << 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9;
  std::vector<size_t> orbital_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto indices =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices_alpha = indices.first;
  auto indices_beta = indices.second;
  EXPECT_EQ(indices_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST_F(WavefunctionActiveSpaceTest, AutocasEntropyThreshold) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas");
  selector->settings().set("entropy_threshold", 0.91);

  // all entropies below threshold
  Eigen::VectorXd entropies(10);
  entropies << 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.02, 0.02, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto indices =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices_alpha = indices.first;
  auto indices_beta = indices.second;
  EXPECT_EQ(indices_alpha, std::vector<size_t>({}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({}));

  // only two entropies above threshold
  selector = ActiveSpaceSelectorFactory::create("qdk_autocas");
  selector->settings().set("entropy_threshold", 0.5);
  result_wavefunction = selector->run(wfn);
  auto indices2 =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices2_alpha = indices2.first;
  auto indices2_beta = indices2.second;
  EXPECT_EQ(indices2_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices2_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));

  // more normal threshold
  selector = ActiveSpaceSelectorFactory::create("qdk_autocas");
  selector->settings().set("entropy_threshold", 0.1);
  result_wavefunction = selector->run(wfn);
  auto indices3 =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices3_alpha = indices3.first;
  auto indices3_beta = indices3.second;
  EXPECT_EQ(indices3_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices3_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));

  // no plateau for very low entropies
  selector = ActiveSpaceSelectorFactory::create("qdk_autocas");
  selector->settings().set("entropy_threshold", 0.001);
  result_wavefunction = selector->run(wfn);
  auto indices4 =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices4_alpha = indices4.first;
  auto indices4_beta = indices4.second;
  EXPECT_EQ(indices4_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices4_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
}

TEST_F(WavefunctionActiveSpaceTest, AutocasEosEntropyThreshold) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  selector->settings().set("entropy_threshold", 0.91);
  selector->settings().set("normalize_entropies", false);

  // all entropies below threshold
  Eigen::VectorXd entropies(10);
  entropies << 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.02, 0.02, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto indices =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices_alpha = indices.first;
  auto indices_beta = indices.second;
  EXPECT_EQ(indices_alpha, std::vector<size_t>({}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({}));

  // only two entropies above threshold
  selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  selector->settings().set("normalize_entropies", false);
  selector->settings().set("entropy_threshold", 0.5);
  result_wavefunction = selector->run(wfn);
  auto indices2 =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices2_alpha = indices2.first;
  auto indices2_beta = indices2.second;
  EXPECT_EQ(indices2_alpha, std::vector<size_t>({4, 5}));
  EXPECT_EQ(indices2_beta, std::vector<size_t>({4, 5}));

  // more normal threshold
  selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  selector->settings().set("normalize_entropies", false);
  selector->settings().set("entropy_threshold", 0.1);
  result_wavefunction = selector->run(wfn);
  auto indices3 =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices3_alpha = indices3.first;
  auto indices3_beta = indices3.second;
  EXPECT_EQ(indices3_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices3_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));

  // no plateau for very low entropies
  selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  selector->settings().set("normalize_entropies", false);
  selector->settings().set("entropy_threshold", 0.001);
  result_wavefunction = selector->run(wfn);
  auto indices4 =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  auto indices4_alpha = indices4.first;
  auto indices4_beta = indices4.second;
  EXPECT_EQ(indices4_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices4_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
}

TEST_F(WavefunctionActiveSpaceTest, AutocasNonContinuous) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas");
  Eigen::VectorXd entropies(10);
  entropies << 0.4, 0.02, 0.02, 0.4, 0.9, 0.9, 0.02, 0.4, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {4, 8, 10, 12, 15, 23, 29, 34, 37, 38};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  EXPECT_EQ(indices_alpha, std::vector<size_t>({4, 12, 15, 23, 34}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({4, 12, 15, 23, 34}));
}

// Entropy-based active space selector tests
TEST_F(WavefunctionActiveSpaceTest, Entropy) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  EXPECT_NO_THROW({ auto settings = selector->settings(); });
  Eigen::VectorXd entropies(10);
  entropies << 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.02, 0.02, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();

  EXPECT_EQ(indices_alpha, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({0, 1, 2, 3, 4, 5}));
}

TEST_F(WavefunctionActiveSpaceTest, EntropyNonContinuous) {
  auto selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  Eigen::VectorXd entropies(10);
  entropies << 0.4, 0.02, 0.02, 0.4, 0.9, 0.9, 0.02, 0.4, 0.02, 0.02;
  std::vector<size_t> orbital_indices = {4, 8, 10, 12, 15, 23, 29, 34, 37, 38};
  auto wfn = std::make_shared<MockWavefunction>(
      MockWavefunction(entropies, orbital_indices));
  auto result_wavefunction = selector->run(wfn);
  auto [indices_alpha, indices_beta] =
      result_wavefunction->get_orbitals()->get_active_space_indices();
  EXPECT_EQ(indices_alpha, std::vector<size_t>({4, 12, 15, 23, 34}));
  EXPECT_EQ(indices_beta, std::vector<size_t>({4, 12, 15, 23, 34}));
}
