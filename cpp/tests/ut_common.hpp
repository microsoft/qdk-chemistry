// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <vector>

namespace testing {

/// @brief  Tolerance for localization tests
inline static constexpr double localization_tolerance = 1e-5;

/// @brief Tolerance for comparisons with plain text data
inline static constexpr double plain_text_tolerance = 1e-5;

/// @brief Tolerance for JSON comparisons
inline static constexpr double json_tolerance = 1e-10;

///@brief Tolerance for HDF5 comparisons
inline static constexpr double hdf5_tolerance = 1e-10;

/// @brief Tolerance for numerical zeros
inline static constexpr double numerical_zero_tolerance = 1e-12;

/// @brief Tolerance for CI energies
inline static constexpr double ci_energy_tolerance = 1e-8;

/// @brief Tolerance for SCF energies
inline static constexpr double scf_energy_tolerance = 1e-8;

/// @brief Tolerance for MP2 energy match vs pyscf
inline static constexpr double mp2_tolerance = 2e-8;

/// @brief Tolerance for wavefunction operations
inline static constexpr double wf_tolerance = 1e-10;

/// @brief Tolerance for RDMs
inline static constexpr double rdm_tolerance = 1e-6;

///@brief Tolerance lower bound for extremely small values
inline static constexpr double small_value_lower_bound_tolerance = 1e-16;

///@brief Tolerance upper bound for extremely small values
inline static constexpr double small_value_upper_bound_tolerance = 1e-14;

///@brief Tolerance for comparing integral values
inline static constexpr double integral_tolerance = 1e-13;

using namespace qdk::chemistry::data;

/**
 * @brief Creates a random basis set with specified number of atomic orbitals
 * @param n_atomic_orbitals Number of atomic orbitals to generate
 * @param name Name for the basis set (default: "test")
 * @return Shared pointer to a random basis set
 */
inline std::shared_ptr<BasisSet> create_random_basis_set(
    int n_atomic_orbitals = 4, const std::string& name = "test") {
  // Create a simple structure with enough atoms
  int n_atoms = std::max(1, n_atomic_orbitals / 2);
  std::vector<Eigen::Vector3d> coords;
  std::vector<Element> elements;

  for (int i = 0; i < n_atoms; ++i) {
    coords.push_back({static_cast<double>(i), 0.0, 0.0});
    elements.push_back(Element::H);  // Use hydrogen atoms for simplicity
  }

  auto structure = std::make_shared<Structure>(coords, elements);

  // Create shells to match the requested number of atomic orbitals
  std::vector<Shell> shells;
  int functions_created = 0;
  int atom_idx = 0;

  while (functions_created < n_atomic_orbitals) {
    int remaining = n_atomic_orbitals - functions_created;

    // Choose orbital type based on remaining functions needed
    OrbitalType orbital_type;
    int functions_per_shell;

    if (remaining >= 3) {
      orbital_type = OrbitalType::P;
      functions_per_shell = 3;  // px, py, pz
    } else {
      orbital_type = OrbitalType::S;
      functions_per_shell = 1;  // s
    }

    // Create random exponents and coefficients
    Eigen::VectorXd exponents(1);
    Eigen::VectorXd coefficients(1);
    exponents << 1.0 + 0.5 * atom_idx;  // Simple variation
    coefficients << 1.0;

    shells.emplace_back(atom_idx % n_atoms, orbital_type, exponents,
                        coefficients);
    functions_created += functions_per_shell;
    atom_idx++;
  }

  return std::make_shared<BasisSet>(name, shells, structure);
}

/**
 * @brief Creates a simple test orbital object with basic data
 * @param n_basis Number of atomic orbitals
 * @param n_orbitals Number of orbitals
 * @param restricted Whether to create a restricted calculation
 */
inline std::shared_ptr<Orbitals> create_test_orbitals(int n_basis = 3,
                                                      int n_orbitals = 2,
                                                      bool restricted = true) {
  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs.setRandom();

  auto basis_set = create_random_basis_set(n_basis);

  if (restricted) {
    return std::make_shared<Orbitals>(coeffs, std::nullopt, std::nullopt,
                                      basis_set, std::nullopt);
  } else {
    Eigen::MatrixXd coeffs_beta(n_basis, n_orbitals);
    coeffs_beta.setRandom();

    return std::make_shared<Orbitals>(coeffs, coeffs_beta, std::nullopt,
                                      std::nullopt, std::nullopt, basis_set,
                                      std::nullopt);
  }
}

/**
 * @brief Creates a test orbital object with energies and occupations
 */
inline Orbitals create_test_orbitals_with_properties(int n_basis = 3,
                                                     int n_orbitals = 2) {
  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs.setRandom();

  Eigen::VectorXd energies(n_orbitals);
  energies.setLinSpaced(n_orbitals, -1.0, 0.0);

  auto basis_set = create_random_basis_set(n_basis);

  return Orbitals(coeffs, std::make_optional(energies), std::nullopt, basis_set,
                  std::nullopt);
}

/**
 * @brief Creates a new Orbitals object from an existing one with active space
 * information
 * @param existing The existing orbitals object
 * @param active_indices The active space indices
 */
inline std::shared_ptr<Orbitals> with_active_space(
    const std::shared_ptr<Orbitals> existing,
    const std::vector<size_t>& active_indices,
    const std::vector<size_t>& inactive_indices) {
  // Get existing data
  auto [alpha_coeffs, beta_coeffs] = existing->get_coefficients();

  std::optional<Eigen::VectorXd> alpha_energies, beta_energies;
  if (existing->has_energies()) {
    auto [e_a, e_b] = existing->get_energies();
    alpha_energies = e_a;
    beta_energies = e_b;
  }

  std::optional<Eigen::MatrixXd> ao_overlap;
  if (existing->has_overlap_matrix()) {
    ao_overlap = existing->get_overlap_matrix();
  }

  std::shared_ptr<BasisSet> basis_set;
  if (existing->has_basis_set()) {
    basis_set = existing->get_basis_set();
  }

  if (existing->is_restricted()) {
    return std::make_shared<Orbitals>(
        alpha_coeffs, alpha_energies, ao_overlap, basis_set,
        std::make_tuple(std::move(active_indices),
                        std::move(inactive_indices)));
  } else {
    return std::make_shared<Orbitals>(
        alpha_coeffs, beta_coeffs, alpha_energies, beta_energies, ao_overlap,
        basis_set,
        std::make_tuple(active_indices,  // Same for alpha/beta
                        active_indices, inactive_indices, inactive_indices));
  }
}

/**
 * @brief Creates a water molecule structure.
 *
 * Crawford geometry
 */
inline std::shared_ptr<Structure> create_water_structure() {
  std::vector<Eigen::Vector3d> coords = {
      {0.000000000, -0.0757918436, 0.000000000000},
      {0.866811829, 0.6014357793, -0.000000000000},
      {-0.866811829, 0.6014357793, -0.000000000000}};

  // Convert to Bohr
  for (auto& coord : coords) {
    coord *= qdk::chemistry::constants::angstrom_to_bohr;
  }

  std::vector<Element> elements = {qdk::chemistry::data::Element::O,
                                   qdk::chemistry::data::Element::H,
                                   qdk::chemistry::data::Element::H};

  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Creates a Li structure
 */
inline std::shared_ptr<Structure> create_li_structure() {
  std::vector<Eigen::Vector3d> coords = {
      {0.000000000, 0.000000000, 0.000000000}};

  std::vector<Element> elements = {qdk::chemistry::data::Element::Li};

  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Creates an O2 structure
 */
inline std::shared_ptr<Structure> create_o2_structure() {
  std::vector<Eigen::Vector3d> coords = {
      {0.000000000, 0.000000000, 0.000000000},
      {0.000000000, 0.000000000, 1.208000000}};

  // Convert to Bohr
  for (auto& coord : coords) {
    coord *= qdk::chemistry::constants::angstrom_to_bohr;
  }

  std::vector<Element> elements = {qdk::chemistry::data::Element::O,
                                   qdk::chemistry::data::Element::O};

  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Create a stretched N2 structure
 */
inline std::shared_ptr<Structure> create_stretched_n2_structure() {
  std::vector<Eigen::Vector3d> coords = {
      {0.000000000, 0.000000000, 0.000000000},
      {0.000000000, 0.000000000, 3.7794519772}};

  std::vector<Element> elements = {qdk::chemistry::data::Element::N,
                                   qdk::chemistry::data::Element::N};

  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Create a stretched N2 structure
 * @param distance_angstrom Distance between nitrogen atoms in Angstrom
 */
inline std::shared_ptr<Structure> create_stretched_n2_structure(
    double distance_angstrom) {
  std::vector<Eigen::Vector3d> coords = {
      {0.000000000, 0.000000000, 0.000000000},
      {distance_angstrom * qdk::chemistry::constants::angstrom_to_bohr,
       0.000000000, 0.000000000}};

  std::vector<Element> elements = {qdk::chemistry::data::Element::N,
                                   qdk::chemistry::data::Element::N};

  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Create a BN+ cation structure
 */
inline std::shared_ptr<Structure> create_bn_plus_structure() {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.2765}};

  // Convert to Bohr
  for (auto& coord : coords) {
    coord *= qdk::chemistry::constants::angstrom_to_bohr;
  }

  std::vector<Element> elements = {qdk::chemistry::data::Element::B,
                                   qdk::chemistry::data::Element::N};

  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Create a single O structure
 */
inline std::shared_ptr<Structure> create_oxygen_structure() {
  std::vector<Eigen::Vector3d> coords = {
      {0.00000000000, 0.00000000000, 0.00000000000}};
  std::vector<Element> elements = {qdk::chemistry::data::Element::O};
  return std::make_shared<Structure>(coords, elements);
}

/**
 * @brief Creates an AgH (silver hydride) structure
 */
inline std::shared_ptr<Structure> create_agh_structure() {
  std::vector<Eigen::Vector3d> coords = {
      {0.000000000, 0.000000000, 0.000000000},
      {0.000000000, 0.000000000, 1.617000000}};

  // Convert to Bohr
  for (auto& coord : coords) {
    coord *= qdk::chemistry::constants::angstrom_to_bohr;
  }

  std::vector<Element> elements = {qdk::chemistry::data::Element::Ag,
                                   qdk::chemistry::data::Element::H};

  return std::make_shared<Structure>(coords, elements);
}

}  // namespace testing
