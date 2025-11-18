// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class SlaterDeterminantContainer
 * @brief Wavefunction container representing a single Slater determinant
 *
 * This class represents the simplest wavefunction - a single Slater determinant
 * with coefficient 1.0. It provides efficient storage and computation for
 * single-determinant wavefunctions such as Hartree-Fock reference states.
 */
class SlaterDeterminantContainer : public WavefunctionContainer {
 public:
  // Use real values for single determinants (coefficient is always 1.0)
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Constructs a single Slater determinant wavefunction
   *
   * This constructor validates that:
   * - The configuration has sufficient orbital capacity for the active space
   * - Any orbitals beyond the active space size are unoccupied
   *
   * Note: Configurations only represent the active space, not the full orbital
   * space. Inactive and virtual orbitals are
   * not included in the configuration representation.
   *
   * @param det The single determinant configuration (representing active space
   * only)
   * @param orbitals Shared pointer to orbital basis set
   * @param type Type of wavefunction (default: SelfDual)
   * @throws std::invalid_argument If validation fails
   */
  SlaterDeterminantContainer(
      const Configuration& det, std::shared_ptr<Orbitals> orbitals,
      WavefunctionType type = WavefunctionType::SelfDual);

  ~SlaterDeterminantContainer() override = default;

  /**
   * @brief Create a deep copy of this container
   */
  std::unique_ptr<WavefunctionContainer> clone() const override;

  /**
   * @brief Get reference to orbital basis set
   * @return Shared pointer to orbitals
   */
  std::shared_ptr<Orbitals> get_orbitals() const override;

  /**
   * @brief Get all coefficients (always a single coefficient of 1.0)
   * @return Vector of all coefficients (real or complex)
   */
  const VectorVariant& get_coefficients() const override;

  /**
   * @brief Get coefficient for a specific determinant
   *
   * The configuration is expected to be a determinant describing only
   * the wavefunction's active space.
   *
   * For single determinant: returns 1.0 if det matches, 0.0 otherwise.
   *
   * @param det Configuration/determinant to get coefficient for
   * @return Coefficient of the determinant
   */
  ScalarVariant get_coefficient(const Configuration& det) const override;

  /**
   * @brief Get all determinants in the wavefunction
   */
  const DeterminantVector& get_active_determinants() const override;

  /**
   * @brief Get number of determinants
   */
  size_t size() const override;

  /**
   * @brief Calculate overlap with another wavefunction
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  /**
   * @brief Calculate norm of the wavefunction
   */
  double norm() const override;

  /**
   * @brief Get spin-dependent one-particle RDMs for active orbitals only
   */
  std::tuple<const MatrixVariant&, const MatrixVariant&>
  get_active_one_rdm_spin_dependent() const override;

  /**
   * @brief Get spin-dependent two-particle RDMs for active orbitals only
   */
  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_active_two_rdm_spin_dependent() const override;

  /**
   * @brief Get spin-traced one-particle RDM for active orbitals only
   */
  const MatrixVariant& get_active_one_rdm_spin_traced() const override;

  /**
   * @brief Get spin-traced two-particle RDM for active orbitals only
   */
  const VectorVariant& get_active_two_rdm_spin_traced() const override;

  /**
   * @brief Calculate single orbital entropies for active orbitals only
   */
  Eigen::VectorXd get_single_orbital_entropies() const override;

  /**
   * @brief Get total number of alpha and beta electrons (active + inactive)
   * @return Pair of (n_alpha_total, n_beta_total) electrons
   */
  std::pair<size_t, size_t> get_total_num_electrons() const override;

  /**
   * @brief Get number of active alpha and beta electrons
   * @return Pair of (n_alpha_active, n_beta_active) electrons
   */
  std::pair<size_t, size_t> get_active_num_electrons() const override;

  /**
   * @brief Get orbital occupations for all orbitals (total = active + inactive
   * + virtual)
   * @return Pair of (alpha_occupations_total, beta_occupations_total)
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_total_orbital_occupations()
      const override;

  /**
   * @brief Get orbital occupations for active orbitals only
   * @return Pair of (alpha_active_occupations, beta_active_occupations)
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_active_orbital_occupations()
      const override;

  /**
   * @brief Check if spin-dependent one-particle RDMs for active orbitals are
   * available
   * @return True if available
   */
  bool has_one_rdm_spin_dependent() const override;

  /**
   * @brief Check if spin-traced one-particle RDM for active orbitals is
   * available
   * @return True if available
   */
  bool has_one_rdm_spin_traced() const override;

  /**
   * @brief Check if spin-dependent two-particle RDMs for active orbitals are
   * available
   * @return True if available
   */
  bool has_two_rdm_spin_dependent() const override;

  /**
   * @brief Check if spin-traced two-particle RDM for active orbitals is
   * available
   * @return True if available
   */
  bool has_two_rdm_spin_traced() const override;

  /**
   * @brief Check if a determinant matches the stored one
   */
  bool contains_determinant(const Configuration& det) const;

  /**
   * @brief Clear cached data to release memory
   *
   * Clears the following cached data:
   * - Determinant vector cache (_determinant_vector_cache)
   * - One-particle RDMs (spin-traced and spin-dependent)
   * - Two-particle RDMs (spin-traced and spin-dependent)
   *
   * This is particularly useful for freeing memory when the cached
   * RDMs are no longer needed.
   */
  void clear_caches() const override;

  /**
   * @brief Convert container to JSON format
   * @return JSON object containing container data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Load container from JSON format
   * @param j JSON object containing container data
   * @return Unique pointer to SD container created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<SlaterDeterminantContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Convert container to HDF5 group
   * @param group HDF5 group to write container data to
   * @throws std::runtime_error if HDF5 I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load container from HDF5 group
   * @param group HDF5 group containing container data
   * @return Unique pointer to SD container created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::unique_ptr<SlaterDeterminantContainer> from_hdf5(
      H5::Group& group);

  /**
   * @brief Get container type identifier for serialization
   * @return String "sd"
   */
  std::string get_container_type() const override;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return Always false for Slater determinants (coefficients are unity)
   */
  bool is_complex() const override;

 private:
  // Single determinant - optimized storage for exactly one determinant
  const Configuration _determinant;

  // Orbital information (stored directly, not via ConfigurationSet)
  std::shared_ptr<Orbitals> _orbitals;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  // Lazy-initialized determinant vector for interface compatibility
  mutable std::unique_ptr<DeterminantVector> _determinant_vector_cache;

  // Cached RDMs for expensive computations
  mutable std::shared_ptr<MatrixVariant> _one_rdm_spin_traced = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_traced = nullptr;
  mutable std::shared_ptr<MatrixVariant> _one_rdm_spin_dependent_aa = nullptr;
  mutable std::shared_ptr<MatrixVariant> _one_rdm_spin_dependent_bb = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_dependent_aaaa = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_dependent_aabb = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_dependent_bbbb = nullptr;

  // Cached coefficient vector for interface compatibility
  mutable VectorVariant _coefficient_vector;
};
}  // namespace qdk::chemistry::data
