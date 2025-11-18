// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/configuration_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

class SciWavefunctionContainer : public WavefunctionContainer {
 public:
  // Use real values for default FCI
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Constructs a wavefunction with full reduced density matrix data
   *
   * @param coeffs The vector of CI coefficients (can be real or complex)
   * @param dets The vector of determinants
   * @param orbitals Shared pointer to orbital basis set
   * @param type The type of wavefunction
   */
  SciWavefunctionContainer(const VectorVariant& coeffs,
                           const DeterminantVector& dets,
                           std::shared_ptr<Orbitals> orbitals,
                           WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a wavefunction with full reduced density matrix (RDM)
   * data
   *
   * @param coeffs The vector of CI coefficients (can be real or complex)
   * @param dets The vector of determinants
   * @param orbitals Shared pointer to orbital basis set
   * @param one_rdm_spin_traced Spin-traced 1-RDM for active orbitals (optional)
   * @param two_rdm_spin_traced Spin-traced 2-RDM for active orbitals (optional)
   * @param type The type of wavefunction
   */
  SciWavefunctionContainer(
      const VectorVariant& coeffs, const DeterminantVector& dets,
      std::shared_ptr<Orbitals> orbitals,
      const std::optional<MatrixVariant>& one_rdm_spin_traced,
      const std::optional<VectorVariant>& two_rdm_spin_traced,
      WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a wavefunction with full reduced density matrix (RDM)
   * data
   *
   * @param coeffs The vector of CI coefficients (can be real or complex)
   * @param dets The vector of determinants
   * @param orbitals Shared pointer to orbital basis set
   * @param one_rdm_spin_traced Spin-traced 1-RDM for active orbitals (optional)
   * @param one_rdm_aa Alpha-alpha block of 1-RDM for active orbitals (optional)
   * @param one_rdm_bb Beta-beta block of 1-RDM for active orbitals (optional)
   * @param two_rdm_spin_traced Spin-traced 2-RDM for active orbitals (optional)
   * @param two_rdm_aabb Alpha-alpha-beta-beta block of 2-RDM for active
   * orbitals (optional)
   * @param two_rdm_aaaa Alpha-alpha-alpha-alpha block of 2-RDM for active
   * orbitals (optional)
   * @param two_rdm_bbbb Beta-beta-beta-beta block of 2-RDM for active orbitals
   * (optional)
   * @param type The type of wavefunction
   */
  SciWavefunctionContainer(
      const VectorVariant& coeffs, const DeterminantVector& dets,
      std::shared_ptr<Orbitals> orbitals,
      const std::optional<MatrixVariant>& one_rdm_spin_traced,
      const std::optional<MatrixVariant>& one_rdm_aa,
      const std::optional<MatrixVariant>& one_rdm_bb,
      const std::optional<VectorVariant>& two_rdm_spin_traced,
      const std::optional<VectorVariant>& two_rdm_aabb,
      const std::optional<VectorVariant>& two_rdm_aaaa,
      const std::optional<VectorVariant>& two_rdm_bbbb,
      WavefunctionType type = WavefunctionType::SelfDual);

  /** @brief Destructor */
  ~SciWavefunctionContainer() override = default;

  /** @brief Clone method for deep copying */
  std::unique_ptr<WavefunctionContainer> clone() const override;

  /**
   * @brief Get reference to orbital basis set
   * @return Shared pointer to orbitals
   */
  std::shared_ptr<Orbitals> get_orbitals() const override;

  /**
   * @brief Get coefficient for a specific determinant
   *
   * The configuration is expected to be a determinant describing only
   * the wavefunction's active space.
   *
   * @param det Configuration/determinant to get coefficient for
   * @return Coefficient of the determinant
   */
  ScalarVariant get_coefficient(const Configuration& det) const override;

  /**
   * @brief Get all determinants in the wavefunction
   * @return Vector of all configurations/determinants
   */
  const VectorVariant& get_coefficients() const override;

  /**
   * @brief Get all determinants in the wavefunction
   * @return Vector of all configurations/determinants
   */
  const DeterminantVector& get_active_determinants() const override;

  /**
   * @brief Get number of determinants
   * @return Number of determinants in the wavefunction
   */
  size_t size() const override;

  /**
   * @brief Calculate overlap with another wavefunction
   * @param other Other wavefunction
   * @return Overlap value
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  /**
   * @brief Calculate norm of the wavefunction
   * @return Norm
   */
  double norm() const override;

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
   * @brief Clear cached data to release memory
   *
   * Clears the following cached data for Selected Configuration Interaction
   * (SCI) wavefunctions:
   * - One-particle RDMs: spin-traced (_one_rdm_spin_traced) and
   *   spin-dependent (_one_rdm_spin_dependent_aa, _one_rdm_spin_dependent_bb)
   * - Two-particle RDMs: spin-traced (_two_rdm_spin_traced) and
   *   spin-dependent (_two_rdm_spin_dependent_aaaa,
   * _two_rdm_spin_dependent_aabb, _two_rdm_spin_dependent_bbbb)
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
   * @return Unique pointer to SCI container created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<SciWavefunctionContainer> from_json(
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
   * @return Unique pointer to SCI container created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::unique_ptr<SciWavefunctionContainer> from_hdf5(H5::Group& group);

  /**
   * @brief Get container type identifier for serialization
   * @return String "sci"
   */
  std::string get_container_type() const override;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return True if coefficients are complex, false if real
   */
  bool is_complex() const override;

 private:
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
  // Coefficients of the wavefunction
  const VectorVariant _coefficients;
  // Configuration set (contains determinants and orbital information)
  const ConfigurationSet _configuration_set;
};

}  // namespace qdk::chemistry::data
