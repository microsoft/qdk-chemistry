/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/configuration_set.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class MP2Container
 * @brief Wavefunction container representing an MP2 wavefunction
 *
 * This container stores an MP2 wavefunction with a Hamiltonian reference.
 * Amplitudes can be computed on-demand (lazy evaluation) when first requested.
 *
 */
class MP2Container : public WavefunctionContainer {
 public:
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Constructs an MP2 wavefunction.
   *
   * @param hamiltonian Shared pointer to the Hamiltonian
   * @param wavefunction Shared pointer to the wavefunction
   * @param partitioning Choice of partitioning in perbutation theory: the
   * default is Moeller-Plesset Hamiltonian partitioning, keyword "mp"
   */
  MP2Container(std::shared_ptr<Hamiltonian> hamiltonian,
               std::shared_ptr<Wavefunction> wavefunction,
               const std::string& partitioning = "mp");

  /** @brief Destructor */
  ~MP2Container() override = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to cloned container
   */
  std::unique_ptr<WavefunctionContainer> clone() const override;

  /**
   * @brief Get reference to orbitals
   * @return Shared pointer to orbitals
   */
  std::shared_ptr<Orbitals> get_orbitals() const override;

  /**
   * @brief Get reference to Hamiltonian
   * @return Shared pointer to Hamiltonian
   */
  std::shared_ptr<Hamiltonian> get_hamiltonian() const;

  /**
   * @brief Get reference to Wavefunction
   * @return Shared pointer to Wavefunction
   */
  std::shared_ptr<Wavefunction> get_wavefunction() const;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  const VectorVariant& get_coefficients() const override;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  ScalarVariant get_coefficient(const Configuration& det) const override;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  const DeterminantVector& get_active_determinants() const override;

  /**
   * @brief Get T1 amplitudes
   * @param
   * @return Pair of (alpha, beta) T1 amplitudes
   */
  std::pair<const VectorVariant&, const VectorVariant&> get_t1_amplitudes()
      const;

  /**
   * @brief Get T2 amplitudes
   *
   * @return Tuple of (alpha-beta, alpha-alpha, beta-beta) T2 amplitudes
   */
  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_t2_amplitudes() const;

  /**
   * @brief Check if T1 amplitudes are available
   * @return Whether the amplitudes have been calculated
   */
  bool has_t1_amplitudes() const;

  /**
   * @brief Check if T2 amplitudes are available
   * @return Whether the amplitudes have been calculated
   */
  bool has_t2_amplitudes() const;

  /**
   * @brief Get number of determinants
   * @throws std::runtime_error Always throws as this is not meaningful for MP2
   * wavefunctions
   */
  size_t size() const override;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  double norm() const override;

  /**
   * @brief Get total number of electrons (alpha and beta)
   * @return Pair of (n_alpha, n_beta)
   */
  std::pair<size_t, size_t> get_total_num_electrons() const override;

  /**
   * @brief Get number of active electrons (alpha and beta)
   * @return Pair of (n_alpha_active, n_beta_active)
   */
  std::pair<size_t, size_t> get_active_num_electrons() const override;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_total_orbital_occupations()
      const override;

  /**
   * @brief Not implemented for MP2 wavefunctions
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_active_orbital_occupations()
      const override;

  /**
   * @brief Check if a determinant is contained in the wavefunction
   * @param det Configuration to check
   * @return True if determinant is in the wavefunction
   */
  bool contains_determinant(const Configuration& det) const;

  /**
   * @brief Check if a determinant is a reference determinant
   * @param det Configuration to check
   * @return True if determinant is a reference
   */
  bool contains_reference(const Configuration& det) const;

  /**
   * @brief Clear all cached data
   */
  void clear_caches() const override;

  /**
   * @brief Serialize to JSON
   * @return JSON representation of the container
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Deserialize from JSON
   * @param j JSON object
   * @return Unique pointer to MP2Container
   */
  static std::unique_ptr<MP2Container> from_json(const nlohmann::json& j);

  /**
   * @brief Serialize to HDF5
   * @param group HDF5 group to write to
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Deserialize from HDF5
   * @param group HDF5 group to read from
   * @return Unique pointer to MP2Container
   */
  static std::unique_ptr<MP2Container> from_hdf5(H5::Group& group);

  /**
   * @brief Get container type identifier
   * @return String identifying container type
   */
  std::string get_container_type() const override;

  /**
   * @brief Check if wavefunction uses complex numbers
   * @return True if wavefunction is complex
   */
  bool is_complex() const override;

 private:
  /** @brief Cached coefficients */
  VectorVariant _cached_coefficients;
  /** @brief Cached determinants */
  DeterminantVector _cached_determinants;
  /** @brief Shared pointer to wavfunction */
  std::shared_ptr<Wavefunction> _wavefunction;
  /** @brief Shared pointer to Hamiltonian */
  std::shared_ptr<Hamiltonian> _hamiltonian;

  // Amplitude storage
  mutable std::shared_ptr<VectorVariant> _t1_amplitudes_aa = nullptr;
  mutable std::shared_ptr<VectorVariant> _t1_amplitudes_bb = nullptr;
  mutable std::shared_ptr<VectorVariant> _t2_amplitudes_abab = nullptr;
  mutable std::shared_ptr<VectorVariant> _t2_amplitudes_aaaa = nullptr;
  mutable std::shared_ptr<VectorVariant> _t2_amplitudes_bbbb = nullptr;

  /** @brief Serialization version */
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /** @brief Cached determinant vector */
  mutable std::unique_ptr<DeterminantVector> _determinant_vector_cache;

  /**
   * @brief Compute T1 amplitudes from Hamiltonian
   */
  void _compute_t1_amplitudes() const;

  /**
   * @brief Compute T2 amplitudes from Hamiltonian
   */
  void _compute_t2_amplitudes() const;
};
}  // namespace qdk::chemistry::data
