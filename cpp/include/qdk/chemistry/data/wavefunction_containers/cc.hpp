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
#include <qdk/chemistry/data/wavefunction.hpp>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class CoupledClusterContainer
 * @brief Container representing a coupled cluster singles-doubles wavefunction
 *
 * CI coefficients are generated from the CC amplitudes on-demand (lazy
 * evaluation) and are costly. CI coefficients and determinants are cached after
 * first computation and are calculated by truncating the expansion at the
 * fourth order.
 */
class CoupledClusterContainer : public WavefunctionContainer {
 public:
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Constructs a coupled cluster wavefunction with amplitudes.
   *
   * T1/T2 amplitudes are stored if provided
   *
   * @param orbitals Shared pointer to orbitals
   * @param wavefunction Shared pointer to wavefunction
   * @param t1_amplitudes T1 amplitudes (optional)
   * @param t2_amplitudes T2 amplitudes (optional)
   */
  CoupledClusterContainer(std::shared_ptr<Orbitals> orbitals,
                          std::shared_ptr<Wavefunction> wavefunction,
                          const std::optional<VectorVariant>& t1_amplitudes,
                          const std::optional<VectorVariant>& t2_amplitudes);

  /**
   * @brief Constructs a coupled cluster wavefunction with spin-separated
   * amplitudes
   *
   * T1/T2 amplitudes are stored if provided
   *
   * @param orbitals Shared pointer to orbitals
   * @param wavefunction Shared pointer to wavefunction
   * @param t1_amplitudes_aa Alpha T1 amplitudes (optional)
   * @param t1_amplitudes_bb Beta T1 amplitudes (optional)
   * @param t2_amplitudes_abab Alpha-beta T2 amplitudes (optional)
   * @param t2_amplitudes_aaaa Alpha-alpha T2 amplitudes (optional)
   * @param t2_amplitudes_bbbb Beta-beta T2 amplitudes (optional)
   */
  CoupledClusterContainer(
      std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<Wavefunction> wavefunction,
      const std::optional<VectorVariant>& t1_amplitudes_aa,
      const std::optional<VectorVariant>& t1_amplitudes_bb,
      const std::optional<VectorVariant>& t2_amplitudes_abab,
      const std::optional<VectorVariant>& t2_amplitudes_aaaa,
      const std::optional<VectorVariant>& t2_amplitudes_bbbb);

  /** @brief Destructor */
  ~CoupledClusterContainer() override = default;

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
   * @brief Get reference to wavefunction
   * @return Shared pointer to wavefunction
   */
  std::shared_ptr<Wavefunction> get_wavefunction() const;

  /**
   * @brief Not implemented for CC wavefunctions
   */
  const VectorVariant& get_coefficients() const override;

  /**
   * @brief Not implemented for CC wavefunctions
   */
  ScalarVariant get_coefficient(const Configuration& det) const override;

  /**
   * @brief Not implemented for CC wavefunctions
   */
  const DeterminantVector& get_active_determinants() const override;

  /**
   * @brief Get T1 amplitudes
   *
   * Returns T1 amplitudes if they were provided at construction.
   *
   * @return Pair of (alpha, beta) T1 amplitudes
   * @throws std::runtime_error if T1 amplitudes are not available
   */
  std::pair<const VectorVariant&, const VectorVariant&> get_t1_amplitudes()
      const;

  /**
   * @brief Get T2 amplitudes
   *
   * Returns T2 amplitudes if they were provided at construction.
   *
   * @return Tuple of (alpha-beta, alpha-alpha, beta-beta) T2 amplitudes
   * @throws std::runtime_error if T2 amplitudes are not available
   */
  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_t2_amplitudes() const;

  /**
   * @brief Check if T1 amplitudes are available
   * @return True if T1 amplitudes are available
   */
  bool has_t1_amplitudes() const;

  /**
   * @brief Check if T2 amplitudes are available
   * @return True if T2 amplitudes are available
   */
  bool has_t2_amplitudes() const;

  /**
   * @brief Get number of determinants (not meaningful for CC wavefunctions)
   * @throws std::runtime_error Always throws as this is not meaningful for CC
   * wavefunctions
   */
  size_t size() const override;

  /**
   * @brief Not implemented for CC wavefunctions
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  /**
   * @brief Not implemented for CC wavefunctions
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
   * @brief Check if a determinant is in the reference wavefunction
   * @param det Configuration/determinant to check
   * @return True if determinant matches any reference determinant
   */
  bool contains_determinant(const Configuration& det) const;

  /**
   * @brief Check if a determinant is in the reference wavefunction
   * @param det Configuration/determinant to check
   * @return True if determinant matches any reference determinant
   */
  bool contains_reference(const Configuration& det) const;

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
   * @return Unique pointer to CC container created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<CoupledClusterContainer> from_json(
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
   * @return Unique pointer to CC container created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::unique_ptr<CoupledClusterContainer> from_hdf5(H5::Group& group);

  /**
   * @brief Get container type identifier for serialization
   * @return String "coupled_cluster"
   */
  std::string get_container_type() const override;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return True if amplitudes contain complex values
   */
  bool is_complex() const override;

  /**
   * @brief Check if spin-dependent one-particle RDMs are available
   *
   * Returns true only if RDMs were explicitly set. RDMs cannot be computed
   * from ket amplitudes alone - the adjoint (bra) wavefunction with lambda
   * amplitudes is required.
   *
   * @return True if spin-dependent 1-RDMs are available
   */
  bool has_one_rdm_spin_dependent() const override;

  /**
   * @brief Check if spin-traced one-particle RDM is available
   *
   * Returns true only if RDMs were explicitly set. RDMs cannot be computed
   * from ket amplitudes alone - the adjoint (bra) wavefunction with lambda
   * amplitudes is required.
   *
   * @return True if spin-traced 1-RDM is available
   */
  bool has_one_rdm_spin_traced() const override;

  /**
   * @brief Check if spin-dependent two-particle RDMs are available
   *
   * Returns true only if RDMs were explicitly set. RDMs cannot be computed
   * from ket amplitudes alone - the adjoint (bra) wavefunction with lambda
   * amplitudes is required.
   *
   * @return True if spin-dependent 2-RDMs are available
   */
  bool has_two_rdm_spin_dependent() const override;

  /**
   * @brief Check if spin-traced two-particle RDM is available
   *
   * Returns true only if RDMs were explicitly set. RDMs cannot be computed
   * from ket amplitudes alone - the adjoint (bra) wavefunction with lambda
   * amplitudes is required.
   *
   * @return True if spin-traced 2-RDM is available
   */
  bool has_two_rdm_spin_traced() const override;

  /**
   * @brief Get spin-dependent one-particle RDMs
   *
   * @note RDMs cannot be computed from the ket (T amplitudes) alone.
   *       Coupled cluster RDM computation requires the adjoint (bra)
   *       wavefunction with lambda amplitudes.
   *
   * @return Tuple of (alpha-alpha, beta-beta) one-particle RDMs
   * @throws std::runtime_error Always throws - requires adjoint wavefunction
   */
  std::tuple<const MatrixVariant&, const MatrixVariant&>
  get_active_one_rdm_spin_dependent() const override;

  /**
   * @brief Get spin-traced one-particle RDM
   *
   * @note RDMs cannot be computed from the ket (T amplitudes) alone.
   *       Coupled cluster RDM computation requires the adjoint (bra)
   *       wavefunction with lambda amplitudes.
   *
   * @return Spin-traced one-particle RDM
   * @throws std::runtime_error Always throws - requires adjoint wavefunction
   */
  const MatrixVariant& get_active_one_rdm_spin_traced() const override;

  /**
   * @brief Get spin-dependent two-particle RDMs
   *
   * @note RDMs cannot be computed from the ket (T amplitudes) alone.
   *       Coupled cluster RDM computation requires the adjoint (bra)
   *       wavefunction with lambda amplitudes.
   *
   * @return Tuple of (aabb, aaaa, bbbb) two-particle RDMs
   * @throws std::runtime_error Always throws - requires adjoint wavefunction
   */
  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_active_two_rdm_spin_dependent() const override;

  /**
   * @brief Get spin-traced two-particle RDM
   *
   * @note RDMs cannot be computed from the ket (T amplitudes) alone.
   *       Coupled cluster RDM computation requires the adjoint (bra)
   *       wavefunction with lambda amplitudes.
   *
   * @return Spin-traced two-particle RDM
   * @throws std::runtime_error Always throws - requires adjoint wavefunction
   */
  const VectorVariant& get_active_two_rdm_spin_traced() const override;

 private:
  // Orbital information
  std::shared_ptr<Orbitals> _orbitals;
  // Wfn
  std::shared_ptr<Wavefunction> _wavefunction;

  std::shared_ptr<VectorVariant> _t1_amplitudes_aa = nullptr;
  std::shared_ptr<VectorVariant> _t1_amplitudes_bb = nullptr;

  std::shared_ptr<VectorVariant> _t2_amplitudes_abab = nullptr;
  std::shared_ptr<VectorVariant> _t2_amplitudes_aaaa = nullptr;
  std::shared_ptr<VectorVariant> _t2_amplitudes_bbbb = nullptr;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  // Lazy-initialized determinant vector and coefficients
  mutable std::unique_ptr<DeterminantVector> _determinant_vector_cache;
  mutable std::unique_ptr<VectorVariant> _coefficients_cache;

  /**
   * @brief Generate CI determinants and coefficients from CC amplitudes
   *
   * Expands e^T = 1 + T + T²/2! + T³/3! + T⁴/4! + ...
   * to generate CI-like determinants and coefficients up to 4th order
   * (quadruple excitations).
   *
   * The expansion includes:
   * - Order 0: Reference determinant (coefficient = 1)
   * - Order 1: Singles from T1
   * - Order 2: Doubles from T2 + T1²/2
   * - Order 3: Triples from T1·T2 + T1³/6
   * - Order 4: Quadruples from T2²/2 + T1²·T2/2 + T1⁴/24
   */
  void _generate_ci_expansion() const;

  /**
   * @brief Helper to create a Configuration from excitations
   * @param ref Reference configuration
   * @param alpha_excitations Vector of (from_orbital, to_orbital) for alpha
   * @param beta_excitations Vector of (from_orbital, to_orbital) for beta
   * @return New Configuration with excitations applied
   */
  static Configuration _apply_excitations(
      const Configuration& ref,
      const std::vector<std::pair<size_t, size_t>>& alpha_excitations,
      const std::vector<std::pair<size_t, size_t>>& beta_excitations);
};
}  // namespace qdk::chemistry::data
