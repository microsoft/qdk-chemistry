// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

namespace qdk::chemistry::data {

/**
 * @class CasStateVectorContainer
 * @brief State vector over a complete active space whose determinants are
 *        generated on demand
 *
 * The expansion holds every determinant with a given number of alpha and beta
 * electrons in the active orbitals, but only its coefficients are stored.
 * Coefficient @c i belongs to the determinant whose alpha and beta strings have
 * the ranks @c i / N_beta and @c i % N_beta, where @c N_beta is the number of
 * beta strings and strings are ranked in lexicographic order of their sorted
 * occupied orbitals. The determinant list is built on first use and released
 * by @ref clear_caches.
 *
 * The container behaves and serializes as a @ref StateVectorContainer; files it
 * writes load as a StateVectorContainer holding the same expansion.
 */
class CasStateVectorContainer final : public StateVectorContainer {
 public:
  /**
   * @brief Constructs a complete-active-space state vector without reduced
   *        density matrices (RDMs)
   *
   * @param coeffs Coefficients of all determinants, in the order described
   * above
   * @param n_alpha Number of active alpha electrons
   * @param n_beta Number of active beta electrons
   * @param orbitals Shared pointer to orbital basis set; the active orbitals
   * (all orbitals if no active space is set) span the determinants
   * @param sector Name of the single-particle sector the orbitals belong to
   * @param type Wavefunction type (SelfDual or NotSelfDual)
   * @throws std::invalid_argument If the alpha and beta active spaces differ
   * in size, an electron count exceeds the number of active orbitals, or the
   * number of coefficients differs from the number of determinants
   */
  CasStateVectorContainer(
      VectorVariant coeffs, size_t n_alpha, size_t n_beta,
      std::shared_ptr<Orbitals> orbitals,
      const std::string& sector = Wavefunction::DEFAULT_SECTOR,
      WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a complete-active-space state vector with reduced density
   *        matrix (RDM) data
   *
   * @param coeffs Coefficients of all determinants, in the order described
   * above
   * @param n_alpha Number of active alpha electrons
   * @param n_beta Number of active beta electrons
   * @param orbitals Shared pointer to orbital basis set; the active orbitals
   * (all orbitals if no active space is set) span the determinants
   * @param one_rdm_spin_traced Spin-traced 1-RDM for active orbitals (optional)
   * @param one_rdm_aa Alpha-alpha block of 1-RDM for active orbitals (optional)
   * @param one_rdm_bb Beta-beta block of 1-RDM for active orbitals (optional)
   * @param two_rdm_spin_traced Spin-traced 2-RDM for active orbitals (optional)
   * @param two_rdm_aaaa Alpha-alpha-alpha-alpha block of 2-RDM for active
   * orbitals (optional)
   * @param two_rdm_aabb Alpha-alpha-beta-beta block of 2-RDM for active
   * orbitals (optional)
   * @param two_rdm_bbbb Beta-beta-beta-beta block of 2-RDM for active orbitals
   * (optional)
   * @param sector Name of the single-particle sector the orbitals belong to
   * @param entropies Orbital entropies, with optional keys "single_orbital"
   * (1-D), "two_orbital" (2-D), and "mutual_information" (2-D) (optional)
   * @param type Wavefunction type (SelfDual or NotSelfDual)
   * @throws std::invalid_argument If the alpha and beta active spaces differ
   * in size, an electron count exceeds the number of active orbitals, or the
   * number of coefficients differs from the number of determinants
   */
  CasStateVectorContainer(
      VectorVariant coeffs, size_t n_alpha, size_t n_beta,
      std::shared_ptr<Orbitals> orbitals,
      const std::optional<MatrixVariant>& one_rdm_spin_traced,
      const std::optional<MatrixVariant>& one_rdm_aa,
      const std::optional<MatrixVariant>& one_rdm_bb,
      const std::optional<VectorVariant>& two_rdm_spin_traced,
      const std::optional<VectorVariant>& two_rdm_aaaa,
      const std::optional<VectorVariant>& two_rdm_aabb,
      const std::optional<VectorVariant>& two_rdm_bbbb,
      const std::string& sector = Wavefunction::DEFAULT_SECTOR,
      const OrbitalEntropies& entropies = OrbitalEntropies{},
      WavefunctionType type = WavefunctionType::SelfDual);

  std::unique_ptr<WavefunctionContainer> clone() const override;

  /**
   * @brief Get coefficient for a specific determinant, without building the
   *        determinant list
   * @param det Configuration/determinant to get coefficient for
   * @return Coefficient of the determinant, zero if it is not in the space
   */
  ScalarVariant get_coefficient(const Configuration& det) const override;

  /**
   * @brief Get all determinants, building the list on first use
   * @return Vector of all configurations/determinants
   */
  const DeterminantVector& get_active_determinants() const override;

  /**
   * @brief Get the configuration set, building it on first use
   * @return Reference to the configuration set containing determinants and
   * orbitals
   */
  const ConfigurationSet& get_configuration_set() const override;

  size_t size() const override;

  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  bool contains_determinant(const Configuration& det) const override;

  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  total_num_particles() const override;

  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  active_num_particles() const override;

  /**
   * @brief Clear cached data to release memory
   *
   * Clears the cached active-space RDMs and the determinant list.
   */
  void clear_caches() const override;

 private:
  /// Number of active orbitals
  const size_t _num_orbitals;
  /// Number of active alpha electrons
  const size_t _n_alpha;
  /// Number of active beta electrons
  const size_t _n_beta;
  /// Determinant list, built on first use
  mutable std::shared_ptr<const ConfigurationSet> _determinants;

  /**
   * @brief Position of a determinant in the expansion
   * @param det Configuration/determinant to locate
   * @return Index of @p det, or no value if it is not in the space
   */
  std::optional<size_t> _index_of(const Configuration& det) const;
};

}  // namespace qdk::chemistry::data
