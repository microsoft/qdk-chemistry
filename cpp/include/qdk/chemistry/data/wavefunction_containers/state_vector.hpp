// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
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
 * @class StateVectorContainer
 * @brief Wavefunction container for a state expressed as a linear combination
 *        of Slater determinants.
 *
 * This is the single container type for determinant-expansion wavefunctions. It
 * stores a coefficient vector together with the corresponding determinants (as
 * a
 * @ref ConfigurationSet) and subsumes the previous single-determinant (Slater
 * determinant), complete-active-space (CAS), and selected-configuration-
 * interaction (SCI) containers, which were structurally identical apart from a
 * type tag.
 *
 * A single Slater determinant is simply the special case of a one-determinant
 * expansion with coefficient 1.0; use the single-determinant convenience
 * constructor for that case. When the expansion contains exactly one
 * determinant and no reduced density matrices (RDMs) were supplied, the
 * active-space RDMs, orbital occupations, and single-orbital entropies are
 * generated on the fly from the determinant occupations.
 */
class StateVectorContainer : public WavefunctionContainer {
 public:
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;
  using CoeffContainer = ContainerTypes::VectorVariant;

  /**
   * @brief Constructs a state-vector wavefunction without reduced density
   *        matrices (RDMs)
   *
   * @param coeffs The vector of CI coefficients (can be real or complex)
   * @param dets The vector of determinants
   * @param orbitals Shared pointer to orbital basis set
   * @param type Wavefunction type (SelfDual or NotSelfDual)
   */
  StateVectorContainer(const VectorVariant& coeffs,
                       const DeterminantVector& dets,
                       std::shared_ptr<Orbitals> orbitals,
                       WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a single Slater-determinant state vector
   *
   * Convenience constructor for the one-determinant, coefficient-1.0 case (e.g.
   * a Hartree-Fock reference). Validates that the configuration has sufficient
   * orbital capacity for the active space and that any orbitals beyond the
   * active space are unoccupied.
   *
   * Note: Configurations only represent the active space, not the full orbital
   * space. Inactive and virtual orbitals are not included in the configuration
   * representation.
   *
   * @param det The single determinant configuration (active space only)
   * @param orbitals Shared pointer to orbital basis set
   * @param type Type of wavefunction (default: SelfDual)
   * @throws std::invalid_argument If validation fails
   */
  StateVectorContainer(const Configuration& det,
                       std::shared_ptr<Orbitals> orbitals,
                       WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a state vector with spin-traced reduced density matrix
   *        (RDM) data
   *
   * @param coeffs The vector of CI coefficients (can be real or complex)
   * @param dets The vector of determinants
   * @param orbitals Shared pointer to orbital basis set
   * @param one_rdm_spin_traced Spin-traced 1-RDM for active orbitals (optional)
   * @param two_rdm_spin_traced Spin-traced 2-RDM for active orbitals (optional)
   * @param entropies Orbital entropies, with optional keys "single_orbital"
   * (1-D), "two_orbital" (2-D), and "mutual_information" (2-D) (optional)
   * @param type Wavefunction type (SelfDual or NotSelfDual)
   */
  StateVectorContainer(const VectorVariant& coeffs,
                       const DeterminantVector& dets,
                       std::shared_ptr<Orbitals> orbitals,
                       const std::optional<MatrixVariant>& one_rdm_spin_traced,
                       const std::optional<VectorVariant>& two_rdm_spin_traced,
                       const OrbitalEntropies& entropies = OrbitalEntropies{},
                       WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a state vector with full reduced density matrix (RDM)
   *        data
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
   * @param entropies Orbital entropies, with optional keys "single_orbital"
   * (1-D), "two_orbital" (2-D), and "mutual_information" (2-D) (optional)
   * @param type Wavefunction type (SelfDual or NotSelfDual)
   */
  StateVectorContainer(const VectorVariant& coeffs,
                       const DeterminantVector& dets,
                       std::shared_ptr<Orbitals> orbitals,
                       const std::optional<MatrixVariant>& one_rdm_spin_traced,
                       const std::optional<MatrixVariant>& one_rdm_aa,
                       const std::optional<MatrixVariant>& one_rdm_bb,
                       const std::optional<VectorVariant>& two_rdm_spin_traced,
                       const std::optional<VectorVariant>& two_rdm_aabb,
                       const std::optional<VectorVariant>& two_rdm_aaaa,
                       const std::optional<VectorVariant>& two_rdm_bbbb,
                       const OrbitalEntropies& entropies = OrbitalEntropies{},
                       WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a state vector from preconstructed RDM storage.
   *
   * Used by the serialization layer to hand reconstructed @ref
   * SymmetryBlockedTensorVariant objects to the container without going
   * through the per-block construction path.
   *
   * @param coeffs The vector of CI coefficients (real or complex).
   * @param dets The vector of determinants.
   * @param orbitals Shared pointer to orbital basis set.
   * @param one_rdm_spin_traced Spin-traced 1-RDM (may be @c nullptr).
   * @param two_rdm_spin_traced Spin-traced 2-RDM (may be @c nullptr).
   * @param active_one_rdm Spin-dependent active-space 1-RDM (may be
   *        @c nullptr).
   * @param active_two_rdm Spin-dependent active-space 2-RDM (may be
   *        @c nullptr).
   * @param entropies Orbital entropies.
   * @param type Wavefunction type (SelfDual or NotSelfDual).
   */
  StateVectorContainer(
      const VectorVariant& coeffs, const DeterminantVector& dets,
      std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<MatrixVariant> one_rdm_spin_traced,
      std::shared_ptr<VectorVariant> two_rdm_spin_traced,
      std::shared_ptr<const SymmetryBlockedTensorVariant<2>> active_one_rdm,
      std::shared_ptr<const SymmetryBlockedTensorVariant<4>> active_two_rdm,
      const OrbitalEntropies& entropies = OrbitalEntropies{},
      WavefunctionType type = WavefunctionType::SelfDual);

  /** @brief Destructor */
  ~StateVectorContainer() override = default;

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
   * @brief Get all coefficients in the wavefunction
   * @return Vector of all coefficients (real or complex)
   */
  const VectorVariant& get_coefficients() const override;

  /**
   * @brief Get all determinants in the wavefunction
   * @return Vector of all configurations/determinants
   */
  const DeterminantVector& get_active_determinants() const override;

  /**
   * @brief Get the configuration set for this wavefunction
   * @return Reference to the configuration set containing determinants and
   * orbitals
   */
  const ConfigurationSet& get_configuration_set() const override;

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
   * @brief Check if a determinant is present in the expansion
   * @param det Configuration/determinant to check for
   * @return True if the determinant is present
   */
  bool contains_determinant(const Configuration& det) const;

  /**
   * @brief Number of electrons (active + inactive) as a symmetry-blocked
   * scalar.
   * @return Shared pointer to the symmetry-blocked total electron count.
   */
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  total_num_electrons() const override;

  /**
   * @brief Number of active-space electrons as a symmetry-blocked scalar.
   * @return Shared pointer to the symmetry-blocked active electron count.
   */
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  active_num_electrons() const override;

  /**
   * @brief Orbital occupations for all orbitals (total = active + inactive +
   * virtual) as a rank-1 symmetry-blocked tensor.
   * @return Shared pointer to the symmetry-blocked total orbital occupations.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> total_orbital_occupations()
      const override;

  /**
   * @brief Orbital occupations for active orbitals only as a rank-1
   * symmetry-blocked tensor.
   * @return Shared pointer to the symmetry-blocked active orbital occupations.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> active_orbital_occupations()
      const override;

  /**
   * @brief Active-space spin-dependent one-particle RDM as an SBT.
   *
   * Returns stored data when available; for a single-determinant expansion the
   * RDM is generated lazily from the determinant occupations.
   */
  const SymmetryBlockedTensorVariant<2>& active_one_rdm() const override;

  /**
   * @brief Active-space spin-dependent two-particle RDM as an SBT.
   *
   * Returns stored data when available; for a single-determinant expansion the
   * RDM is generated lazily from the determinant occupations.
   */
  const SymmetryBlockedTensorVariant<4>& active_two_rdm() const override;

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
   * @brief Clear cached data to release memory
   *
   * Clears the cached active-space RDMs (spin-traced and spin-dependent).
   */
  void clear_caches() const override;

  /**
   * @brief Convert container to JSON format
   * @return JSON object containing container data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Load container from JSON format
   *
   * Reads the current "state_vector" format as well as the legacy "cas",
   * "sci", and "sd" formats (read-only backward compatibility).
   *
   * @param j JSON object containing container data
   * @return Unique pointer to the container created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<WavefunctionContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Load container from HDF5 group
   *
   * Reads the current "state_vector" format as well as the legacy "cas",
   * "sci", and "sd" formats (read-only backward compatibility).
   *
   * @param group HDF5 group containing container data
   * @return Unique pointer to the container created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::unique_ptr<WavefunctionContainer> from_hdf5(H5::Group& group);

  /**
   * @brief Get container type identifier for serialization
   * @return String "state_vector"
   */
  std::string get_container_type() const override;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return True if coefficients are complex, false if real
   */
  bool is_complex() const override;

  /**
   * @brief Check if this container has coefficients data
   * @return True if coefficients are available, false otherwise
   */
  bool has_coefficients() const override;

  /**
   * @brief Check if this container has configuration set data
   * @return True if configuration set is available, false otherwise
   */
  bool has_configuration_set() const override;

 private:
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.2.0";

  // Coefficients of the wavefunction
  const CoeffContainer _coefficients;
  // Configuration set (contains determinants and orbital information)
  const ConfigurationSet _configuration_set;

  /**
   * @brief Whether this expansion is a single determinant.
   *
   * Single-determinant expansions support on-the-fly generation of RDMs,
   * orbital occupations, and single-orbital entropies from the determinant
   * occupations.
   */
  bool _is_single_determinant() const;

  /**
   * @brief Number of active (alpha, beta) electrons read from the expansion's
   * determinants.
   * @return Pair of (n_alpha_active, n_beta_active).
   */
  std::pair<std::size_t, std::size_t> _active_electron_counts() const;

  /**
   * @brief Number of total (active + inactive) (alpha, beta) electrons.
   * @return Pair of (n_alpha_total, n_beta_total).
   */
  std::pair<std::size_t, std::size_t> _total_electron_counts() const;

  /**
   * @brief Active-orbital (alpha, beta) occupation vectors.
   * @return Pair of (alpha_active_occupations, beta_active_occupations).
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> _active_occupations_pair() const;

  /**
   * @brief Total-orbital (alpha, beta) occupation vectors.
   * @return Pair of (alpha_occupations_total, beta_occupations_total).
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> _total_occupations_pair() const;
};

}  // namespace qdk::chemistry::data
