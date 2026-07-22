// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstddef>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Common interface and metadata for MPS wavefunction representations.
 *
 * Concrete subclasses define the site tensor representation and its bond-space
 * semantics. This base owns only data shared by Abelian and reduced
 * non-Abelian MPS containers.
 */
class MPSContainer : public WavefunctionContainer {
 public:
  /**
   * @brief Get the number of sites in the MPS chain.
   * @return Number of stored MPS sites.
   */
  virtual std::size_t num_sites() const = 0;

  /**
   * @brief Check whether the MPS tensors are complex-valued.
   * @return True for a complex-valued MPS; false for a real-valued MPS.
   */
  virtual bool is_complex() const override = 0;

  /**
   * @brief Not supported for MPS wavefunctions.
   * @param other Wavefunction that would be used in the overlap.
   * @throws std::runtime_error Always.
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  /**
   * @brief Not supported for MPS wavefunctions.
   * @throws std::runtime_error Always.
   */
  double norm() const override;

  /**
   * @brief Get the total number of particles, if supplied by the producer.
   * @return Symmetry-blocked total particle count, or nullptr if unavailable.
   */
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  total_num_particles() const override {
    return _total_num_particles;
  }

  /**
   * @brief Get the active-space particle count, if supplied by the producer.
   * @return Symmetry-blocked active particle count, or nullptr if unavailable.
   */
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  active_num_particles() const override {
    return _active_num_particles;
  }

  /**
   * @brief Not supported for MPS wavefunctions.
   * @throws std::runtime_error Always.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> total_orbital_occupations()
      const override;

  /**
   * @brief Not supported for MPS wavefunctions.
   * @throws std::runtime_error Always.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> active_orbital_occupations()
      const override;

  /** @brief Clear cached derived data; currently a no-op for MPS data. */
  void clear_caches() const override;

  /**
   * @brief Not supported for MPS wavefunctions.
   * @throws std::runtime_error Always.
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Get the serialization type identifier.
   * @return String @c "mps".
   */
  std::string get_container_type() const override;

  /**
   * @brief Get the orbital basis associated with the MPS.
   * @return Shared pointer to the orbitals.
   */
  std::shared_ptr<Orbitals> get_orbitals() const override { return _orbitals; }

  /**
   * @brief Get the single-particle sectors spanned by this container.
   * @return A vector containing only @ref Wavefunction::DEFAULT_SECTOR.
   */
  std::vector<std::string> sectors() const override;

  /**
   * @brief Resolve the orbital basis for a single-particle sector.
   * @param name Sector name to resolve.
   * @return Shared pointer to the orbitals for the default sector.
   * @throws std::out_of_range if @p name is not the default sector.
   */
  std::shared_ptr<const Orbitals> sector_basis(
      const std::string& name) const override;

  /**
   * @brief Get the orthogonality center of the MPS, if specified.
   *
   * Sites before the center are left-normalized and sites after it are
   * right-normalized. Center zero denotes a right-canonical MPS, and the last
   * site denotes a left-canonical MPS.
   * @return Site index of the orthogonality center, or @c std::nullopt if the
   * canonicalization state is unspecified.
   */
  std::optional<std::size_t> orthogonality_center() const {
    return _orthogonality_center;
  }

  /**
   * @brief Get the one-orbital occupation states labeling physical slices.
   * @return Configurations in physical-slice order, shared by every site, or
   *         an empty vector if this metadata was not supplied.
   */
  const std::vector<Configuration>& physical_basis() const {
    return _physical_basis;
  }

  /**
   * @brief Get the molecular orbital represented by each MPS site.
   * @return Unique molecular-orbital indices in MPS chain order. The MPS may
   * represent a subset of the associated orbital basis.
   */
  const std::vector<std::size_t>& site_to_orbital_order() const {
    return _site_to_orbital_order;
  }

 protected:
  /**
   * @brief Construct the representation-independent portion of an MPS.
   * @param orbitals Orbital basis associated with the MPS.
   * @param total_num_particles Optional symmetry-blocked total particle count.
   * @param active_num_particles Optional symmetry-blocked active particle
   * count.
   * @param orthogonality_center Optional site containing the orthogonality
   * center. Sites on either side must be left- and right-normalized,
   * respectively.
   * @param physical_basis Optional one-orbital configurations defining the
   *        physical-slice order shared by every site.
   * @param site_to_orbital_order Unique molecular-orbital indices in MPS chain
   * order. The number of sites may be smaller than the orbital basis size.
   */
  MPSContainer(std::shared_ptr<Orbitals> orbitals,
               std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
                   total_num_particles,
               std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
                   active_num_particles,
               std::optional<std::size_t> orthogonality_center,
               std::vector<Configuration> physical_basis,
               std::vector<std::size_t> site_to_orbital_order);

  /**
   * @brief Validate representation-independent MPS invariants.
   * @param num_sites Number of sites supplied by the concrete container.
   * @param physical_dimension Number of physical slices at each site.
   * @throws std::invalid_argument if there are no sites, orbitals are null,
   * physical-basis configurations are invalid, or the site-to-orbital order
   * is not unique and in range.
   */
  void _validate_common(std::size_t num_sites,
                        std::size_t physical_dimension) const;

 private:
  std::shared_ptr<Orbitals> _orbitals;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
      _total_num_particles;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
      _active_num_particles;
  std::optional<std::size_t> _orthogonality_center;
  std::vector<Configuration> _physical_basis;
  std::vector<std::size_t> _site_to_orbital_order;
};

}  // namespace qdk::chemistry::data
