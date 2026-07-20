/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/** @brief Canonicalization state recorded for an MPS wavefunction. */
enum class MPSCanonicalForm {
  Unspecified,      ///< Producer did not record a canonical form.
  LeftNormalized,   ///< Every site tensor is left-normalized.
  RightNormalized,  ///< Every site tensor is right-normalized.
  Mixed             ///< Sites are normalized about a canonical center.
};

/**
 * @brief One block-sparse MPS site.
 *
 * Slice @c p stores the matrix @f$A^p[l,r]@f$. The number and meaning of the
 * slices are not fixed by this class; @ref MPSContainer::physical_basis may
 * describe their ordering. Bond-sector order is explicit so dense conversion
 * never depends on unordered block-map iteration.
 */
class MPSSite {
 public:
  using PhysicalSlice = SymmetryBlockedTensorVariant<2>;
  using PhysicalSlicePtr = std::shared_ptr<const PhysicalSlice>;
  using DenseMatrixVariant = std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>;

  /**
   * @brief Construct a block-sparse MPS site from its physical slices.
   * @param physical_slices Physical-state slices sharing one scalar type and
   * identical left and right bond spaces.
   * @param left_sector_order Order in which left-bond sectors are packed by
   * @ref to_dense.
   * @param right_sector_order Order in which right-bond sectors are packed by
   * @ref to_dense.
   * @throws std::invalid_argument if no slices are supplied, a slice is null,
   * slices have inconsistent scalar types or bond spaces, or a sector order
   * does not contain every bond sector exactly once.
   */
  MPSSite(std::vector<PhysicalSlicePtr> physical_slices,
          std::vector<SymmetryLabel> left_sector_order,
          std::vector<SymmetryLabel> right_sector_order);

  /**
   * @brief Get the physical-state slices stored at this site.
   * @return Physical slices in local-basis order.
   */
  const std::vector<PhysicalSlicePtr>& physical_slices() const {
    return _physical_slices;
  }

  /**
   * @brief Get the packing order of the left-bond symmetry sectors.
   * @return Left-bond sector labels in dense packing order.
   */
  const std::vector<SymmetryLabel>& left_sector_order() const {
    return _left_sector_order;
  }

  /**
   * @brief Get the packing order of the right-bond symmetry sectors.
   * @return Right-bond sector labels in dense packing order.
   */
  const std::vector<SymmetryLabel>& right_sector_order() const {
    return _right_sector_order;
  }

  /**
   * @brief Get the local physical dimension.
   * @return Number of physical-state slices stored at this site.
   */
  std::size_t physical_dimension() const { return _physical_slices.size(); }

  /**
   * @brief Get the full left-bond dimension across all symmetry sectors.
   * @return Sum of the left-bond sector extents.
   */
  std::size_t left_bond_dimension() const;

  /**
   * @brief Get the full right-bond dimension across all symmetry sectors.
   * @return Sum of the right-bond sector extents.
   */
  std::size_t right_bond_dimension() const;

  /**
   * @brief Check whether the site tensors are complex-valued.
   * @return True for complex-valued slices; false for real-valued slices.
   */
  bool is_complex() const;

  /**
   * @brief Materialize this site as a matrix packed as
   * @c (left * physical, right).
   *
   * Row @c (l * physical_dimension() + p) stores @f$A^p[l,r]@f$.
   * @return Real or complex dense matrix matching the slices' scalar type.
   */
  DenseMatrixVariant to_dense() const;

 private:
  /** @brief Validate slice compatibility and bond-sector packing orders. */
  void _validate() const;

  std::vector<PhysicalSlicePtr> _physical_slices;
  std::vector<SymmetryLabel> _left_sector_order;
  std::vector<SymmetryLabel> _right_sector_order;
};

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
   * @brief Get the orbital basis associated with the MPS.
   * @return Shared pointer to the orbitals.
   */
  std::shared_ptr<const Orbitals> orbitals() const { return _orbitals; }

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
   * @brief Get the canonicalization state of the MPS.
   * @return Recorded canonical form.
   */
  MPSCanonicalForm canonical_form() const { return _canonical_form; }

  /**
   * @brief Get the canonical center of a mixed-canonical MPS.
   * @return Zero-based center site, or std::nullopt for other canonical forms.
   */
  std::optional<std::size_t> canonical_center() const {
    return _canonical_center;
  }

  /**
   * @brief Get the total weight discarded while truncating MPS bonds.
   * @return Non-negative discarded weight.
   */
  double discarded_weight() const { return _discarded_weight; }

  /**
   * @brief Get labels describing the physical slices at every site.
   * @return Physical-basis labels in slice order, or an empty vector if they
   * were not supplied.
   */
  const std::vector<std::string>& physical_basis() const {
    return _physical_basis;
  }

 protected:
  /**
   * @brief Construct the representation-independent portion of an MPS.
   * @param orbitals Orbital basis associated with the MPS.
   * @param total_num_particles Optional symmetry-blocked total particle count.
   * @param active_num_particles Optional symmetry-blocked active particle
   * count.
   * @param canonical_form Canonicalization state of the MPS.
   * @param canonical_center Optional zero-based center of a mixed-canonical
   * MPS.
   * @param discarded_weight Total weight discarded during bond truncation.
   * @param physical_basis Optional labels for the physical slices at each site.
   */
  MPSContainer(std::shared_ptr<Orbitals> orbitals,
               std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
                   total_num_particles,
               std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
                   active_num_particles,
               MPSCanonicalForm canonical_form,
               std::optional<std::size_t> canonical_center,
               double discarded_weight,
               std::vector<std::string> physical_basis);

  /**
   * @brief Validate representation-independent MPS invariants.
   * @param num_sites Number of sites supplied by the concrete container.
   * @param physical_dimension Number of physical slices at each site.
   * @throws std::invalid_argument if there are no sites, orbitals are null,
   * discarded weight is negative, canonical-center metadata is inconsistent,
   * or physical-basis labels do not match @p physical_dimension.
   */
  void _validate_common(std::size_t num_sites,
                        std::size_t physical_dimension) const;

 private:
  std::shared_ptr<Orbitals> _orbitals;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
      _total_num_particles;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
      _active_num_particles;
  MPSCanonicalForm _canonical_form;
  std::optional<std::size_t> _canonical_center;
  double _discarded_weight;
  std::vector<std::string> _physical_basis;
};

/**
 * @brief Immutable Abelian block-sparse MPS wavefunction.
 *
 * Each site stores one symmetry-blocked matrix per local physical basis state.
 * Bond-sector extents are degeneracy dimensions, and dense conversion is a
 * direct assembly of those blocks in the declared sector order.
 */
class AbelianMPSContainer : public MPSContainer {
 public:
  using SitePtr = std::shared_ptr<const MPSSite>;

  /**
   * @brief Construct an Abelian block-sparse MPS wavefunction.
   * @param sites MPS sites in chain order.
   * @param orbitals Orbital basis associated with the MPS.
   * @param total_num_particles Optional symmetry-blocked total particle count.
   * @param active_num_particles Optional symmetry-blocked active particle
   * count.
   * @param canonical_form Canonicalization state of the MPS.
   * @param canonical_center Optional zero-based center of a mixed-canonical
   * MPS.
   * @param discarded_weight Total weight discarded during bond truncation.
   * @param physical_basis Optional labels for the physical slices at each site.
   * @throws std::invalid_argument if common MPS metadata is invalid; a site is
   * null; sites differ in physical dimension or scalar type; or adjacent bond
   * spaces are incompatible.
   */
  AbelianMPSContainer(
      std::vector<SitePtr> sites, std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
          total_num_particles = nullptr,
      std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
          active_num_particles = nullptr,
      MPSCanonicalForm canonical_form = MPSCanonicalForm::Unspecified,
      std::optional<std::size_t> canonical_center = std::nullopt,
      double discarded_weight = 0.0,
      std::vector<std::string> physical_basis = {});

  /**
   * @brief Get the block-sparse sites in chain order.
   * @return Immutable site pointers in chain order.
   */
  const std::vector<SitePtr>& sites() const { return _sites; }

  /**
   * @brief Copy this MPS container.
   * @return A new container sharing the immutable site data and metadata.
   */
  std::unique_ptr<WavefunctionContainer> clone() const override;

  /**
   * @brief Get the number of sites in the MPS chain.
   * @return Number of stored sites.
   */
  std::size_t num_sites() const override { return _sites.size(); }

  /**
   * @brief Get the largest total bond dimension in the MPS.
   * @return Maximum left or right bond dimension across all sites.
   */
  std::size_t max_bond_dimension() const;

  /**
   * @brief Check whether the site tensors are complex-valued.
   * @return True for complex-valued sites; false for real-valued sites.
   */
  bool is_complex() const override;

 private:
  /** @brief Validate site consistency and adjacent bond compatibility. */
  void _validate() const;

  std::vector<SitePtr> _sites;
};

}  // namespace qdk::chemistry::data