// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief One block-sparse site in an Abelian MPS.
 *
 * An MPS site tensor @f$A[l,p,r]@f$ has a left-bond index @f$l@f$, a local
 * physical-state index @f$p@f$, and a right-bond index @f$r@f$. This class
 * stores it as a list of rank-2 matrices @f$A^p[l,r]@f$ per physical state.
 * For a spin-half spatial orbital, the usual four slices represent the empty,
 * alpha-occupied, beta-occupied, and doubly occupied states.
 * @ref MPSContainer::physical_basis records the physical-slice ordering when
 * that metadata is available.
 *
 * Each matrix is a symmetry-blocked rank-2 tensor. Its rows and columns are
 * partitioned into labeled Abelian bond sectors, including particle number.
 * Only populated sector pairs are stored; an absent block is interpreted as
 * zero. All physical slices use the same real or complex scalar type and the
 * same definitions of their left and right bond spaces, but they may populate
 * different blocks.
 */
class AbelianMPSSite {
 public:
  using PhysicalSlice = SymmetryBlockedTensorVariant<2>;
  using PhysicalSlicePtr = std::shared_ptr<const PhysicalSlice>;
  using DenseMatrixVariant = std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>;

  /**
   * @brief Construct a block-sparse Abelian MPS site from physical slices.
   * @param physical_slices Matrices @f$A^p[l,r]@f$ in physical-state order.
   *        Every slice must use the same scalar type (all real or all complex),
   *        the same left-bond sectors and extents, and the same right-bond
   *        sectors and extents. The left and right bond spaces need not equal
   *        each other.
   * @param left_sector_order Left-bond sector labels in the order used to
   *        concatenate their rows into a dense left-bond index. Every declared
   *        left sector must occur exactly once.
   * @param right_sector_order Right-bond sector labels in the order used to
   *        concatenate their columns into a dense right-bond index. Every
   *        declared right sector must occur exactly once.
   * @throws std::invalid_argument if no slices are supplied, a slice is null,
   * slices have inconsistent scalar types or bond spaces, or a sector order
   * does not contain every bond sector exactly once.
   */
  AbelianMPSSite(std::vector<PhysicalSlicePtr> physical_slices,
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
   * @return Labels determining how sector-local row indices are concatenated
   *         into the dense left-bond index.
   */
  const std::vector<SymmetryLabel>& left_sector_order() const {
    return _left_sector_order;
  }

  /**
   * @brief Get the packing order of the right-bond symmetry sectors.
   * @return Labels determining how sector-local column indices are
   *         concatenated into the dense right-bond index.
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
   * The sector orders first map each sector-local bond index to a dense bond
   * index. Row @c (l * physical_dimension() + p) then stores
   * @f$A^p[l,r]@f$. Entries belonging to absent symmetry blocks remain zero.
   * The Python binding reshapes this packed matrix to an array of shape
   * @c (left, physical, right).
   * @return Packed real or complex matrix matching the slices' scalar type.
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
 * @brief Immutable Abelian block-sparse MPS wavefunction.
 *
 * Each site stores one symmetry-blocked matrix per local physical basis state.
 * Every left and right bond space carries a particle-number axis. A sector's
 * extent is the number of bond states carrying that particle-number label.
 * Missing sector-pair blocks are zero, while populated blocks hold amplitudes
 * between states within the corresponding sectors. Dense conversion
 * concatenates sectors in each site's explicitly declared sector order.
 */
class AbelianMPSContainer : public MPSContainer {
 public:
  using SitePtr = std::shared_ptr<const AbelianMPSSite>;

  /**
   * @brief Construct an Abelian block-sparse MPS wavefunction.
   * @param sites MPS sites in chain order.
   * @param orbitals Orbital basis associated with the MPS.
   * @param total_num_particles Optional symmetry-blocked total particle count.
   * @param active_num_particles Optional symmetry-blocked active particle
   * count.
   * @param orthogonality_center Optional site containing the orthogonality
   *        center. Defaults to site zero, representing right-canonical form
   *        with every later site right-normalized.
   * @param physical_basis Optional one-orbital configurations defining the
   *        physical-slice order shared by every site.
   * @param site_to_orbital_order Molecular-orbital indices in MPS chain order;
   * defaults to identity ordering.
   * @throws std::invalid_argument if common MPS metadata is invalid; a site is
   *        null; a bond lacks particle-number symmetry; sites differ in
   *        physical dimension or scalar type; or adjacent bond spaces are
   *        incompatible.
   */
  AbelianMPSContainer(
      std::vector<SitePtr> sites, std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
          total_num_particles = nullptr,
      std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
          active_num_particles = nullptr,
      std::optional<std::size_t> orthogonality_center = std::size_t{0},
      std::vector<Configuration> physical_basis = {},
      std::vector<std::size_t> site_to_orbital_order = {});

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