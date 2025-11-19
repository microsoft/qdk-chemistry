/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>

namespace macis {

/**
 * @brief Base class for Hamiltonian generators that provides common
 * functionality for handling molecular integrals and computing matrix elements.
 *
 * This class serves as the foundation for various Hamiltonian generator
 * implementations, providing storage and methods for one- and two-electron
 * integrals, as well as precomputed intermediate quantities that accelerate
 * matrix element calculations.
 *
 * @tparam Scalar The scalar type for integral values (typically double)
 */
template <typename Scalar>
class HamiltonianGeneratorBase {
 protected:
  /// @brief Alias for sparse matrix type with templated index type
  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<Scalar, index_t>;

  /// @brief Type alias for 2D matrix span
  using matrix_span_t = matrix_span<Scalar>;
  /// @brief Type alias for 3D tensor span
  using rank3_span_t = rank3_span<Scalar>;
  /// @brief Type alias for 4D tensor span
  using rank4_span_t = rank4_span<Scalar>;

  /// @brief Number of orbitals
  size_t norb_;
  /// @brief Square of number of orbitals (norb^2)
  size_t norb2_;
  /// @brief Cube of number of orbitals (norb^3)
  size_t norb3_;
  /// @brief One-electron integrals T_pq
  matrix_span_t T_pq_;
  /// @brief Two-electron integrals V_pqrs = (pq|rs)
  rank4_span_t V_pqrs_;

  /// @brief Storage for antisymmetrized two-electron integrals G(i,j,k,l) =
  /// (ij|kl) - (il|kj)
  std::vector<Scalar> G_pqrs_data_;
  /// @brief Span view of antisymmetrized two-electron integrals
  rank4_span_t G_pqrs_;

  /// @brief Storage for reduced antisymmetrized integrals G_red(i,j,k) =
  /// G(i,j,k,k)
  std::vector<Scalar> G_red_data_;
  /// @brief Span view of reduced antisymmetrized integrals
  rank3_span_t G_red_;

  /// @brief Storage for reduced two-electron integrals V_red(i,j,k) = (ij|kk)
  std::vector<Scalar> V_red_data_;
  /// @brief Span view of reduced two-electron integrals
  rank3_span_t V_red_;

  /// @brief Storage for doubly reduced antisymmetrized integrals G2_red(i,j) =
  /// 0.5 * G(i,i,j,j)
  std::vector<Scalar> G2_red_data_;
  /// @brief Span view of doubly reduced antisymmetrized integrals
  matrix_span_t G2_red_;

  /// @brief Storage for doubly reduced two-electron integrals V2_red(i,j) =
  /// (ii|jj)
  std::vector<Scalar> V2_red_data_;
  /// @brief Span view of doubly reduced two-electron integrals
  matrix_span_t V2_red_;

  /**
   * @brief Internal method to generate integral intermediates from two-electron
   * integrals
   * @param V The four-dimensional tensor of two-electron integrals
   */
  void generate_integral_intermediates_(rank4_span_t V);

 public:
  /**
   * @brief Constructor for HamiltonianGeneratorBase
   * @param T Matrix span containing one-electron integrals
   * @param V Four-dimensional tensor containing two-electron integrals
   */
  HamiltonianGeneratorBase(matrix_span_t T, rank4_span_t V);

  /**
   * @brief Virtual destructor
   */
  virtual ~HamiltonianGeneratorBase() noexcept = default;

  /**
   * @brief Get pointer to one-electron integral data
   * @return Pointer to the underlying data of T_pq_
   */
  inline auto* T() const { return T_pq_.data_handle(); }

  /**
   * @brief Get pointer to reduced antisymmetrized integral data
   * @return Pointer to the underlying data of G_red_data_
   */
  inline auto* G_red() const { return G_red_data_.data(); }

  /**
   * @brief Get pointer to reduced two-electron integral data
   * @return Pointer to the underlying data of V_red_data_
   */
  inline auto* V_red() const { return V_red_data_.data(); }

  /**
   * @brief Get pointer to antisymmetrized integral data
   * @return Pointer to the underlying data of G_pqrs_data_
   */
  inline auto* G() const { return G_pqrs_data_.data(); }

  /**
   * @brief Get pointer to two-electron integral data
   * @return Pointer to the underlying data of V_pqrs_
   */
  inline auto* V() const { return V_pqrs_.data_handle(); }

  /**
   * @brief Generate integral intermediates using the stored two-electron
   * integrals
   */
  inline void generate_integral_intermediates() {
    generate_integral_intermediates_(V_pqrs_);
  }

  /**
   * @brief Compute single orbital energy for a given orbital from integrals and
   * occupied orbitals
   * @param orb The orbital index
   * @param ss_occ Vector of same-spin occupied orbitals
   * @param os_occ Vector of opposite-spin occupied orbitals
   * @return Single orbital energy
   */
  double single_orbital_en(uint32_t orb, const std::vector<uint32_t>& ss_occ,
                           const std::vector<uint32_t>& os_occ) const;

  /**
   * @brief Compute single orbital energies for all orbitals
   * @param norb Number of orbitals
   * @param ss_occ Vector of same-spin occupied orbitals
   * @param os_occ Vector of opposite-spin occupied orbitals
   * @return Vector of single orbital energies
   */
  std::vector<double> single_orbital_ens(
      size_t norb, const std::vector<uint32_t>& ss_occ,
      const std::vector<uint32_t>& os_occ) const;

  /**
   * @brief Fast computation of diagonal matrix element for single excitation
   * @param ss_occ Vector of same-spin occupied orbitals
   * @param os_occ Vector of opposite-spin occupied orbitals
   * @param orb_hol Hole orbital index
   * @param orb_par Particle orbital index
   * @param orig_det_Hii Original determinant diagonal element
   * @return Diagonal matrix element
   */
  double fast_diag_single(const std::vector<uint32_t>& ss_occ,
                          const std::vector<uint32_t>& os_occ, uint32_t orb_hol,
                          uint32_t orb_par, double orig_det_Hii) const;

  /**
   * @brief Fast computation of diagonal matrix element for single excitation
   * with precomputed energies
   * @param hol_en Hole orbital energy
   * @param par_en Particle orbital energy
   * @param orb_hol Hole orbital index
   * @param orb_par Particle orbital index
   * @param orig_det_Hii Original determinant diagonal element
   * @return Diagonal matrix element
   */
  double fast_diag_single(double hol_en, double par_en, uint32_t orb_hol,
                          uint32_t orb_par, double orig_det_Hii) const;

  /**
   * @brief Fast computation of diagonal matrix element for same-spin double
   * excitation with precomputed energies
   * @param en_hol1 First hole orbital energy
   * @param en_hol2 Second hole orbital energy
   * @param en_par1 First particle orbital energy
   * @param en_par2 Second particle orbital energy
   * @param orb_hol1 First hole orbital index
   * @param orb_hol2 Second hole orbital index
   * @param orb_par1 First particle orbital index
   * @param orb_par2 Second particle orbital index
   * @param orig_det_Hii Original determinant diagonal element
   * @return Diagonal matrix element
   */
  double fast_diag_ss_double(double en_hol1, double en_hol2, double en_par1,
                             double en_par2, uint32_t orb_hol1,
                             uint32_t orb_hol2, uint32_t orb_par1,
                             uint32_t orb_par2, double orig_det_Hii) const;

  /**
   * @brief Fast computation of diagonal matrix element for same-spin double
   * excitation
   * @param ss_occ Vector of same-spin occupied orbitals
   * @param os_occ Vector of opposite-spin occupied orbitals
   * @param orb_hol1 First hole orbital index
   * @param orb_hol2 Second hole orbital index
   * @param orb_par1 First particle orbital index
   * @param orb_par2 Second particle orbital index
   * @param orig_det_Hii Original determinant diagonal element
   * @return Diagonal matrix element
   */
  double fast_diag_ss_double(const std::vector<uint32_t>& ss_occ,
                             const std::vector<uint32_t>& os_occ,
                             uint32_t orb_hol1, uint32_t orb_hol2,
                             uint32_t orb_par1, uint32_t orb_par2,
                             double orig_det_Hii) const;

  /**
   * @brief Fast computation of diagonal matrix element for opposite-spin double
   * excitation with precomputed energies
   * @param en_holu Alpha hole orbital energy
   * @param en_hold Beta hole orbital energy
   * @param en_paru Alpha particle orbital energy
   * @param en_pard Beta particle orbital energy
   * @param orb_holu Alpha hole orbital index
   * @param orb_hold Beta hole orbital index
   * @param orb_paru Alpha particle orbital index
   * @param orb_pard Beta particle orbital index
   * @param orig_det_Hii Original determinant diagonal element
   * @return Diagonal matrix element
   */
  double fast_diag_os_double(double en_holu, double en_hold, double en_paru,
                             double en_pard, uint32_t orb_holu,
                             uint32_t orb_hold, uint32_t orb_paru,
                             uint32_t orb_pard, double orig_det_Hii) const;

  /**
   * @brief Fast computation of diagonal matrix element for opposite-spin double
   * excitation
   * @param up_occ Vector of alpha occupied orbitals
   * @param do_occ Vector of beta occupied orbitals
   * @param orb_holu Alpha hole orbital index
   * @param orb_hold Beta hole orbital index
   * @param orb_paru Alpha particle orbital index
   * @param orb_pard Beta particle orbital index
   * @param orig_det_Hii Original determinant diagonal element
   * @return Diagonal matrix element
   */
  double fast_diag_os_double(const std::vector<uint32_t>& up_occ,
                             const std::vector<uint32_t>& do_occ,
                             uint32_t orb_holu, uint32_t orb_hold,
                             uint32_t orb_paru, uint32_t orb_pard,
                             double orig_det_Hii) const;

  /**
   * @brief Rotate Hamiltonian using one-electron reduced density matrix
   * @param ordm Pointer to one-electron reduced density matrix data
   */
  void rotate_hamiltonian_ordm(const Scalar* ordm);
};

}  // namespace macis
