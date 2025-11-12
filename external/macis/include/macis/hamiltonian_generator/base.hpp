/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>

namespace macis {

template <typename Scalar>
class HamiltonianGeneratorBase {
 protected:
  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<Scalar, index_t>;

  using matrix_span_t = matrix_span<Scalar>;
  using rank3_span_t = rank3_span<Scalar>;
  using rank4_span_t = rank4_span<Scalar>;

  size_t norb_;
  size_t norb2_;
  size_t norb3_;
  matrix_span_t T_pq_;
  rank4_span_t V_pqrs_;

  // G(i,j,k,l) = (ij|kl) - (il|kj)
  std::vector<Scalar> G_pqrs_data_;
  rank4_span_t G_pqrs_;

  // G_red(i,j,k) = G(i,j,k,k)
  std::vector<Scalar> G_red_data_;
  rank3_span_t G_red_;

  // V_red(i,j,k) = (ij|kk)
  std::vector<Scalar> V_red_data_;
  rank3_span_t V_red_;

  // G2_red(i,j)  = 0.5 * G(i,i,j,j)
  std::vector<Scalar> G2_red_data_;
  matrix_span_t G2_red_;

  // V2_red(i,j)  = (ii|jj)
  std::vector<Scalar> V2_red_data_;
  matrix_span_t V2_red_;

  void generate_integral_intermediates_(rank4_span_t V);

 public:
  HamiltonianGeneratorBase(matrix_span_t T, rank4_span_t V);
  virtual ~HamiltonianGeneratorBase() noexcept = default;

  inline auto* T() const { return T_pq_.data_handle(); }
  inline auto* G_red() const { return G_red_data_.data(); }
  inline auto* V_red() const { return V_red_data_.data(); }
  inline auto* G() const { return G_pqrs_data_.data(); }
  inline auto* V() const { return V_pqrs_.data_handle(); }

  inline void generate_integral_intermediates() {
    generate_integral_intermediates_(V_pqrs_);
  }

  double single_orbital_en(uint32_t orb, const std::vector<uint32_t>& ss_occ,
                           const std::vector<uint32_t>& os_occ) const;

  std::vector<double> single_orbital_ens(
      size_t norb, const std::vector<uint32_t>& ss_occ,
      const std::vector<uint32_t>& os_occ) const;

  double fast_diag_single(const std::vector<uint32_t>& ss_occ,
                          const std::vector<uint32_t>& os_occ, uint32_t orb_hol,
                          uint32_t orb_par, double orig_det_Hii) const;

  double fast_diag_single(double hol_en, double par_en, uint32_t orb_hol,
                          uint32_t orb_par, double orig_det_Hii) const;

  double fast_diag_ss_double(double en_hol1, double en_hol2, double en_par1,
                             double en_par2, uint32_t orb_hol1,
                             uint32_t orb_hol2, uint32_t orb_par1,
                             uint32_t orb_par2, double orig_det_Hii) const;

  double fast_diag_ss_double(const std::vector<uint32_t>& ss_occ,
                             const std::vector<uint32_t>& os_occ,
                             uint32_t orb_hol1, uint32_t orb_hol2,
                             uint32_t orb_par1, uint32_t orb_par2,
                             double orig_det_Hii) const;

  double fast_diag_os_double(double en_holu, double en_hold, double en_paru,
                             double en_pard, uint32_t orb_holu,
                             uint32_t orb_hold, uint32_t orb_paru,
                             uint32_t orb_pard, double orig_det_Hii) const;

  double fast_diag_os_double(const std::vector<uint32_t>& up_occ,
                             const std::vector<uint32_t>& do_occ,
                             uint32_t orb_holu, uint32_t orb_hold,
                             uint32_t orb_paru, uint32_t orb_pard,
                             double orig_det_Hii) const;

  void rotate_hamiltonian_ordm(const Scalar* ordm);
};

}  // namespace macis
