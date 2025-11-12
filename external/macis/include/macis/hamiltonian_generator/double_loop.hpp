/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator.hpp>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>

namespace macis {

template <typename WfnType>
class DoubleLoopHamiltonianGenerator : public HamiltonianGenerator<WfnType> {
 public:
  using base_type = HamiltonianGenerator<WfnType>;
  using full_det_t = typename base_type::full_det_t;
  using spin_det_t = typename base_type::spin_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;
  using matrix_span_t = typename base_type::matrix_span_t;
  using rank4_span_t = typename base_type::rank4_span_t;

  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

 protected:
  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    using wfn_traits = wavefunction_traits<WfnType>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    std::vector<index_t> colind, rowptr(nbra_dets + 1);
    std::vector<double> nzval;

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    rowptr[0] = 0;

    // Loop over bra determinants
    for (size_t i = 0; i < nbra_dets; ++i) {
      // if( (i%1000) == 0 ) std::cout << i << ", " << rowptr[i] << std::endl;
      const auto bra = *(bra_begin + i);

      size_t nrow = 0;
      if (wfn_traits::count(bra)) {
        // Separate out into alpha/beta components
        spin_det_t bra_alpha = wfn_traits::alpha_string(bra);
        spin_det_t bra_beta = wfn_traits::beta_string(bra);

        // Get occupied indices
        bits_to_indices(bra_alpha, bra_occ_alpha);
        bits_to_indices(bra_beta, bra_occ_beta);

        // Loop over ket determinants
        for (size_t j = 0; j < nket_dets; ++j) {
          const auto ket = *(ket_begin + j);
          if (wfn_traits::count(ket)) {
            spin_det_t ket_alpha = wfn_traits::alpha_string(ket);
            spin_det_t ket_beta = wfn_traits::beta_string(ket);

            full_det_t ex_total = bra ^ ket;
            if (wfn_traits::count(ex_total) <= 4) {
              spin_det_t ex_alpha = wfn_traits::alpha_string(ex_total);
              spin_det_t ex_beta = wfn_traits::beta_string(ex_total);

              // Compute Matrix Element
              const auto h_el = this->matrix_element(
                  bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                  bra_occ_alpha, bra_occ_beta);

              if (std::abs(h_el) > H_thresh) {
                nrow++;
                colind.emplace_back(j);
                nzval.emplace_back(h_el);
              }

            }  // Possible non-zero connection (Hamming distance)

          }  // Non-zero ket determinant
        }  // Loop over ket determinants

      }  // Non-zero bra determinant

      rowptr[i + 1] = rowptr[i] + nrow;  // Update rowptr

    }  // Loop over bra determinants

    colind.shrink_to_fit();
    nzval.shrink_to_fit();

    return sparse_matrix_type<index_t>(nbra_dets, nket_dets, std::move(rowptr),
                                       std::move(colind), std::move(nzval));
  }

  sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int32_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

  sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int64_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

 public:
  void form_rdms(full_det_iterator bra_begin, full_det_iterator bra_end,
                 full_det_iterator ket_begin, full_det_iterator ket_end,
                 double *C, matrix_span_t ordm, rank4_span_t trdm) override {
    using wfn_traits = wavefunction_traits<WfnType>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

    // Loop over bra determinants
    for (size_t i = 0; i < nbra_dets; ++i) {
      const auto bra = *(bra_begin + i);
      // if( (i%1000) == 0 ) std::cout << i  << std::endl;
      if (wfn_traits::count(bra)) {
        // Separate out into alpha/beta components
        spin_det_t bra_alpha = wfn_traits::alpha_string(bra);
        spin_det_t bra_beta = wfn_traits::beta_string(bra);

        // Get occupied indices
        bits_to_indices(bra_alpha, bra_occ_alpha);
        bits_to_indices(bra_beta, bra_occ_beta);

        // Loop over ket determinants
        for (size_t j = 0; j < nket_dets; ++j) {
          const auto ket = *(ket_begin + j);
          if (wfn_traits::count(ket)) {
            spin_det_t ket_alpha = wfn_traits::alpha_string(ket);
            spin_det_t ket_beta = wfn_traits::beta_string(ket);

            full_det_t ex_total = bra ^ ket;
            if (wfn_traits::count(ex_total) <= 4) {
              spin_det_t ex_alpha = wfn_traits::alpha_string(ex_total);
              spin_det_t ex_beta = wfn_traits::beta_string(ex_total);

              const double val = C[i] * C[j];

              // Compute Matrix Element
              if (std::abs(val) > 1e-16) {
                rdm_contributions(bra_alpha, ket_alpha, ex_alpha, bra_beta,
                                  ket_beta, ex_beta, bra_occ_alpha,
                                  bra_occ_beta, val, ordm, trdm);
              }
            }  // Possible non-zero connection (Hamming distance)

          }  // Non-zero ket determinant
        }  // Loop over ket determinants

      }  // Non-zero bra determinant
    }  // Loop over bra determinants
  }

 public:
  template <typename... Args>
  DoubleLoopHamiltonianGenerator(Args &&...args)
      : HamiltonianGenerator<WfnType>(std::forward<Args>(args)...) {}
};

}  // namespace macis
