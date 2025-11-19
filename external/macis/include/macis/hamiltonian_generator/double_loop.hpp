/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator.hpp>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>

namespace macis {

/**
 * @brief Double loop implementation of Hamiltonian generator
 *
 * This class implements a straightforward double loop algorithm for generating
 * Hamiltonian matrix elements between determinants. It uses nested loops over
 * bra and ket determinants to compute matrix elements directly.
 *
 * @tparam WfnType The wavefunction type that defines the determinant
 * representation
 */
template <typename WfnType>
class DoubleLoopHamiltonianGenerator : public HamiltonianGenerator<WfnType> {
 public:
  /// @brief Base class type alias
  using base_type = HamiltonianGenerator<WfnType>;
  /// @brief Full determinant type from base class
  using full_det_t = typename base_type::full_det_t;
  /// @brief Spin determinant type from base class
  using spin_det_t = typename base_type::spin_det_t;
  /// @brief Iterator type for determinants from base class
  using full_det_iterator = typename base_type::full_det_iterator;
  /// @brief Matrix span type from base class
  using matrix_span_t = typename base_type::matrix_span_t;
  /// @brief Rank-4 tensor span type from base class
  using rank4_span_t = typename base_type::rank4_span_t;

  /// @brief Sparse matrix type template
  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

 protected:
  /**
   * @brief Template method to create CSR Hamiltonian matrix block
   *
   * This method implements the core double loop algorithm to generate a sparse
   * Hamiltonian matrix block in CSR format between ranges of bra and ket
   * determinants.
   *
   * @tparam index_t Integer type for matrix indices
   * @param bra_begin Iterator to beginning of bra determinants
   * @param bra_end Iterator to end of bra determinants
   * @param ket_begin Iterator to beginning of ket determinants
   * @param ket_end Iterator to end of ket determinants
   * @param H_thresh Threshold for including matrix elements
   * @return CSR sparse matrix containing the Hamiltonian block
   */
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

  /**
   * @brief Create 32-bit CSR Hamiltonian matrix block (override)
   *
   * This method provides the interface for creating CSR matrices with 32-bit
   * indices.
   *
   * @param bra_begin Iterator to beginning of bra determinants
   * @param bra_end Iterator to end of bra determinants
   * @param ket_begin Iterator to beginning of ket determinants
   * @param ket_end Iterator to end of ket determinants
   * @param H_thresh Threshold for including matrix elements
   * @return 32-bit CSR sparse matrix
   */
  sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int32_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

  /**
   * @brief Create 64-bit CSR Hamiltonian matrix block (override)
   *
   * This method provides the interface for creating CSR matrices with 64-bit
   * indices.
   *
   * @param bra_begin Iterator to beginning of bra determinants
   * @param bra_end Iterator to end of bra determinants
   * @param ket_begin Iterator to beginning of ket determinants
   * @param ket_end Iterator to end of ket determinants
   * @param H_thresh Threshold for including matrix elements
   * @return 64-bit CSR sparse matrix
   */
  sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int64_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

 public:
  /**
   * @brief Form reduced density matrices (RDMs) from determinant expansions
   *
   * This method computes one-electron and two-electron reduced density matrices
   * by iterating over pairs of determinants and accumulating their
   * contributions.
   *
   * @param bra_begin Iterator to beginning of bra determinants
   * @param bra_end Iterator to end of bra determinants
   * @param ket_begin Iterator to beginning of ket determinants
   * @param ket_end Iterator to end of ket determinants
   * @param C Coefficient array for the determinants
   * @param ordm Matrix span for one-electron reduced density matrix
   * @param trdm Rank-4 tensor span for two-electron reduced density matrix
   */
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
                rdm_contributions<false>(
                    bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                    bra_occ_alpha, bra_occ_beta, val, ordm, trdm);
              }
            }  // Possible non-zero connection (Hamming distance)

          }  // Non-zero ket determinant
        }  // Loop over ket determinants

      }  // Non-zero bra determinant
    }  // Loop over bra determinants
  }

  void form_rdms_spin_dep(full_det_iterator bra_begin,
                          full_det_iterator bra_end,
                          full_det_iterator ket_begin,
                          full_det_iterator ket_end, double *C,
                          matrix_span_t ordm_aa, matrix_span_t ordm_bb,
                          rank4_span_t trdm_aaaa, rank4_span_t trdm_bbbb,
                          rank4_span_t trdm_aabb) override {
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
                rdm_contributions_spin_dep<false>(
                    bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                    bra_occ_alpha, bra_occ_beta, val, ordm_aa, ordm_bb,
                    trdm_aaaa, trdm_bbbb, trdm_aabb);
              }
            }  // Possible non-zero connection (Hamming distance)

          }  // Non-zero ket determinant
        }  // Loop over ket determinants

      }  // Non-zero bra determinant
    }  // Loop over bra determinants
  }

 public:
  /**
   * @brief Constructor for DoubleLoopHamiltonianGenerator
   *
   * Perfect forwarding constructor that passes all arguments to the base class.
   *
   * @tparam Args Parameter pack for constructor arguments
   * @param args Arguments to forward to HamiltonianGenerator constructor
   */
  template <typename... Args>
  DoubleLoopHamiltonianGenerator(Args &&...args)
      : HamiltonianGenerator<WfnType>(std::forward<Args>(args)...) {}
};

}  // namespace macis
