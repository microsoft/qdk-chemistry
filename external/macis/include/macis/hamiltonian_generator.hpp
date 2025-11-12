/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator/base.hpp>
#include <macis/sd_operations.hpp>

namespace macis {

template <typename WfnType>
class HamiltonianGenerator : public HamiltonianGeneratorBase<double> {
 public:
  using full_det_t = WfnType;
  using spin_det_t = spin_wfn_t<WfnType>;

  using full_det_container = std::vector<WfnType>;
  using full_det_iterator = typename full_det_container::iterator;

  virtual sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator, full_det_iterator, full_det_iterator,
      full_det_iterator, double) = 0;

  virtual sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator, full_det_iterator, full_det_iterator,
      full_det_iterator, double) = 0;

 public:
  HamiltonianGenerator(matrix_span_t T, rank4_span_t V)
      : HamiltonianGeneratorBase<double>(T, V) {};

  virtual ~HamiltonianGenerator() noexcept = default;

  double matrix_element_4(spin_det_t bra, spin_det_t ket, spin_det_t ex) const;
  double matrix_element_22(spin_det_t bra_alpha, spin_det_t ket_alpha,
                           spin_det_t ex_alpha, spin_det_t bra_beta,
                           spin_det_t ket_beta, spin_det_t ex_beta) const;

  double matrix_element_2(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                          const std::vector<uint32_t>& bra_occ_alpha,
                          const std::vector<uint32_t>& bra_occ_beta) const;

  double matrix_element_diag(const std::vector<uint32_t>& occ_alpha,
                             const std::vector<uint32_t>& occ_beta) const;

  double matrix_element(spin_det_t bra_alpha, spin_det_t ket_alpha,
                        spin_det_t ex_alpha, spin_det_t bra_beta,
                        spin_det_t ket_beta, spin_det_t ex_beta,
                        const std::vector<uint32_t>& bra_occ_alpha,
                        const std::vector<uint32_t>& bra_occ_beta) const;

  double matrix_element(full_det_t bra, full_det_t ket) const;

  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    if constexpr (std::is_same_v<index_t, int32_t>)
      return make_csr_hamiltonian_block_32bit_(bra_begin, bra_end, ket_begin,
                                               ket_end, H_thresh);
    else if constexpr (std::is_same_v<index_t, int64_t>)
      return make_csr_hamiltonian_block_64bit_(bra_begin, bra_end, ket_begin,
                                               ket_end, H_thresh);
    else {
      throw std::runtime_error("Unsupported index_t");
      abort();
    }
  }

  void rdm_contributions_4(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                           double val, rank4_span_t trdm);
  void rdm_contributions_22(spin_det_t bra_alpha, spin_det_t ket_alpha,
                            spin_det_t ex_alpha, spin_det_t bra_beta,
                            spin_det_t ket_beta, spin_det_t ex_beta, double val,
                            rank4_span_t trdm);
  void rdm_contributions_2(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                           const std::vector<uint32_t>& bra_occ_alpha,
                           const std::vector<uint32_t>& bra_occ_beta,
                           double val, matrix_span_t ordm, rank4_span_t trdm);
  void rdm_contributions_diag(const std::vector<uint32_t>& occ_alpha,
                              const std::vector<uint32_t>& occ_beta, double val,
                              matrix_span_t ordm, rank4_span_t trdm);

  void rdm_contributions(spin_det_t bra_alpha, spin_det_t ket_alpha,
                         spin_det_t ex_alpha, spin_det_t bra_beta,
                         spin_det_t ket_beta, spin_det_t ex_beta,
                         const std::vector<uint32_t>& bra_occ_alpha,
                         const std::vector<uint32_t>& bra_occ_beta, double val,
                         matrix_span_t ordm, rank4_span_t trdm);

  virtual void form_rdms(full_det_iterator, full_det_iterator,
                         full_det_iterator, full_det_iterator, double* C,
                         matrix_span_t ordm, rank4_span_t trdm) = 0;

  virtual void SetJustSingles(bool /*_js*/) {}
  virtual bool GetJustSingles() { return false; }
  // virtual size_t GetNimp() const { return N / 2; }
};

}  // namespace macis

// Implementation
#include <macis/hamiltonian_generator/impl.hpp>
