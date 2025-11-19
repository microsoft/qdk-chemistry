/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator/base.hpp>
#include <macis/sd_operations.hpp>

namespace macis {

/**
 *  @brief Template class for generating Hamiltonian matrices
 *
 *  HamiltonianGenerator is an abstract base class that provides interfaces
 *  for computing matrix elements of quantum mechanical Hamiltonians between
 *  Slater determinants. It supports various excitation types (single, double,
 *  quadruple) and can generate sparse matrix representations suitable for
 *  Configuration Interaction (CI) calculations.
 *
 *  The class also provides functionality for computing reduced density matrices
 *  (RDMs) which are essential for property calculations and post-processing
 *  of CI wavefunctions.
 *
 *  @tparam WfnType Type representing wavefunction determinants (typically
 * bitsets)
 */
template <typename WfnType>
class HamiltonianGenerator : public HamiltonianGeneratorBase<double> {
 public:
  /// Type alias for full determinant representation
  using full_det_t = WfnType;
  /// Type alias for spin-specific determinant representation
  using spin_det_t = spin_wfn_t<WfnType>;

  /// Type alias for container of full determinants
  using full_det_container = std::vector<WfnType>;
  /// Type alias for iterator over full determinants
  using full_det_iterator = typename full_det_container::iterator;

  /**
   *  @brief Create 32-bit CSR Hamiltonian matrix block (pure virtual)
   *
   *  Pure virtual function that must be implemented by derived classes to
   *  generate a CSR matrix block using 32-bit indices for small systems.
   *
   *  @param[in] bra_begin Iterator to beginning of bra determinants
   *  @param[in] bra_end Iterator to end of bra determinants
   *  @param[in] ket_begin Iterator to beginning of ket determinants
   *  @param[in] ket_end Iterator to end of ket determinants
   *  @param[in] H_thresh Threshold for matrix element inclusion
   *  @returns 32-bit indexed CSR sparse matrix
   */
  virtual sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator, full_det_iterator, full_det_iterator,
      full_det_iterator, double) = 0;

  /**
   *  @brief Create 64-bit CSR Hamiltonian matrix block (pure virtual)
   *
   *  Pure virtual function that must be implemented by derived classes to
   *  generate a CSR matrix block using 64-bit indices for large systems.
   *
   *  @param[in] bra_begin Iterator to beginning of bra determinants
   *  @param[in] bra_end Iterator to end of bra determinants
   *  @param[in] ket_begin Iterator to beginning of ket determinants
   *  @param[in] ket_end Iterator to end of ket determinants
   *  @param[in] H_thresh Threshold for matrix element inclusion
   *  @returns 64-bit indexed CSR sparse matrix
   */
  virtual sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator, full_det_iterator, full_det_iterator,
      full_det_iterator, double) = 0;

 public:
  /**
   *  @brief Constructor for HamiltonianGenerator
   *
   *  Initializes the Hamiltonian generator with one-electron and two-electron
   *  integral tensors.
   *
   *  @param[in] T One-electron integral matrix
   *  @param[in] V Two-electron integral tensor
   */
  HamiltonianGenerator(matrix_span_t T, rank4_span_t V)
      : HamiltonianGeneratorBase<double>(T, V) {};

  /**
   *  @brief Virtual destructor
   *
   *  Default virtual destructor for proper cleanup of derived classes.
   */
  virtual ~HamiltonianGenerator() noexcept = default;

  /**
   *  @brief Compute matrix element for quadruple excitation
   *
   *  Calculates the Hamiltonian matrix element between bra and ket
   *  determinants that differ by four electrons (quadruple excitation).
   *
   *  @param[in] bra Bra spin determinant
   *  @param[in] ket Ket spin determinant
   *  @param[in] ex Excitation pattern between bra and ket
   *  @returns Hamiltonian matrix element value
   */
  double matrix_element_4(spin_det_t bra, spin_det_t ket, spin_det_t ex) const;

  /**
   *  @brief Compute matrix element for double excitation (alpha-beta)
   *
   *  Calculates the Hamiltonian matrix element for determinants that differ
   *  by two electrons, one in alpha and one in beta spin channels.
   *
   *  @param[in] bra_alpha Alpha bra spin determinant
   *  @param[in] ket_alpha Alpha ket spin determinant
   *  @param[in] ex_alpha Alpha excitation pattern
   *  @param[in] bra_beta Beta bra spin determinant
   *  @param[in] ket_beta Beta ket spin determinant
   *  @param[in] ex_beta Beta excitation pattern
   *  @returns Hamiltonian matrix element value
   */
  double matrix_element_22(spin_det_t bra_alpha, spin_det_t ket_alpha,
                           spin_det_t ex_alpha, spin_det_t bra_beta,
                           spin_det_t ket_beta, spin_det_t ex_beta) const;

  /**
   *  @brief Compute matrix element for double excitation (same spin)
   *
   *  Calculates the Hamiltonian matrix element for determinants that differ
   *  by two electrons in the same spin channel.
   *
   *  @param[in] bra Bra spin determinant
   *  @param[in] ket Ket spin determinant
   *  @param[in] ex Excitation pattern between bra and ket
   *  @param[in] bra_occ_alpha Alpha occupied orbitals in bra
   *  @param[in] bra_occ_beta Beta occupied orbitals in bra
   *  @returns Hamiltonian matrix element value
   */
  double matrix_element_2(spin_det_t bra, spin_det_t ket, spin_det_t ex,
                          const std::vector<uint32_t>& bra_occ_alpha,
                          const std::vector<uint32_t>& bra_occ_beta) const;

  /**
   *  @brief Compute diagonal matrix element
   *
   *  Calculates the diagonal Hamiltonian matrix element for a determinant
   *  (expectation value of the Hamiltonian).
   *
   *  @param[in] occ_alpha Alpha occupied orbital indices
   *  @param[in] occ_beta Beta occupied orbital indices
   *  @returns Diagonal Hamiltonian matrix element value
   */
  double matrix_element_diag(const std::vector<uint32_t>& occ_alpha,
                             const std::vector<uint32_t>& occ_beta) const;

  /**
   *  @brief Compute general matrix element with occupation information
   *
   *  General interface for computing Hamiltonian matrix elements between
   *  spin determinants, with precomputed occupation vectors for efficiency.
   *
   *  @param[in] bra_alpha Alpha bra spin determinant
   *  @param[in] ket_alpha Alpha ket spin determinant
   *  @param[in] ex_alpha Alpha excitation pattern
   *  @param[in] bra_beta Beta bra spin determinant
   *  @param[in] ket_beta Beta ket spin determinant
   *  @param[in] ex_beta Beta excitation pattern
   *  @param[in] bra_occ_alpha Alpha occupied orbitals in bra
   *  @param[in] bra_occ_beta Beta occupied orbitals in bra
   *  @returns Hamiltonian matrix element value
   */
  double matrix_element(spin_det_t bra_alpha, spin_det_t ket_alpha,
                        spin_det_t ex_alpha, spin_det_t bra_beta,
                        spin_det_t ket_beta, spin_det_t ex_beta,
                        const std::vector<uint32_t>& bra_occ_alpha,
                        const std::vector<uint32_t>& bra_occ_beta) const;

  /**
   *  @brief Compute matrix element between full determinants
   *
   *  High-level interface for computing Hamiltonian matrix elements
   *  between full (alpha + beta) determinants.
   *
   *  @param[in] bra Full bra determinant
   *  @param[in] ket Full ket determinant
   *  @returns Hamiltonian matrix element value
   */
  double matrix_element(full_det_t bra, full_det_t ket) const;

  /**
   *  @brief Create CSR Hamiltonian matrix block
   *
   *  Generates a Compressed Sparse Row (CSR) matrix representation of the
   *  Hamiltonian for specified ranges of bra and ket determinants.
   *
   *  @tparam index_t Integer type for matrix indices (int32_t or int64_t)
   *
   *  @param[in] bra_begin Iterator to beginning of bra determinants
   *  @param[in] bra_end Iterator to end of bra determinants
   *  @param[in] ket_begin Iterator to beginning of ket determinants
   *  @param[in] ket_end Iterator to end of ket determinants
   *  @param[in] H_thresh Threshold for matrix element inclusion
   *  @returns CSR sparse matrix representation
   */
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

  /**
   *  @brief Form reduced density matrices
   *
   *  Pure virtual function to compute one-electron and two-electron
   *  reduced density matrices from CI coefficients and determinant ranges.
   *  Must be implemented by derived classes.
   *
   *  @param[in] bra_begin Iterator to beginning of bra determinants
   *  @param[in] bra_end Iterator to end of bra determinants
   *  @param[in] ket_begin Iterator to beginning of ket determinants
   *  @param[in] ket_end Iterator to end of ket determinants
   *  @param[in] C CI coefficient array
   *  @param[in,out] ordm One-electron reduced density matrix
   *  @param[in,out] trdm Two-electron reduced density matrix
   */
  virtual void form_rdms(full_det_iterator, full_det_iterator,
                         full_det_iterator, full_det_iterator, double* C,
                         matrix_span_t ordm, rank4_span_t trdm) = 0;

  /**
   *  @brief Form spin-dependent reduced density matrices
   *
   *  Pure virtual function to compute spin-separated one-electron and
   *  two-electron reduced density matrices from CI coefficients and
   *  determinant ranges. This provides more detailed information than
   *  the spin-averaged RDMs by separating alpha-alpha, beta-beta, and
   *  alpha-beta contributions. Must be implemented by derived classes.
   *
   *  @param[in] bra_begin Iterator to beginning of bra determinants
   *  @param[in] bra_end Iterator to end of bra determinants
   *  @param[in] ket_begin Iterator to beginning of ket determinants
   *  @param[in] ket_end Iterator to end of ket determinants
   *  @param[in] C CI coefficient array
   *  @param[in,out] ordm_aa Alpha-alpha one-electron reduced density matrix
   *  @param[in,out] ordm_bb Beta-beta one-electron reduced density matrix
   *  @param[in,out] trdm_aaaa Alpha-alpha-alpha-alpha two-electron RDM
   *  @param[in,out] trdm_bbbb Beta-beta-beta-beta two-electron RDM
   *  @param[in,out] trdm_aabb Alpha-alpha-beta-beta two-electron RDM
   */
  virtual void form_rdms_spin_dep(full_det_iterator, full_det_iterator,
                                  full_det_iterator, full_det_iterator,
                                  double* C, matrix_span_t ordm_aa,
                                  matrix_span_t ordm_bb, rank4_span_t trdm_aaaa,
                                  rank4_span_t trdm_bbbb,
                                  rank4_span_t trdm_aabb) = 0;
};

}  // namespace macis

// Implementation
#include <macis/hamiltonian_generator/impl.hpp>
