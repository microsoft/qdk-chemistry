/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/asci/determinant_search.hpp>
#include <macis/mcscf/mcscf.hpp>

namespace macis {

/**
 * @brief Optional single-rank executor for the numerical steps of ASCI.
 *
 * Grow, refine, core selection and warm starts remain in the shared ASCI
 * implementation. A null context selects the ordinary CPU implementation.
 * Contexts are borrowed, non-reentrant, and may retain integral-dependent
 * state across iterations. Call update_integrals after changing integrals.
 */
template <size_t N>
class ASCIExecutionContext {
 public:
  using Hamiltonian = HamiltonianGenerator<wfn_t<N>>;
  using Iterator = wavefunction_iterator_t<N>;

  virtual ~ASCIExecutionContext() noexcept = default;

  /// Synchronous search of alpha-major core determinants and aligned
  /// coefficients.
  virtual std::vector<wfn_t<N>> search(ASCISettings settings, size_t ndets_max,
                                       Iterator begin, Iterator end, double E0,
                                       const std::vector<double>& C,
                                       size_t norb, Hamiltonian& ham_gen) = 0;

  /**
   * @brief Begin diagonalization, possibly overlapping Hamiltonian
   * construction.
   *
   * At most one operation may be outstanding. The sorted, contiguous
   * determinants and ham_gen must remain alive and unmodified until finish
   * or drain. Capture settings by value: the caller need not retain them.
   */
  virtual void start_diagonalization(Iterator begin, Iterator end,
                                     const Hamiltonian& ham_gen,
                                     const MCSCFSettings& settings) = 0;

  /// Complete the pending solve; C is the guess on input and eigenvector on
  /// output.
  virtual double finish_diagonalization(std::vector<double>& C) = 0;

  /// Refresh integral-dependent state while no diagonalization is outstanding.
  virtual void update_integrals(const Hamiltonian& ham_gen) = 0;

  /// Join outstanding accesses during unwinding without replacing the
  /// exception.
  virtual void drain() noexcept = 0;
};

/// Fence borrowed input lifetimes even if preparation or diagonalization
/// throws.
template <size_t N>
class ASCIExecutionGuard {
 public:
  explicit ASCIExecutionGuard(ASCIExecutionContext<N>* context)
      : _context(context) {}
  ~ASCIExecutionGuard() noexcept {
    if (_context) _context->drain();
  }
  ASCIExecutionGuard(const ASCIExecutionGuard&) = delete;
  ASCIExecutionGuard& operator=(const ASCIExecutionGuard&) = delete;

 private:
  ASCIExecutionContext<N>* _context;
};

}  // namespace macis
