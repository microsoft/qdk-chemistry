/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/asci/iteration.hpp>

namespace macis {

/**
 * @brief Perform ASCI (Adaptive Sampling Configuration Interaction)
 * wavefunction refinement phase
 *
 * This function implements the refinement phase of the ASCI algorithm, where
 * the size of the wave function is kept fixed while iteratively improving
 * the configurations and CI coefficients and energy until convergence.
 *
 * @tparam N Size of the wavefunction bitset representation
 * @tparam index_t Integer type for indexing operations (default: int32_t)
 *
 * @param[in] asci_settings ASCI algorithm parameters including refinement
 * tolerance
 * @param[in] mcscf_settings MCSCF parameters for CI diagonalization
 * @param[in] E0 Initial reference energy from growth phase
 * @param[in] wfn Final wavefunction determinants from growth phase
 * @param[in] X Initial CI coefficients corresponding to wavefunction
 * @param[in] ham_gen Hamiltonian generator containing integrals and methods
 * @param[in] norb Number of molecular orbitals
 * @param[in] comm MPI communicator for parallel execution (MPI builds only)
 *
 * @return Tuple containing:
 *   - Final refined energy
 *   - Unchanged wavefunction determinants
 *   - Refined CI coefficients
 *
 * @see asci_iter, asci_grow
 */
template <size_t N, typename index_t = int32_t>
auto asci_refine(ASCISettings asci_settings, MCSCFSettings mcscf_settings,
                 double E0, std::vector<wfn_t<N>> wfn, std::vector<double> X,
                 HamiltonianGenerator<wfn_t<N>>& ham_gen,
                 size_t norb MACIS_MPI_CODE(, MPI_Comm comm)) {
  auto logger = spdlog::get("asci_refine");
#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
#else
  int world_rank = 0;
#endif /* MACIS_ENABLE_MPI */
  if (!logger)
    logger = world_rank ? spdlog::null_logger_mt("asci_refine")
                        : spdlog::stdout_color_mt("asci_refine");

  logger->info("[ASCI Refine Settings]:");
  logger->info(
      "  NTDETS = {:6}, NCDETS = {:6}, MAX_REFINE_ITER = {:4}, REFINE_TOL = "
      "{:.2e}",
      wfn.size(), asci_settings.ncdets_max, asci_settings.max_refine_iter,
      asci_settings.refine_energy_tol);

  constexpr const char* fmt_string =
      "iter = {:4}, E0 = {:20.12e}, dE = {:14.6e}";

  logger->info(fmt_string, 0, E0, 0.0);

  // Refinement Loop
  size_t ndets = wfn.size();
  bool converged = false;
  for (size_t iter = 0; iter < asci_settings.max_refine_iter; ++iter) {
    double E;
    std::tie(E, wfn, X) = asci_iter<N, index_t>(
        asci_settings, mcscf_settings, ndets, E0, std::move(wfn), std::move(X),
        ham_gen, norb MACIS_MPI_CODE(, comm));

    // Check if wavefunction size changed
    if (wfn.size() != ndets) {
      logger->warn(
          "Wavefunction size changed from {} to {} during refinement iteration "
          "{}",
          ndets, wfn.size(), iter + 1);

      // Update target size for next iteration
      ndets = wfn.size();

      // If wavefunction became too small, stop refinement
      if (wfn.size() < asci_settings.ntdets_min) {
        logger->error(
            "Wavefunction shrunk below ntdets_min ({}), stopping refinement",
            asci_settings.ntdets_min);
        break;
      }
    }

    const auto E_delta = E - E0;
    logger->info(fmt_string, iter + 1, E, E_delta);
    E0 = E;
    if (std::abs(E_delta) < asci_settings.refine_energy_tol) {
      converged = true;
      break;
    }
  }  // Refinement loop

  if (converged)
    logger->info("ASCI Refine Converged!");
  else
    throw std::runtime_error("ASCI Refine did not converge");

  return std::make_tuple(E0, wfn, X);
}

}  // namespace macis
