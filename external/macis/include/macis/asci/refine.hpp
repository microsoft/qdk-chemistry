/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <macis/asci/iteration.hpp>
#include <macis/bitset_operations.hpp>

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
  assert_fast_bitset_word_access_runtime_once();
  auto logger = spdlog::get("asci_refine");
#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
#else
  int world_rank = 0;
#endif /* MACIS_ENABLE_MPI */
  if (!logger) {
    logger = world_rank ? spdlog::null_logger_mt("asci_refine")
                        : spdlog::stdout_color_mt("asci_refine");
    logger->flush_on(logger->level());
  }

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
  double prev_E_delta = 0.0;
  int oscillation_count = 0;
  size_t max_iter = asci_settings.max_refine_iter;
  // Cap total extensions at the original budget to prevent runaway loops
  size_t total_extensions = 0;
  const size_t max_extensions = asci_settings.max_refine_iter;
  std::vector<wfn_t<N>> prev_wfn;  // saved for union stabilization
  // Incremental H_build cache — carried across refine iterations
  CachedHamiltonianState<wfn_t<N>, index_t> h_cache;

  for (size_t iter = 0; iter < max_iter; ++iter) {
    double E;
    std::tie(E, wfn, X) = asci_iter<N, index_t>(
        asci_settings, mcscf_settings, ndets, E0, std::move(wfn), std::move(X),
        ham_gen, norb, &h_cache MACIS_MPI_CODE(, comm));

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

    if (std::abs(E_delta) < asci_settings.refine_energy_tol) {
      E0 = E;
      converged = true;
      break;
    }

    // Detect oscillation: sign change in energy delta with similar magnitude
    if (iter > 0 && prev_E_delta * E_delta < 0 &&
        std::abs(prev_E_delta + E_delta) < asci_settings.refine_energy_tol) {
      oscillation_count++;

      if (oscillation_count >= 2 && !prev_wfn.empty()) {
        // Stabilize by rediagonalizing in the union of the two oscillating
        // determinant sets.
        using wfn_comp =
            typename wavefunction_traits<wfn_t<N>>::spin_comparator;

        std::vector<wfn_t<N>> union_wfn;
        union_wfn.reserve(wfn.size() + prev_wfn.size());
        std::set_union(prev_wfn.begin(), prev_wfn.end(), wfn.begin(), wfn.end(),
                       std::back_inserter(union_wfn), wfn_comp{});

        logger->info(
            "  Oscillation detected — stabilizing via union of last two "
            "det sets ({} + {} → {} unique dets)",
            prev_wfn.size(), wfn.size(), union_wfn.size());

        // Rediagonalize in the enlarged space — keep the cache since the
        // union is a superset of the cached dets (high overlap → patched
        // build). Warm-start from current X mapped onto the union ordering.
        std::vector<double> X_union(union_wfn.size(), 0.0);
        {
          // wfn is sorted and is a subset of union_wfn (also sorted).
          // Binary-search each wfn det into union_wfn to map coefficients.
          for (size_t i = 0; i < wfn.size(); ++i) {
            auto it = std::lower_bound(union_wfn.begin(), union_wfn.end(),
                                       wfn[i], wfn_comp{});
            if (it != union_wfn.end() && *it == wfn[i]) {
              X_union[static_cast<size_t>(
                  std::distance(union_wfn.begin(), it))] = X[i];
            }
          }
        }
        double E_union = selected_ci_diag<index_t>(
            union_wfn.begin(), union_wfn.end(), ham_gen,
            mcscf_settings.ci_matel_tol, mcscf_settings.ci_max_subspace,
            mcscf_settings.ci_res_tol, X_union, &h_cache,
            asci_settings.min_patch_overlap MACIS_MPI_CODE(, comm));

        // In MPI builds, X_union is the local portion — gather it.
#ifdef MACIS_ENABLE_MPI
        {
          auto ws = comm_size(comm);
          if (ws > 1) {
            const size_t n = union_wfn.size();
            const size_t lc = n / ws;
            std::vector<double> X_full(n);
            MPI_Allgather(X_union.data(), lc, MPI_DOUBLE, X_full.data(), lc,
                          MPI_DOUBLE, comm);
            if (n % ws) {
              auto wr = comm_rank(comm);
              auto* rem = X_full.data() + ws * lc;
              if (wr == ws - 1) std::copy_n(X_union.data() + lc, n % ws, rem);
              MPI_Bcast(rem, n % ws, MPI_DOUBLE, ws - 1, comm);
            }
            X_union = std::move(X_full);
          }
        }
#endif

        logger->info("  Union diag: E = {:20.12e}", E_union);

        // Extend iteration budget by the number of oscillation iterations
        // that were wasted, capped so we don't loop forever.
        const size_t extension =
            std::min(static_cast<size_t>(oscillation_count),
                     max_extensions - total_extensions);
        if (extension > 0) {
          max_iter += extension;
          total_extensions += extension;
          logger->info(
              "  Extending max_iter by {} → {} (total extended: {}/{})",
              extension, max_iter, total_extensions, max_extensions);
        }

        wfn = std::move(union_wfn);
        X = std::move(X_union);
        ndets = wfn.size();  // allow the enlarged set through
        E0 = E_union;
        prev_wfn.clear();
        oscillation_count = 0;
        prev_E_delta = 0.0;
        continue;  // re-enter refinement loop for real convergence check
      }
    } else {
      oscillation_count = 0;
    }
    prev_E_delta = E_delta;
    prev_wfn = wfn;

    E0 = E;
  }  // Refinement loop

  if (converged)
    logger->info("ASCI Refine Converged!");
  else {
    std::string msg = "ASCI Refine did not converge";
    if (total_extensions > 0) {
      msg += " (oscillation detected, " + std::to_string(total_extensions) +
             " extra iterations granted). "
             "Consider using percentage core_selection_strategy, increasing "
             "ncdets_max, or loosening refine_energy_tol.";
    }
    throw std::runtime_error(msg);
  }

  return std::make_tuple(E0, wfn, X);
}

}  // namespace macis
