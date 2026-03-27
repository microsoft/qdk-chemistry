/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/asci/determinant_sort.hpp>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <macis/util/dist_quickselect.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/mpi.hpp>

namespace macis {

/**
 * @brief Strategy for selecting core determinants in ASCI calculations
 *
 * Determines how the number of core determinants is chosen from the
 * trial determinant set during ASCI refinement iterations.
 */
enum class CoreSelectionStrategy {
  /// @brief Use a fixed number of core determinants
  Fixed,
  /// @brief Select core determinants to capture a target percentage of
  /// wavefunction weight
  Percentage
};

/**
 * @brief Convert CoreSelectionStrategy enum to string representation
 * @param strategy The core selection strategy to convert
 * @return String representation ("fixed" or "percentage")
 */
inline std::string core_selection_strategy_to_string(
    CoreSelectionStrategy strategy) {
  switch (strategy) {
    case CoreSelectionStrategy::Fixed:
      return "fixed";
    case CoreSelectionStrategy::Percentage:
      return "percentage";
  }
  throw std::invalid_argument("Invalid CoreSelectionStrategy value");
}

/**
 * @brief Comparator for selecting top-k ASCI contributions
 *
 * This comparator orders ASCI contributions by their absolute ratio value (rv)
 * in descending order, allowing efficient selection of the most important
 * determinant contributions for variational space expansion.
 *
 * @tparam WfnT Wavefunction type representing quantum states
 */
template <typename WfnT>
struct asci_contrib_topk_comparator {
  using type = asci_contrib<WfnT>;

  /**
   * @brief Compare two ASCI contributions by their absolute ratio values
   * @param a First ASCI contribution
   * @param b Second ASCI contribution
   * @return true if |a.rv()| > |b.rv()|, false otherwise
   */
  constexpr bool operator()(const type& a, const type& b) const {
    return std::abs(a.rv()) > std::abs(b.rv());
  }
};

/**
 * @brief Configuration parameters for ASCI (Adaptive Sampling Configuration
 * Interaction) calculations
 *
 * This structure contains all the settings and thresholds used to control the
 * behavior of ASCI calculations, including determinant selection criteria,
 * perturbation theory corrections, parallelization parameters, and convergence
 * tolerances.
 */
struct ASCISettings {
  /// @brief Maximum number of trial determinants in the variational space
  size_t ntdets_max = 1e5;
  /// @brief Minimum number of trial determinants required
  size_t ntdets_min = 100;
  /// @brief Strategy for selecting the number of core determinants
  CoreSelectionStrategy core_selection_strategy =
      CoreSelectionStrategy::Percentage;
  /// @brief Threshold for percentage-based core selection (fraction of
  /// wavefunction weight to retain, 0.0-1.0)
  double core_selection_threshold = 0.95;
  /// @brief Maximum number of core determinants
  size_t ncdets_max = 100;
  /// @brief Threshold for Hamiltonian matrix element magnitude
  double h_el_tol = 1e-8;
  /// @brief Threshold for ratio value pruning in determinant selection
  double rv_prune_tol = 1e-8;
  /// @brief Maximum number of ASCI contribution pairs to store in memory
  size_t pair_size_max = 5e8;

  /// @brief Tolerance for second-order perturbation theory corrections
  double pt2_tol = 1e-16;
  /// @brief Reserve count for PT2 calculations
  size_t pt2_reserve_count = 70000000;
  /// @brief Enable pruning in PT2 calculations
  bool pt2_prune = false;
  /// @brief Precompute orbital energies for PT2
  bool pt2_precompute_eps = false;
  /// @brief Precompute indices for PT2
  bool pt2_precompute_idx = false;
  /// @brief Print progress information during PT2 calculations
  bool pt2_print_progress = false;
  /// @brief Threshold for big constraint handling in PT2
  size_t pt2_bigcon_thresh = 250;

  /// @brief Threshold for next value batch count
  size_t nxtval_bcount_thresh = 1000;
  /// @brief Increment for next value batch count
  size_t nxtval_bcount_inc = 10;

  /// @brief If true, only consider single excitations (no doubles)
  bool just_singles = false;
  /// @brief Growth factor for determinant space expansion
  double grow_factor = 8.0;
  /// @brief Minimum allowed growth factor
  double min_grow_factor = 1.01;
  /// @brief Rate to reduce grow_factor on failure
  double growth_backoff_rate = 0.5;
  /// @brief Rate to restore grow_factor on success
  double growth_recovery_rate = 1.1;

  /// @brief Maximum number of refinement iterations
  size_t max_refine_iter = 6;
  /// @brief Energy convergence tolerance for refinement
  double refine_energy_tol = 1e-6;

  /// @brief Enable warm-starting Davidson from previous eigenvector.
  /// When true, CI coefficients from the previous iteration are mapped
  /// onto the new determinant set and used as the initial guess.
  bool warm_start_davidson = true;

  /// @brief Minimum determinant overlap fraction for warm-starting Davidson.
  /// If the fraction of new dets also present in the old set falls below
  /// this, the warm-start guess is discarded in favour of a diagonal guess.
  double min_warm_start_overlap = 0.5;

  /// @brief Minimum determinant overlap fraction for using incremental
  /// (patched) Hamiltonian build.  Below this threshold the full CSR is
  /// rebuilt from scratch.  Range: (0, 1].
  // This is misleading as it will still enforce alternation. We should check whether that's actually required or if it just requires tinkering with the grow_factor parameters.
  double min_patch_overlap = 0.3;

  /// @brief CI residual tolerance used during the grow phase.
  /// A looser tolerance here speeds up growth iterations where tight
  /// convergence is unnecessary.  Set to 0 to use the refine tolerance.
  double grow_ci_residual_tolerance = 0;

  /// @brief Growth factor applied when the next expansion would exceed
  /// ntdets_max.  0 disables tapering (the normal grow_factor is used).
  double taper_grow_factor = 0;

  /// @brief Enable growing with rotations
  bool grow_with_rot = false;
  /// @brief Starting size for rotations
  size_t rot_size_start = 1000;

  /// @brief Constraint level for excitation generation (0=triplets,
  /// 1=quadruplets, 2=quintuplets, etc.)
  int constraint_level = 2;
  /// @brief Maximum constraint level for PT2 calculations
  int pt2_max_constraint_level = 5;
  /// @brief Minimum constraint level for PT2 calculations
  int pt2_min_constraint_level = 0;
  /// @brief Force constraint refinement for PT2 calculations
  int64_t pt2_constraint_refine_force = 0;

  /// @brief Hamiltonian build algorithm for diagonal blocks.
  /// "" or "sorted_double_loop" = default alpha-sorted double loop;
  /// "residue_arrays" = Algorithm 5 from Tubman et al. (arXiv:1807.00821);
  /// "dynamic_bit_masking" = Algorithm 6 from Tubman et al.
  std::string h_build_algo;
};

/**
 * @brief Generate ASCI contributions using standard determinant-by-determinant
 * approach
 *
 * This function computes ASCI contributions for all possible single and double
 * excitations from the current determinant set using a straightforward approach
 * that loops over each determinant individually. It generates matrix elements
 * for all connected excitations and stores them for later selection.
 *
 * @tparam N Size of the wavefunction bitset
 * @param[in] asci_settings Configuration parameters for the ASCI calculation
 * @param[in] cdets_begin Iterator to the beginning of core determinants
 * @param[in] cdets_end Iterator to the end of core determinants
 * @param[in] E_ASCI Reference energy for the ASCI calculation
 * @param[in] C Coefficients of the core determinants
 * @param[in] norb Number of orbitals in the system
 * @param[in] T_pq One-electron integral matrix
 * @param[in] G_red Reduced same-spin two-electron integral tensor
 * @param[in] V_red Reduced opposite-spin two-electron integral tensor
 * @param[in] G_pqrs Full same-spin two-electron integral tensor
 * @param[in] V_pqrs Full opposite-spin two-electron integral tensor
 * @param[in] ham_gen Hamiltonian generator for matrix element evaluation
 * @return Container of ASCI contributions with their associated scores
 */
template <size_t N>
asci_contrib_container<wfn_t<N>> asci_contributions_standard(
    ASCISettings asci_settings, wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs, HamiltonianGenerator<wfn_t<N>>& ham_gen) {
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  auto logger = spdlog::get("asci_search");

  const size_t ncdets = std::distance(cdets_begin, cdets_end);

  asci_contrib_container<wfn_t<N>> asci_pairs;
  std::vector<uint32_t> occ_alpha, vir_alpha;
  std::vector<uint32_t> occ_beta, vir_beta;
  asci_pairs.reserve(asci_settings.pair_size_max);
  for (size_t i = 0; i < ncdets; ++i) {
    // Alias state data
    auto state = *(cdets_begin + i);
    auto state_alpha = wfn_traits::alpha_string(state);
    auto state_beta = wfn_traits::beta_string(state);
    auto coeff = C[i];

    // Get occupied and virtual indices
    spin_wfn_traits::state_to_occ_vir(norb, state_alpha, occ_alpha, vir_alpha);
    spin_wfn_traits::state_to_occ_vir(norb, state_beta, occ_beta, vir_beta);

    // Precompute orbital energies
    auto eps_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
    auto eps_beta = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);

    // Compute base diagonal matrix element
    double h_diag = ham_gen.matrix_element(state, state);

    const double h_el_tol = asci_settings.h_el_tol;

    // Singles - AA
    append_singles_asci_contributions<Spin::Alpha>(
        coeff, state, state_alpha, occ_alpha, vir_alpha, occ_beta,
        eps_alpha.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol,
        h_diag, E_ASCI, ham_gen, asci_pairs);

    // Singles - BB
    append_singles_asci_contributions<Spin::Beta>(
        coeff, state, state_beta, occ_beta, vir_beta, occ_alpha,
        eps_beta.data(), T_pq, norb, G_red, norb, V_red, norb, h_el_tol, h_diag,
        E_ASCI, ham_gen, asci_pairs);

    if (not asci_settings.just_singles) {
      // Doubles - AAAA
      append_ss_doubles_asci_contributions<Spin::Alpha>(
          coeff, state, state_alpha, state_beta, occ_alpha, vir_alpha, occ_beta,
          eps_alpha.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
          asci_pairs);

      // Doubles - BBBB
      append_ss_doubles_asci_contributions<Spin::Beta>(
          coeff, state, state_beta, state_alpha, occ_beta, vir_beta, occ_alpha,
          eps_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI, ham_gen,
          asci_pairs);

      // Doubles - AABB
      append_os_doubles_asci_contributions(
          coeff, state, state_alpha, state_beta, occ_alpha, occ_beta, vir_alpha,
          vir_beta, eps_alpha.data(), eps_beta.data(), V_pqrs, norb, h_el_tol,
          h_diag, E_ASCI, ham_gen, asci_pairs);
    }

    // Prune Down Contributions
    if (asci_pairs.size() > asci_settings.pair_size_max) {
      // Remove small contributions
      auto it = std::partition(
          asci_pairs.begin(), asci_pairs.end(), [=](const auto& x) {
            return std::abs(x.rv()) > asci_settings.rv_prune_tol;
          });
      asci_pairs.erase(it, asci_pairs.end());
      logger->info("  * Pruning at DET = {} NSZ = {}", i, asci_pairs.size());

      // Extra Pruning if not sufficient
      if (asci_pairs.size() > asci_settings.pair_size_max) {
        logger->info("    * Removing Duplicates");
        sort_and_accumulate_asci_pairs(asci_pairs);
        logger->info("    * NSZ = {}", asci_pairs.size());
      }

    }  // Pruning
  }  // Loop over search determinants

  return asci_pairs;
}

/**
 * @brief Generate ASCI contributions using constraint-based approach for
 * parallel efficiency
 *
 * This function computes ASCI contributions using a constraint-based algorithm
 * that groups determinants by common alpha strings and generates excitations
 * systematically. This approach is more memory-efficient and parallelizes
 * better than the standard approach, particularly for large systems and
 * distributed computing environments.
 *
 * The algorithm works by:
 * 1. Grouping determinants by unique alpha strings
 * 2. Generating mask constraints for systematic excitation enumeration
 * 3. Processing constraints in parallel across MPI ranks and threads
 * 4. Accumulating and pruning contributions on-the-fly
 *
 * @tparam N Size of the wavefunction bitset
 * @param[in] asci_settings Configuration parameters for the ASCI calculation
 * @param[in] ntdets Target number of determinants for the expanded space
 * @param[in] cdets_begin Iterator to the beginning of core determinants
 * @param[in] cdets_end Iterator to the end of core determinants
 * @param[in] E_ASCI Reference energy for the ASCI calculation
 * @param[in] C Coefficients of the core determinants
 * @param[in] norb Number of orbitals in the system
 * @param[in] T_pq One-electron integral matrix
 * @param[in] G_red Reduced same-spin two-electron integral tensor
 * @param[in] V_red Reduced opposite-spin two-electron integral tensor
 * @param[in] G_pqrs Full same-spin two-electron integral tensor
 * @param[in] V_pqrs Full opposite-spin two-electron integral tensor
 * @param[in] ham_gen Hamiltonian generator for matrix element evaluation
 * @param[in] comm MPI communicator for parallel execution (if MPI enabled)
 * @return Container of ASCI contributions with their associated scores
 */
template <size_t N>
asci_contrib_container<wfn_t<N>> asci_contributions_constraint(
    ASCISettings asci_settings, const size_t ntdets,
    wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs,
    HamiltonianGenerator<wfn_t<N>>& ham_gen MACIS_MPI_CODE(, MPI_Comm comm)) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  using wfn_comp = typename wfn_traits::spin_comparator;
  if (!std::is_sorted(cdets_begin, cdets_end, wfn_comp{}))
    throw std::runtime_error("ASCI Search Only Works with Sorted Wfns");

  auto logger = spdlog::get("asci_search");
  const size_t ncdets = std::distance(cdets_begin, cdets_end);

#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
#else
  auto world_rank = 0;
  auto world_size = 1;
#endif /* MACIS_ENABLE_MPI */

  // For each unique alpha, create a list of beta string and store metadata
  /**
   * @brief Data structure for storing beta string metadata in constraint-based
   * ASCI
   *
   * This structure efficiently stores precomputed information for beta strings
   * that share the same alpha string, enabling fast evaluation of matrix
   * elements during excitation generation. All necessary data is precomputed to
   * avoid redundant calculations during the constraint processing loop.
   */
  struct beta_coeff_data {
    /// @brief The beta spin string component of the determinant
    spin_wfn_type beta_string;
    /// @brief Occupied orbital indices in the beta string
    std::vector<uint32_t> occ_beta;
    /// @brief Virtual orbital indices in the beta string
    std::vector<uint32_t> vir_beta;
    /// @brief Precomputed alpha orbital energies for this determinant
    std::vector<double> orb_ens_alpha;
    /// @brief Precomputed beta orbital energies for this determinant
    std::vector<double> orb_ens_beta;
    /// @brief Coefficient of this determinant in the wavefunction
    double coeff;
    /// @brief Diagonal Hamiltonian matrix element <det|H|det>
    double h_diag;

    /**
     * @brief Constructor to initialize beta coefficient data
     * @param[in] c Coefficient of the determinant
     * @param[in] norb Number of orbitals in the system
     * @param[in] occ_alpha Occupied alpha orbital indices
     * @param[in] w Full determinant wavefunction
     * @param[in] ham_gen Hamiltonian generator for matrix element computation
     */
    beta_coeff_data(double c, size_t norb,
                    const std::vector<uint32_t>& occ_alpha, wfn_t<N> w,
                    const HamiltonianGenerator<wfn_t<N>>& ham_gen) {
      coeff = c;

      beta_string = wfn_traits::beta_string(w);

      // Compute diagonal matrix element
      h_diag = ham_gen.matrix_element(w, w);

      // Compute occ/vir for beta string
      spin_wfn_traits::state_to_occ_vir(norb, beta_string, occ_beta, vir_beta);

      // Precompute orbital energies
      orb_ens_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
      orb_ens_beta = ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);
    }
  };

  auto uniq_alpha = get_unique_alpha(cdets_begin, cdets_end);
  const size_t nuniq_alpha = uniq_alpha.size();
  std::vector<wfn_t<N>> uniq_alpha_wfn(nuniq_alpha);
  std::transform(
      uniq_alpha.begin(), uniq_alpha.end(), uniq_alpha_wfn.begin(),
      [](const auto& p) { return wfn_traits::from_spin(p.first, 0); });

  using unique_alpha_data = std::vector<beta_coeff_data>;
  std::vector<unique_alpha_data> uad(nuniq_alpha);
  for (auto i = 0, iw = 0; i < nuniq_alpha; ++i) {
    std::vector<uint32_t> occ_alpha, vir_alpha;
    spin_wfn_traits::state_to_occ_vir(norb, uniq_alpha[i].first, occ_alpha,
                                      vir_alpha);

    const auto nbeta = uniq_alpha[i].second;
    uad[i].reserve(nbeta);
    for (auto j = 0; j < nbeta; ++j, ++iw) {
      const auto& w = *(cdets_begin + iw);
      uad[i].emplace_back(C[iw], norb, occ_alpha, w, ham_gen);
    }
  }

  // Precompute cumulative alpha offsets for direct indexing
  std::vector<size_t> uniq_alpha_ioff(nuniq_alpha);
  std::transform_exclusive_scan(
      uniq_alpha.begin(), uniq_alpha.end(), uniq_alpha_ioff.begin(), 0ul,
      std::plus<size_t>(), [](const auto& p) { return p.second; });

  const auto num_alpha_occupied_orbitals =
      spin_wfn_traits::count(uniq_alpha[0].first);
  const auto num_alpha_virtual_orbitals = norb - num_alpha_occupied_orbitals;
  const auto n_sing_alpha =
      num_alpha_occupied_orbitals * num_alpha_virtual_orbitals;
  const auto n_doub_alpha = (n_sing_alpha * (n_sing_alpha - norb + 1)) / 4;

  const auto num_beta_occupied_orbitals =
      cdets_begin->count() - num_alpha_occupied_orbitals;
  const auto num_beta_virtual_orbitals = norb - num_beta_occupied_orbitals;
  const auto n_sing_beta =
      num_beta_occupied_orbitals * num_beta_virtual_orbitals;
  const auto n_doub_beta = (n_sing_beta * (n_sing_beta - norb + 1)) / 4;

  // Generate mask constraints
  if (!world_rank) {
    std::string cl_string;
    switch (asci_settings.constraint_level) {
      case 0:
        cl_string = "Triplets";
        break;
      case 1:
        cl_string = "Quadruplets";
        break;
      case 2:
        cl_string = "Quintuplets";
        break;
      case 3:
        cl_string = "Hextuplets";
        break;
      default:
        cl_string = "Something I dont recognize (" +
                    std::to_string(asci_settings.constraint_level) + ")";
        break;
    }
    logger->info("  * Will Generate up to {}", cl_string);
  }

  // Use total worker count (MPI ranks * OpenMP threads) so constraints
  // are split finely enough for thread-level load balancing.
  int num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif
  const int total_workers = world_size * num_threads;

  logger->info("  * NUNIQ_ALPHA = {}, NCDETS = {}", nuniq_alpha, ncdets);
  logger->info("  * NS_ALPHA = {}, ND_ALPHA = {}", n_sing_alpha, n_doub_alpha);
  logger->info("  * NS_BETA  = {}, ND_BETA  = {}", n_sing_beta, n_doub_beta);
  logger->info("  * MPI_RANKS = {}, OMP_THREADS = {}, TOTAL_WORKERS = {}",
               world_size, num_threads, total_workers);

  auto gen_c_st = clock_type::now();
  auto constraints = gen_constraints_general<wfn_t<N>>(
      asci_settings.constraint_level, norb, n_sing_beta, n_doub_beta,
      uniq_alpha, total_workers);
  auto gen_c_en = clock_type::now();
  duration_type gen_c_dur = gen_c_en - gen_c_st;
  logger->info("  * NCONSTRAINTS = {}, GEN_DUR = {:.2e} ms", constraints.size(),
               gen_c_dur.count());

  // Log constraint work distribution for load-balance diagnostics (debug only)
  if (logger->should_log(spdlog::level::debug) && !constraints.empty()) {
    size_t total_cwork = 0, max_cwork = 0;
    for (const auto& [c, w] : constraints) {
      total_cwork += w;
      if (w > max_cwork) max_cwork = w;
    }
    size_t avg_per_worker = total_cwork / total_workers;
    logger->debug(
        "  * CONSTRAINT_WORK: total={}, max_single={}, avg/worker={}, "
        "max/avg={:.2f}",
        total_cwork, max_cwork, avg_per_worker,
        avg_per_worker > 0 ? static_cast<double>(max_cwork) / avg_per_worker
                           : 1.0);
  }

  size_t max_size =
      std::min(std::min(ntdets, asci_settings.pair_size_max),
               ncdets * (n_sing_alpha + n_sing_beta +  // AA + BB
                         n_doub_alpha + n_doub_beta +  // AAAA + BBBB
                         n_sing_alpha * n_sing_beta    // AABB
                         ));

  const size_t ncon_total = constraints.size();

// Global atomic task-id counter
#ifdef MACIS_ENABLE_MPI
  global_atomic<size_t> nxtval(comm);
#else
  std::atomic<size_t> nxtval(0);
#endif /* MACIS_ENABLE_MPI */

  // Each thread writes its accumulated results to its own slot;
  // merged serially after the parallel region (no omp critical needed).
  std::vector<asci_contrib_container<wfn_t<N>>> thread_pairs(num_threads);
  const bool debug_timing = logger->should_log(spdlog::level::debug);
  std::vector<double> thread_wall(debug_timing ? num_threads : 0, 0.0);
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const asci_contrib_topk_comparator<wfn_t<N>> contrib_cmp{};
    auto t_wall_st = clock_type::now();

    // Accumulated results from all constraints processed by this thread.
    asci_contrib_container<wfn_t<N>> accumulated_pairs;
    const size_t per_thread_reserve =
        std::max(size_t(1024), max_size / num_threads);
    accumulated_pairs.reserve(per_thread_reserve);

    // Working set for the current constraint — grows during excitation
    // generation, then gets S&A'd and merged into accumulated_pairs.
    // clear() preserves capacity, so after the first constraint this
    // buffer never allocates again.
    asci_contrib_container<wfn_t<N>> working_pairs;

    // Thread-local reusable buffers — capacity persists across iterations,
    // so subsequent calls reuse existing memory.
    std::vector<uint32_t> occ_alpha_buf;
    using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
    std::vector<spin_wfn_type> O_buf, V_buf;
    std::vector<uint32_t> virt_ind_buf, occ_ind_buf;

    size_t ic = 0;
    while (ic < ncon_total) {
      const double h_el_tol = asci_settings.h_el_tol;

      // Atomically get the next task ID and increment for other
      // MPI ranks and threads
      size_t ntake = ic < asci_settings.nxtval_bcount_thresh
                         ? 1
                         : asci_settings.nxtval_bcount_inc;
      ic = nxtval.fetch_add(ntake);

      // Loop over assigned tasks
      const size_t c_end = std::min(ncon_total, ic + ntake);
      for (; ic < c_end; ++ic) {
        const auto& con = constraints[ic].first;
        working_pairs.clear();

        for (size_t i_alpha = 0; i_alpha < nuniq_alpha; ++i_alpha) {
          const auto& alpha_det = uniq_alpha[i_alpha].first;
          const auto ncon_alpha = constraint_histogram(alpha_det, 1, 1, con);
          if (!ncon_alpha) continue;
          bits_to_indices(alpha_det, occ_alpha_buf);
          const auto& occ_alpha = occ_alpha_buf;
          const bool alpha_satisfies_con = satisfies_constraint(alpha_det, con);

          const auto& bcd = uad[i_alpha];
          const size_t nbeta = bcd.size();
          for (size_t j_beta = 0; j_beta < nbeta; ++j_beta) {
            const size_t iw = uniq_alpha_ioff[i_alpha] + j_beta;
            const auto w = *(cdets_begin + iw);
            const auto c = C[iw];
            const auto& beta_det = bcd[j_beta].beta_string;
            const auto h_diag = bcd[j_beta].h_diag;
            const auto& occ_beta = bcd[j_beta].occ_beta;
            const auto& vir_beta = bcd[j_beta].vir_beta;
            const auto& orb_ens_alpha = bcd[j_beta].orb_ens_alpha;
            const auto& orb_ens_beta = bcd[j_beta].orb_ens_beta;

            // AA excitations
            generate_constraint_singles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), T_pq,
                norb, G_red, norb, V_red, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, working_pairs);

            // AAAA excitations
            generate_constraint_doubles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), G_pqrs,
                norb, h_el_tol, h_diag, E_ASCI, ham_gen, working_pairs, O_buf,
                V_buf, virt_ind_buf, occ_ind_buf);

            // AABB excitations
            generate_constraint_doubles_contributions_os(
                c, w, con, occ_alpha, occ_beta, vir_beta, orb_ens_alpha.data(),
                orb_ens_beta.data(), V_pqrs, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, working_pairs);

            if (alpha_satisfies_con) {
              // BB excitations
              append_singles_asci_contributions<Spin::Beta>(
                  c, w, beta_det, occ_beta, vir_beta, occ_alpha,
                  orb_ens_beta.data(), T_pq, norb, G_red, norb, V_red, norb,
                  h_el_tol, h_diag, E_ASCI, ham_gen, working_pairs);

              // BBBB excitations
              append_ss_doubles_asci_contributions<Spin::Beta>(
                  c, w, beta_det, alpha_det, occ_beta, vir_beta, occ_alpha,
                  orb_ens_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI,
                  ham_gen, working_pairs);

              // No excitation (push inf to remove from list)
              working_pairs.push_back(
                  {w, std::numeric_limits<double>::infinity(), 1.0});
            }
          }

          // Prune working set if it gets too large
          if (working_pairs.size() > asci_settings.pair_size_max) {
            logger->info("  * PRUNING AT CON = {} IALPHA = {}", ic, i_alpha);
            auto uit = sort_and_accumulate_asci_pairs(working_pairs.begin(),
                                                      working_pairs.end());
            working_pairs.erase(uit, working_pairs.end());
          }

        }  // Unique Alpha Loop

        // S&A the working set and prune small contributions
        {
          sort_and_accumulate_asci_pairs(working_pairs);

          auto uit = std::partition(
              working_pairs.begin(), working_pairs.end(), [=](const auto& x) {
                return std::abs(x.rv()) > asci_settings.rv_prune_tol;
              });
          working_pairs.erase(uit, working_pairs.end());
        }

        // Merge into accumulated results
        if (ntdets) {
          accumulated_pairs.insert(
              accumulated_pairs.end(),
              std::make_move_iterator(working_pairs.begin()),
              std::make_move_iterator(working_pairs.end()));
          working_pairs.clear();

          if (accumulated_pairs.size() > ntdets) {
            const size_t cutoff_idx = ntdets - 1;
            std::nth_element(accumulated_pairs.begin(),
                             accumulated_pairs.begin() + cutoff_idx,
                             accumulated_pairs.end(), contrib_cmp);

            const double threshold =
                std::abs(accumulated_pairs[cutoff_idx].rv());

            // Keep all contributions tied at the cutoff so equal-|rv| entries
            // survive trimming
            auto new_end = std::partition(
                accumulated_pairs.begin(), accumulated_pairs.end(),
                [threshold](const auto& contrib) {
                  return std::abs(contrib.rv()) >= threshold;
                });
            accumulated_pairs.erase(new_end, accumulated_pairs.end());
          }
        } else {
          working_pairs.clear();
        }

        logger->info(
            "after debug, thread {} accumulated_pairs.size() = {}, ntdets = {} "
            "at CON = {}",
            tid, accumulated_pairs.size(), ntdets, ic);
      }  // Loc constraint loop
    }  // Constraint Loop

    // Move thread results to shared slot — no synchronization needed
    // since each thread writes to its own index.
    thread_pairs[tid] = std::move(accumulated_pairs);
    if (debug_timing) {
      auto t_wall_en = clock_type::now();
      thread_wall[tid] = duration_type(t_wall_en - t_wall_st).count();
    }

  }  // OpenMP

  // Log per-thread load balance (debug only)
  if (debug_timing) {
    double t_max = *std::max_element(thread_wall.begin(), thread_wall.end());
    double t_min = *std::min_element(thread_wall.begin(), thread_wall.end());
    double t_sum = 0;
    for (auto t : thread_wall) t_sum += t;
    double t_avg = t_sum / num_threads;
    logger->debug(
        "  * THREAD_WALL: min={:.3f}s max={:.3f}s avg={:.3f}s "
        "imbalance={:.2f}x efficiency={:.1f}%",
        t_min, t_max, t_avg, t_avg > 0 ? t_max / t_avg : 1.0,
        t_max > 0 ? 100.0 * t_avg / t_max : 100.0);
  }

  // Serial merge: compute total size, reserve once, concatenate.
  size_t total_size = 0;
  for (const auto& tp : thread_pairs) total_size += tp.size();
  asci_contrib_container<wfn_t<N>> asci_pairs_total;
  asci_pairs_total.reserve(total_size);
  for (auto& tp : thread_pairs) {
    asci_pairs_total.insert(asci_pairs_total.end(),
                            std::make_move_iterator(tp.begin()),
                            std::make_move_iterator(tp.end()));
    // Release thread-local memory immediately
    asci_contrib_container<wfn_t<N>>().swap(tp);
  }

  return asci_pairs_total;
}

/**
 * @brief Main ASCI determinant search algorithm
 *
 * This is the primary function for ASCI (Adaptive Sampling Configuration
 * Interaction) determinant selection. It expands the variational space by
 * identifying the most important determinants connected to the current
 * reference space through single and double excitations.
 *
 * The algorithm performs the following steps:
 * 1. Generate all possible excitation contributions from reference determinants
 * 2. Score each contribution using perturbative estimates
 * 3. Select the top-scoring determinants up to the specified limit
 * 4. Return the expanded determinant set for the next CI iteration
 *
 * The function uses either standard or constraint-based contribution generation
 * depending on the system size and parallelization requirements.
 *
 * @tparam N Size of the wavefunction bitset
 * @param[in] asci_settings Configuration parameters controlling the search
 * @param[in] ndets_max Maximum number of determinants to include in expanded
 * space
 * @param[in] cdets_begin Iterator to the beginning of current reference
 * determinants
 * @param[in] cdets_end Iterator to the end of current reference determinants
 * @param[in] E_ASCI Current ASCI energy estimate
 * @param[in] C Coefficients of the reference determinants
 * @param[in] norb Number of orbitals in the system
 * @param[in] T_pq One-electron integral matrix
 * @param[in] G_red Reduced same-spin two-electron integral tensor
 * @param[in] V_red Reduced opposite-spin two-electron integral tensor
 * @param[in] G_pqrs Full same-spin two-electron integral tensor
 * @param[in] V_pqrs Full opposite-spin two-electron integral tensor
 * @param[in] ham_gen Hamiltonian generator for matrix element evaluation
 * @param[in] comm MPI communicator for parallel execution (if MPI enabled)
 * @return Vector of determinants for the expanded variational space
 */
template <size_t N>
std::vector<wfn_t<N>> asci_search(
    ASCISettings asci_settings, size_t ndets_max,
    wavefunction_iterator_t<N> cdets_begin,
    wavefunction_iterator_t<N> cdets_end, const double E_ASCI,
    const std::vector<double>& C, size_t norb, const double* T_pq,
    const double* G_red, const double* V_red, const double* G_pqrs,
    const double* V_pqrs,
    HamiltonianGenerator<wfn_t<N>>& ham_gen MACIS_MPI_CODE(, MPI_Comm comm)) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double>;

  // MPI Info
#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
#else
  int world_rank = 0;
  int world_size = 1;
#endif /* MACIS_ENABLE_MPI */

  auto logger = spdlog::get("asci_search");
  if (!logger) {
    logger = world_rank ? spdlog::null_logger_mt("asci_search")
                        : spdlog::stdout_color_mt("asci_search");
    logger->flush_on(logger->level());
  }

#ifdef MACIS_ENABLE_MPI
  auto print_mpi_stats = [&](auto str, auto vmin, auto vmax, auto vavg) {
    constexpr const char* fmt_string =
        "    * {0}_MIN = {1}, {0}_MAX = {2}, {0}_AVG = {3}, RATIO = {4:.2e}";
    if constexpr (std::is_floating_point_v<std::decay_t<decltype(vmin)>>)
      logger->info(
          "    * {0}_MIN = {1:.2e}, {0}_MAX = {2:.2e}, {0}_AVG = {3:.2e}, "
          "RATIO = {4:.2e}",
          str, vmin, vmax, vavg, vmax / float(vmin));
    else
      logger->info(fmt_string, str, vmin, vmax, vavg, vmax / float(vmin));
  };
#endif /* MACIS_ENABLE_MPI */

  // Print Search Header to logger
  const size_t ncdets = std::distance(cdets_begin, cdets_end);
  logger->info("[ASCI Search Settings]:");
  logger->info("  * NCDETS                 = {}", ncdets);
  logger->info("  * NDETS_MAX              = {}", ndets_max);
  logger->info("  * NORB                   = {}", norb);
  logger->info("  * H_EL_TOL               = {:.4e}", asci_settings.h_el_tol);
  logger->info("  * RV_PRUNE_TOL           = {:.4e}",
               asci_settings.rv_prune_tol);
  logger->info("  * PAIR_SIZE_MAX          = {}", asci_settings.pair_size_max);
  logger->info("  * JUST_SINGLES           = {}", asci_settings.just_singles);
  logger->info("  * CONSTRAINT_LEVEL       = {}",
               asci_settings.constraint_level);
  logger->info("  * GROW_FACTOR            = {:.2f}",
               asci_settings.grow_factor);
  logger->info("  * MIN_GROW_FACTOR        = {:.2f}",
               asci_settings.min_grow_factor);
  logger->info("  * GROWTH_BACKOFF_RATE    = {:.2f}",
               asci_settings.growth_backoff_rate);
  logger->info("  * GROWTH_RECOVERY_RATE   = {:.2f}",
               asci_settings.growth_recovery_rate);
  logger->info("  * MAX_REFINE_ITER        = {}",
               asci_settings.max_refine_iter);
  logger->info("  * REFINE_ENERGY_TOL      = {:.4e}",
               asci_settings.refine_energy_tol);
  logger->info(
      "  * CORE_SELECTION         = {}",
      core_selection_strategy_to_string(asci_settings.core_selection_strategy));
  logger->info("  * CORE_SELECT_THRESHOLD  = {:.4f}",
               asci_settings.core_selection_threshold);
  logger->info("  * NCDETS_MAX             = {}", asci_settings.ncdets_max);
  logger->info("  * NXTVAL_BCOUNT_THRESH   = {}",
               asci_settings.nxtval_bcount_thresh);
  logger->info("  * NXTVAL_BCOUNT_INC      = {}",
               asci_settings.nxtval_bcount_inc);
  logger->info("  * GROW_WITH_ROT          = {}", asci_settings.grow_with_rot);
  logger->info("  * E_ASCI                 = {:.10e}", E_ASCI);
  logger->info("  * CDET_SUM               = {:.6e}",
               std::accumulate(C.begin(), C.begin() + ncdets, 0.0,
                               [](auto s, auto c) { return s + c * c; }));
  logger->info("");

  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto asci_search_st = clock_type::now();

  // Expand Search Space with Connected ASCI Contributions
  auto pairs_st = clock_type::now();
  asci_contrib_container<wfn_t<N>> asci_pairs;
  asci_pairs = asci_contributions_constraint(
      asci_settings, ndets_max, cdets_begin, cdets_end, E_ASCI, C, norb, T_pq,
      G_red, V_red, G_pqrs, V_pqrs, ham_gen MACIS_MPI_CODE(, comm));
  auto pairs_en = clock_type::now();

  {
#ifdef MACIS_ENABLE_MPI
    size_t npairs = allreduce(asci_pairs.size(), MPI_SUM, comm);
#else
    size_t npairs = asci_pairs.size();
#endif /* MACIS_ENABLE_MPI */
    logger->info("  * ASCI Kept {} Pairs", npairs);
    if (npairs < ndets_max)
      logger->info("    * WARNING: Kept ASCI pairs less than requested TDETS");

#ifdef MACIS_ENABLE_MPI
    if (world_size > 1) {
      size_t npairs_max = allreduce(asci_pairs.size(), MPI_MAX, comm);
      size_t npairs_min = allreduce(asci_pairs.size(), MPI_MIN, comm);
      print_mpi_stats("PAIRS_LOC", npairs_min, npairs_max, npairs / world_size);
    }
#endif /* MACIS_ENABLE_MPI */

    if (world_size == 1) {
      logger->info("  * Pairs Mem = {:.2e} GiB", to_gib(asci_pairs));
    } else {
#ifdef MACIS_ENABLE_MPI
      float local_mem = to_gib(asci_pairs);
      float total_mem = allreduce(local_mem, MPI_SUM, comm);
      float min_mem = allreduce(local_mem, MPI_MIN, comm);
      float max_mem = allreduce(local_mem, MPI_MAX, comm);
      print_mpi_stats("PAIRS_MEM", min_mem, max_mem, total_mem / world_size);
#endif /* MACIS_ENABLE_MPI */
    }
  }

  // Accumulate unique score contributions
  // MPI + Constraint Search already does S&A
  auto bit_sort_st = clock_type::now();
  if (world_size == 1) parallel_sort_and_accumulate_asci_pairs(asci_pairs);
  auto bit_sort_en = clock_type::now();

  {
#ifdef MACIS_ENABLE_MPI
    size_t npairs = allreduce(asci_pairs.size(), MPI_SUM, comm);
#else
    size_t npairs = asci_pairs.size();
#endif /* MACIS_ENABLE_MPI */
    logger->info("  * ASCI will search over {} unique determinants", npairs);

    float pairs_dur = duration_type(pairs_en - pairs_st).count();
    float bit_sort_dur = duration_type(bit_sort_en - bit_sort_st).count();

    if (world_size > 1) {
#ifdef MACIS_ENABLE_MPI
      float timings = pairs_dur;
      float timings_max, timings_min, timings_avg;
      allreduce(&timings, &timings_max, 1, MPI_MAX, comm);
      allreduce(&timings, &timings_min, 1, MPI_MIN, comm);
      allreduce(&timings, &timings_avg, 1, MPI_SUM, comm);
      timings_avg /= world_size;
      print_mpi_stats("PAIRS_DUR", timings_min, timings_max, timings_avg);
#endif /* MACIS_ENABLE_MPI */
    } else {
      logger->info("  * PAIR_DUR = {:.2e} s, SORT_ACC_DUR = {:.2e} s",
                   pairs_dur, bit_sort_dur);
    }
  }

  auto keep_large_st = clock_type::now();

  // Remove core dets
  // This assumes the constraint search
  {
    auto inf_ptr = std::partition(asci_pairs.begin(), asci_pairs.end(),
                                  [](auto& p) { return !std::isinf(p.rv()); });
    asci_pairs.erase(inf_ptr, asci_pairs.end());
  }

  auto keep_large_en = clock_type::now();
  duration_type keep_large_dur = keep_large_en - keep_large_st;
  if (world_size > 1) {
#ifdef MACIS_ENABLE_MPI
    float dur = keep_large_dur.count();
    auto dmin = allreduce(dur, MPI_MIN, comm);
    auto dmax = allreduce(dur, MPI_MAX, comm);
    auto davg = allreduce(dur, MPI_SUM, comm) / world_size;
    print_mpi_stats("KEEP_LARG_DUR", dmin, dmax, davg);
#endif /* MACIS_ENABLE_MPI */
  } else {
    logger->info("  * KEEP_LARG_DUR = {:.2e} s", keep_large_dur.count());
  }

  // Only do top-K on (ndets_max - ncdets) b/c CDETS will be added later

  // N.B. Mutable because we will include pairs with equal RV in the case
  // that the oribital top-K splits the equal partition
  size_t top_k_elements = ndets_max - ncdets;

  // Do Top-K to get the largest determinant contributions
  auto asci_sort_st = clock_type::now();
  if (world_size > 1 or asci_pairs.size() > top_k_elements) {
    std::vector<asci_contrib<wfn_t<N>>> topk(top_k_elements);
    if (world_size > 1) {
#ifdef MACIS_ENABLE_MPI
      // Strip scores
      std::vector<double> scores(asci_pairs.size());
      std::transform(asci_pairs.begin(), asci_pairs.end(), scores.begin(),
                     [](const auto& p) { return std::abs(p.rv()); });

      // Determine kth-ranked scores
      auto kth_score =
          dist_quickselect(scores.begin(), scores.end(), top_k_elements, comm,
                           std::greater<double>{}, std::equal_to<double>{});

      logger->info("  * Kth Score Pivot = {:.16e}", kth_score);
      // Partition local pairs into less / eq batches
      auto [g_begin, e_begin, l_begin, _end] = leg_partition(
          asci_pairs.begin(), asci_pairs.end(), kth_score,
          [=](const auto& p, const auto& s) { return std::abs(p.rv()) > s; },
          [=](const auto& p, const auto& s) { return std::abs(p.rv()) == s; });

      // Determine local counts
      size_t n_greater = std::distance(g_begin, e_begin);
      size_t n_equal = std::distance(e_begin, l_begin);
      size_t n_less = std::distance(l_begin, _end);
      const int n_geq_local = n_greater + n_equal;

      // Strip bitsrings
      std::vector<wfn_t<N>> keep_strings_local(n_geq_local);
      std::transform(g_begin, l_begin, keep_strings_local.begin(),
                     [](const auto& p) { return p.state; });

      // Gather global strings
      std::vector<int> local_sizes, displ;
      auto n_geq_global = total_gather_and_exclusive_scan(
          n_geq_local, local_sizes, displ, comm);

      std::vector<wfn_t<N>> keep_strings_global(n_geq_global);
      auto string_dtype = mpi_traits<wfn_t<N>>::datatype();
      MPI_Allgatherv(keep_strings_local.data(), n_geq_local, string_dtype,
                     keep_strings_global.data(), local_sizes.data(),
                     displ.data(), string_dtype, comm);

      // Resize to global size
      if (n_geq_global > top_k_elements) {
        top_k_elements = n_geq_global;
        keep_strings_global.resize(n_geq_global);
      }

      // Make fake strings
      topk.resize(n_geq_global);
      std::transform(
          keep_strings_global.begin(), keep_strings_global.end(), topk.begin(),
          [](const auto& s) { return asci_contrib<wfn_t<N>>{s, -1.0}; });

#endif /* MACIS_ENABLE_MPI */
    } else {
      std::nth_element(asci_pairs.begin(), asci_pairs.begin() + top_k_elements,
                       asci_pairs.end(),
                       asci_contrib_topk_comparator<wfn_t<N>>{});
      // Back out equivalent contributions
      const auto kth_score =
          std::abs(std::max_element(asci_pairs.begin(),
                                    asci_pairs.begin() + top_k_elements,
                                    asci_contrib_topk_comparator<wfn_t<N>>{})
                       ->rv());
      logger->info("  * Kth Score Pivot = {:.16e}", kth_score);
      auto [g_begin, e_begin, l_begin, _end] = leg_partition(
          asci_pairs.begin(), asci_pairs.end(), kth_score,
          [=](const auto& p, const auto& s) { return std::abs(p.rv()) > s; },
          [=](const auto& p, const auto& s) { return std::abs(p.rv()) == s; });
      size_t n_greater = std::distance(g_begin, e_begin);
      size_t n_equal = std::distance(e_begin, l_begin);
      size_t n_less = std::distance(l_begin, _end);
      const int n_geq = n_greater + n_equal;
      top_k_elements = n_geq;
      topk.resize(n_geq);

      std::copy(asci_pairs.begin(), asci_pairs.begin() + top_k_elements,
                topk.begin());
    }
    asci_pairs = std::move(topk);
  }
  auto asci_sort_en = clock_type::now();
  if (world_size > 1) {
#ifdef MACIS_ENABLE_MPI
    float dur = duration_type(asci_sort_en - asci_sort_st).count();
    auto dmin = allreduce(dur, MPI_MIN, comm);
    auto dmax = allreduce(dur, MPI_MAX, comm);
    auto davg = allreduce(dur, MPI_SUM, comm) / world_size;
    print_mpi_stats("ASCI_SORT_DUR", dmin, dmax, davg);
#endif /* MACIS_ENABLE_MPI */
  } else {
    logger->info("  * ASCI_SORT_DUR = {:.2e} s",
                 duration_type(asci_sort_en - asci_sort_st).count());
  }

  if (top_k_elements != ndets_max - ncdets) {
    logger->warn(
        "  * ASCI Search requested {} determinants, but will return {}.",
        ndets_max, top_k_elements + ncdets);
    logger->warn(
        "    This is due to the presence of multiple determinants with the "
        "    same predicted weight at the cutoff size. By policy, equivalent "
        "    determinants are retained.");
  }
  // Shrink to max search space
  asci_pairs.shrink_to_fit();

  // Extract new search determinants
  std::vector<std::bitset<N>> new_dets(asci_pairs.size());
  std::transform(asci_pairs.begin(), asci_pairs.end(), new_dets.begin(),
                 [](auto x) { return x.state; });

  // Insert the CDETS back in
  new_dets.insert(new_dets.end(), cdets_begin, cdets_end);
  new_dets.shrink_to_fit();

  logger->info("  * New Dets Mem = {:.2e} GiB", to_gib(new_dets));

  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto asci_search_en = clock_type::now();
  duration_type asci_search_dur = asci_search_en - asci_search_st;
  logger->info("  * ASCI_SEARCH DUR = {:.2e} s", asci_search_dur.count());
  return new_dets;
}

}  // namespace macis
