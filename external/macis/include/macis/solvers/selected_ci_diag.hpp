/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <chrono>
#include <optional>

#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator.hpp>
#include <macis/solvers/davidson.hpp>
#include <macis/solvers/incremental_h_build.hpp>
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>
#include <sparsexx/io/write_dist_mm.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/util/submatrix.hpp>

namespace macis {

#ifdef MACIS_ENABLE_MPI
/**
 * @brief Parallel (MPI) selected CI diagonalization using Davidson solver.
 *
 * Performs diagonalization of a distributed sparse Hamiltonian matrix using
 * the parallel Davidson eigensolver to find the ground state energy and
 * wavefunction.
 *
 * @tparam SpMatType Type of the distributed sparse matrix
 * @param[in] H Distributed sparse Hamiltonian matrix
 * @param[in] davidson_max_m Maximum dimension of the Davidson subspace
 * @param[in] davidson_res_tol Convergence tolerance for Davidson residual
 * @param[in,out] C_local Local portion of eigenvector (input: guess, output:
 * converged)
 * @param[in] comm MPI communicator
 * @return Ground state energy
 */
template <typename SpMatType>
double parallel_selected_ci_diag(const SpMatType& H, size_t davidson_max_m,
                                 double davidson_res_tol,
                                 std::vector<double>& C_local, MPI_Comm comm) {
  auto logger = spdlog::get("ci_solver");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Resize eigenvector size
  C_local.resize(H.local_row_extent(), 0);

  // Extract Diagonal
  auto D_local = extract_diagonal_elements(H.diagonal_tile());

  // Setup guess
  auto max_c = *std::max_element(
      C_local.begin(), C_local.end(),
      [](auto a, auto b) { return std::abs(a) < std::abs(b); });
  max_c = std::abs(max_c);

  if (max_c > (1. / C_local.size())) {
    logger->info("  * Will use passed vector as guess");
  } else {
    logger->info("  * Will generate identity guess");
    p_diagonal_guess(C_local.size(), H, C_local.data());
  }

  // Setup Davidson Functor
  SparseMatrixOperator op(H);

  // Solve EVP
  MPI_Barrier(comm);
  auto dav_st = clock_type::now();

  auto [niter, E] =
      p_davidson(H.local_row_extent(), davidson_max_m, op, D_local.data(),
                 davidson_res_tol, C_local.data() MACIS_MPI_CODE(, H.comm()));

  MPI_Barrier(comm);
  auto dav_en = clock_type::now();

  logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
               niter, "E0", E, "DAVIDSON_DUR",
               duration_type(dav_en - dav_st).count());

  return E;
}
#endif /* MACIS_ENABLE_MPI */

/**
 * @brief Serial selected CI diagonalization using Davidson solver.
 *
 * Performs diagonalization of a sparse Hamiltonian matrix using the serial
 * Davidson eigensolver to find the ground state energy and wavefunction.
 *
 * @tparam SpMatType Type of the sparse matrix
 * @param[in] H Sparse Hamiltonian matrix
 * @param[in] davidson_max_m Maximum dimension of the Davidson subspace
 * @param[in] davidson_res_tol Convergence tolerance for Davidson residual
 * @param[in,out] C Eigenvector (input: guess, output: converged)
 * @return Ground state energy
 */
template <typename SpMatType>
double serial_selected_ci_diag(const SpMatType& H, size_t davidson_max_m,
                               double davidson_res_tol,
                               std::vector<double>& C) {
  auto logger = spdlog::get("ci_solver");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Resize eigenvector size
  C.resize(H.m(), 0);

  // Extract Diagonal
  auto D = extract_diagonal_elements(H);

  // Setup guess
  auto max_c = *std::max_element(C.begin(), C.end(), [](auto a, auto b) {
    return std::abs(a) < std::abs(b);
  });
  max_c = std::abs(max_c);

  if (max_c > (1. / C.size())) {
    logger->info("  * Will use passed vector as guess");
  } else {
    logger->info("  * Will generate identity guess");
    diagonal_guess(C.size(), H, C.data());
  }

  // Setup Davidson Functor
  SparseMatrixOperator op(H);

  // Solve EVP
  auto dav_st = clock_type::now();

  auto [niter, E] =
      davidson(H.m(), davidson_max_m, op, D.data(), davidson_res_tol, C.data());

  auto dav_en = clock_type::now();

  logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
               niter, "E0", E, "DAVIDSON_DUR",
               duration_type(dav_en - dav_st).count());

  return E;
}

/**
 * @brief Main selected CI diagonalization routine with Hamiltonian
 * construction.
 *
 * Builds the Hamiltonian and runs Davidson.  If a CachedHamiltonianState is
 * provided, attempts an incremental build first and falls back to a full
 * build when overlap is low.  Automatically chooses serial vs parallel
 * Davidson based on compilation flags.
 *
 * @param cache  Optional pointer to incremental H_build cache.  nullptr
 *               disables incremental mode.
 * @param min_patch_overlap  Overlap threshold for patched build (ignored
 *                           when cache is nullptr).
 */
template <typename index_t, typename WfnType, typename WfnIterator>
double selected_ci_diag(
    WfnIterator dets_begin, WfnIterator dets_end,
    HamiltonianGenerator<WfnType>& ham_gen, double h_el_tol,
    size_t davidson_max_m, double davidson_res_tol,
    std::vector<double>& C_local,
    CachedHamiltonianState<WfnType, index_t>* cache,
    double min_patch_overlap
    MACIS_MPI_CODE(, MPI_Comm comm = MPI_COMM_WORLD)) {
  auto logger = spdlog::get("ci_solver");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }
  // Ensure sub-loggers exist for downstream code
  if (!spdlog::get("h_build"))
    spdlog::stdout_color_mt("h_build");
  if (!spdlog::get("h_build_inc"))
    spdlog::stdout_color_mt("h_build_inc");

  const size_t ndets = std::distance(dets_begin, dets_end);
  const bool incremental = cache != nullptr;

  logger->info("[Selected CI Solver{}]:", incremental ? " (incremental)" : "");
  logger->info("  {} = {:6}, {} = {:.5e}, {} = {:.5e}, {} = {:4}", "NDETS",
               ndets, "MATEL_TOL", h_el_tol, "RES_TOL", davidson_res_tol,
               "MAX_SUB", davidson_max_m);

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto H_st = clock_type::now();

  double E;

  // ---- Incremental path: try patched operator first ----
  if (incremental && cache->valid) {
    std::vector<WfnType> new_dets(dets_begin, dets_end);
    auto patched_op = build_patched_operator<index_t>(
        *cache, new_dets, ham_gen, h_el_tol, min_patch_overlap);
    auto H_en = clock_type::now();
    logger->info("  PATCH_DUR = {:.5e} ms",
                 duration_type(H_en - H_st).count());

    if (patched_op) {
      // Set up guess
      C_local.resize(ndets, 0.0);
      auto max_c = *std::max_element(
          C_local.begin(), C_local.end(),
          [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      if (std::abs(max_c) > (1.0 / ndets)) {
        logger->info("  * Will use passed vector as guess");
      } else {
        logger->info("  * Will generate diagonal guess");
        const auto& D = patched_op->diagonal();
        auto D_min = std::min_element(D.begin(), D.end());
        auto min_idx = std::distance(D.begin(), D_min);
        std::fill(C_local.begin(), C_local.end(), 0.0);
        C_local[min_idx] = 1.0;
      }

      auto dav_st = clock_type::now();
      auto [niter, eigval] = davidson(
          ndets, davidson_max_m, *patched_op,
          patched_op->diagonal().data(), davidson_res_tol, C_local.data());
      auto dav_en = clock_type::now();

      logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
                   niter, "E0", eigval, "DAVIDSON_DUR",
                   duration_type(dav_en - dav_st).count());
      E = eigval;


      return E;
    }
    // patched_op is nullopt — fall through to full build
  }

  // ---- Full build path ----
  auto H =
#ifdef MACIS_ENABLE_MPI
      make_dist_csr_hamiltonian<index_t>(comm, dets_begin, dets_end, ham_gen,
                                         h_el_tol);
#else
      make_csr_hamiltonian<index_t>(dets_begin, dets_end, ham_gen, h_el_tol);
#endif

  auto H_en = clock_type::now();
  MACIS_MPI_CODE(MPI_Barrier(comm);)

#ifdef MACIS_ENABLE_MPI
  size_t local_nnz = H.nnz();
  size_t total_nnz = allreduce(local_nnz, MPI_SUM, comm);
  auto world_size = comm_size(comm);
  if (world_size > 1) {
    size_t max_nnz = allreduce(local_nnz, MPI_MAX, comm);
    size_t min_nnz = allreduce(local_nnz, MPI_MIN, comm);
    double local_hdur = duration_type(H_en - H_st).count();
    double max_hdur = allreduce(local_hdur, MPI_MAX, comm);
    double min_hdur = allreduce(local_hdur, MPI_MIN, comm);
    double avg_hdur = allreduce(local_hdur, MPI_SUM, comm) / world_size;
    logger->info(
        "  H_DUR_MAX = {:.2e} ms, H_DUR_MIN = {:.2e} ms, H_DUR_AVG = {:.2e} ms",
        max_hdur, min_hdur, avg_hdur);
    logger->info("  NNZ_MAX = {}, NNZ_MIN = {}, NNZ_AVG = {}", max_nnz,
                 min_nnz, total_nnz / double(world_size));
  }
#else
  size_t total_nnz = H.nnz();
#endif

  logger->info("  {}   = {:6}, {}     = {:.5e} ms", "NNZ", total_nnz, "H_DUR",
               duration_type(H_en - H_st).count());
  logger->info("  {} = {:.2e} GiB", "HMEM_LOC",
               H.mem_footprint() / 1073741824.);
  logger->info("  {} = {:.2e}%", "H_SPARSE",
               total_nnz / double(H.n() * H.n()) * 100);

#ifdef MACIS_ENABLE_MPI
  E = parallel_selected_ci_diag(H, davidson_max_m, davidson_res_tol, C_local,
                                comm);
#else
  E = serial_selected_ci_diag(H, davidson_max_m, davidson_res_tol, C_local);
#endif

  // Cache for next iteration if incremental mode is active
  if (incremental) {
    cache->store(std::vector<WfnType>(dets_begin, dets_end), std::move(H));
  }

  return E;
}

/// @brief Convenience overload without incremental cache.
template <typename index_t, typename WfnType, typename WfnIterator>
double selected_ci_diag(
    WfnIterator dets_begin, WfnIterator dets_end,
    HamiltonianGenerator<WfnType>& ham_gen, double h_el_tol,
    size_t davidson_max_m, double davidson_res_tol,
    std::vector<double>& C_local
    MACIS_MPI_CODE(, MPI_Comm comm = MPI_COMM_WORLD)) {
  return selected_ci_diag<index_t>(
      dets_begin, dets_end, ham_gen, h_el_tol, davidson_max_m,
      davidson_res_tol, C_local,
      static_cast<CachedHamiltonianState<WfnType, index_t>*>(nullptr), 0.3
      MACIS_MPI_CODE(, comm));
}

}  // namespace macis
