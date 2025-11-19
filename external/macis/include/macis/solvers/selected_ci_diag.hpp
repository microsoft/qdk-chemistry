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
#include <macis/csr_hamiltonian.hpp>
#include <macis/hamiltonian_generator.hpp>
#include <macis/solvers/davidson.hpp>
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
 * This function constructs the selected CI Hamiltonian matrix from determinants
 * and performs diagonalization to find the ground state energy. It
 * automatically chooses between serial and parallel implementations based on
 * compilation flags.
 *
 * @tparam index_t Type for matrix indices
 * @tparam WfnType Type of wavefunction/determinant
 * @tparam WfnIterator Iterator type for determinant container
 * @param[in] dets_begin Iterator to beginning of determinant list
 * @param[in] dets_end Iterator to end of determinant list
 * @param[in,out] ham_gen Hamiltonian generator for matrix elements
 * @param[in] h_el_tol Tolerance for Hamiltonian matrix elements
 * @param[in] davidson_max_m Maximum dimension of Davidson subspace
 * @param[in] davidson_res_tol Davidson convergence tolerance
 * @param[in,out] C_local Eigenvector coefficients (local portion for MPI)
 * @param[in] comm MPI communicator (MPI builds only)
 * @param[in] quiet Flag to suppress output (default: false)
 * @return Ground state energy
 */
template <typename index_t, typename WfnType, typename WfnIterator>
double selected_ci_diag(WfnIterator dets_begin, WfnIterator dets_end,
                        HamiltonianGenerator<WfnType>& ham_gen, double h_el_tol,
                        size_t davidson_max_m, double davidson_res_tol,
                        std::vector<double>& C_local,
                        MACIS_MPI_CODE(MPI_Comm comm, )
                            const bool quiet = false) {
  auto logger = spdlog::get("ci_solver");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  logger->info("[Selected CI Solver]:");
  logger->info("  {} = {:6}, {} = {:.5e}, {} = {:.5e}, {} = {:4}", "NDETS",
               std::distance(dets_begin, dets_end), "MATEL_TOL", h_el_tol,
               "RES_TOL", davidson_res_tol, "MAX_SUB", davidson_max_m);

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;

  // Generate Hamiltonian
  MACIS_MPI_CODE(MPI_Barrier(comm);)
  auto H_st = clock_type::now();

#ifdef MACIS_ENABLE_MPI

  auto world_size = comm_size(comm);
  auto world_rank = comm_rank(comm);

  auto H = make_dist_csr_hamiltonian<index_t>(comm, dets_begin, dets_end,
                                              ham_gen, h_el_tol);
#else
  auto H =
      make_csr_hamiltonian<index_t>(dets_begin, dets_end, ham_gen, h_el_tol);
#endif /* MACIS_ENABLE_MPI */

  auto H_en = clock_type::now();
  MACIS_MPI_CODE(MPI_Barrier(comm);)

  // Get total NNZ
#ifdef MACIS_ENABLE_MPI
  size_t local_nnz = H.nnz();
  size_t total_nnz = allreduce(local_nnz, MPI_SUM, comm);
  size_t max_nnz = allreduce(local_nnz, MPI_MAX, comm);
  size_t min_nnz = allreduce(local_nnz, MPI_MIN, comm);
#else
  size_t total_nnz = H.nnz();
#endif /* MACIS_ENABLE_MPI */

  logger->info("  {}   = {:6}, {}     = {:.5e} ms", "NNZ", total_nnz, "H_DUR",
               duration_type(H_en - H_st).count());

#ifdef MACIS_ENABLE_MPI
  if (world_size > 1) {
    double local_hdur = duration_type(H_en - H_st).count();
    double max_hdur = allreduce(local_hdur, MPI_MAX, comm);
    double min_hdur = allreduce(local_hdur, MPI_MIN, comm);
    double avg_hdur = allreduce(local_hdur, MPI_SUM, comm);
    avg_hdur /= world_size;
    logger->info(
        "  H_DUR_MAX = {:.2e} ms, H_DUR_MIN = {:.2e} ms, H_DUR_AVG = {:.2e} ms",
        max_hdur, min_hdur, avg_hdur);
  }
#endif /* MACIS_ENABLE_MPI */
  logger->info("  {} = {:.2e} GiB", "HMEM_LOC",
               H.mem_footprint() / 1073741824.);
  logger->info("  {} = {:.2e}%", "H_SPARSE",
               total_nnz / double(H.n() * H.n()) * 100);
#ifdef MACIS_ENABLE_MPI
  if (world_size > 1) {
    logger->info("  NNZ_MAX = {}, NNZ_MIN = {}, NNZ_AVG = {}", max_nnz, min_nnz,
                 total_nnz / double(world_size));
  }
#endif /* MACIS_ENABLE_MPI */

  // Solve EVP
#ifdef MACIS_ENABLE_MPI
  auto E = parallel_selected_ci_diag(H, davidson_max_m, davidson_res_tol,
                                     C_local, comm);
#else
  auto E =
      serial_selected_ci_diag(H, davidson_max_m, davidson_res_tol, C_local);
#endif /* MACIS_ENABLE_MPI */

  return E;
}

}  // namespace macis
