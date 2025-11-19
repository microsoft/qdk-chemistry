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
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>

#ifdef MACIS_ENABLE_MPI
#include <sparsexx/matrix_types/dist_sparse_matrix.hpp>
#endif /* MACIS_ENABLE_MPI */

#include <sparsexx/matrix_types/csr_matrix.hpp>
namespace macis {

/**
 *  @brief Base implementation of CSR Hamiltonian generation for a block
 *
 *  Generates a Compressed Sparse Row (CSR) matrix representation of a
 *  Hamiltonian block between specified bra and ket wavefunction ranges.
 *  Returns an empty matrix if either range is empty.
 *
 *  @tparam index_t Integer type for matrix indices
 *  @tparam WfnType Type of the wavefunction determinants
 *  @tparam WfnIterator Iterator type for wavefunction containers
 *
 *  @param[in] bra_begin Iterator to the beginning of bra determinants
 *  @param[in] bra_end Iterator to the end of bra determinants
 *  @param[in] ket_begin Iterator to the beginning of ket determinants
 *  @param[in] ket_end Iterator to the end of ket determinants
 *  @param[in] ham_gen Hamiltonian generator object
 *  @param[in] H_thresh Threshold for matrix element inclusion
 *  @returns CSR matrix representation of the Hamiltonian block
 */
template <typename index_t, typename WfnType, typename WfnIterator>
sparsexx::csr_matrix<double, index_t> make_csr_hamiltonian_block(
    WfnIterator bra_begin, WfnIterator bra_end, WfnIterator ket_begin,
    WfnIterator ket_end, HamiltonianGenerator<WfnType>& ham_gen,
    double H_thresh) {
  size_t nbra = std::distance(bra_begin, bra_end);
  size_t nket = std::distance(ket_begin, ket_end);

  if (nbra and nket) {
    return ham_gen.template make_csr_hamiltonian_block<index_t>(
        bra_begin, bra_end, ket_begin, ket_end, H_thresh);
  } else {
    return sparsexx::csr_matrix<double, index_t>(nbra, nket, 0, 0);
  }
}

/**
 *  @brief Generate CSR Hamiltonian matrix for a set of determinants
 *
 *  Creates a full Compressed Sparse Row (CSR) matrix representation of
 *  the Hamiltonian for a given set of wavefunction determinants. This is
 *  a convenience wrapper around make_csr_hamiltonian_block for square matrices.
 *
 *  @tparam index_t Integer type for matrix indices
 *  @tparam WfnType Type of the wavefunction determinants
 *  @tparam WfnIterator Iterator type for wavefunction containers
 *
 *  @param[in] sd_begin Iterator to the beginning of determinants
 *  @param[in] sd_end Iterator to the end of determinants
 *  @param[in] ham_gen Hamiltonian generator object
 *  @param[in] H_thresh Threshold for matrix element inclusion
 *  @returns CSR matrix representation of the full Hamiltonian
 */
template <typename index_t, typename WfnType, typename WfnIterator>
sparsexx::csr_matrix<double, index_t> make_csr_hamiltonian(
    WfnIterator sd_begin, WfnIterator sd_end,
    HamiltonianGenerator<WfnType>& ham_gen, double H_thresh) {
  return make_csr_hamiltonian_block<index_t>(sd_begin, sd_end, sd_begin, sd_end,
                                             ham_gen, H_thresh);
}

#ifdef MACIS_ENABLE_MPI
/**
 *  @brief Generate distributed CSR Hamiltonian matrix
 *
 *  Creates a distributed Compressed Sparse Row (CSR) matrix representation
 *  of the Hamiltonian using MPI for parallel computation. The matrix is
 *  distributed across MPI ranks with each rank handling a subset of rows.
 *  Constructs both diagonal and off-diagonal tiles separately for efficiency.
 *
 *  @tparam index_t Integer type for matrix indices
 *  @tparam WfnType Type of the wavefunction determinants
 *  @tparam WfnIterator Iterator type for wavefunction containers
 *
 *  @param[in] comm MPI communicator for distributed computation
 *  @param[in] sd_begin Iterator to the beginning of determinants
 *  @param[in] sd_end Iterator to the end of determinants
 *  @param[in] ham_gen Hamiltonian generator object
 *  @param[in] H_thresh Threshold for matrix element inclusion
 *  @returns Distributed CSR matrix representation of the Hamiltonian
 */
template <typename index_t, typename WfnType, typename WfnIterator>
sparsexx::dist_sparse_matrix<sparsexx::csr_matrix<double, index_t>>
make_dist_csr_hamiltonian(MPI_Comm comm, WfnIterator sd_begin,
                          WfnIterator sd_end,
                          HamiltonianGenerator<WfnType>& ham_gen,
                          const double H_thresh) {
  using namespace sparsexx;
  using namespace sparsexx::detail;

  size_t ndets = std::distance(sd_begin, sd_end);
  dist_sparse_matrix<csr_matrix<double, index_t>> H_dist(comm, ndets, ndets);

  // Get local row bounds
  auto [bra_st, bra_en] = H_dist.row_bounds(get_mpi_rank(comm));

  // Build diagonal part
  H_dist.set_diagonal_tile(make_csr_hamiltonian_block<index_t>(
      sd_begin + bra_st, sd_begin + bra_en, sd_begin + bra_st,
      sd_begin + bra_en, ham_gen, H_thresh));

  auto world_size = get_mpi_size(comm);

  if (world_size > 1) {
    // Create a copy of SD's with local bra dets zero'd out
    std::vector<WfnType> sds_offdiag(sd_begin, sd_end);
    for (auto i = bra_st; i < bra_en; ++i) sds_offdiag[i] = 0ul;

    // Build off-diagonal part
    H_dist.set_off_diagonal_tile(make_csr_hamiltonian_block<index_t>(
        sd_begin + bra_st, sd_begin + bra_en, sds_offdiag.begin(),
        sds_offdiag.end(), ham_gen, H_thresh));
  }

  return H_dist;
}
#endif /* MACIS_ENABLE_MPI */

}  // namespace macis
