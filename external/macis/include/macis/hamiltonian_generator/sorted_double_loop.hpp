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
#include <macis/hamiltonian_generator.hpp>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>
#ifdef _OPENMP
#include <omp.h>
#else
// Fallbacks for non-OpenMP environments
namespace {
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
}  // namespace
#endif /* _OPENMP */

namespace macis {

/**
 * @brief Optimized double loop Hamiltonian generator with alpha string sorting
 *
 * This class implements an optimized version of the double loop algorithm that
 * takes determinants sorted by their alpha string components to reduce
 * computational complexity. It groups determinants with identical alpha strings
 * and processes them together, leading to significant performance improvements
 * for large determinant spaces.
 *
 * @tparam WfnType The wavefunction type that defines the determinant
 * representation
 */
template <typename WfnType>
class SortedDoubleLoopHamiltonianGenerator
    : public HamiltonianGenerator<WfnType> {
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
   * @brief Template method to create optimized CSR Hamiltonian matrix block
   *
   * This method implements the optimized double loop algorithm with alpha
   * string sorting to generate a sparse Hamiltonian matrix block in CSR format.
   * The optimization groups determinants by their alpha strings and processes
   * beta strings within each group, significantly reducing the computational
   * cost.
   *
   * The method uses OpenMP parallelization when available and implements a
   * two-pass algorithm to efficiently construct the CSR matrix structure.
   *
   * @tparam index_t Integer type for matrix indices (int32_t or int64_t)
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
    using spin_wfn_type = typename wfn_traits::spin_wfn_type;
    using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    const bool is_symm = bra_begin == ket_begin and bra_end == ket_end;
#ifdef MACIS_ENABLE_MPI
    auto world_rank = comm_rank(MPI_COMM_WORLD);
#else
    auto world_rank = 0;
#endif /* MACIS_ENABLE_MPI */

    // Get unique alpha strings
    auto setup_st = std::chrono::high_resolution_clock::now();
    auto unique_alpha_bra = get_unique_alpha(bra_begin, bra_end);
    auto unique_alpha_ket =
        is_symm ? unique_alpha_bra : get_unique_alpha(ket_begin, ket_end);

    const size_t nuniq_bra = unique_alpha_bra.size();
    const size_t nuniq_ket = unique_alpha_ket.size();

    // Compute offsets
    std::vector<size_t> unique_alpha_bra_idx(nuniq_bra + 1);
    std::transform_exclusive_scan(
        unique_alpha_bra.begin(), unique_alpha_bra.end(),
        unique_alpha_bra_idx.begin(), 0ul, std::plus<size_t>{},
        [](auto& x) { return x.second; });
    std::vector<size_t> unique_alpha_ket_idx(nuniq_ket + 1);
    if (is_symm) {
      unique_alpha_ket_idx = unique_alpha_bra_idx;
    } else {
      std::transform_exclusive_scan(
          unique_alpha_ket.begin(), unique_alpha_ket.end(),
          unique_alpha_ket_idx.begin(), 0ul, std::plus<size_t>{},
          [](auto& x) { return x.second; });
    }

    unique_alpha_bra_idx.back() = nbra_dets;
    unique_alpha_ket_idx.back() = nket_dets;

    // Two-pass algorithm:
    // 1. First pass - Count non-zero matrix elements per row
    // 2. Second pass - Compute and fill the CSR matrix directly

    auto count_st = std::chrono::high_resolution_clock::now();

    // First pass: Count non-zeros per row to allocate CSR structure
    std::vector<index_t> row_nnz(nbra_dets, 0);

#pragma omp parallel
    {
      std::vector<index_t> row_nnz_local(nbra_dets, 0);
      std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

#pragma omp for schedule(dynamic)
      for (size_t ia_bra = 0; ia_bra < nuniq_bra; ++ia_bra) {
        if (!unique_alpha_bra[ia_bra].first.any()) continue;

        // Extract alpha bra
        const auto bra_alpha = unique_alpha_bra[ia_bra].first;
        const size_t beta_st_bra = unique_alpha_bra_idx[ia_bra];
        const size_t beta_en_bra = unique_alpha_bra_idx[ia_bra + 1];

        const auto ket_lower = is_symm ? ia_bra : 0;
        for (size_t ia_ket = ket_lower; ia_ket < nuniq_ket; ++ia_ket) {
          if (!unique_alpha_ket[ia_ket].first.any()) continue;

          // Extract alpha ket
          const auto ket_alpha = unique_alpha_ket[ia_ket].first;

          // Compute alpha excitation
          const auto ex_alpha = bra_alpha ^ ket_alpha;
          const auto ex_alpha_count = spin_wfn_traits::count(ex_alpha);

          // Early exit if excitation level too high
          if (ex_alpha_count > 4) continue;

          // Early exit if the only possible matrix element is 4x alpha
          if (ex_alpha_count == 4) {
            const double h_el_all_alpha_4 =
                this->matrix_element_4(bra_alpha, ket_alpha, ex_alpha);
            if (std::abs(h_el_all_alpha_4) < H_thresh) continue;
          }

          const size_t beta_st_ket = unique_alpha_ket_idx[ia_ket];
          const size_t beta_en_ket = unique_alpha_ket_idx[ia_ket + 1];

          // Loop over betas
          for (size_t ibra = beta_st_bra; ibra < beta_en_bra; ++ibra) {
            const auto bra_beta = wfn_traits::beta_string(*(bra_begin + ibra));

            for (size_t iket = beta_st_ket; iket < beta_en_ket; ++iket) {
              if (is_symm && (iket < ibra)) continue;

              const auto ket_beta =
                  wfn_traits::beta_string(*(ket_begin + iket));
              const auto ex_beta = bra_beta ^ ket_beta;
              const auto ex_beta_count = spin_wfn_traits::count(ex_beta);

              // Skip if total excitation level too high
              if ((ex_alpha_count + ex_beta_count) > 4) continue;

              // Integral lookups are cheap
              if (ex_beta_count == 4) {
                const double h_el_all_beta_4 =
                    this->matrix_element_4(bra_beta, ket_beta, ex_beta);
                if (std::abs(h_el_all_beta_4) < H_thresh) continue;
              }

              // Need to consider the threshold here to ensure accurate count
              // For this first counting pass, we'll just count everything
              // that could potentially be non-zero based on excitation rank
              row_nnz_local[ibra]++;

              // For symmetric matrices, count both entries
              if (is_symm && ibra != iket) {
                row_nnz_local[iket]++;
              }
            }
          }
        }
      }

      // Merge local counts into global counts using atomic operations
      for (size_t i = 0; i < nbra_dets; ++i) {
        if (row_nnz_local[i] > 0) {  // Only update if there's something to add
#ifdef _OPENMP
#pragma omp atomic
          row_nnz[i] += row_nnz_local[i];
#else
          row_nnz[i] += row_nnz_local[i];
#endif /* _OPENMP */
        }
      }
    }
    auto count_en = std::chrono::high_resolution_clock::now();

    // Create CSR structure using the count results
    // First, determine rowptr
    std::vector<index_t> rowptr(nbra_dets + 1);
    rowptr[0] = 0;
    for (size_t i = 0; i < nbra_dets; ++i) {
      rowptr[i + 1] = rowptr[i] + row_nnz[i];
    }

    auto alloc_st = std::chrono::high_resolution_clock::now();

    // Total number of non-zeros
    size_t total_nnz = rowptr[nbra_dets];

    // Allocate arrays for CSR format
    std::vector<index_t> colind(total_nnz);
    std::vector<double> nzval(total_nnz);

    // Create working copies of row pointers for each thread to use as insertion
    // points
    std::vector<index_t> row_offsets(rowptr);

    auto alloc_en = std::chrono::high_resolution_clock::now();

    // Second pass: Calculate matrix elements and populate CSR arrays directly
    auto fill_st = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
      // Thread-local working memory
      std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

#pragma omp for schedule(dynamic)
      for (size_t ia_bra = 0; ia_bra < nuniq_bra; ++ia_bra) {
        if (!unique_alpha_bra[ia_bra].first.any()) continue;

        // Extract alpha bra
        const auto bra_alpha = unique_alpha_bra[ia_bra].first;
        const size_t beta_st_bra = unique_alpha_bra_idx[ia_bra];
        const size_t beta_en_bra = unique_alpha_bra_idx[ia_bra + 1];
        spin_wfn_traits::state_to_occ(bra_alpha, bra_occ_alpha);

        const auto ket_lower = is_symm ? ia_bra : 0;
        for (size_t ia_ket = ket_lower; ia_ket < nuniq_ket; ++ia_ket) {
          if (!unique_alpha_ket[ia_ket].first.any()) continue;

          // Extract alpha ket
          const auto ket_alpha = unique_alpha_ket[ia_ket].first;

          // Compute alpha excitation
          const auto ex_alpha = bra_alpha ^ ket_alpha;
          const auto ex_alpha_count = spin_wfn_traits::count(ex_alpha);

          // Early exit
          if (ex_alpha_count > 4) continue;

          const size_t beta_st_ket = unique_alpha_ket_idx[ia_ket];
          const size_t beta_en_ket = unique_alpha_ket_idx[ia_ket + 1];

          // Loop over betas
          for (size_t ibra = beta_st_bra; ibra < beta_en_bra; ++ibra) {
            const auto bra_beta = wfn_traits::beta_string(*(bra_begin + ibra));
            spin_wfn_traits::state_to_occ(bra_beta, bra_occ_beta);

            for (size_t iket = beta_st_ket; iket < beta_en_ket; ++iket) {
              if (is_symm && (iket < ibra)) continue;

              const auto ket_beta =
                  wfn_traits::beta_string(*(ket_begin + iket));
              const auto ex_beta = bra_beta ^ ket_beta;
              const auto ex_beta_count = spin_wfn_traits::count(ex_beta);

              if ((ex_alpha_count + ex_beta_count) > 4) continue;

              // Compute matrix element value
              double h_el = 0.0;
              if (ex_alpha_count == 4) {
                h_el = this->matrix_element_4(bra_alpha, ket_alpha, ex_alpha);
                if (std::abs(h_el) < H_thresh) continue;
              } else if (ex_beta_count == 4) {
                h_el = this->matrix_element_4(bra_beta, ket_beta, ex_beta);
                if (std::abs(h_el) < H_thresh) continue;
              } else if (ex_alpha_count == 2) {
                if (ex_beta_count == 2) {
                  h_el = this->matrix_element_22(bra_alpha, ket_alpha, ex_alpha,
                                                 bra_beta, ket_beta, ex_beta);
                } else {
                  h_el = this->matrix_element_2(bra_alpha, ket_alpha, ex_alpha,
                                                bra_occ_alpha, bra_occ_beta);
                }
              } else if (ex_beta_count == 2) {
                h_el = this->matrix_element_2(bra_beta, ket_beta, ex_beta,
                                              bra_occ_beta, bra_occ_alpha);
              } else {
                // Diagonal matrix element
                h_el = this->matrix_element_diag(bra_occ_alpha, bra_occ_beta);
              }

              // Direct insertion into the CSR arrays using atomic operations
              // Get the current insertion position for this row and atomically
              // increment
              index_t offset_row;
#ifdef _OPENMP
// Use OpenMP atomic capture to safely get current value and increment
#pragma omp atomic capture
              offset_row = row_offsets[ibra]++;
#else
              // Sequential fallback
              offset_row = row_offsets[ibra]++;
#endif /* _OPENMP */

              // Insert values (no synchronization needed as each thread writes
              // to unique locations)
              colind[offset_row] = iket;
              nzval[offset_row] = h_el;

              // For symmetric matrices, also handle the symmetric entry
              if (is_symm && ibra != iket) {
                index_t offset_col;
#ifdef _OPENMP
#pragma omp atomic capture
                offset_col = row_offsets[iket]++;
#else
                offset_col = row_offsets[iket]++;
#endif /* _OPENMP */

                colind[offset_col] = ibra;
                nzval[offset_col] = h_el;
              }
            }
          }
        }
      }
    }

    // Matrix fill operation is complete but columns might not be sorted
    auto fill_en = std::chrono::high_resolution_clock::now();

    // Sort CSR matrix columns within each row to maintain CSR format
    // requirements
    auto sort_st = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
      std::vector<size_t> indices;
      std::vector<index_t> sorted_cols;
      std::vector<double> sorted_vals;
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < nbra_dets; i++) {
        // Get the range for this row
        const index_t row_start = rowptr[i];
        const index_t row_end = rowptr[i + 1];
        const size_t row_size = row_end - row_start;

        if (row_size > 1) {
          // Create indices for sorting
          indices.resize(row_size);
          std::iota(indices.begin(), indices.end(),
                    0);  // Fill with 0, 1, 2, ...

          // Sort indices based on column values
          std::sort(indices.begin(), indices.end(),
                    [&colind, row_start](size_t i1, size_t i2) {
                      return colind[row_start + i1] < colind[row_start + i2];
                    });

          // Create temporary storage for sorted data
          sorted_cols.resize(row_size);
          sorted_vals.resize(row_size);

          // Reorder column indices and values using the sorted indices
          for (size_t j = 0; j < row_size; j++) {
            sorted_cols[j] = colind[row_start + indices[j]];
            sorted_vals[j] = nzval[row_start + indices[j]];
          }

          // Copy sorted data back to the original arrays
          std::copy(sorted_cols.begin(), sorted_cols.end(),
                    colind.begin() + row_start);
          std::copy(sorted_vals.begin(), sorted_vals.end(),
                    nzval.begin() + row_start);
        }
      }
    }

    auto sort_en = std::chrono::high_resolution_clock::now();

    auto thresh_st = std::chrono::high_resolution_clock::now();

    // Create CSR matrix directly
    sparse_matrix_type<index_t> csr_mat(nbra_dets, nket_dets, std::move(rowptr),
                                        std::move(colind), std::move(nzval));

    // Apply thresholding to the CSR matrix
    if (H_thresh > 0.0) {
      csr_mat.threshold_parallel(H_thresh);
      total_nnz = csr_mat.nnz();  // Update total_nnz after thresholding
    }
    auto thresh_en = std::chrono::high_resolution_clock::now();

    // Print timing information if needed
    auto duration_setup =
        std::chrono::duration<double>(count_st - setup_st).count();
    auto duration_compute =
        std::chrono::duration<double>(count_en - count_st).count();
    auto duration_alloc =
        std::chrono::duration<double>(alloc_en - alloc_st).count();
    auto duration_fill =
        std::chrono::duration<double>(fill_en - fill_st).count();
    auto duration_sort =
        std::chrono::duration<double>(sort_en - sort_st).count();
    auto duration_thresh =
        std::chrono::duration<double>(thresh_en - thresh_st).count();

    return csr_mat;
  }

  /**
   * @brief Create 32-bit CSR Hamiltonian matrix block (override)
   *
   * This method provides the interface for creating CSR matrices with 32-bit
   * indices using the optimized sorted double loop algorithm.
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
   * indices using the optimized sorted double loop algorithm.
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
   * @brief Form reduced density matrices using optimized algorithm
   *
   * This method computes one-electron and two-electron reduced density matrices
   * from determinant expansions using the same double loop approach as the
   * standard implementation, but could potentially benefit from the same
   * sorting optimizations.
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
                 double* C, matrix_span_t ordm, rank4_span_t trdm) override {
    using wfn_traits = wavefunction_traits<WfnType>;
    using spin_wfn_type = typename wfn_traits::spin_wfn_type;
    using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    const bool is_symm = bra_begin == ket_begin and bra_end == ket_end;
#ifdef MACIS_ENABLE_MPI
    auto world_rank = comm_rank(MPI_COMM_WORLD);
#else
    auto world_rank = 0;
#endif /* MACIS_ENABLE_MPI */

    // Get unique alpha strings
    auto setup_st = std::chrono::high_resolution_clock::now();
    auto unique_alpha_bra = get_unique_alpha(bra_begin, bra_end);
    auto unique_alpha_ket =
        is_symm ? unique_alpha_bra : get_unique_alpha(ket_begin, ket_end);

    const size_t nuniq_bra = unique_alpha_bra.size();
    const size_t nuniq_ket = unique_alpha_ket.size();

    // Compute offsets
    std::vector<size_t> unique_alpha_bra_idx(nuniq_bra + 1);
    std::transform_exclusive_scan(
        unique_alpha_bra.begin(), unique_alpha_bra.end(),
        unique_alpha_bra_idx.begin(), 0ul, std::plus<size_t>{},
        [](auto& x) { return x.second; });
    std::vector<size_t> unique_alpha_ket_idx(nuniq_ket + 1);
    if (is_symm) {
      unique_alpha_ket_idx = unique_alpha_bra_idx;
    } else {
      std::transform_exclusive_scan(
          unique_alpha_ket.begin(), unique_alpha_ket.end(),
          unique_alpha_ket_idx.begin(), 0ul, std::plus<size_t>{},
          [](auto& x) { return x.second; });
    }

    unique_alpha_bra_idx.back() = nbra_dets;
    unique_alpha_ket_idx.back() = nket_dets;

    auto count_st = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
      std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

#pragma omp for schedule(dynamic)
      for (size_t ia_bra = 0; ia_bra < nuniq_bra; ++ia_bra) {
        if (!unique_alpha_bra[ia_bra].first.any()) continue;

        // Extract alpha bra
        const auto bra_alpha = unique_alpha_bra[ia_bra].first;
        const size_t beta_st_bra = unique_alpha_bra_idx[ia_bra];
        const size_t beta_en_bra = unique_alpha_bra_idx[ia_bra + 1];

        const auto ket_lower = is_symm ? ia_bra : 0;
        for (size_t ia_ket = ket_lower; ia_ket < nuniq_ket; ++ia_ket) {
          if (!unique_alpha_ket[ia_ket].first.any()) continue;

          // Extract alpha ket
          const auto ket_alpha = unique_alpha_ket[ia_ket].first;

          // Compute alpha excitation
          const auto ex_alpha = bra_alpha ^ ket_alpha;
          const auto ex_alpha_count = spin_wfn_traits::count(ex_alpha);

          // Early exit if excitation level too high
          if (ex_alpha_count > 4) continue;

          const size_t beta_st_ket = unique_alpha_ket_idx[ia_ket];
          const size_t beta_en_ket = unique_alpha_ket_idx[ia_ket + 1];

          // Get occupied alpha indices
          spin_wfn_traits::state_to_occ(bra_alpha, bra_occ_alpha);

          // Loop over betas
          for (size_t ibra = beta_st_bra; ibra < beta_en_bra; ++ibra) {
            const auto bra_beta = wfn_traits::beta_string(*(bra_begin + ibra));

            // Get occupied beta indices
            spin_wfn_traits::state_to_occ(bra_beta, bra_occ_beta);

            for (size_t iket = beta_st_ket; iket < beta_en_ket; ++iket) {
              if (is_symm && (iket < ibra)) continue;

              const auto ket_beta =
                  wfn_traits::beta_string(*(ket_begin + iket));
              const auto ex_beta = bra_beta ^ ket_beta;
              const auto ex_beta_count = spin_wfn_traits::count(ex_beta);

              // Skip if total excitation level too high
              if ((ex_alpha_count + ex_beta_count) > 4) continue;

              const double val = C[ibra] * C[iket];

              // Compute Matrix Element
              if (std::abs(val) > 1e-16) {
                rdm_contributions<true>(
                    bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                    bra_occ_alpha, bra_occ_beta, val, ordm, trdm);
              }
            }
          }
        }
      }
    }
  }

  void form_rdms_spin_dep(full_det_iterator bra_begin,
                          full_det_iterator bra_end,
                          full_det_iterator ket_begin,
                          full_det_iterator ket_end, double* C,
                          matrix_span_t ordm_aa, matrix_span_t ordm_bb,
                          rank4_span_t trdm_aaaa, rank4_span_t trdm_bbbb,
                          rank4_span_t trdm_aabb) override {
    using wfn_traits = wavefunction_traits<WfnType>;
    using spin_wfn_type = typename wfn_traits::spin_wfn_type;
    using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
    const size_t nbra_dets = std::distance(bra_begin, bra_end);
    const size_t nket_dets = std::distance(ket_begin, ket_end);

    const bool is_symm = bra_begin == ket_begin and bra_end == ket_end;
#ifdef MACIS_ENABLE_MPI
    auto world_rank = comm_rank(MPI_COMM_WORLD);
#else
    auto world_rank = 0;
#endif /* MACIS_ENABLE_MPI */

    // Get unique alpha strings
    auto setup_st = std::chrono::high_resolution_clock::now();
    auto unique_alpha_bra = get_unique_alpha(bra_begin, bra_end);
    auto unique_alpha_ket =
        is_symm ? unique_alpha_bra : get_unique_alpha(ket_begin, ket_end);

    const size_t nuniq_bra = unique_alpha_bra.size();
    const size_t nuniq_ket = unique_alpha_ket.size();

    // Compute offsets
    std::vector<size_t> unique_alpha_bra_idx(nuniq_bra + 1);
    std::transform_exclusive_scan(
        unique_alpha_bra.begin(), unique_alpha_bra.end(),
        unique_alpha_bra_idx.begin(), 0ul, std::plus<size_t>{},
        [](auto& x) { return x.second; });
    std::vector<size_t> unique_alpha_ket_idx(nuniq_ket + 1);
    if (is_symm) {
      unique_alpha_ket_idx = unique_alpha_bra_idx;
    } else {
      std::transform_exclusive_scan(
          unique_alpha_ket.begin(), unique_alpha_ket.end(),
          unique_alpha_ket_idx.begin(), 0ul, std::plus<size_t>{},
          [](auto& x) { return x.second; });
    }

    unique_alpha_bra_idx.back() = nbra_dets;
    unique_alpha_ket_idx.back() = nket_dets;

    auto count_st = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
      std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

#pragma omp for schedule(dynamic)
      for (size_t ia_bra = 0; ia_bra < nuniq_bra; ++ia_bra) {
        if (!unique_alpha_bra[ia_bra].first.any()) continue;

        // Extract alpha bra
        const auto bra_alpha = unique_alpha_bra[ia_bra].first;
        const size_t beta_st_bra = unique_alpha_bra_idx[ia_bra];
        const size_t beta_en_bra = unique_alpha_bra_idx[ia_bra + 1];

        const auto ket_lower = is_symm ? ia_bra : 0;
        for (size_t ia_ket = ket_lower; ia_ket < nuniq_ket; ++ia_ket) {
          if (!unique_alpha_ket[ia_ket].first.any()) continue;

          // Extract alpha ket
          const auto ket_alpha = unique_alpha_ket[ia_ket].first;

          // Compute alpha excitation
          const auto ex_alpha = bra_alpha ^ ket_alpha;
          const auto ex_alpha_count = spin_wfn_traits::count(ex_alpha);

          // Early exit if excitation level too high
          if (ex_alpha_count > 4) continue;

          const size_t beta_st_ket = unique_alpha_ket_idx[ia_ket];
          const size_t beta_en_ket = unique_alpha_ket_idx[ia_ket + 1];

          // Get occupied alpha indices
          spin_wfn_traits::state_to_occ(bra_alpha, bra_occ_alpha);

          // Loop over betas
          for (size_t ibra = beta_st_bra; ibra < beta_en_bra; ++ibra) {
            const auto bra_beta = wfn_traits::beta_string(*(bra_begin + ibra));

            // Get occupied beta indices
            spin_wfn_traits::state_to_occ(bra_beta, bra_occ_beta);

            for (size_t iket = beta_st_ket; iket < beta_en_ket; ++iket) {
              if (is_symm && (iket < ibra)) continue;

              const auto ket_beta =
                  wfn_traits::beta_string(*(ket_begin + iket));
              const auto ex_beta = bra_beta ^ ket_beta;
              const auto ex_beta_count = spin_wfn_traits::count(ex_beta);

              // Skip if total excitation level too high
              if ((ex_alpha_count + ex_beta_count) > 4) continue;

              const double val = C[ibra] * C[iket];

              // Compute Matrix Element
              if (std::abs(val) > 1e-16) {
                rdm_contributions_spin_dep<true>(
                    bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                    bra_occ_alpha, bra_occ_beta, val, ordm_aa, ordm_bb,
                    trdm_aaaa, trdm_bbbb, trdm_aabb);
              }
            }
          }
        }
      }
    }
  }

 public:
  /**
   * @brief Constructor for SortedDoubleLoopHamiltonianGenerator
   *
   * Perfect forwarding constructor that passes all arguments to the base class.
   * The sorting optimization is implemented transparently in the matrix
   * construction methods.
   *
   * @tparam Args Parameter pack for constructor arguments
   * @param args Arguments to forward to HamiltonianGenerator constructor
   */
  template <typename... Args>
  SortedDoubleLoopHamiltonianGenerator(Args&&... args)
      : HamiltonianGenerator<WfnType>(std::forward<Args>(args)...) {}
};

}  // namespace macis
