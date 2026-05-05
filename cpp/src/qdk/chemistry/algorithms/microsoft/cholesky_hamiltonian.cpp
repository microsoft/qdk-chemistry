// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "cholesky_hamiltonian.hpp"

// STL Headers
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
#include <tuple>
#include <unordered_set>

// MACIS Headers
#include <macis/mcscf/fock_matrices.hpp>

// QDK/Chemistry SCF headers
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/util/int1e.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

// Schwarz screening
#include "scf/src/eri/schwarz.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <blas.hh>

// QDK/Chemistry data::Hamiltonian headers
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail {

/**
 * @brief Compute Cholesky decomposition of the two-electron integral tensor
 *
 * Performs a pivoted Cholesky decomposition of the (μν|λσ) ERI tensor using
 * the Libint2 library for on-the-fly integral evaluation. The decomposition
 * approximates the ERI tensor as:
 *   (μν|λσ) ≈ Σ_K L^K_{μν} L^K_{λσ}
 *
 * where L^K are the Cholesky vectors and K is the Cholesky rank.
 *
 * @param basis_set QDK basis set for the molecule
 * @param threshold Cholesky decomposition threshold controlling accuracy
 * @param eri_threshold ERI screening threshold for skipping negligible
 *        shell quartets during integral evaluation
 * @return Tuple of (pointer to array of Cholesky vectors stored in column-major
 *         order with dimensions (num_aos * num_aos) × num_vectors, number of
 *         Cholesky vectors generated (rank))
 */
std::tuple<std::vector<double>, size_t> compute_cholesky_vectors(
    qdk::chemistry::scf::BasisSet& basis_set, double threshold,
    double eri_threshold) {
  QDK_LOG_TRACE_ENTERING();
  QDK_LOGGER().info("Cholesky decomposition threshold: {}", threshold);
  QDK_LOGGER().info("ERI screening threshold: {}", eri_threshold);

  using qdk::chemistry::scf::RowMajorMatrix;

  // Convert to libint2 basis set
  auto obs =
      qdk::chemistry::scf::libint2_util::convert_to_libint_basisset(basis_set);
  auto shell2bf = obs.shell2bf();

  // Compute Schwarz screening matrix
  const size_t num_shells_schwarz = obs.size();
  RowMajorMatrix K_schwarz(num_shells_schwarz, num_shells_schwarz);
  auto mpi = qdk::chemistry::scf::mpi_default_input();
  qdk::chemistry::scf::schwarz_integral(&basis_set, mpi, K_schwarz.data());

  const size_t num_aos = obs.nbf();
  const size_t num_aos2 = num_aos * num_aos;
  const size_t num_shells = obs.size();
  const size_t num_shell_pairs = num_shells * (num_shells + 1) / 2;

  // Cholesky decomposition (rank is bounded by num_aos*(num_aos+1)/2)
  const size_t max_rank = num_aos * (num_aos + 1) / 2;
  QDK_LOGGER().debug("Maximum possible Cholesky rank: {}", max_rank);

  // Precompute upper bound for shell-pair block columns: n_cols = n1 * n2.
  // This enables reusing ERI column buffers across iterations.
  size_t max_shell_size = 0;
  for (size_t s = 0; s < num_shells; ++s) {
    max_shell_size = std::max(max_shell_size, obs[s].size());
  }
  const size_t max_n_cols = max_shell_size * max_shell_size;

  // Fix threshold to (= sqrt(max_rank) * eps), to prevent numerical noise.
  const double min_threshold = std::sqrt(static_cast<double>(max_rank)) *
                               std::numeric_limits<double>::epsilon();
  if (threshold < min_threshold) {
    QDK_LOGGER().warn(
        "Cholesky threshold {:.2e} set to {:.2e} (= sqrt({}) * eps) to prevent "
        "noise.",
        threshold, min_threshold, max_rank);
    threshold = min_threshold;
  }

  // map shell pair to index
  auto shell_pair_index = [num_shell_pairs](size_t s1, size_t s2) {
    if (s1 < s2) std::swap(s1, s2);
    return s2 + (s1 * (s1 + 1)) / 2;
  };

  // Precompute sp_index to (s1, s2) mapping
  std::vector<std::pair<size_t, size_t>> sp_index_to_shells(num_shell_pairs);
  for (size_t s1 = 0; s1 < num_shells; ++s1) {
    for (size_t s2 = 0; s2 <= s1; ++s2) {
      const size_t sp_idx = shell_pair_index(s1, s2);
      sp_index_to_shells[sp_idx] = {s1, s2};
    }
  }

  // setup libint engine for ERI computation
  const auto engine_precision = std::numeric_limits<double>::epsilon();
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif
  std::vector<::libint2::Engine> engines_coulomb(nthreads);
  engines_coulomb[0] = ::libint2::Engine(::libint2::Operator::coulomb,
                                         obs.max_nprim(), obs.max_l(), 0);
  engines_coulomb[0].set(::libint2::ScreeningMethod::Original);
  engines_coulomb[0].set_precision(engine_precision);
  for (int i = 1; i < nthreads; ++i) engines_coulomb[i] = engines_coulomb[0];

  // index of current cholesky vector
  size_t current_col = 0;
  std::vector<double> L_data;
  // Reserve number of aos for Cholesky vectors
  size_t estimated_rank = num_aos;
  L_data.reserve(num_aos2 * estimated_rank);

  // Diagonal elements and indices
  std::vector<std::vector<double>> D_shell_pair(num_shell_pairs);
  std::unordered_set<size_t> active_shell_pairs;

#ifdef _OPENMP
  //  Thread-local active shell pair lists
  std::vector<std::vector<size_t>> active_shell_pairs_local(nthreads);
#endif

  // Reusable ERI column buffer — single shared buffer, no per-thread copies.
  std::vector<double> eri_col_max(num_aos2 * max_n_cols, 0.0);

  // Compute diagonal elements for all shell pairs
  QDK_LOGGER().debug("Computing diagonal elements");
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    const auto thread_id = omp_get_thread_num();
#else
    const auto thread_id = 0;
#endif
    auto& engine = engines_coulomb[thread_id];
    const auto& buf = engine.results();

    for (size_t s1 = 0, s12 = 0; s1 < num_shells; ++s1) {
      const auto n1 = obs[s1].size();
      for (size_t s2 = 0; s2 <= s1; ++s2) {
        // Assign to threads
        if ((s12++) % nthreads != thread_id) continue;

        const auto n2 = obs[s2].size();
        const size_t sp_index = shell_pair_index(s1, s2);
        const size_t n12 = n1 * n2;

        // screening via schwarz bounds
        if (K_schwarz(s1, s2) * K_schwarz(s1, s2) < eri_threshold) {
          continue;
        }

        // compute diagonal block (s1,s2|s1,s2)
        engine.compute2<::libint2::Operator::coulomb, ::libint2::BraKet::xx_xx,
                        0>(obs[s1], obs[s2], obs[s1], obs[s2]);
        const auto& res = buf[0];
        if (res == nullptr) {
          continue;
        }

        // local diagonal block
        D_shell_pair[sp_index].resize(n12);
        for (size_t i = 0; i < n12; ++i) {
          D_shell_pair[sp_index][i] = res[i * n12 + i];
        }
#ifdef _OPENMP
        active_shell_pairs_local[thread_id].push_back(sp_index);
#else
        active_shell_pairs.insert(sp_index);
#endif
      }  // s2
    }  // s1
  }  // omp parallel

  // Merge thread-local active lists
#ifdef _OPENMP
  for (int t = 0; t < nthreads; ++t) {
    for (size_t sp_index : active_shell_pairs_local[t]) {
      active_shell_pairs.insert(sp_index);
    }
  }
#endif

  // Target number of GEMM columns per batched orthogonalization call.
  // Amortizes memory-bandwidth cost of reading L_data by combining multiple
  // shell pairs into one GEMM. Auto-adapts to basis set: small shells (s,p)
  // batch many; large shells (d,f) batch few. Value of 20 saturates bandwidth
  // on typical hardware.
  constexpr size_t TARGET_GEMM_COLS = 20;

  QDK_LOGGER().debug("Cholesky Rank | Max Diagonal Element");
  double D_max = 0.0;
  while (current_col < max_rank) {
    if (active_shell_pairs.empty()) {
      QDK_LOGGER().debug("{:>13} | all shell pairs converged", current_col);
      break;
    }

    // === Step 1: Select top-B shell pairs by max diagonal ===
    struct SPInfo {
      size_t sp_index, s1, s2;
      double max_diag;
    };
    std::vector<SPInfo> sp_list;
    sp_list.reserve(active_shell_pairs.size());
    for (const auto sp_index : active_shell_pairs) {
      const auto [s1, s2] = sp_index_to_shells[sp_index];
      const auto& diag = D_shell_pair[sp_index];
      const double block_max = *std::max_element(diag.begin(), diag.end());
      sp_list.push_back({sp_index, s1, s2, block_max});
    }
    // Sort all active shell pairs by max diagonal (descending)
    std::sort(sp_list.begin(), sp_list.end(),
              [](const SPInfo& a, const SPInfo& b) {
                return a.max_diag > b.max_diag;
              });

    D_max = sp_list[0].max_diag;
    if (D_max < threshold) {
      QDK_LOGGER().debug("{:>13} | {}", current_col, D_max);
      break;
    }

    // Accumulate shell pairs until we reach the target column count
    size_t n_batch = 0;
    size_t total_n_cols = 0;
    for (size_t b = 0; b < sp_list.size(); ++b) {
      if (sp_list[b].max_diag < threshold) break;
      const size_t n1 = obs[sp_list[b].s1].size();
      const size_t n2 = obs[sp_list[b].s2].size();
      total_n_cols += n1 * n2;
      n_batch++;
      if (total_n_cols >= TARGET_GEMM_COLS) break;
    }
    if (n_batch == 0) break;

    // Build batch entries
    struct BatchEntry {
      size_t sp_index, s1, s2, n1, n2, n_cols, col_offset;
      size_t bf1_st, bf2_st;
    };
    std::vector<BatchEntry> batch(n_batch);
    total_n_cols = 0;  // recompute for accurate offsets
    for (size_t b = 0; b < n_batch; ++b) {
      const auto& sp = sp_list[b];
      const size_t n1 = obs[sp.s1].size();
      const size_t n2 = obs[sp.s2].size();
      batch[b] = {
          sp.sp_index,  sp.s1,           sp.s2,          n1, n2, n1 * n2,
          total_n_cols, shell2bf[sp.s1], shell2bf[sp.s2]};
      total_n_cols += n1 * n2;
    }

    QDK_LOGGER().debug("{:>13} | {} | batch={}", current_col, D_max, n_batch);

    // === Step 2: Compute ERI columns for ALL batch entries ===
    // Each shell pair writes to its own section of eri_col_batch.
    // No dependency between shell pairs — fully parallel.
    const size_t eri_batch_size = num_aos2 * total_n_cols;
    std::vector<double> eri_col_batch(eri_batch_size, 0.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const auto thread_id = omp_get_thread_num();
#else
      const auto thread_id = 0;
#endif
      auto& engine = engines_coulomb[thread_id];
      const auto& buf = engine.results();

      for (size_t b = 0; b < n_batch; ++b) {
        const auto& be = batch[b];
        double* eri_out = eri_col_batch.data() + be.col_offset * num_aos2;

        for (size_t s3 = 0, s34 = 0; s3 < num_shells; ++s3) {
          const size_t n3 = obs[s3].size();
          const size_t bf3_st = shell2bf[s3];
          for (size_t s4 = 0; s4 < num_shells; ++s4) {
            if ((s34++) % nthreads != thread_id) continue;

            if (K_schwarz(be.s1, be.s2) * K_schwarz(s3, s4) < eri_threshold) {
              continue;
            }

            const size_t n4 = obs[s4].size();
            const size_t bf4_st = shell2bf[s4];

            engine.compute2<::libint2::Operator::coulomb,
                            ::libint2::BraKet::xx_xx, 0>(obs[be.s1], obs[be.s2],
                                                         obs[s3], obs[s4]);
            const auto& res = buf[0];
            if (res == nullptr) continue;

            for (size_t i = 0, ijkl = 0; i < be.n1; ++i) {
              const size_t ind_i = i * be.n2;
              for (size_t j = 0; j < be.n2; ++j) {
                const size_t ind_ij = (ind_i + j) * num_aos2;
                for (size_t k = 0; k < n3; ++k) {
                  const size_t ind_ijk = ind_ij + (bf3_st + k) * num_aos;
                  for (size_t l = 0; l < n4; ++l, ++ijkl) {
                    eri_out[ind_ijk + (bf4_st + l)] += res[ijkl];
                  }
                }
              }
            }
          }  // s4
        }  // s3
      }  // for each batch entry
    }  // omp parallel
    // === Step 3: ONE big GEMM for all columns at once ===
    // Precompute lookup for all columns
    std::vector<size_t> all_lookup(total_n_cols);
    for (size_t b = 0; b < n_batch; ++b) {
      const auto& be = batch[b];
      for (size_t i = 0; i < be.n1; ++i) {
        for (size_t j = 0; j < be.n2; ++j) {
          const size_t local_idx = i * be.n2 + j;
          all_lookup[be.col_offset + local_idx] =
              (be.bf1_st + i) * num_aos + (be.bf2_st + j);
        }
      }
    }

    if (current_col > 0) {
      // Gather rows: L_rows[col, :] = L_data[all_lookup[col], :]
      std::vector<double> L_rows(total_n_cols * current_col);
      for (size_t col = 0; col < total_n_cols; ++col) {
        blas::copy(current_col, L_data.data() + all_lookup[col], num_aos2,
                   L_rows.data() + col, total_n_cols);
      }
      // One big GEMM: eri_col_batch -= L_data * L_rows^T
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                 num_aos2, total_n_cols, current_col, -1.0, L_data.data(),
                 num_aos2, L_rows.data(), total_n_cols, 1.0,
                 eri_col_batch.data(), num_aos2);
    }
    // === Step 4: Form vectors (sequentially within batch) ===
    // Process each batch entry, forming vectors and doing in-batch
    // Gram-Schmidt against vectors formed earlier in this batch.
    for (size_t b = 0; b < n_batch; ++b) {
      const auto& be = batch[b];
      double* eri_col = eri_col_batch.data() + be.col_offset * num_aos2;

      for (size_t local_i = 0; local_i < be.n1; ++local_i) {
        const size_t j_max = (be.s1 == be.s2) ? (local_i + 1) : be.n2;
        for (size_t local_j = 0; local_j < j_max; ++local_j) {
          const size_t local_index = local_i * be.n2 + local_j;
          const double D_val = D_shell_pair[be.sp_index][local_index];

          if (D_val < threshold) continue;

          double Q_max = std::sqrt(1.0 / D_val);
          std::vector<double> L_col_vec(num_aos2);
          blas::copy(num_aos2, eri_col + local_index * num_aos2, 1,
                     L_col_vec.data(), 1);
          blas::scal(num_aos2, Q_max, L_col_vec.data(), 1);

          L_data.insert(L_data.end(), L_col_vec.data(),
                        L_col_vec.data() + num_aos2);

          const double* L_col = L_data.data() + current_col * num_aos2;

          // In-batch Gram-Schmidt: update remaining ERI columns in this
          // shell pair (same as before)
          for (size_t col = local_index + 1; col < be.n1 * be.n2; ++col) {
            const size_t global_col_idx =
                (be.bf1_st + col / be.n2) * num_aos + (be.bf2_st + col % be.n2);
            const double scale_factor = -L_col[global_col_idx];
            blas::axpy(num_aos2, scale_factor, L_col, 1,
                       eri_col + col * num_aos2, 1);
          }

          // ALSO update ERI columns of LATER batch entries
          // (cross-batch Gram-Schmidt)
          for (size_t b2 = b + 1; b2 < n_batch; ++b2) {
            const auto& be2 = batch[b2];
            double* eri_col2 = eri_col_batch.data() + be2.col_offset * num_aos2;
            for (size_t col2 = 0; col2 < be2.n1 * be2.n2; ++col2) {
              const size_t g_idx = (be2.bf1_st + col2 / be2.n2) * num_aos +
                                   (be2.bf2_st + col2 % be2.n2);
              const double sf = -L_col[g_idx];
              blas::axpy(num_aos2, sf, L_col, 1, eri_col2 + col2 * num_aos2, 1);
            }
          }

          // Update diagonal elements
          std::vector<size_t> shell_pairs_to_remove;
          for (const auto sp_index : active_shell_pairs) {
            const auto [s1, s2] = sp_index_to_shells[sp_index];
            const auto n1 = obs[s1].size();
            const auto n2 = obs[s2].size();
            const auto bf1_st = shell2bf[s1];
            const auto bf2_st = shell2bf[s2];
            for (size_t i = 0; i < n1; ++i) {
              const size_t bf1_st_i = (bf1_st + i) * num_aos;
              for (size_t j = 0; j < n2; ++j) {
                const size_t idx = i * n2 + j;
                const size_t global_idx = bf1_st_i + (bf2_st + j);
                D_shell_pair[sp_index][idx] -=
                    L_col[global_idx] * L_col[global_idx];
              }
            }
            const auto& diag = D_shell_pair[sp_index];
            const double max_diag = *std::max_element(diag.begin(), diag.end());
            if (max_diag < threshold) {
              shell_pairs_to_remove.push_back(sp_index);
            }
          }
          for (const auto sp_index : shell_pairs_to_remove) {
            active_shell_pairs.erase(sp_index);
          }
          current_col += 1;
        }
      }
    }  // for each batch entry
  }

  QDK_LOGGER().info("Cholesky rank: {}", current_col);

  if (current_col == max_rank) {
    QDK_LOGGER().warn(
        "Cholesky decomposition reached maximum rank (num_aos*(num_aos+1)/2 = "
        "{}). The requested threshold {} may not have been achieved.",
        max_rank, threshold);
  }

  return {std::move(L_data), current_col};
}

Eigen::MatrixXd transform_cholesky_to_mo(
    const Eigen::MatrixXd& ao_cholesky_vectors,
    const Eigen::MatrixXd& mo_coeffs) {
  QDK_LOG_TRACE_ENTERING();
  size_t n_ao = mo_coeffs.rows();
  size_t n_mo = mo_coeffs.cols();
  size_t rank = ao_cholesky_vectors.cols();

  // Validate dimensions
  if (n_ao == 0 || n_mo == 0) {
    throw std::invalid_argument("C matrix has zero dimensions");
  }
  if (ao_cholesky_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_cholesky_vectors dimensions do not match n_ao");
  }

  Eigen::MatrixXd mo_vectors(n_mo * n_mo, rank);

  // iterate over each Cholesky vector
  for (size_t k = 0; k < rank; ++k) {
    // Reshape the flat AO vector to a matrix (n_ao x n_ao)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        V_ao(ao_cholesky_vectors.col(k).data(), n_ao, n_ao);

    // Transform from AO to MO basis: C^T * V_ao * C
    // Write directly to output column
    Eigen::Map<Eigen::MatrixXd> V_mo_map(mo_vectors.col(k).data(), n_mo, n_mo);
    V_mo_map.noalias() = mo_coeffs.transpose() * V_ao * mo_coeffs;
  }

  return mo_vectors;
}

Eigen::MatrixXd build_J_from_cholesky(
    const Eigen::MatrixXd& ao_cholesky_vectors,
    const Eigen::MatrixXd& density) {
  QDK_LOG_TRACE_ENTERING();
  size_t n_ao = density.rows();
  size_t rank = ao_cholesky_vectors.cols();

  // Validate dimensions
  if (density.cols() != density.rows()) {
    throw std::invalid_argument("Density matrix must be square");
  }
  if (ao_cholesky_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_cholesky_vectors dimensions do not match density matrix");
  }

  // Flatten density matrix
  Eigen::Map<const Eigen::VectorXd> density_vec(density.data(), n_ao * n_ao);

  // Compute all inner products at once: V_k = sum_{mu,nu} L^k_{mu,nu} *
  // P_{mu,nu} V = L^T * vec(P)
  Eigen::VectorXd V = ao_cholesky_vectors.transpose() * density_vec;

  // Reconstruct J: J = sum_k L_k * V_k = L * V (reshaped)
  Eigen::VectorXd J_vec = ao_cholesky_vectors * V;

  // Reshape back to matrix
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      J(J_vec.data(), n_ao, n_ao);
  return J;
}

Eigen::MatrixXd build_K_from_cholesky(
    const Eigen::MatrixXd& ao_cholesky_vectors, const Eigen::MatrixXd& coeffs,
    const std::vector<size_t>& occ_orb_ind) {
  QDK_LOG_TRACE_ENTERING();
  size_t n_ao = coeffs.rows();
  size_t n_occ = occ_orb_ind.size();
  size_t rank = ao_cholesky_vectors.cols();

  // Validate dimensions
  if (ao_cholesky_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_cholesky_vectors dimensions do not match density matrix");
  }

  // Extract occupied orbital coefficients only
  Eigen::MatrixXd C_occ(n_ao, n_occ);
  for (size_t idx = 0; idx < n_occ; ++idx) {
    C_occ.col(idx) = coeffs.col(occ_orb_ind[idx]);
  }

  // Transform to occupied MO basis only: L^k_{\sigma,i_occ} = L^k_{\mu\sigma} *
  // C_{\mu,i_occ}
  Eigen::MatrixXd L_sigma_occ(n_ao * n_occ, rank);
  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        L_k(ao_cholesky_vectors.col(k).data(), n_ao, n_ao);
    Eigen::Map<Eigen::MatrixXd> L_k_occ(L_sigma_occ.col(k).data(), n_ao, n_occ);
    L_k_occ.noalias() = L_k * C_occ;
  }

  // Build K_{\lambda\sigma} = \sum_k L^k_{\lambda,i} * L^k_{\sigma,i}
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_ao, n_ao);
  for (size_t k = 0; k < rank; ++k) {
    Eigen::Map<const Eigen::MatrixXd> L_k_occ(L_sigma_occ.col(k).data(), n_ao,
                                              n_occ);
    K.noalias() += L_k_occ * L_k_occ.transpose();
  }

  return K;
}

}  // namespace detail

namespace detail_chol {
/**
 * @brief Validate active orbital indices
 * @param indices The indices to validate
 * @param spin_label Label for error messages (e.g., "Alpha", "Beta")
 * @param num_molecular_orbitals Total number of molecular orbitals
 * @return true if the indices are contiguous, false otherwise
 */
bool validate_active_contiguous_indices(const std::vector<size_t>& indices,
                                        const std::string& spin_label,
                                        size_t num_molecular_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  if (indices.empty()) return true;

  // Cannot contain more than the total number of MOs
  if (indices.size() > num_molecular_orbitals) {
    throw std::runtime_error("Number of requested " + spin_label +
                             " active orbitals exceeds total number of MOs");
  }

  // Make sure that the indices are within bounds
  for (const auto& idx : indices) {
    if (static_cast<size_t>(idx) >= num_molecular_orbitals) {
      throw std::runtime_error(
          spin_label +
          " active orbital index out of bounds: " + std::to_string(idx));
    }
  }

  // Make sure that the indices are unique
  std::set<size_t> unique_indices(indices.begin(), indices.end());
  if (unique_indices.size() != indices.size()) {
    throw std::runtime_error(spin_label +
                             " active orbital indices must be unique");
  }

  // Make sure that the indices are sorted
  std::vector<size_t> sorted_indices(indices.begin(), indices.end());
  std::sort(sorted_indices.begin(), sorted_indices.end());
  if (indices != sorted_indices) {
    throw std::runtime_error(spin_label +
                             " active orbital indices must be sorted");
  }

  // Check if indices are contiguous
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    if (indices[i + 1] - indices[i] != 1) {
      return false;
    }
  }

  return true;
}
}  // namespace detail_chol

std::shared_ptr<data::Hamiltonian> CholeskyHamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals) const {
  QDK_LOG_TRACE_ENTERING();
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  auto basis_set = orbitals->get_basis_set();
  const auto& [Ca, Cb] = orbitals->get_coefficients();
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Get alpha and beta active space indices
  auto active_space_indices = orbitals->get_active_space_indices();
  auto active_indices_alpha = active_space_indices.first;
  auto active_indices_beta = active_space_indices.second;

  if (orbitals->is_restricted() && active_indices_alpha.empty()) {
    throw std::runtime_error("Need to specify an active space.");
  } else if (orbitals->is_unrestricted() &&
             (active_indices_alpha.empty() || active_indices_beta.empty())) {
    throw std::runtime_error(
        "Need to specify an active space for alpha and beta.");
  }

  const size_t nactive_alpha = active_indices_alpha.size();
  const size_t nactive_beta = active_indices_beta.size();

  // Validate alpha active orbitals and check contiguity
  bool alpha_space_is_contiguous =
      detail_chol::validate_active_contiguous_indices(
          active_indices_alpha, "Alpha", num_molecular_orbitals);

  // Validate beta active orbitals (if different from alpha) and check
  // contiguity
  bool beta_space_is_contiguous = true;
  if (active_indices_beta != active_indices_alpha) {
    beta_space_is_contiguous = detail_chol::validate_active_contiguous_indices(
        active_indices_beta, "Beta", num_molecular_orbitals);
  } else {
    beta_space_is_contiguous = alpha_space_is_contiguous;
  }

  // Ensure alpha and beta active spaces have the same size
  if (nactive_alpha != nactive_beta) {
    throw std::runtime_error(
        "Alpha and beta active spaces must have the same size. "
        "Alpha: " +
        std::to_string(nactive_alpha) +
        ", Beta: " + std::to_string(nactive_beta));
  }

  // Create internal Molecule
  auto structure = basis_set->get_structure();

  // Create internal BasisSet (includes ECP-adjusted nuclear charges)
  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  // Create dummy SCFConfig
  auto scf_config = std::make_unique<qcs::SCFConfig>();

  // Use the default MPI configuration (fallback to serial if MPI not enabled)
  scf_config->mpi = qcs::mpi_default_input();
  scf_config->require_gradient = false;
  scf_config->basis = internal_basis_set->name;
  scf_config->cartesian = !internal_basis_set->pure;
  scf_config->scf_orbital_type = qcs::SCFOrbitalType::Restricted;
  scf_config->eri.method = qcs::ERIMethod::Libint2Direct;
  scf_config->k_eri.method = qcs::ERIMethod::Libint2Direct;

  // Create Integral Instance
  auto eri = qcs::ERIMultiplexer::create(*internal_basis_set, *scf_config, 0.0);
  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      internal_basis_set.get(), internal_basis_set->mol.get(), scf_config->mpi);

  // Compute Core Hamiltonian in AO basis
  Eigen::MatrixXd T_full(num_atomic_orbitals, num_atomic_orbitals),
      V_full(num_atomic_orbitals, num_atomic_orbitals);
  int1e->kinetic_integral(T_full.data());
  int1e->nuclear_integral(V_full.data());
  Eigen::MatrixXd H_full = T_full + V_full;

  // Add ECP integrals if present
  if (internal_basis_set->ecp_shells.size() > 0) {
    Eigen::MatrixXd ECP_full =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    int1e->ecp_integral(ECP_full.data());
    H_full += ECP_full;
  }

  // Build active coefficient matrices for alpha and beta (can have different
  // sizes)
  Eigen::MatrixXd Ca_active(num_atomic_orbitals, nactive_alpha);
  Eigen::MatrixXd Cb_active(num_atomic_orbitals, nactive_beta);

  if (alpha_space_is_contiguous) {
    // Contiguous alpha indices
    Ca_active = Ca.block(0, active_indices_alpha.front(), num_atomic_orbitals,
                         nactive_alpha);
  } else {
    // Non-contiguous alpha indices
    for (size_t i = 0; i < nactive_alpha; i++) {
      Ca_active.col(i) = Ca.col(active_indices_alpha[i]);
    }
  }

  if (beta_space_is_contiguous) {
    // Contiguous beta indices
    Cb_active = Cb.block(0, active_indices_beta.front(), num_atomic_orbitals,
                         nactive_beta);
  } else {
    // Non-contiguous beta indices
    for (size_t i = 0; i < nactive_beta; i++) {
      Cb_active.col(i) = Cb.col(active_indices_beta[i]);
    }
  }

  // Convert to row-major for MOERI
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ca_active_rm = Ca_active;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Cb_active_rm = Cb_active;

  // Determine SCF type from settings
  std::string scf_type = _settings->get<std::string>("scf_type");

  bool is_restricted_calc;
  if (scf_type == "restricted") {
    is_restricted_calc = true;
  } else if (scf_type == "unrestricted") {
    is_restricted_calc = false;
  } else {  // "auto"
    is_restricted_calc = (active_indices_alpha == active_indices_beta) &&
                         orbitals->is_restricted();
  }

  // SCFOrbitalType::RestrictedOpenShell is not supported for Hamiltonian
  // construction, so we only set Restricted for all restricted cases
  scf_config->scf_orbital_type = is_restricted_calc
                                     ? qcs::SCFOrbitalType::Restricted
                                     : qcs::SCFOrbitalType::Unrestricted;

  // Compute integrals (same size for alpha and beta)
  const size_t nactive = nactive_alpha;

  // Use Cholesky Decomposition
  double cholesky_tol = _settings->get<double>("cholesky_tolerance");
  double eri_tol = _settings->get<double>("eri_threshold");

  // get cholesky vectors
  auto [output, num_cholesky_vectors] = detail::compute_cholesky_vectors(
      *internal_basis_set, cholesky_tol, eri_tol);

  // map output to Eigen matrix
  Eigen::Map<const Eigen::MatrixXd> L_ao(
      output.data(), num_atomic_orbitals * num_atomic_orbitals,
      num_cholesky_vectors);

  Eigen::MatrixXd L_mo;
  Eigen::MatrixXd L_mo_alpha;
  Eigen::MatrixXd L_mo_beta;

  if (is_restricted_calc) {
    // Transform to MO
    L_mo = detail::transform_cholesky_to_mo(L_ao, Ca_active);
  } else {
    // Transform to MO (Alpha and Beta)
    L_mo_alpha = detail::transform_cholesky_to_mo(L_ao, Ca_active);
    L_mo_beta = detail::transform_cholesky_to_mo(L_ao, Cb_active);
  }

  // Get inactive space indices for both alpha and beta
  auto [inactive_indices_alpha, inactive_indices_beta] =
      orbitals->get_inactive_space_indices();

  // For restricted calculations, alpha and beta inactive spaces should be
  // identical
  if (orbitals->is_restricted() &&
      inactive_indices_alpha != inactive_indices_beta) {
    throw std::runtime_error(
        "For restricted orbitals, alpha and beta inactive spaces must be "
        "identical");
  }

  // all occupied orbitals specified as active
  if (inactive_indices_alpha.empty() && inactive_indices_beta.empty()) {
    if (is_restricted_calc) {
      // Use restricted constructor
      Eigen::MatrixXd H_active(nactive, nactive);
      H_active = Ca_active.transpose() * H_full * Ca_active;
      Eigen::MatrixXd dummy_fock = Eigen::MatrixXd::Zero(0, 0);
      if (_settings->get<bool>("store_ao_cholesky_vectors")) {
        return std::make_shared<data::Hamiltonian>(
            std::make_unique<data::CholeskyHamiltonianContainer>(
                H_active, L_mo, orbitals,
                structure->calculate_nuclear_repulsion_energy(), dummy_fock,
                L_ao));
      }
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::CholeskyHamiltonianContainer>(
              H_active, L_mo, orbitals,
              structure->calculate_nuclear_repulsion_energy(), dummy_fock));
    } else {
      // Use unrestricted constructor
      Eigen::MatrixXd H_active_alpha(nactive, nactive);
      Eigen::MatrixXd H_active_beta(nactive, nactive);
      H_active_alpha = Ca_active.transpose() * H_full * Ca_active;
      H_active_beta = Cb_active.transpose() * H_full * Cb_active;
      Eigen::MatrixXd dummy_fock_alpha = Eigen::MatrixXd::Zero(0, 0);
      Eigen::MatrixXd dummy_fock_beta = Eigen::MatrixXd::Zero(0, 0);
      if (_settings->get<bool>("store_ao_cholesky_vectors")) {
        return std::make_shared<data::Hamiltonian>(
            std::make_unique<data::CholeskyHamiltonianContainer>(
                H_active_alpha, H_active_beta, L_mo_alpha, L_mo_beta, orbitals,
                structure->calculate_nuclear_repulsion_energy(),
                dummy_fock_alpha, dummy_fock_beta, L_ao));
      }
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::CholeskyHamiltonianContainer>(
              H_active_alpha, H_active_beta, L_mo_alpha, L_mo_beta, orbitals,
              structure->calculate_nuclear_repulsion_energy(), dummy_fock_alpha,
              dummy_fock_beta));
    }
  }

  if (is_restricted_calc) {
    // Restricted case
    const auto& inactive_indices = inactive_indices_alpha;

    // Determine whether the inactive space is contiguous
    bool inactive_space_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices.size() - 1; ++i) {
      if (inactive_indices[i + 1] - inactive_indices[i] != 1) {
        inactive_space_is_contiguous = false;
        break;
      }
    }

    // Compute the inactive density matrix
    Eigen::MatrixXd D_inactive =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    if (inactive_space_is_contiguous) {
      auto C_inactive = Ca.block(0, inactive_indices.front(),
                                 num_atomic_orbitals, inactive_indices.size());
      D_inactive = C_inactive * C_inactive.transpose();
    } else {
      for (size_t i : inactive_indices) {
        D_inactive += Ca.col(i) * Ca.col(i).transpose();
      }
    }

    // Compute the two electron part of the inactive fock matrix
    Eigen::MatrixXd J_inactive_ao, K_inactive_ao;
    // Use Cholesky vectors to build J and K
    J_inactive_ao = detail::build_J_from_cholesky(L_ao, D_inactive);
    K_inactive_ao = detail::build_K_from_cholesky(L_ao, Ca, inactive_indices);
    Eigen::MatrixXd G_inactive_ao = 2 * J_inactive_ao - K_inactive_ao;

    // Compute the inactive Fock matrix
    Eigen::MatrixXd F_inactive_ao = G_inactive_ao + H_full;
    Eigen::MatrixXd F_inactive(num_molecular_orbitals, num_molecular_orbitals);
    F_inactive = Ca.transpose() * F_inactive_ao * Ca;

    // Compute the inactive energy
    double E_inactive = 0.0;
    Eigen::MatrixXd H_mo = Ca.transpose() * H_full * Ca;
    for (auto i : inactive_indices) {
      E_inactive += H_mo(i, i) + F_inactive(i, i);
    }

    // Extract active space Hamiltonian
    Eigen::MatrixXd H_active(nactive, nactive);
    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active(i, j) =
            F_inactive(active_indices_alpha[i], active_indices_alpha[j]);
      }
    }

    if (_settings->get<bool>("store_ao_cholesky_vectors")) {
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::CholeskyHamiltonianContainer>(
              H_active, L_mo, orbitals,
              E_inactive + structure->calculate_nuclear_repulsion_energy(),
              F_inactive, L_ao));
    }
    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::CholeskyHamiltonianContainer>(
            H_active, L_mo, orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive));
  } else {
    // Unrestricted case

    // Determine whether the alpha inactive space is contiguous
    bool alpha_inactive_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices_alpha.size() - 1; ++i) {
      if (inactive_indices_alpha[i + 1] - inactive_indices_alpha[i] != 1) {
        alpha_inactive_is_contiguous = false;
        break;
      }
    }

    // Determine whether the beta inactive space is contiguous
    bool beta_inactive_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices_beta.size() - 1; ++i) {
      if (inactive_indices_beta[i + 1] - inactive_indices_beta[i] != 1) {
        beta_inactive_is_contiguous = false;
        break;
      }
    }

    // Compute separate alpha and beta inactive density matrices
    Eigen::MatrixXd D_inactive_alpha =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    Eigen::MatrixXd D_inactive_beta =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);

    // Build alpha inactive density
    if (alpha_inactive_is_contiguous && !inactive_indices_alpha.empty()) {
      auto C_inactive_alpha =
          Ca.block(0, inactive_indices_alpha.front(), num_atomic_orbitals,
                   inactive_indices_alpha.size());
      D_inactive_alpha = C_inactive_alpha * C_inactive_alpha.transpose();
    } else {
      for (size_t i : inactive_indices_alpha) {
        D_inactive_alpha += Ca.col(i) * Ca.col(i).transpose();
      }
    }

    // Build beta inactive density
    if (beta_inactive_is_contiguous && !inactive_indices_beta.empty()) {
      auto C_inactive_beta =
          Cb.block(0, inactive_indices_beta.front(), num_atomic_orbitals,
                   inactive_indices_beta.size());
      D_inactive_beta = C_inactive_beta * C_inactive_beta.transpose();
    } else {
      for (size_t i : inactive_indices_beta) {
        D_inactive_beta += Cb.col(i) * Cb.col(i).transpose();
      }
    }

    // Compute J and K matrices for alpha and beta densities
    Eigen::MatrixXd J_alpha_ao, K_alpha_ao, J_beta_ao, K_beta_ao;
    // Use Cholesky vectors to build J and K
    J_alpha_ao = detail::build_J_from_cholesky(L_ao, D_inactive_alpha);
    K_alpha_ao =
        detail::build_K_from_cholesky(L_ao, Ca, inactive_indices_alpha);
    J_beta_ao = detail::build_J_from_cholesky(L_ao, D_inactive_beta);
    K_beta_ao = detail::build_K_from_cholesky(L_ao, Cb, inactive_indices_beta);

    Eigen::MatrixXd F_inactive_alpha_ao =
        H_full + J_alpha_ao + J_beta_ao - K_alpha_ao;
    Eigen::MatrixXd F_inactive_beta_ao =
        H_full + J_alpha_ao + J_beta_ao - K_beta_ao;

    // Transform to MO basis
    Eigen::MatrixXd F_inactive_alpha(num_molecular_orbitals,
                                     num_molecular_orbitals);
    Eigen::MatrixXd F_inactive_beta(num_molecular_orbitals,
                                    num_molecular_orbitals);
    F_inactive_alpha = Ca.transpose() * F_inactive_alpha_ao * Ca;
    F_inactive_beta = Cb.transpose() * F_inactive_beta_ao * Cb;

    // Compute inactive energy
    Eigen::MatrixXd H_mo_alpha = Ca.transpose() * H_full * Ca;
    Eigen::MatrixXd H_mo_beta = Cb.transpose() * H_full * Cb;

    double E_inactive = 0.0;
    for (auto i : inactive_indices_alpha) {
      E_inactive += H_mo_alpha(i, i) + F_inactive_alpha(i, i);
    }
    for (auto i : inactive_indices_beta) {
      E_inactive += H_mo_beta(i, i) + F_inactive_beta(i, i);
    }
    // Avoid double counting of two-electron interactions
    E_inactive *= 0.5;

    // Extract active space Hamiltonians
    Eigen::MatrixXd H_active_alpha(nactive, nactive);
    Eigen::MatrixXd H_active_beta(nactive, nactive);

    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active_alpha(i, j) =
            F_inactive_alpha(active_indices_alpha[i], active_indices_alpha[j]);
        H_active_beta(i, j) =
            F_inactive_beta(active_indices_beta[i], active_indices_beta[j]);
      }
    }

    if (_settings->get<bool>("store_ao_cholesky_vectors")) {
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::CholeskyHamiltonianContainer>(
              H_active_alpha, H_active_beta, L_mo_alpha, L_mo_beta, orbitals,
              E_inactive + structure->calculate_nuclear_repulsion_energy(),
              F_inactive_alpha, F_inactive_beta, L_ao));
    }
    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::CholeskyHamiltonianContainer>(
            H_active_alpha, H_active_beta, L_mo_alpha, L_mo_beta, orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive_alpha, F_inactive_beta));
  }
}
}  // namespace qdk::chemistry::algorithms::microsoft
