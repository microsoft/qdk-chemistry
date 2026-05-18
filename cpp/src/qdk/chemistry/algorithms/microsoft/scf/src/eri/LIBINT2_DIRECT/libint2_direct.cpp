// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "libint2_direct.h"

#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <qdk/chemistry/utils/logger.hpp>

#include "../schwarz.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <blas.hh>
#include <cassert>
#include <stdexcept>
#include <unordered_set>

#include "util/timer.h"

namespace qdk::chemistry::scf {

namespace libint2::direct {
using qdk::chemistry::scf::RowMajorMatrix;

const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

/**
 * @brief Compute shell block norms of a matrix for screening purposes
 *
 * Computes the infinity norm (maximum absolute value) for each shell block
 * of matrix A. This is used for integral screening to determine which
 * shell quartets can be safely neglected based on density matrix magnitudes.
 *
 * @param obs Libint2 orbital basis set
 * @param A Matrix data in row-major format
 * @param LDA Leading dimension of matrix A
 * @return Matrix of shell block norms for screening
 */
/**
 * @brief Compute shell block norms of a density matrix for screening purposes
 *
 * Computes the infinity norm (maximum absolute value) for each shell block
 * of matrix A. When multiple density matrices are present (UHF: ndm=2),
 * takes the elementwise maximum across all densities so that screening
 * is conservative for all spin channels.
 *
 * @param obs Libint2 orbital basis set
 * @param A Matrix data in row-major format (ndm consecutive NAO x NAO blocks)
 * @param LDA Leading dimension of each matrix block (= NAO)
 * @param ndm Number of density matrices (1 for RHF, 2 for UHF)
 * @return Matrix of shell block norms for screening
 */
RowMajorMatrix compute_shellblock_norm(const ::libint2::BasisSet& obs,
                                       const double* A, size_t LDA,
                                       size_t ndm = 1) {
  QDK_LOG_TRACE_ENTERING();

  auto shell2bf = obs.shell2bf();
  const size_t nsh = obs.size();
  const size_t nbf = obs.nbf();
  RowMajorMatrix shnrms(nsh, nsh);
  shnrms.setZero();

  for (size_t idm = 0; idm < ndm; ++idm) {
    Eigen::Map<const RowMajorMatrix> A_map(A + idm * nbf * LDA, nbf, LDA);
    for (size_t ish = 0; ish < nsh; ++ish)
      for (size_t jsh = 0; jsh < nsh; ++jsh) {
        double block_norm = A_map
                                .block(shell2bf[ish], shell2bf[jsh],
                                       obs[ish].size(), obs[jsh].size())
                                .lpNorm<Eigen::Infinity>();
        shnrms(ish, jsh) = std::max(shnrms(ish, jsh), block_norm);
      }
  }
  return shnrms;
}

/**
 * @brief CSR (Compressed Sparse Row) shell-pair list
 *
 * Cache-friendly replacement for the unordered_map + shared_ptr shell-pair
 * data structures. Stores shell-pair neighbor indices in CSR format and
 * ShellPair geometric data contiguously by value.
 *
 * For each shell s, the significant neighbors are stored at indices
 * [offsets[s], offsets[s+1]) in the neighbors and data arrays. Neighbors
 * within each row are sorted in ascending order, which is required for the
 * `if (s4 > s4_max) break;` early-exit optimization.
 */
struct ShellPairCSR {
  std::vector<size_t> offsets;           ///< size nsh+1; row boundaries
  std::vector<size_t> neighbors;         ///< size = total significant pairs
  std::vector<::libint2::ShellPair> data;///< same length as neighbors; by value

  /// Number of significant pairs for shell s
  size_t row_size(size_t s) const { return offsets[s + 1] - offsets[s]; }
};

/**
 * @brief Precompute significant shell pairs in CSR format
 *
 * Replaces compute_shellpairs() with a cache-friendly CSR layout.
 * ShellPair objects are stored by value (no shared_ptr indirection).
 * Neighbors within each row are sorted ascending (required by the
 * early-exit break in the ket-pair loop).
 */
ShellPairCSR compute_shellpairs_csr(const ::libint2::BasisSet& obs,
                                    double threshold) {
  QDK_LOG_TRACE_ENTERING();

  const auto ln_max_engine_precision = std::log(max_engine_precision);
  const size_t nsh = obs.size();
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif
  std::vector<::libint2::Engine> engines(
      nthreads, ::libint2::Engine(::libint2::Operator::overlap, obs.max_nprim(),
                                  obs.max_l(), 0));
  for (auto& e : engines) e.set_precision(0.);

  // Phase 1: Determine significant pairs per shell (thread-local lists)
  std::vector<std::vector<size_t>> per_shell_neighbors(nsh);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    const int thread_id = omp_get_thread_num();
#else
    const int thread_id = 0;
#endif
    auto& engine = engines[thread_id];
    const auto& buf = engine.results();
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (size_t s1 = 0; s1 < nsh; ++s1) {
      const auto n1 = obs[s1].size();
      for (size_t s2 = 0; s2 <= s1; ++s2) {
        const auto n2 = obs[s2].size();
        bool on_same_center = (obs[s1].O == obs[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          engine.compute(obs[s1], obs[s2]);
          Eigen::Map<const RowMajorMatrix> buf_map(buf[0], n1, n2);
          const auto norm = buf_map.norm();
          significant = (norm >= threshold);
        }
        if (significant) per_shell_neighbors[s1].emplace_back(s2);
      }
    }
  }

  // Phase 2: Build CSR offsets
  ShellPairCSR csr;
  csr.offsets.resize(nsh + 1);
  csr.offsets[0] = 0;
  for (size_t s = 0; s < nsh; ++s) {
    csr.offsets[s + 1] = csr.offsets[s] + per_shell_neighbors[s].size();
  }
  const size_t total_pairs = csr.offsets[nsh];

  // Phase 3: Fill neighbors array (preserving ascending order per row)
  csr.neighbors.resize(total_pairs);
  for (size_t s = 0; s < nsh; ++s) {
    assert(std::is_sorted(per_shell_neighbors[s].begin(),
                          per_shell_neighbors[s].end()));
    std::copy(per_shell_neighbors[s].begin(), per_shell_neighbors[s].end(),
              csr.neighbors.begin() + csr.offsets[s]);
  }

  // Phase 4: Construct ShellPair data by value (parallel, indexed by position)
  csr.data.resize(total_pairs);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t s1 = 0; s1 < nsh; ++s1) {
    for (size_t idx = csr.offsets[s1]; idx < csr.offsets[s1 + 1]; ++idx) {
      const size_t s2 = csr.neighbors[idx];
      csr.data[idx] = ::libint2::ShellPair(obs[s1], obs[s2],
                                           ln_max_engine_precision,
                                           ::libint2::ScreeningMethod::Original);
    }
  }

  return csr;
}

/**
 * @brief ERI class for direct computation of electron repulsion integrals using
 * Libint2
 *
 * This class implements efficient direct evaluation of two-electron repulsion
 * integrals using the Libint2 library. It employs Schwarz screening for
 * computational efficiency and supports both restricted and unrestricted
 * Hartree-Fock and DFT calculations.
 *
 * The implementation uses shell-pair pre-screening to avoid computation of
 * negligible integrals, significantly reducing computational cost for large
 * basis sets. All integrals are computed on-the-fly without storage, making it
 * suitable for large-scale calculations with limited memory.
 *
 * @note This class requires Libint2 library for integral evaluation
 */
class ERI {
  size_t spin_density_factor_;     ///< Number of spin density matrices (1 or 2)
  bool use_thread_local_buffers_;  ///< Use thread-local buffers (true) or
                                   ///< atomic ops (false)
  double eri_threshold_;           ///< Integral screening threshold
  ::libint2::BasisSet obs_;        ///< Libint2 orbital basis set representation
  std::vector<size_t>
      shell2bf_;  ///< Mapping from shell index to first atomic orbital
  ShellPairCSR sp_csr_;            ///< CSR shell-pair list + data (by value)
  RowMajorMatrix K_schwarz_;       ///< Schwarz screening matrix for integral bounds

  // Persistent per-thread Libint2 engine pool (constructed once, reused across calls)
  std::vector<::libint2::Engine> engines_; ///< One engine per thread
  int nthreads_;                           ///< Number of threads engines_ is sized for

  // Per-shell POD arrays for fast inner-loop access (avoid obs_[s].size() calls)
  std::vector<size_t> shell_nbf_;       ///< shell_nbf_[s] = number of basis functions in shell s
  std::vector<size_t> shell_bf_offset_; ///< shell_bf_offset_[s] = first basis function index
  size_t max_shellblock_bf_;            ///< max over all s of shell_nbf_[s]

  // Persistent per-thread TLS accumulation buffers.
  // Allocated once with first-touch in a parallel region so each thread's
  // buffer lands on its own NUMA node. Reused across build_JK calls to
  // avoid repeated heap allocation of nthreads × N² doubles.
  std::vector<std::vector<double>> J_tls_; ///< Per-thread J accumulation
  std::vector<std::vector<double>> K_tls_; ///< Per-thread K accumulation
  size_t tls_mat_size_ = 0;               ///< Current allocation size per buffer

  // Per-thread touched shell-pair tracking for sparse reduction + zeroing.
  // Each entry is an nsh×nsh bitmap (uint8_t). On reduction, only touched
  // shell-pair blocks are summed — O(touched × shell_size²) instead of O(N²).
  // On zeroing, only previously-touched blocks are cleared.
  std::vector<std::vector<uint8_t>> touched_J_; ///< Per-thread J touched bitmap
  std::vector<std::vector<uint8_t>> touched_K_; ///< Per-thread K touched bitmap
  bool first_build_jk_ = true; ///< True until first build_JK completes (skip sparse zeroing)

 public:
  /**
   * @brief Construct Libint2 direct ERI engine
   *
   * Initializes the direct integral evaluation engine with the given basis set.
   * Precomputes shell pair lists and Schwarz bounds for efficient screening
   * during integral evaluation. All screening data is computed once during
   * construction and reused throughout the calculation.
   *
   * @param spin_density_factor Number of spin density matrices (1 or 2)
   * @param basis_set QDK basis set (converted to Libint2 format internally)
   * @param use_atomics Use atomic operations (true) or thread-local buffers
   *        (false)
   * @param eri_threshold ERI screening threshold for skipping negligible
   *        shell quartets during J/K builds and quarter transformations
   * @param shell_pair_threshold Overlap-based shell pair pre-screening
   *        threshold
   *
   * @note Construction involves significant overhead due to screening setup
   * @note Shell pair and Schwarz data is computed using OpenMP parallelization
   */
  ERI(size_t spin_density_factor, qdk::chemistry::scf::BasisSet& basis_set,
      bool use_atomics, double eri_threshold, double shell_pair_threshold)
      : spin_density_factor_(spin_density_factor),
        use_thread_local_buffers_(!use_atomics),
        eri_threshold_(eri_threshold),
        obs_(libint2_util::convert_to_libint_basisset(basis_set)) {
    QDK_LOG_TRACE_ENTERING();

    shell2bf_ = obs_.shell2bf();

    // Build per-shell POD arrays
    const size_t nsh = obs_.size();
    shell_nbf_.resize(nsh);
    shell_bf_offset_.resize(nsh);
    max_shellblock_bf_ = 0;
    for (size_t s = 0; s < nsh; ++s) {
      shell_nbf_[s] = obs_[s].size();
      shell_bf_offset_[s] = shell2bf_[s];
      max_shellblock_bf_ = std::max(max_shellblock_bf_, shell_nbf_[s]);
    }

    // Compute Shell Pairs in CSR format
    sp_csr_ = compute_shellpairs_csr(obs_, shell_pair_threshold);

    // Compute Schwarz Screening
    K_schwarz_ = RowMajorMatrix(obs_.size(), obs_.size());
    auto mpi = mpi_default_input();
    schwarz_integral(&basis_set, mpi, K_schwarz_.data(), true);

    // Construct persistent per-thread engine pool
#ifdef _OPENMP
    nthreads_ = omp_get_max_threads();
#else
    nthreads_ = 1;
#endif
    engines_.resize(nthreads_);
    engines_[0] = ::libint2::Engine(::libint2::Operator::coulomb,
                                    obs_.max_nprim(), obs_.max_l(), 0);
    engines_[0].set(::libint2::ScreeningMethod::Original);
    for (int i = 1; i < nthreads_; ++i) engines_[i] = engines_[0];

    // Pre-allocate per-thread TLS buffers with first-touch NUMA placement.
    // Each thread allocates and touches its own buffer inside a parallel
    // region so the OS maps the pages to the thread's NUMA node.
    const size_t nbf = obs_.nbf();
    tls_mat_size_ = spin_density_factor_ * nbf * nbf;
    J_tls_.resize(nthreads_);
    K_tls_.resize(nthreads_);
    touched_J_.resize(nthreads_);
    touched_K_.resize(nthreads_);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      J_tls_[tid].resize(tls_mat_size_);
      K_tls_[tid].resize(tls_mat_size_);
      std::memset(J_tls_[tid].data(), 0, tls_mat_size_ * sizeof(double));
      std::memset(K_tls_[tid].data(), 0, tls_mat_size_ * sizeof(double));
      touched_J_[tid].assign(nsh * nsh, 0);
      touched_K_[tid].assign(nsh * nsh, 0);
    }
  }

  /**
   * @brief Build Coulomb (J) and exchange (K) matrices using direct integral
   * evaluation
   *
   * Computes J and K matrices by direct evaluation of two-electron repulsion
   * integrals using the Libint2 library. Employs multiple levels of screening:
   * 1. Shell pair pre-screening based on overlap
   * 2. Schwarz inequality bounds: (μν|λσ) ≤ K(μν) × K(λσ)
   * 3. Density-based screening using shell block norms
   * 4. Libint2 internal screening based on primitive overlap
   *
   * The implementation uses OpenMP parallelization and atomic operations for
   * thread-safe accumulation of matrix elements.
   *
   * @param P Density matrix in AO basis (packed for unrestricted: [Pα, Pβ])
   * @param J Output Coulomb matrix (nullptr if not needed)
   * @param K Output exchange matrix (nullptr if not needed)
   * @param alpha Scaling factor for Hartree-Fock exchange
   * @param beta Scaling factor for DFT exchange
   * @param omega Range-separation parameter (not supported, must be ~0)
   *
   * @throws std::runtime_error If range-separated hybrid is requested
   *
   * @note Both J and K matrices are symmetrized at the end
   * @note K matrix is scaled by (alpha + beta)
   * @note Supports both restricted (num_density_matrices=1) and unrestricted
   * (num_density_matrices=2) calculations
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega) {
    QDK_LOG_TRACE_ENTERING();

    AutoTimer t("ERI::build_JK");
    const size_t N = obs_.nbf();
    const size_t nsh = obs_.size();
    const size_t ndm = spin_density_factor_;
    const size_t mat_size = ndm * N * N;
    const bool is_rsx = std::abs(omega) > 1e-12;

    if (is_rsx) throw std::runtime_error("RSX + LIBINT2_DIRECT NYI");

    // Compute shell block norm of P (max over all density matrices for UHF)
    const auto P_shnrm =
        compute_shellblock_norm(obs_, P, N, ndm);
    const auto P_shmax = P_shnrm.maxCoeff();

    if (!std::isfinite(P_shmax)) {
      throw std::runtime_error("Density matrix contains NaN/Inf values.");
    }

    double engine_precision;
    if (P_shmax <= 0.0) {
      engine_precision = std::numeric_limits<double>::epsilon();
    } else {
      engine_precision = std::numeric_limits<double>::epsilon() / P_shmax;
    }
    engine_precision = std::max(engine_precision, max_engine_precision);

    for (int i = 0; i < nthreads_; ++i)
      engines_[i].set_precision(engine_precision);

    if (J) std::memset(J, 0, mat_size * sizeof(double));
    if (K) std::memset(K, 0, mat_size * sizeof(double));

    // ---------------------------------------------------------------
    // Phase 3: Bra-pair-chunked dynamic scheduling
    // ---------------------------------------------------------------
    // Build flat canonical bra-pair list with estimated cost for load balancing
    struct BraPair {
      size_t s1, s2;
      size_t sp_idx;
      size_t est_ket_pairs;  // estimated number of ket-pairs for this bra-pair
    };
    std::vector<BraPair> bra_pairs;
    bra_pairs.reserve(sp_csr_.neighbors.size());
    for (size_t s1 = 0; s1 < nsh; ++s1) {
      // Estimate ket-pair count: sum of ket-pairs for all s3 <= s1
      size_t est_kets = 0;
      for (size_t s3 = 0; s3 <= s1; ++s3) {
        est_kets += sp_csr_.row_size(s3);
      }
      for (size_t idx = sp_csr_.offsets[s1]; idx < sp_csr_.offsets[s1 + 1]; ++idx) {
        bra_pairs.push_back({s1, sp_csr_.neighbors[idx], idx, est_kets});
      }
    }
    const size_t n_bra_pairs = bra_pairs.size();

    // Cost-balanced reordering: sort by descending cost, then interleave
    // across thread bins so each thread gets a mix of expensive and cheap
    // bra-pairs. Final order is deterministic (pure function of basis).
    {
      // Sort by descending estimated cost
      std::vector<size_t> order(n_bra_pairs);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(),
                       [&](size_t a, size_t b) {
                         return bra_pairs[a].est_ket_pairs >
                                bra_pairs[b].est_ket_pairs;
                       });

      // Interleave into thread bins: bp_ranked[0] -> thread 0,
      // bp_ranked[1] -> thread 1, ..., bp_ranked[nthreads] -> thread 0, etc.
      std::vector<std::vector<size_t>> bins(nthreads_);
      for (size_t r = 0; r < n_bra_pairs; ++r) {
        bins[r % nthreads_].push_back(order[r]);
      }

      // Rebuild bra_pairs in bin order (thread 0's bra-pairs first, then
      // thread 1's, etc.) so schedule(static) assigns them correctly.
      std::vector<BraPair> reordered;
      reordered.reserve(n_bra_pairs);
      for (int t = 0; t < nthreads_; ++t) {
        for (size_t idx : bins[t]) {
          reordered.push_back(bra_pairs[idx]);
        }
      }
      bra_pairs = std::move(reordered);
    }

    // Accumulators for per-thread sparse work cost (computed in parallel)
    size_t j_sparse_total = 0;
    size_t k_sparse_total = 0;

    // Single parallel region: zero TLS buffers + compute + cost model
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      // Sparse zeroing: only clear shell-pair blocks touched in the previous
      // call, then clear the touched bitmap. Much cheaper than full N² memset
      // when each thread only touches a subset of shell pairs.
      // On the very first build_JK call, TLS buffers are already zeroed by the
      // constructor, so we skip the scan entirely.
      auto& tj = touched_J_[tid];
      auto& tk = touched_K_[tid];
      if (!first_build_jk_) {
        if (J) {
          double* jd = J_tls_[tid].data();
          for (size_t sa = 0; sa < nsh; ++sa)
            for (size_t sb = 0; sb < nsh; ++sb)
              if (tj[sa * nsh + sb]) {
                const size_t na = shell_nbf_[sa], nb = shell_nbf_[sb];
                for (size_t idm = 0; idm < ndm; ++idm) {
                  const size_t off = idm * N * N + shell_bf_offset_[sa] * N + shell_bf_offset_[sb];
                  for (size_t a = 0; a < na; ++a)
                    std::memset(jd + off + a * N, 0, nb * sizeof(double));
                }
                tj[sa * nsh + sb] = 0;
              }
        }
        if (K) {
          double* kd = K_tls_[tid].data();
          for (size_t sa = 0; sa < nsh; ++sa)
            for (size_t sb = 0; sb < nsh; ++sb)
              if (tk[sa * nsh + sb]) {
                const size_t na = shell_nbf_[sa], nb = shell_nbf_[sb];
                for (size_t idm = 0; idm < ndm; ++idm) {
                  const size_t off = idm * N * N + shell_bf_offset_[sa] * N + shell_bf_offset_[sb];
                  for (size_t a = 0; a < na; ++a)
                    std::memset(kd + off + a * N, 0, nb * sizeof(double));
                }
                tk[sa * nsh + sb] = 0;
              }
        }
      }

#ifdef _OPENMP
#pragma omp barrier
#endif

      auto& engine = engines_[tid];
      const auto& buf = engine.results();
      double* J_thr = J ? J_tls_[tid].data() : nullptr;
      double* K_thr = K ? K_tls_[tid].data() : nullptr;

      // Dynamic scheduling over bra-pairs — each thread processes
      // contiguous bra-pair ranges, improving cache locality for bra-side
      // density rows and Schwarz data vs the old round-robin.
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (size_t bp_idx = 0; bp_idx < n_bra_pairs; ++bp_idx) {
        const auto& bp = bra_pairs[bp_idx];
        const size_t s1 = bp.s1;
        const size_t s2 = bp.s2;
        const auto bf1_st = shell_bf_offset_[s1];
        const auto bf2_st = shell_bf_offset_[s2];
        const auto n1 = shell_nbf_[s1];
        const auto n2 = shell_nbf_[s2];
        const auto* sp12_data = &sp_csr_.data[bp.sp_idx];
        const auto P12_nrm = P_shnrm(s1, s2);

        for (size_t s3 = 0; s3 <= s1; ++s3) {
          const auto bf3_st = shell_bf_offset_[s3];
          const auto n3 = shell_nbf_[s3];
          const auto P13_nrm = P_shnrm(s1, s3);
          const auto P23_nrm = P_shnrm(s2, s3);

          const size_t sp34_begin = sp_csr_.offsets[s3];
          const size_t sp34_end = sp_csr_.offsets[s3 + 1];
          const size_t s4_max = (s1 == s3) ? s2 : s3;

          for (size_t sp34_idx = sp34_begin; sp34_idx < sp34_end; ++sp34_idx) {
            const size_t s4 = sp_csr_.neighbors[sp34_idx];
            if (s4 > s4_max) break;

            const auto* sp34_data = &sp_csr_.data[sp34_idx];

            // Density-aware Schwarz screening
            const auto schwarz_bound = K_schwarz_(s1, s2) * K_schwarz_(s3, s4);
            const auto P14_nrm = P_shnrm(s1, s4);
            const auto P24_nrm = P_shnrm(s2, s4);
            const auto P34_nrm = P_shnrm(s3, s4);

            double P_screen;
            if (J && K) {
              P_screen = std::max(std::max(P12_nrm, P34_nrm),
                                  std::max({P13_nrm, P14_nrm, P23_nrm, P24_nrm}));
            } else if (J) {
              P_screen = std::max(P12_nrm, P34_nrm);
            } else {
              P_screen = std::max({P13_nrm, P14_nrm, P23_nrm, P24_nrm});
            }
            if (P_screen * schwarz_bound < eri_threshold_) continue;

            const auto bf4_st = shell_bf_offset_[s4];
            const auto n4 = shell_nbf_[s4];

            // Mark touched shell pairs for sparse reduction/zeroing
            if (J_thr) {
              tj[s1 * nsh + s2] = 1; tj[s3 * nsh + s4] = 1;
            }
            if (K_thr) {
              tk[s1 * nsh + s3] = 1; tk[s1 * nsh + s4] = 1;
              tk[s2 * nsh + s3] = 1; tk[s2 * nsh + s4] = 1;
            }

            auto s12_deg = (s1 == s2) ? 1 : 2;
            auto s34_deg = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

            engine.compute2<::libint2::Operator::coulomb,
                            ::libint2::BraKet::xx_xx, 0>(
                obs_[s1], obs_[s2], obs_[s3], obs_[s4], sp12_data, sp34_data);

            const auto buf_1234 = buf[0];
            if (buf_1234 == nullptr) continue;

            // Stack-tile contraction: accumulate shell-block contributions
            // in small L1-hot tiles, then flush once to the N² TLS buffer.
            // This reduces N-strided cache misses from O(n1*n2*n3*n4) to
            // O(tile_size) per shell quartet — critical for large N where
            // the TLS buffer doesn't fit in L2.
            // Skip tiling for small quartets (s-shells) where the setup
            // overhead exceeds the cache benefit.
            constexpr size_t MAX_BF = 15;  // supports up to Cartesian g-shells
            constexpr size_t TILE_THRESHOLD = 4;  // min n3*n4 to justify tiling
            const bool use_tiles = (n1 <= MAX_BF && n2 <= MAX_BF &&
                                    n3 <= MAX_BF && n4 <= MAX_BF &&
                                    n3 * n4 >= TILE_THRESHOLD);

            // Contract J
            if (J_thr)
              for (size_t idm = 0; idm < ndm; ++idm) {
                auto* J_cur = J_thr + idm * N * N;
                const auto* P_cur = P + idm * N * N;
                if (use_tiles) {
                  // Pre-load P(s3,s4) tile
                  double P_34[MAX_BF * MAX_BF];
                  for (size_t k = 0; k < n3; ++k)
                    for (size_t l = 0; l < n4; ++l)
                      P_34[k * n4 + l] = P_cur[(bf3_st + k) * N + (bf4_st + l)];

                  // J(s3,s4) accumulator tile
                  double J_34[MAX_BF * MAX_BF];
                  std::memset(J_34, 0, n3 * n4 * sizeof(double));

                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    for (size_t j = 0; j < n2; ++j) {
                      double J_ij = 0.0;
                      const double P_ij = P_cur[(bf1_st + i) * N + (bf2_st + j)];
                      for (size_t k = 0; k < n3; ++k)
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const auto value = buf_1234[ijkl] * s1234_deg;
                          J_ij += P_34[k * n4 + l] * value;
                          J_34[k * n4 + l] += P_ij * value;
                        }
                      J_cur[(bf1_st + i) * N + (bf2_st + j)] += J_ij;
                    }
                  }
                  // Flush J(s3,s4) tile
                  for (size_t k = 0; k < n3; ++k)
                    for (size_t l = 0; l < n4; ++l)
                      J_cur[(bf3_st + k) * N + (bf4_st + l)] += J_34[k * n4 + l];
                } else {
                  // Direct contraction fallback for oversized shells
                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    const size_t bf1 = bf1_st + i;
                    for (size_t j = 0; j < n2; ++j) {
                      const size_t bf2 = bf2_st + j;
                      double J_ij = 0.0;
                      const double P_ij = P_cur[bf1 * N + bf2];
                      for (size_t k = 0; k < n3; ++k) {
                        const size_t bf3 = bf3_st + k;
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const auto value = buf_1234[ijkl] * s1234_deg;
                          J_ij += P_cur[bf3 * N + (bf4_st + l)] * value;
                          J_cur[bf3 * N + (bf4_st + l)] += P_ij * value;
                        }
                      }
                      J_cur[bf1 * N + bf2] += J_ij;
                    }
                  }
                }
              }

            // Contract K
            if (K_thr)
              for (size_t idm = 0; idm < ndm; ++idm) {
                auto* K_cur = K_thr + idm * N * N;
                const auto* P_cur = P + idm * N * N;
                if (use_tiles) {
                  // Pre-load density tiles (with 0.25 pre-multiplied)
                  double P_13[MAX_BF * MAX_BF], P_23[MAX_BF * MAX_BF];
                  double P_14[MAX_BF * MAX_BF], P_24[MAX_BF * MAX_BF];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t k = 0; k < n3; ++k)
                      P_13[i * n3 + k] = 0.25 * P_cur[(bf1_st + i) * N + (bf3_st + k)];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t k = 0; k < n3; ++k)
                      P_23[j * n3 + k] = 0.25 * P_cur[(bf2_st + j) * N + (bf3_st + k)];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t l = 0; l < n4; ++l)
                      P_14[i * n4 + l] = 0.25 * P_cur[(bf1_st + i) * N + (bf4_st + l)];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t l = 0; l < n4; ++l)
                      P_24[j * n4 + l] = 0.25 * P_cur[(bf2_st + j) * N + (bf4_st + l)];

                  // K accumulator tiles
                  double K_13[MAX_BF * MAX_BF], K_23[MAX_BF * MAX_BF];
                  double K_14[MAX_BF * MAX_BF], K_24[MAX_BF * MAX_BF];
                  std::memset(K_13, 0, n1 * n3 * sizeof(double));
                  std::memset(K_23, 0, n2 * n3 * sizeof(double));
                  std::memset(K_14, 0, n1 * n4 * sizeof(double));
                  std::memset(K_24, 0, n2 * n4 * sizeof(double));

                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    for (size_t j = 0; j < n2; ++j) {
                      for (size_t k = 0; k < n3; ++k) {
                        double K_ik = 0.0, K_jk = 0.0;
                        const double P_ik = P_13[i * n3 + k];
                        const double P_jk = P_23[j * n3 + k];
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const auto value = buf_1234[ijkl] * s1234_deg;
                          K_ik += P_24[j * n4 + l] * value;
                          K_jk += P_14[i * n4 + l] * value;
                          K_14[i * n4 + l] += P_jk * value;
                          K_24[j * n4 + l] += P_ik * value;
                        }
                        K_13[i * n3 + k] += K_ik;
                        K_23[j * n3 + k] += K_jk;
                      }
                    }
                  }
                  // Flush K tiles
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t k = 0; k < n3; ++k)
                      K_cur[(bf1_st + i) * N + (bf3_st + k)] += K_13[i * n3 + k];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t k = 0; k < n3; ++k)
                      K_cur[(bf2_st + j) * N + (bf3_st + k)] += K_23[j * n3 + k];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t l = 0; l < n4; ++l)
                      K_cur[(bf1_st + i) * N + (bf4_st + l)] += K_14[i * n4 + l];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t l = 0; l < n4; ++l)
                      K_cur[(bf2_st + j) * N + (bf4_st + l)] += K_24[j * n4 + l];
                } else {
                  // Direct contraction fallback for oversized shells
                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    const size_t bf1 = bf1_st + i;
                    for (size_t j = 0; j < n2; ++j) {
                      const size_t bf2 = bf2_st + j;
                      for (size_t k = 0; k < n3; ++k) {
                        const size_t bf3 = bf3_st + k;
                        double K_ik = 0.0, K_jk = 0.0;
                        const double P_ik = 0.25 * P_cur[bf1 * N + bf3];
                        const double P_jk = 0.25 * P_cur[bf2 * N + bf3];
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const size_t bf4 = bf4_st + l;
                          const auto value = buf_1234[ijkl] * s1234_deg;
                          K_ik += 0.25 * P_cur[bf2 * N + bf4] * value;
                          K_jk += 0.25 * P_cur[bf1 * N + bf4] * value;
                          K_cur[bf1 * N + bf4] += P_jk * value;
                          K_cur[bf2 * N + bf4] += P_ik * value;
                        }
                        K_cur[bf1 * N + bf3] += K_ik;
                        K_cur[bf2 * N + bf3] += K_jk;
                      }
                    }
                  }
                }
              }

          }  // s4
        }  // s3
      }  // bra-pair (omp for)

      // Parallel cost model: each thread counts its own touched AO pairs.
      // Runs inside the compute region to avoid serial overhead and cache
      // eviction between the compute and reduction phases.
      {
        size_t my_j_sparse = 0, my_k_sparse = 0;
        if (J) {
          const auto& tj_local = touched_J_[tid];
          for (size_t sa = 0; sa < nsh; ++sa)
            for (size_t sb = 0; sb < nsh; ++sb)
              if (tj_local[sa * nsh + sb])
                my_j_sparse += shell_nbf_[sa] * shell_nbf_[sb];
        }
        if (K) {
          const auto& tk_local = touched_K_[tid];
          for (size_t sa = 0; sa < nsh; ++sa)
            for (size_t sb = 0; sb < nsh; ++sb)
              if (tk_local[sa * nsh + sb])
                my_k_sparse += shell_nbf_[sa] * shell_nbf_[sb];
        }
#ifdef _OPENMP
#pragma omp atomic
#endif
        j_sparse_total += my_j_sparse * ndm;
#ifdef _OPENMP
#pragma omp atomic
#endif
        k_sparse_total += my_k_sparse * ndm;
      }
    }  // parallel region

    // Hybrid deterministic reduction: choose dense or sparse based on the
    // AO-element-weighted work cost computed inside the parallel region.
    // Dense reduction uses a tight vectorized loop (unit-stride, AVX-friendly).
    // Sparse reduction skips untouched shell-pair blocks using the bitmap.
    //
    // The sparse inner loop has non-unit-stride access (rows of a shell block
    // are N apart in the row-major N² buffer), making it ~3-5x slower per
    // element than vectorized dense. However, sparse reads less total data,
    // which matters at high thread counts where memory bandwidth is limiting.
    // The 50% threshold balances these: switch to dense only when sparse
    // would process more than half the data that dense would process.
    constexpr size_t dense_threshold_num = 1;
    constexpr size_t dense_threshold_den = 2;
    const size_t dense_work = static_cast<size_t>(nthreads_) * ndm * N * N;

    if (J) {
      const bool use_dense_j =
          j_sparse_total * dense_threshold_den >
          dense_work * dense_threshold_num;
      if (use_dense_j) {
        // Dense reduction: parallel over rows, unit-stride vectorized add
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t row = 0; row < ndm * N; ++row) {
          double* dst = J + row * N;
          for (int t = 0; t < nthreads_; ++t) {
            const double* src = J_tls_[t].data() + row * N;
#pragma omp simd
            for (size_t col = 0; col < N; ++col)
              dst[col] += src[col];
          }
        }
      } else {
        // Sparse reduction: skip untouched shell-pair blocks
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
        for (size_t sa = 0; sa < nsh; ++sa)
          for (size_t sb = 0; sb < nsh; ++sb) {
            const size_t na = shell_nbf_[sa], nb = shell_nbf_[sb];
            for (int t = 0; t < nthreads_; ++t) {
              if (!touched_J_[t][sa * nsh + sb]) continue;
              const double* src = J_tls_[t].data();
              for (size_t idm = 0; idm < ndm; ++idm) {
                const size_t off = idm * N * N + shell_bf_offset_[sa] * N + shell_bf_offset_[sb];
                for (size_t a = 0; a < na; ++a)
                  for (size_t b = 0; b < nb; ++b)
                    J[off + a * N + b] += src[off + a * N + b];
              }
            }
          }
      }
    }
    if (K) {
      const bool use_dense_k =
          k_sparse_total * dense_threshold_den >
          dense_work * dense_threshold_num;
      if (use_dense_k) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t row = 0; row < ndm * N; ++row) {
          double* dst = K + row * N;
          for (int t = 0; t < nthreads_; ++t) {
            const double* src = K_tls_[t].data() + row * N;
#pragma omp simd
            for (size_t col = 0; col < N; ++col)
              dst[col] += src[col];
          }
        }
      } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
        for (size_t sa = 0; sa < nsh; ++sa)
          for (size_t sb = 0; sb < nsh; ++sb) {
            const size_t na = shell_nbf_[sa], nb = shell_nbf_[sb];
            for (int t = 0; t < nthreads_; ++t) {
              if (!touched_K_[t][sa * nsh + sb]) continue;
              const double* src = K_tls_[t].data();
              for (size_t idm = 0; idm < ndm; ++idm) {
                const size_t off = idm * N * N + shell_bf_offset_[sa] * N + shell_bf_offset_[sb];
                for (size_t a = 0; a < na; ++a)
                  for (size_t b = 0; b < nb; ++b)
                    K[off + a * N + b] += src[off + a * N + b];
              }
            }
          }
      }
    }

    first_build_jk_ = false;

    // Symmetrize J (parallel)
    if (J)
      for (size_t idm = 0; idm < ndm; ++idm) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j <= i; ++j) {
            auto v = J[idm * N * N + i * N + j];
            auto w = J[idm * N * N + j * N + i];
            v = 0.25 * (v + w);
            J[idm * N * N + i * N + j] = v;
            J[idm * N * N + j * N + i] = v;
          }
      }

    // Symmetrize K + scale (parallel)
    if (K)
      for (size_t idm = 0; idm < ndm; ++idm) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j <= i; ++j) {
            auto v = K[idm * N * N + i * N + j];
            auto w = K[idm * N * N + j * N + i];
            v = (alpha + beta) * 0.5 * (v + w);
            K[idm * N * N + i * N + j] = v;
            K[idm * N * N + j * N + i] = v;
          }
      }
  }

  /**
   * @brief Compute nuclear gradients for Coulomb and exchange energies
   *
   * Computes the contribution to nuclear gradients from the Coulomb and
   * exchange energies using direct evaluation of derivative integrals. This
   * involves computing nuclear derivatives of two-electron integrals and
   * contracting with density matrices to obtain energy gradient contributions:
   * ∂E_J/∂R and ∂E_K/∂R.
   *
   * @param P Density matrix in AO basis
   * @param dJ Output nuclear gradient contribution from Coulomb energy
   * @param dK Output nuclear gradient contribution from exchange energy
   * @param alpha Scaling factor for Hartree-Fock exchange
   * @param beta Scaling factor for DFT exchange
   * @param omega Range-separation parameter
   *
   * @throws std::runtime_error Always - energy gradients not yet implemented
   *
   * @note These are energy derivatives, not matrix element derivatives
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) {
    QDK_LOG_TRACE_ENTERING();

    throw std::runtime_error("LIBINT2_DIRECT + Gradients Not Yet Implemented");
  }

  /**
   * @brief Perform quarter transformation of two-electron integrals
   *
   * Transforms the first index of the four-center integrals from AO to MO
   * basis: (pj|kl) = Σᵢ C(i,p) × (ij|kl)
   *
   * This is the first step in integral transformations for post-HF methods.
   * The implementation uses the same screening as build_JK but performs the
   * transformation during integral evaluation to minimize memory usage.
   *
   * @param nt Number of transformed orbitals (size of MO space)
   * @param C Transformation matrix C(AO,MO) in row-major format
   * @param out Output buffer for transformed integrals (pj|kl)
   *           Layout: out[p + nt*(l + num_atomic_orbitals*(j +
   * num_atomic_orbitals*k))] for (kj|lp)
   *
   * @note First index (i) is transformed to MO index (p): i → p
   * @note Output tensor has mixed AO/MO indices with p as fastest index
   * @note Uses same shell pair and Schwarz screening as build_JK
   * @note Result includes proper symmetrization for shell permutations
   */
  void quarter_trans(size_t nt, const double* C, double* out) {
    QDK_LOG_TRACE_ENTERING();

    AutoTimer t("ERI::quarter_trans");
    const size_t num_atomic_orbitals = obs_.nbf();
    const size_t nsh = obs_.size();

    // Clear output tensor: (ij|kp) with p as fast index
    const size_t out_size =
        nt * num_atomic_orbitals * num_atomic_orbitals * num_atomic_orbitals;
    std::memset(out, 0, out_size * sizeof(double));
    const size_t inner_size = nt * num_atomic_orbitals;

    // No shell block norm needed for quarter transformation
    // We'll use Schwarz screening directly

    // Setup required precision for libint2 engine
    const auto engine_precision = std::numeric_limits<double>::epsilon();

    // Update engine precision for quarter transformation
    for (int i = 0; i < nthreads_; ++i)
      engines_[i].set_precision(engine_precision);

    // Thread-local accumulation buffers for reproducibility
    std::vector<std::vector<double>> out_local(0);
    if (use_thread_local_buffers_) {
      out_local.resize(nthreads_);
      for (int t = 0; t < nthreads_; ++t) {
        out_local[t].resize(out_size, 0.0);
      }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const auto thread_id = omp_get_thread_num();
#else
      const auto thread_id = 0;
#endif
      auto& engine = engines_[thread_id];
      const auto& buf = engine.results();

      // Get pointer to thread-local buffer
      double* out_thread = nullptr;
      if (use_thread_local_buffers_) {
        out_thread = out_local[thread_id].data();
      }

      for (size_t s1 = 0ul, s1234 = 0ul; s1 < nsh; ++s1) {
        const auto bf1_st = shell_bf_offset_[s1];
        const auto n1 = shell_nbf_[s1];

        const size_t sp12_begin = sp_csr_.offsets[s1];
        const size_t sp12_end = sp_csr_.offsets[s1 + 1];

        // Only loop over significant shell pairs (CSR row for s1)
        for (size_t sp12_idx = sp12_begin; sp12_idx < sp12_end; ++sp12_idx) {
          const size_t s2 = sp_csr_.neighbors[sp12_idx];
          const auto bf2_st = shell_bf_offset_[s2];
          const auto n2 = shell_nbf_[s2];
          const auto* sp12_data = &sp_csr_.data[sp12_idx];

          // Permutational Degeneracy
          auto s12_deg = (s1 == s2) ? 1 : 2;

          for (size_t s3 = 0; s3 <= s1; ++s3) {
            const auto bf3_st = shell_bf_offset_[s3];
            const auto n3 = shell_nbf_[s3];

            const size_t sp34_begin = sp_csr_.offsets[s3];
            const size_t sp34_end = sp_csr_.offsets[s3 + 1];
            const size_t s4_max = (s1 == s3) ? s2 : s3;

            // Only loop over significant shell pairs (CSR row for s3)
            for (size_t sp34_idx = sp34_begin; sp34_idx < sp34_end; ++sp34_idx) {
              const size_t s4 = sp_csr_.neighbors[sp34_idx];
              if (s4 > s4_max) break;

              const auto* sp34_data = &sp_csr_.data[sp34_idx];

              // Assign to threads
              if ((s1234++) % nthreads_ != thread_id) continue;
              if (K_schwarz_(s1, s2) * K_schwarz_(s3, s4) < eri_threshold_)
                continue;

              const auto bf4_st = shell_bf_offset_[s4];
              const auto n4 = shell_nbf_[s4];

              // Compute the integral shell quartet
              engine.compute2<::libint2::Operator::coulomb,
                              ::libint2::BraKet::xx_xx, 0>(
                  obs_[s1], obs_[s2], obs_[s3], obs_[s4], sp12_data, sp34_data);

              // Coarse integral screening
              const auto buf_1234 = buf[0];
              if (buf_1234 == nullptr) continue;

              auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
              auto s34_deg = (s3 == s4) ? 1 : 2;

              // Perform quarter transformation: (pn|lk) = sum_m C(p,m) *
              // (mn|lk) Here: m=s1i, n=s2j, l=s3k, k=s4l Output layout: p is
              // fast index, so out[p + nt*(k + num_atomic_orbitals*(j +
              // num_atomic_orbitals*i))]
              for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                const size_t bf1 = bf1_st + i;
                for (size_t j = 0; j < n2; ++j) {
                  const size_t bf2 = bf2_st + j;
                  for (size_t k = 0; k < n3; ++k) {
                    const size_t bf3 = bf3_st + k;
                    for (size_t l = 0; l < n4; ++l, ++ijkl) {
                      const size_t bf4 = bf4_st + l;

                      const auto value = buf_1234[ijkl] * s12_34_deg;
                      // p and l are fast indices separately
                      for (size_t p = 0; p < nt; ++p) {
                        // (ij|kp) = \sum_l (ij|kl) * C(l,p)
                        const size_t idx1 =
                            (bf1 * num_atomic_orbitals + bf2) * inner_size +
                            bf3 * nt + p;
                        if (use_thread_local_buffers_) {
                          out_thread[idx1] += C[bf4 * nt + p] * value * s12_deg;
                        } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                          out[idx1] += C[bf4 * nt + p] * value * s12_deg;
                        }

                        // (ij|lp) = \sum_k (ij|lk) * C(k,p) = \sum_l (ij|kl) *
                        // C(k,p)
                        if (s3 != s4) {
                          const size_t idx2 =
                              (bf1 * num_atomic_orbitals + bf2) * inner_size +
                              bf4 * nt + p;
                          if (use_thread_local_buffers_) {
                            out_thread[idx2] +=
                                C[bf3 * nt + p] * value * s12_deg;
                          } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            out[idx2] += C[bf3 * nt + p] * value * s12_deg;
                          }
                        }

                        // (kl|ip) = \sum_j (kl|ij) * C(j,p)
                        const size_t idx3 =
                            (bf3 * num_atomic_orbitals + bf4) * inner_size +
                            bf1 * nt + p;
                        if (use_thread_local_buffers_) {
                          out_thread[idx3] += C[bf2 * nt + p] * value * s34_deg;
                        } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                          out[idx3] += C[bf2 * nt + p] * value * s34_deg;
                        }

                        // (kl|jp) = \sum_i (kl|ji) * C(i,p)
                        if (s1 != s2) {
                          const size_t idx4 =
                              (bf3 * num_atomic_orbitals + bf4) * inner_size +
                              bf2 * nt + p;
                          if (use_thread_local_buffers_) {
                            out_thread[idx4] +=
                                C[bf1 * nt + p] * value * s34_deg;
                          } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            out[idx4] += C[bf1 * nt + p] * value * s34_deg;
                          }
                        }
                      }
                    }  // l (bf4)
                  }  // k (bf3)
                }  // j (bf2)
              }  // i (bf1)

            }  // s4
          }  // s3
        }  // s2
      }  // s1
    }

    // Deterministic reduction: combine thread-local buffers in order
    if (use_thread_local_buffers_) {
      for (int t = 0; t < nthreads_; ++t) {
        for (size_t i = 0; i < out_size; ++i) {
          out[i] += out_local[t][i];
        }
      }
    }

// Symmetrize the first and second index: (ij|kp) = 0.5 * ( (ji|kp) + (ij|kp) ),
// then cut half due to s12_34_deg
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < num_atomic_orbitals; ++i)
      for (size_t j = 0; j <= i; ++j) {
        for (size_t k = 0; k < num_atomic_orbitals; ++k) {
          for (size_t p = 0; p < nt; ++p) {
            const size_t idx_ij =
                (i * num_atomic_orbitals + j) * inner_size + k * nt + p;
            const size_t idx_ji =
                (j * num_atomic_orbitals + i) * inner_size + k * nt + p;
            const double a = out[idx_ij];
            const double b = out[idx_ji];
            const double avg =
                0.25 * (a + b);  // average and then cut half due to s12_34_deg
            out[idx_ij] = avg;
            out[idx_ji] = avg;
          }
        }
      }
  }

  /**
   * @brief Factory method to create Libint2 direct ERI instance
   *
   * Template factory method that forwards arguments to the ERI constructor.
   * Provides a clean interface for creating ERI instances with perfect
   * forwarding of constructor arguments.
   *
   * @tparam Args Parameter pack for constructor arguments
   * @param args Arguments to forward to ERI constructor
   * @return Unique pointer to new Libint2 direct ERI instance
   */
  template <typename... Args>
  static std::unique_ptr<ERI> make_libint2_direct_eri(Args&&... args) {
    QDK_LOG_TRACE_ENTERING();

    return std::make_unique<ERI>(std::forward<Args>(args)...);
  }
};

}  // namespace libint2::direct

LIBINT2_DIRECT::LIBINT2_DIRECT(SCFOrbitalType scf_orbital_type,
                               BasisSet& basis_set, ParallelConfig _mpi,
                               bool use_atomics, double eri_threshold,
                               double shell_pair_threshold)
    : ERI(scf_orbital_type, eri_threshold, basis_set, _mpi),
      eri_impl_(libint2::direct::ERI::make_libint2_direct_eri(
          scf_orbital_type == SCFOrbitalType::Restricted ? 1 : 2, basis_set,
          use_atomics, eri_threshold, shell_pair_threshold)) {
  QDK_LOG_TRACE_ENTERING();
  if (_mpi.world_size > 1) throw std::runtime_error("LIBINT2_DIRECT + MPI NYI");
}

LIBINT2_DIRECT::~LIBINT2_DIRECT() noexcept = default;

// Public interface implementation - delegates to internal ERI implementation
void LIBINT2_DIRECT::build_JK(const double* P, double* J, double* K,
                              double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

// Override implementation for ERI base class
void LIBINT2_DIRECT::build_JK_impl_(const double* P, double* J, double* K,
                                    double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

// Gradient computation interface
void LIBINT2_DIRECT::get_gradients(const double* P, double* dJ, double* dK,
                                   double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
}

// Quarter transformation interface
void LIBINT2_DIRECT::quarter_trans_impl(size_t nt, const double* C,
                                        double* out) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
};

}  // namespace qdk::chemistry::scf
