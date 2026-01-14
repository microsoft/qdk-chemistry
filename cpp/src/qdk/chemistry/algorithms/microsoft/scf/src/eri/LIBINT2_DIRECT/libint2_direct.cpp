// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "libint2_direct.h"

#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <qdk/chemistry/utils/logger.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <blas.hh>
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
RowMajorMatrix compute_shellblock_norm(const ::libint2::BasisSet& obs,
                                       const double* A, size_t LDA) {
  QDK_LOG_TRACE_ENTERING();

  auto shell2bf = obs.shell2bf();
  const size_t nsh = obs.size();
  RowMajorMatrix shnrms(nsh, nsh);
  Eigen::Map<const RowMajorMatrix> A_map(A, obs.nbf(), LDA);
  for (auto ish = 0; ish < nsh; ++ish)
    for (auto jsh = 0; jsh < nsh; ++jsh) {
      shnrms(ish, jsh) = A_map
                             .block(shell2bf[ish], shell2bf[jsh],
                                    obs[ish].size(), obs[jsh].size())
                             .lpNorm<Eigen::Infinity>();
    }
  return shnrms;
}

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t = std::vector<
    std::vector<std::shared_ptr<::libint2::ShellPair>>>;  // in same order as
                                                          // shellpair_list_t

/**
 * @brief Precompute significant shell pairs for integral screening
 *
 * Determines which shell pairs have significant overlap and should be included
 * in integral evaluation. Uses overlap integral screening to eliminate
 * negligible shell pairs, reducing computational cost in subsequent ERI
 * evaluation. Also precomputes ShellPair objects containing geometric screening
 * data.
 *
 * @param obs Libint2 orbital basis set
 * @param threshold Overlap threshold for shell pair significance (default:
 * 1e-12)
 * @return Tuple containing shell pair list and precomputed ShellPair data
 *
 * @note Shell pairs on the same center are always considered significant
 * @note ShellPair objects contain logarithmic precision bounds for screening
 */
std::tuple<shellpair_list_t, shellpair_data_t> compute_shellpairs(
    const ::libint2::BasisSet& obs, double threshold = 1e-12) {
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

  shellpair_list_t splist;

  // Initialize with empty lists
  for (size_t i = 0; i < nsh; ++i) {
    splist.insert(std::make_pair(i, std::vector<size_t>()));
  }

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

        if (significant) splist[s1].emplace_back(s2);
      }  // s2
    }  // s1
  }  // parallel

  shellpair_data_t spdata(splist.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t s1 = 0; s1 < nsh; ++s1)
    for (size_t s2 : splist[s1]) {
      spdata[s1].emplace_back(std::make_shared<::libint2::ShellPair>(
          obs[s1], obs[s2], ln_max_engine_precision,
          ::libint2::ScreeningMethod::Original));
    }

  return std::make_tuple(splist, spdata);
}

/**
 * @brief Compute Schwarz integral bounds for screening
 *
 * Calculates the Schwarz inequality bounds (μν|μν)^{1/2} for all shell pairs,
 * which provide upper bounds for two-electron integrals. These bounds are used
 * for efficient integral screening: if (μν|λσ) ≤ K(μν) × K(λσ), then the
 * integral can be neglected if this product is below the required precision.
 *
 * @param obs Libint2 orbital basis set
 * @param use_2norm Whether to use 2-norm (true) or infinity norm (false)
 * @return Matrix of Schwarz bounds K(μν) = sqrt(|(μν|μν)|)
 *
 * @note The returned matrix is symmetric: K(μν) = K(νμ)
 * @note 2-norm typically provides tighter bounds but is more expensive
 */
RowMajorMatrix compute_schwarz_ints(const ::libint2::BasisSet& obs,
                                    bool use_2norm) {
  QDK_LOG_TRACE_ENTERING();

  const size_t nsh = obs.size();

  // Setup the engine
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif
  std::vector<::libint2::Engine> engines(nthreads);
  engines[0] = ::libint2::Engine(::libint2::Operator::coulomb, obs.max_nprim(),
                                 obs.max_l(), 0, 0.0);
  for (int i = 1; i < nthreads; ++i) engines[i] = engines[0];

  RowMajorMatrix K(nsh, nsh);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto& engine = engines[omp_get_thread_num()];
#else
    auto& engine = engines[0];
#endif
    const auto& buf = engine.results();
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
    for (auto i = 0; i < nsh; ++i) {
      for (auto j = 0; j <= i; ++j) {
        const size_t ni = obs[i].size();
        const size_t nj = obs[i].size();
        const size_t nij = ni * nj;
        engine.compute2<::libint2::Operator::coulomb, ::libint2::BraKet::xx_xx,
                        0>(obs[i], obs[j], obs[i], obs[j]);

        Eigen::Map<const RowMajorMatrix> bmap(buf[0], nij, nij);
        auto norm = use_2norm ? bmap.norm() : bmap.lpNorm<Eigen::Infinity>();
        K(i, j) = std::sqrt(norm);
        K(j, i) = K(i, j);
      }
    }
  }

  return K;
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
  bool unrestricted_;              ///< Whether to use unrestricted formalism
  bool use_thread_local_buffers_;  ///< Use thread-local buffers (true) or
                                   ///< atomic ops (false)
  ::libint2::BasisSet obs_;        ///< Libint2 orbital basis set representation
  std::vector<size_t>
      shell2bf_;  ///< Mapping from shell index to first atomic orbital
  shellpair_list_t splist_;   ///< Pre-computed shell pair list for screening
  shellpair_data_t spdata_;   ///< Shell pair geometric data and bounds
  RowMajorMatrix K_schwarz_;  ///< Schwarz screening matrix for integral bounds

 public:
  /**
   * @brief Construct Libint2 direct ERI engine
   *
   * Initializes the direct integral evaluation engine with the given basis set.
   * Precomputes shell pair lists and Schwarz bounds for efficient screening
   * during integral evaluation. All screening data is computed once during
   * construction and reused throughout the calculation.
   *
   * @param unr Whether to use unrestricted formalism
   * @param basis_set QDK basis set (converted to Libint2 format internally)
   *
   * @note Construction involves significant overhead due to screening setup
   * @note Shell pair and Schwarz data is computed using OpenMP parallelization
   */
  ERI(bool unr, qdk::chemistry::scf::BasisSet& basis_set, bool use_atomics)
      : unrestricted_(unr),
        use_thread_local_buffers_(!use_atomics),
        obs_(libint2_util::convert_to_libint_basisset(basis_set)) {
    QDK_LOG_TRACE_ENTERING();

    shell2bf_ = obs_.shell2bf();

    // Compute Shell Pairs
    std::tie(splist_, spdata_) = compute_shellpairs(obs_);

    // Compute Schwarz Screening
    K_schwarz_ = compute_schwarz_ints(obs_, true);
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
    const size_t num_atomic_orbitals = obs_.nbf();
    const size_t nsh = obs_.size();
    const size_t num_density_matrices = unrestricted_ ? 2 : 1;
    const size_t mat_size =
        num_density_matrices * num_atomic_orbitals * num_atomic_orbitals;
    const bool is_rsx = std::abs(omega) > 1e-12;
    const double precision = std::numeric_limits<double>::epsilon();

    if (is_rsx) throw std::runtime_error("RSX + LIBINT2_DIRECT NYI");

    // Compute shell block norm of P
    const auto P_shnrm = compute_shellblock_norm(obs_, P, num_atomic_orbitals);
    const auto P_shmax = P_shnrm.maxCoeff();

    // Check for NaN/Inf values
    if (!std::isfinite(P_shmax)) {
      throw std::runtime_error("Density matrix contains NaN/Inf values.");
    }

    // Setup required precision
    const auto engine_precision = precision / P_shmax;

    // Setup the engine
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<::libint2::Engine> engines_coulomb(nthreads);
    engines_coulomb[0] = ::libint2::Engine(::libint2::Operator::coulomb,
                                           obs_.max_nprim(), obs_.max_l(), 0);
    engines_coulomb[0].set(::libint2::ScreeningMethod::Original);
    engines_coulomb[0].set_precision(engine_precision);
    for (int i = 1; i < nthreads; ++i) engines_coulomb[i] = engines_coulomb[0];

    if (J) std::memset(J, 0, mat_size * sizeof(double));
    if (K) std::memset(K, 0, mat_size * sizeof(double));

    // Thread-local accumulation buffers for reproducibility
    std::vector<std::vector<double>> J_local(0);
    std::vector<std::vector<double>> K_local(0);

    if (use_thread_local_buffers_) {
      J_local.resize(nthreads);
      K_local.resize(nthreads);
      if (J) {
        for (int t = 0; t < nthreads; ++t) {
          J_local[t].resize(mat_size, 0.0);
        }
      }
      if (K) {
        for (int t = 0; t < nthreads; ++t) {
          K_local[t].resize(mat_size, 0.0);
        }
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
      auto& engine = engines_coulomb[thread_id];
      const auto& buf = engine.results();

      // Get pointers to thread-local buffers
      double* J_thread = nullptr;
      double* K_thread = nullptr;
      if (use_thread_local_buffers_) {
        J_thread = J ? J_local[thread_id].data() : nullptr;
        K_thread = K ? K_local[thread_id].data() : nullptr;
      }

      for (size_t s1 = 0ul, s1234 = 0ul; s1 < nsh; ++s1) {
        const auto bf1_st = shell2bf_[s1];
        const auto n1 = obs_[s1].size();

        auto sp12_data_it = spdata_.at(s1).begin();

        // Only loop over significant shell pairs
        for (size_t s2 : splist_[s1]) {
          const auto bf2_st = shell2bf_[s2];
          const auto n2 = obs_[s2].size();
          const auto* sp12_data = sp12_data_it->get();
          sp12_data_it++;

          const auto P12_nrm = P_shnrm(s1, s2);

          for (size_t s3 = 0; s3 <= s1; ++s3) {
            const auto bf3_st = shell2bf_[s3];
            const auto n3 = obs_[s3].size();

            const auto P13_nrm = P_shnrm(s1, s3);
            const auto P23_nrm = P_shnrm(s2, s3);
            const auto P123_nrm = std::max({P12_nrm, P13_nrm, P23_nrm});

            auto sp34_data_it = spdata_.at(s3).begin();
            const size_t s4_max = (s1 == s3) ? s2 : s3;

            // Only loop over significant shell pairs
            for (size_t s4 : splist_[s3]) {
              if (s4 > s4_max) break;

              const auto* sp34_data = sp34_data_it->get();
              sp34_data_it++;

              // Assign to threads
              if ((s1234++) % nthreads != thread_id) continue;

              // Determine if we need to compute this integral via Schwarz
              const auto P14_nrm = P_shnrm(s1, s4);
              const auto P24_nrm = P_shnrm(s2, s4);
              const auto P34_nrm = P_shnrm(s3, s4);
              const auto P1234_nrm =
                  std::max({P14_nrm, P24_nrm, P34_nrm, P123_nrm});

              if (P1234_nrm * K_schwarz_(s1, s2) * K_schwarz_(s3, s4) <
                  precision)
                continue;

              const auto bf4_st = shell2bf_[s4];
              const auto n4 = obs_[s4].size();

              // Permutational Degeneracy
              auto s12_deg = (s1 == s2) ? 1 : 2;
              auto s34_deg = (s3 == s4) ? 1 : 2;
              auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
              auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

              // Compute the integral shell quartet
              engine.compute2<::libint2::Operator::coulomb,
                              ::libint2::BraKet::xx_xx, 0>(
                  obs_[s1], obs_[s2], obs_[s3], obs_[s4], sp12_data, sp34_data);

              // Coarse integral screening
              const auto buf_1234 = buf[0];
              if (buf_1234 == nullptr) continue;

              // Contract shell quartet (J)
              if (J)
                for (size_t idm = 0; idm < num_density_matrices; ++idm) {
                  auto* J_cur =
                      use_thread_local_buffers_
                          ? J_thread +
                                idm * num_atomic_orbitals * num_atomic_orbitals
                          : J + idm * num_atomic_orbitals * num_atomic_orbitals;
                  auto* P_cur =
                      P + idm * num_atomic_orbitals * num_atomic_orbitals;
                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    const size_t bf1 = bf1_st + i;
                    for (size_t j = 0; j < n2; ++j) {
                      const size_t bf2 = bf2_st + j;
                      double J_ij = 0.0;
                      const double P_ij =
                          P_cur[bf1 * num_atomic_orbitals + bf2];
                      for (size_t k = 0; k < n3; ++k) {
                        const size_t bf3 = bf3_st + k;
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const size_t bf4 = bf4_st + l;

                          const auto value = buf_1234[ijkl] * s1234_deg;

                          // J contractions
                          J_ij +=
                              P_cur[bf3 * num_atomic_orbitals + bf4] * value;
                          if (use_thread_local_buffers_) {
                            J_cur[bf3 * num_atomic_orbitals + bf4] +=
                                P_ij * value;
                          } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            J_cur[bf3 * num_atomic_orbitals + bf4] +=
                                P_ij * value;
                          }

                        }  // l
                      }  // k

                      // Update J
                      if (use_thread_local_buffers_) {
                        J_cur[bf1 * num_atomic_orbitals + bf2] += J_ij;
                      } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                        J_cur[bf1 * num_atomic_orbitals + bf2] += J_ij;
                      }
                    }  // j
                  }  // i
                }  // idm

              // Contract shell quartet (K)
              if (K)
                for (size_t idm = 0; idm < num_density_matrices; ++idm) {
                  auto* K_cur =
                      use_thread_local_buffers_
                          ? K_thread +
                                idm * num_atomic_orbitals * num_atomic_orbitals
                          : K + idm * num_atomic_orbitals * num_atomic_orbitals;
                  auto* P_cur =
                      P + idm * num_atomic_orbitals * num_atomic_orbitals;
                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    const size_t bf1 = bf1_st + i;
                    for (size_t j = 0; j < n2; ++j) {
                      const size_t bf2 = bf2_st + j;
                      for (size_t k = 0; k < n3; ++k) {
                        const size_t bf3 = bf3_st + k;
                        double K_ik = 0.0;
                        double K_jk = 0.0;
                        const double P_ik =
                            0.25 * P_cur[bf1 * num_atomic_orbitals + bf3];
                        const double P_jk =
                            0.25 * P_cur[bf2 * num_atomic_orbitals + bf3];
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const size_t bf4 = bf4_st + l;

                          const auto value = buf_1234[ijkl] * s1234_deg;

                          // K contractions
                          K_ik += 0.25 *
                                  P_cur[bf2 * num_atomic_orbitals + bf4] *
                                  value;
                          K_jk += 0.25 *
                                  P_cur[bf1 * num_atomic_orbitals + bf4] *
                                  value;
                          if (use_thread_local_buffers_) {
                            K_cur[bf1 * num_atomic_orbitals + bf4] +=
                                P_jk * value;
                            K_cur[bf2 * num_atomic_orbitals + bf4] +=
                                P_ik * value;
                          } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            K_cur[bf1 * num_atomic_orbitals + bf4] +=
                                P_jk * value;
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            K_cur[bf2 * num_atomic_orbitals + bf4] +=
                                P_ik * value;
                          }

                        }  // l

                        // Update K
                        if (use_thread_local_buffers_) {
                          K_cur[bf1 * num_atomic_orbitals + bf3] += K_ik;
                          K_cur[bf2 * num_atomic_orbitals + bf3] += K_jk;
                        } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                          K_cur[bf1 * num_atomic_orbitals + bf3] += K_ik;
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                          K_cur[bf2 * num_atomic_orbitals + bf3] += K_jk;
                        }
                      }  // k
                    }  // j
                  }  // i
                }  // idm

            }  // s4
          }  // s3
        }  // s2
      }  // s1

    }  // End parallel region

    // Deterministic reduction: combine thread-local buffers in order
    if (use_thread_local_buffers_) {
      if (J) {
        for (int t = 0; t < nthreads; ++t) {
          for (size_t i = 0; i < mat_size; ++i) {
            J[i] += J_local[t][i];
          }
        }
      }
      if (K) {
        for (int t = 0; t < nthreads; ++t) {
          for (size_t i = 0; i < mat_size; ++i) {
            K[i] += K_local[t][i];
          }
        }
      }
    }

    // Symmetrize J
    if (J)
      for (size_t idm = 0; idm < num_density_matrices; ++idm)
        for (size_t i = 0; i < num_atomic_orbitals; ++i)
          for (size_t j = 0; j <= i; ++j) {
            auto J_ij = J[idm * num_atomic_orbitals * num_atomic_orbitals +
                          i * num_atomic_orbitals + j];
            auto J_ji = J[idm * num_atomic_orbitals * num_atomic_orbitals +
                          j * num_atomic_orbitals + i];
            J_ij = 0.25 * (J_ij + J_ji);
            J[idm * num_atomic_orbitals * num_atomic_orbitals +
              i * num_atomic_orbitals + j] = J_ij;
            J[idm * num_atomic_orbitals * num_atomic_orbitals +
              j * num_atomic_orbitals + i] = J_ij;
          }

    // Symmetrize K + scale by alpha/beta
    if (K)
      for (size_t idm = 0; idm < num_density_matrices; ++idm)
        for (size_t i = 0; i < num_atomic_orbitals; ++i)
          for (size_t j = 0; j <= i; ++j) {
            auto K_ij = K[idm * num_atomic_orbitals * num_atomic_orbitals +
                          i * num_atomic_orbitals + j];
            auto K_ji = K[idm * num_atomic_orbitals * num_atomic_orbitals +
                          j * num_atomic_orbitals + i];
            K_ij = (alpha + beta) * 0.5 * (K_ij + K_ji);
            K[idm * num_atomic_orbitals * num_atomic_orbitals +
              i * num_atomic_orbitals + j] = K_ij;
            K[idm * num_atomic_orbitals * num_atomic_orbitals +
              j * num_atomic_orbitals + i] = K_ij;
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

  std::unique_ptr<double[]> get_cholesky_vectors(double threshold,
                                                 size_t* num_vectors) {
    QDK_LOG_TRACE_ENTERING();
    QDK_LOG_INFO("Cholesky decomposition threshold: {}", threshold);

    const size_t num_aos = obs_.nbf();
    const size_t num_aos2 = num_aos * num_aos;
    const size_t num_aos3 = num_aos2 * num_aos;
    const size_t num_shells = obs_.size();
    const size_t num_shell_pairs = num_shells * (num_shells + 1) / 2;

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
    const double precision = std::numeric_limits<double>::epsilon();
    const auto engine_precision = precision;
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<::libint2::Engine> engines_coulomb(nthreads);
    engines_coulomb[0] = ::libint2::Engine(::libint2::Operator::coulomb,
                                           obs_.max_nprim(), obs_.max_l(), 0);
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

    // Compute diagonal elements for all shell pairs
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
        const auto n1 = obs_[s1].size();
        for (size_t s2 = 0; s2 <= s1; ++s2) {
          // Assign to threads
          if ((s12++) % nthreads != thread_id) continue;

          const auto n2 = obs_[s2].size();
          const size_t sp_index = shell_pair_index(s1, s2);
          const size_t n12 = n1 * n2;

          // screening via schwarz bounds
          if (K_schwarz_(s1, s2) * K_schwarz_(s1, s2) < precision) {
            continue;
          }

          // compute diagonal block (s1,s2|s1,s2)
          engine.compute2<::libint2::Operator::coulomb,
                          ::libint2::BraKet::xx_xx, 0>(obs_[s1], obs_[s2],
                                                       obs_[s1], obs_[s2]);
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

    // Cholesky decomposition
    while (true) {
      // get max diagonal element
      size_t q_shell_pair_max;
      double D_max = 0.0;
      size_t s1_max, s2_max;
      for (const auto sp_index : active_shell_pairs) {
        const auto [s1, s2] = sp_index_to_shells[sp_index];
        // get block max
        const auto& diag = D_shell_pair[sp_index];
        const double block_max = *std::max_element(diag.begin(), diag.end());
        if (block_max > D_max) {
          D_max = block_max;
          q_shell_pair_max = sp_index;
          s1_max = s1;
          s2_max = s2;
        }
      }

      // check convergence
      if (D_max < threshold) {
        break;
      }

      // get column for max shell pair (s1_max, s2_max)
      const size_t n1_max = obs_[s1_max].size();
      const size_t n2_max = obs_[s2_max].size();
      const size_t n_cols = n1_max * n2_max;
      std::vector<double> eri_col(num_aos2 * n_cols, 0.0);
#ifdef _OPENMP
      std::vector<std::vector<double>> eri_col_threads(nthreads);
      for (int t = 0; t < nthreads; ++t) {
        eri_col_threads[t].resize(num_aos2 * n_cols, 0.0);
      }
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
        const auto thread_id = omp_get_thread_num();
        // get thread-local eri_col
        auto& eri_col_local = eri_col_threads[thread_id];
#else
        const auto thread_id = 0;
#endif
        auto& engine = engines_coulomb[thread_id];
        const auto& buf = engine.results();
        for (size_t s3 = 0, s34 = 0; s3 < num_shells; ++s3) {
          const size_t n3 = obs_[s3].size();
          const size_t bf3_st = shell2bf_[s3];
          for (size_t s4 = 0; s4 < num_shells; ++s4) {
            // Assign to threads
            if ((s34++) % nthreads != thread_id) continue;

            // screening via schwarz bounds
            if (K_schwarz_(s1_max, s2_max) * K_schwarz_(s3, s4) < precision) {
              continue;
            }

            const size_t n4 = obs_[s4].size();
            const size_t bf4_st = shell2bf_[s4];

            // compute integral shell quartet
            engine.compute2<::libint2::Operator::coulomb,
                            ::libint2::BraKet::xx_xx, 0>(
                obs_[s1_max], obs_[s2_max], obs_[s3], obs_[s4]);
            const auto& res = buf[0];
            if (res == nullptr) continue;

            // fill in eri_col
            for (size_t i = 0, ijkl = 0; i < n1_max; ++i) {
              const size_t ind_i = i * n2_max;
              for (size_t j = 0; j < n2_max; ++j) {
                const size_t ind_ij = (ind_i + j) * num_aos2;
                for (size_t k = 0; k < n3; ++k) {
                  const size_t ind_ijk = ind_ij + (bf3_st + k) * num_aos;
                  for (size_t l = 0; l < n4; ++l, ++ijkl) {
                    const size_t ind_ijkl = ind_ijk + (bf4_st + l);
#ifdef _OPENMP
                    eri_col_local[ind_ijkl] += res[ijkl];
#else
                    eri_col[ind_ijkl] += res[ijkl];
#endif
                  }  // l
                }  // k
              }  // j
            }  // i
          }  // s4
        }  // s3
      }  // omp parallel

      // merge thread-local eri_col
#ifdef _OPENMP
      for (int t = 0; t < nthreads; ++t) {
        blas::axpy(eri_col.size(), 1.0, eri_col_threads[t].data(), 1,
                   eri_col.data(), 1);
      }
#endif

      // precompute lookup
      const size_t bf1_max_st = shell2bf_[s1_max];
      const size_t bf2_max_st = shell2bf_[s2_max];
      std::vector<size_t> shell_pairs_to_lookup(n1_max * n2_max);
      for (size_t i = 0; i < n1_max; ++i) {
        for (size_t j = 0; j < n2_max; ++j) {
          const size_t local_index = i * n2_max + j;
          shell_pairs_to_lookup[local_index] =
              (bf1_max_st + i) * num_aos + (bf2_max_st + j);
        }
      }

      // correct for cholesky contributions by subtracting previous vectors
      if (current_col > 0) {
        // Extract rows from L_data corresponding to shell pairs
        std::vector<double> L_rows(n_cols * current_col);
        for (size_t col = 0; col < n_cols; ++col) {
          const size_t global_index = shell_pairs_to_lookup[col];
          // Copy row from L_data: L_rows[col, :] = L_data[global_index, :]
          blas::copy(current_col, L_data.data() + global_index, num_aos2,
                     L_rows.data() + col, n_cols);
        }
        // Compute eri_col -= L_data * L_rows^T
        // eri_col is num_aos2 x n_cols, L_data is num_aos2 x current_col,
        // L_rows^T is current_col x n_cols
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   num_aos2, n_cols, current_col, -1.0, L_data.data(), num_aos2,
                   L_rows.data(), n_cols, 1.0, eri_col.data(), num_aos2);
      }

      // form new cholesky vector for each index in shell pair
      for (size_t local_i = 0; local_i < n1_max; ++local_i) {
        for (size_t local_j = 0; local_j < n2_max; ++local_j) {
          const size_t local_index = local_i * n2_max + local_j;
          const double D_val = D_shell_pair[q_shell_pair_max][local_index];

          // skip if below threshold
          if (D_val < threshold) {
            continue;
          }

          // Form Cholesky vector
          double Q_max = std::sqrt(1.0 / D_val);
          std::vector<double> L_col_vec(num_aos2);
          // Copy column from eri_col
          blas::copy(num_aos2, eri_col.data() + local_index * num_aos2, 1,
                     L_col_vec.data(), 1);
          // Scale by Q_max
          blas::scal(num_aos2, Q_max, L_col_vec.data(), 1);

          // append to L_data
          L_data.insert(L_data.end(), L_col_vec.data(),
                        L_col_vec.data() + num_aos2);

          // reference to current column
          const double* L_col = L_data.data() + current_col * num_aos2;

          // Update remaining columns in eri_col for vectors formed within this
          // shell pair
          for (size_t col = local_index + 1; col < n1_max * n2_max; ++col) {
            const size_t global_col_idx = shell_pairs_to_lookup[col];
            const double scale_factor = -L_col[global_col_idx];
            double* eri_col_ptr = eri_col.data() + col * num_aos2;
            blas::axpy(num_aos2, scale_factor, L_col, 1, eri_col_ptr, 1);
          }

          // Update diagonal elements
          std::vector<size_t> shell_pairs_to_remove;
          for (const auto sp_index : active_shell_pairs) {
            const auto [s1, s2] = sp_index_to_shells[sp_index];
            const auto n1 = obs_[s1].size();
            const auto n2 = obs_[s2].size();
            const auto bf1_st = shell2bf_[s1];
            const auto bf2_st = shell2bf_[s2];
            // update diagonal block
            for (size_t i = 0; i < n1; ++i) {
              const size_t bf1_st_i = (bf1_st + i) * num_aos;
              for (size_t j = 0; j < n2; ++j) {
                const size_t idx = i * n2 + j;
                const size_t global_idx = bf1_st_i + (bf2_st + j);
                D_shell_pair[sp_index][idx] -=
                    L_col[global_idx] * L_col[global_idx];
              }
            }
            // remove if below threshold
            const auto& diag = D_shell_pair[sp_index];
            const double max_diag = *std::max_element(diag.begin(), diag.end());
            if (max_diag < threshold) {
              shell_pairs_to_remove.push_back(sp_index);
            }
          }
          // remove inactive shell pairs
          for (const auto sp_index : shell_pairs_to_remove) {
            active_shell_pairs.erase(sp_index);
          }
          current_col += 1;
        }
      }
    }

    QDK_LOG_INFO("Cholesky rank: {}", current_col);

    // Allocate and return exact size
    const size_t data_size = current_col * num_aos * num_aos;
    auto output = std::make_unique<double[]>(data_size);
    std::memcpy(output.get(), L_data.data(), data_size * sizeof(double));
    *num_vectors = current_col;
    return output;
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
    const double precision = std::numeric_limits<double>::epsilon();

    // Clear output tensor: (ij|kp) with p as fast index
    const size_t out_size =
        nt * num_atomic_orbitals * num_atomic_orbitals * num_atomic_orbitals;
    std::memset(out, 0, out_size * sizeof(double));
    const size_t inner_size = nt * num_atomic_orbitals;

    // No shell block norm needed for quarter transformation
    // We'll use Schwarz screening directly

    // Setup required precision (using default Schwarz screening)
    const auto engine_precision = precision;

    // Setup the engine
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<::libint2::Engine> engines_coulomb(nthreads);
    engines_coulomb[0] = ::libint2::Engine(::libint2::Operator::coulomb,
                                           obs_.max_nprim(), obs_.max_l(), 0);
    engines_coulomb[0].set(::libint2::ScreeningMethod::Original);
    engines_coulomb[0].set_precision(engine_precision);
    for (int i = 1; i < nthreads; ++i) engines_coulomb[i] = engines_coulomb[0];

    // Thread-local accumulation buffers for reproducibility
    std::vector<std::vector<double>> out_local(0);
    if (use_thread_local_buffers_) {
      out_local.resize(nthreads);
      for (int t = 0; t < nthreads; ++t) {
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
      auto& engine = engines_coulomb[thread_id];
      const auto& buf = engine.results();

      // Get pointer to thread-local buffer
      double* out_thread = nullptr;
      if (use_thread_local_buffers_) {
        out_thread = out_local[thread_id].data();
      }

      for (size_t s1 = 0ul, s1234 = 0ul; s1 < nsh; ++s1) {
        const auto bf1_st = shell2bf_[s1];
        const auto n1 = obs_[s1].size();

        auto sp12_data_it = spdata_.at(s1).begin();

        // Only loop over significant shell pairs
        for (size_t s2 : splist_[s1]) {
          const auto bf2_st = shell2bf_[s2];
          const auto n2 = obs_[s2].size();
          const auto* sp12_data = sp12_data_it->get();
          sp12_data_it++;

          // Permutational Degeneracy
          auto s12_deg = (s1 == s2) ? 1 : 2;

          for (size_t s3 = 0; s3 <= s1; ++s3) {
            const auto bf3_st = shell2bf_[s3];
            const auto n3 = obs_[s3].size();

            auto sp34_data_it = spdata_.at(s3).begin();
            const size_t s4_max = (s1 == s3) ? s2 : s3;

            // Only loop over significant shell pairs
            for (size_t s4 : splist_[s3]) {
              if (s4 > s4_max) break;

              const auto* sp34_data = sp34_data_it->get();
              sp34_data_it++;

              // Assign to threads
              if ((s1234++) % nthreads != thread_id) continue;

              // Use Schwarz screening only
              if (K_schwarz_(s1, s2) * K_schwarz_(s3, s4) < precision) continue;

              const auto bf4_st = shell2bf_[s4];
              const auto n4 = obs_[s4].size();

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
      for (int t = 0; t < nthreads; ++t) {
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

LIBINT2_DIRECT::LIBINT2_DIRECT(bool unr, BasisSet& basis_set,
                               ParallelConfig _mpi, bool use_atomics)
    : ERI(unr, 0.0, basis_set, _mpi),
      eri_impl_(libint2::direct::ERI::make_libint2_direct_eri(unr, basis_set,
                                                              use_atomics)) {
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

// Cholesky vectors interface
std::unique_ptr<double[]> LIBINT2_DIRECT::get_cholesky_vectors(
    double threshold, size_t* num_vectors) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  return eri_impl_->get_cholesky_vectors(threshold, num_vectors);
}
}  // namespace qdk::chemistry::scf
