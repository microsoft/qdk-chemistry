// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <qdk/chemistry/data/majorana_map_engine.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

namespace {

// Compile-time Majorana decomposition coefficients for a†_p a_q:
//   a†_p a_q = (1/4) Σ_{a,b} c[a][b] · γ_{2p+a} · γ_{2q+b}
//   c[a][b] = (-i)^a · (i)^b
//   c[0][0] = 1,  c[0][1] = i,  c[1][0] = -i,  c[1][1] = 1
constexpr std::complex<double> kC00{1.0, 0.0};
constexpr std::complex<double> kC01{0.0, 1.0};
constexpr std::complex<double> kC10{0.0, -1.0};
constexpr std::complex<double> kC11{1.0, 0.0};

// All four coefficients in indexable form: kC[a][b]
constexpr std::complex<double> kC[2][2] = {{kC00, kC01}, {kC10, kC11}};

// Quarter factor applied to each one-body Majorana product
constexpr double kQuarter = 0.25;

// 1/16 factor for two-body (product of two E operators, each with 1/4)
constexpr double kSixteenth = 0.0625;

/// Convert a SparsePauliWord to a dense little-endian string.
/// (qubit 0 = rightmost character)
std::string sparse_to_le_string(const SparsePauliWord& word,
                                std::size_t num_qubits) {
  std::string result(num_qubits, 'I');
  for (const auto& [qubit, op_type] : word) {
    if (qubit < num_qubits) {
      switch (op_type) {
        case 1:
          result[qubit] = 'X';
          break;
        case 2:
          result[qubit] = 'Y';
          break;
        case 3:
          result[qubit] = 'Z';
          break;
        default:
          break;
      }
    }
  }
  // Reverse to get little-endian (qubit 0 = rightmost)
  std::reverse(result.begin(), result.end());
  return result;
}

}  // namespace

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy,
    const double* h1_alpha, const double* h1_beta, const double* eri_aaaa,
    const double* eri_aabb, const double* eri_bbbb, std::size_t n_spatial,
    bool is_restricted, double threshold, double integral_threshold) {
  const std::size_t n_modes = 2 * n_spatial;  // spin-orbitals
  PauliTermAccumulator acc;

  // Helper: compute spin-orbital mode index from spatial + spin
  // Blocked ordering: mode(p, α) = p, mode(p, β) = p + n_spatial
  auto mode_alpha = [](std::size_t p) -> std::size_t { return p; };
  auto mode_beta = [n_spatial](std::size_t p) -> std::size_t {
    return p + n_spatial;
  };

  // Helper: accumulate one-body E_pq = a†_p a_q for specific mode pair
  // E_pq = (1/4) Σ_{a,b} c[a][b] · γ_{2p+a} · γ_{2q+b}
  auto accumulate_epq = [&](std::size_t mode_p, std::size_t mode_q,
                            double h_pq) {
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        std::complex<double> coeff =
            h_pq * kQuarter * kC[a][b];
        acc.accumulate_product(mapping(2 * mode_p + a),
                               mapping(2 * mode_q + b), coeff);
      }
    }
  };

  // ─── Core energy ──────────────────────────────────────────────────
  if (std::abs(core_energy) > integral_threshold) {
    acc.accumulate({}, std::complex<double>(core_energy, 0.0));
  }

  // ─── One-body terms ───────────────────────────────────────────────
  // H_1 = Σ_{p,q,σ} h_pq a†_{p,σ} a_{q,σ}
  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t q = 0; q < n_spatial; ++q) {
      double h_pq_a = h1_alpha[p * n_spatial + q];
      if (std::abs(h_pq_a) > integral_threshold) {
        accumulate_epq(mode_alpha(p), mode_alpha(q), h_pq_a);
      }

      double h_pq_b = is_restricted ? h_pq_a : h1_beta[p * n_spatial + q];
      if (std::abs(h_pq_b) > integral_threshold) {
        accumulate_epq(mode_beta(p), mode_beta(q), h_pq_b);
      }
    }
  }

  // ─── Two-body terms ───────────────────────────────────────────────
  // H_2 = (1/2) Σ_{pqrs,σ,τ} (pq|rs) a†_{p,σ} a†_{r,τ} a_{s,τ} a_{q,σ}
  //      = (1/2) Σ_{pqrs,σ,τ} (pq|rs) [E_{pσ,qσ} E_{rτ,sτ} - δ_{qσ,rτ} E_{pσ,sτ}]

  // Precompute Majorana pair products for SAME-SPIN blocks only.
  // Majorana indices for α: [0, 2*n_spatial), for β: [2*n_spatial, 4*n_spatial).
  // Downstream only ever pairs indices within the same spin block.
  // Two flat caches (α and β), each (2*n_spatial)² entries.
  struct MajPairProduct {
    std::complex<double> phase;
    SparsePauliWord word;
  };
  const std::size_t maj_per_spin = 2 * n_spatial;  // Majorana indices per spin
  std::vector<MajPairProduct> pair_cache_alpha(maj_per_spin * maj_per_spin);
  std::vector<MajPairProduct> pair_cache_beta(maj_per_spin * maj_per_spin);

  for (std::size_t i = 0; i < maj_per_spin; ++i) {
    for (std::size_t j = 0; j < maj_per_spin; ++j) {
      // α block: Majorana indices i, j (for modes 0..n_spatial-1)
      auto [ph_a, w_a] = PauliTermAccumulator::multiply_uncached(
          mapping(i), mapping(j));
      pair_cache_alpha[i * maj_per_spin + j] = {ph_a, std::move(w_a)};

      // β block: Majorana indices i+2*n_spatial, j+2*n_spatial
      auto [ph_b, w_b] = PauliTermAccumulator::multiply_uncached(
          mapping(i + maj_per_spin), mapping(j + maj_per_spin));
      pair_cache_beta[i * maj_per_spin + j] = {ph_b, std::move(w_b)};
    }
  }

  // Accessor lambdas: given a mode index m and Majorana sub-index a (0 or 1),
  // return the pair product for two Majorana operators of the same spin.
  // For α: mode m ∈ [0, n_spatial), Majorana index = 2*m+a
  // For β: mode m ∈ [n_spatial, 2*n_spatial), Majorana index = 2*(m-n_spatial)+a
  auto alpha_pair = [&](std::size_t local_i, std::size_t local_j)
      -> const MajPairProduct& {
    return pair_cache_alpha[local_i * maj_per_spin + local_j];
  };
  auto beta_pair = [&](std::size_t local_i, std::size_t local_j)
      -> const MajPairProduct& {
    return pair_cache_beta[local_i * maj_per_spin + local_j];
  };

  // Helper: multiply two SparsePauliWords and accumulate, bypassing the
  // LRU cache (the cache has near-zero hit rate in the O(N^4) loop since
  // keys are unique across (p,q,r,s,a,b,c,d) tuples).
  auto accumulate_product_uncached =
      [&](const SparsePauliWord& w1, const SparsePauliWord& w2,
          std::complex<double> scale) {
        auto [phase, word] =
            PauliTermAccumulator::multiply_uncached(w1, w2);
        acc.accumulate(word, scale * phase);
      };

  // Helper: accumulate E_pσ E_rτ product using precomputed pair products
  auto accumulate_two_body_product = [&](std::size_t mode_p,
                                         std::size_t mode_q,
                                         std::size_t mode_r,
                                         std::size_t mode_s, double eri) {
    // Determine which pair cache to use for each E operator
    bool pq_is_alpha = mode_p < n_spatial;
    bool rs_is_alpha = mode_r < n_spatial;
    auto& cache_pq = pq_is_alpha ? pair_cache_alpha : pair_cache_beta;
    auto& cache_rs = rs_is_alpha ? pair_cache_alpha : pair_cache_beta;
    std::size_t pq_base_p = pq_is_alpha ? mode_p : (mode_p - n_spatial);
    std::size_t pq_base_q = pq_is_alpha ? mode_q : (mode_q - n_spatial);
    std::size_t rs_base_r = rs_is_alpha ? mode_r : (mode_r - n_spatial);
    std::size_t rs_base_s = rs_is_alpha ? mode_s : (mode_s - n_spatial);

    double half_eri = 0.5 * eri;
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        std::complex<double> c_ab = kC[a][b];
        const auto& [phase1, prod1] =
            cache_pq[(2 * pq_base_p + a) * maj_per_spin +
                     (2 * pq_base_q + b)];
        for (int c = 0; c < 2; ++c) {
          for (int d = 0; d < 2; ++d) {
            std::complex<double> coeff =
                half_eri * kSixteenth * c_ab * kC[c][d];
            const auto& [phase2, prod2] =
                cache_rs[(2 * rs_base_r + c) * maj_per_spin +
                         (2 * rs_base_s + d)];
            accumulate_product_uncached(prod1, prod2,
                                        coeff * phase1 * phase2);
          }
        }
      }
    }
  };

  auto idx4 = [n_spatial](std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s) -> std::size_t {
    return ((p * n_spatial + q) * n_spatial + r) * n_spatial + s;
  };

  // Spin-free case: use spin-summed E operators for 4x fewer products
  if (is_restricted) {
    // Precompute spin-summed E_pq: for each spatial (p,q), accumulate
    // the 4 Majorana pair products from both α and β into a single set.
    // E^σ_pq = (1/4) Σ_{a,b} c[a][b] * pair_cache_σ[2*p+a, 2*q+b]
    // E_pq = E^α_pq + E^β_pq
    struct SpinSummedE {
      // Up to 8 terms (4 per spin, some may combine)
      std::vector<std::pair<std::complex<double>, SparsePauliWord>> terms;
    };
    std::vector<SpinSummedE> ss_e(n_spatial * n_spatial);

    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto& sse = ss_e[p * n_spatial + q];
        // α terms: local Majorana indices 2*p+a, 2*q+b
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const auto& [phase, word] =
                alpha_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(kQuarter * kC[a][b] * phase, word);
          }
        }
        // β terms: same local indices but from β cache
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const auto& [phase, word] =
                beta_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(kQuarter * kC[a][b] * phase, word);
          }
        }
      }
    }

    // Two-body: (1/2) Σ_{pqrs} (pq|rs) [E_pq · E_rs - δ_{qr} E_ps]
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        const auto& e_pq = ss_e[p * n_spatial + q];
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aaaa[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            double half_eri = 0.5 * eri;
            const auto& e_rs = ss_e[r * n_spatial + s];

            // E_pq · E_rs product — bypass LRU cache
            for (const auto& [c1, w1] : e_pq.terms) {
              for (const auto& [c2, w2] : e_rs.terms) {
                accumulate_product_uncached(w1, w2, half_eri * c1 * c2);
              }
            }

            // δ_{qr} correction
            if (q == r) {
              const auto& e_ps = ss_e[p * n_spatial + s];
              for (const auto& [c, w] : e_ps.terms) {
                acc.accumulate(w, -half_eri * c);
              }
            }
          }
        }
      }
    }
  } else {
    // Unrestricted: explicit spin-channel ERIs

    // aaaa channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aaaa[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_alpha(r), mode_alpha(s), eri);
            if (q == r) {
              accumulate_epq(mode_alpha(p), mode_alpha(s), -0.5 * eri);
            }
          }
        }
      }
    }

    // bbbb channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_bbbb[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_beta(r), mode_beta(s), eri);
            if (q == r) {
              accumulate_epq(mode_beta(p), mode_beta(s), -0.5 * eri);
            }
          }
        }
      }
    }

    // aabb channel (alpha-beta)
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aabb[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_beta(r), mode_beta(s), eri);
            // No δ correction (different spin)
          }
        }
      }
    }

    // bbaa channel (beta-alpha) — uses same eri_aabb
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aabb[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_alpha(r), mode_alpha(s), eri);
            // No δ correction (different spin)
          }
        }
      }
    }
  }

  // ─── Extract results ──────────────────────────────────────────────
  auto terms = acc.get_terms(threshold);
  std::size_t num_qubits = mapping.num_qubits();

  MajoranaMapResult result;
  result.pauli_strings.reserve(terms.size());
  result.coefficients.reserve(terms.size());

  for (auto& [coeff, word] : terms) {
    result.pauli_strings.push_back(sparse_to_le_string(word, num_qubits));
    result.coefficients.push_back(coeff);
  }

  return result;
}

}  // namespace qdk::chemistry::data
