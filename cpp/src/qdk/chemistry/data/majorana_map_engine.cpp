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
  //
  // For same-spin (σ==τ), δ_{qσ,rτ} = δ_{q,r} and E_{pσ,sτ} = E_{pσ,sσ}
  // For cross-spin (σ≠τ), δ_{qσ,rτ} = 0 (different spin)

  // Helper: accumulate E_pσ E_rτ product with two-body coefficient
  // E_pq · E_rs = (1/16) Σ_{a,b,c,d} c[a][b] c[c][d] γ_{2p+a} γ_{2q+b} γ_{2r+c} γ_{2s+d}
  auto accumulate_two_body_product = [&](std::size_t mode_p,
                                         std::size_t mode_q,
                                         std::size_t mode_r,
                                         std::size_t mode_s, double eri) {
    // Factor: 0.5 * eri / 16 = eri * kSixteenth * 0.5
    double half_eri = 0.5 * eri;
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        std::complex<double> c_ab = kC[a][b];
        const auto& w_pa = mapping(2 * mode_p + a);
        const auto& w_qb = mapping(2 * mode_q + b);
        for (int c = 0; c < 2; ++c) {
          for (int d = 0; d < 2; ++d) {
            std::complex<double> coeff =
                half_eri * kSixteenth * c_ab * kC[c][d];
            const auto& w_rc = mapping(2 * mode_r + c);
            const auto& w_sd = mapping(2 * mode_s + d);

            // Need to compute (w_pa · w_qb) · (w_rc · w_sd)
            // Use the accumulator's multiply + accumulate
            auto [phase1, prod1] =
                PauliTermAccumulator::multiply_uncached(w_pa, w_qb);
            auto [phase2, prod2] =
                PauliTermAccumulator::multiply_uncached(w_rc, w_sd);
            acc.accumulate_product(prod1, prod2,
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

  // Spin-free case: use spatial ERIs with both spin channels
  if (is_restricted) {
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aaaa[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            // αα channel
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_alpha(r), mode_alpha(s), eri);
            // ββ channel (same ERI for restricted)
            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_beta(r), mode_beta(s), eri);
            // αβ channel
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_beta(r), mode_beta(s), eri);
            // βα channel
            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_alpha(r), mode_alpha(s), eri);

            // δ corrections for same-spin channels
            if (q == r) {
              // -0.5 * eri * E_{p,α,s,α}
              accumulate_epq(mode_alpha(p), mode_alpha(s), -0.5 * eri);
              // -0.5 * eri * E_{p,β,s,β}
              accumulate_epq(mode_beta(p), mode_beta(s), -0.5 * eri);
            }
            // No δ correction for cross-spin channels (σ ≠ τ)
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
