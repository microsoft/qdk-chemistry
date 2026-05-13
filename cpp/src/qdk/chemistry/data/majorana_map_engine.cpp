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

// ─── Bitpacked Pauli representation ────────────────────────────────
//
// Represents a Pauli string as a pair of bitmasks (x_bits, z_bits)
// using the symplectic encoding:
//   I: x=0, z=0    X: x=1, z=0    Z: x=0, z=1    Y: x=1, z=1
//
// Each uint64_t covers 64 qubits. Templated on NW (number of uint64
// words) to keep everything on the stack. The engine dispatches to
// NW ∈ {1, 2, 3, 4} at runtime (covers up to 256 qubits).

template <std::size_t NW>
struct PackedPauliWord {
  std::array<std::uint64_t, NW> x{};
  std::array<std::uint64_t, NW> z{};

  bool operator==(const PackedPauliWord& other) const = default;
};

template <std::size_t NW>
struct PackedPauliWordHash {
  std::size_t operator()(const PackedPauliWord<NW>& w) const noexcept {
    std::size_t seed = NW;
    for (std::size_t i = 0; i < NW; ++i) {
      seed ^= w.x[i] * 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
    for (std::size_t i = 0; i < NW; ++i) {
      seed ^= w.z[i] * 0x517cc1b727220a95ULL + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template <std::size_t NW>
PackedPauliWord<NW> sparse_to_packed(const SparsePauliWord& word) {
  PackedPauliWord<NW> pw{};
  for (const auto& [qubit, op] : word) {
    std::size_t wi = qubit / 64;
    std::uint64_t bit = std::uint64_t(1) << (qubit % 64);
    if (wi < NW) {
      if (op == 1 || op == 2) pw.x[wi] |= bit;
      if (op == 2 || op == 3) pw.z[wi] |= bit;
    }
  }
  return pw;
}

template <std::size_t NW>
SparsePauliWord packed_to_sparse(const PackedPauliWord<NW>& pw) {
  SparsePauliWord word;
  for (std::size_t wi = 0; wi < NW; ++wi) {
    std::uint64_t x = pw.x[wi];
    std::uint64_t z = pw.z[wi];
    std::uint64_t active = x | z;
    while (active) {
      int bit = __builtin_ctzll(active);
      std::uint64_t mask = std::uint64_t(1) << bit;
      std::uint64_t qubit = wi * 64 + bit;
      bool has_x = (x & mask) != 0;
      bool has_z = (z & mask) != 0;
      std::uint8_t op = has_x ? (has_z ? 2 : 1) : 3;
      word.emplace_back(qubit, op);
      active &= ~mask;
    }
  }
  return word;
}

template <std::size_t NW>
std::pair<int, PackedPauliWord<NW>> multiply_packed(
    const PackedPauliWord<NW>& p1, const PackedPauliWord<NW>& p2) {
  PackedPauliWord<NW> result;
  int phase_exp = 0;
  for (std::size_t i = 0; i < NW; ++i) {
    std::uint64_t x1 = p1.x[i], z1 = p1.z[i];
    std::uint64_t x2 = p2.x[i], z2 = p2.z[i];
    result.x[i] = x1 ^ x2;
    result.z[i] = z1 ^ z2;
    std::uint64_t nx1 = ~x1, nz1 = ~z1, nx2 = ~x2, nz2 = ~z2;
    std::uint64_t cyc =
        (x1 & nz1 & x2 & z2) | (x1 & z1 & nx2 & z2) | (nx1 & z1 & x2 & nz2);
    std::uint64_t anti =
        (x1 & z1 & x2 & nz2) | (nx1 & z1 & x2 & z2) | (x1 & nz1 & nx2 & z2);
    phase_exp += __builtin_popcountll(cyc);
    phase_exp -= __builtin_popcountll(anti);
  }
  // Return phase index (0..3) instead of complex — callers apply phase
  // using branchless real/imag swap, avoiding complex<double> multiply.
  return {phase_exp & 3, result};
}

/// Apply a phase index (0=+1, 1=+i, 2=-1, 3=-i) to a complex scale factor.
/// Returns the scaled value without a full complex multiply.
inline std::complex<double> apply_phase(int phase_idx,
                                        std::complex<double> scale) {
  switch (phase_idx) {
    case 0:
      return scale;
    case 1:
      return {-scale.imag(), scale.real()};
    case 2:
      return {-scale.real(), -scale.imag()};
    case 3:
      return {scale.imag(), -scale.real()};
    default:
      __builtin_unreachable();
  }
}

template <std::size_t NW>
class PackedAccumulator {
 public:
  void accumulate(const PackedPauliWord<NW>& word, std::complex<double> coeff) {
    // Single hash-map probe: operator[] default-constructs (0,0) on miss.
    terms_[word] += coeff;
  }

  void accumulate_product(const PackedPauliWord<NW>& w1,
                          const PackedPauliWord<NW>& w2,
                          std::complex<double> scale) {
    auto [phase_idx, word] = multiply_packed(w1, w2);
    terms_[word] += apply_phase(phase_idx, scale);
  }

  /// Extract terms above threshold as (coefficient, little-endian string)
  /// pairs.
  std::vector<std::pair<std::complex<double>, std::string>>
  get_terms_as_strings(double threshold, std::size_t num_qubits) const {
    std::vector<std::pair<std::complex<double>, std::string>> result;
    result.reserve(terms_.size());
    for (const auto& [pw, coeff] : terms_) {
      if (std::abs(coeff) >= threshold) {
        result.emplace_back(coeff, packed_to_le_string(pw, num_qubits));
      }
    }
    return result;
  }

 private:
  std::unordered_map<PackedPauliWord<NW>, std::complex<double>,
                     PackedPauliWordHash<NW>>
      terms_;
};

/// Convert a PackedPauliWord directly to a dense little-endian string,
/// skipping the intermediate SparsePauliWord allocation.
template <std::size_t NW>
std::string packed_to_le_string(const PackedPauliWord<NW>& pw,
                                std::size_t num_qubits) {
  std::string result(num_qubits, 'I');
  for (std::size_t wi = 0; wi < NW; ++wi) {
    std::uint64_t x = pw.x[wi];
    std::uint64_t z = pw.z[wi];
    std::uint64_t active = x | z;
    while (active) {
      int bit = __builtin_ctzll(active);
      std::uint64_t mask = std::uint64_t(1) << bit;
      std::size_t qubit = wi * 64 + bit;
      if (qubit < num_qubits) {
        bool has_x = (x & mask) != 0;
        bool has_z = (z & mask) != 0;
        // Little-endian: qubit 0 = rightmost
        result[num_qubits - 1 - qubit] = has_x ? (has_z ? 'Y' : 'X') : 'Z';
      }
      active &= ~mask;
    }
  }
  return result;
}

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

namespace {

/// Templated engine implementation, parameterized on NW (number of uint64
/// words per Pauli bitmask). NW is selected at runtime by the dispatcher.
template <std::size_t NW>
MajoranaMapResult majorana_map_impl(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool is_restricted,
    double threshold, double integral_threshold) {
  const std::size_t n_modes = 2 * n_spatial;
  const std::size_t num_qubits = mapping.num_qubits();

  PackedAccumulator<NW> acc;

  // Convert all Majorana table entries to packed form + cache phases
  std::vector<PackedPauliWord<NW>> packed_mapping(2 * n_modes);
  std::vector<std::int8_t> maj_phases(2 * n_modes);
  for (std::size_t k = 0; k < 2 * n_modes; ++k) {
    packed_mapping[k] = sparse_to_packed<NW>(mapping(k));
    maj_phases[k] = mapping.phase(k);
  }
  const bool has_neg_phases = !mapping.all_phases_positive();

  auto mode_alpha = [](std::size_t p) -> std::size_t { return p; };
  auto mode_beta = [n_spatial](std::size_t p) -> std::size_t {
    return p + n_spatial;
  };

  // Helper: accumulate one-body E_pq for a mode pair, using packed types
  // E_pq = (1/4) * sum_{a,b} c[a][b] * phase(2p+a) * phase(2q+b) * P(2p+a) *
  // P(2q+b)
  auto accumulate_epq = [&](std::size_t mode_p, std::size_t mode_q,
                            double h_pq) {
    for (int a = 0; a < 2; ++a) {
      std::size_t idx_pa = 2 * mode_p + a;
      for (int b = 0; b < 2; ++b) {
        std::size_t idx_qb = 2 * mode_q + b;
        auto [ph, word] =
            multiply_packed(packed_mapping[idx_pa], packed_mapping[idx_qb]);
        double sign = has_neg_phases ? static_cast<double>(maj_phases[idx_pa]) *
                                           maj_phases[idx_qb]
                                     : 1.0;
        acc.accumulate(word,
                       apply_phase(ph, sign * h_pq * kQuarter * kC[a][b]));
      }
    }
  };

  // ─── Core energy ──────────────────────────────────────────────────
  if (std::abs(core_energy) > integral_threshold) {
    PackedPauliWord<NW> identity{};
    acc.accumulate(identity, std::complex<double>(core_energy, 0.0));
  }

  auto idx4 = [n_spatial](std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s) -> std::size_t {
    return ((p * n_spatial + q) * n_spatial + r) * n_spatial + s;
  };

  // ─── One-body terms (with δ correction folded in for restricted) ──
  std::vector<double> h1_eff_alpha(n_spatial * n_spatial);
  std::vector<double> h1_eff_beta(n_spatial * n_spatial);
  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t s = 0; s < n_spatial; ++s) {
      double h_a = h1_alpha[p * n_spatial + s];
      double h_b = is_restricted ? h_a : h1_beta[p * n_spatial + s];
      if (is_restricted) {
        double delta_corr = 0.0;
        for (std::size_t q = 0; q < n_spatial; ++q) {
          delta_corr += eri_aaaa[idx4(p, q, q, s)];
        }
        h_a -= 0.5 * delta_corr;
        h_b = h_a;
      }
      h1_eff_alpha[p * n_spatial + s] = h_a;
      h1_eff_beta[p * n_spatial + s] = h_b;
    }
  }

  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t q = 0; q < n_spatial; ++q) {
      double h_pq_a = h1_eff_alpha[p * n_spatial + q];
      if (std::abs(h_pq_a) > integral_threshold) {
        accumulate_epq(mode_alpha(p), mode_alpha(q), h_pq_a);
      }
      double h_pq_b = h1_eff_beta[p * n_spatial + q];
      if (std::abs(h_pq_b) > integral_threshold) {
        accumulate_epq(mode_beta(p), mode_beta(q), h_pq_b);
      }
    }
  }

  // ─── Two-body terms ───────────────────────────────────────────────

  // Precompute Majorana pair products in packed form (same-spin only).
  // Each entry stores the Pauli multiply phase index AND the combined
  // Majorana sign factor (phases[i] * phases[j]).
  struct PackedPairProduct {
    int phase;         // Pauli multiply phase index (0..3)
    std::int8_t sign;  // maj_phases[i] * maj_phases[j] (±1)
    PackedPauliWord<NW> word;
  };
  const std::size_t maj_per_spin = 2 * n_spatial;
  std::vector<PackedPairProduct> ppair_alpha(maj_per_spin * maj_per_spin);
  std::vector<PackedPairProduct> ppair_beta(maj_per_spin * maj_per_spin);

  for (std::size_t i = 0; i < maj_per_spin; ++i) {
    for (std::size_t j = 0; j < maj_per_spin; ++j) {
      // alpha block: Majorana indices i, j
      auto [ph_a, w_a] = multiply_packed(packed_mapping[i], packed_mapping[j]);
      std::int8_t sign_a = has_neg_phases ? maj_phases[i] * maj_phases[j]
                                          : static_cast<std::int8_t>(1);
      ppair_alpha[i * maj_per_spin + j] = {ph_a, sign_a, std::move(w_a)};

      // beta block: Majorana indices i+2*n_spatial, j+2*n_spatial
      auto [ph_b, w_b] = multiply_packed(packed_mapping[i + maj_per_spin],
                                         packed_mapping[j + maj_per_spin]);
      std::int8_t sign_b = has_neg_phases ? maj_phases[i + maj_per_spin] *
                                                maj_phases[j + maj_per_spin]
                                          : static_cast<std::int8_t>(1);
      ppair_beta[i * maj_per_spin + j] = {ph_b, sign_b, std::move(w_b)};
    }
  }

  auto alpha_pair = [&](std::size_t i,
                        std::size_t j) -> const PackedPairProduct& {
    return ppair_alpha[i * maj_per_spin + j];
  };
  auto beta_pair = [&](std::size_t i,
                       std::size_t j) -> const PackedPairProduct& {
    return ppair_beta[i * maj_per_spin + j];
  };

  if (is_restricted) {
    // Precompute spin-summed E_pq for all (p,q):
    struct SpinSummedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord<NW>>> terms;
    };
    std::vector<SpinSummedE> ss_e(n_spatial * n_spatial);

    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto& sse = ss_e[p * n_spatial + q];
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const auto& [phase_a, sign_a, word_a] =
                alpha_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(
                apply_phase(phase_a,
                            static_cast<double>(sign_a) * kQuarter * kC[a][b]),
                word_a);
            const auto& [phase_b, sign_b, word_b] =
                beta_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(
                apply_phase(phase_b,
                            static_cast<double>(sign_b) * kQuarter * kC[a][b]),
                word_b);
          }
        }
      }
    }

    // Precompute symmetrized S_pq = E_pq + E_qp (merged) for p ≤ q.
    struct SymmetrizedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord<NW>>> terms;
    };
    std::vector<SymmetrizedE> sym_e;
    std::vector<std::size_t> sym_map(n_spatial * n_spatial);
    {
      std::size_t idx = 0;
      sym_e.resize(n_spatial * (n_spatial + 1) / 2);
      for (std::size_t p = 0; p < n_spatial; ++p) {
        for (std::size_t q = p; q < n_spatial; ++q) {
          sym_map[p * n_spatial + q] = idx;
          if (p != q) sym_map[q * n_spatial + p] = idx;

          std::unordered_map<PackedPauliWord<NW>, std::complex<double>,
                             PackedPauliWordHash<NW>>
              merged;
          for (const auto& [c, w] : ss_e[p * n_spatial + q].terms) {
            merged[w] += c;
          }
          if (p != q) {
            for (const auto& [c, w] : ss_e[q * n_spatial + p].terms) {
              merged[w] += c;
            }
          }
          auto& se = sym_e[idx];
          for (auto& [w, c] : merged) {
            if (std::abs(c) > 1e-15) {
              se.terms.emplace_back(c, std::move(w));
            }
          }
          ++idx;
        }
      }
    }

    // Two-body product: exploit full 8-fold ERI symmetry.
    // Already using p≤q, r≤s (4-fold). Now add (pq)≤(rs) exchange:
    // (pq|rs) = (rs|pq), so S_pq·S_rs + S_rs·S_pq covers both.
    // For pq_idx < rs_idx: accumulate both products with scale 2×½·eri.
    // For pq_idx == rs_idx: accumulate one product with scale ½·eri.
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = p; q < n_spatial; ++q) {
        std::size_t pq_idx = sym_map[p * n_spatial + q];
        const auto& s_pq = sym_e[pq_idx];
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = r; s < n_spatial; ++s) {
            std::size_t rs_idx = sym_map[r * n_spatial + s];
            if (pq_idx > rs_idx) continue;  // skip; (rs,pq) handles this

            double eri = eri_aaaa[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            const auto& s_rs = sym_e[rs_idx];

            if (pq_idx == rs_idx) {
              // Diagonal: S_pq · S_pq (single product)
              double half_eri = 0.5 * eri;
              for (const auto& [c1, w1] : s_pq.terms) {
                for (const auto& [c2, w2] : s_rs.terms) {
                  acc.accumulate_product(w1, w2, half_eri * c1 * c2);
                }
              }
            } else {
              // Off-diagonal: S_pq · S_rs + S_rs · S_pq (both products)
              double half_eri = 0.5 * eri;
              for (const auto& [c1, w1] : s_pq.terms) {
                for (const auto& [c2, w2] : s_rs.terms) {
                  acc.accumulate_product(w1, w2, half_eri * c1 * c2);
                  acc.accumulate_product(w2, w1, half_eri * c2 * c1);
                }
              }
            }
          }
        }
      }
    }

    // (δ correction is folded into the one-body integrals above)
  } else {
    // Unrestricted: explicit spin-channel ERIs using packed types

    auto accumulate_two_body_product =
        [&](std::size_t mode_p, std::size_t mode_q, std::size_t mode_r,
            std::size_t mode_s, double eri) {
          bool pq_is_alpha = mode_p < n_spatial;
          bool rs_is_alpha = mode_r < n_spatial;
          auto& cache_pq = pq_is_alpha ? ppair_alpha : ppair_beta;
          auto& cache_rs = rs_is_alpha ? ppair_alpha : ppair_beta;
          std::size_t bp = pq_is_alpha ? mode_p : (mode_p - n_spatial);
          std::size_t bq = pq_is_alpha ? mode_q : (mode_q - n_spatial);
          std::size_t br = rs_is_alpha ? mode_r : (mode_r - n_spatial);
          std::size_t bs = rs_is_alpha ? mode_s : (mode_s - n_spatial);

          double half_eri = 0.5 * eri;
          for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
              const auto& [ph1, s1, w1] =
                  cache_pq[(2 * bp + a) * maj_per_spin + (2 * bq + b)];
              for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                  const auto& [ph2, s2, w2] =
                      cache_rs[(2 * br + c) * maj_per_spin + (2 * bs + d)];
                  double sign = static_cast<double>(s1) * s2;
                  std::complex<double> scale = apply_phase(
                      (ph1 + ph2) & 3,
                      sign * half_eri * kSixteenth * kC[a][b] * kC[c][d]);
                  acc.accumulate_product(w1, w2, scale);
                }
              }
            }
          }
        };

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

    // aabb channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aabb[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_beta(r), mode_beta(s), eri);
          }
        }
      }
    }

    // bbaa channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = eri_aabb[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;
            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_alpha(r), mode_alpha(s), eri);
          }
        }
      }
    }
  }

  // ─── Extract results directly as strings ────────────────────────
  auto terms = acc.get_terms_as_strings(threshold, num_qubits);

  MajoranaMapResult result;
  result.pauli_strings.reserve(terms.size());
  result.coefficients.reserve(terms.size());

  for (auto& [coeff, str] : terms) {
    result.pauli_strings.push_back(std::move(str));
    result.coefficients.push_back(coeff);
  }

  return result;
}

}  // anonymous namespace

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool is_restricted,
    double threshold, double integral_threshold) {
  const std::size_t num_qubits = mapping.num_qubits();
  if (num_qubits == 0) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: mapping has zero qubits; the encoding "
        "must produce at least one qubit. This usually indicates an "
        "uninitialized or malformed MajoranaMapping.");
  }
  const std::size_t num_words = (num_qubits + 63) / 64;

  // Dispatch to the appropriate template instantiation.
  // Each instantiation uses std::array<uint64_t, NW> (stack-allocated,
  // no heap overhead per Pauli word).
  switch (num_words) {
    case 1:
      return majorana_map_impl<1>(mapping, core_energy, h1_alpha, h1_beta,
                                  eri_aaaa, eri_aabb, eri_bbbb, n_spatial,
                                  is_restricted, threshold, integral_threshold);
    case 2:
      return majorana_map_impl<2>(mapping, core_energy, h1_alpha, h1_beta,
                                  eri_aaaa, eri_aabb, eri_bbbb, n_spatial,
                                  is_restricted, threshold, integral_threshold);
    case 3:
      return majorana_map_impl<3>(mapping, core_energy, h1_alpha, h1_beta,
                                  eri_aaaa, eri_aabb, eri_bbbb, n_spatial,
                                  is_restricted, threshold, integral_threshold);
    case 4:
      return majorana_map_impl<4>(mapping, core_energy, h1_alpha, h1_beta,
                                  eri_aaaa, eri_aabb, eri_bbbb, n_spatial,
                                  is_restricted, threshold, integral_threshold);
    default:
      throw std::invalid_argument(
          "majorana_map_hamiltonian: num_qubits=" + std::to_string(num_qubits) +
          " requires " + std::to_string(num_words) +
          " uint64 words, but max supported is 4 (256 qubits). "
          "Contact the developers to extend the template dispatch.");
  }
}

}  // namespace qdk::chemistry::data
