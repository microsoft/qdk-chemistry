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
// Each uint64_t covers 64 qubits. A vector of uint64_t scales to
// arbitrary qubit counts for FTQC applications.
//
// Multiplication is O(num_words) via XOR + popcount for phase:
//   (x1,z1) * (x2,z2) = phase * (x1^x2, z1^z2)
//   where phase = i^{2 * popcount(x1 & z2)} * (-1)^{popcount(z1 & x2 & ~(x1&z2))}
//   (simplified: see multiply implementation below)

struct PackedPauliWord {
  std::vector<std::uint64_t> x;  // X-component bitmask
  std::vector<std::uint64_t> z;  // Z-component bitmask

  bool operator==(const PackedPauliWord& other) const = default;
};

struct PackedPauliWordHash {
  std::size_t operator()(const PackedPauliWord& w) const noexcept {
    std::size_t seed = w.x.size();
    for (auto v : w.x) seed ^= v * 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    for (auto v : w.z) seed ^= v * 0x517cc1b727220a95ULL + (seed << 6) + (seed >> 2);
    return seed;
  }
};

/// Convert SparsePauliWord to packed form.
PackedPauliWord sparse_to_packed(const SparsePauliWord& word,
                                 std::size_t num_words) {
  PackedPauliWord pw;
  pw.x.resize(num_words, 0);
  pw.z.resize(num_words, 0);
  for (const auto& [qubit, op] : word) {
    std::size_t wi = qubit / 64;
    std::uint64_t bit = std::uint64_t(1) << (qubit % 64);
    if (wi < num_words) {
      if (op == 1 || op == 2) pw.x[wi] |= bit;  // X or Y
      if (op == 2 || op == 3) pw.z[wi] |= bit;  // Y or Z
    }
  }
  return pw;
}

/// Convert packed form back to SparsePauliWord.
SparsePauliWord packed_to_sparse(const PackedPauliWord& pw) {
  SparsePauliWord word;
  for (std::size_t wi = 0; wi < pw.x.size(); ++wi) {
    std::uint64_t x = pw.x[wi];
    std::uint64_t z = pw.z[wi];
    std::uint64_t active = x | z;
    while (active) {
      int bit = __builtin_ctzll(active);
      std::uint64_t mask = std::uint64_t(1) << bit;
      std::uint64_t qubit = wi * 64 + bit;
      bool has_x = (x & mask) != 0;
      bool has_z = (z & mask) != 0;
      std::uint8_t op = has_x ? (has_z ? 2 : 1) : 3;  // Y:2, X:1, Z:3
      word.emplace_back(qubit, op);
      active &= ~mask;
    }
  }
  return word;
}

/// Popcount for a vector of uint64_t words.
inline int vec_popcount(const std::vector<std::uint64_t>& v) {
  int count = 0;
  for (auto w : v) count += __builtin_popcountll(w);
  return count;
}

/// Multiply two PackedPauliWords: result = pw1 * pw2.
/// Returns (phase, product) where phase is ±1 or ±i.
///
/// Using the symplectic formula:
///   P1 * P2 = i^{phase_exp} * P_result
///   P_result.x = P1.x ^ P2.x,  P_result.z = P1.z ^ P2.z
///   phase_exp = 2*popcount(P1.x & P2.z) - 2*popcount(P1.z & P2.x)  (mod 4)
///   but we need the FULL Pauli algebra phase, which is:
///   For each qubit: op1 * op2 gives a phase from the multiplication table.
///
/// Efficient formula (from Dehaene & De Moor 2003):
///   phase_exp = Σ_j f(P1_j, P2_j) mod 4
///   where f(a,b) for symplectic vectors encodes the single-qubit phase.
///
/// We use the direct computation: for each qubit where both operators are
/// non-identity, look up the phase from {X,Y,Z} × {X,Y,Z}.
///
/// Actually, the most efficient approach uses the identity:
///   phase = i^{2·popcount(x1&z2)} · (-1)^{popcount(z1&x2)}
///         · (-1)^{popcount(x1&x2&z1&z2)}
///   (this follows from the qubit-by-qubit Pauli multiplication table)
std::pair<std::complex<double>, PackedPauliWord> multiply_packed(
    const PackedPauliWord& p1, const PackedPauliWord& p2) {
  std::size_t nw = p1.x.size();
  PackedPauliWord result;
  result.x.resize(nw);
  result.z.resize(nw);

  // Per-qubit phase from Pauli multiplication table, vectorized over 64 bits.
  // Cyclic pairs (XY, YZ, ZX) contribute +1; anti-cyclic (YX, ZY, XZ) give -1.
  int phase_exp = 0;
  for (std::size_t i = 0; i < nw; ++i) {
    std::uint64_t x1 = p1.x[i], z1 = p1.z[i];
    std::uint64_t x2 = p2.x[i], z2 = p2.z[i];

    result.x[i] = x1 ^ x2;
    result.z[i] = z1 ^ z2;

    std::uint64_t nx1 = ~x1, nz1 = ~z1, nx2 = ~x2, nz2 = ~z2;
    std::uint64_t cyc = (x1 & nz1 & x2 & z2) |   // XY
                        (x1 & z1 & nx2 & z2) |    // YZ
                        (nx1 & z1 & x2 & nz2);    // ZX
    std::uint64_t anti = (x1 & z1 & x2 & nz2) |   // YX
                         (nx1 & z1 & x2 & z2) |    // ZY
                         (x1 & nz1 & nx2 & z2);    // XZ
    phase_exp += __builtin_popcountll(cyc);
    phase_exp -= __builtin_popcountll(anti);
  }

  static constexpr std::complex<double> powers_of_i[4] = {
      {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  int mod = ((phase_exp % 4) + 4) % 4;
  return {powers_of_i[mod], result};
}

/// Bitpacked term accumulator: maps PackedPauliWord → complex coefficient.
class PackedAccumulator {
 public:
  void accumulate(const PackedPauliWord& word, std::complex<double> coeff) {
    auto it = terms_.find(word);
    if (it != terms_.end()) {
      it->second += coeff;
    } else {
      terms_[word] = coeff;
    }
  }

  void accumulate_product(const PackedPauliWord& w1,
                          const PackedPauliWord& w2,
                          std::complex<double> scale) {
    auto [phase, word] = multiply_packed(w1, w2);
    accumulate(word, scale * phase);
  }

  /// Extract terms above threshold, converting back to SparsePauliWord.
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> get_terms(
      double threshold) const {
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> result;
    result.reserve(terms_.size());
    for (const auto& [pw, coeff] : terms_) {
      if (std::abs(coeff) >= threshold) {
        result.emplace_back(coeff, packed_to_sparse(pw));
      }
    }
    return result;
  }

 private:
  std::unordered_map<PackedPauliWord, std::complex<double>,
                     PackedPauliWordHash>
      terms_;
};

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
  const std::size_t num_qubits = mapping.num_qubits();
  const std::size_t num_words = (num_qubits + 63) / 64;

  // Use bitpacked accumulator for O(1) hash/compare/multiply
  PackedAccumulator acc;

  // Convert all Majorana table entries to packed form
  std::vector<PackedPauliWord> packed_mapping(2 * n_modes);
  for (std::size_t k = 0; k < 2 * n_modes; ++k) {
    packed_mapping[k] = sparse_to_packed(mapping(k), num_words);
  }

  auto mode_alpha = [](std::size_t p) -> std::size_t { return p; };
  auto mode_beta = [n_spatial](std::size_t p) -> std::size_t {
    return p + n_spatial;
  };

  // Helper: accumulate one-body E_pq for a mode pair, using packed types
  auto accumulate_epq = [&](std::size_t mode_p, std::size_t mode_q,
                            double h_pq) {
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        auto [phase, word] = multiply_packed(
            packed_mapping[2 * mode_p + a], packed_mapping[2 * mode_q + b]);
        acc.accumulate(word, h_pq * kQuarter * kC[a][b] * phase);
      }
    }
  };

  // ─── Core energy ──────────────────────────────────────────────────
  if (std::abs(core_energy) > integral_threshold) {
    PackedPauliWord identity;
    identity.x.resize(num_words, 0);
    identity.z.resize(num_words, 0);
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
  struct PackedPairProduct {
    std::complex<double> phase;
    PackedPauliWord word;
  };
  const std::size_t maj_per_spin = 2 * n_spatial;
  std::vector<PackedPairProduct> ppair_alpha(maj_per_spin * maj_per_spin);
  std::vector<PackedPairProduct> ppair_beta(maj_per_spin * maj_per_spin);

  for (std::size_t i = 0; i < maj_per_spin; ++i) {
    for (std::size_t j = 0; j < maj_per_spin; ++j) {
      auto [ph_a, w_a] = multiply_packed(
          packed_mapping[i], packed_mapping[j]);
      ppair_alpha[i * maj_per_spin + j] = {ph_a, std::move(w_a)};

      auto [ph_b, w_b] = multiply_packed(
          packed_mapping[i + maj_per_spin],
          packed_mapping[j + maj_per_spin]);
      ppair_beta[i * maj_per_spin + j] = {ph_b, std::move(w_b)};
    }
  }

  auto alpha_pair = [&](std::size_t i, std::size_t j)
      -> const PackedPairProduct& {
    return ppair_alpha[i * maj_per_spin + j];
  };
  auto beta_pair = [&](std::size_t i, std::size_t j)
      -> const PackedPairProduct& {
    return ppair_beta[i * maj_per_spin + j];
  };

  if (is_restricted) {
    // Precompute spin-summed E_pq for all (p,q):
    struct SpinSummedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord>> terms;
    };
    std::vector<SpinSummedE> ss_e(n_spatial * n_spatial);

    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto& sse = ss_e[p * n_spatial + q];
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const auto& [phase_a, word_a] = alpha_pair(2*p+a, 2*q+b);
            sse.terms.emplace_back(kQuarter * kC[a][b] * phase_a, word_a);
            const auto& [phase_b, word_b] = beta_pair(2*p+a, 2*q+b);
            sse.terms.emplace_back(kQuarter * kC[a][b] * phase_b, word_b);
          }
        }
      }
    }

    // Precompute symmetrized S_pq = E_pq + E_qp (merged) for p ≤ q.
    struct SymmetrizedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord>> terms;
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

          std::unordered_map<PackedPauliWord, std::complex<double>,
                             PackedPauliWordHash> merged;
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

    // Two-body product: Σ_{p≤q, r≤s} (pq|rs) S_pq · S_rs
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = p; q < n_spatial; ++q) {
        const auto& s_pq = sym_e[sym_map[p * n_spatial + q]];
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = r; s < n_spatial; ++s) {
            double eri = eri_aaaa[idx4(p, q, r, s)];
            if (std::abs(eri) < integral_threshold) continue;

            double half_eri = 0.5 * eri;
            const auto& s_rs = sym_e[sym_map[r * n_spatial + s]];

            for (const auto& [c1, w1] : s_pq.terms) {
              for (const auto& [c2, w2] : s_rs.terms) {
                acc.accumulate_product(w1, w2, half_eri * c1 * c2);
              }
            }
          }
        }
      }
    }

    // (δ correction is folded into the one-body integrals above)
  } else {
    // Unrestricted: explicit spin-channel ERIs using packed types

    auto accumulate_two_body_product = [&](std::size_t mode_p,
                                           std::size_t mode_q,
                                           std::size_t mode_r,
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
          const auto& [ph1, w1] =
              cache_pq[(2*bp+a) * maj_per_spin + (2*bq+b)];
          for (int c = 0; c < 2; ++c) {
            for (int d = 0; d < 2; ++d) {
              const auto& [ph2, w2] =
                  cache_rs[(2*br+c) * maj_per_spin + (2*bs+d)];
              acc.accumulate_product(
                  w1, w2,
                  half_eri * kSixteenth * kC[a][b] * kC[c][d] * ph1 * ph2);
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

  // ─── Extract results ──────────────────────────────────────────────
  auto terms = acc.get_terms(threshold);

  MajoranaMapResult result;
  result.pauli_strings.reserve(terms.size());
  result.coefficients.reserve(terms.size());

  for (auto& [coeff, word] : terms) {
    result.pauli_strings.push_back(
        sparse_to_le_string(word, num_qubits));
    result.coefficients.push_back(coeff);
  }

  return result;
}

}  // namespace qdk::chemistry::data
