// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::data {
namespace detail {

// Majorana decomposition coefficients for the excitation operator a†_p a_q:
//   a†_p a_q = (1/4) Σ_{a,b} coeff[a][b] · γ_{2p+a} · γ_{2q+b}
//   coeff[a][b] = (-i)^a · (i)^b
constexpr std::complex<double> excitation_coeff[2][2] = {
    {{1.0, 0.0}, {0.0, 1.0}},
    {{0.0, -1.0}, {1.0, 0.0}},
};

// ─── Bitpacked Pauli representation ────────────────────────────────
//
// Represents a Pauli string as a pair of bitmasks (x_bits, z_bits)
// using the symplectic encoding:
//   I: x=0, z=0    X: x=1, z=0    Z: x=0, z=1    Y: x=1, z=1
//
// Each uint64_t covers 64 qubits. Templated on NW (number of uint64
// words) to keep everything on the stack. The engine dispatches to
// NW in {1, ..., 16} at runtime (covers up to 1024 qubits).

template <std::size_t NW>
struct PackedPauliWord {
  std::array<std::uint64_t, NW> x{};
  std::array<std::uint64_t, NW> z{};

  bool operator==(const PackedPauliWord& other) const = default;
};

template <std::size_t NW>
struct PackedPauliWordHash {
  std::size_t operator()(const PackedPauliWord<NW>& w) const noexcept {
    utils::HashContext ctx;
    hash_value(ctx, "packed_pauli_word");
    hash_value(ctx, static_cast<uint64_t>(NW));
    for (std::size_t i = 0; i < NW; ++i) {
      hash_value(ctx, static_cast<uint64_t>(w.x[i]));
    }
    for (std::size_t i = 0; i < NW; ++i) {
      hash_value(ctx, static_cast<uint64_t>(w.z[i]));
    }
    return ctx.hash_code();
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
      int bit = std::countr_zero(active);
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
    phase_exp += std::popcount(cyc);
    phase_exp -= std::popcount(anti);
  }
  // Pauli multiplication phases cycle mod 4:
  // 0: +1, 1: +i, 2: −1, 3: −i.
  return {phase_exp & 3, result};
}

/// Apply a phase index (0=+1, 1=+i, 2=-1, 3=-i) to a complex scale factor.
inline std::complex<double> apply_phase(int phase_idx,
                                        std::complex<double> scale) {
  switch (phase_idx & 3) {
    case 0:
      return scale;
    case 1:
      return {-scale.imag(), scale.real()};
    case 2:
      return {-scale.real(), -scale.imag()};
    case 3:
      return {scale.imag(), -scale.real()};
  }
  return scale;  // unreachable
}

template <std::size_t NW>
class PackedAccumulator {
 public:
  void accumulate(const PackedPauliWord<NW>& word, std::complex<double> coeff) {
    terms_[word] += coeff;
  }

  void accumulate_product(const PackedPauliWord<NW>& w1,
                          const PackedPauliWord<NW>& w2,
                          std::complex<double> scale) {
    auto [phase_idx, word] = multiply_packed(w1, w2);
    terms_[word] += apply_phase(phase_idx, scale);
  }

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
  std::unordered_map<PackedPauliWord<NW>, std::complex<double>,
                     PackedPauliWordHash<NW>>
      terms_;
};

// ─── Engine implementation ─────────────────────────────────────────
//
// The standard non-relativistic electronic Hamiltonian is spin-free: it
// contains no spin operators.  The Majorana decomposition of excitation
// operators E_pq = a†_p a_q into bilinear products of γ's works for
// any spin-free Hamiltonian regardless of the orbital basis.
//
// The spin_symmetric flag selects one of two evaluation strategies:
//
//   spin_symmetric = true   (restricted orbitals)
//     All spin channels share the same integrals (h_alpha == h_beta,
//     eri_aaaa == eri_bbbb == eri_aabb).  The engine precomputes
//     spin-summed E_pq = E^α_pq + E^β_pq and exploits 8-fold ERI
//     symmetry, roughly halving the two-body work.  The δ_{qr}
//     two-body contraction is folded into the one-body integrals.
//
//   spin_symmetric = false  (unrestricted orbitals)
//     Each spin channel (αα, ββ, αβ, βα) is handled independently
//     with its own integrals.  This is needed for unrestricted orbitals,
//     where h^α ≠ h^β and the ERI channels differ — even though the
//     underlying Hamiltonian is still spin-free.  The αβ and βα
//     cross-spin channels are related by Coulomb symmetry (pq|rs) =
//     (rs|pq) and are handled in a single merged loop.

inline std::size_t idx4(std::size_t n_spatial, std::size_t p, std::size_t q,
                        std::size_t r, std::size_t s) {
  return ((p * n_spatial + q) * n_spatial + r) * n_spatial + s;
}

inline std::size_t canonical_pair_index(std::size_t n_spatial, std::size_t p,
                                        std::size_t q) {
  if (p > q) std::swap(p, q);
  return p * n_spatial - (p * (p - 1)) / 2 + (q - p);
}

struct DenseEriProvider {
  const double* eri_aaaa;
  const double* eri_aabb;
  const double* eri_bbbb;
  std::size_t n_spatial;
  std::size_t n2;

  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return eri_aaaa[idx4(n_spatial, p, q, r, s)];
  }

  std::span<const double> row_aaaa(std::size_t p, std::size_t q) const {
    return {eri_aaaa + (p * n_spatial + q) * n2, n2};
  }

  std::span<const double> row_aabb(std::size_t p, std::size_t q) const {
    return {eri_aabb + (p * n_spatial + q) * n2, n2};
  }

  std::span<const double> row_bbbb(std::size_t p, std::size_t q) const {
    return {eri_bbbb + (p * n_spatial + q) * n2, n2};
  }

  template <typename Callback>
  void for_each_restricted_eri(const std::vector<std::size_t>& sym_map,
                               double integral_threshold,
                               Callback&& callback) const {
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = p; q < n_spatial; ++q) {
        std::size_t pq_idx = sym_map[p * n_spatial + q];
        auto row = row_aaaa(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = r; s < n_spatial; ++s) {
            std::size_t rs_idx = sym_map[r * n_spatial + s];
            if (pq_idx > rs_idx) continue;

            double eri = row[r * n_spatial + s];
            if (std::abs(eri) < integral_threshold) continue;
            callback(p, q, r, s, pq_idx, rs_idx, eri);
          }
        }
      }
    }
  }
};

struct CholeskyEriProvider {
  const Eigen::MatrixXd& l_alpha;
  const Eigen::MatrixXd& l_beta;
  std::size_t n_spatial;
  std::size_t n2;
  std::size_t naux;

  double dot_rows(const Eigen::MatrixXd& left, std::size_t left_row,
                  const Eigen::MatrixXd& right, std::size_t right_row) const {
    double value = 0.0;
    for (std::size_t aux = 0; aux < naux; ++aux) {
      value += left(left_row, aux) * right(right_row, aux);
    }
    return value;
  }

  std::vector<double> build_row(const Eigen::MatrixXd& left,
                                const Eigen::MatrixXd& right, std::size_t p,
                                std::size_t q) const {
    std::vector<double> row(n2, 0.0);
    const std::size_t pq = p * n_spatial + q;
    for (std::size_t aux = 0; aux < naux; ++aux) {
      const double scale = left(pq, aux);
      if (scale == 0.0) continue;
      for (std::size_t rs = 0; rs < n2; ++rs) {
        row[rs] += scale * right(rs, aux);
      }
    }
    return row;
  }

  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return dot_rows(l_alpha, p * n_spatial + q, l_alpha, r * n_spatial + s);
  }

  std::vector<double> row_aaaa(std::size_t p, std::size_t q) const {
    return build_row(l_alpha, l_alpha, p, q);
  }

  std::vector<double> row_aabb(std::size_t p, std::size_t q) const {
    return build_row(l_alpha, l_beta, p, q);
  }

  std::vector<double> row_bbbb(std::size_t p, std::size_t q) const {
    return build_row(l_beta, l_beta, p, q);
  }

  template <typename Callback>
  void for_each_restricted_eri(const std::vector<std::size_t>& sym_map,
                               double integral_threshold,
                               Callback&& callback) const {
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = p; q < n_spatial; ++q) {
        std::size_t pq_idx = sym_map[p * n_spatial + q];
        auto row = row_aaaa(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = r; s < n_spatial; ++s) {
            std::size_t rs_idx = sym_map[r * n_spatial + s];
            if (pq_idx > rs_idx) continue;

            double eri = row[r * n_spatial + s];
            if (std::abs(eri) < integral_threshold) continue;
            callback(p, q, r, s, pq_idx, rs_idx, eri);
          }
        }
      }
    }
  }
};

struct SparseEriProvider {
  struct Key {
    std::size_t p;
    std::size_t q;
    std::size_t r;
    std::size_t s;

    friend bool operator<(const Key& left, const Key& right) {
      return std::tie(left.p, left.q, left.r, left.s) <
             std::tie(right.p, right.q, right.r, right.s);
    }
  };

  std::size_t n_spatial;
  std::size_t n2;
  std::map<Key, double> entries;

  SparseEriProvider(std::size_t n,
                    const SparseHamiltonianContainer::TwoBodyMap& values)
      : n_spatial(n), n2(n * n) {
    for (const auto& [indices, value] : values) {
      auto [p, q, r, s] = indices;
      Key key = canonicalize_checked(p, q, r, s);
      auto [it, inserted] = entries.emplace(key, value);
      if (!inserted) {
        double scale = std::max({1.0, std::abs(it->second), std::abs(value)});
        if (std::abs(it->second - value) > 1e-12 * scale) {
          throw std::invalid_argument(
              "Sparse two-body integrals contain inconsistent values for "
              "symmetry-equivalent indices.");
        }
      }
    }
  }

  Key canonicalize_checked(int p, int q, int r, int s) const {
    auto checked = [&](int index) -> std::size_t {
      if (index < 0 || static_cast<std::size_t>(index) >= n_spatial) {
        throw std::invalid_argument(
            "Sparse two-body integral index is outside the active-space "
            "orbital range.");
      }
      return static_cast<std::size_t>(index);
    };
    return canonicalize(checked(p), checked(q), checked(r), checked(s));
  }

  Key canonicalize(std::size_t p, std::size_t q, std::size_t r,
                   std::size_t s) const {
    if (p > q) std::swap(p, q);
    if (r > s) std::swap(r, s);
    if (canonical_pair_index(n_spatial, p, q) >
        canonical_pair_index(n_spatial, r, s)) {
      std::swap(p, r);
      std::swap(q, s);
    }
    return {p, q, r, s};
  }

  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    auto it = entries.find(canonicalize(p, q, r, s));
    return it == entries.end() ? 0.0 : it->second;
  }

  std::vector<double> row_aaaa(std::size_t p, std::size_t q) const {
    std::vector<double> row(n2, 0.0);
    for (std::size_t r = 0; r < n_spatial; ++r) {
      for (std::size_t s = 0; s < n_spatial; ++s) {
        row[r * n_spatial + s] = aaaa(p, q, r, s);
      }
    }
    return row;
  }

  std::vector<double> row_aabb(std::size_t p, std::size_t q) const {
    return row_aaaa(p, q);
  }

  std::vector<double> row_bbbb(std::size_t p, std::size_t q) const {
    return row_aaaa(p, q);
  }

  template <typename Callback>
  void for_each_restricted_eri(const std::vector<std::size_t>& sym_map,
                               double integral_threshold,
                               Callback&& callback) const {
    for (const auto& [key, eri] : entries) {
      if (std::abs(eri) < integral_threshold) continue;
      std::size_t pq_idx = sym_map[key.p * n_spatial + key.q];
      std::size_t rs_idx = sym_map[key.r * n_spatial + key.s];
      callback(key.p, key.q, key.r, key.s, pq_idx, rs_idx, eri);
    }
  }
};

template <std::size_t NW, typename EriProvider>
MajoranaMapResult majorana_map_impl(const MajoranaMapping& mapping,
                                    double core_energy, const double* h1_alpha,
                                    const double* h1_beta,
                                    const EriProvider& eri_provider,
                                    std::size_t n_spatial, bool spin_symmetric,
                                    double threshold,
                                    double integral_threshold) {
  const std::size_t n_modes = 2 * n_spatial;

  PackedAccumulator<NW> acc;

  // ─── Pair product cache ───────────────────────────────────────────
  //
  // Precompute γ_j · γ_k = (-i) · bilinear(j, k) for each within-spin-block
  // pair. Using bilinear() as the access primitive makes this work for both
  // Majorana-atomic and bilinear-only mappings.

  struct PackedPairProduct {
    std::complex<double> coeff;
    PackedPauliWord<NW> word;
  };
  const std::size_t maj_per_spin = 2 * n_spatial;
  const std::size_t alpha_offset = 0;
  const std::size_t beta_offset = maj_per_spin;

  auto build_pair_cache =
      [&](std::size_t global_offset) -> std::vector<PackedPairProduct> {
    std::vector<PackedPairProduct> cache(maj_per_spin * maj_per_spin);
    for (std::size_t i = 0; i < maj_per_spin; ++i) {
      cache[i * maj_per_spin + i] = {{1.0, 0.0}, PackedPauliWord<NW>{}};
      for (std::size_t j = 0; j < maj_per_spin; ++j) {
        if (i == j) continue;
        auto [bl_coeff, bl_word] =
            mapping.bilinear(i + global_offset, j + global_offset);
        cache[i * maj_per_spin + j] = {
            std::complex<double>(0.0, -1.0) * bl_coeff,
            sparse_to_packed<NW>(bl_word)};
      }
    }
    return cache;
  };

  auto ppair_alpha = build_pair_cache(alpha_offset);
  auto ppair_beta = build_pair_cache(beta_offset);

  auto alpha_pair = [&](std::size_t i,
                        std::size_t j) -> const PackedPairProduct& {
    return ppair_alpha[i * maj_per_spin + j];
  };
  auto beta_pair = [&](std::size_t i,
                       std::size_t j) -> const PackedPairProduct& {
    return ppair_beta[i * maj_per_spin + j];
  };

  auto mode_alpha = [](std::size_t p) -> std::size_t { return p; };
  auto mode_beta = [n_spatial](std::size_t p) -> std::size_t {
    return p + n_spatial;
  };

  auto accumulate_epq = [&](std::size_t mode_p, std::size_t mode_q,
                            double h_pq) {
    bool is_alpha = mode_p < n_spatial;
    auto& cache = is_alpha ? ppair_alpha : ppair_beta;
    std::size_t bp = is_alpha ? mode_p : (mode_p - n_spatial);
    std::size_t bq = is_alpha ? mode_q : (mode_q - n_spatial);
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        const auto& [coeff, word] =
            cache[(2 * bp + a) * maj_per_spin + (2 * bq + b)];
        acc.accumulate(word, coeff * h_pq * 0.25 * excitation_coeff[a][b]);
      }
    }
  };

  // ─── Core energy ──────────────────────────────────────────────────
  if (std::abs(core_energy) > integral_threshold) {
    PackedPauliWord<NW> identity{};
    acc.accumulate(identity, std::complex<double>(core_energy, 0.0));
  }

  // ─── One-body terms ──────────────────────────────────────────────
  std::vector<double> h1_eff_alpha(n_spatial * n_spatial);
  std::vector<double> h1_eff_beta(n_spatial * n_spatial);
  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t s = 0; s < n_spatial; ++s) {
      double h_a = h1_alpha[p * n_spatial + s];
      double h_b = spin_symmetric ? h_a : h1_beta[p * n_spatial + s];
      if (spin_symmetric) {
        double delta_corr = 0.0;
        for (std::size_t q = 0; q < n_spatial; ++q) {
          delta_corr += eri_provider.aaaa(p, q, q, s);
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

  if (spin_symmetric) {
    struct SpinSummedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord<NW>>> terms;
    };
    std::vector<SpinSummedE> ss_e(n_spatial * n_spatial);

    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto& sse = ss_e[p * n_spatial + q];
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const auto& [coeff_a, word_a] = alpha_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(coeff_a * 0.25 * excitation_coeff[a][b],
                                   word_a);
            const auto& [coeff_b, word_b] = beta_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(coeff_b * 0.25 * excitation_coeff[a][b],
                                   word_b);
          }
        }
      }
    }

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

    eri_provider.for_each_restricted_eri(
        sym_map, integral_threshold,
        [&](std::size_t, std::size_t, std::size_t, std::size_t,
            std::size_t pq_idx, std::size_t rs_idx, double eri) {
          const auto& s_pq = sym_e[pq_idx];
          const auto& s_rs = sym_e[rs_idx];

          if (pq_idx == rs_idx) {
            double half_eri = 0.5 * eri;
            for (const auto& [c1, w1] : s_pq.terms) {
              for (const auto& [c2, w2] : s_rs.terms) {
                acc.accumulate_product(w1, w2, half_eri * c1 * c2);
              }
            }
          } else {
            double half_eri = 0.5 * eri;
            for (const auto& [c1, w1] : s_pq.terms) {
              for (const auto& [c2, w2] : s_rs.terms) {
                acc.accumulate_product(w1, w2, half_eri * c1 * c2);
                acc.accumulate_product(w2, w1, half_eri * c2 * c1);
              }
            }
          }
        });

  } else {
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
              const auto& [coeff1, w1] =
                  cache_pq[(2 * bp + a) * maj_per_spin + (2 * bq + b)];
              for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                  const auto& [coeff2, w2] =
                      cache_rs[(2 * br + c) * maj_per_spin + (2 * bs + d)];
                  std::complex<double> scale = coeff1 * coeff2 * half_eri *
                                               0.0625 * excitation_coeff[a][b] *
                                               excitation_coeff[c][d];
                  acc.accumulate_product(w1, w2, scale);
                }
              }
            }
          }
        };

    // αα channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto row = eri_provider.row_aaaa(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = row[r * n_spatial + s];
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

    // ββ channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto row = eri_provider.row_bbbb(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = row[r * n_spatial + s];
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

    // αβ + βα cross-spin channels, related by Coulomb symmetry (pq|rs)=(rs|pq)
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto row = eri_provider.row_aabb(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = row[r * n_spatial + s];
            if (std::abs(eri) < integral_threshold) continue;
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_beta(r), mode_beta(s), eri);
            accumulate_two_body_product(mode_beta(r), mode_beta(s),
                                        mode_alpha(p), mode_alpha(q), eri);
          }
        }
      }
    }
  }

  // ─── Extract results as sparse Pauli words ──────────────────────
  auto terms = acc.get_terms(threshold);

  MajoranaMapResult result;
  result.words.reserve(terms.size());
  result.coefficients.reserve(terms.size());

  for (auto& [coeff, word] : terms) {
    result.words.push_back(std::move(word));
    result.coefficients.push_back(coeff);
  }

  return result;
}

// Dispatch table: function pointer per NW, covering 1..16 (up to 1024 qubits).
template <typename EriProvider>
using DispatchFn = MajoranaMapResult (*)(const MajoranaMapping&, double,
                                         const double*, const double*,
                                         const EriProvider&, std::size_t, bool,
                                         double, double);

template <typename EriProvider, std::size_t... Is>
constexpr std::array<DispatchFn<EriProvider>, sizeof...(Is)>
make_dispatch_table(std::index_sequence<Is...>) {
  return {{&majorana_map_impl<Is + 1, EriProvider>...}};
}

constexpr std::size_t max_nw = 16;

template <typename EriProvider>
MajoranaMapResult dispatch_majorana_map(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const EriProvider& eri_provider,
    std::size_t n_spatial, bool spin_symmetric, double threshold,
    double integral_threshold) {
  const std::size_t num_qubits = mapping.num_qubits();
  if (num_qubits == 0) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: mapping has zero qubits; the encoding "
        "must produce at least one qubit.");
  }
  const std::size_t num_words = (num_qubits + 63) / 64;

  if (num_words > max_nw) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: num_qubits=" + std::to_string(num_qubits) +
        " exceeds the maximum of " + std::to_string(max_nw * 64) + " qubits.");
  }

  static const auto table =
      make_dispatch_table<EriProvider>(std::make_index_sequence<max_nw>{});
  return table[num_words - 1](mapping, core_energy, h1_alpha, h1_beta,
                              eri_provider, n_spatial, spin_symmetric,
                              threshold, integral_threshold);
}

}  // namespace detail

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold) {
  detail::DenseEriProvider provider{eri_aaaa, eri_aabb, eri_bbbb, n_spatial,
                                    n_spatial * n_spatial};
  return detail::dispatch_majorana_map(mapping, core_energy, h1_alpha, h1_beta,
                                       provider, n_spatial, spin_symmetric,
                                       threshold, integral_threshold);
}

MajoranaMapResult majorana_map_hamiltonian(const MajoranaMapping& mapping,
                                           const Hamiltonian& hamiltonian,
                                           double threshold,
                                           double integral_threshold) {
  auto [h1_alpha, h1_beta] = hamiltonian.get_one_body_integrals();
  if (h1_alpha.rows() != h1_alpha.cols()) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: alpha one-body integrals must be square.");
  }

  const std::size_t n_spatial = static_cast<std::size_t>(h1_alpha.rows());
  const bool spin_symmetric = hamiltonian.is_restricted();

  std::vector<double> h1_alpha_flat(n_spatial * n_spatial);
  std::vector<double> h1_beta_flat(n_spatial * n_spatial);
  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t q = 0; q < n_spatial; ++q) {
      h1_alpha_flat[p * n_spatial + q] = h1_alpha(p, q);
      h1_beta_flat[p * n_spatial + q] =
          spin_symmetric ? h1_alpha(p, q) : h1_beta(p, q);
    }
  }

  constexpr double excluded_core_energy = 0.0;

  if (hamiltonian.has_container_type<SparseHamiltonianContainer>()) {
    if (!spin_symmetric) {
      throw std::invalid_argument(
          "majorana_map_hamiltonian: sparse container mapping currently "
          "requires restricted orbitals.");
    }
    const auto& sparse =
        hamiltonian.get_container<SparseHamiltonianContainer>();
    detail::SparseEriProvider provider(n_spatial,
                                       sparse.sparse_two_body_integrals());
    return detail::dispatch_majorana_map(
        mapping, excluded_core_energy, h1_alpha_flat.data(),
        h1_beta_flat.data(), provider, n_spatial, spin_symmetric, threshold,
        integral_threshold);
  }

  if (hamiltonian.has_container_type<CholeskyHamiltonianContainer>()) {
    const auto& cholesky =
        hamiltonian.get_container<CholeskyHamiltonianContainer>();
    auto [l_alpha, l_beta] = cholesky.get_three_center_integrals();
    if (static_cast<std::size_t>(l_alpha.rows()) != n_spatial * n_spatial) {
      throw std::invalid_argument(
          "majorana_map_hamiltonian: Cholesky three-center rows must equal "
          "n_spatial^2.");
    }
    detail::CholeskyEriProvider provider{
        l_alpha, l_beta, n_spatial, n_spatial * n_spatial,
        static_cast<std::size_t>(l_alpha.cols())};
    return detail::dispatch_majorana_map(
        mapping, excluded_core_energy, h1_alpha_flat.data(),
        h1_beta_flat.data(), provider, n_spatial, spin_symmetric, threshold,
        integral_threshold);
  }

  auto [eri_aaaa, eri_aabb, eri_bbbb] = hamiltonian.get_two_body_integrals();
  return majorana_map_hamiltonian(
      mapping, excluded_core_energy, h1_alpha_flat.data(), h1_beta_flat.data(),
      eri_aaaa.data(), eri_aabb.data(), eri_bbbb.data(), n_spatial,
      spin_symmetric, threshold, integral_threshold);
}

}  // namespace qdk::chemistry::data
