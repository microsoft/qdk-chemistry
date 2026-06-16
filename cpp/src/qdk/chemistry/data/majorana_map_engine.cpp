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
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

// ─── Two-body integral sources ──────────────────────────────────────
//
// The Pauli accumulation code below is shared across dense, sparse, and
// Cholesky-backed Hamiltonians. Each source presents the same scalar accessors
// and, when profitable, a sparse restricted iterator.

inline std::size_t idx4(std::size_t n_spatial, std::size_t p, std::size_t q,
                        std::size_t r, std::size_t s) {
  return ((p * n_spatial + q) * n_spatial + r) * n_spatial + s;
}

struct DenseTwoBodySource {
  static constexpr bool sparse_restricted_iteration = false;

  const double* eri_aaaa;
  const double* eri_aabb;
  const double* eri_bbbb;
  std::size_t n_spatial;

  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return eri_aaaa[idx4(n_spatial, p, q, r, s)];
  }

  double aabb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return eri_aabb[idx4(n_spatial, p, q, r, s)];
  }

  double bbbb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return eri_bbbb[idx4(n_spatial, p, q, r, s)];
  }
};

class CholeskyTwoBodySource {
 public:
  static constexpr bool sparse_restricted_iteration = false;

  CholeskyTwoBodySource(const CholeskyHamiltonianContainer& container,
                        std::size_t n_spatial)
      : n_spatial_(n_spatial) {
    const auto factors = container.get_three_center_integrals();
    const Eigen::MatrixXd& aa = factors.first;
    const Eigen::MatrixXd& bb = factors.second;
    const auto expected_rows =
        static_cast<Eigen::Index>(n_spatial_ * n_spatial_);
    if (aa.rows() != expected_rows || bb.rows() != expected_rows ||
        aa.cols() != bb.cols()) {
      throw std::invalid_argument(
          "majorana_map_hamiltonian: Cholesky three-center integrals must have "
          "shape [n_spatial^2, n_aux] for matching alpha and beta factors.");
    }
    aaaa_cache_ = std::make_unique<ContractedRowCache>(aa, aa);
    aabb_cache_ = std::make_unique<ContractedRowCache>(aa, bb);
    bbbb_cache_ = std::make_unique<ContractedRowCache>(bb, bb);
  }

  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aaaa_cache_->value(pair_index(p, q), pair_index(r, s));
  }

  double aabb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aabb_cache_->value(pair_index(p, q), pair_index(r, s));
  }

  double bbbb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return bbbb_cache_->value(pair_index(p, q), pair_index(r, s));
  }

 private:
  class ContractedRowCache {
   public:
    ContractedRowCache(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right)
        : left_(left), right_(right) {}

    double value(std::size_t left_row, std::size_t right_row) const {
      const auto& row = contracted_row(left_row);
      return row(static_cast<Eigen::Index>(right_row));
    }

   private:
    struct Entry {
      std::size_t left_row = 0;
      std::uint64_t stamp = 0;
      Eigen::VectorXd values;
    };

    static constexpr std::size_t max_cached_rows = 64;

    const Eigen::MatrixXd& left_;
    const Eigen::MatrixXd& right_;
    mutable std::vector<Entry> entries_;
    mutable std::uint64_t next_stamp_ = 0;

    const Eigen::VectorXd& contracted_row(std::size_t left_row) const {
      ++next_stamp_;
      for (auto& entry : entries_) {
        if (entry.left_row == left_row) {
          entry.stamp = next_stamp_;
          return entry.values;
        }
      }

      if (entries_.size() >= max_cached_rows) {
        auto victim = std::min_element(
            entries_.begin(), entries_.end(),
            [](const Entry& lhs, const Entry& rhs) {
              return lhs.stamp < rhs.stamp;
            });
        *victim = make_entry(left_row, next_stamp_);
        return victim->values;
      }

      entries_.push_back(make_entry(left_row, next_stamp_));
      return entries_.back().values;
    }

    Entry make_entry(std::size_t left_row, std::uint64_t stamp) const {
      Entry entry;
      entry.left_row = left_row;
      entry.stamp = stamp;
      entry.values =
          right_ * left_.row(static_cast<Eigen::Index>(left_row)).transpose();
      return entry;
    }
  };

  std::unique_ptr<ContractedRowCache> aaaa_cache_;
  std::unique_ptr<ContractedRowCache> aabb_cache_;
  std::unique_ptr<ContractedRowCache> bbbb_cache_;
  std::size_t n_spatial_ = 0;

  std::size_t pair_index(std::size_t p, std::size_t q) const {
    return p * n_spatial_ + q;
  }
};

class SparseTwoBodySource {
 public:
  static constexpr bool sparse_restricted_iteration = true;

  explicit SparseTwoBodySource(const SparseHamiltonianContainer& container,
                               std::size_t n_spatial) {
    if (!container.has_two_body_integrals()) {
      return;
    }
    const auto& block = container.two_body_integrals_sparse().block(
        {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()});

    for (const auto& [idx, eri] : block) {
      EriKey key = exact_key(idx[0], idx[1], idx[2], idx[3], n_spatial);
      values_.emplace(key, eri);
    }

    entries_.reserve(values_.size());
    for (const auto& [key, value] : values_) {
      entries_.push_back({key, value});
    }
    std::sort(entries_.begin(), entries_.end(),
              [](const EriEntry& lhs, const EriEntry& rhs) {
                return lhs.key < rhs.key;
              });
  }

  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    auto it = values_.find(EriKey{p, q, r, s});
    return it == values_.end() ? 0.0 : it->second;
  }

  double aabb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aaaa(p, q, r, s);
  }

  double bbbb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aaaa(p, q, r, s);
  }

  template <typename F>
  void for_each_restricted_dense_coordinate(
      const std::vector<std::size_t>& sym_map, std::size_t n_spatial,
      double integral_threshold, F&& f) const {
    for (const auto& [key, eri] : entries_) {
      if (std::abs(eri) < integral_threshold) continue;
      if (key.q < key.p || key.s < key.r) continue;
      std::size_t pq_idx = sym_map[key.p * n_spatial + key.q];
      std::size_t rs_idx = sym_map[key.r * n_spatial + key.s];
      if (pq_idx > rs_idx) continue;

      f(key.p, key.q, key.r, key.s, eri);
    }
  }

 private:
  struct EriKey {
    std::size_t p = 0;
    std::size_t q = 0;
    std::size_t r = 0;
    std::size_t s = 0;

    bool operator==(const EriKey& other) const = default;

    bool operator<(const EriKey& other) const {
      return std::array{p, q, r, s} <
             std::array{other.p, other.q, other.r, other.s};
    }
  };

  struct EriKeyHash {
    std::size_t operator()(const EriKey& key) const noexcept {
      std::size_t seed = 0;
      auto combine = [&seed](std::size_t value) {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      };
      combine(key.p);
      combine(key.q);
      combine(key.r);
      combine(key.s);
      return seed;
    }
  };

  struct EriEntry {
    EriKey key;
    double value = 0.0;
  };

  static EriKey exact_key(std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s, std::size_t n_spatial) {
    if (p >= n_spatial || q >= n_spatial || r >= n_spatial ||
        s >= n_spatial) {
      throw std::invalid_argument(
          "majorana_map_hamiltonian: sparse two-body integral index is out "
          "of range for the Hamiltonian one-body dimension.");
    }
    return {p, q, r, s};
  }

  std::unordered_map<EriKey, double, EriKeyHash> values_;
  std::vector<EriEntry> entries_;
};

template <typename Source, typename F>
void for_each_restricted_representative_integral(
    const Source& source, const std::vector<std::size_t>& sym_map,
    std::size_t n_spatial, double integral_threshold, F&& f) {
  if constexpr (Source::sparse_restricted_iteration) {
    source.for_each_restricted_dense_coordinate(
        sym_map, n_spatial, integral_threshold, std::forward<F>(f));
  } else {
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = p; q < n_spatial; ++q) {
        std::size_t pq_idx = sym_map[p * n_spatial + q];
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = r; s < n_spatial; ++s) {
            std::size_t rs_idx = sym_map[r * n_spatial + s];
            if (pq_idx > rs_idx) continue;

            double eri = source.aaaa(p, q, r, s);
            if (std::abs(eri) < integral_threshold) continue;
            f(p, q, r, s, eri);
          }
        }
      }
    }
  }
}

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

template <std::size_t NW, typename TwoBodySource>
MajoranaMapResult majorana_map_impl_from_source(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const TwoBodySource& two_body,
    std::size_t n_spatial, bool spin_symmetric, double threshold,
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
          delta_corr += two_body.aaaa(p, q, q, s);
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

    for_each_restricted_representative_integral(
        two_body, sym_map, n_spatial, integral_threshold,
        [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s,
            double eri) {
          std::size_t pq_idx = sym_map[p * n_spatial + q];
          std::size_t rs_idx = sym_map[r * n_spatial + s];
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
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = two_body.aaaa(p, q, r, s);
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
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = two_body.bbbb(p, q, r, s);
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
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = two_body.aabb(p, q, r, s);
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
template <typename Source>
using SourceDispatchFn = MajoranaMapResult (*)(
    const MajoranaMapping&, double, const double*, const double*, const Source&,
    std::size_t, bool, double, double);

template <std::size_t NW, typename Source>
MajoranaMapResult majorana_map_source_dispatch(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const Source& source, std::size_t n_spatial,
    bool spin_symmetric, double threshold, double integral_threshold) {
  return majorana_map_impl_from_source<NW>(
      mapping, core_energy, h1_alpha, h1_beta, source, n_spatial,
      spin_symmetric, threshold, integral_threshold);
}

template <typename Source, std::size_t... Is>
constexpr std::array<SourceDispatchFn<Source>, sizeof...(Is)>
make_source_dispatch_table(std::index_sequence<Is...>) {
  return {{&majorana_map_source_dispatch<Is + 1, Source>...}};
}

constexpr std::size_t max_nw = 16;

inline std::size_t validated_num_words(const MajoranaMapping& mapping) {
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
  return num_words;
}

template <typename Source>
MajoranaMapResult dispatch_majorana_map_from_source(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const Source& source, std::size_t n_spatial,
    bool spin_symmetric, double threshold, double integral_threshold) {
  const std::size_t num_words = validated_num_words(mapping);
  static const auto table =
      make_source_dispatch_table<Source>(std::make_index_sequence<max_nw>{});
  return table[num_words - 1](mapping, core_energy, h1_alpha, h1_beta, source,
                              n_spatial, spin_symmetric, threshold,
                              integral_threshold);
}

std::vector<double> flatten_row_major(const Eigen::MatrixXd& matrix) {
  std::vector<double> result(
      static_cast<std::size_t>(matrix.rows() * matrix.cols()));
  const std::size_t cols = static_cast<std::size_t>(matrix.cols());
  for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
    for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
      result[static_cast<std::size_t>(r) * cols + static_cast<std::size_t>(c)] =
          matrix(r, c);
    }
  }
  return result;
}

}  // namespace detail

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold) {
  detail::DenseTwoBodySource source{eri_aaaa, eri_aabb, eri_bbbb, n_spatial};
  return detail::dispatch_majorana_map_from_source(
      mapping, core_energy, h1_alpha, h1_beta, source, n_spatial,
      spin_symmetric, threshold, integral_threshold);
}

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, const Hamiltonian& hamiltonian,
    double threshold, double integral_threshold) {
  auto [h1_alpha, h1_beta] = hamiltonian.get_one_body_integrals();
  const auto n_spatial = static_cast<std::size_t>(h1_alpha.rows());
  if (h1_alpha.rows() != h1_alpha.cols()) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: alpha one-body integrals must be square.");
  }
  const bool spin_symmetric = hamiltonian.is_restricted();
  if (!spin_symmetric &&
      (h1_beta.rows() != h1_alpha.rows() || h1_beta.cols() != h1_alpha.cols())) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: beta one-body integrals must match alpha "
        "dimensions.");
  }

  const std::size_t n_spin_orbitals = 2 * n_spatial;
  if (mapping.num_modes() != n_spin_orbitals) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: mapping has " +
        std::to_string(mapping.num_modes()) + " modes but the Hamiltonian has " +
        std::to_string(n_spin_orbitals) + " spin-orbitals.");
  }

  auto h1_alpha_flat = detail::flatten_row_major(h1_alpha);
  auto h1_beta_flat =
      spin_symmetric ? h1_alpha_flat : detail::flatten_row_major(h1_beta);
  const double* h1_beta_ptr =
      spin_symmetric ? h1_alpha_flat.data() : h1_beta_flat.data();

  if (spin_symmetric &&
      hamiltonian.has_container_type<SparseHamiltonianContainer>()) {
    const auto& container =
        hamiltonian.get_container<SparseHamiltonianContainer>();
    detail::SparseTwoBodySource source(container, n_spatial);
    return detail::dispatch_majorana_map_from_source(
        mapping, 0.0, h1_alpha_flat.data(), h1_beta_ptr, source,
        n_spatial, spin_symmetric, threshold, integral_threshold);
  }

  if (hamiltonian.has_container_type<CholeskyHamiltonianContainer>()) {
    const auto& container =
        hamiltonian.get_container<CholeskyHamiltonianContainer>();
    detail::CholeskyTwoBodySource source(container, n_spatial);
    return detail::dispatch_majorana_map_from_source(
        mapping, 0.0, h1_alpha_flat.data(), h1_beta_ptr, source,
        n_spatial, spin_symmetric, threshold, integral_threshold);
  }

  auto [eri_aaaa, eri_aabb, eri_bbbb] = hamiltonian.get_two_body_integrals();
  detail::DenseTwoBodySource source{eri_aaaa.data(), eri_aabb.data(),
                                    eri_bbbb.data(), n_spatial};
  return detail::dispatch_majorana_map_from_source(
      mapping, 0.0, h1_alpha_flat.data(), h1_beta_ptr, source, n_spatial,
      spin_symmetric, threshold, integral_threshold);
}

}  // namespace qdk::chemistry::data
