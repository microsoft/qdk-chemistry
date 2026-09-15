// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <utility>

namespace qdk::chemistry::data::detail {

/// Relative to the largest integral in the tensor.  A four-index
/// transformation leaves round-off proportional to the integral magnitude
/// rather than at an absolute floor: measured worst-case asymmetry runs from
/// 3e-15 relative (cc-pVDZ) to 9e-12 relative (cc-pVQZ), while a genuinely
/// reduced-symmetry tensor breaks the symmetry at O(1) relative magnitude.
inline constexpr double two_body_symmetry_tolerance = 1e-8;

using TwoBodyKey = std::array<std::size_t, 4>;

/// Permutation symmetry of a two-body tensor, in increasing strength.
///
/// The three generators have different standing.  a+_p a+_r a_s a_q is
/// invariant under (p,q) <-> (r,s), so the operator depends only on the
/// bra-ket average gbar_pqrs = (g_pqrs + g_rspq) / 2: bra-ket symmetry is a
/// gauge choice, not a property of the physics.  Given that average,
/// (pq|rs) = (qp|sr) is exactly Hermiticity of the operator, and the
/// independent bra swap (pq|rs) = (qp|rs) additionally requires real orbitals.
///
/// Genuine electron-repulsion integrals over real orbitals therefore always
/// reach `EightFold`.  `FourFold` is the general Hermitian two-body operator,
/// which is what downfolding and effective-Hamiltonian constructions produce.
enum class TwoBodySymmetry {
  NonHermitian,        ///< gbar_pqrs != gbar_qpsr: no Hermitian operator
  NotBraKetSymmetric,  ///< g_pqrs != g_rspq: Hermitian, but g != gbar
  FourFold,            ///< (pq|rs) = (qp|sr) = (rs|pq) = (sr|qp)
  EightFold,           ///< additionally (pq|rs) = (qp|rs) = (pq|sr)
};

/// Classify `integrals`, a row-major (pq|rs) tensor of extent `n` per index.
inline TwoBodySymmetry classify_two_body_symmetry(const double* integrals,
                                                  std::size_t n,
                                                  double relative_tolerance) {
  const std::size_t count = n * n * n * n;
  double scale = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    scale = std::max(scale, std::abs(integrals[i]));
  }
  const double tolerance = relative_tolerance * scale;

  auto idx = [n](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * n + q) * n + r) * n + s;
  };
  auto equal = [tolerance](double lhs, double rhs) {
    return std::abs(lhs - rhs) <= tolerance;
  };

  bool braket_symmetric = true;
  bool eightfold = true;
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          const double value = integrals[idx(p, q, r, s)];
          const double swapped = integrals[idx(r, s, p, q)];
          // Hermiticity and the 8-fold test apply to the bra-ket average,
          // since that is all the operator sees.
          const double mean = 0.5 * (value + swapped);
          if (!equal(mean, 0.5 * (integrals[idx(q, p, s, r)] +
                                  integrals[idx(s, r, q, p)]))) {
            return TwoBodySymmetry::NonHermitian;
          }
          if (braket_symmetric && !equal(value, swapped)) {
            braket_symmetric = false;
          }
          if (eightfold && !equal(mean, 0.5 * (integrals[idx(q, p, r, s)] +
                                               integrals[idx(r, s, q, p)]))) {
            eightfold = false;
          }
        }
      }
    }
  }
  // Both evaluation strategies read the stored tensor directly in the
  // cross-spin channel, so a Hermitian operator in the wrong gauge still has
  // to be symmetrized by the caller before it can be mapped.
  if (!braket_symmetric) return TwoBodySymmetry::NotBraKetSymmetric;
  return eightfold ? TwoBodySymmetry::EightFold : TwoBodySymmetry::FourFold;
}

/// Whether every three-center factor is symmetric in its orbital pair.
///
/// Factors are column-major `[n^2 x naux]` with the pair index in row-major
/// order, so element (pq, Q) lives at `factors[p * n + q + Q * n^2]`.  The
/// reconstruction sum_Q L^Q_pq L^Q_rs is bra-ket symmetric by construction but
/// carries the full eightfold symmetry only when each L^Q is symmetric.
inline bool cholesky_factors_are_pair_symmetric(const double* factors,
                                                std::size_t n, std::size_t naux,
                                                double relative_tolerance) {
  const std::size_t n2 = n * n;
  double scale = 0.0;
  for (std::size_t i = 0; i < n2 * naux; ++i) {
    scale = std::max(scale, std::abs(factors[i]));
  }
  const double tolerance = relative_tolerance * scale;

  for (std::size_t aux = 0; aux < naux; ++aux) {
    const double* factor = factors + aux * n2;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        if (std::abs(factor[p * n + q] - factor[q * n + p]) > tolerance) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Representative shared by all eight permutations of (p,q,r,s).  It is also
/// the lexicographic minimum of the class, which sparse ingestion relies on to
/// pick a deterministic representative value.
inline TwoBodyKey eightfold_canonical(TwoBodyKey k) {
  if (k[0] > k[1]) std::swap(k[0], k[1]);
  if (k[2] > k[3]) std::swap(k[2], k[3]);
  if (k[0] > k[2] || (k[0] == k[2] && k[1] > k[3])) {
    std::swap(k[0], k[2]);
    std::swap(k[1], k[3]);
  }
  return k;
}

/// Number of *distinct* index tuples in the permutation class of a canonical
/// key: repeated indices make some of the eight permutations coincide.
inline std::size_t eightfold_class_size(const TwoBodyKey& canonical) {
  const std::size_t bra = canonical[0] == canonical[1] ? 1 : 2;
  const std::size_t ket = canonical[2] == canonical[3] ? 1 : 2;
  const std::size_t swap =
      canonical[0] == canonical[2] && canonical[1] == canonical[3] ? 1 : 2;
  return bra * ket * swap;
}

/// Sparse counterpart of @ref classify_two_body_symmetry.
///
/// Consumers of sparse data — the mapping engine and ordinary FCIDUMP readers
/// alike — expand one stored record over its whole permutation class, while
/// the container implies zero everywhere nothing is stored.  The two readings
/// agree only when a class is stored exactly once (unambiguously a
/// representative) or in full; in between, the expansion would overwrite the
/// implied zeros.
template <class Entries>
bool sparse_entries_have_eightfold_symmetry(const Entries& entries,
                                            double relative_tolerance) {
  std::map<TwoBodyKey, double> stored;
  double scale = 0.0;
  for (const auto& [idx, value] : entries) {
    stored[{static_cast<std::size_t>(idx[0]), static_cast<std::size_t>(idx[1]),
            static_cast<std::size_t>(idx[2]),
            static_cast<std::size_t>(idx[3])}] = value;
    scale = std::max(scale, std::abs(value));
  }
  const double tolerance = relative_tolerance * scale;

  std::map<TwoBodyKey, std::pair<double, std::size_t>> classes;
  for (const auto& [key, value] : stored) {
    auto [it, inserted] =
        classes.try_emplace(eightfold_canonical(key), value, 1);
    if (inserted) continue;
    if (std::abs(it->second.first - value) > tolerance) return false;
    ++it->second.second;
  }
  for (const auto& [key, aggregate] : classes) {
    if (aggregate.second != 1 &&
        aggregate.second != eightfold_class_size(key)) {
      return false;
    }
  }
  return true;
}

}  // namespace qdk::chemistry::data::detail
