// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <qdk/chemistry/utils/logger.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Consumers use Hermitian two-body integrals through their Hermitian part: the
// average over (pq|rs), (rs|pq), (qp|sr), (sr|qp), or over (pq|rs), (qp|sr)
// for an unrestricted alpha-beta block, which has no exchange partner.

namespace qdk::chemistry::data::detail {

using TwoBodyKey = std::array<std::size_t, 4>;

inline void require_finite_two_body(const double* integrals, std::size_t count,
                                    const std::string& consumer) {
  if (!std::all_of(integrals, integrals + count,
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        consumer + ": the two-body integrals contain a non-finite value.");
  }
}

/// Accumulates ||x - y||^2, ||x||^2 and ||y||^2 element by element.
struct TwoBodyComparison {
  double difference = 0.0;
  double first = 0.0;
  double second = 0.0;

  void add(double x, double y) {
    const double delta = x - y;
    difference += delta * delta;
    first += x * x;
    second += y * y;
  }
  /// Eigen's isApprox: ||x - y|| <= p min(||x||, ||y||).
  bool approx() const {
    const double precision = Eigen::NumTraits<double>::dummy_precision();
    return difference <= precision * precision * std::min(first, second);
  }
  /// ||x - y|| / ||x||.
  double relative() const {
    return first == 0.0 ? 0.0 : std::sqrt(difference / first);
  }
};

/// Warns if `change`, made by averaging over `partners`, exceeds round-off.
inline void warn_if_not_hermitian(const TwoBodyComparison& change,
                                  const std::string& consumer,
                                  const std::string& partners) {
  if (change.approx()) return;
  QDK_LOGGER().warn(
      "{}: the two-body integrals differ from their Hermitian part, the "
      "average over {}, by {:.1e} relative to their norm; the Hermitian part "
      "is used. Symmetrize the integrals to avoid this warning.",
      consumer, partners, change.relative());
}

inline constexpr const char* fourfold_partners =
    "(pq|rs), (rs|pq), (qp|sr) and (sr|qp)";
inline constexpr const char* reversal_partners = "(pq|rs) and (qp|sr)";

/// Average of a row-major restricted or same-spin (pq|rs) block over (pq|rs),
/// (rs|pq), (qp|sr), (sr|qp): `integrals` if they equal it exactly, otherwise
/// `storage`, filled with it.
inline const double* fourfold_symmetrized(const double* integrals,
                                          std::size_t n,
                                          Eigen::VectorXd& storage,
                                          const std::string& consumer) {
  const std::size_t n2 = n * n;
  require_finite_two_body(integrals, n2 * n2, consumer);
  auto at = [integrals, n, n2](std::size_t p, std::size_t q, std::size_t r,
                               std::size_t s) {
    return integrals[(p * n + q) * n2 + r * n + s];
  };
  auto symmetric = [&] {
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s) {
            const double value = at(p, q, r, s);
            if (value != at(r, s, p, q) || value != at(q, p, s, r))
              return false;
          }
    return true;
  };
  if (symmetric()) return integrals;

  storage.resize(static_cast<Eigen::Index>(n2 * n2));
  // The operator depends only on the bra-ket average.
  TwoBodyComparison change;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const double direct = at(p, q, r, s) + at(r, s, p, q);
          const double reversed = at(q, p, s, r) + at(s, r, q, p);
          const double average = 0.25 * (direct + reversed);
          storage[static_cast<Eigen::Index>((p * n + q) * n2 + r * n + s)] =
              average;
          change.add(0.5 * direct, average);
        }
  warn_if_not_hermitian(change, consumer, fourfold_partners);
  return storage.data();
}

/// Average of a row-major alpha-beta (pq|rs) block over (pq|rs), (qp|sr), as
/// for @ref fourfold_symmetrized.
inline const double* reversal_symmetrized(const double* integrals,
                                          std::size_t n,
                                          Eigen::VectorXd& storage,
                                          const std::string& consumer) {
  const std::size_t n2 = n * n;
  require_finite_two_body(integrals, n2 * n2, consumer);
  auto at = [integrals, n, n2](std::size_t p, std::size_t q, std::size_t r,
                               std::size_t s) {
    return integrals[(p * n + q) * n2 + r * n + s];
  };
  auto symmetric = [&] {
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s)
            if (at(p, q, r, s) != at(q, p, s, r)) return false;
    return true;
  };
  if (symmetric()) return integrals;

  storage.resize(static_cast<Eigen::Index>(n2 * n2));
  TwoBodyComparison change;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const double value = at(p, q, r, s);
          const double average = 0.5 * (value + at(q, p, s, r));
          storage[static_cast<Eigen::Index>((p * n + q) * n2 + r * n + s)] =
              average;
          change.add(value, average);
        }
  warn_if_not_hermitian(change, consumer, reversal_partners);
  return storage.data();
}

/// Representative shared by all eight permutations of (p,q,r,s); also the
/// lexicographic minimum of the class.
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

/// Images of (pq|rs) under exchange and reversal; may repeat positions.
inline std::array<TwoBodyKey, 4> fourfold_orbit(const TwoBodyKey& k) {
  return {{k,
           {k[2], k[3], k[0], k[1]},
           {k[1], k[0], k[3], k[2]},
           {k[3], k[2], k[1], k[0]}}};
}

/// Images of (pq|rs) under the eightfold group; may repeat positions.
inline std::array<TwoBodyKey, 8> eightfold_orbit(const TwoBodyKey& k) {
  const auto direct = fourfold_orbit(k);
  const auto swapped = fourfold_orbit({k[1], k[0], k[2], k[3]});
  std::array<TwoBodyKey, 8> orbit;
  std::copy(direct.begin(), direct.end(), orbit.begin());
  std::copy(swapped.begin(), swapped.end(), orbit.begin() + 4);
  return orbit;
}

/// Average of `at` over the eightfold class of `key`, summed pairwise so that
/// equal values average to themselves exactly.
template <class At>
double eightfold_average(const TwoBodyKey& key, const At& at) {
  const auto o = eightfold_orbit(key);
  return (((at(o[0]) + at(o[1])) + (at(o[2]) + at(o[3]))) +
          ((at(o[4]) + at(o[5])) + (at(o[6]) + at(o[7])))) /
         8.0;
}

/// Compares a row-major (pq|rs) tensor with its eightfold class averages.
inline TwoBodyComparison eightfold_comparison(const double* integrals,
                                              std::size_t n) {
  auto at = [integrals, n](const TwoBodyKey& k) {
    return integrals[((k[0] * n + k[1]) * n + k[2]) * n + k[3]];
  };
  TwoBodyComparison comparison;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = p; q < n; ++q)
      for (std::size_t r = p; r < n; ++r)
        for (std::size_t s = r; s < n; ++s) {
          const TwoBodyKey key{p, q, r, s};
          if (eightfold_canonical(key) != key) continue;
          const double average = eightfold_average(key, at);
          auto orbit = eightfold_orbit(key);
          std::sort(orbit.begin(), orbit.end());
          const auto end = std::unique(orbit.begin(), orbit.end());
          for (auto position = orbit.begin(); position != end; ++position)
            comparison.add(at(*position), average);
        }
  return comparison;
}

inline std::runtime_error not_eightfold_error(
    const std::string& consumer, const TwoBodyComparison& comparison) {
  char relative[32];
  std::snprintf(relative, sizeof(relative), "%.1e", comparison.relative());
  return std::runtime_error(
      consumer +
      " requires 8-fold two-body permutation symmetry; the integrals differ "
      "from their 8-fold average by " +
      relative +
      " relative to their norm, so they cannot be represented losslessly. If "
      "this is round-off, as in Hamiltonians saved by earlier versions, "
      "rebuild the Hamiltonian.");
}

// The averages below are taken one swap at a time, (rs) <-> (sr) within each
// row, then (pq) <-> (qp) across rows, then (pq|rs) <-> (rs|pq), which visits
// memory in order and leaves partners bitwise equal.

/// Replaces a row-major alpha-beta (pq|rs) tensor by its average over (pq|rs),
/// (qp|rs), (pq|sr), (qp|sr).
inline void pair_symmetrize(double* integrals, std::size_t n) {
  const std::size_t n2 = n * n;
  for (std::size_t pq = 0; pq < n2; ++pq) {
    double* row = integrals + pq * n2;
    for (std::size_t r = 0; r < n; ++r)
      for (std::size_t s = r + 1; s < n; ++s) {
        double& forward = row[r * n + s];
        double& reverse = row[s * n + r];
        forward = reverse = 0.5 * (forward + reverse);
      }
  }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = p + 1; q < n; ++q) {
      double* forward = integrals + (p * n + q) * n2;
      double* reverse = integrals + (q * n + p) * n2;
      for (std::size_t rs = 0; rs < n2; ++rs)
        forward[rs] = reverse[rs] = 0.5 * (forward[rs] + reverse[rs]);
    }
}

/// Replaces a row-major (pq|rs) tensor by its eightfold class averages.
inline void eightfold_symmetrize(double* integrals, std::size_t n) {
  pair_symmetrize(integrals, n);
  const std::size_t n2 = n * n;
  constexpr std::size_t tile = 64;
  for (std::size_t i0 = 0; i0 < n2; i0 += tile)
    for (std::size_t j0 = i0; j0 < n2; j0 += tile)
      for (std::size_t i = i0; i < std::min(i0 + tile, n2); ++i)
        for (std::size_t j = std::max(j0, i + 1); j < std::min(j0 + tile, n2);
             ++j) {
          double& upper = integrals[i * n2 + j];
          double& lower = integrals[j * n2 + i];
          upper = lower = 0.5 * (upper + lower);
        }
}

/// Whether three-center factors, column-major [n^2 x naux], satisfy
/// L_pq = L_qp exactly; the integrals they reconstruct are then eightfold.
inline bool factors_are_pair_symmetric(const double* factors, std::size_t n,
                                       std::size_t naux) {
  const std::size_t n2 = n * n;
  for (std::size_t aux = 0; aux < naux; ++aux)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = p + 1; q < n; ++q)
        if (factors[p * n + q + aux * n2] != factors[q * n + p + aux * n2])
          return false;
  return true;
}

/// Literal sparse two-body tensor: positions that are absent are zero.
using SparseTwoBodyTensor = std::map<TwoBodyKey, double>;

inline std::string two_body_key_string(const TwoBodyKey& key) {
  return "(" + std::to_string(key[0]) + ", " + std::to_string(key[1]) + ", " +
         std::to_string(key[2]) + ", " + std::to_string(key[3]) + ")";
}

/// Distinct positions of an orbit.
template <std::size_t N>
std::vector<TwoBodyKey> distinct_positions(std::array<TwoBodyKey, N> orbit) {
  std::sort(orbit.begin(), orbit.end());
  return {orbit.begin(), std::unique(orbit.begin(), orbit.end())};
}

/// Sparse counterpart of @ref eightfold_comparison, reading absent positions
/// as zero.  Only classes with a stored member can differ from their average.
inline TwoBodyComparison sparse_eightfold_comparison(
    const SparseTwoBodyTensor& tensor) {
  auto at = [&tensor](const TwoBodyKey& key) {
    const auto it = tensor.find(key);
    return it == tensor.end() ? 0.0 : it->second;
  };
  std::set<TwoBodyKey> classes;
  for (const auto& [key, value] : tensor) {
    classes.insert(eightfold_canonical(key));
  }
  TwoBodyComparison comparison;
  for (const auto& canonical : classes) {
    const double average = eightfold_average(canonical, at);
    for (const auto& position : distinct_positions(eightfold_orbit(canonical)))
      comparison.add(at(position), average);
  }
  return comparison;
}

/// Replaces `tensor` by its average as for @ref fourfold_symmetrized, reading
/// absent positions as zero.
inline void sparse_fourfold_symmetrize(SparseTwoBodyTensor& tensor,
                                       const std::string& consumer) {
  auto at = [&tensor](const TwoBodyKey& key) {
    const auto it = tensor.find(key);
    return it == tensor.end() ? 0.0 : it->second;
  };
  auto symmetric = [&] {
    for (const auto& [key, value] : tensor) {
      const auto [p, q, r, s] = key;
      if (value != at({r, s, p, q}) || value != at({q, p, s, r})) return false;
    }
    return true;
  };
  if (symmetric()) return;

  SparseTwoBodyTensor averaged;
  TwoBodyComparison change;
  for (const auto& [key, value] : tensor) {
    if (averaged.count(key)) continue;
    const auto orbit = fourfold_orbit(key);
    const double direct = at(orbit[0]) + at(orbit[1]);
    const double reversed = at(orbit[2]) + at(orbit[3]);
    const double average = 0.25 * (direct + reversed);
    for (const auto& position : distinct_positions(orbit)) {
      averaged.emplace(position, average);
      const auto [p, q, r, s] = position;
      change.add(0.5 * (at(position) + at({r, s, p, q})), average);
    }
  }
  warn_if_not_hermitian(change, consumer, fourfold_partners);
  tensor = std::move(averaged);
}

/// Nonzero entries that are the only stored member of a larger eightfold
/// class: the deprecated sparse shorthand in which one record stands for its
/// whole class.
inline std::vector<TwoBodyKey> lone_eightfold_representatives(
    const SparseTwoBodyTensor& tensor) {
  std::map<TwoBodyKey, std::pair<TwoBodyKey, std::size_t>> classes;
  for (const auto& [key, value] : tensor) {
    ++classes.try_emplace(eightfold_canonical(key), key, 0)
          .first->second.second;
  }
  std::vector<TwoBodyKey> lone;
  for (const auto& [canonical, stored] : classes) {
    const auto& [key, count] = stored;
    if (count == 1 && eightfold_class_size(canonical) > 1 &&
        tensor.at(key) != 0.0) {
      lone.push_back(key);
    }
  }
  return lone;
}

/// Expands every lone eightfold representative over its class.
inline void expand_legacy_eightfold_representatives(
    SparseTwoBodyTensor& tensor) {
  for (const auto& key : lone_eightfold_representatives(tensor)) {
    const double value = tensor.at(key);
    for (const auto& position : eightfold_orbit(key)) {
      tensor.emplace(position, value);
    }
  }
}

}  // namespace qdk::chemistry::data::detail
