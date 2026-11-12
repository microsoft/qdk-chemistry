/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <limits>
#include <macis/fci/string_space.hpp>
#include <numeric>
#include <stdexcept>

namespace macis {

FCIStringSpace::FCIStringSpace(size_t norb, size_t nelec)
    : norb_(norb), nelec_(nelec) {
  if (nelec > norb)
    throw std::invalid_argument("FCIStringSpace: nelec exceeds norb");
  if (norb > std::numeric_limits<uint16_t>::max())
    throw std::invalid_argument("FCIStringSpace: too many orbitals");
  nexc_ = nelec * (norb - nelec);

  // Saturating Pascal triangle, binom[m * (nelec+1) + j] = C(m, j)
  const size_t ncol = nelec + 1;
  std::vector<size_t> binom((norb + 1) * ncol, 0);
  for (size_t m = 0; m <= norb; ++m) {
    binom[m * ncol] = 1;
    for (size_t j = 1; j <= std::min(m, nelec); ++j) {
      const size_t a = binom[(m - 1) * ncol + j - 1];
      const size_t b = binom[(m - 1) * ncol + j];
      binom[m * ncol + j] = a > std::numeric_limits<size_t>::max() - b
                                ? std::numeric_limits<size_t>::max()
                                : a + b;
    }
  }
  nstrings_ = binom[norb * ncol + nelec];
  if (nstrings_ > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("FCIStringSpace: too many strings");

  // Lexicographic rank = C(norb,nelec) - 1 - sum_i C(norb-1-o_i, nelec-i);
  // only orbitals reachable at position i (i <= o <= norb-nelec+i) are needed
  rank_weight_.assign(nelec * norb, 0);
  for (size_t i = 0; i < nelec; ++i)
    for (size_t o = i; o + nelec <= norb + i; ++o)
      rank_weight_[i * norb + o] = binom[(norb - 1 - o) * ncol + nelec - i];

  occupied_.resize(nstrings_ * nelec);
  std::vector<uint32_t> occ(nelec);
  std::iota(occ.begin(), occ.end(), 0u);
  for (size_t s = 0; s < nstrings_; ++s) {
    std::copy(occ.begin(), occ.end(), occupied_.begin() + s * nelec);
    size_t i = nelec;
    while (i > 0 and occ[i - 1] == norb - nelec + i - 1) --i;
    if (i == 0) break;
    ++occ[i - 1];
    for (size_t j = i; j < nelec; ++j) occ[j] = occ[j - 1] + 1;
  }

  excitations_.resize(nstrings_ * nexc_);
#pragma omp parallel
  {
    std::vector<uint32_t> excited(nelec);
    std::vector<bool> is_occupied(norb);
#pragma omp for schedule(static)
    for (size_t s = 0; s < nstrings_; ++s) {
      const uint32_t* o = occupied(s);
      std::fill(is_occupied.begin(), is_occupied.end(), false);
      for (size_t i = 0; i < nelec; ++i) is_occupied[o[i]] = true;

      Excitation* ex = excitations_.data() + s * nexc_;
      for (size_t di = 0; di < nelec; ++di) {
        const uint32_t d = o[di];
        for (uint32_t c = 0; c < norb; ++c) {
          if (is_occupied[c]) continue;
          // Sorted occupied list of the excited string and the number of
          // occupied orbitals strictly between c and d (fermionic sign)
          size_t n = 0, nbetween = 0;
          bool inserted = false;
          for (size_t i = 0; i < nelec; ++i) {
            if (i == di) continue;
            if (not inserted and c < o[i]) {
              excited[n++] = c;
              inserted = true;
            }
            excited[n++] = o[i];
            if (o[i] > std::min(c, d) and o[i] < std::max(c, d)) ++nbetween;
          }
          if (not inserted) excited[n] = c;

          *ex++ = Excitation{static_cast<uint32_t>(address(excited.data())),
                             static_cast<uint32_t>(fci_pair_index(c, d)),
                             static_cast<uint16_t>(c), static_cast<uint16_t>(d),
                             nbetween % 2 ? -1 : 1};
        }
      }
    }
  }
}

size_t FCIStringSpace::address(const uint32_t* occ) const {
  size_t r = 0;
  for (size_t i = 0; i < nelec_; ++i) r += rank_weight_[i * norb_ + occ[i]];
  return nstrings_ - 1 - r;
}

}  // namespace macis
