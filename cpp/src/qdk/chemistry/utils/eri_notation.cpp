// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cstddef>
#include <qdk/chemistry/utils/eri_notation.hpp>
#include <stdexcept>
#include <vector>

namespace qdk::chemistry::utils {

namespace {

inline std::size_t idx4(std::size_t p, std::size_t q, std::size_t r,
                        std::size_t s, std::size_t n) {
  return ((p * n + q) * n + r) * n + s;
}

void check_size(const std::vector<double>& t, std::size_t n) {
  if (t.size() != n * n * n * n)
    throw std::invalid_argument(
        "eri_notation: tensor size does not match n^4.");
}

}  // namespace

std::vector<double> chemist_to_antisymmetrized(
    const std::vector<double>& chemist, std::size_t n) {
  check_size(chemist, n);
  std::vector<double> out(n * n * n * n);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          out[idx4(p, q, r, s, n)] =
              chemist[idx4(p, r, q, s, n)] - chemist[idx4(p, s, q, r, n)];
  return out;
}

std::vector<double> antisymmetrized_to_chemist(
    const std::vector<double>& antisymmetrized, std::size_t n) {
  check_size(antisymmetrized, n);
  std::vector<double> out(n * n * n * n);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          out[idx4(p, q, r, s, n)] = 0.5 * antisymmetrized[idx4(p, r, q, s, n)];
  return out;
}

}  // namespace qdk::chemistry::utils
