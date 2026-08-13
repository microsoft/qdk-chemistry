// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstddef>

namespace qdk::chemistry::data {

/// Whether @p value is a non-zero power of two.
constexpr bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

/// Smallest @c bits satisfying @f$2^{bits} \ge value@f$, which is
/// @f$\log_2 value@f$ exactly when @p value is a power of two.
constexpr std::size_t log2_exact(std::size_t value) {
  std::size_t bits = 0;
  while ((std::size_t{1} << bits) < value) {
    ++bits;
  }
  return bits;
}

}  // namespace qdk::chemistry::data
