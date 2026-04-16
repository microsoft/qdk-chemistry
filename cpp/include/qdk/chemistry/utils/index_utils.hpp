// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::utils {

/// Compute n^4 with overflow checking. Throws std::overflow_error if the
/// result would wrap size_t.
inline size_t checked_n4(size_t n) {
  if (n == 0) return 0;
  size_t n2 = n * n;
  if (n2 / n != n) {
    throw std::overflow_error("n^4 overflows size_t for n = " +
                              std::to_string(n));
  }
  size_t n4 = n2 * n2;
  if (n4 / n2 != n2) {
    throw std::overflow_error("n^4 overflows size_t for n = " +
                              std::to_string(n));
  }
  return n4;
}

}  // namespace qdk::chemistry::utils
