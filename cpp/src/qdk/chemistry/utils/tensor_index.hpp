// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace qdk::chemistry::utils {

/// Number of elements of a tensor; throws std::length_error if unaddressable.
inline std::size_t tensor_size(std::span<const std::size_t> extents) {
  std::size_t size = 1;
  for (const auto extent : extents) {
    if (extent && size > std::vector<double>().max_size() / extent)
      throw std::length_error("Tensor extents exceed addressable storage.");
    size *= extent;
  }
  return size;
}

inline std::size_t tensor_size(std::initializer_list<std::size_t> extents) {
  return tensor_size(std::span(extents.begin(), extents.size()));
}

/// Row-major offset of element (p, q, r, s) of an n x n x n x n tensor.
inline std::size_t index4(std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s, std::size_t n) {
  return ((p * n + q) * n + r) * n + s;
}

}  // namespace qdk::chemistry::utils
