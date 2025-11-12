/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cassert>
#include <fstream>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <string>

namespace sparsexx {

template <typename T, typename index_t, typename Alloc>
csc_matrix<T, index_t, Alloc> read_rb_as_csc(std::string fname) {
  std::ifstream f_in(fname);

  std::string line;

  // Skip the first two lines
  std::getline(f_in, line);  // comments
  std::getline(f_in, line);  // some strange metadata

  // Get useful meta data
  int64_t m, n, nnz;
  // bool is_sym;
  {
    std::getline(f_in, line);
    auto tokens = tokenize(line);
    assert(tokens.size() == 5);

    auto type = tokens[0];
    m = std::stoll(tokens[1]);
    n = std::stoll(tokens[2]);
    nnz = std::stoll(tokens[3]);
    // Whats the last one for??

    // is_sym = type[1] == 's' or type[1] == 'S';
    assert(nnz <= m * n);
  }

  // Skip format line
  std::getline(f_in, line);

  csc_matrix<T, index_t, Alloc> A(m, n, nnz);

  int64_t curcount = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    for (const auto& t : tokens) A.colptr()[curcount++] = std::stoi(t);

    if (curcount == (n + 1)) break;
  }

  curcount = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    for (const auto& t : tokens) A.rowind()[curcount++] = std::stoi(t);

    if (curcount == nnz) break;
  }

  curcount = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    for (const auto& t : tokens) A.nzval()[curcount++] = std::stod(t);

    if (curcount == nnz) break;
  }

  assert(!std::getline(f_in, line));
  return A;
}

template <typename SpMatType>
SpMatType read_rb(std::string fname) {
  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  if constexpr (detail::is_csc_matrix_v<SpMatType>)
    return read_rb_as_csc<value_t, index_t, allocator_t>(fname);
  else
    return SpMatType(read_rb_as_csc<value_t, index_t, allocator_t>(fname));
  abort();
}

}  // namespace sparsexx
