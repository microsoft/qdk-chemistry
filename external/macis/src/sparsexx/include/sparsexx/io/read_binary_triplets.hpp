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
#include <iostream>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <stdexcept>
#include <string>

namespace sparsexx {

template <typename SpMatType>
detail::enable_if_coo_matrix_t<SpMatType, SpMatType> read_binary_triplet_as_coo(
    std::string fname) {
  using index_t = detail::index_type_t<SpMatType>;
  using value_t = detail::value_type_t<SpMatType>;

  std::ifstream f_in(fname, std::ios::binary);
  size_t nnz;
  index_t m, n;
  f_in.read((char*)&m, sizeof(index_t));
  f_in.read((char*)&n, sizeof(index_t));
  f_in.read((char*)&nnz, sizeof(size_t));

  std::cout << "Reading bin data" << std::endl;
  SpMatType A(m, n, nnz);
  f_in.read((char*)A.rowind().data(), nnz * sizeof(index_t));
  f_in.read((char*)A.colind().data(), nnz * sizeof(index_t));
  f_in.read((char*)A.nzval().data(), nnz * sizeof(value_t));

  A.determine_indexing_from_adj();
  A.expand_from_triangle();
  A.sort_by_row_index();

  assert(A.is_sorted_by_row_index());

  return A;
}

template <typename SpMatType>
SpMatType read_binary_triplet(std::string fname) {
  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  if constexpr (detail::is_coo_matrix_v<SpMatType>)
    return read_binary_triplet_as_coo<SpMatType>(fname);
  else
    return SpMatType(
        read_binary_triplet_as_coo<coo_matrix<value_t, index_t, allocator_t> >(
            fname));
}

}  // namespace sparsexx
