/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <sparsexx/matrix_types/coo_matrix.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>

#include "catch2/catch.hpp"

TEST_CASE("COO Matrix", "[matrix_types]") {
  using mat_type = sparsexx::coo_matrix<double, int32_t>;

  SECTION("Insert") {
    size_t n = 6;
    mat_type A(n, n, 0, 0);  // empty matrix zero-indexing
    for (int i = 0; i < n; ++i) {
      A.insert<false>(i, i, 2);
      if (i < n - 1) A.insert<false>(i, i + 1, 1);
      if (i > 0) A.insert<false>(i, i - 1, 1);
    }
    A.sort_by_row_index();  // Put into canonical order
    REQUIRE(A.nnz() == 16);
    REQUIRE(A.rowind() == std::vector<int32_t>{0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                               4, 4, 4, 5, 5});
    REQUIRE(A.colind() == std::vector<int32_t>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4,
                                               3, 4, 5, 4, 5});
    REQUIRE(A.nzval() == std::vector<double>{2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1,
                                             2, 1, 1, 2});
  }
}
