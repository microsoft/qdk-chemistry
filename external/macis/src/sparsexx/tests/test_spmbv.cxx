/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/spblas/spmbv.hpp>

#include "catch2/catch.hpp"

TEST_CASE("CSR SPMBV", "[spmbv]") {
  using mat_type = sparsexx::csr_matrix<double, int32_t>;
  // [2 1 0 0 0 0]
  // [1 2 1 0 0 0]
  // [0 1 2 1 0 0]
  // [0 0 1 2 1 0]
  // [0 0 0 1 2 1]
  // [0 0 0 0 1 2]

  int indexing;
  SECTION("Indexing = 0") { indexing = 0; }

  SECTION("Indexing = 1") { indexing = 1; }

  int N = 6;
  int NNZ = 16;
  mat_type A(N, N, NNZ, indexing);

  A.rowptr() = {
      indexing,      indexing + 2,  indexing + 5,  indexing + 8,
      indexing + 11, indexing + 14, indexing + 16,
  };

  A.colind() = {indexing + 0, indexing + 1, indexing + 0, indexing + 1,
                indexing + 2, indexing + 1, indexing + 2, indexing + 3,
                indexing + 2, indexing + 3, indexing + 4, indexing + 3,
                indexing + 4, indexing + 5, indexing + 4, indexing + 5};

  A.nzval() = {2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2};

  std::vector<double> V = {1, 2, 3, 4, 5, 6};

  std::vector<double> AV_ref = {4, 8, 12, 16, 20, 17};

  std::vector<double> AV(N, 1.);
  sparsexx::spblas::gespmbv(1, 1., A, V.data(), N, 0., AV.data(), N);
  for (int i = 0; i < N; ++i) {
    CHECK(AV[i] == Approx(AV_ref[i]));
  }
}
