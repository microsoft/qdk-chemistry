/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <sparsexx/io/read_binary_triplets.hpp>
#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/io/read_rb.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>

#include "catch2/catch.hpp"

TEST_CASE("CSR", "[io]") {
  using mat_type = sparsexx::csr_matrix<double, int32_t>;
  const int M(7), N(7), NNZ(15);
  const std::vector<int32_t> ref_colind = {2, 3, 4, 2, 5, 3, 6, 4,
                                           7, 1, 5, 1, 6, 1, 7};
  const std::vector<int32_t> ref_rowptr = {1, 4, 6, 8, 10, 12, 14, 16};
  const std::vector<double> ref_nzval = {
      1,    1,          1, -1,         0.45, -1,          0.1, -1,
      0.45, -0.0359994, 1, -0.0176371, 1,    -0.00772178, 1};

  mat_type A;
  SECTION("MatrixMarket") {
    std::string ref_fname = SPARSEXX_DATA_DIR "/b1_ss/b1_ss.mtx";
    A = sparsexx::read_mm<mat_type>(ref_fname);
  }
  SECTION("RutherfordBoeing") {
    std::string ref_fname = SPARSEXX_DATA_DIR "/b1_ss/b1_ss.rb";
    A = sparsexx::read_rb<mat_type>(ref_fname);
  }

  // Check output dims
  REQUIRE(A.m() == M);
  REQUIRE(A.n() == N);
  REQUIRE(A.nnz() == NNZ);
  REQUIRE(A.colind().size() == NNZ);
  REQUIRE(A.rowptr().size() == (M + 1));

  // Check data
  CHECK(ref_colind == A.colind());
  CHECK(ref_rowptr == A.rowptr());
  for (auto i = 0; i < NNZ; ++i) {
    CHECK(A.nzval()[i] == Approx(ref_nzval[i]));
  }
}

TEST_CASE("CSC", "[io]") {
  using mat_type = sparsexx::csc_matrix<double, int32_t>;
  const int M(7), N(7), NNZ(15);
  const std::vector<int32_t> ref_rowind = {5, 6, 7, 1, 2, 1, 3, 1,
                                           4, 2, 5, 3, 6, 4, 7};
  const std::vector<int32_t> ref_colptr = {1, 4, 6, 8, 10, 12, 14, 16};
  const std::vector<double> ref_nzval = {
      -0.0359994, -0.0176371, -0.00772178, 1,   -1, 1,    -1, 1,
      -1,         0.45,       1,           0.1, 1,  0.45, 1};

  mat_type A;
  SECTION("MatrixMarket") {
    std::string ref_fname = SPARSEXX_DATA_DIR "/b1_ss/b1_ss.mtx";
    A = sparsexx::read_mm<mat_type>(ref_fname);
  }
  SECTION("RutherfordBoeing") {
    std::string ref_fname = SPARSEXX_DATA_DIR "/b1_ss/b1_ss.rb";
    A = sparsexx::read_rb<mat_type>(ref_fname);
  }

  // Check output dims
  REQUIRE(A.m() == M);
  REQUIRE(A.n() == N);
  REQUIRE(A.nnz() == NNZ);
  REQUIRE(A.rowind().size() == NNZ);
  REQUIRE(A.colptr().size() == (N + 1));

  // Check data
  CHECK(ref_rowind == A.rowind());
  CHECK(ref_colptr == A.colptr());
  for (auto i = 0; i < NNZ; ++i) {
    CHECK(A.nzval()[i] == Approx(ref_nzval[i]));
  }
}
