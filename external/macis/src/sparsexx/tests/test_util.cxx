/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/util/submatrix.hpp>

#include "catch2/catch.hpp"

TEST_CASE("CSR Utilities", "[util]") {
  using mat_type = sparsexx::csr_matrix<double, int32_t>;
  std::string ref_fname = SPARSEXX_DATA_DIR "/b1_ss/b1_ss.mtx";
  auto A = sparsexx::read_mm<mat_type>(ref_fname);

  std::vector<double> A_dense(A.m() * A.n());
  sparsexx::convert_to_dense(A, A_dense.data(), A.m());
  // for( auto i = 0; i < A.m(); ++i ) {
  //   for( auto j = 0; j < A.n(); ++j ) std::cout << A_dense[i+j*A.m()] << ",
  //   "; std::cout << std::endl;
  // }
  //  [ 0,           1,  1,  1, 0,    0,   0,
  //  [ 0,          -1,  0,  0, 0.45, 0,   0,
  //  [ 0,           0, -1,  0, 0,    0.1, 0,
  //  [ 0,           0,  0, -1, 0,    0,   0.45,
  //  [-0.0359994,   0,  0,  0, 1,    0,   0,
  //  [-0.0176371,   0,  0,  0, 0,    1,   0,
  //  [-0.00772178,  0,  0,  0, 0,    0,   1,

  SECTION("Extract Submatrix (Row/Col Inclusive)") {
    std::pair<int64_t, int64_t> lo = {1, 4};
    std::pair<int64_t, int64_t> up = {5, 6};
    auto A_sub = sparsexx::extract_submatrix(A, lo, up);
    REQUIRE(A_sub.m() == 4);
    REQUIRE(A_sub.n() == 2);
    REQUIRE(A_sub.nnz() == 3);

    REQUIRE(A_sub.rowptr().size() == 5);
    REQUIRE(A_sub.colind().size() == 3);

    const std::vector<int32_t> ref_colind = {1, 2, 1};
    const std::vector<int32_t> ref_rowptr = {1, 2, 3, 3, 4};
    const std::vector<double> ref_nzval = {0.45, .1, 1};
    CHECK(ref_colind == A_sub.colind());
    CHECK(ref_rowptr == A_sub.rowptr());

    for (int i = 0; i < 3; ++i) {
      CHECK(A_sub.nzval()[i] == Approx(ref_nzval[i]));
    }
  }

  SECTION("Extract Submatrix (Row Inclusive, Col Exclusive)") {
    std::pair<int64_t, int64_t> lo = {1, 4};
    std::pair<int64_t, int64_t> up = {5, 6};
    auto A_sub = sparsexx::extract_submatrix_inclrow_exclcol(A, lo, up);
    REQUIRE(A_sub.m() == 4);
    REQUIRE(A_sub.n() == A.n());
    REQUIRE(A_sub.nnz() == 5);

    const std::vector<int32_t> ref_colind = {2, 3, 4, 7, 1};
    const std::vector<int32_t> ref_rowptr = {1, 2, 3, 5, 6};
    const std::vector<double> ref_nzval = {-1, -1, -1, 0.45, -0.0359994};

    CHECK(ref_colind == A_sub.colind());
    CHECK(ref_rowptr == A_sub.rowptr());
    for (int i = 0; i < 3; ++i) {
      CHECK(A_sub.nzval()[i] == Approx(ref_nzval[i]));
    }
  }

  SECTION("Extract Upper Triangle") {
    auto L = sparsexx::extract_upper_triangle(A);
    REQUIRE(L.m() == A.m());
    REQUIRE(L.n() == A.n());
    REQUIRE(L.nnz() == A.nnz() - 3);

    REQUIRE(L.rowptr().size() == A.rowptr().size());
    REQUIRE(L.colind().size() == L.nnz());

    const std::vector<int32_t> ref_colind = {2, 3, 4, 2, 5, 3,
                                             6, 4, 7, 5, 6, 7};
    const std::vector<int32_t> ref_rowptr = {1, 4, 6, 8, 10, 11, 12, 13};
    const std::vector<double> ref_nzval = {1,   1,  1,    -1, 0.45, -1,
                                           0.1, -1, 0.45, 1,  1,    1};

    CHECK(ref_colind == L.colind());
    CHECK(ref_rowptr == L.rowptr());
    for (int i = 0; i < L.nnz(); ++i) {
      CHECK(L.nzval()[i] == Approx(ref_nzval[i]));
    }
  }

  SECTION("Extract Diagonal") {
    auto D = sparsexx::extract_diagonal_elements(A);
    const std::vector<double> ref_diag = {0, -1, -1, -1, 1, 1, 1};
    REQUIRE(D.size() == ref_diag.size());
    for (auto i = 0; i < D.size(); ++i) CHECK(D[i] == Approx(ref_diag[i]));
  }

  SECTION("Trace") { CHECK(sparsexx::trace(A) == Approx(0.)); }
}
