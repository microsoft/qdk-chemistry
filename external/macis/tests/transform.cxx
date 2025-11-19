/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <macis/util/transform.hpp>
#include <numeric>

#include "ut_common.hpp"

#define FOUR_IDX(arr, i, j, k, l, n) arr[i + j * n + k * n * n + l * n * n * n]
#define TWO_IDX(arr, i, j, n) arr[i + j * n]

TEST_CASE("One Index Transform") {
  // A = [ 1 2  3  4 ]
  //     [ 5 6  7  8 ]
  //     [ 9 10 11 12]
  //     [13 14 15 16]
  size_t n = 4;
  std::vector<double> A = {// col major
                           1.0, 5.0, 9.0,  13.0, 2.0, 6.0, 10.0, 14.0,
                           3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0};

  SECTION("Identity Transform") {
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> B(n * n);
    macis::two_index_transform(n, n, A.data(), n, C.data(), n, B.data(), n);

    for (auto i = 0; i < n * n; ++i)
      REQUIRE_THAT(A[i], Catch::Matchers::WithinAbs(
                             B[i], testing::numerical_zero_tolerance));
  }

  SECTION("Permutation") {
    // C = [ 1 0 0 0 ]
    //     [ 0 0 0 1 ]
    //     [ 0 1 0 0 ]
    //     [ 0 0 1 0 ]
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<double> B(A.size());
    macis::two_index_transform(n, n, A.data(), n, C.data(), n, B.data(), n);

    std::vector<double> refB = {1, 9,  13, 5, 3, 11, 15, 7,
                                4, 12, 16, 8, 2, 10, 14, 6};

    for (auto i = 0; i < n * n; ++i)
      REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(
                             refB[i], testing::numerical_zero_tolerance));
  }

  SECTION("Submatrix") {
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                             0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    size_t m = 3;
    std::vector<double> B(m * m);
    macis::two_index_transform(n, m, A.data(), n, C.data(), n, B.data(), m);

    std::vector<double> refB = {1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0};

    for (auto i = 0; i < m * m; ++i)
      REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(
                             refB[i], testing::numerical_zero_tolerance));
  }

  SECTION("Sub Permutation") {
    // C = [ 1 0 0 ]
    //     [ 0 0 1 ]
    //     [ 0 0 0 ]
    //     [ 0 1 0 ]
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    size_t m = 3;
    std::vector<double> B(m * m);
    macis::two_index_transform(n, m, A.data(), n, C.data(), n, B.data(), m);

    std::vector<double> refB = {1, 13, 5, 4, 16, 8, 2, 14, 6};

    for (auto i = 0; i < m * m; ++i)
      REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(
                             refB[i], testing::numerical_zero_tolerance));
  }
}

TEST_CASE("Four Index Transform") {
  size_t n = 4;
  size_t n2 = n * n;
  size_t n3 = n2 * n;
  size_t n4 = n2 * n2;
  std::vector<double> A(n4);
  std::iota(A.begin(), A.end(), 1);

  SECTION("Identity Transform") {
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> B(n4);
    macis::four_index_transform(n, n, A.data(), n, C.data(), n, B.data(), n);

    for (size_t i = 0; i < n4; ++i)
      REQUIRE_THAT(A[i], Catch::Matchers::WithinAbs(
                             B[i], testing::numerical_zero_tolerance));
  }

  SECTION("Permutation") {
    // C = [ 1 0 0 0
    //       0 0 0 1
    //       0 1 0 0
    //       0 0 1 0 ]
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<double> B(n4);
    macis::four_index_transform(n, n, A.data(), n, C.data(), n, B.data(), n);

    std::vector<double> refB(n4);
    for (size_t p = 0; p < n; ++p)
      for (size_t q = 0; q < n; ++q)
        for (size_t r = 0; r < n; ++r)
          for (size_t s = 0; s < n; ++s) {
            for (size_t i = 0; i < n; ++i)
              for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                  for (size_t l = 0; l < n; ++l) {
                    FOUR_IDX(refB, p, q, r, s, n) +=
                        TWO_IDX(C, i, p, n) * TWO_IDX(C, j, q, n) *
                        TWO_IDX(C, k, r, n) * TWO_IDX(C, l, s, n) *
                        FOUR_IDX(A, i, j, k, l, n);
                  }
          }

    for (auto i = 0; i < A.size(); ++i)
      REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(
                             refB[i], testing::numerical_zero_tolerance));
  }

  SECTION("Subtensor") {
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                             0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    size_t m = 3;
    size_t m2 = m * m;
    size_t m3 = m2 * m;
    size_t m4 = m2 * m2;
    std::vector<double> B(m4);
    macis::four_index_transform(n, m, A.data(), n, C.data(), n, B.data(), m);

    std::vector<double> refB(m4);
    for (size_t p = 0; p < m; ++p)
      for (size_t q = 0; q < m; ++q)
        for (size_t r = 0; r < m; ++r)
          for (size_t s = 0; s < m; ++s) {
            for (size_t i = 0; i < n; ++i)
              for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                  for (size_t l = 0; l < n; ++l) {
                    FOUR_IDX(refB, p, q, r, s, m) +=
                        TWO_IDX(C, i, p, n) * TWO_IDX(C, j, q, n) *
                        TWO_IDX(C, k, r, n) * TWO_IDX(C, l, s, n) *
                        FOUR_IDX(A, i, j, k, l, n);
                  }
          }

    for (auto i = 0; i < m4; ++i)
      REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(
                             refB[i], testing::numerical_zero_tolerance));
  }

  SECTION("Sub Permutation") {
    // C = [ 1 0 0 ]
    //     [ 0 0 1 ]
    //     [ 0 0 0 ]
    //     [ 0 1 0 ]
    std::vector<double> C = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    size_t m = 3;
    size_t m2 = m * m;
    size_t m3 = m2 * m;
    size_t m4 = m2 * m2;
    std::vector<double> B(m4);
    macis::four_index_transform(n, m, A.data(), n, C.data(), n, B.data(), m);

    std::vector<double> refB(m4);
    for (size_t p = 0; p < m; ++p)
      for (size_t q = 0; q < m; ++q)
        for (size_t r = 0; r < m; ++r)
          for (size_t s = 0; s < m; ++s) {
            for (size_t i = 0; i < n; ++i)
              for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                  for (size_t l = 0; l < n; ++l) {
                    FOUR_IDX(refB, p, q, r, s, m) +=
                        TWO_IDX(C, i, p, n) * TWO_IDX(C, j, q, n) *
                        TWO_IDX(C, k, r, n) * TWO_IDX(C, l, s, n) *
                        FOUR_IDX(A, i, j, k, l, n);
                  }
          }

    for (auto i = 0; i < m4; ++i)
      REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(
                             refB[i], testing::numerical_zero_tolerance));
  }
}
