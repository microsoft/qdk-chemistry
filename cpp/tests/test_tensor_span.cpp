// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <qdk/chemistry/utils/tensor_span.hpp>
#include <vector>

namespace qdk::chemistry::test {

class TensorSpanTest : public ::testing::Test {
 protected:
  // Create a 2x2x2x2 tensor with known values for testing
  // Values are stored in column-major (Fortran) order: i + j*n + k*n^2 + l*n^3
  void SetUp() override {
    const size_t n = 2;
    data_.resize(n * n * n * n);
    for (size_t l = 0; l < n; ++l) {
      for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n; ++j) {
          for (size_t i = 0; i < n; ++i) {
            size_t idx = i + j * n + k * n * n + l * n * n * n;
            data_[idx] = static_cast<double>(idx + 1);  // 1-indexed values
          }
        }
      }
    }
  }

  std::vector<double> data_;
};

TEST_F(TensorSpanTest, MakeRank4SpanCreatesCorrectExtents) {
  const size_t n = 2;
  auto span = make_rank4_span(data_.data(), n);

  EXPECT_EQ(span.extent(0), n);
  EXPECT_EQ(span.extent(1), n);
  EXPECT_EQ(span.extent(2), n);
  EXPECT_EQ(span.extent(3), n);
}

TEST_F(TensorSpanTest, MakeRank4SpanProvidesCorrectDataHandle) {
  const size_t n = 2;
  auto span = make_rank4_span(data_.data(), n);

  EXPECT_EQ(span.data_handle(), data_.data());
}

TEST_F(TensorSpanTest, MakeRank4SpanIndexingMatchesRowMajor) {
  const size_t n = 2;
  auto span = make_rank4_span(data_.data(), n);

  // Verify indexing matches row-major layout
  // Element at [i,j,k,l] should be at linear index i*n^3 + j*n^2 + k*n + l
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        for (size_t l = 0; l < n; ++l) {
          size_t expected_idx = i * n * n * n + j * n * n + k * n + l;
          double expected_val = static_cast<double>(expected_idx + 1);
          EXPECT_EQ(span(i, j, k, l), expected_val)
              << "Mismatch at [" << i << "," << j << "," << k << "," << l
              << "]";
        }
      }
    }
  }
}

TEST_F(TensorSpanTest, ConstSpanSupportsReadAccess) {
  const size_t n = 2;
  const double* const_data = data_.data();
  auto span = make_rank4_span(const_data, n);

  // Should be able to read from const span
  EXPECT_EQ(span(0, 0, 0, 0), data_[0]);
  EXPECT_EQ(span(0, 0, 0, 1), data_[1]);
}

TEST_F(TensorSpanTest, LargerTensorWorks) {
  const size_t n = 4;
  std::vector<double> large_data(n * n * n * n);
  for (size_t i = 0; i < large_data.size(); ++i) {
    large_data[i] = static_cast<double>(i);
  }

  auto span = make_rank4_span(large_data.data(), n);

  EXPECT_EQ(span.extent(0), n);
  EXPECT_EQ(span.extent(1), n);
  EXPECT_EQ(span.extent(2), n);
  EXPECT_EQ(span.extent(3), n);

  // Test a few specific indices
  EXPECT_EQ(span(0, 0, 0, 0), 0.0);
  EXPECT_EQ(span(0, 0, 0, 1), 1.0);                           
  EXPECT_EQ(span(0, 0, 1, 0), static_cast<double>(n));        
  EXPECT_EQ(span(0, 1, 0, 0), static_cast<double>(n * n));    
  EXPECT_EQ(span(1, 0, 0, 0), static_cast<double>(n * n * n));
}

}  // namespace qdk::chemistry::test
