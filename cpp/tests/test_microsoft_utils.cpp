// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include "../src/qdk/chemistry/algorithms/microsoft/utils.hpp"
#include <stdexcept>

using namespace qdk::chemistry::utils::microsoft;

// Test fixture for factorial and binomial_coefficient tests
class MicrosoftUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// ========== Tests for factorial function ==========

TEST_F(MicrosoftUtilsTest, FactorialEdgeCaseZero) {
  // 0! = 1 by definition
  EXPECT_EQ(factorial(0), 1);
}

TEST_F(MicrosoftUtilsTest, FactorialEdgeCaseOne) {
  // 1! = 1
  EXPECT_EQ(factorial(1), 1);
}

TEST_F(MicrosoftUtilsTest, FactorialSmallValues) {
  // Test typical small values
  EXPECT_EQ(factorial(2), 2);     // 2! = 2
  EXPECT_EQ(factorial(3), 6);     // 3! = 6
  EXPECT_EQ(factorial(4), 24);    // 4! = 24
  EXPECT_EQ(factorial(5), 120);   // 5! = 120
  EXPECT_EQ(factorial(6), 720);   // 6! = 720
  EXPECT_EQ(factorial(7), 5040);  // 7! = 5040
}

TEST_F(MicrosoftUtilsTest, FactorialMediumValues) {
  // Test medium values that are commonly used
  EXPECT_EQ(factorial(10), 3628800);         // 10! = 3,628,800
  EXPECT_EQ(factorial(12), 479001600);       // 12! = 479,001,600
  EXPECT_EQ(factorial(15), 1307674368000);   // 15! = 1,307,674,368,000
}

TEST_F(MicrosoftUtilsTest, FactorialLargeValues) {
  // Test larger values that are still within size_t range
  // Note: 20! = 2,432,902,008,176,640,000 is the largest factorial that fits in 64-bit unsigned
  EXPECT_EQ(factorial(20), 2432902008176640000ULL);
}

TEST_F(MicrosoftUtilsTest, FactorialOverflowBoundary) {
  // Document the maximum safe input value before overflow occurs
  // For size_t (typically 64-bit unsigned), 20! is the largest factorial that fits
  // 21! = 51,090,942,171,709,440,000 > 2^64-1, so it overflows
  
  // This test documents that n=20 is the maximum safe value
  EXPECT_EQ(factorial(20), 2432902008176640000ULL);
  
  // Test that n=21 throws an overflow error
  EXPECT_THROW(factorial(21), std::overflow_error);
  
  // Test that larger values also throw
  EXPECT_THROW(factorial(25), std::overflow_error);
  EXPECT_THROW(factorial(100), std::overflow_error);
}

// ========== Tests for binomial_coefficient function ==========

TEST_F(MicrosoftUtilsTest, BinomialCoefficientEdgeCaseKZero) {
  // C(n, 0) = 1 for any n
  EXPECT_EQ(binomial_coefficient(0, 0), 1);
  EXPECT_EQ(binomial_coefficient(5, 0), 1);
  EXPECT_EQ(binomial_coefficient(10, 0), 1);
  EXPECT_EQ(binomial_coefficient(100, 0), 1);
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientEdgeCaseKEqualsN) {
  // C(n, n) = 1 for any n
  EXPECT_EQ(binomial_coefficient(0, 0), 1);
  EXPECT_EQ(binomial_coefficient(1, 1), 1);
  EXPECT_EQ(binomial_coefficient(5, 5), 1);
  EXPECT_EQ(binomial_coefficient(10, 10), 1);
  EXPECT_EQ(binomial_coefficient(50, 50), 1);
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientEdgeCaseKGreaterThanN) {
  // C(n, k) = 0 when k > n
  EXPECT_EQ(binomial_coefficient(5, 6), 0);
  EXPECT_EQ(binomial_coefficient(10, 15), 0);
  EXPECT_EQ(binomial_coefficient(0, 1), 0);
  EXPECT_EQ(binomial_coefficient(3, 10), 0);
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientSmallValues) {
  // Test typical small values
  EXPECT_EQ(binomial_coefficient(4, 2), 6);   // C(4,2) = 6
  EXPECT_EQ(binomial_coefficient(5, 2), 10);  // C(5,2) = 10
  EXPECT_EQ(binomial_coefficient(5, 3), 10);  // C(5,3) = 10
  EXPECT_EQ(binomial_coefficient(6, 3), 20);  // C(6,3) = 20
  EXPECT_EQ(binomial_coefficient(7, 3), 35);  // C(7,3) = 35
  EXPECT_EQ(binomial_coefficient(8, 4), 70);  // C(8,4) = 70
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientSymmetry) {
  // Test symmetry property: C(n, k) = C(n, n-k)
  EXPECT_EQ(binomial_coefficient(10, 3), binomial_coefficient(10, 7));
  EXPECT_EQ(binomial_coefficient(15, 5), binomial_coefficient(15, 10));
  EXPECT_EQ(binomial_coefficient(20, 8), binomial_coefficient(20, 12));
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientMediumValues) {
  // Test medium values commonly used in quantum chemistry
  EXPECT_EQ(binomial_coefficient(10, 5), 252);    // C(10,5) = 252
  EXPECT_EQ(binomial_coefficient(15, 7), 6435);   // C(15,7) = 6,435
  EXPECT_EQ(binomial_coefficient(20, 10), 184756);  // C(20,10) = 184,756
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientLargeValues) {
  // Test larger values that are still within size_t range
  // These are representative of realistic quantum chemistry problems
  EXPECT_EQ(binomial_coefficient(30, 15), 155117520);  // C(30,15) = 155,117,520
  EXPECT_EQ(binomial_coefficient(40, 20), 137846528820ULL);  // C(40,20) = 137,846,528,820
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientChemistryRelevant) {
  // Test values relevant to quantum chemistry active space calculations
  // These correspond to choosing electrons/orbitals in active spaces
  
  // Example: 6 orbitals, 3 electrons (common in small active spaces)
  EXPECT_EQ(binomial_coefficient(6, 3), 20);
  
  // Example: 10 orbitals, 5 electrons
  EXPECT_EQ(binomial_coefficient(10, 5), 252);
  
  // Example: 12 orbitals, 6 electrons
  EXPECT_EQ(binomial_coefficient(12, 6), 924);
  
  // Example: 20 orbitals, 10 electrons (larger active space)
  EXPECT_EQ(binomial_coefficient(20, 10), 184756);
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientPascalsTriangle) {
  // Verify Pascal's triangle identity: C(n,k) = C(n-1,k-1) + C(n-1,k)
  for (size_t n = 2; n <= 10; ++n) {
    for (size_t k = 1; k < n; ++k) {
      size_t left = binomial_coefficient(n - 1, k - 1);
      size_t right = binomial_coefficient(n - 1, k);
      size_t expected = binomial_coefficient(n, k);
      EXPECT_EQ(left + right, expected) 
          << "Pascal's identity failed for n=" << n << ", k=" << k;
    }
  }
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientSumOfRow) {
  // Verify that sum of row n in Pascal's triangle equals 2^n
  // C(n,0) + C(n,1) + ... + C(n,n) = 2^n
  for (size_t n = 0; n <= 10; ++n) {
    size_t sum = 0;
    for (size_t k = 0; k <= n; ++k) {
      sum += binomial_coefficient(n, k);
    }
    size_t expected = (1ULL << n);  // 2^n
    EXPECT_EQ(sum, expected) 
        << "Sum of row " << n << " should equal 2^" << n;
  }
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientOverflowProtection) {
  // Test that the implementation avoids intermediate overflow
  // by computing incrementally rather than using full factorials
  
  // These large values would overflow if computed as n!/(k!*(n-k)!)
  // but should work with the iterative approach
  EXPECT_GT(binomial_coefficient(50, 25), 0);  // Should not overflow
  EXPECT_GT(binomial_coefficient(60, 30), 0);  // Should not overflow
  
  // Document the practical limits for quantum chemistry applications
  // Most active space calculations use n <= 30, k <= 15
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientEdgeCaseN1) {
  // C(1, 0) = 1, C(1, 1) = 1
  EXPECT_EQ(binomial_coefficient(1, 0), 1);
  EXPECT_EQ(binomial_coefficient(1, 1), 1);
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientEdgeCaseK1) {
  // C(n, 1) = n for any n >= 1
  EXPECT_EQ(binomial_coefficient(1, 1), 1);
  EXPECT_EQ(binomial_coefficient(5, 1), 5);
  EXPECT_EQ(binomial_coefficient(10, 1), 10);
  EXPECT_EQ(binomial_coefficient(100, 1), 100);
}

TEST_F(MicrosoftUtilsTest, BinomialCoefficientUsedInMacisAsci) {
  // Test the specific use case from macis_asci.cpp line 77-81
  // where binomial_coefficient is used to compute FCI dimension
  
  // Example: num_molecular_orbitals=6, nalpha=3, nbeta=3
  // FCI dimension = C(6,3) * C(6,3) = 20 * 20 = 400
  size_t num_molecular_orbitals = 6;
  size_t nalpha = 3;
  size_t nbeta = 3;
  
  size_t fci_dimension = 
      binomial_coefficient(num_molecular_orbitals, nalpha) *
      binomial_coefficient(num_molecular_orbitals, nbeta);
  
  EXPECT_EQ(fci_dimension, 400);
  
  // Another example: num_molecular_orbitals=10, nalpha=5, nbeta=5
  // FCI dimension = C(10,5) * C(10,5) = 252 * 252 = 63,504
  num_molecular_orbitals = 10;
  nalpha = 5;
  nbeta = 5;
  
  fci_dimension = 
      binomial_coefficient(num_molecular_orbitals, nalpha) *
      binomial_coefficient(num_molecular_orbitals, nbeta);
  
  EXPECT_EQ(fci_dimension, 63504);
}
