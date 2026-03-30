// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/data/element_data.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class ElementDataTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test Element enum values
TEST_F(ElementDataTest, ElementEnumValues) {
  // Test some elements
  EXPECT_EQ(static_cast<unsigned>(Element::H), 1);
  EXPECT_EQ(static_cast<unsigned>(Element::He), 2);
  EXPECT_EQ(static_cast<unsigned>(Element::Og), 118);
}

// Test isotope() helper function
TEST_F(ElementDataTest, IsotopeHelperFunction) {
  // Test encoding of specific isotopes
  // For H-1: Z=1, A=1
  unsigned h1_value = isotope(1, 1);
  EXPECT_EQ(h1_value, 129);  // (1 << 7) + 1 = 128 + 1 = 129

  // For H-2: Z=1, A=2
  unsigned h2_value = isotope(1, 2);
  EXPECT_EQ(h2_value, 257);  // (2 << 7) + 1 = 256 + 1 = 257

  // For H-3: Z=1, A=3
  unsigned h3_value = isotope(1, 3);
  EXPECT_EQ(h3_value, 385);  // (3 << 7) + 1 = 384 + 1 = 385

  // For Og-295: Z=118, A=295
  unsigned og295_value = isotope(118, 295);
  EXPECT_EQ(og295_value, 37878);  // (295 << 7) + 118 = 37760 + 118 = 37878
}

// Test that isotope values encode Z in lower 7 bits
TEST_F(ElementDataTest, IsotopeEncodingZExtraction) {
  // Extract Z from isotope encoded values
  unsigned h1_encoded = isotope(1, 1);
  unsigned z_h1 = h1_encoded & 0x7F;
  EXPECT_EQ(z_h1, 1);

  unsigned h2_encoded = isotope(1, 2);
  unsigned z_h2 = h2_encoded & 0x7F;
  EXPECT_EQ(z_h2, 1);

  unsigned h3_encoded = isotope(1, 3);
  unsigned z_h3 = h3_encoded & 0x7F;
  EXPECT_EQ(z_h3, 1);

  unsigned og295_encoded = isotope(118, 295);
  unsigned z_og295 = og295_encoded & 0x7F;
  EXPECT_EQ(z_og295, 118);
}

// Test CHARGE_TO_SYMBOL map
TEST_F(ElementDataTest, ChargeToSymbol) {
  EXPECT_EQ(CHARGE_TO_SYMBOL.at(1), "H");
  EXPECT_EQ(CHARGE_TO_SYMBOL.at(2), "He");
  EXPECT_EQ(CHARGE_TO_SYMBOL.at(118), "Og");
}

// Test that all elements from 1-118 have symbols
TEST_F(ElementDataTest, AllElementsHaveSymbols) {
  for (unsigned z = 1; z <= 118; ++z) {
    EXPECT_NO_THROW(CHARGE_TO_SYMBOL.at(z));
    EXPECT_FALSE(CHARGE_TO_SYMBOL.at(z).empty());
  }
}

// Test get_atomic_weight with Element enum
TEST_F(ElementDataTest, GetAtomicWeightFromElement) {
  using namespace ciaaw_2024;

  // Test that Element values return standard atomic weights
  EXPECT_DOUBLE_EQ(get_atomic_weight(Element::H), 1.0080);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Element::He), 4.0026);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Element::Og), 294.0);
}

// Test standard atomic weights
TEST_F(ElementDataTest, GetStandardAtomicWeights) {
  using namespace ciaaw_2024;

  // Test standard atomic weights (using Z, A=0)
  EXPECT_DOUBLE_EQ(get_atomic_weight(1, 0), 1.0080);   // H
  EXPECT_DOUBLE_EQ(get_atomic_weight(2, 0), 4.0026);   // He
  EXPECT_DOUBLE_EQ(get_atomic_weight(118, 0), 294.0);  // Og
}

// Test CIAAW 2024 updated values (Gd, Lu, Zr as mentioned in documentation)
TEST_F(ElementDataTest, CIAAW2024UpdatedWeights) {
  using namespace ciaaw_2024;

  // Test updated values for Gd (64), Lu (71), Zr (40)
  EXPECT_DOUBLE_EQ(get_atomic_weight(64, 0), 157.25);  // Gd
  EXPECT_DOUBLE_EQ(get_atomic_weight(71, 0), 174.97);  // Lu
  EXPECT_DOUBLE_EQ(get_atomic_weight(40, 0), 91.222);  // Zr
}

// Test specific isotope masses
TEST_F(ElementDataTest, GetSpecificIsotopeMasses) {
  using namespace ciaaw_2024;

  // Test specific isotope masses
  EXPECT_DOUBLE_EQ(get_atomic_weight(1, 1), 1.007825032);     // H-1
  EXPECT_DOUBLE_EQ(get_atomic_weight(1, 2), 2.014101778);     // H-2 (Deuterium)
  EXPECT_DOUBLE_EQ(get_atomic_weight(1, 3), 3.016049281);     // H-3 (Tritium)
  EXPECT_DOUBLE_EQ(get_atomic_weight(118, 295), 295.216178);  // Og-295
}

// Test error handling for unknown element/isotope
TEST_F(ElementDataTest, UnknownElementIsotopeThrows) {
  using namespace ciaaw_2024;

  // Test with invalid atomic number
  EXPECT_THROW(get_atomic_weight(120, 0), std::invalid_argument);

  // Test with valid Z but invalid A
  EXPECT_THROW(get_atomic_weight(1, 999), std::invalid_argument);
}

// Test that all elements 1-118 have atomic weights
TEST_F(ElementDataTest, AllElementsHaveAtomicWeights) {
  using namespace ciaaw_2024;

  for (unsigned z = 1; z <= 118; ++z) {
    Element element = static_cast<Element>(z);
    EXPECT_NO_THROW(get_atomic_weight(element));
    double weight = get_atomic_weight(element);
    EXPECT_GT(weight, 0.0);
    EXPECT_LT(weight, 300.0);  // All atomic weights should be less than 300 AMU
  }
}

// Test CIAAW version string
TEST_F(ElementDataTest, CIAAWVersion) {
  const char* version = get_current_ciaaw_version();
  EXPECT_STREQ(version, "CIAAW 2024");
}

// Test that standard atomic weights are reasonable
TEST_F(ElementDataTest, AtomicWeightsAreReasonable) {
  using namespace ciaaw_2024;

  // Check that atomic weights generally increase with atomic number
  // (with some exceptions due to isotopic abundances)
  for (unsigned z = 1; z < 118; ++z) {
    double weight_z = get_atomic_weight(static_cast<Element>(z));
    double weight_z_plus_1 = get_atomic_weight(static_cast<Element>(z + 1));

    // Most elements should have increasing atomic weights
    // Allow for exceptions (like Te/I, Co/Ni, Ar/K)
    if (z != 18 && z != 27 && z != 52) {  // Ar, Co, Te exceptions
      EXPECT_LE(weight_z, weight_z_plus_1 + 2.0)
          << "Unexpected weight ordering at Z=" << z;
    }
  }
}
