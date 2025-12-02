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

// Test Isotope enum values for atomic weights
TEST_F(ElementDataTest, IsotopeEnumStandardValues) {
  // Standard isotopes should match Element enum values
  EXPECT_EQ(static_cast<unsigned>(Isotope::H), 1);
  EXPECT_EQ(static_cast<unsigned>(Isotope::He), 2);
  EXPECT_EQ(static_cast<unsigned>(Isotope::Og), 118);
}

// Test isotope() helper function
TEST_F(ElementDataTest, IsotopeHelperFunction) {
  // Test encoding of specific isotopes
  // For H-1: Z=1, A=1
  unsigned h1_value = isotope(1, 1);
  EXPECT_EQ(static_cast<unsigned>(Isotope::H1), h1_value);

  // For H-2: Z=1, A=2
  unsigned h2_value = isotope(1, 2);
  EXPECT_EQ(static_cast<unsigned>(Isotope::H2), h2_value);

  // For D: Z=1, A=2
  unsigned d_value = isotope(1, 2);
  EXPECT_EQ(static_cast<unsigned>(Isotope::D), d_value);

  // For T: Z=1, A=3
  unsigned t_value = isotope(1, 3);
  EXPECT_EQ(static_cast<unsigned>(Isotope::T), t_value);

  // For Og-295: Z=118, A=295
  unsigned og295_value = isotope(118, 295);
  EXPECT_EQ(static_cast<unsigned>(Isotope::Og295), og295_value);
}

// Test that isotope values encode Z in lower 7 bits
TEST_F(ElementDataTest, IsotopeEncodingZExtraction) {
  // Extract Z from isotope enum values
  unsigned z_h1 = static_cast<unsigned>(Isotope::H1) & 0x7F;
  EXPECT_EQ(z_h1, 1);

  unsigned z_h2 = static_cast<unsigned>(Isotope::H2) & 0x7F;
  EXPECT_EQ(z_h2, 1);

  unsigned z_d = static_cast<unsigned>(Isotope::D) & 0x7F;
  EXPECT_EQ(z_d, 1);

  unsigned z_t = static_cast<unsigned>(Isotope::T) & 0x7F;
  EXPECT_EQ(z_t, 1);

  unsigned z_og295 = static_cast<unsigned>(Isotope::Og295) & 0x7F;
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

// Test CIAAW 2024 atomic weights for standard elements
TEST_F(ElementDataTest, CIAAW2024StandardAtomicWeights) {
  using namespace ciaaw_2024;

  // Test some well-known standard atomic weights
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::H), 1.0080);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::He), 4.0026);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::U), 238.03);
}

// Test CIAAW 2024 updated values (Gd, Lu, Zr as mentioned in documentation)
TEST_F(ElementDataTest, CIAAW2024UpdatedWeights) {
  using namespace ciaaw_2024;

  // Test updated values for Gd (64), Lu (71), Zr (40)
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::Gd), 157.25);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::Lu), 174.97);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::Zr), 91.222);
}

// Test specific isotope masses
TEST_F(ElementDataTest, SpecificIsotopeMasses) {
  using namespace ciaaw_2024;

  // Test hydrogen isotopes
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::H1), 1.007825032);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::H2), 2.014101778);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::D), 2.014101778);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::T), 3.016049281);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Isotope::Og295), 295.216178);
}

// Test get_atomic_weight with Element enum
TEST_F(ElementDataTest, GetAtomicWeightFromElement) {
  using namespace ciaaw_2024;

  // Test that Element values can be converted to Isotope and return standard
  // atomic weights
  EXPECT_DOUBLE_EQ(get_atomic_weight(Element::H), 1.0080);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Element::He), 4.0026);
  EXPECT_DOUBLE_EQ(get_atomic_weight(Element::Og), 294.0);
}

// Test error handling for unknown isotope
TEST_F(ElementDataTest, UnknownIsotopeThrows) {
  using namespace ciaaw_2024;

  // Create an invalid isotope value
  Isotope invalid_isotope = static_cast<Isotope>(120);

  EXPECT_THROW(get_atomic_weight(invalid_isotope), std::invalid_argument);
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
