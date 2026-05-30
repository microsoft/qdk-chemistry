// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/data/errors.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>

using namespace qdk::chemistry::data;

TEST(SymmetryVocabTest, SpinValueBasics) {
  EXPECT_EQ(axes::alpha()->value(), 1);
  EXPECT_EQ(axes::beta()->value(), -1);
  EXPECT_TRUE(axes::alpha()->equals(*axes::spin_value(1)));
  EXPECT_FALSE(axes::alpha()->equals(*axes::beta()));
  EXPECT_EQ(axes::alpha()->axis(), AxisName::Spin);
  EXPECT_EQ(axes::alpha()->kind_name(), "spin");
}

TEST(SymmetryVocabTest, InternedAlphaBeta) {
  // The interned alpha/beta shared_ptrs are reused.
  EXPECT_EQ(axes::spin_value(1).get(), axes::alpha().get());
  EXPECT_EQ(axes::spin_value(-1).get(), axes::beta().get());
}

TEST(SymmetryVocabTest, SpinAxisAdmits) {
  SymmetryAxis spin = axes::spin(/*two_s=*/0, /*equivalent=*/true);
  EXPECT_EQ(spin.name(), AxisName::Spin);
  EXPECT_TRUE(spin.equivalent());
  EXPECT_TRUE(spin.admits(*axes::alpha()));
  EXPECT_TRUE(spin.admits(*axes::beta()));
  EXPECT_FALSE(spin.admits(*axes::spin_value(3)));
}

TEST(SymmetryVocabTest, SymmetriesAxisLookup) {
  Symmetries sym({axes::spin(0, true)});
  EXPECT_TRUE(sym.has_axis(AxisName::Spin));
  EXPECT_FALSE(sym.has_axis(AxisName::PointGroup));
  EXPECT_EQ(sym.axis(AxisName::Spin).name(), AxisName::Spin);
  EXPECT_THROW(sym.axis(AxisName::PointGroup), SymmetryConditionError);
}

TEST(SymmetryVocabTest, DuplicateAxisRejected) {
  EXPECT_THROW(Symmetries({axes::spin(0, true), axes::spin(0, false)}),
               SymmetryConditionError);
}

TEST(SymmetryVocabTest, SymmetryLabelOnePerAxis) {
  SymmetryLabel label({axes::alpha()});
  EXPECT_TRUE(label.has(AxisName::Spin));
  EXPECT_EQ(label.get(AxisName::Spin)->equals(*axes::alpha()), true);
  EXPECT_THROW(label.get(AxisName::PointGroup), SymmetryConditionError);
  EXPECT_THROW(SymmetryLabel({axes::alpha(), axes::beta()}),
               SymmetryConditionError);
}

TEST(SymmetryVocabTest, SymmetryLabelEqualityAndHash) {
  SymmetryLabel a({axes::alpha()});
  SymmetryLabel b({axes::spin_value(1)});
  SymmetryLabel c({axes::beta()});
  EXPECT_EQ(a, b);
  EXPECT_EQ(a.hash(), b.hash());
  EXPECT_NE(a, c);
  EXPECT_EQ(std::hash<SymmetryLabel>{}(a), a.hash());
}

TEST(SymmetryVocabTest, SymmetriesEquality) {
  Symmetries a({axes::spin(0, true)});
  Symmetries b({axes::spin(0, true)});
  Symmetries c({axes::spin(0, false)});
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(SymmetryVocabTest, RoundTripSpinValueJson) {
  auto json = axes::alpha()->to_json();
  auto value = symmetry_axis_value_from_json(json);
  EXPECT_TRUE(value->equals(*axes::alpha()));
}

TEST(SymmetryVocabTest, RoundTripSymmetriesJson) {
  Symmetries sym({axes::spin(0, true)});
  auto restored = Symmetries::from_json(sym.to_json());
  EXPECT_EQ(sym, restored);
}

TEST(SymmetryVocabTest, RoundTripSymmetryLabelJson) {
  SymmetryLabel label({axes::beta()});
  auto restored = SymmetryLabel::from_json(label.to_json());
  EXPECT_EQ(label, restored);
}

TEST(SymmetryVocabTest, UnknownKindThrows) {
  nlohmann::json bad = {{"kind", "not_a_real_kind"}};
  EXPECT_THROW(symmetry_axis_value_from_json(bad), SymmetryConditionError);
}
