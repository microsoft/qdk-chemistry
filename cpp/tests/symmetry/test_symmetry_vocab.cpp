// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <stdexcept>

using namespace qdk::chemistry::data;

TEST(SymmetryVocabTest, SpinValueBasics) {
  EXPECT_EQ(axes::alpha()->value(), 1);
  EXPECT_EQ(axes::beta()->value(), -1);
  EXPECT_TRUE(axes::alpha()->equals(*axes::spin_value(1)));
  EXPECT_FALSE(axes::alpha()->equals(*axes::beta()));
  EXPECT_EQ(axes::alpha()->axis(), AxisName::Spin);
  EXPECT_EQ(to_string(axes::alpha()->axis()), "spin");
}

TEST(SymmetryVocabTest, InternedAlphaBeta) {
  // The interned alpha/beta shared_ptrs are reused.
  EXPECT_EQ(axes::spin_value(1).get(), axes::alpha().get());
  EXPECT_EQ(axes::spin_value(-1).get(), axes::beta().get());
}

TEST(SymmetryVocabTest, SpinAxisAdmits) {
  SymmetryAxis spin = axes::spin(/*two_s=*/1, /*equivalent=*/true);
  EXPECT_EQ(spin.name(), AxisName::Spin);
  EXPECT_TRUE(spin.equivalent());
  EXPECT_TRUE(spin.admits(*axes::alpha()));
  EXPECT_TRUE(spin.admits(*axes::beta()));
  EXPECT_FALSE(spin.admits(*axes::spin_value(3)));
}

TEST(SymmetryVocabTest, SymmetriesAxisLookup) {
  SymmetryProduct sym({axes::spin(1, true)});
  EXPECT_TRUE(sym.has_axis(AxisName::Spin));
  EXPECT_EQ(sym.axis(AxisName::Spin).name(), AxisName::Spin);
}

TEST(SymmetryVocabTest, DuplicateAxisRejected) {
  EXPECT_THROW(SymmetryProduct({axes::spin(1, true), axes::spin(1, false)}),
               std::runtime_error);
}

TEST(SymmetryVocabTest, SymmetryLabelOnePerAxis) {
  SymmetryLabel label({axes::alpha()});
  EXPECT_TRUE(label.has(AxisName::Spin));
  EXPECT_EQ(label.get(AxisName::Spin)->equals(*axes::alpha()), true);
  EXPECT_THROW(SymmetryLabel({axes::alpha(), axes::beta()}),
               std::runtime_error);
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
  SymmetryProduct a({axes::spin(1, true)});
  SymmetryProduct b({axes::spin(1, true)});
  SymmetryProduct c({axes::spin(1, false)});
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(SymmetryVocabTest, RoundTripSpinValueJson) {
  auto json = axes::alpha()->to_json();
  auto value = symmetry_axis_value_from_json(json);
  EXPECT_TRUE(value->equals(*axes::alpha()));
}

TEST(SymmetryVocabTest, RoundTripSymmetriesJson) {
  SymmetryProduct sym({axes::spin(1, true)});
  auto restored = SymmetryProduct::from_json(sym.to_json());
  EXPECT_EQ(sym, *restored);
}

TEST(SymmetryVocabTest, RoundTripSymmetryLabelJson) {
  SymmetryLabel label({axes::beta()});
  auto restored = SymmetryLabel::from_json(label.to_json());
  EXPECT_EQ(label, restored);
}

TEST(SymmetryVocabTest, UnknownKindThrows) {
  nlohmann::json bad = {{"kind", "not_a_real_kind"}};
  EXPECT_THROW(symmetry_axis_value_from_json(bad), std::runtime_error);
}

TEST(SymmetryVocabTest, SymmetryAxisDataClassMetadata) {
  SymmetryAxis spin = axes::spin(1, true);
  EXPECT_EQ(spin.get_data_type_name(), "symmetry_axis");
  EXPECT_NE(spin.get_summary().find("SymmetryAxis"), std::string::npos);
  EXPECT_NE(spin.get_summary().find("spin"), std::string::npos);
}

TEST(SymmetryVocabTest, SymmetriesDataClassMetadata) {
  SymmetryProduct sym({axes::spin(1, true)});
  EXPECT_EQ(sym.get_data_type_name(), "symmetry_product");
  EXPECT_NE(sym.get_summary().find("SymmetryProduct"), std::string::npos);
}

TEST(SymmetryVocabTest, SymmetryAxisRoundTripJson) {
  SymmetryAxis original = axes::spin(1, true);
  auto restored = SymmetryAxis::from_json(original.to_json());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(original, *restored);
}

TEST(SymmetryVocabTest, SymmetriesRoundTripJson) {
  SymmetryProduct original({axes::spin(1, true)});
  auto restored = SymmetryProduct::from_json(original.to_json());
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(original, *restored);
}

TEST(SymmetryVocabTest, SymmetryAxisRoundTripJsonFile) {
  const std::filesystem::path filename = "test.symmetry_axis.json";
  std::filesystem::remove(filename);
  SymmetryAxis original = axes::spin(1, true);
  original.to_json_file(filename.string());
  auto restored = SymmetryAxis::from_json_file(filename.string());
  std::filesystem::remove(filename);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(original, *restored);
}

TEST(SymmetryVocabTest, SymmetryAxisRoundTripHdf5File) {
  const std::filesystem::path filename = "test.symmetry_axis.h5";
  std::filesystem::remove(filename);
  SymmetryAxis original = axes::spin(1, true);
  original.to_hdf5_file(filename.string());
  auto restored = SymmetryAxis::from_hdf5_file(filename.string());
  std::filesystem::remove(filename);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(original, *restored);
}

TEST(SymmetryVocabTest, SymmetriesRoundTripJsonFile) {
  const std::filesystem::path filename = "test.symmetries.json";
  std::filesystem::remove(filename);
  SymmetryProduct original({axes::spin(1, true)});
  original.to_json_file(filename.string());
  auto restored = SymmetryProduct::from_json_file(filename.string());
  std::filesystem::remove(filename);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(original, *restored);
}

TEST(SymmetryVocabTest, SymmetriesRoundTripHdf5File) {
  const std::filesystem::path filename = "test.symmetries.h5";
  std::filesystem::remove(filename);
  SymmetryProduct original({axes::spin(1, true)});
  original.to_hdf5_file(filename.string());
  auto restored = SymmetryProduct::from_hdf5_file(filename.string());
  std::filesystem::remove(filename);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(original, *restored);
}

TEST(SymmetryVocabTest, SymmetryAxisToFileDispatch) {
  const std::filesystem::path json_filename =
      "test.dispatch.symmetry_axis.json";
  const std::filesystem::path h5_filename = "test.dispatch.symmetry_axis.h5";
  std::filesystem::remove(json_filename);
  std::filesystem::remove(h5_filename);
  SymmetryAxis original = axes::spin(1, true);

  original.to_file(json_filename.string(), "json");
  original.to_file(h5_filename.string(), "hdf5");
  auto via_json = SymmetryAxis::from_file(json_filename.string(), "json");
  auto via_h5 = SymmetryAxis::from_file(h5_filename.string(), "hdf5");
  std::filesystem::remove(json_filename);
  std::filesystem::remove(h5_filename);

  ASSERT_NE(via_json, nullptr);
  ASSERT_NE(via_h5, nullptr);
  EXPECT_EQ(original, *via_json);
  EXPECT_EQ(original, *via_h5);

  EXPECT_THROW(original.to_file("ignored.dat", "xml"), std::invalid_argument);
  EXPECT_THROW(SymmetryAxis::from_file("ignored.dat", "xml"),
               std::invalid_argument);
}

TEST(SymmetryVocabTest, SymmetryAxisFromJsonRejectsMissingVersion) {
  auto j = axes::spin(1, true).to_json();
  j.erase("version");
  EXPECT_THROW(SymmetryAxis::from_json(j), std::runtime_error);
}

TEST(SymmetryVocabTest, SymmetryAxisFromJsonRejectsMismatchedVersion) {
  auto j = axes::spin(1, true).to_json();
  j["version"] = "99.0.0";
  EXPECT_THROW(SymmetryAxis::from_json(j), std::runtime_error);
}

TEST(SymmetryVocabTest, SymmetriesFromJsonRejectsMissingVersion) {
  SymmetryProduct sym({axes::spin(1, true)});
  auto j = sym.to_json();
  j.erase("version");
  EXPECT_THROW(SymmetryProduct::from_json(j), std::runtime_error);
}

TEST(SymmetryVocabTest, SymmetriesFromJsonRejectsMismatchedVersion) {
  SymmetryProduct sym({axes::spin(1, true)});
  auto j = sym.to_json();
  j["version"] = "99.0.0";
  EXPECT_THROW(SymmetryProduct::from_json(j), std::runtime_error);
}
