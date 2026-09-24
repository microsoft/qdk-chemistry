// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <string>
#include <utility>

namespace {

namespace data = qdk::chemistry::data;
namespace utils = qdk::chemistry::utils;

class HashTestData final : public data::DataClass {
 public:
  HashTestData(double value, std::string label)
      : value_(value), label_(std::move(label)) {}

  std::string get_data_type_name() const override { return "hash_test_data"; }
  std::string get_summary() const override { return label_; }
  nlohmann::json to_json() const override {
    return {{"value", value_}, {"label", label_}};
  }
  void to_file(const std::string&, const std::string&) const override {}
  void to_json_file(const std::string&) const override {}
  void to_hdf5(H5::Group&) const override {}
  void to_hdf5_file(const std::string&) const override {}

 protected:
  void hash_update(utils::HashContext& ctx) const override {
    utils::hash_value(ctx, value_);
    utils::hash_value(ctx, label_);
  }

 private:
  double value_;
  std::string label_;
};

TEST(DataClassTest, ContentHashUsesIdentifyingFieldsAndRequestedLength) {
  const HashTestData value(7.5, "sample");
  const data::DataClass& base = value;
  utils::HashContext expected;
  utils::hash_value(expected, 7.5);
  utils::hash_value(expected, std::string("sample"));

  EXPECT_EQ(base.content_hash(), expected.hexdigest(16));
  EXPECT_EQ(base.content_hash(32), expected.hexdigest(32));
}

TEST(DataClassTest, HashValueFoldsInNestedContentHash) {
  const HashTestData child(7.5, "sample");
  utils::HashContext actual;
  data::hash_value(actual, child);
  utils::HashContext expected;
  utils::hash_value(expected, child.content_hash());

  EXPECT_EQ(actual.hexdigest(), expected.hexdigest());
}

}  // namespace
