// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "util/cache.h"

#include <string>

#include "gtest/gtest.h"

using namespace qdk::chemistry::scf;

struct Dummy {
  int x;
  std::string y;
  Dummy(int x_, std::string y_) : x(x_), y(std::move(y_)) {}
};

TEST(CacheTest, InsertAndRetrieve) {
  util::Cache<Dummy> cache;
  Dummy* obj = cache.emplace(1, 42, "bar");
  ASSERT_NE(obj, nullptr);
  EXPECT_EQ(obj->x, 42);
  EXPECT_EQ(obj->y, "bar");
  // Retrieval should return the same pointer
  Dummy* obj2 = cache.get(1);
  EXPECT_EQ(obj, obj2);
}

TEST(CacheTest, NotFound) {
  util::Cache<Dummy> cache;
  EXPECT_EQ(cache.get(999), nullptr);
}

TEST(CacheTest, EraseAndClear) {
  util::Cache<Dummy> cache;
  cache.emplace(1, 1, "one");
  cache.emplace(2, 2, "two");
  cache.erase(1);
  EXPECT_EQ(cache.get(1), nullptr);
  EXPECT_NE(cache.get(2), nullptr);
  cache.clear();
  EXPECT_EQ(cache.get(2), nullptr);
}

TEST(CacheTest, FindByIdentifierAndOperator) {
  util::Cache<Dummy> cache;
  std::string key = "hello";
  auto idx = util::hash_index(key);
  cache.emplace(idx, 123, "world");
  // Test find_by_identifier
  Dummy* found = cache.find_by_identifier(key);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->x, 123);
  EXPECT_EQ(found->y, "world");
  // Test operator[]
  Dummy* found2 = cache[key];
  EXPECT_EQ(found2, found);
  // Test const version
  const util::Cache<Dummy>& ccache = cache;
  const Dummy* cfound = ccache.find_by_identifier(key);
  EXPECT_EQ(cfound, found);
  const Dummy* cfound2 = ccache[key];
  EXPECT_EQ(cfound2, found);
}
