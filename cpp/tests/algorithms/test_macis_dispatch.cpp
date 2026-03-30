// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Unit test for the QDK dispatch_by_norb threshold table.
//
// This lives in its own file rather than test_macis.cpp because it requires a
// direct include of the private implementation header macis_base.hpp in order
// to exercise the dispatch ladder.  test_macis.cpp tests MACIS behaviour
// exclusively through the public calculator API, so this file keeps that
// boundary clean while still giving us a fast, deterministic guard against
// accidental threshold regressions (e.g. reverting the 2048-orbital limit
// back to 255).  The test runs in zero measurable time and the boundary
// cannot practically be covered through integration tests, which would
// require a 2048-active-orbital system.

#include <gtest/gtest.h>

#include "../src/qdk/chemistry/algorithms/microsoft/macis_base.hpp"

namespace {

/// Returns `N` so we can observe which bitset tier dispatch_by_norb chose
/// without instantiating any real MACIS solver.
struct DispatchProbe {
  template <size_t N>
  static constexpr size_t impl() {
    return N;
  }
};

}  // namespace

TEST(MacisDispatchTest, ThresholdTableSupportsUpTo2048Orbitals) {
  using qdk::chemistry::algorithms::microsoft::dispatch_by_norb;

  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(31), 64u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(32), 128u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(127), 256u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(255), 512u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(511), 1024u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(1023), 2048u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(1024), 4096u);
  EXPECT_EQ(dispatch_by_norb<DispatchProbe>(2048), 4096u);
  EXPECT_THROW(dispatch_by_norb<DispatchProbe>(2049), std::runtime_error);
}
