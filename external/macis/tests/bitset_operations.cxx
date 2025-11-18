/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <macis/bitset_operations.hpp>

#include "ut_common.hpp"

template <size_t N>
using bs = std::bitset<N>;

template <size_t N>
void uint128_conversion_test() {
  using ref_type = macis::uint128_t;
  std::bitset<N> one_bs{1};
  std::bitset<N> val_bs{0};
  val_bs |= (one_bs << 4);
  if (N > 32) val_bs |= (one_bs << 33);
  if (N > 64) {
    val_bs |= (one_bs << 65);
    val_bs |= (one_bs << 115);
  }

  ref_type one{1};
  ref_type val{0};
  val |= (one << 4);
  if (N > 32) val |= (one << 33);
  if (N > 64) {
    val |= (one << 65);
    val |= (one << 115);
  }

  auto val_cmp = macis::to_uint128(val_bs);
  REQUIRE(val_cmp == val);
}

template <size_t N>
void mask_test() {
  bs<N> full = ~bs<N>(0);
  bs<N> full_128 = N > 128 ? full >> (N - 128) : 0;
  bs<N> full_64 = N > 64 ? full >> (N - 64) : 0;
  bs<N> full_32 = N > 32 ? full >> (N - 32) : 0;

  REQUIRE(full == macis::full_mask<N>());
  if constexpr (N > 128) REQUIRE(full_128 == macis::full_mask<128, N>());
  if constexpr (N > 64) REQUIRE(full_64 == macis::full_mask<64, N>());
  if constexpr (N > 32) REQUIRE(full_32 == macis::full_mask<32, N>());

  REQUIRE(full == macis::full_mask<N>(N));
  if constexpr (N > 128) REQUIRE(full_128 == macis::full_mask<N>(128));
  if constexpr (N > 64) REQUIRE(full_64 == macis::full_mask<N>(64));
  if constexpr (N > 32) REQUIRE(full_32 == macis::full_mask<N>(32));
}

template <size_t N>
void ffs_test() {
  REQUIRE(macis::ffs(bs<N>()) == 0);
  REQUIRE(macis::ffs(bs<N>(1)) == 1);

  bs<N> a(0xABC);
  REQUIRE(macis::ffs(a) == 3);
  if (N > 32) REQUIRE(macis::ffs(a << 32) == 3 + 32);
  if (N > 64) REQUIRE(macis::ffs(a << 64) == 3 + 64);
  if (N > 128) REQUIRE(macis::ffs(a << 128) == 3 + 128);
}

template <size_t N>
void fls_test() {
  REQUIRE(macis::fls(bs<N>()) == UINT32_MAX);
  REQUIRE(macis::fls(bs<N>(1)) == 0);

  bs<N> a(0xDEADBEEF);
  REQUIRE(macis::fls(a) == 31);
  if (N > 32) REQUIRE(macis::fls(a << 32) == 31 + 32);
  if (N > 64) REQUIRE(macis::fls(a << 64) == 31 + 64);
  if (N > 128) REQUIRE(macis::fls(a << 128) == 31 + 128);
}

template <size_t N>
void indices_test() {
  std::vector<uint32_t> indices = {5};
  if (N > 32) indices.emplace_back(6 + 32);
  if (N > 64) indices.emplace_back(7 + 64);
  if (N > 128) indices.emplace_back(8 + 128);

  bs<N> one(1), a(0);
  for (auto i : indices) a |= one << i;
  REQUIRE_THAT(macis::bits_to_indices(a), Catch::Matchers::Equals(indices));
}

template <size_t N>
void lo_word_test() {
  bs<N> a(0);
  bs<N / 2> b(0);
  size_t n_32_words = N / 32;
  for (int i = 0; i < n_32_words; ++i) {
    a |= bs<N>(0xDEADBEEF + i) << (i * 32);
  }
  for (int i = 0; i < n_32_words / 2; ++i) {
    b |= bs<N / 2>(0xDEADBEEF + i) << (i * 32);
  }

  REQUIRE(macis::bitset_lo_word(a) == b);

  bs<N> c(0);
  macis::set_bitset_lo_word(c, b);
  REQUIRE((a << N / 2) >> N / 2 == c);
}

template <size_t N>
void hi_word_test() {
  bs<N> a(0);
  bs<N / 2> b(0);
  size_t n_32_words = N / 32;
  for (int i = 0; i < n_32_words; ++i) {
    a |= bs<N>(0xDEADBEEF + i) << (i * 32);
  }
  for (int i = 0; i < n_32_words / 2; ++i) {
    b |= bs<N / 2>(0xDEADBEEF + (i + n_32_words / 2)) << (i * 32);
  }

  REQUIRE(macis::bitset_hi_word(a) == b);

  bs<N> c(0);
  macis::set_bitset_hi_word(c, b);
  REQUIRE((a >> N / 2) << N / 2 == c);
}

template <size_t N>
void bitset_less_test() {
  bs<N> a(65), b(42);
  REQUIRE(macis::bitset_less(b, a));
  REQUIRE_FALSE(macis::bitset_less(a, a));
  REQUIRE_FALSE(macis::bitset_less(a, b));

  if (N > 32) {
    auto aa = a << 32;
    auto bb = b << 32;
    REQUIRE(macis::bitset_less(bb, aa));
    REQUIRE_FALSE(macis::bitset_less(aa, aa));
    REQUIRE_FALSE(macis::bitset_less(aa, bb));
    REQUIRE_FALSE(macis::bitset_less(aa, b));
    REQUIRE_FALSE(macis::bitset_less(aa, a));
  }

  if (N > 64) {
    auto aa = a << 64;
    auto bb = b << 64;
    REQUIRE(macis::bitset_less(bb, aa));
    REQUIRE_FALSE(macis::bitset_less(aa, aa));
    REQUIRE_FALSE(macis::bitset_less(aa, bb));
    REQUIRE_FALSE(macis::bitset_less(aa, b));
    REQUIRE_FALSE(macis::bitset_less(aa, a));
  }

  if (N > 128) {
    auto aa = a << 128;
    auto bb = b << 128;
    REQUIRE(macis::bitset_less(bb, aa));
    REQUIRE_FALSE(macis::bitset_less(aa, aa));
    REQUIRE_FALSE(macis::bitset_less(aa, bb));
    REQUIRE_FALSE(macis::bitset_less(aa, b));
    REQUIRE_FALSE(macis::bitset_less(aa, a));
  }
}

TEST_CASE("Bitset Operations") {
  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("INT128") {
    using macis::uint128_t;
    using ref_type = macis::uint128_t;

    SECTION("TYPE") { REQUIRE(std::is_same_v<uint128_t, ref_type>); }

    SECTION("From bitset<128>") { uint128_conversion_test<128>(); }
    SECTION("From bitset<64>") { uint128_conversion_test<64>(); }
    SECTION("From bitset<32>") { uint128_conversion_test<32>(); }
  }

  SECTION("Full Mask") {
    SECTION("32") { mask_test<32>(); }
    SECTION("64") { mask_test<64>(); }
    SECTION("128") { mask_test<128>(); }
    SECTION("256") { mask_test<256>(); }
  }

  SECTION("FFS") {
    SECTION("32") { ffs_test<32>(); }
    SECTION("64") { ffs_test<64>(); }
    SECTION("128") { ffs_test<128>(); }
    SECTION("256") { ffs_test<256>(); }
  }

  SECTION("FLS") {
    SECTION("32") { fls_test<32>(); }
    SECTION("64") { fls_test<64>(); }
    SECTION("128") { fls_test<128>(); }
    SECTION("256") { fls_test<256>(); }
  }

  SECTION("Indices") {
    SECTION("32") { indices_test<32>(); }
    SECTION("64") { indices_test<64>(); }
    SECTION("128") { indices_test<128>(); }
    SECTION("256") { indices_test<256>(); }
  }

  SECTION("LO WORDS") {
    SECTION("64") { lo_word_test<64>(); }
    SECTION("128") { lo_word_test<128>(); }
    SECTION("256") { lo_word_test<256>(); }
    SECTION("512") { lo_word_test<512>(); }
    SECTION("1024") { lo_word_test<1024>(); }
  }

  SECTION("HI WORDS") {
    SECTION("64") { hi_word_test<64>(); }
    SECTION("128") { hi_word_test<128>(); }
    SECTION("256") { hi_word_test<256>(); }
    SECTION("512") { hi_word_test<512>(); }
    SECTION("1024") { hi_word_test<1024>(); }
  }

  SECTION("Compare") {
    SECTION("32") { bitset_less_test<32>(); }
    SECTION("64") { bitset_less_test<64>(); }
    SECTION("128") { bitset_less_test<128>(); }
    SECTION("256") { bitset_less_test<256>(); }
  }
}
