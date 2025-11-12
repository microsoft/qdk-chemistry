/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <bit>
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <macis/types.hpp>

namespace macis {

/**
 *  @brief Typesafe CLZ
 *
 *  Unsigned int overload
 *
 *  @param[in] i integral input for CLZ
 *  @returns CLZ for `i`
 */
// #TODO: Guard against zero inputs; __builtin_clz is undefined for 0 and we
// should throw or handle explicitly.
inline auto clz(unsigned int i) { return __builtin_clz(i); }

/**
 *  @brief Typesafe CLZ
 *
 *  Unsigned long int overload
 *
 *  @param[in] i integral input for CLZ
 *  @returns CLZ for `i`
 */
// #TODO: Same zero-input concern as above
inline auto clz(unsigned long int i) { return __builtin_clzl(i); }

/**
 *  @brief Typesafe CLZ
 *
 *  Unsigned long long int overload
 *
 *  @param[in] i integral input for CLZ
 *  @returns CLZ for `i`
 */
// #TODO: Same zero-input concern as above
inline auto clz(unsigned long long int i) { return __builtin_clzll(i); }

/**
 *  @brief Typesafe FLS
 *
 *  Returns the index (0-based) of the last set bit of input integer
 *
 *  @tparam Integral Integral type, must be `Integral` and not `Signed`
 *
 *  @param[in] i integral input for FLS
 *  @returns FLS for `i`
 */
template <typename Integral>
std::enable_if_t<std::is_integral_v<Integral> and !std::is_signed_v<Integral>,
                 unsigned>
fls(Integral i) {
  return CHAR_BIT * sizeof(Integral) - clz(i) - 1;
}

/// Fast conversion of bitset to unsigned long long
template <size_t N>
unsigned long long fast_to_ullong(const std::bitset<N>& bits) {
  // Low words
  if constexpr (N == 64 or N == 128)
    return *reinterpret_cast<const uint64_t*>(&bits);
  if constexpr (N == 32) return *reinterpret_cast<const uint32_t*>(&bits);
  return bits.to_ullong();
}

/// Fast conversion of bitset to unsigned long
template <size_t N>
unsigned long fast_to_ulong(const std::bitset<N>& bits) {
  // Low words
  if constexpr (N == 32 or N == 64 or N == 128)
    // #TODO: I am a bit confused here. On a 64-bit system, unsigned long is 64
    // bits, so are we throwing away data? Is this intended?
    return *reinterpret_cast<const uint32_t*>(&bits);
  return bits.to_ulong();
}

/// Conversion of bitset to uint128
template <size_t N>
uint128_t to_uint128(std::bitset<N> bits) {
  static_assert(N <= 128, "N > 128");
  if constexpr (N == 128) {
    alignas(alignof(uint128_t)) std::bitset<N> cpy = bits;
    auto _x = reinterpret_cast<uint128_t*>(&cpy);
    return *_x;
  } else {
    return fast_to_ullong(bits);
  }
}

/**
 *  @brief Full bitmask generator (compile time)
 *
 *  Generates an all-true bitmask of a specified width
 *
 *  e.g. full_mask<2,4> === 0b0011
 *
 *  @tparam N Number of true bits
 *  @tparam M Width of resulting mask
 *
 *  @returns `N`-true bitmask of width `M`
 */
template <size_t N, size_t M = N>
std::bitset<M> full_mask() {
  static_assert(M >= N, "M < N");
  std::bitset<M> mask(0ul);
  if constexpr (N == M / 2) {
    if constexpr (N == 64) {
      reinterpret_cast<uint64_t*>(&mask)[0] = UINT64_MAX;
    } else if constexpr (N == 32) {
      reinterpret_cast<uint32_t*>(&mask)[0] = UINT32_MAX;
    } else
      mask = (~mask) >> (M - N);
    return mask;
  } else
    return (~mask) >> (M - N);
}

/**
 *  @brief Full bitmask generator (dynamic)
 *
 *  Generates an all-true bitmask of a specified width
 *
 *  e.g. full_mask<4>(2) === 0b0011
 *
 *  @tparam N Width of resulting mask
 *
 *  @param[in] i Number of true bits
 *  @returns `i`-true bitmask of width `N`
 */
template <size_t N>
std::bitset<N> full_mask(size_t i) {
  assert(i <= N);
  std::bitset<N> mask(0ul);
  return (~mask) >> (N - i);
}

/**
 *  @brief FFS for bitset
 *
 *  Returns the index (0-based) of the first set bit of a bitset
 *
 *  @tparam `N` Width of bitset
 *
 *  @param[in] bits input for FFS
 *  @returns FFS for `bits`
 */
template <size_t N>
uint32_t ffs(std::bitset<N> bits) {
  if constexpr (N <= 32)
    return ffsl(fast_to_ulong(bits));
  else if constexpr (N <= 64)
    return ffsll(fast_to_ullong(bits));
  else if constexpr (N % 64 == 0) {
    auto as_words = reinterpret_cast<uint64_t*>(&bits);
    constexpr int n_words = N / 64;
    int off = 0;
    for (int i = 0; i < n_words; ++i) {
      if (as_words[i]) return ffsll(as_words[i]) + off;
      off += 64;
    }
    return 0;
  } else {
    uint32_t ind = 0;
    for (ind = 0; ind < N; ++ind)
      if (bits[ind]) return (ind + 1);
    return ind;
  }
  abort();
}

/**
 *  @brief FLS for bitset
 *
 *  Returns the index (0-based) of the last set bit of a bitset
 *
 *  @tparam `N` Width of bitset
 *
 *  @param[in] bits input for FLS
 *  @returns FLS for `bits`
 */
template <size_t N>
uint32_t fls(std::bitset<N> bits) {
  if constexpr (N <= 32)
    return fls(fast_to_ulong(bits));
  else if constexpr (N <= 64)
    return fls(fast_to_ullong(bits));
  else if constexpr (N % 64 == 0) {
    auto as_words = reinterpret_cast<uint64_t*>(&bits);
    constexpr int n_words = N / 64;
    int off = N - 64;
    for (int i = n_words - 1; i >= 0; --i) {
      if (as_words[i]) return fls(as_words[i]) + off;
      off -= 64;
    }
    return UINT32_MAX;
  } else {
    uint32_t ind = 0;
    for (ind = N - 1; ind >= 0; ind--)
      if (bits[ind]) return ind;
    return UINT32_MAX;
  }
  abort();
}

/// Convert bitset to a list of indices (inplace)
template <size_t N>
void bits_to_indices(std::bitset<N> bits, std::vector<uint32_t>& indices) {
  indices.clear();
  auto c = bits.count();
  indices.resize(c);
  if (!c) return;
  for (int i = 0; i < c; ++i) {
    const auto ind = ffs(bits) - 1;
    bits.flip(ind);
    indices[i] = ind;
  }
}

/// Convert bitset to a list of indices (out-of-place)
template <size_t N>
std::vector<uint32_t> bits_to_indices(std::bitset<N> bits) {
  std::vector<uint32_t> indices;
  bits_to_indices(bits, indices);
  return indices;
}

#if 0
// #TODO: Dead code, need to be removed?
/// Truncate a bitset to one of smaller width
template <size_t N, size_t M>
inline std::bitset<N> truncate_bitset(std::bitset<M> bits) {
  static_assert(M >= N, "M < N");
  if constexpr(M == N) return bits;

  const auto mask = full_mask<N, M>();
  if constexpr(N <= 32) {
    return (bits & mask).to_ulong();
  } else if constexpr(N <= 64) {
    return (bits & mask).to_ullong();
  } else {
    std::bitset<N> trunc_bits = 0;
    for(size_t i = 0; i < N; ++i)
      if(bits[i]) trunc_bits[i] = 1;
    return trunc_bits;
  }
}
#endif /* 0 */

/// Expand a bitset to one of larger width
template <size_t N, size_t M>
inline std::bitset<N> expand_bitset(std::bitset<M> bits) {
  static_assert(N >= M, "N < M");
  if constexpr (M == N) return bits;

  if constexpr (M <= 32) {
    return bits.to_ulong();
  } else if constexpr (M <= 64) {
    return bits.to_ullong();
  } else {
    std::bitset<N> exp_bits = 0;
    for (size_t i = 0; i < M; ++i)
      if (bits[i]) exp_bits[i] = 1;
    return exp_bits;
  }
}

/// Extract to lo word of a bitset of even width
template <size_t N>
inline std::bitset<N / 2> bitset_lo_word(std::bitset<N> bits) {
  static_assert((N % 64 == 0) and (N == 64 or N / 2 % 64 == 0),
                "Not Supported");
  if constexpr (N == 64) {
    return std::bitset<32>(reinterpret_cast<uint32_t*>(&bits)[0]);
  } else {
    std::bitset<N / 2> lo;
    constexpr int nword = (N / 2) / 64;
    auto lo_as_words = reinterpret_cast<uint64_t*>(&lo);
    auto bi_as_words = reinterpret_cast<uint64_t*>(&bits);
    for (int i = 0; i < nword; ++i) lo_as_words[i] = bi_as_words[i];
    return lo;
  }
}

template <size_t N>
inline void set_bitset_lo_word(std::bitset<N>& bits, std::bitset<N / 2> word) {
  static_assert((N % 64 == 0) and (N == 64 or N / 2 % 64 == 0),
                "Not Supported");

  if constexpr (N == 64) {
    auto bi_as_words = reinterpret_cast<uint32_t*>(&bits);
    auto wo_as_words = reinterpret_cast<uint32_t*>(&word);
    bi_as_words[0] = wo_as_words[0];
  } else {
    constexpr int nword = (N / 2) / 64;
    auto bi_as_words = reinterpret_cast<uint64_t*>(&bits);
    auto wo_as_words = reinterpret_cast<uint64_t*>(&word);
    for (int i = 0; i < nword; ++i) bi_as_words[i] = wo_as_words[i];
  }
}

/// Extract to hi word of a bitset of even width
template <size_t N>
inline std::bitset<N / 2> bitset_hi_word(std::bitset<N> bits) {
  static_assert((N % 64 == 0) and (N == 64 or N / 2 % 64 == 0),
                "Not Supported");
  if constexpr (N == 64) {
    return std::bitset<32>(reinterpret_cast<uint32_t*>(&bits)[1]);
  } else {
    std::bitset<N / 2> hi;
    constexpr int nword = (N / 2) / 64;
    auto hi_as_words = reinterpret_cast<uint64_t*>(&hi);
    auto bi_as_words = reinterpret_cast<uint64_t*>(&bits);
    for (int i = 0; i < nword; ++i) hi_as_words[i] = bi_as_words[i + nword];
    return hi;
  }
}

template <size_t N>
inline void set_bitset_hi_word(std::bitset<N>& bits, std::bitset<N / 2> word) {
  static_assert((N % 64 == 0) and (N == 64 or N / 2 % 64 == 0),
                "Not Supported");

  if constexpr (N == 64) {
    auto bi_as_words = reinterpret_cast<uint32_t*>(&bits);
    auto wo_as_words = reinterpret_cast<uint32_t*>(&word);
    bi_as_words[1] = wo_as_words[0];
  } else {
    constexpr int nword = (N / 2) / 64;
    auto bi_as_words = reinterpret_cast<uint64_t*>(&bits);
    auto wo_as_words = reinterpret_cast<uint64_t*>(&word);
    for (int i = 0; i < nword; ++i) bi_as_words[i + nword] = wo_as_words[i];
  }
}

/// Bitwise less-than operator for bitset
template <size_t N>
bool bitset_less(std::bitset<N> x, std::bitset<N> y) {
  if constexpr (N <= 32)
    return fast_to_ulong(x) < fast_to_ulong(y);
  else if constexpr (N <= 64)
    return fast_to_ullong(x) < fast_to_ullong(y);
  else if constexpr (N == 128) {
    auto _x = to_uint128(x);
    auto _y = to_uint128(y);
    return _x < _y;
  } else if constexpr (N % 64 == 0) {
    auto x_as_words = reinterpret_cast<uint64_t*>(&x);
    auto y_as_words = reinterpret_cast<uint64_t*>(&y);
    constexpr auto nwords = N / 64;
    for (int i = nwords - 1; i >= 0; --i) {
      const auto x_i = x_as_words[i];
      const auto y_i = y_as_words[i];
      if (x_i != y_i) return x_i < y_i;
    }
    return false;
  } else {
    for (int i = N - 1; i >= 0; i--) {
      if (x[i] ^ y[i]) return y[i];
    }
    return false;
  }
  abort();
}

/// Bitwise less-than comparator for bitset
template <size_t N>
struct bitset_less_comparator {
  bool operator()(std::bitset<N> x, std::bitset<N> y) const {
    return bitset_less(x, y);
  }
};

}  // namespace macis
