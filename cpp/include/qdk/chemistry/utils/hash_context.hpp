// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @class HashContext
 * @brief Streaming context for deterministic content hashes.
 *
 * Provides an incremental byte-oriented interface for feeding data into a
 * content digest. Producers must add fields in a deterministic order; this is
 * a sequential streaming hash interface, not a tree hash suitable for parallel
 * producers. Use hash_value() to hash typed values.
 *
 * Content hashes are intended for cache keys and checkpoint/restart workflows
 * among compatible builds. They are not a data preservation format, and the
 * exact digest values are not guaranteed to remain stable across releases.
 *
 * HashContext favors deterministic byte encodings over lookup speed. For
 * performance-sensitive in-memory hash tables, prefer hash_combine() from
 * qdk/chemistry/utils/hash.hpp unless the table key must intentionally use the
 * deterministic content digest.
 *
 * Usage:
 * @code
 *   HashContext ctx;
 *   hash_value(ctx, some_matrix);
 *   hash_value(ctx, some_double);
 *   hash_value(ctx, "some_string");
 *   std::string hash = ctx.hexdigest();
 * @endcode
 */
class HashContext {
 public:
  HashContext();

  /// Feed raw bytes into the hash
  void update(const void* data, size_t len);

  // -- Common variant types --
  using VectorVariant = std::variant<Eigen::VectorXd, Eigen::VectorXcd>;
  using MatrixVariant = std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>;

  // -- Finalization --

  /// Finalize and return truncated hex digest
  /// @param truncate_chars Number of hex characters to return (default 16)
  std::string hexdigest(size_t truncate_chars = 16) const;

  /// Finalize and return the leading digest bytes as a size_t value.
  /// Prefer hash_combine() for hot std::hash-compatible table hashers.
  std::size_t hash_code() const;

 private:
  // SHA-256 state
  std::array<uint32_t, 8> _state;
  std::array<uint8_t, 64> _buffer;
  size_t _buffer_len;
  uint64_t _total_len;

  void _process_block(const uint8_t block[64]);
};

/**
 * @brief Hash a double value as deterministic little-endian bytes.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, double value);

/**
 * @brief Hash a float value through the canonical double encoding.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, float value);

/**
 * @brief Hash a long double value as a canonical hexadecimal string.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, long double value);

/**
 * @brief Hash a signed 64-bit integer as deterministic little-endian bytes.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, int64_t value);

/**
 * @brief Hash an unsigned 64-bit integer as deterministic little-endian bytes.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, uint64_t value);

/**
 * @brief Hash a single sentinel byte.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, uint8_t value);

/**
 * @brief Stable byte tags used to delimit generic hash_value encodings.
 *
 * These are explicit hash-format markers, not serialized C++ type identity.
 * RTTI names are implementation-defined and may vary by compiler, standard
 * library, and build flags; fixed tags keep the digest format deterministic.
 */
enum class HashValueTag : uint8_t {
  RealAlternative = 0,
  ComplexAlternative = 1,
  SharedPtr = 2,
  Optional = 3,
  Vector = 4,
  Variant = 5,
};

/**
 * @brief Hash a generic hash-format tag.
 *
 * @param ctx Hash context to update
 * @param tag Tag to hash
 */
inline void hash_value(HashContext& ctx, HashValueTag tag) {
  hash_value(ctx, static_cast<uint8_t>(tag));
}

/**
 * @brief Hash a boolean value as one byte.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, bool value);

/**
 * @brief Hash a string value with a length prefix.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const std::string& value);

/**
 * @brief Hash a string value with a length prefix.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, std::string_view value);

/**
 * @brief Hash a null-terminated string value.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const char* value);

/**
 * @brief Hash a dense matrix.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const Eigen::MatrixXd& value);

/**
 * @brief Hash a dense vector.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const Eigen::VectorXd& value);

/**
 * @brief Hash a dense integer vector.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const Eigen::VectorXi& value);

/**
 * @brief Hash a complex dense matrix.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const Eigen::MatrixXcd& value);

/**
 * @brief Hash a complex dense vector.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const Eigen::VectorXcd& value);

/**
 * @brief Hash a sparse matrix.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const Eigen::SparseMatrix<double>& value);

/**
 * @brief Hash a vector variant.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const HashContext::VectorVariant& value);

/**
 * @brief Hash a matrix variant.
 *
 * @param ctx Hash context to update
 * @param value Value to hash
 */
void hash_value(HashContext& ctx, const HashContext::MatrixVariant& value);

// Fallback overloads normalize C++ scalar aliases to the fixed encodings
// above. User-defined types should provide their own hash_value() overload,
// preferably in the same namespace as the type so ADL can find it.

/**
 * @brief Concept for signed integral types that need hash normalization.
 * @tparam T The type to check
 *
 * Matches signed integral types other than bool, uint8_t, and the canonical
 * int64_t overload. Matching values are promoted to int64_t before hashing.
 */
template <typename T>
concept HashSignedIntegral =
    std::integral<T> && !std::same_as<T, bool> && !std::same_as<T, uint8_t> &&
    std::signed_integral<T> && !std::same_as<T, int64_t>;

/**
 * @brief Concept for unsigned integral types that need hash normalization.
 * @tparam T The type to check
 *
 * Matches unsigned integral types other than bool, uint8_t, and the canonical
 * uint64_t overload. Matching values are promoted to uint64_t before hashing.
 */
template <typename T>
concept HashUnsignedIntegral =
    std::integral<T> && !std::same_as<T, bool> && !std::same_as<T, uint8_t> &&
    std::unsigned_integral<T> && !std::same_as<T, uint64_t>;

/**
 * @brief Concept for scalar types supported by complex hashing.
 * @tparam T The scalar type to check
 *
 * Matches the floating-point scalar types used by std::complex overloads.
 */
template <typename T>
concept HashComplexScalar = std::same_as<T, float> || std::same_as<T, double> ||
                            std::same_as<T, long double>;

/**
 * @brief Concept for enum types that hash through their underlying type.
 * @tparam T The type to check
 *
 * Matching enum values are cast to their declared underlying type before
 * hashing so callers do not need one-off casts at each use site.
 */
template <typename T>
concept HashEnum = std::is_enum_v<T>;

template <HashSignedIntegral T>
void hash_value(HashContext& ctx, T value) {
  hash_value(ctx, static_cast<int64_t>(value));
}

template <HashUnsignedIntegral T>
void hash_value(HashContext& ctx, T value) {
  hash_value(ctx, static_cast<uint64_t>(value));
}

template <HashEnum T>
void hash_value(HashContext& ctx, T value) {
  using UnderlyingType = std::underlying_type_t<T>;
  hash_value(ctx, static_cast<UnderlyingType>(value));
}

template <HashComplexScalar T>
void hash_value(HashContext& ctx, const std::complex<T>& value) {
  hash_value(ctx, value.real());
  hash_value(ctx, value.imag());
}

template <typename T>
void hash_value(HashContext& ctx, const std::shared_ptr<T>& value) {
  hash_value(ctx, HashValueTag::SharedPtr);
  if (value) {
    hash_value(ctx, true);
    hash_value(ctx, *value);
  } else {
    hash_value(ctx, false);
  }
}

template <typename T>
void hash_value(HashContext& ctx, const std::optional<T>& value) {
  hash_value(ctx, HashValueTag::Optional);
  if (value) {
    hash_value(ctx, true);
    hash_value(ctx, *value);
  } else {
    hash_value(ctx, false);
  }
}

template <typename T>
void hash_value(HashContext& ctx, const std::vector<T>& values) {
  hash_value(ctx, HashValueTag::Vector);
  hash_value(ctx, static_cast<uint64_t>(values.size()));
  for (const auto& value : values) {
    hash_value(ctx, value);
  }
}

template <typename... Ts>
void hash_value(HashContext& ctx, const std::variant<Ts...>& value) {
  hash_value(ctx, HashValueTag::Variant);
  hash_value(ctx, static_cast<uint64_t>(value.index()));
  std::visit([&ctx](const auto& selected) { hash_value(ctx, selected); },
             value);
}

}  // namespace qdk::chemistry::utils
