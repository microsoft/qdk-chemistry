// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @class HashContext
 * @brief Streaming context for deterministic content hashes.
 *
 * Provides an incremental interface for feeding data into a content digest.
 * Producers must add fields in a deterministic order; this is a sequential
 * streaming hash interface, not a tree hash suitable for parallel producers.
 * Supports Eigen matrices, vectors, sparse matrices, and primitive types.
 *
 * Content hashes are intended for cache keys and checkpoint/restart workflows
 * among compatible builds. They are not a data preservation format, and the
 * exact digest values are not guaranteed to remain stable across releases.
 *
 * Usage:
 * @code
 *   HashContext ctx;
 *   ctx.update(some_matrix);
 *   ctx.update(some_double);
 *   ctx.update("some_string");
 *   std::string hash = ctx.hexdigest();
 * @endcode
 */
class HashContext {
 public:
  HashContext();

  // -- Primitive types --

  /// Feed raw bytes into the hash
  void update(const void* data, size_t len);

  /// Hash a double value (8 bytes, little-endian)
  void update(double val);

  /// Hash an int64_t value (8 bytes, little-endian)
  void update(int64_t val);

  /// Hash a uint64_t value (8 bytes, little-endian)
  void update(uint64_t val);

  /// Hash a uint8_t sentinel byte
  void update(uint8_t val);

  /// Hash a boolean value (1 byte: 0x00 or 0x01)
  void update(bool val);

  /// Hash a string (length-prefixed to avoid collisions)
  void update(const std::string& s);

  // -- Eigen types --

  /// Hash a dense matrix (raw contiguous bytes)
  void update(const Eigen::MatrixXd& m);

  /// Hash a dense vector (raw contiguous bytes)
  void update(const Eigen::VectorXd& v);

  /// Hash a dense integer vector
  void update(const Eigen::VectorXi& v);

  /// Hash a complex dense matrix
  void update(const Eigen::MatrixXcd& m);

  /// Hash a complex dense vector
  void update(const Eigen::VectorXcd& v);

  /// Hash a sparse matrix (sorted triplets)
  void update(const Eigen::SparseMatrix<double>& m);

  // -- Variant types (real or complex) --
  using VectorVariant = std::variant<Eigen::VectorXd, Eigen::VectorXcd>;
  using MatrixVariant = std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>;

  /// Hash a vector variant (tags real=0, complex=1 then raw data)
  void update(const VectorVariant& v);

  /// Hash a matrix variant
  void update(const MatrixVariant& m);

  // -- Optional helpers --

  /// Hash an optional value: 0x00 if nullopt, 0x01 + data if present
  template <typename T>
  void update_optional(const std::optional<T>& opt) {
    if (opt.has_value()) {
      update(uint8_t(1));
      update(opt.value());
    } else {
      update(uint8_t(0));
    }
  }

  /// Hash an optional shared_ptr: 0x00 if null, 0x01 + data if present
  template <typename T>
  void update_optional(const std::shared_ptr<T>& ptr) {
    if (ptr) {
      update(uint8_t(1));
      update(*ptr);
    } else {
      update(uint8_t(0));
    }
  }

  /// Hash a vector of size_t values
  void update(const std::vector<size_t>& v);

  // -- Finalization --

  /// Finalize and return truncated hex digest
  /// @param truncate_chars Number of hex characters to return (default 16)
  std::string hexdigest(size_t truncate_chars = 16) const;

  /// Finalize and return the leading digest bytes as a size_t hash code.
  /// Intended for std::hash-compatible local hash tables.
  std::size_t hash_code() const;

 private:
  // SHA-256 state
  std::array<uint32_t, 8> _state;
  std::array<uint8_t, 64> _buffer;
  size_t _buffer_len;
  uint64_t _total_len;

  void _process_block(const uint8_t block[64]);
};

}  // namespace qdk::chemistry::utils
