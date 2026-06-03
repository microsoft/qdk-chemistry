// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked.hpp>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Sorted-unique integer index subset, blocked by symmetry.
 *
 * A @ref SymmetryBlockedIndexSet describes, for each admissible
 * @ref SymmetryLabel of a single @ref Symmetries, a sorted set of
 * unique indices drawn from @c [0, extent) for that label. The
 * @em extent is the universe size for that label (e.g. the total number
 * of α MOs) and is stored separately from the @em indices subset, so
 * the universe boundary survives serialization even when the subset is
 * sparse or empty.
 *
 * Typical use: carving out symmetry-respecting subspaces
 * (core / active / virtual orbital partitions) of a
 * @ref SymmetryBlockedTensor.
 *
 * Example — active α = @c {2,3,4} drawn from 10 α MOs:
 * @code
 * std::unordered_map<SymmetryLabel, std::size_t> extents;
 * extents[SymmetryLabel({axes::alpha()})] = 10;           // universe size
 *
 * std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
 * indices[SymmetryLabel({axes::alpha()})] = {2u, 3u, 4u}; // chosen subset
 *
 * SymmetryBlockedIndexSet active(symmetries, extents, indices);
 * @endcode
 */
class SymmetryBlockedIndexSet
    : public SymmetryBlocked<1, std::vector<std::uint32_t>> {
  using Base = SymmetryBlocked<1, std::vector<std::uint32_t>>;

 public:
  /**
   * @brief Construct from symmetry definitions, per-label extents, and
   * per-label index lists.
   *
   * @param symmetries The symmetry definitions this index set is blocked
   *        under. Determines the admissible @ref SymmetryLabel keys for
   *        @p extents and @p indices.
   * @param extents Per-label universe size. @c extents[label] is the
   *        upper bound (exclusive) of the index range from which the
   *        @p indices entries for that label are drawn. The extent is
   *        @b not derived from @c indices[label].size() because the
   *        subset may be strictly smaller than (or sparser than, or
   *        empty within) its universe, and the universe boundary must
   *        survive serialization independently of the chosen subset.
   * @param indices Per-label chosen subset, sorted strictly increasing.
   *        Every index must satisfy @c 0 <= idx < extents[label].
   *
   * @throws std::invalid_argument if a label is not admissible under
   *         @p symmetries, lacks a declared extent, or an index list is not
   *         strictly increasing.
   * @throws std::out_of_range if an index is >= its label's extent.
   */
  SymmetryBlockedIndexSet(
      std::shared_ptr<const Symmetries> symmetries,
      std::unordered_map<SymmetryLabel, std::size_t> extents,
      std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices);

  /** @brief The symmetry definitions this index set is blocked under. */
  std::shared_ptr<const Symmetries> symmetries() const {
    return Base::symmetries()[0];
  }

  /** @brief Per-label extents. */
  const std::unordered_map<SymmetryLabel, std::size_t>& extents() const {
    return Base::SymmetryBlocked::extents()[0];
  }

  /**
   * @brief View of the (sorted, unique) indices stored for @p label.
   * @throws std::invalid_argument if no indices are stored for @p label.
   */
  std::span<const std::uint32_t> indices(const SymmetryLabel& label) const;

  /** @brief True iff indices are stored for @p label. */
  bool has(const SymmetryLabel& label) const {
    return Base::has_block(Labels{label});
  }

  /** @brief The labels for which indices are stored. */
  std::vector<SymmetryLabel> labels() const;

  // ---- DataClass interface ------------------------------------------------

  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(SymmetryBlockedIndexSet);
  }
  std::string get_summary() const override;
  void to_file(const std::string& filename,
               const std::string& type) const override;
  nlohmann::json to_json() const override;
  void to_json_file(const std::string& filename) const override;
  void to_hdf5(H5::Group& group) const override;
  void to_hdf5_file(const std::string& filename) const override;

  static std::shared_ptr<SymmetryBlockedIndexSet> from_json(
      const nlohmann::json& j);
  static std::shared_ptr<SymmetryBlockedIndexSet> from_json_file(
      const std::string& filename);
  static std::shared_ptr<SymmetryBlockedIndexSet> from_hdf5(H5::Group& group);
  static std::shared_ptr<SymmetryBlockedIndexSet> from_hdf5_file(
      const std::string& filename);
  static std::shared_ptr<SymmetryBlockedIndexSet> from_file(
      const std::string& filename, const std::string& type);

 private:
  static Base::BlockMap _build_block_map(
      std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>& indices);
  void _validate_indices() const;

  /// On-disk serialization format version. Bump on any change to the JSON
  /// or HDF5 shape produced by @ref to_json / @ref to_hdf5.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data
