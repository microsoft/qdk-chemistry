// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Immutable set of symmetry-blocked, sorted-unique integer indices.
 *
 * A @ref SymmetryBlockedIndexSet describes, for each admissible
 * @ref SymmetryLabel of a single @ref Symmetries vocabulary, a sorted set of
 * unique indices drawn from @c [0, extent) for that label. It is used to carve
 * out symmetry-respecting subspaces (for example core / active / virtual
 * orbital partitions) of a @ref SymmetryBlockedTensor.
 */
class SymmetryBlockedIndexSet : public DataClass {
 public:
  /**
   * @brief Construct from a symmetry vocabulary, per-label extents, and
   * per-label index lists.
   *
   * @throws BlockLabelInvalidError       if a label is not admissible under
   *         @p symmetries or lacks a declared extent.
   * @throws IndexSetOutOfRangeError      if an index is >= its label's extent.
   * @throws IndexSetNotSortedUniqueError if an index list is not strictly
   *         increasing.
   */
  SymmetryBlockedIndexSet(
      std::shared_ptr<const Symmetries> symmetries,
      std::unordered_map<SymmetryLabel, std::size_t> extents,
      std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices);

  /** @brief The symmetry vocabulary this index set is blocked under. */
  std::shared_ptr<const Symmetries> symmetries() const { return _symmetries; }

  /** @brief Per-label extents. */
  const std::unordered_map<SymmetryLabel, std::size_t>& extents() const {
    return _extents;
  }

  /**
   * @brief View of the (sorted, unique) indices stored for @p label.
   * @throws BlockLabelInvalidError if no indices are stored for @p label.
   */
  std::span<const std::uint32_t> indices(const SymmetryLabel& label) const;

  /** @brief True iff indices are stored for @p label. */
  bool has(const SymmetryLabel& label) const {
    return _indices.find(label) != _indices.end();
  }

  /** @brief The labels for which indices are stored. */
  std::vector<SymmetryLabel> labels() const;

  // ---- DataClass interface ------------------------------------------------

  std::string get_data_type_name() const override {
    return "symmetry_blocked_index_set";
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
  std::shared_ptr<const Symmetries> _symmetries;
  std::unordered_map<SymmetryLabel, std::size_t> _extents;
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> _indices;

  void _validate() const;
};

}  // namespace qdk::chemistry::data
