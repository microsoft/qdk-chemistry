// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/errors.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Hash of an array of @ref SymmetryLabel used to key blocks by their
 * per-slot symmetry labels.
 */
template <std::size_t Rank>
struct LabelsHash {
  std::size_t operator()(
      const std::array<SymmetryLabel, Rank>& labels) const noexcept {
    std::size_t seed = 0;
    for (const auto& label : labels) {
      seed ^= label.hash() + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

namespace detail {

// Reassemble a std::array of labels from a vector of exactly Rank labels.
template <std::size_t Rank>
std::array<SymmetryLabel, Rank> make_labels(
    const std::vector<SymmetryLabel>& values) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::array<SymmetryLabel, Rank>{values[I]...};
  }(std::make_index_sequence<Rank>{});
}

}  // namespace detail

/**
 * @brief Immutable symmetry-addressed container of per-block storage.
 *
 * A @ref SymmetryBlocked stores a sparse map from per-slot
 * @ref SymmetryLabel arrays to opaque block values of type @p Block.
 * Each slot carries its own @ref Symmetries vocabulary and a per-label extent.
 * Blocks are held via @c shared_ptr<const Block> so that orbit-equivalent
 * sectors can alias the same storage.
 *
 * Orbit equivalence is defined on the spin axis: a simultaneous
 * @f$\alpha \leftrightarrow \beta@f$ swap across all slots. When every slot
 * shares the same @ref Symmetries instance and the spin axis is marked
 * @c equivalent (restricted), the constructor auto-aliases each orbit's
 * partner block to the supplied representative. When the slots carry distinct
 * @ref Symmetries (intertwiner storage such as basis coefficients), no
 * auto-aliasing is performed.
 *
 * Instances are immutable: the full block map is supplied at construction and
 * there is no mutation API.
 *
 * Derived classes (e.g. @ref SymmetryBlockedTensor) add block-type-specific
 * validation and serialization.
 *
 * @tparam Rank   Number of label slots (1, 2, or 4 are typical).
 * @tparam Block  The per-block storage type (e.g. @c Eigen::MatrixXd,
 *                @c std::map, @c std::vector).
 */
template <std::size_t Rank, typename Block>
class SymmetryBlocked : public DataClass {
 public:
  using Labels = std::array<SymmetryLabel, Rank>;
  using BlockPtr = std::shared_ptr<const Block>;
  using BlockMap = std::unordered_map<Labels, BlockPtr, LabelsHash<Rank>>;
  using SymmetriesArray = std::array<std::shared_ptr<const Symmetries>, Rank>;
  using ExtentsArray =
      std::array<std::unordered_map<SymmetryLabel, std::size_t>, Rank>;

  /**
   * @brief Construct from per-slot symmetries, per-slot extents, and a block
   * map.
   *
   * Validates symmetries and extents, checks block label admissibility and
   * null pointers, then applies orbit aliasing.  Derived classes should call
   * their own block-shape validation after this constructor returns.
   *
   * @throws BlockLabelInvalidError   if a block or extent label is not
   *         admissible under the matching slot's @ref Symmetries, or if a
   *         block pointer is null.
   * @throws BlockExtentMismatchError if restricted orbit partners have
   *         unequal extents.
   * @throws BlockAliasMismatchError  if both orbit partners are supplied but
   *         do not share storage.
   */
  SymmetryBlocked(SymmetriesArray symmetries, ExtentsArray extents,
                  BlockMap blocks)
      : _symmetries(std::move(symmetries)),
        _extents(std::move(extents)),
        _blocks(std::move(blocks)) {
    _validate_symmetries_and_extents();
    _validate_block_labels();
    _apply_orbit_aliasing();
  }

  /** @brief Per-slot symmetry vocabularies. */
  const SymmetriesArray& symmetries() const { return _symmetries; }

  /** @brief Per-slot per-label extents. */
  const ExtentsArray& extents() const { return _extents; }

  /** @brief True iff a block is stored for @p labels. */
  bool has_block(const Labels& labels) const {
    return _blocks.find(labels) != _blocks.end();
  }

  /**
   * @brief Reference to the block stored for @p labels.
   * @throws BlockLabelInvalidError if no such block exists.
   */
  const Block& block(const Labels& labels) const { return *block_ptr(labels); }

  /**
   * @brief Shared pointer to the block stored for @p labels.
   * @throws BlockLabelInvalidError if no such block exists.
   */
  BlockPtr block_ptr(const Labels& labels) const {
    auto it = _blocks.find(labels);
    if (it == _blocks.end()) {
      throw BlockLabelInvalidError(
          "SymmetryBlocked has no block for the requested labels.");
    }
    return it->second;
  }

  /** @brief Total number of stored blocks (including aliases). */
  std::size_t num_blocks() const { return _blocks.size(); }

  /**
   * @brief Representative labels for each independent (non-aliased) block.
   *
   * Blocks that share storage (orbit partners under a restricted spin axis,
   * or producer-shared blocks) are represented once, by the spin-canonical
   * key among those sharing the pointer.
   */
  std::vector<Labels> canonical_block_labels() const {
    std::vector<Labels> result;
    for (const auto& group : _group_by_pointer()) {
      result.push_back(group.representative);
    }
    return result;
  }

  /** @brief Representative (labels, block-pointer) pairs, one per unique data
   * pointer. */
  std::vector<std::pair<Labels, BlockPtr>> canonical_blocks() const {
    std::vector<std::pair<Labels, BlockPtr>> result;
    for (const auto& group : _group_by_pointer()) {
      result.emplace_back(group.representative, group.ptr);
    }
    return result;
  }

  /**
   * @brief True iff any two distinct labels alias the same storage.
   *
   * Mirrors the historical restricted check of pointer equality between the
   * @f$\alpha@f$ and @f$\beta@f$ coefficient blocks.
   */
  bool is_restricted() const {
    for (const auto& [labels_a, ptr_a] : _blocks) {
      for (const auto& [labels_b, ptr_b] : _blocks) {
        if (!(labels_a == labels_b) && ptr_a.get() == ptr_b.get() &&
            ptr_a != nullptr) {
          return true;
        }
      }
    }
    return false;
  }

 protected:
  SymmetriesArray _symmetries;
  ExtentsArray _extents;
  BlockMap _blocks;

  /** @brief Internal grouping for canonical-block enumeration. */
  struct PointerGroup {
    Labels representative;
    std::vector<Labels> keys;
    BlockPtr ptr;
  };

  static bool _label_admissible(const Symmetries& sym,
                                const SymmetryLabel& label) {
    if (label.values().size() != sym.axes().size()) {
      return false;
    }
    for (const auto& axis : sym.axes()) {
      if (!label.has(axis.name())) {
        return false;
      }
      if (!axis.admits(*label.get(axis.name()))) {
        return false;
      }
    }
    return true;
  }

  void _validate_symmetries_and_extents() const {
    for (std::size_t i = 0; i < Rank; ++i) {
      if (_symmetries[i] == nullptr) {
        throw BlockLabelInvalidError(
            "SymmetryBlocked slot symmetries must not be null.");
      }
      for (const auto& [label, extent] : _extents[i]) {
        (void)extent;
        if (!_label_admissible(*_symmetries[i], label)) {
          throw BlockLabelInvalidError(
              "SymmetryBlocked extent label is not admissible under the "
              "slot's symmetries.");
        }
      }
    }
  }

  void _validate_block_labels() const {
    for (const auto& [labels, ptr] : _blocks) {
      if (ptr == nullptr) {
        throw BlockLabelInvalidError(
            "SymmetryBlocked block pointers must not be null.");
      }
      for (std::size_t i = 0; i < Rank; ++i) {
        if (!_label_admissible(*_symmetries[i], labels[i])) {
          throw BlockLabelInvalidError(
              "SymmetryBlocked block label is not admissible under the "
              "slot's symmetries.");
        }
      }
    }
  }

  std::size_t _extent_for(std::size_t slot, const SymmetryLabel& label) const {
    auto it = _extents[slot].find(label);
    if (it == _extents[slot].end()) {
      throw BlockLabelInvalidError(
          "SymmetryBlocked block label has no declared extent.");
    }
    return it->second;
  }

  bool _spin_aliasing_active() const {
    for (std::size_t i = 1; i < Rank; ++i) {
      if (_symmetries[i] != _symmetries[0]) {
        return false;
      }
    }
    if (_symmetries[0] == nullptr ||
        !_symmetries[0]->has_axis(AxisName::Spin)) {
      return false;
    }
    return _symmetries[0]->axis(AxisName::Spin).equivalent();
  }

  static SymmetryLabel _swap_spin(const SymmetryLabel& label) {
    std::vector<std::shared_ptr<const SymmetryAxisValue>> values;
    for (const auto& [axis, value] : label.values()) {
      if (axis == AxisName::Spin) {
        const auto* spin = dynamic_cast<const SpinValue*>(value.get());
        if (spin != nullptr) {
          values.push_back(axes::spin_value(-spin->value()));
          continue;
        }
      }
      values.push_back(value);
    }
    return SymmetryLabel(std::move(values));
  }

  static Labels _swap_spin_labels(const Labels& labels) {
    Labels swapped = labels;
    for (std::size_t i = 0; i < Rank; ++i) {
      swapped[i] = _swap_spin(labels[i]);
    }
    return swapped;
  }

  static std::array<int, Rank> _spin_sequence(const Labels& labels) {
    std::array<int, Rank> seq{};
    for (std::size_t i = 0; i < Rank; ++i) {
      if (labels[i].has(AxisName::Spin)) {
        const auto* spin =
            dynamic_cast<const SpinValue*>(labels[i].get(AxisName::Spin).get());
        seq[i] = spin != nullptr ? spin->value() : 0;
      }
    }
    return seq;
  }

  static bool _is_spin_canonical(const Labels& labels) {
    auto seq = _spin_sequence(labels);
    for (std::size_t i = 0; i < Rank; ++i) {
      if (seq[i] != 0) {
        return seq[i] > 0;
      }
    }
    return true;
  }

  void _apply_orbit_aliasing() {
    if (!_spin_aliasing_active()) {
      return;
    }
    for (std::size_t i = 0; i < Rank; ++i) {
      for (const auto& [label, extent] : _extents[i]) {
        SymmetryLabel partner = _swap_spin(label);
        auto it = _extents[i].find(partner);
        if (it != _extents[i].end() && it->second != extent) {
          throw BlockExtentMismatchError(
              "Restricted spin orbit partners must share the same extent.");
        }
      }
    }

    std::vector<std::pair<Labels, BlockPtr>> to_insert;
    for (const auto& [labels, ptr] : _blocks) {
      Labels partner = _swap_spin_labels(labels);
      if (partner == labels) {
        continue;
      }
      auto partner_it = _blocks.find(partner);
      if (partner_it != _blocks.end()) {
        if (partner_it->second.get() != ptr.get()) {
          throw BlockAliasMismatchError(
              "Restricted spin orbit partners must share block storage.");
        }
      } else {
        to_insert.emplace_back(partner, ptr);
      }
    }
    for (auto& [labels, ptr] : to_insert) {
      _blocks.emplace(std::move(labels), std::move(ptr));
    }
  }

  std::vector<PointerGroup> _group_by_pointer() const {
    std::vector<PointerGroup> groups;
    std::unordered_map<const Block*, std::size_t> index;
    for (const auto& [labels, ptr] : _blocks) {
      const auto* raw = ptr.get();
      auto it = index.find(raw);
      if (it == index.end()) {
        index.emplace(raw, groups.size());
        groups.push_back(PointerGroup{labels, {labels}, ptr});
      } else {
        auto& group = groups[it->second];
        group.keys.push_back(labels);
        if (_is_spin_canonical(labels) &&
            !_is_spin_canonical(group.representative)) {
          group.representative = labels;
        }
      }
    }
    return groups;
  }

  // ---- JSON helpers for symmetries/extents (shared by all derived types) ---

  nlohmann::json _symmetries_to_json() const {
    nlohmann::json symmetries = nlohmann::json::array();
    for (const auto& sym : _symmetries) {
      symmetries.push_back(sym ? sym->to_json() : nlohmann::json());
    }
    return symmetries;
  }

  nlohmann::json _extents_to_json() const {
    nlohmann::json extents = nlohmann::json::array();
    for (const auto& slot : _extents) {
      nlohmann::json slot_json = nlohmann::json::array();
      for (const auto& [label, extent] : slot) {
        slot_json.push_back(
            nlohmann::json{{"label", label.to_json()}, {"extent", extent}});
      }
      extents.push_back(std::move(slot_json));
    }
    return extents;
  }

  static SymmetriesArray _symmetries_from_json(const nlohmann::json& j) {
    SymmetriesArray symmetries;
    const auto& sym_json = j.at("symmetries");
    for (std::size_t i = 0; i < Rank; ++i) {
      symmetries[i] = std::make_shared<const Symmetries>(
          Symmetries::from_json(sym_json[i]));
    }
    return symmetries;
  }

  static ExtentsArray _extents_from_json(const nlohmann::json& j) {
    ExtentsArray extents;
    const auto& ext_json = j.at("extents");
    for (std::size_t i = 0; i < Rank; ++i) {
      for (const auto& entry : ext_json[i]) {
        extents[i].emplace(SymmetryLabel::from_json(entry.at("label")),
                           entry.at("extent").template get<std::size_t>());
      }
    }
    return extents;
  }

  nlohmann::json _block_keys_to_json() const {
    nlohmann::json blocks = nlohmann::json::array();
    for (const auto& group : _group_by_pointer()) {
      nlohmann::json keys = nlohmann::json::array();
      for (const auto& key : group.keys) {
        nlohmann::json key_json = nlohmann::json::array();
        for (const auto& label : key) {
          key_json.push_back(label.to_json());
        }
        keys.push_back(std::move(key_json));
      }
      blocks.push_back(nlohmann::json{{"keys", std::move(keys)}});
    }
    return blocks;
  }

  static std::vector<std::vector<Labels>> _block_keys_from_json(
      const nlohmann::json& j) {
    std::vector<std::vector<Labels>> result;
    for (const auto& entry : j.at("blocks")) {
      std::vector<Labels> key_group;
      for (const auto& key_json : entry.at("keys")) {
        std::vector<SymmetryLabel> labels;
        for (const auto& label_json : key_json) {
          labels.push_back(SymmetryLabel::from_json(label_json));
        }
        key_group.push_back(detail::make_labels<Rank>(labels));
      }
      result.push_back(std::move(key_group));
    }
    return result;
  }
};

}  // namespace qdk::chemistry::data
