// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/utils/hash.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Hash of an array of @ref SymmetryLabel used to key blocks by their
 * per-slot symmetry labels.
 *
 * @tparam Rank Number of label slots.
 */
template <std::size_t Rank>
struct LabelsHash {
  /**
   * @brief Combine the per-slot @ref SymmetryLabel hashes into a single
   * hash suitable for use as an @c unordered_map key.
   *
   * @param labels Per-slot symmetry labels keying one block.
   * @return Hash value derived from the concatenation of per-slot label
   *         hashes via @ref qdk::chemistry::utils::hash_combine.
   */
  std::size_t operator()(
      const std::array<SymmetryLabel, Rank>& labels) const noexcept {
    std::size_t seed = 0;
    for (const auto& label : labels) {
      seed = utils::hash_combine(seed, label.hash());
    }
    return seed;
  }
};

namespace detail {

// Reassemble a std::array of labels from a vector of exactly Rank labels.
template <std::size_t Rank>
std::array<SymmetryLabel, Rank> make_labels(
    const std::vector<SymmetryLabel>& values) {
  if (values.size() != Rank) {
    throw std::invalid_argument("make_labels: expected " +
                                std::to_string(Rank) + " labels, got " +
                                std::to_string(values.size()));
  }
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::array<SymmetryLabel, Rank>{values[I]...};
  }(std::make_index_sequence<Rank>{});
}

}  // namespace detail

/**
 * @brief Symmetry-addressed container of per-block storage.
 *
 * A @ref SymmetryBlocked stores a sparse map from per-slot
 * @ref SymmetryLabel arrays to opaque block values of type @p Block.
 * Each slot carries its own @ref SymmetryProduct and a per-label extent.
 * Blocks are held via @c shared_ptr<const Block> so that symmetry-equivalent
 * sectors can alias the same storage.
 *
 * Symmetry aliasing is defined on the spin axis: a simultaneous
 * @f$\alpha \leftrightarrow \beta@f$ swap across all slots. When every slot
 * shares the same @ref SymmetryProduct instance and the spin axis is marked
 * @c equivalent, the constructor auto-aliases each partner block to the
 * supplied representative. Aliasing can also be achieved by the producer
 * supplying the same @c shared_ptr for both spin partners (e.g. restricted
 * basis coefficients share the same MO matrix for both spins).
 *
 * The full block map is supplied at construction.
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
  /**
   * @brief Per-slot block label tuple: one @ref SymmetryLabel per index slot.
   *
   * Used as the key type of @ref BlockMap so that blocks are addressed by the
   * full tuple of per-slot symmetry labels.
   */
  using Labels = std::array<SymmetryLabel, Rank>;
  /**
   * @brief Shared pointer to immutable per-block storage.
   *
   * Held as @c shared_ptr<const Block> so that symmetry-equivalent sectors
   * can alias the same underlying storage.
   */
  using BlockPtr = std::shared_ptr<const Block>;
  /**
   * @brief Sparse map from per-slot label tuples to block storage.
   *
   * Aliased sectors (e.g. restricted-spin partners) map to the same
   * @ref BlockPtr. Keys are hashed via @ref LabelsHash.
   */
  using BlockMap = std::unordered_map<Labels, BlockPtr, LabelsHash<Rank>>;
  /**
   * @brief Per-slot symmetry definitions.
   *
   * One @ref SymmetryProduct per index slot; supplied at construction and
   * used to validate block labels and apply orbit aliasing.
   */
  using SymmetriesArray =
      std::array<std::shared_ptr<const SymmetryProduct>, Rank>;
  /**
   * @brief Per-slot per-label extents.
   *
   * For each index slot, maps every admissible @ref SymmetryLabel to the
   * universe size (number of basis vectors) carried under that label.
   */
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
   * @param symmetries Per-slot @ref SymmetryProduct definitions. None of the
   *                   pointers may be null.
   * @param extents Per-slot per-label universe sizes; every declared label
   *                must be admissible under its slot's symmetries.
   * @param blocks Block storage keyed by per-slot @ref SymmetryLabel
   *               tuples; pointers must be non-null and labels admissible.
   *
   * @throws std::invalid_argument if a block or extent label is not
   *         admissible under the matching slot's @ref SymmetryProduct, if a
   *         block pointer is null, if restricted orbit partners have
   *         unequal extents, or if both orbit partners are supplied but do
   *         not share storage.
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

  /**
   * @brief Per-slot symmetry definitions.
   * @return Reference to the @ref SymmetriesArray supplied at construction.
   */
  const SymmetriesArray& symmetries() const { return _symmetries; }

  /**
   * @brief Per-slot per-label extents.
   * @return Reference to the @ref ExtentsArray supplied at construction.
   */
  const ExtentsArray& extents() const { return _extents; }

  /**
   * @brief True iff a block is stored for @p labels.
   *
   * If all labels in the key are trivial (empty), returns @c true when
   * exactly one unique block exists.
   *
   * @param labels Per-slot label tuple to look up.
   * @return @c true if @p labels (or the unambiguous trivial-key shortcut)
   *         maps to a stored block.
   */
  bool has_block(const Labels& labels) const {
    if (_is_trivial_key(labels)) {
      return _blocks.size() == 1 || _group_by_pointer().size() == 1;
    }
    return _blocks.find(labels) != _blocks.end();
  }

  /**
   * @brief Reference to the block stored for @p labels.
   *
   * If all labels in the key are trivial (empty), returns the sole block
   * when exactly one unique block exists.
   *
   * @param labels Per-slot label tuple identifying the block.
   * @return Const reference to the requested block storage.
   * @throws std::invalid_argument if no such block exists or the trivial
   *         key is ambiguous.
   */
  const Block& block(const Labels& labels) const { return *block_ptr(labels); }

  /**
   * @brief Shared pointer to the block stored for @p labels.
   *
   * If all labels in the key are trivial (empty), returns the sole block
   * when exactly one unique block exists.
   *
   * @param labels Per-slot label tuple identifying the block.
   * @return Shared pointer to the requested block storage; same instance is
   *         returned for every key that aliases the same storage.
   * @throws std::invalid_argument if no such block exists or the trivial
   *         key is ambiguous.
   */
  BlockPtr block_ptr(const Labels& labels) const {
    if (_is_trivial_key(labels)) {
      if (_blocks.empty()) {
        throw std::invalid_argument("SymmetryBlocked has no blocks.");
      }
      if (_blocks.size() == 1) {
        return _blocks.begin()->second;
      }
      // Check if all blocks alias the same pointer (restricted).
      const auto* first = _blocks.begin()->second.get();
      bool all_same = true;
      for (const auto& [k, v] : _blocks) {
        if (v.get() != first) {
          all_same = false;
          break;
        }
      }
      if (all_same) {
        return _blocks.begin()->second;
      }
      throw std::invalid_argument(
          "Trivial (empty) label key is ambiguous: multiple independent "
          "blocks exist.");
    }
    auto it = _blocks.find(labels);
    if (it == _blocks.end()) {
      throw std::invalid_argument(
          "SymmetryBlocked has no block for the requested labels.");
    }
    return it->second;
  }

  /**
   * @brief Total number of stored blocks (including aliases).
   * @return Size of the underlying @ref BlockMap; counts each key
   *         independently even if multiple keys alias the same storage.
   */
  std::size_t num_blocks() const { return _blocks.size(); }

  /**
   * @brief Whether every stored key in @p keys aliases the same block.
   *
   * Keys absent from the container are skipped (they do not break aliasing).
   * Returns @c true when zero or one of @p keys is stored, or when all
   * stored keys map to the same underlying block pointer.
   *
   * @param keys Per-slot label tuples to compare.
   * @return @c true iff the stored keys all share storage; @c false if any
   *         two stored keys point at different blocks.
   */
  bool all_aliased(const std::vector<Labels>& keys) const {
    const Block* first = nullptr;
    for (const auto& k : keys) {
      if (!has_block(k)) continue;
      const Block* p = block_ptr(k).get();
      if (first == nullptr) {
        first = p;
      } else if (p != first) {
        return false;
      }
    }
    return true;
  }

 protected:
  SymmetriesArray _symmetries;
  ExtentsArray _extents;
  BlockMap _blocks;

  /**
   * @brief True iff every label slot in @p labels is the trivial (empty)
   * label.
   *
   * Trivial keys are special-cased by @ref has_block / @ref block_ptr so
   * that callers that do not care about symmetry can address the unique
   * block of a single-block container without constructing a matching
   * label.
   *
   * @param labels Per-slot label tuple to inspect.
   * @return @c true iff every slot's label is empty.
   */
  static bool _is_trivial_key(const Labels& labels) {
    for (const auto& label : labels) {
      if (!label.empty()) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Internal grouping of blocks by underlying storage pointer, used
   * for canonical-key enumeration during serialization and aliasing
   * detection.
   */
  struct PointerGroup {
    /// One canonical key chosen as the group representative.
    Labels representative;
    /// All keys whose blocks alias this group's pointer.
    std::vector<Labels> keys;
    /// The shared block storage backing every key in @ref keys.
    BlockPtr ptr;
  };

  /**
   * @brief True iff @p label is admissible under @p sym (same axes carrying
   * values that @p sym 's axes admit).
   *
   * @param sym Per-slot symmetries to validate against.
   * @param label Candidate label to test.
   * @return @c true iff @p label carries exactly one value per axis of
   *         @p sym and each value is admissible under the matching axis.
   */
  static bool _label_admissible(const SymmetryProduct& sym,
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

  /**
   * @brief Validate that every slot's @ref SymmetryProduct is non-null and
   * every declared extent label is admissible under its slot's symmetries.
   * @throws std::invalid_argument on failure.
   */
  void _validate_symmetries_and_extents() const {
    for (std::size_t i = 0; i < Rank; ++i) {
      if (_symmetries[i] == nullptr) {
        throw std::invalid_argument(
            "SymmetryBlocked slot symmetries must not be null.");
      }
      for (const auto& [label, extent] : _extents[i]) {
        (void)extent;
        if (!_label_admissible(*_symmetries[i], label)) {
          throw std::invalid_argument(
              "SymmetryBlocked extent label is not admissible under the "
              "slot's symmetries.");
        }
      }
    }
  }

  /**
   * @brief Validate that every block pointer is non-null and every block
   * key carries labels admissible under their slot's symmetries.
   * @throws std::invalid_argument on failure.
   */
  void _validate_block_labels() const {
    for (const auto& [labels, ptr] : _blocks) {
      if (ptr == nullptr) {
        throw std::invalid_argument(
            "SymmetryBlocked block pointers must not be null.");
      }
      for (std::size_t i = 0; i < Rank; ++i) {
        if (!_label_admissible(*_symmetries[i], labels[i])) {
          throw std::invalid_argument(
              "SymmetryBlocked block label is not admissible under the "
              "slot's symmetries.");
        }
      }
    }
  }

  /**
   * @brief Look up the declared extent for @p label on slot @p slot.
   *
   * @param slot Index slot to look up (must be < @p Rank).
   * @param label Candidate label whose extent is requested.
   * @return The declared universe size for @p label on @p slot.
   * @throws std::invalid_argument if @p label is not in @p slot's extents.
   */
  std::size_t _extent_for(std::size_t slot, const SymmetryLabel& label) const {
    auto it = _extents[slot].find(label);
    if (it == _extents[slot].end()) {
      throw std::invalid_argument(
          "SymmetryBlocked block label has no declared extent.");
    }
    return it->second;
  }

  /**
   * @brief True iff every slot carries a spin axis marked
   * @ref SymmetryAxis::equivalent (restricted-spin storage), in which case
   * the constructor auto-aliases each block's spin partner to the same
   * storage.
   *
   * @return @c true iff every slot is restricted-spin.
   */
  bool _spin_aliasing_active() const {
    for (std::size_t i = 0; i < Rank; ++i) {
      if (_symmetries[i] == nullptr ||
          !_symmetries[i]->has_axis(AxisName::Spin) ||
          !_symmetries[i]->axis(AxisName::Spin).equivalent()) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Return a copy of @p label with its spin value flipped
   * (@f$\alpha \leftrightarrow \beta@f$). Non-spin values are unchanged.
   *
   * @param label Source label.
   * @return Spin-flipped copy of @p label.
   */
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

  /**
   * @brief Apply @ref _swap_spin to every slot of @p labels to obtain the
   * spin-partner key.
   *
   * @param labels Source per-slot label tuple.
   * @return Per-slot spin-flipped tuple suitable for indexing the partner
   *         block.
   */
  static Labels _swap_spin_labels(const Labels& labels) {
    Labels swapped = labels;
    for (std::size_t i = 0; i < Rank; ++i) {
      swapped[i] = _swap_spin(labels[i]);
    }
    return swapped;
  }

  /**
   * @brief Extract the per-slot @f$2 M_s@f$ values of @p labels (0 if a
   * slot carries no spin axis).
   *
   * @param labels Per-slot label tuple.
   * @return Per-slot integer array of @f$2 M_s@f$ values.
   */
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

  /**
   * @brief True iff @p labels is the canonical representative of its
   * spin-partner pair (positive @f$2 M_s@f$ on the first spinful slot).
   *
   * @param labels Per-slot label tuple.
   * @return @c true iff @p labels is the canonical (positive-leading-spin)
   *         representative.
   */
  static bool _is_spin_canonical(const Labels& labels) {
    auto seq = _spin_sequence(labels);
    for (std::size_t i = 0; i < Rank; ++i) {
      if (seq[i] != 0) {
        return seq[i] > 0;
      }
    }
    return true;
  }

  /**
   * @brief Materialize the implicit spin-partner aliases produced by
   * restricted-spin storage.
   *
   * If @ref _spin_aliasing_active is true, inserts a block at every
   * @ref _swap_spin_labels key whose partner exists but whose key does
   * not, sharing the partner's storage. Validates that explicitly supplied
   * partners share both extents and storage pointers.
   *
   * @throws std::invalid_argument if restricted spin partners have unequal
   *         extents or are supplied with distinct backing storage.
   */
  void _apply_orbit_aliasing() {
    if (!_spin_aliasing_active()) {
      return;
    }
    for (std::size_t i = 0; i < Rank; ++i) {
      for (const auto& [label, extent] : _extents[i]) {
        SymmetryLabel partner = _swap_spin(label);
        auto it = _extents[i].find(partner);
        if (it != _extents[i].end() && it->second != extent) {
          throw std::invalid_argument(
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
          throw std::invalid_argument(
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

  /**
   * @brief Group block keys by their underlying storage pointer.
   *
   * Used by serialization to enumerate canonical (non-aliased) blocks
   * exactly once; the @ref PointerGroup::representative is chosen so that
   * a spin-canonical key is preferred when available.
   *
   * @return Vector of @ref PointerGroup entries, one per unique block
   *         pointer in the container.
   */
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

  /**
   * @brief Serialize the per-slot @ref SymmetryProduct to a JSON array.
   * @return JSON array of length @p Rank; each entry is either the
   *         @ref SymmetryProduct::to_json output or @c null for an unset
   *         slot.
   */
  nlohmann::json _symmetries_to_json() const {
    nlohmann::json symmetries = nlohmann::json::array();
    for (const auto& sym : _symmetries) {
      symmetries.push_back(sym ? sym->to_json() : nlohmann::json());
    }
    return symmetries;
  }

  /**
   * @brief Serialize the per-slot extents (label → size) to a JSON array.
   * @return JSON array of length @p Rank; each entry is a list of
   *         <tt>{"label": ..., "extent": ...}</tt> objects.
   */
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

  /**
   * @brief Reconstruct a @ref SymmetriesArray from the @c "symmetries" entry
   * of a JSON document produced by @ref _symmetries_to_json.
   *
   * @param j JSON document containing a top-level @c "symmetries" array.
   * @return The reconstructed per-slot @ref SymmetriesArray.
   * @throws nlohmann::json::exception if @p j lacks the @c "symmetries"
   *         field or its entries are malformed.
   */
  static SymmetriesArray _symmetries_from_json(const nlohmann::json& j) {
    SymmetriesArray symmetries;
    const auto& sym_json = j.at("symmetries");
    for (std::size_t i = 0; i < Rank; ++i) {
      symmetries[i] = SymmetryProduct::from_json(sym_json[i]);
    }
    return symmetries;
  }

  /**
   * @brief Reconstruct an @ref ExtentsArray from the @c "extents" entry of a
   * JSON document produced by @ref _extents_to_json.
   *
   * @param j JSON document containing a top-level @c "extents" array.
   * @return The reconstructed per-slot @ref ExtentsArray.
   * @throws nlohmann::json::exception if @p j lacks the @c "extents" field
   *         or its entries are malformed.
   */
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

  /**
   * @brief Serialize the block key groups (canonical key + aliases per
   * group) as a JSON array. Derived classes append per-block payload to
   * each entry before writing.
   *
   * @return JSON array; one entry per unique storage pointer with a
   *         @c "keys" field enumerating the per-slot label tuples that
   *         alias that storage.
   */
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

  /**
   * @brief Reconstruct the per-group block keys from the @c "blocks" entry
   * of a JSON document produced by @ref _block_keys_to_json.
   *
   * @param j JSON document containing a top-level @c "blocks" array.
   * @return Outer vector: one entry per stored block group. Inner vector:
   *         the per-slot label tuples that alias the same block.
   * @throws nlohmann::json::exception if @p j lacks the @c "blocks" field
   *         or any entry is malformed.
   * @throws std::invalid_argument if a key tuple has the wrong arity.
   */
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
