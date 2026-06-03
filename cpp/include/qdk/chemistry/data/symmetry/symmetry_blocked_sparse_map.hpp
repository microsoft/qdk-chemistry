// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Per-block sparse storage: maps an index tuple to a scalar value.
 *
 * Each key is an @c std::array<unsigned, Rank> of per-slot local indices
 * (offsets within the block's declared extents).
 *
 * @tparam Rank   Number of index slots.
 * @tparam Scalar Element type (default @c double).
 */
template <std::size_t Rank, class Scalar = double>
using SparseMapBlock = std::map<std::array<unsigned, Rank>, Scalar>;

/**
 * @brief Symmetry-blocked sparse map.
 *
 * A @ref SymmetryBlockedSparseMap stores the non-zero symmetry sectors of a
 * rank-@p Rank sparse index-value map.  Each block is a
 * @ref SparseMapBlock — a sorted map from per-slot local-index tuples to
 * scalar values.  The block map, SymmetryProduct instances, and orbit aliasing
 * are inherited from @ref SymmetryBlocked.
 *
 * @tparam Rank   Number of label slots (typically 4 for two-body integrals).
 * @tparam Scalar Element type (@c double).
 */
template <std::size_t Rank, class Scalar = double>
class SymmetryBlockedSparseMap
    : public SymmetryBlocked<Rank, SparseMapBlock<Rank, Scalar>> {
  using Base = SymmetryBlocked<Rank, SparseMapBlock<Rank, Scalar>>;

 public:
  using typename Base::BlockMap;
  using typename Base::BlockPtr;
  using typename Base::ExtentsArray;
  using typename Base::Labels;
  using typename Base::SymmetriesArray;
  using IndexTuple = std::array<unsigned, Rank>;
  using SparseBlock = SparseMapBlock<Rank, Scalar>;

  /**
   * @brief Construct from per-slot symmetries, per-slot extents, and a block
   * map of sparse blocks.
   *
   * @throws std::invalid_argument if a label is not admissible, if restricted
   *         orbit partners have unequal extents, if a sparse entry index
   *         exceeds the extent, or if orbit partners do not share storage.
   */
  SymmetryBlockedSparseMap(SymmetriesArray symmetries, ExtentsArray extents,
                           BlockMap blocks)
      : Base(std::move(symmetries), std::move(extents), std::move(blocks)) {
    _validate_sparse_blocks();
  }

  /** @brief Total number of stored (non-zero) entries across all blocks. */
  std::size_t num_entries() const {
    std::size_t count = 0;
    for (const auto& [labels, ptr] : this->_blocks) {
      count += ptr->size();
    }
    return count;
  }

  /** @brief Look up a single entry by labels and index tuple.
   *  @return The value if present, or zero. */
  Scalar get(const Labels& labels, const IndexTuple& idx) const {
    auto it = this->_blocks.find(labels);
    if (it == this->_blocks.end()) {
      return Scalar{};
    }
    auto entry = it->second->find(idx);
    if (entry == it->second->end()) {
      return Scalar{};
    }
    return entry->second;
  }

  // ---- DataClass interface ------------------------------------------------

  /** @brief @ref DataClass type identifier:
   * @c "symmetry_blocked_sparse_map". */
  std::string get_data_type_name() const override {
    return "symmetry_blocked_sparse_map";
  }

  /**
   * @brief Single-line summary including rank, number of stored blocks, and
   * total number of non-zero entries.
   */
  std::string get_summary() const override {
    std::ostringstream oss;
    oss << "SymmetryBlockedSparseMap(rank=" << Rank
        << ", blocks=" << this->num_blocks() << ", entries=" << num_entries()
        << ")";
    return oss.str();
  }

  /**
   * @brief Serialize this sparse map to JSON. One entry per group of
   * pointer-equivalent blocks; each entry lists the canonical key, the
   * aliased keys, and the sparse-entry payload as
   * @c [idx0, idx1, ..., value] tuples.
   */
  nlohmann::json to_json() const override {
    nlohmann::json j;
    j["type"] = "SymmetryBlockedSparseMap";
    j["rank"] = Rank;
    j["symmetries"] = this->_symmetries_to_json();
    j["extents"] = this->_extents_to_json();

    nlohmann::json blocks = nlohmann::json::array();
    for (const auto& group : this->_group_by_pointer()) {
      nlohmann::json keys = nlohmann::json::array();
      for (const auto& key : group.keys) {
        nlohmann::json key_json = nlohmann::json::array();
        for (const auto& label : key) {
          key_json.push_back(label.to_json());
        }
        keys.push_back(std::move(key_json));
      }
      nlohmann::json entries = nlohmann::json::array();
      for (const auto& [idx, value] : *group.ptr) {
        nlohmann::json entry = nlohmann::json::array();
        for (unsigned i : idx) {
          entry.push_back(i);
        }
        entry.push_back(value);
        entries.push_back(std::move(entry));
      }
      blocks.push_back(nlohmann::json{{"keys", std::move(keys)},
                                      {"entries", std::move(entries)}});
    }
    j["blocks"] = std::move(blocks);
    return j;
  }

  /** @brief Serialize this sparse map to a JSON file at @p filename. */
  void to_json_file(const std::string& filename) const override {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    out << to_json().dump(2);
  }

  /** @brief HDF5 serialization is not yet implemented for sparse maps;
   * always throws. */
  void to_hdf5(H5::Group& /*group*/) const override {
    throw std::runtime_error(
        "SymmetryBlockedSparseMap HDF5 serialization not yet implemented.");
  }

  /** @brief HDF5 serialization is not yet implemented for sparse maps;
   * always throws. */
  void to_hdf5_file(const std::string& /*filename*/) const override {
    throw std::runtime_error(
        "SymmetryBlockedSparseMap HDF5 serialization not yet implemented.");
  }

  /**
   * @brief Dispatch to JSON serialization. HDF5 is not yet implemented for
   * sparse maps and will throw.
   * @param filename Target file path.
   * @param type Either @c "json" (supported) or @c "hdf5" (throws).
   * @throws std::invalid_argument if @p type is not supported.
   * @throws std::runtime_error if @p type is @c "hdf5".
   */
  void to_file(const std::string& filename,
               const std::string& type) const override {
    if (type == "json") {
      to_json_file(filename);
    } else if (type == "hdf5") {
      throw std::runtime_error(
          "SymmetryBlockedSparseMap HDF5 serialization not yet implemented.");
    } else {
      throw std::invalid_argument("Unsupported file type: " + type);
    }
  }

  /** @brief Reconstruct a @ref SymmetryBlockedSparseMap from a JSON object
   * produced by @ref to_json. */
  static std::shared_ptr<SymmetryBlockedSparseMap> from_json(
      const nlohmann::json& j) {
    auto symmetries = Base::_symmetries_from_json(j);
    auto extents = Base::_extents_from_json(j);

    BlockMap blocks;
    for (const auto& entry : j.at("blocks")) {
      auto sparse_block = std::make_shared<SparseBlock>();
      for (const auto& e : entry.at("entries")) {
        IndexTuple idx{};
        for (std::size_t i = 0; i < Rank; ++i) {
          idx[i] = e.at(i).template get<unsigned>();
        }
        (*const_cast<SparseBlock*>(sparse_block.get()))[idx] =
            e.at(Rank).template get<Scalar>();
      }
      auto const_block =
          std::const_pointer_cast<const SparseBlock>(sparse_block);
      for (const auto& key_json : entry.at("keys")) {
        std::vector<SymmetryLabel> labels;
        for (const auto& label_json : key_json) {
          labels.push_back(SymmetryLabel::from_json(label_json));
        }
        blocks.emplace(detail::make_labels<Rank>(labels), const_block);
      }
    }
    return std::make_shared<SymmetryBlockedSparseMap>(
        std::move(symmetries), std::move(extents), std::move(blocks));
  }

 private:
  void _validate_sparse_blocks() const {
    for (const auto& [labels, ptr] : this->_blocks) {
      for (const auto& [idx, value] : *ptr) {
        (void)value;
        for (std::size_t i = 0; i < Rank; ++i) {
          auto extent = this->_extent_for(i, labels[i]);
          if (idx[i] >= extent) {
            throw std::invalid_argument(
                "SymmetryBlockedSparseMap entry index exceeds block extent.");
          }
        }
      }
    }
  }
};

}  // namespace qdk::chemistry::data
