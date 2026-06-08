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
  /**
   * @brief Sparse map from per-slot label tuples to sparse block storage.
   *
   * Inherited from @ref SymmetryBlocked. Aliased sectors map to the same
   * @ref BlockPtr; keys are hashed via @ref LabelsHash.
   */
  using typename Base::BlockMap;
  /**
   * @brief Shared pointer to immutable per-block sparse storage.
   *
   * Equivalent to the base @ref SymmetryBlocked @c BlockPtr with the block
   * type resolved to @ref SparseMapBlock. Held as
   * @c shared_ptr<const SparseMapBlock> so that symmetry-equivalent sectors can
   * alias the same storage.
   */
  using BlockPtr = std::shared_ptr<const SparseMapBlock<Rank, Scalar>>;
  /**
   * @brief Per-slot per-label extents.
   *
   * Inherited from @ref SymmetryBlocked. For each index slot, maps every
   * admissible @ref SymmetryLabel to its universe size.
   */
  using typename Base::ExtentsArray;
  /**
   * @brief Per-slot block label tuple: one @ref SymmetryLabel per index slot.
   *
   * Inherited from @ref SymmetryBlocked. Used as the key type of
   * @ref BlockMap.
   */
  using typename Base::Labels;
  /**
   * @brief Per-slot symmetry definitions.
   *
   * Inherited from @ref SymmetryBlocked. One @ref SymmetryProduct per index
   * slot, supplied at construction.
   */
  using typename Base::SymmetriesArray;
  /** @brief Per-slot local-index tuple keying a single sparse entry within
   *  one block. */
  using IndexTuple = std::array<unsigned, Rank>;
  /** @brief Per-block sparse storage: a sorted map from per-slot local-index
   *  tuples to scalar values. */
  using SparseBlock = SparseMapBlock<Rank, Scalar>;

  /**
   * @brief Construct from per-slot symmetries, per-slot extents, and a block
   * map of sparse blocks.
   *
   * @param symmetries Per-slot @ref SymmetryProduct definitions.
   * @param extents Per-slot per-label universe sizes.
   * @param blocks Block storage keyed by per-slot label tuples; entries of
   *               each block carry per-slot local indices.
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

  /**
   * @brief Total number of stored (non-zero) entries across all blocks.
   * @return Sum of the sparse-entry counts of every stored block.
   */
  std::size_t num_entries() const {
    std::size_t count = 0;
    for (const auto& [labels, ptr] : this->_blocks) {
      count += ptr->size();
    }
    return count;
  }

  /**
   * @brief Look up a single entry by labels and index tuple.
   * @param labels Per-slot symmetry label tuple identifying the block.
   * @param idx Per-slot local index tuple within the block.
   * @return The stored value if present, or @c Scalar{} (zero) otherwise.
   */
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

  /**
   * @brief @ref DataClass type identifier.
   * @return The stable string @c "symmetry_blocked_sparse_map".
   */
  std::string get_data_type_name() const override {
    return "symmetry_blocked_sparse_map";
  }

  /**
   * @brief Single-line summary including rank, number of stored blocks, and
   * total number of non-zero entries.
   * @return A short diagnostic string suitable for logging.
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
   *
   * @return JSON object carrying rank, scalar type, per-slot symmetries and
   *         extents, and the sparse-entry payload.
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

  /**
   * @brief Serialize this sparse map to a JSON file.
   * @param filename Path to the JSON file to create or overwrite.
   * @throws std::runtime_error if the file cannot be opened for writing.
   */
  void to_json_file(const std::string& filename) const override {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    out << to_json().dump(2);
  }

  /**
   * @brief HDF5 serialization is not yet implemented for sparse maps;
   * always throws.
   * @throws std::runtime_error unconditionally.
   */
  void to_hdf5(H5::Group& /*group*/) const override {
    throw std::runtime_error(
        "SymmetryBlockedSparseMap HDF5 serialization not yet implemented.");
  }

  /**
   * @brief HDF5 serialization is not yet implemented for sparse maps;
   * always throws.
   * @throws std::runtime_error unconditionally.
   */
  void to_hdf5_file(const std::string& /*filename*/) const override {
    throw std::runtime_error(
        "SymmetryBlockedSparseMap HDF5 serialization not yet implemented.");
  }

  /**
   * @brief Dispatch to JSON serialization. HDF5 is not yet implemented for
   * sparse maps and will throw.
   * @param filename Target file path.
   * @param type Either @c "json" (supported) or @c "hdf5" (throws).
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if @p type is @c "hdf5" (HDF5 not
   *         implemented) or if the JSON I/O fails.
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

  /**
   * @brief Reconstruct a @ref SymmetryBlockedSparseMap from a JSON object
   * produced by @ref to_json.
   *
   * @param j JSON object produced by a prior @ref to_json call.
   * @return Shared pointer to the reconstructed sparse map.
   * @throws std::invalid_argument if a block label is not admissible or a
   *         sparse entry index exceeds the declared extent.
   * @throws nlohmann::json::exception if @p j is otherwise malformed.
   */
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
        (*sparse_block)[idx] = e.at(Rank).template get<Scalar>();
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

/**
 * @brief Build a single-channel restricted rank-4
 * @ref SymmetryBlockedSparseMap whose @c alpha-alpha-alpha-alpha block is
 * aliased into the @c alpha-alpha-beta-beta key (orbit aliasing on the
 * restricted spin axis fills @c bbbb from @c aaaa and @c bbaa from @c aabb,
 * so all four equivalent spin patterns share a single underlying block).
 *
 * Use for spin-restricted sparse two-electron integrals where
 * @f$(\alpha\alpha|\alpha\alpha) = (\alpha\alpha|\beta\beta) =
 * (\beta\beta|\beta\beta)@f$ holds physically. Mirrors the single-channel
 * dense overload @ref make_spin_diagonal_rank4_sbt(const
 * Eigen::MatrixBase<Derived>&).
 *
 * @tparam Scalar Map value type.
 * @param block Sparse entries for the single channel; keys are per-slot
 *           local indices, values are the integral magnitudes. Moved into
 *           the resulting map.
 * @param n_active Per-spin extent. The alpha and beta extents both equal
 *           @p n_active on every slot.
 * @return Shared pointer to the constructed sparse map, or @c nullptr when
 *         @p block is empty.
 */
template <class Scalar>
std::shared_ptr<const SymmetryBlockedSparseMap<4, Scalar>>
make_spin_diagonal_rank4_sbsm(SparseMapBlock<4, Scalar> block,
                              std::size_t n_active) {
  if (block.empty()) {
    return nullptr;
  }
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/true)}));
  std::unordered_map<SymmetryLabel, std::size_t> ext;
  ext[axes::alpha()] = n_active;
  ext[axes::beta()] = n_active;

  typename SymmetryBlockedSparseMap<4, Scalar>::SymmetriesArray symmetries = {
      sym, sym, sym, sym};
  typename SymmetryBlockedSparseMap<4, Scalar>::ExtentsArray extents = {
      ext, ext, ext, ext};

  auto shared_block =
      std::make_shared<SparseMapBlock<4, Scalar>>(std::move(block));

  typename SymmetryBlockedSparseMap<4, Scalar>::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}] =
      shared_block;
  blocks[{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}] =
      shared_block;

  return std::make_shared<const SymmetryBlockedSparseMap<4, Scalar>>(
      std::move(symmetries), std::move(extents), std::move(blocks));
}

}  // namespace qdk::chemistry::data
