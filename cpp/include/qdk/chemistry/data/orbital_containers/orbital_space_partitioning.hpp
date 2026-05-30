// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <string>
#include <unordered_map>

namespace qdk::chemistry::data {

/**
 * @enum OrbitalSpace
 * @brief The five single-particle orbital subspaces of a partitioning.
 *
 * The integer values are stable and define the canonical storage order.
 */
enum class OrbitalSpace : std::size_t {
  Frozen = 0,    ///< Frozen-core orbitals (occupied, never correlated).
  Inactive = 1,  ///< Inactive orbitals (occupied, correlated externally).
  Active = 2,    ///< Active orbitals (treated explicitly by the solver).
  Virtual = 3,   ///< Virtual orbitals (unoccupied, correlated externally).
  External = 4   ///< External orbitals (outside the modeled space).
};

/**
 * @class OrbitalSpacePartitioning
 * @brief Symmetry-blocked assignment of single-particle modes to the five
 *        @ref OrbitalSpace subspaces.
 *
 * Holds one @ref SymmetryBlockedIndexSet per @ref OrbitalSpace. All five index
 * sets share the same symmetry vocabulary and mode extents. A default (empty)
 * partitioning, in which every mode is @ref OrbitalSpace::Active, can be built
 * with @ref all_active.
 */
class OrbitalSpacePartitioning : public DataClass {
 public:
  static constexpr std::size_t kNumSpaces = 5;
  using IndexSet = SymmetryBlockedIndexSet;
  using IndexSetPtr = std::shared_ptr<const IndexSet>;
  using IndexSetArray = std::array<IndexSetPtr, kNumSpaces>;

  /**
   * @brief Construct from the five per-space index sets (Frozen, Inactive,
   *        Active, Virtual, External in that order).
   * @throws std::invalid_argument if any index set is null
   * @throws SymmetryConditionError if the index sets do not share a common
   *         symmetry vocabulary and mode extents
   */
  explicit OrbitalSpacePartitioning(IndexSetArray spaces);

  /** @brief The index set for orbital subspace @p space. */
  const IndexSetPtr& space(OrbitalSpace space) const {
    return _spaces[static_cast<std::size_t>(space)];
  }

  /** @brief The frozen-core index set. */
  const IndexSetPtr& frozen() const { return space(OrbitalSpace::Frozen); }
  /** @brief The inactive index set. */
  const IndexSetPtr& inactive() const { return space(OrbitalSpace::Inactive); }
  /** @brief The active index set. */
  const IndexSetPtr& active() const { return space(OrbitalSpace::Active); }
  /** @brief The virtual index set. */
  const IndexSetPtr& virtual_orbitals() const {
    return space(OrbitalSpace::Virtual);
  }
  /** @brief The external index set. */
  const IndexSetPtr& external() const { return space(OrbitalSpace::External); }

  /** @brief Symmetry vocabulary shared by all five subspaces. */
  std::shared_ptr<const Symmetries> symmetries() const {
    return _spaces[0]->symmetries();
  }

  /** @brief Per-label mode extents shared by all five subspaces. */
  const std::unordered_map<SymmetryLabel, std::size_t>& mo_extents() const {
    return _spaces[0]->extents();
  }

  /**
   * @brief Build a partitioning in which every mode is active.
   *
   * The @ref OrbitalSpace::Active set contains all modes (indices
   * @c [0, extent) for each label); the other four subspaces are empty.
   *
   * @param symmetries Mode symmetry vocabulary
   * @param mo_extents Per-label mode extents
   * @return Shared pointer to the all-active partitioning
   */
  static std::shared_ptr<OrbitalSpacePartitioning> all_active(
      std::shared_ptr<const Symmetries> symmetries,
      std::unordered_map<SymmetryLabel, std::size_t> mo_extents);

  // ---- DataClass interface ------------------------------------------------

  std::string get_data_type_name() const override {
    return "orbital_space_partitioning";
  }
  std::string get_summary() const override;
  void to_file(const std::string& filename,
               const std::string& type) const override;
  nlohmann::json to_json() const override;
  void to_json_file(const std::string& filename) const override;
  void to_hdf5(H5::Group& group) const override;
  void to_hdf5_file(const std::string& filename) const override;

  static std::shared_ptr<OrbitalSpacePartitioning> from_json(
      const nlohmann::json& j);
  static std::shared_ptr<OrbitalSpacePartitioning> from_json_file(
      const std::string& filename);
  static std::shared_ptr<OrbitalSpacePartitioning> from_hdf5(H5::Group& group);
  static std::shared_ptr<OrbitalSpacePartitioning> from_hdf5_file(
      const std::string& filename);
  static std::shared_ptr<OrbitalSpacePartitioning> from_file(
      const std::string& filename, const std::string& type);

 private:
  IndexSetArray _spaces;

  void _validate() const;
};

}  // namespace qdk::chemistry::data
