// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <unordered_map>

namespace qdk::chemistry::data {

/**
 * @class SingleParticleBasis
 * @brief Abstract base for objects describing a single-particle (orbital)
 * basis.
 *
 * A single-particle basis exposes the symmetry vocabulary its modes are blocked
 * under, the per-label mode extents, and the total number of modes. Concrete
 * subclasses (e.g. @ref Orbitals) supply the actual coefficient/energy data.
 *
 * This abstraction lets consumers that only need the symmetry-blocked layout of
 * a one-particle space operate on any such space without depending on a
 * concrete orbital representation.
 *
 * @note This class is abstract; it adds three pure-virtual accessors on top of
 *       the @ref DataClass serialization interface, which concrete subclasses
 *       are responsible for implementing.
 */
class SingleParticleBasis : public DataClass {
 public:
  ~SingleParticleBasis() override = default;

  /**
   * @brief Symmetry vocabulary the single-particle modes are blocked under.
   * @return Shared pointer to the mode @ref Symmetries
   */
  virtual std::shared_ptr<const Symmetries> symmetries() const = 0;

  /**
   * @brief Per-label mode extents.
   * @return Mapping from @ref SymmetryLabel to the number of modes carried by
   *         that symmetry block
   */
  virtual std::unordered_map<SymmetryLabel, std::size_t> mo_extents() const = 0;

  /**
   * @brief Total number of single-particle modes across all symmetry blocks.
   * @return The number of modes
   */
  virtual std::size_t num_modes() const = 0;

 protected:
  SingleParticleBasis() = default;
  SingleParticleBasis(const SingleParticleBasis&) = default;
  SingleParticleBasis& operator=(const SingleParticleBasis&) = default;
  SingleParticleBasis(SingleParticleBasis&&) = default;
  SingleParticleBasis& operator=(SingleParticleBasis&&) = default;
};

}  // namespace qdk::chemistry::data
