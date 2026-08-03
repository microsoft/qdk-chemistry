// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/orbitals.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class BosonicModes
 * @brief Single-particle basis for bosonic modes with a local occupation
 *        cutoff.
 *
 * `BosonicModes` is a @ref ModelOrbitals specialisation that additionally
 * records, for every mode, the dimension @f$d@f$ of the truncated local Fock
 * space. A mode with dimension @f$d@f$ carries occupations
 * @f$n \in \{0, 1, \ldots, d-1\}@f$, i.e. @f$n_{\max} = d - 1@f$, and the
 * truncated ladder operators are
 *
 * @f[
 *   b = \sum_{n=1}^{d-1} \sqrt{n}\, |n-1\rangle\langle n| , \qquad
 *   \hat n = b^\dagger b = \mathrm{diag}(0, 1, \ldots, d-1) .
 * @f]
 *
 * The truncation is exact except on the top level, where
 * @f$[b, b^\dagger] = I - d\,|d-1\rangle\langle d-1|@f$.
 *
 * Because it derives from @ref Orbitals it can be used anywhere the library
 * expects a single-particle basis — in particular as the basis of a
 * @ref SparseHamiltonianContainer holding a Bose-Hubbard Hamiltonian.
 *
 * @note The occupation cutoff lives on the basis, not on the boson-to-qubit
 *       mapping. A @ref BosonMapping reads its cutoff from here; a mismatch is
 *       a hard error rather than silently-wrong physics.
 *
 * @note @f$d@f$ always means the local Hilbert-space @e dimension,
 *       @f$d = n_{\max}+1@f$ — never the largest occupation number. No API on
 *       this class accepts or stores @f$n_{\max}@f$.
 *
 * @note The cutoff is attributed @e per @e mode: one dimension is stored for
 *       every mode, and @ref mode_dimension is the authoritative accessor. All
 *       consumers must read through it and must not assume homogeneity. Phase
 *       1 only offers uniform public construction, so the stored dimensions
 *       are always equal in practice, but the representation, the serialized
 *       schema and every internal loop are already correct for per-mode
 *       cutoffs.
 *
 * @par Choosing the cutoff
 * Qubit encodings (@ref BosonMapping) require a power-of-two dimension so that
 * the code space is the whole register and there is no unphysical subspace.
 * This class stores exactly the dimension it is given and @e never pads
 * silently — padding is a property of the encoding, not of the basis. Use
 * @ref padded_to_power_of_two (or @ref with_padded_dimensions) to opt in
 * explicitly; it is free in Pauli-term count and strictly reduces the
 * truncation error.
 */
class BosonicModes : public ModelOrbitals {
 public:
  /**
   * @brief Construct a uniform-cutoff bosonic mode basis.
   *
   * Every mode is given the same dimension. This is the only public way to
   * build a basis in phase 1, so the uniform invariant holds by construction.
   *
   * @param num_modes Number of bosonic modes.
   * @param mode_dimension Local Fock-space dimension @f$d = n_{\max}+1@f$ of
   *        every mode; must be at least 2.
   * @throws std::invalid_argument If @p num_modes is 0 or @p mode_dimension
   *         is less than 2.
   */
  BosonicModes(std::size_t num_modes, std::size_t mode_dimension);

  /**
   * @brief Construct from an existing model basis plus a uniform cutoff.
   *
   * Copies the active/inactive index sets and symmetries of @p base and
   * attaches the local Fock-space dimension to every mode.
   *
   * @param base Model basis supplying the mode count and index sets.
   * @param mode_dimension Local Fock-space dimension of every mode.
   * @throws std::invalid_argument If @p mode_dimension is less than 2.
   */
  BosonicModes(const ModelOrbitals& base, std::size_t mode_dimension);

  BosonicModes(const BosonicModes& other) = default;
  BosonicModes& operator=(const BosonicModes& other) = default;
  ~BosonicModes() override = default;

  /**
   * @brief Round a requested local dimension up to the next power of two.
   *
   * @param requested_dimension Requested local Fock-space dimension.
   * @return The smallest power of two greater than or equal to
   *         @p requested_dimension, and at least 2.
   */
  static std::size_t padded_dimension(std::size_t requested_dimension);

  /**
   * @brief Construct a bosonic basis whose cutoff is padded to a power of two.
   *
   * Padding is the recommended way to build a mappable basis: it costs nothing
   * in Pauli-term count (@f$d=3@f$ and @f$d=4@f$ both need 32 hopping terms),
   * removes the unphysical subspace entirely, and only lowers the truncation
   * error. It is deliberately explicit: neither this class nor
   * @ref BosonMapping will ever pad a stated cutoff behind the caller's back.
   *
   * @param num_modes Number of bosonic modes.
   * @param requested_dimension Requested local Fock-space dimension.
   * @return A basis whose mode dimensions are powers of two.
   */
  static std::shared_ptr<BosonicModes> padded_to_power_of_two(
      std::size_t num_modes, std::size_t requested_dimension);

  /**
   * @brief Copy of this basis with every mode dimension padded to a power of
   *        two.
   *
   * Modes that already have a power-of-two dimension are left untouched. This
   * is the instance counterpart of @ref padded_to_power_of_two; it has a
   * distinct name because pybind11 cannot expose a static and an instance
   * method under one Python attribute.
   *
   * @return A basis with the same mode count and padded dimensions.
   */
  std::shared_ptr<BosonicModes> with_padded_dimensions() const;

  /**
   * @brief Local Fock-space dimension of a single mode.
   *
   * This is the authoritative accessor for the occupation cutoff. Prefer it
   * over @ref uniform_dimension in any code that loops over modes.
   *
   * @param mode Mode index.
   * @return Dimension @f$d = n_{\max}+1@f$ of that mode's truncated Fock
   *         space.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  std::size_t mode_dimension(std::size_t mode) const;

  /**
   * @brief Largest occupation number representable on a single mode.
   *
   * Derived from @ref mode_dimension; @f$n_{\max}@f$ is never stored.
   *
   * @param mode Mode index.
   * @return @f$n_{\max} = d - 1@f$ for that mode.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  std::size_t max_occupation(std::size_t mode) const;

  /**
   * @brief All mode dimensions, indexed by mode.
   * @return One local Fock-space dimension per mode.
   */
  const std::vector<std::size_t>& mode_dimensions() const {
    return _mode_dimensions;
  }

  /**
   * @brief Common local dimension, when every mode shares one.
   *
   * Always engaged for a basis built through the phase-1 public constructors.
   *
   * @return The uniform dimension, or @c std::nullopt if the modes do not all
   *         share a single dimension.
   */
  std::optional<std::size_t> uniform_dimension() const;

  /**
   * @brief Whether every mode's dimension is a power of two.
   * @return @c true if all mode dimensions are powers of two.
   */
  bool has_power_of_two_dimensions() const;

  /**
   * @brief Dimension of the full truncated Fock space of all modes.
   * @return @f$\prod_i d_i@f$.
   * @throws std::overflow_error If the product does not fit in a
   *         @c std::size_t.
   */
  std::size_t fock_space_dimension() const;

  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(BosonicModes);
  }

  std::string get_summary() const override;

  nlohmann::json to_json() const override;

  /**
   * @brief Load BosonicModes from JSON.
   * @param j JSON object produced by @ref to_json.
   * @return Shared pointer to the reconstructed basis.
   * @throws std::runtime_error If the JSON is malformed.
   */
  static std::shared_ptr<BosonicModes> from_json(const nlohmann::json& j);

  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load BosonicModes from an HDF5 group.
   * @param group HDF5 group produced by @ref to_hdf5.
   * @return Shared pointer to the reconstructed basis.
   * @throws std::runtime_error If the group is malformed.
   */
  static std::shared_ptr<BosonicModes> from_hdf5(H5::Group& group);

 protected:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

 private:
  /**
   * @brief Construct with an explicit per-mode dimension list.
   *
   * Deliberately non-public in phase 1: the only supported public
   * construction is uniform. Deserialization and @ref padded_to_power_of_two
   * use this path, which is also what a future heterogeneous public
   * constructor will forward to.
   *
   * @param base Model basis supplying the mode count and index sets.
   * @param mode_dimensions One local Fock-space dimension per mode.
   */
  BosonicModes(const ModelOrbitals& base,
               std::vector<std::size_t> mode_dimensions);

  /// Local Fock-space dimension of every mode; exactly one entry per mode.
  std::vector<std::size_t> _mode_dimensions;

  static void _validate_dimension(std::size_t mode, std::size_t mode_dimension);
};

}  // namespace qdk::chemistry::data
