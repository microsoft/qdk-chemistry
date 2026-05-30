// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <string>
#include <unordered_map>

namespace qdk::chemistry::data {

/**
 * @class OrbitalEnergies
 * @brief Symmetry-blocked single-particle (orbital) energies.
 *
 * Thin, immutable semantic wrapper around a rank-1 @ref SymmetryBlockedTensor
 * over the MO symmetry vocabulary. Each symmetry block holds the energies of
 * the modes carried by that label.
 */
class OrbitalEnergies : public DataClass {
 public:
  using Sbt = SymmetryBlockedTensor<1, double>;

  /**
   * @brief Construct from a rank-1 symmetry-blocked tensor of energies.
   * @param energies Non-null rank-1 tensor over the MO symmetries
   * @throws std::invalid_argument if @p energies is null
   */
  explicit OrbitalEnergies(std::shared_ptr<const Sbt> energies);

  /** @brief The underlying rank-1 symmetry-blocked tensor. */
  const std::shared_ptr<const Sbt>& tensor() const { return _energies; }

  /** @brief Symmetry vocabulary the orbital energies are blocked under. */
  std::shared_ptr<const Symmetries> symmetries() const {
    return _energies->symmetries()[0];
  }

  /** @brief Per-label mode extents. */
  std::unordered_map<SymmetryLabel, std::size_t> mo_extents() const {
    return _energies->extents()[0];
  }

  /** @brief True iff an energy block is stored for @p label. */
  bool has_block(const SymmetryLabel& label) const {
    return _energies->has_block({label});
  }

  /**
   * @brief Energies for the modes carried by @p label.
   * @throws BlockLabelInvalidError if no such block exists
   */
  const Tensor<1, double>& block(const SymmetryLabel& label) const {
    return _energies->block({label});
  }

  // ---- DataClass interface ------------------------------------------------

  std::string get_data_type_name() const override { return "orbital_energies"; }
  std::string get_summary() const override;
  void to_file(const std::string& filename,
               const std::string& type) const override;
  nlohmann::json to_json() const override;
  void to_json_file(const std::string& filename) const override;
  void to_hdf5(H5::Group& group) const override;
  void to_hdf5_file(const std::string& filename) const override;

  static std::shared_ptr<OrbitalEnergies> from_json(const nlohmann::json& j);
  static std::shared_ptr<OrbitalEnergies> from_json_file(
      const std::string& filename);
  static std::shared_ptr<OrbitalEnergies> from_hdf5(H5::Group& group);
  static std::shared_ptr<OrbitalEnergies> from_hdf5_file(
      const std::string& filename);
  static std::shared_ptr<OrbitalEnergies> from_file(const std::string& filename,
                                                    const std::string& type);

 private:
  std::shared_ptr<const Sbt> _energies;
};

}  // namespace qdk::chemistry::data
