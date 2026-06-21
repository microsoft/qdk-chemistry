// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Specification for post-mapping qubit tapering.
 *
 * Records which qubits to remove after a fermion-to-qubit mapping and the
 * symmetry eigenvalue (+1 or −1) to substitute for each.  Used by
 * MajoranaMapping to carry tapering metadata through the mapping pipeline.
 */
class TaperingSpecification : public DataClass {
 public:
  /**
   * @brief Construct a tapering specification.
   *
   * @param qubit_indices Indices of qubits to taper (remove).
   * @param eigenvalues   Symmetry eigenvalue (+1 or −1) for each qubit.
   * @throws std::invalid_argument If sizes differ, indices contain duplicates,
   *         or any eigenvalue is not +1 or −1.
   */
  TaperingSpecification(std::vector<std::size_t> qubit_indices,
                        std::vector<int> eigenvalues);

  /// @brief Indices of qubits to taper (remove).
  const std::vector<std::size_t>& qubit_indices() const {
    return qubit_indices_;
  }

  /// @brief Symmetry eigenvalue (+1 or −1) for each tapered qubit.
  const std::vector<int>& eigenvalues() const { return eigenvalues_; }

  /// @brief Number of qubits removed by tapering.
  std::size_t num_tapered() const { return qubit_indices_.size(); }

  /**
   * @brief Tapering for the symmetry-conserving Bravyi-Kitaev encoding.
   *
   * Removes the two qubits that encode the alpha and total particle-number
   * parities in a balanced binary-tree (BK-tree) mapping.
   *
   * @param num_modes Total number of spin-orbitals (must be even, ≥ 4).
   * @param n_alpha   Number of alpha electrons.
   * @param n_beta    Number of beta electrons.
   * @throws std::invalid_argument If num_modes is odd, < 4, or electron
   *         counts exceed spatial orbitals.
   */
  static TaperingSpecification symmetry_conserving_bravyi_kitaev(
      std::size_t num_modes, std::size_t n_alpha, std::size_t n_beta);

  /**
   * @brief Tapering for the parity encoding with two-qubit reduction.
   *
   * Removes the same two symmetry qubits as
   * symmetry_conserving_bravyi_kitaev().
   *
   * @param num_modes Total number of spin-orbitals (must be even, ≥ 4).
   * @param n_alpha   Number of alpha electrons.
   * @param n_beta    Number of beta electrons.
   * @throws std::invalid_argument If num_modes is odd, < 4, or electron
   *         counts exceed spatial orbitals.
   */
  static TaperingSpecification parity_two_qubit_reduction(std::size_t num_modes,
                                                          std::size_t n_alpha,
                                                          std::size_t n_beta);

  /// @brief Get the data type name for serialization.
  std::string get_data_type_name() const override {
    return "tapering_specification";
  }

  /// @brief Get a human-readable summary of the tapering specification.
  std::string get_summary() const override;

  /**
   * @brief Save to file in the specified format.
   * @param filename Path to the output file.
   * @param type Format type ("json", "hdf5", or "h5").
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /// @brief Serialize to JSON.
  nlohmann::json to_json() const override;

  /// @brief Deserialize from JSON.
  static TaperingSpecification from_json(const nlohmann::json& data);

  /// @brief Save to a JSON file.
  void to_json_file(const std::string& filename) const override;

  /// @brief Load from a JSON file.
  static TaperingSpecification from_json_file(const std::string& filename);

  /// @brief Save to an HDF5 group.
  void to_hdf5(H5::Group& group) const override;

  /// @brief Load from an HDF5 group.
  static TaperingSpecification from_hdf5(H5::Group& group);

  /// @brief Save to an HDF5 file.
  void to_hdf5_file(const std::string& filename) const override;

  /// @brief Load from an HDF5 file.
  static TaperingSpecification from_hdf5_file(const std::string& filename);

  /**
   * @brief Load from file in the specified format.
   * @param filename Path to the input file.
   * @param type Format type ("json", "hdf5", or "h5").
   */
  static TaperingSpecification from_file(const std::string& filename,
                                         const std::string& type);

  /// @brief Value equality (compares qubit indices and eigenvalues).
  bool operator==(const TaperingSpecification& other) const;

 private:
  std::vector<std::size_t> qubit_indices_;
  std::vector<int> eigenvalues_;

  /// Feed the tapering specification's identifying data into a content hash.
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /// Serialization schema version.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data
