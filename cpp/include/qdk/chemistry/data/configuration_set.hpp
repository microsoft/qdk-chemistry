// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class ConfigurationSet
 * @brief Associates a collection of Configuration objects with orbital
 * information
 *
 * This class manages a set of configurations that share the same
 * single-particle basis, specifically the active space of an Orbitals object.
 * By storing the orbital information at the set level rather than in each
 * configuration, we eliminate redundant storage of the number of orbitals.
 *
 * Key design points:
 * - Configurations represent only the active space, not the full orbital set
 * - Inactive and virtual orbitals are not
 *   included in the configuration representation
 * - All configurations in the set must be consistent with the active space size
 * - Provides iteration and access methods similar to std::vector
 * - Immutable after construction to ensure consistency
 */
class ConfigurationSet : public DataClass {
 public:
  // Type aliases
  using ConfigVector = std::vector<Configuration>;
  using iterator = ConfigVector::const_iterator;
  using const_iterator = ConfigVector::const_iterator;

  /**
   * @brief Construct a ConfigurationSet from configurations and orbital
   * information
   * @param configurations Vector of configurations (representing active space
   * only)
   * @param orbitals Shared pointer to orbital basis set
   * @throws std::invalid_argument if configurations are inconsistent with
   * active space or if orbitals pointer is null
   *
   * @note All configurations must have the same number of orbitals and
   * sufficient capacity to represent the active space defined in the orbitals
   * object. Configurations only represent the active space; inactive and
   * virtual orbitals are not included in the configuration representation.
   */
  ConfigurationSet(const std::vector<Configuration>& configurations,
                   std::shared_ptr<Orbitals> orbitals);

  /**
   * @brief Construct a ConfigurationSet from configurations and orbital
   * information (move)
   * @param configurations Vector of configurations (representing active space
   * only, moved)
   * @param orbitals Shared pointer to orbital basis set
   * @throws std::invalid_argument if configurations are inconsistent with
   * active space or if orbitals pointer is null
   *
   * @note All configurations must have the same number of orbitals and
   * sufficient capacity to represent the active space defined in the orbitals
   * object. Configurations only represent the active space; inactive and
   * virtual orbitals are not included in the configuration representation.
   */
  ConfigurationSet(std::vector<Configuration>&& configurations,
                   std::shared_ptr<Orbitals> orbitals);

  /**
   * @brief Get the configurations
   * @return Const reference to vector of configurations
   */
  const std::vector<Configuration>& get_configurations() const;

  /**
   * @brief Get the orbital information
   * @return Shared pointer to orbitals
   */
  std::shared_ptr<Orbitals> get_orbitals() const;

  /**
   * @brief Get the number of configurations in the set
   * @return Number of configurations
   */
  size_t size() const;

  /**
   * @brief Check if the set is empty
   * @return True if empty
   */
  bool empty() const;

  /**
   * @brief Access configuration by index
   * @param idx Index
   * @return Const reference to configuration
   */
  const Configuration& operator[](size_t idx) const;

  /**
   * @brief Access configuration by index with bounds checking
   * @param idx Index
   * @return Const reference to configuration
   * @throws std::out_of_range if index is invalid
   */
  const Configuration& at(size_t idx) const;

  // Iterator support
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

  /**
   * @brief Check equality with another ConfigurationSet
   * @param other Other set
   * @return True if equal (same configurations and orbitals point to same
   * object)
   */
  bool operator==(const ConfigurationSet& other) const;

  /**
   * @brief Check inequality with another ConfigurationSet
   * @param other Other set
   * @return True if not equal
   */
  bool operator!=(const ConfigurationSet& other) const;

  /**
   * @brief Get a summary string describing the configuration set
   * @return String containing object summary information
   */
  std::string get_summary() const override;

  /**
   * @brief Save object to file in the specified format
   * @param filename Path to the output file
   * @param type Format type (e.g., "json", "hdf5")
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */

  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Serialize configuration set to JSON
   * @return JSON representation (includes configurations and orbitals)
   * @note This serializes both configurations and orbitals recursively
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save object to JSON file
   * @param filename Path to the output JSON file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize configuration set to HDF5
   * @param group HDF5 group to write to
   * @note This serializes both configurations and orbitals recursively
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save object to HDF5 file
   * @param filename Path to the output HDF5 file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load configuration set from file in the specified format
   * @param filename Path to the input file
   * @param type Format type (e.g., "json", "hdf5")
   * @return ConfigurationSet object
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */
  static ConfigurationSet from_file(const std::string& filename,
                                    const std::string& type);

  /**
   * @brief Deserialize configuration set from JSON
   * @param j JSON object containing configuration and orbital data
   * @return ConfigurationSet object
   * @note Orbitals are deserialized from the JSON "orbitals" field
   */
  static ConfigurationSet from_json(const nlohmann::json& j);

  /**
   * @brief Load configuration set from JSON file
   * @param filename Path to the input JSON file
   * @return ConfigurationSet object
   * @throws std::runtime_error if I/O error occurs
   */
  static ConfigurationSet from_json_file(const std::string& filename);

  /**
   * @brief Deserialize configuration set from HDF5
   * @param group HDF5 group to read from
   * @return ConfigurationSet object
   * @note Orbitals are deserialized from the HDF5 "orbitals" subgroup
   */
  static ConfigurationSet from_hdf5(H5::Group& group);

  /**
   * @brief Load configuration set from HDF5 file
   * @param filename Path to the input HDF5 file
   * @return ConfigurationSet object
   * @throws std::runtime_error if I/O error occurs
   */
  static ConfigurationSet from_hdf5_file(const std::string& filename);

 private:
  /// Configurations in the set
  std::vector<Configuration> _configurations;

  /// Orbital information (holds active space definition)
  std::shared_ptr<Orbitals> _orbitals;

  /**
   * @brief Validate that all configurations are consistent with active space
   * @throws std::invalid_argument if validation fails
   *
   * Checks that:
   * - All configurations have the same number of orbitals
   * - All configurations have the same electron count (both alpha and beta)
   * - Configurations have sufficient orbital capacity for the active space size
   * - Any orbitals beyond the active space size are unoccupied (no
   * "overhanging" electrons)
   *
   * Note: Configurations only represent the active space orbitals. Inactive
   * and virtual orbitals are not included in
   * the configuration representation, so no validation is performed for those.
   */
  void _validate_configurations() const;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<ConfigurationSet>,
              "ConfigurationSet must derive from DataClass and implement "
              "all required "
              "deserialization methods");

}  // namespace qdk::chemistry::data
