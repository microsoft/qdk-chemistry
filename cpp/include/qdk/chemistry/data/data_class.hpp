// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace H5 {
class Group;
}

#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>

namespace qdk::chemistry::data {

/**
 * @brief Base interface class providing common methods for data classes
 *
 * This class defines a common interface that all QDK chemistry data classes
 * should implement.
 */
class DataClass {
 public:
  virtual ~DataClass() = default;

  /**
   * @brief Get the data type name for this class
   *
   * This is used for file naming conventions and serialization.
   * Derived classes must override this to return their specific type name.
   *
   * @return String containing the data type name (e.g., "structure",
   * "wavefunction")
   */
  virtual std::string get_data_type_name() const = 0;

  /**
   * @brief Get a summary string describing the object
   * @return String containing object summary information
   */
  virtual std::string get_summary() const = 0;

  /**
   * @brief Save object to file in the specified format
   * @param filename Path to the output file
   * @param type Format type (e.g., "json", "hdf5", "xyz")
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */
  virtual void to_file(const std::string& filename,
                       const std::string& type) const = 0;

  /**
   * @brief Convert object to JSON representation
   * @return JSON object containing the serialized data
   */
  virtual nlohmann::json to_json() const = 0;

  /**
   * @brief Save object to JSON file
   * @param filename Path to the output JSON file
   * @throws std::runtime_error if I/O error occurs
   */
  virtual void to_json_file(const std::string& filename) const = 0;

  /**
   * @brief Save object to HDF5 group
   * @param group HDF5 group to save data to
   * @throws std::runtime_error if I/O error occurs
   */
  virtual void to_hdf5(H5::Group& group) const = 0;

  /**
   * @brief Save object to HDF5 file
   * @param filename Path to the output HDF5 file
   * @throws std::runtime_error if I/O error occurs
   */
  virtual void to_hdf5_file(const std::string& filename) const = 0;

 protected:
  /**
   * @brief Default constructor
   */
  DataClass() = default;

  /**
   * @brief Copy constructor
   * @param other The DataClass object to copy from
   */
  DataClass(const DataClass& other) = default;

  /**
   * @brief Copy assignment operator
   * @param other The DataClass object to copy from
   * @return Reference to this object
   */
  DataClass& operator=(const DataClass& other) = default;

  /**
   * @brief Move constructor
   * @param other The DataClass object to move from
   */
  DataClass(DataClass&& other) = default;

  /**
   * @brief Move assignment operator
   * @param other The DataClass object to move from
   * @return Reference to this object
   */
  DataClass& operator=(DataClass&& other) = default;
};

/**
 * @brief Concept to enforce inheritance of DataClass and presence of
 * static deserialization methods
 */
template <typename T>
concept DataClassCompliant = std::derived_from<T, DataClass> && requires {
  T::from_file(std::declval<std::string>(), std::declval<std::string>());
} && requires { T::from_json_file(std::declval<std::string>()); } && requires {
  T::from_json(std::declval<nlohmann::json>());
} && requires { T::from_hdf5_file(std::declval<std::string>()); } && requires {
  T::from_hdf5(std::declval<H5::Group&>());
};
}  // namespace qdk::chemistry::data
