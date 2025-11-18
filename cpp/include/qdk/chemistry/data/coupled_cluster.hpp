// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/data/orbitals.hpp>

namespace qdk::chemistry::data {

/** @brief Class representing the coupled cluster amplitudes.
 *
 * TODO: this class will undergo API changes as it is currently not suited
 * for the use with unrestricted orbitals. It assumes aufbau occupations.
 * 41285/
 */
class CoupledClusterAmplitudes
    : public DataClass,
      public std::enable_shared_from_this<CoupledClusterAmplitudes> {
 public:
  using amplitude_type = Eigen::VectorXd;

  /// @brief  Default constructor
  CoupledClusterAmplitudes();

  /// @brief  Destructor
  virtual ~CoupledClusterAmplitudes() noexcept;

  /** @brief Construct a CoupledClusterAmplitudes object from amplitudes and
   * orbitals
   *
   * @param orbitals The orbital information
   * @param t1_amplitudes The T1 amplitudes
   * @param t2_amplitudes The T2 amplitudes
   * @param n_alpha_electrons The number of alpha electrons
   * @param n_beta_electrons The number of beta electrons
   */
  CoupledClusterAmplitudes(std::shared_ptr<Orbitals> orbitals,
                           const amplitude_type& t1_amplitudes,
                           const amplitude_type& t2_amplitudes,
                           unsigned int n_alpha_electrons,
                           unsigned int n_beta_electrons);

  /**
   * @brief Copy constructor
   * @param other The other CoupledClusterAmplitudes object
   */
  CoupledClusterAmplitudes(const CoupledClusterAmplitudes& other);

  /**
   * @brief Move constructor
   * @param other The other CoupledClusterAmplitudes object
   */
  CoupledClusterAmplitudes(CoupledClusterAmplitudes&& other) noexcept = default;

  /**
   * @brief Assignment operator
   * @param other The other CoupledClusterAmplitudes object
   * @return A reference to this object
   */
  CoupledClusterAmplitudes& operator=(const CoupledClusterAmplitudes& other);

  /**
   * @brief Move assignment operator
   * @param other The other CoupledClusterAmplitudes object
   * @return A reference to this object
   */
  CoupledClusterAmplitudes& operator=(
      CoupledClusterAmplitudes&& other) noexcept = default;

  /**
   * @brief Get the T1 amplitudes
   * @return The T1 amplitudes
   * @throws std::runtime_error if T1 amplitudes are not set
   */
  const amplitude_type& get_t1_amplitudes() const;

  /**
   * @brief Check if T1 amplitudes are set
   * @return True if T1 amplitudes are set, false otherwise
   */
  bool has_t1_amplitudes() const;

  /**
   * @brief Get the T2 amplitudes
   * @return The T2 amplitudes
   * @throws std::runtime_error if T2 amplitudes are not set
   */
  const amplitude_type& get_t2_amplitudes() const;

  /**
   * @brief Check if T2 amplitudes are set
   * @return True if T2 amplitudes are set, false otherwise
   */
  bool has_t2_amplitudes() const;

  /**
   * @brief Get number of occupied orbitals
   * @return Pair with counts of occupied orbitals for (alpha, beta) channels
   */
  std::pair<size_t, size_t> get_num_occupied() const;

  /**
   * @brief Get number of virtual orbitals
   * @return Pair with counts of virtual orbitals for (alpha, beta) channels
   */
  std::pair<size_t, size_t> get_num_virtual() const;

  /**
   * @brief Get a summary string describing the coupled cluster amplitudes
   * @return String containing amplitude summary information
   */
  std::string get_summary() const override;

  /**
   * @brief Save amplitudes to file in the specified format
   * @param filename Path to the output file
   * @param type Format type ("json" or "hdf5")
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */

  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert amplitudes to JSON representation
   * @return JSON object containing the serialized amplitude data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save amplitudes to JSON file
   * @param filename Path to the output JSON file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Save amplitudes to HDF5 group
   * @param group HDF5 group to save data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save amplitudes to HDF5 file
   * @param filename Path to the output HDF5 file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load amplitudes from file based on type parameter
   * @param filename Path to file to read
   * @param type File format type ("json" or "hdf5")
   * @return New CoupledClusterAmplitudes loaded from file
   * @throws std::runtime_error if file doesn't exist, unsupported type, or I/O
   * error occurs
   */
  static std::shared_ptr<CoupledClusterAmplitudes> from_file(
      const std::string& filename, const std::string& type);

  /**
   * @brief Load amplitudes from JSON
   * @param j JSON object containing amplitude data
   * @return Shared pointer to CoupledClusterAmplitudes loaded from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<CoupledClusterAmplitudes> from_json(
      const nlohmann::json& j);

  /**
   * @brief Load amplitudes from JSON file
   * @param filename Path to JSON file to read
   * @return Shared pointer to CoupledClusterAmplitudes loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<CoupledClusterAmplitudes> from_json_file(
      const std::string& filename);

  /**
   * @brief Load amplitudes from HDF5 group
   * @param group HDF5 group to read data from
   * @return Shared pointer to CoupledClusterAmplitudes loaded from group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<CoupledClusterAmplitudes> from_hdf5(H5::Group& group);

  /**
   * @brief Load amplitudes from HDF5 file
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to CoupledClusterAmplitudes loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<CoupledClusterAmplitudes> from_hdf5_file(
      const std::string& filename);

 private:
  /// Molecular orbital data
  std::shared_ptr<Orbitals> _orbitals;

  /// The T1 Amplitudes (NV, NO)
  std::unique_ptr<amplitude_type> _t1_amplitudes;

  /// The T2 Amplitudes (NV, NV, NO, NO)
  std::unique_ptr<amplitude_type> _t2_amplitudes;

  /// Number of occupied orbitals (alpha, beta)
  std::pair<size_t, size_t> _num_occupied;

  /// Number of virtual orbitals (alpha, beta)
  std::pair<size_t, size_t> _num_virtual;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<CoupledClusterAmplitudes>,
              "CoupledClusterAmplitudes must derive from DataClass"
              " and implement all required "
              "deserialization methods");

}  // namespace qdk::chemistry::data
