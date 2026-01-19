// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::data {

/**
 * @class Ansatz
 * @brief Represents a quantum chemical ansatz combining a Hamiltonian and
 * wavefunction
 *
 * This class represents a complete quantum chemical ansatz, which consists of:
 * - A Hamiltonian operator describing the system's energy
 * - A wavefunction describing the quantum state
 *
 * The class is immutable after construction, meaning all data must be provided
 * during construction and cannot be modified afterwards. This ensures
 * consistency between the Hamiltonian and wavefunction throughout the
 * calculation.
 *
 * Common use cases:
 * - Configuration interaction (CI) methods
 * - Multi-configuration self-consistent field (MultiConfigurationScf)
 * calculations
 * - Coupled cluster calculations
 * - Energy expectation value computations
 */
class Ansatz : public DataClass, public std::enable_shared_from_this<Ansatz> {
 public:
  /**
   * @brief Constructor with Hamiltonian and Wavefunction objects
   * @param hamiltonian The Hamiltonian operator for the system
   * @param wavefunction The wavefunction describing the quantum state
   * @throws std::invalid_argument if orbital dimensions are inconsistent
   * between Hamiltonian and wavefunction
   */
  Ansatz(const Hamiltonian& hamiltonian, const Wavefunction& wavefunction);

  /**
   * @brief Constructor with shared pointers to Hamiltonian and Wavefunction
   * @param hamiltonian Shared pointer to the Hamiltonian operator
   * @param wavefunction Shared pointer to the wavefunction
   * @throws std::invalid_argument if pointers are nullptr or orbital dimensions
   * are inconsistent
   */
  Ansatz(std::shared_ptr<Hamiltonian> hamiltonian,
         std::shared_ptr<Wavefunction> wavefunction);

  /**
   * @brief Copy constructor
   */
  Ansatz(const Ansatz& other);

  /**
   * @brief Move constructor
   */
  Ansatz(Ansatz&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Ansatz& operator=(const Ansatz& other);

  /**
   * @brief Move assignment operator
   */
  Ansatz& operator=(Ansatz&& other) noexcept = default;

  /**
   * @brief Destructor
   */
  virtual ~Ansatz() = default;

  /**
   * @brief Get shared pointer to the Hamiltonian
   * @return Shared pointer to the Hamiltonian object
   * @throws std::runtime_error if Hamiltonian is not set
   */
  std::shared_ptr<Hamiltonian> get_hamiltonian() const;

  /**
   * @brief Check if Hamiltonian is available
   * @return True if Hamiltonian is set
   */
  bool has_hamiltonian() const;

  /**
   * @brief Get shared pointer to the wavefunction
   * @return Shared pointer to the Wavefunction object
   * @throws std::runtime_error if wavefunction is not set
   */
  std::shared_ptr<Wavefunction> get_wavefunction() const;

  /**
   * @brief Check if wavefunction is available
   * @return True if wavefunction is set
   */
  bool has_wavefunction() const;

  /**
   * @brief Get shared pointer to the orbital basis set from the Hamiltonian
   * @return Shared pointer to the Orbitals object
   * @throws std::runtime_error if orbitals are not available
   */
  std::shared_ptr<Orbitals> get_orbitals() const;

  /**
   * @brief Check if orbital data is available
   * @return True if orbitals are set in both Hamiltonian and wavefunction
   */
  bool has_orbitals() const;

  /**
   * @brief Calculate the energy expectation value ⟨ψ|H|ψ⟩
   * @return Energy expectation value in atomic units
   * @throws std::runtime_error if calculation cannot be performed
   * @note This method will be implemented once energy calculation algorithms
   * are available
   */
  double calculate_energy() const;

  /**
   * @brief Validate orbital consistency between Hamiltonian and wavefunction
   * @throws std::runtime_error if orbital dimensions are inconsistent
   */
  void validate_orbital_consistency() const;

  /**
   * @brief Get the data type name for this class
   * @return "ansatz"
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(Ansatz);
  }
  /**
   * @brief Get a summary string describing the Ansatz
   * @return Human-readable summary of the Ansatz
   */
  std::string get_summary() const override;

  /**
   * @brief Generic file I/O - save to file based on type parameter
   * @param filename Path to file to create/overwrite
   * @param type File format type ("json" or "hdf5")
   * @throws std::runtime_error if unsupported type or I/O error occurs
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert Ansatz to JSON
   * @return JSON object containing Ansatz data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save Ansatz to JSON file (with validation)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize Ansatz to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save Ansatz to HDF5 file (with validation)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Generic file I/O - load from file based on type parameter
   * @param filename Path to file to read
   * @param type File format type ("json" or "hdf5")
   * @return New Ansatz loaded from file
   * @throws std::runtime_error if file doesn't exist, unsupported type, or I/O
   * error occurs
   */
  static std::shared_ptr<Ansatz> from_file(const std::string& filename,
                                           const std::string& type);

  /**
   * @brief Load Ansatz from HDF5 file (with validation)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const Ansatz loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Ansatz> from_hdf5_file(const std::string& filename);

  /**
   * @brief Load Ansatz from JSON file (with validation)
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Ansatz loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Ansatz> from_json_file(const std::string& filename);

  /**
   * @brief Load Ansatz from JSON
   * @param j JSON object containing Ansatz data
   * @return Shared pointer to const Ansatz loaded from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<Ansatz> from_json(const nlohmann::json& j);

  /**
   * @brief Load Ansatz from HDF5 group
   * @param group HDF5 group to read data from
   * @return Shared pointer to const Ansatz loaded from group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<Ansatz> from_hdf5(H5::Group& group);

 private:
  /// Hamiltonian operator for the system
  std::shared_ptr<Hamiltonian> _hamiltonian;

  /// Wavefunction describing the quantum state
  std::shared_ptr<Wavefunction> _wavefunction;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /// Validation helpers
  void _validate_construction() const;

  /**
   * @brief Check if the Ansatz data is complete and consistent
   * @return True if both Hamiltonian and wavefunction are set and have
   * consistent orbital dimensions
   */
  bool _is_valid() const;

  /**
   * @brief Save to JSON file without filename validation (internal use)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Load from JSON file without filename validation (internal use)
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Ansatz loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Ansatz> _from_json_file(const std::string& filename);

  /**
   * @brief Save to HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Load from HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const Ansatz loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Ansatz> _from_hdf5_file(const std::string& filename);
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<Ansatz>,
              "Ansatz must derive from DataClass and implement all required "
              "deserialization methods");

}  // namespace qdk::chemistry::data
