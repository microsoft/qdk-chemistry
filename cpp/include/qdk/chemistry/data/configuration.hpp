// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <bitset>
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class Configuration
 * @brief Represents a configuration (or occupation number vector) with
 * efficient bit-packing
 * @details The Configuration class provides a memory-efficient representation
 * of orbital occupations for calculations. Each orbital can be in one of four
 * states: unoccupied, alpha-occupied, beta-occupied, or doubly-occupied. The
 * implementation uses bit-packing (2 bits per orbital) for memory efficiency.
 */

class Configuration : public DataClass {
 public:
  /**
   * @brief Default constructor
   * @details Creates a configuration with 0 orbitals
   */
  Configuration() = default;

  /**
   * @brief Constructor from string representation
   * @param str String representation of configuration (e.g., "22du0d00")
   * where '0' = unoccupied, 'u' = alpha, 'd' = beta, '2' = doubly
   * occupied
   *
   * @note The length of the string determines the number of orbitals
   * @note Orbital indexing is from left to right
   *
   * @throws std::invalid_argument If the string contains invalid characters
   */
  Configuration(const std::string& str);

  /**
   * @brief Constructor from bitset representation
   * @tparam N Size of the bitset (must be even)
   * @param orbs Bitset representation of the state
   * @param num_orbitals Number of spatial orbitals to use from the bitset
   *
   * @note First half of bitset (indices 0 to N/2-1) represents alpha orbitals
   * Second half (indices N/2 to N-1) represents beta orbitals
   *
   * @note Orbital ordering is little-endian (right to left) within each word.
   *
   * @throws std::invalid_argument If num_orbitals exceeds N/2
   */
  template <size_t N>
  Configuration(const std::bitset<N>& orbs, size_t num_orbitals) {
    // Check that N is even
    static_assert(N % 2 == 0, "Bitset size must be even");
    constexpr size_t max_spatial_orbs = N / 2;  // Half for alpha, half for beta

    if (num_orbitals > max_spatial_orbs) {
      throw std::invalid_argument("Number of orbitals exceeds bitset capacity");
    }

    _packed_orbs.resize((num_orbitals + 3) / 4,
                        0);  // Each byte stores 4 orbitals

    for (size_t i = 0; i < num_orbitals; ++i) {
      // Convention: lo-word (alpha) / hi-word (beta)
      bool has_alpha = orbs[i];
      bool has_beta = orbs[max_spatial_orbs + i];

      OccupationState state;
      if (has_alpha && has_beta) {
        state = DOUBLY;
      } else if (has_alpha) {
        state = ALPHA;
      } else if (has_beta) {
        state = BETA;
      } else {
        state = UNOCCUPIED;
      }

      _set_orbital(i, state);
    }
  }

  /**
   * @brief Convert the configuration to a bitset representation
   * @tparam N Size of the bitset (must be even)
   * @return Bitset representation of the configuration
   *
   * @note The returned bitset will have the first half (indices 0 to N/2-1)
   * representing alpha orbitals and the second half (indices N/2 to N-1)
   * representing beta orbitals
   *
   * @note Orbital ordering is little-endian (right to left) within each word.
   *
   * @throws std::invalid_argument If the configuration has more orbitals than
   * N/2
   */
  template <size_t N>
  std::bitset<N> to_bitset() const {
    // check that N is even
    static_assert(N % 2 == 0, "Bitset size must be even");
    constexpr size_t max_spatial_orbs = N / 2;  // Half for alpha, half for beta

    size_t num_orbitals = _packed_orbs.size() * 4;
    if (num_orbitals > max_spatial_orbs) {
      throw std::invalid_argument(
          "Configuration has larger capacity than bitset capacity");
    }

    std::bitset<N> result;
    // Process each spatial orbital in the configuration
    for (size_t i = 0; i < num_orbitals; ++i) {
      OccupationState state = _get_orbital(i);

      // Set alpha bit (lo-word)
      if (state == ALPHA || state == DOUBLY) {
        result.set(i);
      }

      // Set beta bit (hi-word)
      if (state == BETA || state == DOUBLY) {
        result.set(max_spatial_orbs + i);
      }
    }

    return result;
  }

  /**
   * @brief Convert the configuration to a string representation
   * @return String representation where each character represents an orbital:
   *         '0' = unoccupied, 'u' = alpha, 'd' = beta, '2' = doubly occupied
   */
  std::string to_string() const;

  /**
   * @brief Create a canonical Hartree-Fock configuration using the Aufbau
   * principle
   * @param n_alpha Number of alpha electrons
   * @param n_beta Number of beta electrons
   * @param n_orbitals Total number of orbitals
   * @return Configuration representing the HF ground state
   * @details Fills orbitals from lowest energy according to the Aufbau
   * principle:
   *          - Doubly occupied orbitals for paired electrons
   *          - Singly occupied orbitals for unpaired electrons (alpha first if
   * n_alpha > n_beta)
   *          - Unoccupied orbitals for remaining positions
   */
  static Configuration canonical_hf_configuration(size_t n_alpha, size_t n_beta,
                                                  size_t n_orbitals);

  /**
   * @brief Get the number of alpha and beta electrons in the configuration
   * @return A tuple containing (number of alpha electrons, number of beta
   * electrons)
   */
  std::tuple<size_t, size_t> get_n_electrons() const;

  /**
   * @brief Get the max orbital capacity of the configuration
   * @return Number of spatial orbitals the configuration can represent
   */
  size_t get_orbital_capacity() const;

  /**
   * @brief Get the data type name for this class
   * @return "configuration"
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(Configuration);
  }

  /**
   * @brief Check if a specific orbital has an alpha electron
   * @param orbital_idx The orbital index (0-indexed)
   * @return true if the orbital has an alpha electron, false otherwise
   */
  bool has_alpha_electron(size_t orbital_idx) const;

  /**
   * @brief Check if a specific orbital has a beta electron
   * @param orbital_idx The orbital index (0-indexed)
   * @return true if the orbital has a beta electron, false otherwise
   */
  bool has_beta_electron(size_t orbital_idx) const;

  /**
   * @brief Equality comparison operator
   * @param other The configuration to compare with
   * @return true if the configurations are identical, false otherwise
   * @note Used for std::find and other algorithms
   */
  bool operator==(const Configuration& other) const;

  /**
   * @brief Inequality comparison operator
   * @param other The configuration to compare with
   * @return true if the configurations differ, false if they are identical
   */
  bool operator!=(const Configuration& other) const;

  /**
   * @brief Get a summary string describing the configuration
   * @return String containing configuration summary information
   */
  std::string get_summary() const override;

  /**
   * @brief Save configuration to file in the specified format
   * @param filename Path to the output file
   * @param type Format type ("json" or "hdf5")
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert configuration to JSON representation
   * @return JSON object containing the serialized data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save configuration to JSON file
   * @param filename Path to the output JSON file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Save configuration to HDF5 group
   * @param group HDF5 group to save data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save configuration to HDF5 file
   * @param filename Path to the output HDF5 file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load configuration from file in specified format
   * @param filename Path to file to read
   * @param type Format type ("json" or "hdf5")
   * @return New Configuration instance loaded from file
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static Configuration from_file(const std::string& filename,
                                 const std::string& type);

  /**
   * @brief Load configuration from JSON file
   * @param filename Path to JSON file to read
   * @return New Configuration instance loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static Configuration from_json_file(const std::string& filename);

  /**
   * @brief Load configuration from JSON
   * @param j JSON object containing configuration data
   * @return New Configuration instance created from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static Configuration from_json(const nlohmann::json& j);

  /**
   * @brief Load configuration from HDF5 file
   * @param filename Path to HDF5 file to read
   * @return New Configuration instance loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static Configuration from_hdf5_file(const std::string& filename);

  /**
   * @brief Load configuration from HDF5 group
   * @param group HDF5 group to read from
   * @return New Configuration instance created from HDF5 data
   * @throws std::runtime_error if I/O error occurs
   */
  static Configuration from_hdf5(H5::Group& group);

  /**
   * @brief Convert configuration to separate alpha and beta binary strings in
   * little-endian format
   * @param num_orbitals How many orbitals to extract (if we want to slice for
   * active space)
   * @return Pair of binary strings (alpha, beta) where '1' indicates occupied
   *         and '0' indicates unoccupied for each spin channel, in little
   *         endian format
   * @details Converts the compact representation to two binary strings:
   *          - Alpha string: '1' if orbital has alpha or doubly occupied
   *          - Beta string: '1' if orbital has beta or doubly occupied
   *          Example: "2du0" -> ("1010", "1100")
   * @throws std::runtime_error If num_orbitals exceeds configuration capacity
   */
  std::pair<std::string, std::string> to_binary_strings(
      size_t num_orbitals) const;

  /**
   * @brief Convert separate alpha and beta binary strings in little-endian
   * format to a Configuration
   * @param alpha_string Alpha string where '1' indicates occupied
   *         and '0' indicates unoccupied for each spin channel, in little
   *         endian format
   * @param beta_string Beta string where '1' indicates occupied
   *         and '0' indicates unoccupied for each spin channel, in little
   *         endian format
   * @return Configuration object
   */
  static Configuration from_binary_strings(std::string alpha_string,
                                           std::string beta_string);

 private:
  // Friend classes that need direct access to packed data for efficient
  // serialization
  friend class ConfigurationSet;

  /**
   * @brief Enumeration of possible orbital occupation states
   * @details Each state is stored using 2 bits in the packed representation
   */
  enum OccupationState : uint8_t {
    UNOCCUPIED = 0,  // 00 in binary - No electrons
    ALPHA = 1,       // 01 in binary - One alpha electron
    BETA = 2,        // 10 in binary - One beta electron
    DOUBLY = 3       // 11 in binary - Both alpha and beta electrons
  };

  /**
   * @brief Get the occupation state of a specific orbital
   * @param pos The orbital position (0-indexed)
   * @return The occupation state (UNOCCUPIED, ALPHA, BETA, or DOUBLY)
   */
  OccupationState _get_orbital(size_t pos) const;

  /**
   * @brief Set the occupation state of a specific orbital
   * @param pos The orbital position (0-indexed)
   * @param value The occupation state to set (UNOCCUPIED, ALPHA, BETA, or
   * DOUBLY)
   */
  void _set_orbital(size_t pos, OccupationState value);

  /**
   * @brief Storage for packed orbital occupation data
   * @details Each byte stores 4 orbitals (2 bits per orbital occupation state)
   */
  std::vector<uint8_t> _packed_orbs;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(
    DataClassCompliant<Configuration>,
    "Configuration must derive from DataClass and implement all required "
    "deserialization methods");
}  // namespace qdk::chemistry::data
