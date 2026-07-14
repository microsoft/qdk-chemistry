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
 * efficient bit-packing.
 * @details The Configuration class provides a memory-efficient representation
 * of single-particle mode occupations. For spin-½ systems (2 bits per mode),
 * each mode can be unoccupied, alpha-occupied, beta-occupied, or doubly
 * occupied. For generic systems (1 bit per mode), each mode is simply occupied
 * or unoccupied.
 */

class Configuration : public DataClass {
 public:
  /**
   * @brief Default constructor.
   * @details Creates an empty spin-½ configuration with 0 modes.
   */
  Configuration() = default;

  /**
   * @brief Construct from a spin-½ string representation.
   * @deprecated Use from_spin_half_string().
   */
  [[deprecated("Use Configuration::from_spin_half_string() instead.")]]
  Configuration(const std::string& str);

  /**
   * @brief Construct from an alpha/beta bitset.
   * @deprecated Use from_spin_half_bitset().
   */
  template <size_t N>
  [[deprecated("Use Configuration::from_spin_half_bitset() instead.")]]
  Configuration(const std::bitset<N>& orbs, size_t num_orbitals)
      : Configuration(from_spin_half_bitset(orbs, num_orbitals)) {}

  // ---- Factories ----------------------------------------------------------

  /**
   * @brief Construct from a spin-½ string representation.
   * @param str String with alphabet @c '0'/@c 'u'/@c 'd'/@c '2'.
   * @return Configuration with bits_per_mode() == 2.
   * @throws std::invalid_argument If the string contains invalid characters.
   */
  static Configuration from_spin_half_string(const std::string& str);

  /**
   * @brief Construct from a bitstring (1 bit per mode).
   * @param str String with alphabet @c '0'/@c '1'.
   * @return Configuration with bits_per_mode() == 1.
   * @throws std::invalid_argument If the string contains invalid characters.
   */
  static Configuration from_bitstring(const std::string& str);

  /**
   * @brief Construct from an alpha/beta bitset (spin-½).
   * @tparam N Size of the bitset (must be even).
   * @param orbs Bitset with alpha in the low half and beta in the high half.
   * @param num_orbitals Number of spatial orbitals to extract.
   * @return Configuration with bits_per_mode() == 2.
   * @throws std::invalid_argument If num_orbitals exceeds N/2.
   */
  template <size_t N>
  static Configuration from_spin_half_bitset(const std::bitset<N>& orbs,
                                             size_t num_orbitals) {
    static_assert(N % 2 == 0, "Bitset size must be even");
    constexpr size_t max_spatial_orbs = N / 2;

    if (num_orbitals > max_spatial_orbs) {
      throw std::invalid_argument("Number of orbitals exceeds bitset capacity");
    }

    Configuration result;
    result._bits_per_mode = 2;
    result._packed_orbs.resize(_packed_bytes(num_orbitals, 2), 0);

    for (size_t i = 0; i < num_orbitals; ++i) {
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

      result._set_orbital(i, state);
    }
    return result;
  }

  /**
   * @brief Construct from a flat bitset (1 bit per mode).
   * @tparam N Size of the bitset.
   * @param orbs Bitset with one bit per mode.
   * @param num_modes Number of modes to extract.
   * @return Configuration with bits_per_mode() == 1.
   * @throws std::invalid_argument If num_modes exceeds N.
   */
  template <size_t N>
  static Configuration from_bitstring_bitset(const std::bitset<N>& orbs,
                                             size_t num_modes) {
    if (num_modes > N) {
      throw std::invalid_argument("Number of modes exceeds bitset capacity");
    }
    Configuration result;
    result._bits_per_mode = 1;
    result._packed_orbs.resize(_packed_bytes(num_modes, 1), 0);
    for (size_t i = 0; i < num_modes; ++i) {
      if (orbs[i]) result._set_mode_raw(i, 1);
    }
    return result;
  }

  // ---- Conversions --------------------------------------------------------

  /**
   * @brief Convert to a bitset representation.
   *
   * For spin-½ (2 bits/mode): @p N must be even; alpha occupations fill the
   * low half, beta the high half.
   * For bitstring (1 bit/mode): one bit per mode in the low @c capacity()
   * positions.
   *
   * @tparam N Size of the returned bitset.
   * @return Bitset representation of the configuration.
   * @throws std::invalid_argument If the configuration doesn't fit in N bits.
   */
  template <size_t N>
  std::bitset<N> to_bitset() const {
    if (_bits_per_mode == 2) {
      static_assert(N % 2 == 0, "Bitset size must be even for spin-½");
      constexpr size_t max_spatial_orbs = N / 2;
      if (capacity() > max_spatial_orbs) {
        throw std::invalid_argument(
            "Configuration has more modes than bitset capacity");
      }
      std::bitset<N> result;
      for (size_t i = 0; i < capacity(); ++i) {
        OccupationState state = _get_orbital(i);
        if (state == ALPHA || state == DOUBLY) result.set(i);
        if (state == BETA || state == DOUBLY) result.set(max_spatial_orbs + i);
      }
      return result;
    }
    // 1-bit: flat mapping
    if (capacity() > N) {
      throw std::invalid_argument(
          "Configuration has more modes than bitset capacity");
    }
    std::bitset<N> result;
    for (size_t i = 0; i < capacity(); ++i) {
      if (_get_mode_raw(i)) result.set(i);
    }
    return result;
  }

  /**
   * @brief Convert the configuration to a string representation.
   * @return For spin-½ (2 bits/mode): @c '0'/@c 'u'/@c 'd'/@c '2'.
   *         For bitstring (1 bit/mode): @c '0'/@c '1'.
   */
  std::string to_string() const;

  /**
   * @brief Create a canonical Hartree-Fock configuration using the Aufbau
   * principle (spin-½ only).
   * @param n_alpha Number of alpha electrons
   * @param n_beta Number of beta electrons
   * @param n_orbitals Total number of orbitals
   * @return Configuration representing the HF ground state (2 bits/mode)
   */
  static Configuration canonical_hf_configuration(size_t n_alpha, size_t n_beta,
                                                  size_t n_orbitals);

  // ---- Generic (statistics-agnostic) accessors ----------------------------

  /**
   * @brief Number of single-particle modes in the configuration.
   * @return Number of modes (same as @ref get_orbital_capacity).
   */
  size_t capacity() const;

  /**
   * @brief Bits used to encode each mode (1 for spinless, 2 for spin-½).
   * @return Bits per mode.
   */
  uint8_t bits_per_mode() const;

  /**
   * @brief Raw state value for mode @p idx (range 0 to 2^bits_per_mode - 1).
   * @param idx Mode index (0-indexed).
   * @return Packed state value for the mode.
   * @throws std::out_of_range if @p idx >= capacity().
   */
  uint8_t get_mode_state(size_t idx) const;

  /**
   * @brief Return a const reference to the raw packed byte storage.
   *
   * Each byte packs (8 / bits_per_mode) modes. Use bits_per_mode() and
   * the true mode count from the owning container to interpret the data.
   *
   * @return Const reference to the internal packed byte vector.
   */
  const std::vector<uint8_t>& packed_data() const;

  /**
   * @brief Total occupation summed over all modes.
   *
   * For spin-½ modes the per-mode occupation is the popcount of the 2-bit
   * state (0, 1, 1, or 2). For spinless modes it is the 1-bit value itself.
   *
   * @return Sum of per-mode occupations.
   */
  size_t total_occupation() const;

  // ---- Spin-½ specific accessors (throw if bits_per_mode != 2) ------------

  /**
   * @brief Get the number of alpha and beta electrons in the configuration.
   * @return A tuple containing (number of alpha electrons, number of beta
   * electrons)
   * @throws std::runtime_error if bits_per_mode() != 2
   */
  std::tuple<size_t, size_t> get_n_electrons() const;

  /**
   * @brief Get the max orbital capacity of the configuration.
   * @return Number of modes the configuration can represent.
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
   * @brief Check if a specific orbital has an alpha electron (spin-½ only).
   * @param orbital_idx The orbital index (0-indexed)
   * @return true if the orbital has an alpha electron, false otherwise
   * @throws std::runtime_error if bits_per_mode() != 2
   */
  bool has_alpha_electron(size_t orbital_idx) const;

  /**
   * @brief Check if a specific orbital has a beta electron (spin-½ only).
   * @param orbital_idx The orbital index (0-indexed)
   * @return true if the orbital has a beta electron, false otherwise
   * @throws std::runtime_error if bits_per_mode() != 2
   */
  bool has_beta_electron(size_t orbital_idx) const;

  /**
   * @brief Whether the configuration is closed-shell (spin-½ only).
   *
   * Closed-shell means every spatial orbital is either unoccupied or doubly
   * occupied — there are no singly-occupied (open-shell) orbitals. The
   * resulting alpha and beta occupation patterns are identical.
   *
   * @return @c true iff no orbital is singly occupied.
   */
  bool is_closed_shell() const;

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
   * little-endian format (spin-½ only).
   * @param num_orbitals How many orbitals to extract
   * @return Pair of binary strings (alpha, beta)
   * @throws std::runtime_error If bits_per_mode() != 2 or num_orbitals exceeds
   * capacity
   */
  std::pair<std::string, std::string> to_binary_strings(
      size_t num_orbitals) const;

  /**
   * @brief Convert separate alpha and beta binary strings to a spin-½
   * Configuration (2 bits/mode).
   * @param alpha_string Alpha occupation string ('0'/'1')
   * @param beta_string Beta occupation string ('0'/'1')
   * @return Configuration object with bits_per_mode() == 2
   */
  static Configuration from_binary_strings(std::string alpha_string,
                                           std::string beta_string);

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  friend class ConfigurationSet;

  /// Spin-½ occupation states (2-bit interpretation layer).
  enum OccupationState : uint8_t {
    UNOCCUPIED = 0,
    ALPHA = 1,
    BETA = 2,
    DOUBLY = 3
  };

  /// Get the 2-bit occupation state of a spin-½ mode (asserts
  /// _bits_per_mode==2).
  OccupationState _get_orbital(size_t pos) const;

  /// Set the 2-bit occupation state of a spin-½ mode.
  void _set_orbital(size_t pos, OccupationState value);

  /// Generic mode read — returns the raw _bits_per_mode-wide value at @p pos.
  uint8_t _get_mode_raw(size_t pos) const;

  /// Generic mode write — stores a _bits_per_mode-wide @p value at @p pos.
  void _set_mode_raw(size_t pos, uint8_t value);

  /// Throw if _bits_per_mode != 2 with a message naming @p caller.
  void _require_spin_half(const char* caller) const;

  /// Number of packed bytes needed for @p n_modes at @p bpm bits per mode.
  static size_t _packed_bytes(size_t n_modes, uint8_t bpm);

  /// Packed mode data. Layout depends on _bits_per_mode.
  std::vector<uint8_t> _packed_orbs;

  /// Bits used to encode each single-particle mode (1 or 2).
  uint8_t _bits_per_mode = 2;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(
    DataClassCompliant<Configuration>,
    "Configuration must derive from DataClass and implement all required "
    "deserialization methods");
}  // namespace qdk::chemistry::data
