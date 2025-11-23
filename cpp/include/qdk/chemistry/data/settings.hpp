// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <any>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/data_class.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Type-safe variant for storing different setting value types
 *
 * This variant can hold common types used in settings configurations.
 * Note: All integer types are stored internally as int64_t (signed) or uint64_t
 * (unsigned) for simplicity. Other integer types can be requested via get()
 * with automatic conversion.
 */
using SettingValue =
    std::variant<bool, int64_t, uint64_t, float, double, std::string,
                 std::vector<int64_t>, std::vector<uint64_t>,
                 std::vector<double>, std::vector<std::string>>;

/**
 * @brief Exception thrown when modification of locked settings is requested
 */
class SettingAreLocked : public std::runtime_error {
 public:
  explicit SettingAreLocked()
      : std::runtime_error("Settings are locked: please modify a copy.") {}
};

/**
 * @brief Exception thrown when a setting is not found
 */
class SettingNotFound : public std::runtime_error {
 public:
  explicit SettingNotFound(const std::string& key)
      : std::runtime_error("Setting not found: " + key) {}
};

/**
 * @brief Exception thrown when a setting type conversion fails
 */
class SettingTypeMismatch : public std::runtime_error {
 public:
  explicit SettingTypeMismatch(const std::string& key,
                               const std::string& expected_type)
      : std::runtime_error("Type mismatch for setting '" + key +
                           "'. Expected: " + expected_type) {}
};

/**
 * @brief Base class for extensible settings objects
 *
 * This class provides a flexible settings system that can:
 * - Store arbitrary typed values using a variant system
 * - Be easily extended by derived classes during construction only
 * - Map seamlessly to Python dictionaries via pybind11
 * - Provide type-safe access to settings with default values
 * - Support nested settings structures
 * - Prevent extension of the settings map after class initialization
 *
 * The settings map can only be populated during construction using the
 * protected set_default methods. This design ensures that the available
 * settings are fixed at initialization time and cannot be extended later.
 *
 * Usage:
 * ```cpp
 * class MySettings : public Settings {
 * public:
 *     MySettings() {
 *         // Can only call set_default during construction
 *         set_default("max_iterations", 100);
 *         set_default("tolerance", 1e-6);
 *         set_default("method", std::string("default"));
 *     }
 *
 *     // Convenience getters with validation (optional)
 *     int32_t get_max_iterations() const { return
 * get<int32_t>("max_iterations"); } double get_tolerance() const { return
 * get<double>("tolerance"); } std::string get_method() const { return
 * get<std::string>("method"); }
 *
 *     // After construction, only existing settings can be modified
 *     void set_max_iterations(int32_t value) { set("max_iterations", value); }
 *     void set_tolerance(double value) { set("tolerance", value); }
 * };
 * ```
 */
class Settings : public DataClass,
                 public std::enable_shared_from_this<Settings> {
 public:
  /**
   * @brief Get a summary string describing the settings
   * @return String containing settings summary information
   */
  std::string get_summary() const override;

  /**
   * @brief Default constructor
   */
  Settings() = default;

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~Settings() = default;

  /**
   * @brief Copy constructor
   */
  Settings(const Settings& other) = default;

  /**
   * @brief Move constructor
   */
  Settings(Settings&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Settings& operator=(const Settings& other) = delete;

  /**
   * @brief Move assignment operator
   */
  Settings& operator=(Settings&& other) noexcept = default;

  /**
   * @brief Set a setting value
   * @param key The setting key
   * @param value The setting value
   */
  // TODO (NAB):  Doesn't this function also throw exceptions if the key doesn't
  // exist? Workitem: 38750
  void set(const std::string& key, const SettingValue& value);

  /**
   * @brief Set a setting value (template version for convenience)
   * @param key The setting key
   * @param value The setting value
   */
  template <typename T>
  void set(const std::string& key, const T& value) {
    static_assert(is_supported_type<T>(),
                  "Type not supported in SettingValue variant");
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      throw SettingNotFound(key);
    }

    // If the type is directly in the variant, use it as-is
    if constexpr (is_variant_member_v<T, SettingValue>) {
      settings_[key] = value;
    }
    // Handle integral types - store as int64_t (signed) or uint64_t (unsigned)
    else if constexpr (is_non_bool_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        settings_[key] = static_cast<int64_t>(value);
      } else {
        settings_[key] = static_cast<uint64_t>(value);
      }
    }
    // Handle integer vector types
    else if constexpr (is_non_bool_integral_vector_v<T>) {
      if constexpr (std::is_signed_v<typename T::value_type>) {
        // Signed integer vectors -> vector<int64_t>
        settings_[key] = _convert_to_int64_vector(value);
      } else {
        // Unsigned integer vectors -> vector<uint64_t>
        settings_[key] = _convert_to_uint64_vector(value);
      }
    } else {
      settings_[key] = value;
    }
  }

  /**
   * @brief Sets the value for a given key in the settings.
   *
   * This overload allows setting the value using a C-style string.
   * Internally, the value is converted to a std::string and passed to the main
   * set function.
   *
   * @param key The key to associate with the value.
   * @param value The C-style string value to set.
   */
  void set(const std::string& key, const char* value);

  /**
   * @brief Get a setting value as variant
   * @param key The setting key
   * @return The setting value as SettingValue variant
   * @throws SettingNotFound if key doesn't exist
   */
  SettingValue get(const std::string& key) const;

  /**
   * @brief Get a setting value with type checking
   * @param key The setting key
   * @return The setting value
   * @throws SettingNotFound if key doesn't exist
   * @throws SettingTypeMismatch if type conversion fails
   */
  template <typename T>
  T get(const std::string& key) const {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      throw SettingNotFound(key);
    }

    // If the type is directly in the variant, return it
    if constexpr (is_variant_member_v<T, SettingValue>) {
      try {
        return std::get<T>(it->second);
      } catch (const std::bad_variant_access&) {
        throw SettingTypeMismatch(key, typeid(T).name());
      }
    }
    // Handle integral type conversions
    else if constexpr (is_non_bool_integral_v<T>) {
      try {
        // Check which type is actually stored to avoid unnecessary conversion
        // attempts
        if (std::holds_alternative<int64_t>(it->second)) {
          if (auto result = _try_convert_from<T, int64_t>(it->second)) {
            return *result;
          }
        } else if (std::holds_alternative<uint64_t>(it->second)) {
          if (auto result = _try_convert_from<T, uint64_t>(it->second)) {
            return *result;
          }
        }
        throw SettingTypeMismatch(key, typeid(T).name());
      } catch (...) {
        throw SettingTypeMismatch(key, typeid(T).name());
      }
    }
    // Handle vector<int> and other integer vector conversions
    else if constexpr (is_vector_v<T>) {
      using ElementType = typename T::value_type;
      if constexpr (is_non_bool_integral_v<ElementType>) {
        // Try signed vector first (vector<int64_t>)
        if (std::holds_alternative<std::vector<int64_t>>(it->second)) {
          const auto& vec64 = std::get<std::vector<int64_t>>(it->second);
          T result;
          result.reserve(vec64.size());
          for (const auto& val : vec64) {
            if (auto converted = _safe_convert<ElementType>(val)) {
              result.push_back(*converted);
            } else {
              throw SettingTypeMismatch(key, "vector element out of range");
            }
          }
          return result;
        }
        // Try unsigned vector (vector<uint64_t>)
        else if (std::holds_alternative<std::vector<uint64_t>>(it->second)) {
          const auto& vecu64 = std::get<std::vector<uint64_t>>(it->second);
          T result;
          result.reserve(vecu64.size());
          for (const auto& val : vecu64) {
            if (auto converted = _safe_convert<ElementType>(val)) {
              result.push_back(*converted);
            } else {
              throw SettingTypeMismatch(key, "vector element out of range");
            }
          }
          return result;
        }
      }
      throw SettingTypeMismatch(key, typeid(T).name());
    } else {
      throw SettingTypeMismatch(key, typeid(T).name());
    }
  }

  /**
   * @brief Get a setting value with a default if not found (variant version)
   * @param key The setting key
   * @param default_value The default value to return if key not found
   * @return The setting value or default
   */
  SettingValue get_or_default(const std::string& key,
                              const SettingValue& default_value) const;

  /**
   * @brief Get a setting value with a default if not found (template version)
   * @param key The setting key
   * @param default_value The default value to return if key not found
   * @return The setting value or default
   */
  template <typename T>
  T get_or_default(const std::string& key, const T& default_value) const {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      return default_value;
    }

    // If the type is directly in the variant, try to get it
    if constexpr (is_variant_member_v<T, SettingValue>) {
      try {
        return std::get<T>(it->second);
      } catch (const std::bad_variant_access&) {
        // Fall through to try conversion if T is an integral type
        if constexpr (is_non_bool_integral_v<T>) {
          // Type is in variant but wrong signedness - try conversion
          if (auto result = _try_convert_from<T, int64_t>(it->second)) {
            return *result;
          } else if (auto result = _try_convert_from<T, uint64_t>(it->second)) {
            return *result;
          }
        }
        return default_value;
      }
    }
    // Handle integral type conversions
    else if constexpr (is_non_bool_integral_v<T>) {
      try {
        // Try conversion from int64_t and uint64_t only (simplified)
        if (auto result = _try_convert_from<T, int64_t>(it->second)) {
          return *result;
        } else if (auto result = _try_convert_from<T, uint64_t>(it->second)) {
          return *result;
        }
      } catch (...) {
        // Conversion failed or would lose data
      }
      return default_value;
    } else {
      return default_value;
    }
  }

  /**
   * @brief Check if a setting exists
   * @param key The setting key
   * @return true if the setting exists
   */
  bool has(const std::string& key) const;

  /**
   * @brief Get all setting keys
   * @return Vector of all setting keys
   */
  std::vector<std::string> keys() const;

  /**
   * @brief Get the number of settings
   * @return Number of settings
   */
  size_t size() const;

  /**
   * @brief Check if settings are empty
   * @return true if no settings are stored
   */
  bool empty() const;

  /**
   * @brief Get a setting value as a string representation
   * @param key The setting key
   * @return String representation of the value
   * @throws SettingNotFound if key doesn't exist
   */
  std::string get_as_string(const std::string& key) const;

  /**
   * @brief Get all settings as a map for Python interoperability
   * @return Map of setting keys to their SettingValue variants
   */
  const std::map<std::string, SettingValue>& get_all_settings() const;

  /**
   * @brief Set settings from a map (useful for Python dictionary conversion)
   * @param settings_map Map of settings to set
   */
  void set_from_map(const std::map<std::string, SettingValue>& settings_map);

  /**
   * @brief Convert settings to JSON
   * @return JSON object containing all settings
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Create settings from JSON (static factory method)
   * @param json_obj JSON object containing settings data
   * @return Shared pointer to new Settings instance
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<Settings> from_json(const nlohmann::json& json_obj);

  /**
   * @brief Save settings to JSON file
   * @param filename Path to JSON file to write
   * @throws std::runtime_error if file cannot be opened or written
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Create settings from JSON file (static factory method)
   * @param filename Path to JSON file to read
   * @return Shared pointer to new Settings instance
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Settings> from_json_file(const std::string& filename);

  /**
   * @brief Save settings to HDF5 group
   * @param group HDF5 group to save data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save settings to HDF5 file
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Create settings from HDF5 file (static factory method)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to new Settings instance
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Settings> from_hdf5_file(const std::string& filename);

  /**
   * @brief Create settings from HDF5 group (static factory method)
   * @param group HDF5 group to read
   * @return Shared pointer to new Settings instance
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<Settings> from_hdf5(H5::Group& group);

  /**
   * @brief Save settings to file in specified format
   * @param filename Path to file to create/overwrite
   * @param type Format type ("json" or "hdf5")
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if I/O error occurs
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Create settings from file in specified format (static factory
   * method)
   * @param filename Path to file to read
   * @param type Format type ("json" or "hdf5")
   * @return Shared pointer to new Settings instance
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Settings> from_file(const std::string& filename,
                                             const std::string& type);

  /**
   * @brief Validate that all required settings are present
   * @param required_keys Vector of required setting keys
   * @throws SettingNotFound if any required setting is missing
   */
  void validate_required(const std::vector<std::string>& required_keys) const;

  /**
   * @brief Try to get a setting value with type checking, returns optional
   * @param key The setting key
   * @return Optional containing the value if found and correct type, empty
   * otherwise
   */
  template <typename T>
  std::optional<T> try_get(const std::string& key) const {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      return std::nullopt;
    }

    try {
      return std::get<T>(it->second);
    } catch (const std::bad_variant_access&) {
      return std::nullopt;
    }
  }

  /**
   * @brief Check if a setting exists and has the expected type
   * @param key The setting key
   * @return true if the setting exists and has the correct type
   */
  template <typename T>
  bool has_type(const std::string& key) const {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      return false;
    }
    return std::holds_alternative<T>(it->second);
  }

  /**
   * @brief Get the type name of a setting value
   * @param key The setting key
   * @return String representation of the type, or "not_found" if key doesn't
   * exist
   */
  std::string get_type_name(const std::string& key) const;

  /**
   * @brief Update a setting value, throwing if key doesn't exist
   * @param key The setting key
   * @param value The new value
   * @throws SettingNotFound if key doesn't exist
   */
  template <typename T>
  void update(const std::string& key, const T& value) {
    if (!has(key)) {
      throw SettingNotFound(key);
    }
    set(key, value);
  }

  /**
   * @brief Update a setting value (variant version), throwing if key doesn't
   * exist
   * @param key The setting key
   * @param value The new value
   * @throws SettingNotFound if key doesn't exist
   */
  void update(const std::string& key, const SettingValue& value);

  /**
   * @brief Apply multiple settings from a map
   * @param updates_map Map of setting keys to their new values
   * @throws SettingNotFound if any key doesn't exist
   * @throws SettingTypeMismatch if any value type is incompatible
   *
   * This method performs a bulk update of settings. All keys in the map must
   * already exist in the settings. If any key is missing, the entire operation
   * fails and no settings are modified. This ensures atomicity of the bulk
   * update.
   */
  void update(const std::map<std::string, SettingValue>& updates_map);

  /**
   * @brief Apply multiple settings from a string-to-string map
   * @param updates_map Map of setting keys to their new values as strings
   * @throws SettingNotFound if any key doesn't exist
   * @throws std::runtime_error if any string value cannot be parsed to the
   * expected type
   *
   * This method performs a bulk update of settings from string representations.
   * The string values are parsed based on the current type of each setting.
   * All keys in the map must already exist in the settings. If any key is
   * missing or any value cannot be parsed, the entire operation fails and no
   * settings are modified. This ensures atomicity of the bulk update.
   *
   * Supported string formats:
   * - bool: "true"/"false", "1"/"0", "yes"/"no", "on"/"off" (case-insensitive)
   * - integers: Standard integer format (e.g., "123", "-456")
   * - floating-point: Standard float format (e.g., "3.14", "1e-6")
   * - string: Direct string value
   * - vectors: JSON array format (e.g., "[1,2,3]" or "[\"a\",\"b\",\"c\"]")
   */
  void update(const std::map<std::string, std::string>& updates_map);

  /**
   * @brief Apply settings from another Settings object
   * @param other_settings The Settings object to copy values from
   * @throws SettingNotFound if any key from other_settings doesn't exist in
   * this object
   * @throws SettingTypeMismatch if any value type is incompatible
   *
   * This method performs a bulk update of settings from another Settings
   * object. Only keys that exist in both this object and the other_settings
   * object will be updated. All keys in other_settings must already exist in
   * this settings object. If any key is missing, the entire operation fails and
   * no settings are modified. This ensures atomicity of the bulk update.
   */
  void update(const Settings& other_settings);

  /**
   * @brief Lock the settings to prevent further modifications
   */
  void lock() const;

 protected:
  /**
   * @brief Set a default value for a setting (only if not already set) -
   * variant version
   *
   * This method is protected to ensure that default values can only be set
   * during class initialization, preventing extension of the settings map after
   * construction.
   *
   * @param key The setting key
   * @param value The default value
   */
  void set_default(const std::string& key, const SettingValue& value);

  /**
   * @brief Set a default value for a setting (only if not already set) -
   * template version
   *
   * This method is protected to ensure that default values can only be set
   * during class initialization, preventing extension of the settings map after
   * construction.
   *
   * @param key The setting key
   * @param value The default value
   */
  template <typename T>
  void set_default(const std::string& key, const T& value) {
    if (!has(key)) {
      // If the type is directly in the variant, use it as-is
      if constexpr (is_variant_member_v<T, SettingValue>) {
        settings_[key] = value;
      }
      // Handle integral types - store as int64_t (signed) or uint64_t
      // (unsigned)
      else if constexpr (is_non_bool_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
          settings_[key] = static_cast<int64_t>(value);
        } else {
          settings_[key] = static_cast<uint64_t>(value);
        }
      }
      // Handle integer vector types
      else if constexpr (is_non_bool_integral_vector_v<T>) {
        if constexpr (std::is_signed_v<typename T::value_type>) {
          // Signed integer vectors -> vector<int64_t>
          settings_[key] = _convert_to_int64_vector(value);
        } else {
          // Unsigned integer vectors -> vector<uint64_t>
          settings_[key] = _convert_to_uint64_vector(value);
        }
      } else {
        settings_[key] = value;
      }
    }
  }

  /**
   * @brief Sets a default value for a given key using a C-style string.
   *
   * This overload allows setting the default value for the specified key
   * by passing a C-style string (`const char*`). Internally, it converts
   * the C-style string to a `std::string` and delegates to the corresponding
   * `set_default` function that accepts a `std::string` value.
   *
   * @param key The key for which to set the default value.
   * @param value The default value to associate with the key, as a C-style
   * string.
   */
  void set_default(const std::string& key, const char* value);

 private:
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Helper to check if a type is in the SettingValue variant
   */
  template <typename T, typename Variant>
  struct is_variant_member;

  template <typename T, typename... Ts>
  struct is_variant_member<T, std::variant<Ts...>>
      : std::disjunction<std::is_same<T, Ts>...> {};

  template <typename T, typename Variant>
  inline static constexpr bool is_variant_member_v =
      is_variant_member<T, Variant>::value;

  /**
   * @brief Helper to check if a type is a std::vector
   */
  template <typename T>
  struct is_vector : std::false_type {};

  template <typename T, typename A>
  struct is_vector<std::vector<T, A>> : std::true_type {};

  template <typename T>
  inline static constexpr bool is_vector_v = is_vector<T>::value;

  template <typename T>
  struct is_non_bool_integral
      : std::conjunction<std::is_integral<T>,
                         std::negation<std::is_same<T, bool>>> {};

  template <typename T>
  inline static constexpr bool is_non_bool_integral_v =
      is_non_bool_integral<T>::value;

  template <typename T>
  inline static constexpr bool is_non_bool_integral_vector_v =
      is_vector_v<T> && is_non_bool_integral_v<typename T::value_type>;

  /**
   * @brief Helper to try converting from a specific type if it's in the variant
   */
  template <typename TargetT, typename SourceT>
  static std::optional<TargetT> _try_convert_from(const SettingValue& value) {
    if constexpr (is_variant_member_v<SourceT, SettingValue>) {
      if (std::holds_alternative<SourceT>(value)) {
        return _safe_convert<TargetT>(std::get<SourceT>(value));
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Safely convert between types with range checks
   *
   * This function attempts to convert a value of type SourceT to TargetT,
   * ensuring that the conversion is safe and within the bounds of the target
   * type.
   *
   * @tparam TargetT The target type to convert to
   * @tparam SourceT The source type to convert from
   * @param value The value to convert
   * @return std::optional<TargetT> containing the converted value if
   * successful, or std::nullopt if conversion fails
   */
  template <typename TargetT, typename SourceT>
  static std::optional<TargetT> _safe_convert(const SourceT& value) {
    // Fast path: no conversion needed for 64-bit to same 64-bit type
    if constexpr (std::is_same_v<TargetT, SourceT> &&
                  (std::is_same_v<TargetT, int64_t> ||
                   std::is_same_v<TargetT, uint64_t>)) {
      return value;
    }

    if constexpr (std::is_signed_v<TargetT>) {
      // Target type is signed, check both lower and upper bounds
      if (value >= static_cast<SourceT>(std::numeric_limits<TargetT>::min()) &&
          value <= static_cast<SourceT>(std::numeric_limits<TargetT>::max())) {
        return static_cast<TargetT>(value);
      }
    } else {
      // Target type is unsigned, only check non-negative and upper bound
      if (value >= 0 && static_cast<std::make_unsigned_t<SourceT>>(value) <=
                            std::numeric_limits<TargetT>::max()) {
        return static_cast<TargetT>(value);
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Convert a vector of signed integers to vector<int64_t>
   * @tparam T The source integer type
   * @param value The vector to convert
   * @return vector<int64_t> with converted values
   */
  template <typename T>
  static std::vector<int64_t> _convert_to_int64_vector(
      const std::vector<T>& value) {
    std::vector<int64_t> int64_vec(value.size());
    std::transform(value.begin(), value.end(), int64_vec.begin(),
                   [](const T& v) { return static_cast<int64_t>(v); });
    return int64_vec;
  }

  /**
   * @brief Convert a vector of unsigned integers to vector<uint64_t>
   * @tparam T The source integer type
   * @param value The vector to convert
   * @return vector<uint64_t> with converted values
   */
  template <typename T>
  static std::vector<uint64_t> _convert_to_uint64_vector(
      const std::vector<T>& value) {
    std::vector<uint64_t> uint64_vec(value.size());
    std::transform(value.begin(), value.end(), uint64_vec.begin(),
                   [](const T& v) { return static_cast<uint64_t>(v); });
    return uint64_vec;
  }

  /// Storage for all settings
  std::map<std::string, SettingValue> settings_;

  /// Flag to indicate if settings are locked
  mutable bool _locked = false;

  /**
   * @brief Check if a type is supported by SettingValue variant
   */
  template <typename T>
  static constexpr bool is_supported_type() {
    // Use is_variant_member to check against the actual variant types
    // Also allow integral types and integer vectors that are convertible
    if constexpr (is_variant_member_v<T, SettingValue>) {
      return true;
    } else if constexpr (is_non_bool_integral_v<T>) {
      // Allow other integral types that can be safely converted to
      // int64_t/uint64_t
      return true;
    } else if constexpr (is_non_bool_integral_vector_v<T>) {
      // Allow integer vector types that can be converted to vector<int64_t> or
      // vector<uint64_t>
      return true;
    } else {
      return false;
    }
  }

  /**
   * @brief Convert a SettingValue to string representation
   */
  std::string visit_to_string(const SettingValue& value) const;

  /**
   * @brief Convert SettingValue to JSON
   */
  nlohmann::json convert_setting_value_to_json(const SettingValue& value) const;

  /**
   * @brief Convert JSON to SettingValue
   */
  SettingValue convert_json_to_setting_value(const nlohmann::json& j) const;

  /**
   * @brief Convert string to SettingValue based on the current type of the
   * setting
   * @param key The setting key (used to determine expected type)
   * @param str_value The string value to convert
   * @return SettingValue with the converted value
   * @throws std::runtime_error if conversion fails
   */
  SettingValue parse_string_to_setting_value(
      const std::string& key, const std::string& str_value) const;

  /**
   * @brief Save a SettingValue to HDF5 dataset
   */
  void save_setting_value_to_hdf5(H5::Group& group, const std::string& name,
                                  const SettingValue& value) const;

  /**
   * @brief Load a SettingValue from HDF5 dataset
   */
  static SettingValue load_setting_value_from_hdf5(H5::Group& group,
                                                   const std::string& name);

  /**
   * @brief Check if an HDF5 group exists
   */
  static bool group_exists(H5::H5File& file, const std::string& group_name);

  /**
   * @brief Private function to save settings to JSON file without validation
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Private function to load settings from JSON file without validation
   * @param filename Path to JSON file to read
   * @return Shared pointer to new Settings instance
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Settings> _from_json_file(const std::string& filename);

  /**
   * @brief Private function to save settings to HDF5 file without validation
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Private function to load settings from HDF5 file without validation
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to new Settings instance
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Settings> _from_hdf5_file(const std::string& filename);
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<Settings>,
              "Settings must derive from DataClass and implement all required "
              "deserialization methods");

}  // namespace qdk::chemistry::data
