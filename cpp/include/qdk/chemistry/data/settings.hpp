// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <any>
#include <concepts>
#include <fstream>
#include <limits>
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
 * Note: All integer types are stored internally as int64_t (signed).
 * Other integer types can be requested via get()
 * with automatic conversion.
 */
using SettingValue =
    std::variant<bool, int64_t, double, std::string, std::vector<int64_t>,
                 std::vector<double>, std::vector<std::string>>;

/**
 * @brief Constraint specifying minimum and maximum bounds for a setting value
 * @tparam T The type of the bounded value (int64_t or double)
 *
 * This constraint type defines inclusive bounds [min, max] for numeric
 * settings. By default, min and max are set to the type's limits.
 */
template <typename T>
struct BoundConstraint {
  T min = std::numeric_limits<T>::min();  ///< Minimum allowed value (inclusive)
  T max = std::numeric_limits<T>::max();  ///< Maximum allowed value (inclusive)
};

/**
 * @brief Constraint specifying a list of allowed values for a setting
 * @tparam T The type of the allowed values (int64_t or std::string)
 *
 * This constraint type defines an explicit set of allowed values for a setting.
 * The setting value must match one of the values in the allowed_values vector.
 */
template <typename T>
struct ListConstraint {
  std::vector<T> allowed_values;  ///< Vector of allowed values for the setting
};

/**
 * @brief Type for specifying limits on setting values
 */
using Constraint =
    std::variant<BoundConstraint<int64_t>, ListConstraint<int64_t>,
                 BoundConstraint<double>, ListConstraint<std::string>>;

/**
 * @brief Concept to detect if a type is a std::vector
 * @tparam T The type to check
 */
template <typename T>
struct is_vector_impl : std::false_type {};

template <typename T, typename A>
struct is_vector_impl<std::vector<T, A>> : std::true_type {};

template <typename T>
concept Vector = is_vector_impl<T>::value;

/**
 * @brief Helper variable template for is_vector (for backward compatibility)
 * @tparam T The type to check
 */
template <typename T>
inline constexpr bool is_vector_v = Vector<T>;

/**
 * @brief Concept for non-bool integral types
 * @tparam T The type to check
 *
 * Matches integral types other than bool.
 */
template <typename T>
concept NonBoolIntegral = std::integral<T> && !std::same_as<T, bool>;

/**
 * @brief Concept for types that are not non-bool integral types (except
 * int64_t)
 * @tparam T The type to check
 *
 * Matches any type that is either not integral, or is bool, or is int64_t.
 * Useful for constraining templates to exclude integer types except int64_t.
 */
template <typename T>
concept NonIntegralBool = !NonBoolIntegral<T> || std::same_as<T, int64_t>;

/**
 * @brief Helper variable template for non-bool integral (for backward
 * compatibility)
 * @tparam T The type to check
 */
template <typename T>
inline constexpr bool is_non_bool_integral_v = NonBoolIntegral<T>;

/**
 * @brief Concept for vectors of non-bool integral types
 * @tparam T The type to check
 *
 * Matches std::vector whose element type is a non-bool integral type
 * (e.g., std::vector<int>, std::vector<uint32_t>, but not std::vector<bool>).
 */
template <typename T>
concept NonBoolIntegralVector =
    Vector<T> && NonBoolIntegral<typename T::value_type>;

/**
 * @brief Helper variable template for non-bool integral vector (for backward
 * compatibility)
 * @tparam T The type to check
 */
template <typename T>
inline constexpr bool is_non_bool_integral_vector_v = NonBoolIntegralVector<T>;

/**
 * @brief Concept to check if a type is a member of a std::variant
 * @tparam T The type to check
 * @tparam Variant The variant type to check against
 */
template <typename T, typename Variant>
struct is_variant_member_impl;

template <typename T, typename... Ts>
struct is_variant_member_impl<T, std::variant<Ts...>>
    : std::disjunction<std::is_same<T, Ts>...> {};

template <typename T, typename Variant>
concept VariantMember = is_variant_member_impl<T, Variant>::value;

/**
 * @brief Helper variable template for variant member check
 * @tparam T The type to check
 * @tparam Variant The variant type to check against
 */
template <typename T, typename Variant>
inline constexpr bool is_variant_member_v = VariantMember<T, Variant>;

/**
 * @brief Concept for types supported by the SettingValue variant
 * @tparam T The type to check
 *
 * A type is supported if it's directly in the variant, or is a non-bool
 * integral type (convertible to int64_t), or is a vector of non-bool integral
 * types (convertible to vector<int64_t>).
 */
template <typename T>
concept SupportedSettingType = VariantMember<T, SettingValue> ||
                               NonBoolIntegral<T> || NonBoolIntegralVector<T>;

/**
 * @brief Exception thrown when modification of locked settings is requested
 */
class SettingsAreLocked : public std::runtime_error {
 public:
  explicit SettingsAreLocked()
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
 *         set_default("convergence_threshold", 1e-6);
 *         set_default("method", std::string("default"));
 *     }
 *
 *     // Convenience getters with validation (optional)
 *     int32_t get_max_iterations() const { return
 * get<int32_t>("max_iterations"); } double get_convergence_threshold() const {
 * return get<double>("convergence_threshold"); } std::string get_method() const
 * { return get<std::string>("method"); }
 *
 *     // After construction, only existing settings can be modified
 *     void set_max_iterations(int32_t value) { set("max_iterations", value); }
 *     void set_convergence_threshold(double value) {
 * set("convergence_threshold", value); }
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
   * @note This template is disabled for non-int64_t integers to avoid ambiguity
   */
  template <NonIntegralBool T>
  void set(const std::string& key, const T& value) {
    static_assert(SupportedSettingType<T>,
                  "Type not supported in SettingValue variant");
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      throw SettingNotFound(key);
    }

    // If the type is directly in the variant, use it as-is
    if constexpr (VariantMember<T, SettingValue>) {
      SettingValue variant_value = value;
      set(key, variant_value);
    }
    // Handle integral types - store as int64_t (signed)
    else if constexpr (NonBoolIntegral<T>) {
      settings_[key] = static_cast<int64_t>(value);
    }
    // Handle integer vector types
    else if constexpr (NonBoolIntegralVector<T>) {
      // Signed integer vectors -> vector<int64_t>
      settings_[key] = _convert_to_int64_vector(value);
    } else {
      settings_[key] = value;
    }
  }

  template <typename Integer>
    requires NonBoolIntegral<Integer> && (!std::same_as<Integer, int64_t>)
  void set(const std::string& key, Integer value) {
    // Check if key exists first
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      throw SettingNotFound(key);
    }

    // Range check for non-int64_t integers
    if constexpr (std::is_signed_v<Integer>) {
      if constexpr (sizeof(Integer) > sizeof(int64_t)) {
        if (value < static_cast<Integer>(std::numeric_limits<int64_t>::min()) ||
            value > static_cast<Integer>(std::numeric_limits<int64_t>::max())) {
          throw std::out_of_range("Value for setting '" + key +
                                  "' cannot be represented as int64_t.");
        }
      }
    } else {
      // Unsigned type
      if (value > static_cast<Integer>(std::numeric_limits<int64_t>::max())) {
        throw std::out_of_range("Value for setting '" + key +
                                "' cannot be represented as int64_t.");
      }
    }
    settings_[key] = static_cast<int64_t>(value);
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
    if constexpr (VariantMember<T, SettingValue>) {
      try {
        return std::get<T>(it->second);
      } catch (const std::bad_variant_access&) {
        throw SettingTypeMismatch(key, typeid(T).name());
      }
    }
    // Handle integral type conversions
    else if constexpr (NonBoolIntegral<T>) {
      try {
        // Check which type is actually stored (only int64_t now)
        if (std::holds_alternative<int64_t>(it->second)) {
          if (auto result = _try_convert_from<T, int64_t>(it->second)) {
            return *result;
          }
        }
        throw SettingTypeMismatch(key, typeid(T).name());
      } catch (...) {
        throw SettingTypeMismatch(key, typeid(T).name());
      }
    }
    // Handle vector<int> and other integer vector conversions
    else if constexpr (Vector<T>) {
      using ElementType = typename T::value_type;
      if constexpr (NonBoolIntegral<ElementType>) {
        // Try signed vector (vector<int64_t>)
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
    if constexpr (VariantMember<T, SettingValue>) {
      try {
        return std::get<T>(it->second);
      } catch (const std::bad_variant_access&) {
        // Fall through to try conversion if T is an integral type
        if constexpr (NonBoolIntegral<T>) {
          // Type is in variant but wrong signedness - try conversion
          if (auto result = _try_convert_from<T, int64_t>(it->second)) {
            return *result;
          }
        }
        return default_value;
      }
    }
    // Handle integral type conversions
    else if constexpr (NonBoolIntegral<T>) {
      try {
        // Try conversion from int64_t only
        if (auto result = _try_convert_from<T, int64_t>(it->second)) {
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
   * @brief Check if a setting has a description
   * @param key The setting key
   * @return true if the setting has a description
   */
  bool has_description(const std::string& key) const;

  /**
   * @brief Get the description of a setting
   * @param key The setting key
   * @return The description string
   * @throws SettingNotFound if key doesn't exist or has no description
   */
  std::string get_description(const std::string& key) const;

  /**
   * @brief Check if a setting has defined limits
   * @param key The setting key
   * @return true if the setting has limits defined
   */
  bool has_limits(const std::string& key) const;

  /**
   * @brief Get the limits of a setting
   * @param key The setting key
   * @return The limit value (can be range or enumeration)
   * @throws SettingNotFound if key doesn't exist or has no limits
   */
  Constraint get_limits(const std::string& key) const;

  /**
   * @brief Check if a setting is documented
   * @param key The setting key
   * @return true if the setting is marked as documented
   * @throws SettingNotFound if key doesn't exist
   */
  bool is_documented(const std::string& key) const;

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

  /**
   * @brief Print settings as a formatted table
   * @param max_width Maximum total width of the table (default: 120)
   * @param show_undocumented Whether to show undocumented settings (default:
   * false)
   * @return Formatted table string
   *
   * Prints all documented settings in a table format with columns:
   * Key, Value, Limits, Description
   *
   * The table fits within the specified width with multi-line descriptions
   * as needed. Non-integer numeric values are displayed in scientific notation.
   */
  std::string as_table(size_t max_width = 120,
                       bool show_undocumented = false) const;

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
   * @param description Optional description of the setting for documentation
   * @param limit Optional constraint on the allowed values
   * @param documented Whether the setting should be included in documentation
   */
  void set_default(const std::string& key, const SettingValue& value,
                   std::optional<std::string> description = std::nullopt,
                   std::optional<Constraint> limit = std::nullopt,
                   bool documented = true);

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
   * @param description Optional description of the setting for documentation
   * @param limit Optional constraint on the allowed values
   * @param documented Whether the setting should be included in documentation
   * @tparam T The type of the value
   */
  template <typename T>
  void set_default(
      const std::string& key, const T& value,
      std::optional<std::string> description = std::nullopt,
      std::optional<std::variant<BoundConstraint<T>, ListConstraint<T>>> limit =
          std::nullopt,
      bool documented = true) {
    static_assert(SupportedSettingType<T>,
                  "Type not supported in SettingValue variant");
    if (!has(key)) {
      // If the type is directly in the variant, use it as-is
      if constexpr (VariantMember<T, SettingValue>) {
        settings_[key] = value;
        if (description.has_value()) {
          descriptions_[key] = *description;
        }
        if (limit.has_value()) {
          // Convert template limit variant to Constraint variant
          std::visit(
              [this, &key](const auto& limit_val) {
                using LimitValType = std::decay_t<decltype(limit_val)>;
                // Convert to the appropriate Constraint type
                if constexpr (std::same_as<LimitValType,
                                           BoundConstraint<int64_t>>) {
                  limits_[key] = limit_val;
                } else if constexpr (std::same_as<LimitValType,
                                                  ListConstraint<int64_t>>) {
                  limits_[key] = limit_val;
                } else if constexpr (std::same_as<LimitValType,
                                                  BoundConstraint<double>>) {
                  limits_[key] = limit_val;
                } else if constexpr (std::same_as<
                                         LimitValType,
                                         ListConstraint<std::string>>) {
                  limits_[key] = limit_val;
                } else {
                  throw std::invalid_argument(
                      "Unsupported limit type for this value type");
                }
              },
              *limit);
        }
        documented_[key] = documented;
      }
      // Handle integral types - store as int64_t (signed)
      else if constexpr (NonBoolIntegral<T>) {
        settings_[key] = static_cast<int64_t>(value);
        if (description.has_value()) {
          descriptions_[key] = *description;
        }
        if (limit.has_value()) {
          std::visit(
              [this, &key](const auto& limit_val) {
                using LimitValType = std::decay_t<decltype(limit_val)>;
                if constexpr (std::same_as<LimitValType, BoundConstraint<T>>) {
                  // Convert T to int64_t for storage
                  limits_[key] = BoundConstraint<int64_t>{
                      static_cast<int64_t>(limit_val.min),
                      static_cast<int64_t>(limit_val.max)};
                } else if constexpr (std::same_as<LimitValType,
                                                  ListConstraint<T>>) {
                  // Convert T values to int64_t for storage
                  ListConstraint<int64_t> constraint;
                  constraint.allowed_values.reserve(
                      limit_val.allowed_values.size());
                  for (const auto& val : limit_val.allowed_values) {
                    constraint.allowed_values.push_back(
                        static_cast<int64_t>(val));
                  }
                  limits_[key] = std::move(constraint);
                } else {
                  // Unsupported limit type for this value type
                  throw std::invalid_argument(
                      "Unsupported limit type for integral value");
                }
              },
              *limit);
        }
        documented_[key] = documented;
      }
      // Handle integer vector types
      else if constexpr (NonBoolIntegralVector<T>) {
        // All integer vectors -> vector<int64_t>
        settings_[key] = _convert_to_int64_vector(value);
        if (description.has_value()) {
          descriptions_[key] = *description;
        }
        if (limit.has_value()) {
          throw std::invalid_argument(
              "Limits are not supported for integral vector defaults when "
              "implicit conversions are required. Use SettingValue types "
              "directly instead.");
        }
        documented_[key] = documented;
      } else {
        settings_[key] = value;
        if (description.has_value()) {
          descriptions_[key] = *description;
        }
        if (limit.has_value()) {
          throw std::invalid_argument(
              "Unsupported limit type for the provided default value");
        }
        documented_[key] = documented;
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
   * @param description Optional description of the setting for documentation
   * @param limit Optional constraint on the allowed string values
   * @param documented Whether the setting should be included in documentation
   */
  void set_default(const std::string& key, const char* value,
                   std::optional<const char*> description = std::nullopt,
                   std::optional<std::vector<const char*>> limit = std::nullopt,
                   bool documented = true);

 private:
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Helper to try converting from a specific type if it's in the variant
   */
  template <typename TargetT, typename SourceT>
  static std::optional<TargetT> _try_convert_from(const SettingValue& value) {
    if constexpr (VariantMember<SourceT, SettingValue>) {
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
    if constexpr (std::same_as<TargetT, SourceT> &&
                  std::same_as<TargetT, int64_t>) {
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

  /// Storage for all settings
  std::map<std::string, SettingValue> settings_;
  std::map<std::string, std::string> descriptions_;
  std::map<std::string, Constraint> limits_;
  std::map<std::string, bool> documented_;

  /// Flag to indicate if settings are locked
  mutable bool _locked = false;

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
