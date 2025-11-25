// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <qdk/chemistry/data/settings.hpp>

#include "filename_utils.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

void Settings::set(const std::string& key, const SettingValue& value) {
  if (_locked) {
    throw SettingAreLocked();
  }
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    throw SettingNotFound(key);
  }
  settings_[key] = value;
}

void Settings::set(const std::string& key, const char* value) {
  if (_locked) {
    throw SettingAreLocked();
  }
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    throw SettingNotFound(key);
  }
  set(key, std::string(value));
}

SettingValue Settings::get(const std::string& key) const {
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    throw SettingNotFound(key);
  }
  return it->second;
}

SettingValue Settings::get_or_default(const std::string& key,
                                      const SettingValue& default_value) const {
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    return default_value;
  }
  return it->second;
}

bool Settings::has(const std::string& key) const {
  return settings_.find(key) != settings_.end();
}

std::vector<std::string> Settings::keys() const {
  std::vector<std::string> result;
  result.reserve(settings_.size());
  for (const auto& [key, value] : settings_) {
    result.push_back(key);
  }
  return result;
}

size_t Settings::size() const { return settings_.size(); }

bool Settings::empty() const { return settings_.empty(); }

std::string Settings::get_summary() const {
  std::ostringstream oss;
  oss << "Settings Summary:\n";

  if (empty()) {
    oss << "  No settings configured.\n";
    return oss.str();
  }

  oss << "  Settings:\n";
  for (const auto& [key, value] : settings_) {
    std::string value_str;
    try {
      value_str = get_as_string(key);
      // Truncate very long values for readability
      if (value_str.length() > 50) {
        value_str = value_str.substr(0, 47) + "...";
      }
    } catch (const std::exception&) {
      value_str = "<error>";
    }

    oss << "    " << key << " = " << value_str << "\n";
  }

  return oss.str();
}

std::string Settings::get_as_string(const std::string& key) const {
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    throw SettingNotFound(key);
  }

  return visit_to_string(it->second);
}

const std::map<std::string, SettingValue>& Settings::get_all_settings() const {
  return settings_;
}

void Settings::set_from_map(
    const std::map<std::string, SettingValue>& settings_map) {
  if (_locked) {
    throw SettingAreLocked();
  }
  // First, validate that all keys exist to ensure atomicity
  for (const auto& [key, value] : settings_map) {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      throw SettingNotFound(key);
    }
  }

  // If all keys exist, perform the updates
  for (const auto& [key, value] : settings_map) {
    settings_[key] = value;
  }
}

nlohmann::json Settings::to_json() const {
  nlohmann::json json_obj;

  // Store version first
  json_obj["version"] = SERIALIZATION_VERSION;

  for (const auto& [key, value] : settings_) {
    json_obj[key] = convert_setting_value_to_json(value);
  }
  return json_obj;
}

std::shared_ptr<Settings> Settings::from_json(const nlohmann::json& json_obj) {
  auto settings = std::make_shared<Settings>();

  if (!json_obj.is_object()) {
    throw std::runtime_error("JSON must be an object");
  }

  // Validate version first (only if version field exists, for backward
  // compatibility)
  if (json_obj.contains("version")) {
    validate_serialization_version(SERIALIZATION_VERSION, json_obj["version"]);
  }

  // For base Settings class, we create settings dynamically from JSON
  // Note: This bypasses the usual restriction of only updating existing keys
  // because the base Settings class needs to be able to deserialize arbitrary
  // settings
  for (const auto& [key, value] : json_obj.items()) {
    // Skip the version field during processing
    if (key == "version") {
      continue;
    }
    settings->settings_[key] = settings->convert_json_to_setting_value(value);
  }

  return settings;
}

void Settings::validate_required(
    const std::vector<std::string>& required_keys) const {
  for (const auto& key : required_keys) {
    if (!has(key)) {
      throw SettingNotFound(key);
    }
  }
}

std::string Settings::get_type_name(const std::string& key) const {
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    return "not_found";
  }

  return std::visit(
      [](const auto& value) -> std::string {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<ValueType, bool>)
          return "bool";
        else if constexpr (std::is_same_v<ValueType, int64_t>)
          return "int64_t";
        else if constexpr (std::is_same_v<ValueType, uint64_t>)
          return "uint64_t";
        else if constexpr (std::is_same_v<ValueType, float>)
          return "float";
        else if constexpr (std::is_same_v<ValueType, double>)
          return "double";
        else if constexpr (std::is_same_v<ValueType, std::string>)
          return "string";
        else if constexpr (std::is_same_v<ValueType, std::vector<int64_t>>)
          return "vector<int64_t>";
        else if constexpr (std::is_same_v<ValueType, std::vector<uint64_t>>)
          return "vector<uint64_t>";
        else if constexpr (std::is_same_v<ValueType, std::vector<double>>)
          return "vector<double>";
        else if constexpr (std::is_same_v<ValueType, std::vector<std::string>>)
          return "vector<string>";
        else
          return "unknown";
      },
      it->second);
}

void Settings::update(const std::string& key, const SettingValue& value) {
  if (_locked) {
    throw SettingAreLocked();
  }
  if (!has(key)) {
    throw SettingNotFound(key);
  }
  set(key, value);
}

void Settings::update(const std::map<std::string, SettingValue>& updates_map) {
  if (_locked) {
    throw SettingAreLocked();
  }
  set_from_map(updates_map);
}

void Settings::update(const std::map<std::string, std::string>& updates_map) {
  if (_locked) {
    throw SettingAreLocked();
  }
  // First, validate that all keys exist and convert string values
  std::map<std::string, SettingValue> converted_updates;

  for (const auto& [key, str_value] : updates_map) {
    auto it = settings_.find(key);
    if (it == settings_.end()) {
      throw SettingNotFound(key);
    }

    // Convert string to appropriate SettingValue based on current type
    try {
      converted_updates[key] = parse_string_to_setting_value(key, str_value);
    } catch (const std::exception& conversion_exception) {
      throw std::runtime_error("Failed to convert value for key '" + key +
                               "': " + conversion_exception.what());
    }
  }

  // If all conversions succeeded, perform the updates
  for (const auto& [key, value] : converted_updates) {
    settings_[key] = value;
  }
}

void Settings::update(const Settings& other_settings) {
  if (_locked) {
    throw SettingAreLocked();
  }

  // Get all settings from the other object
  const auto& other_map = other_settings.get_all_settings();

  // Use the existing apply method which handles validation and atomicity
  update(other_map);
}

void Settings::set_default(const std::string& key, const SettingValue& value) {
  if (!has(key)) {
    settings_[key] = value;  // Direct assignment for set_default - this is
                             // allowed to create new keys
  }
}

void Settings::set_default(const std::string& key, const char* value) {
  set_default(key, std::string(value));
}

std::string Settings::visit_to_string(const SettingValue& value) const {
  return std::visit(
      [](const auto& variant_value) -> std::string {
        using ValueType = std::decay_t<decltype(variant_value)>;

        if constexpr (std::is_same_v<ValueType, bool>) {
          return variant_value ? "true" : "false";
        } else if constexpr (std::is_same_v<ValueType, std::string>) {
          return variant_value;
        } else if constexpr (std::is_floating_point_v<ValueType>) {
          // Use scientific notation to avoid truncation of small values
          std::ostringstream oss;
          oss << std::scientific << variant_value;
          return oss.str();
        } else if constexpr (std::is_arithmetic_v<ValueType>) {
          return std::to_string(variant_value);
        } else if constexpr (is_vector_v<ValueType>) {
          using ElementType = typename ValueType::value_type;
          std::ostringstream oss;
          oss << "[";
          for (size_t idx = 0; idx < variant_value.size(); ++idx) {
            if (idx > 0) oss << ", ";
            if constexpr (std::is_same_v<ElementType, std::string>) {
              oss << "\"" << variant_value[idx] << "\"";
            } else if constexpr (std::is_same_v<ValueType,
                                                std::vector<double>>) {
              oss << std::scientific << variant_value[idx];
            } else {
              oss << variant_value[idx];
            }
          }
          oss << "]";
          return oss.str();
        } else {
          return "unknown_type";
        }
      },
      value);
}

nlohmann::json Settings::convert_setting_value_to_json(
    const SettingValue& value) const {
  return std::visit(
      [](const auto& variant_value) -> nlohmann::json {
        using ValueType = std::decay_t<decltype(variant_value)>;

        if constexpr (is_vector_v<ValueType>) {
          nlohmann::json json_array = nlohmann::json::array();
          for (const auto& elem : variant_value) {
            json_array.push_back(elem);
          }
          return json_array;
        } else {
          return nlohmann::json(variant_value);
        }
      },
      value);
}

SettingValue Settings::convert_json_to_setting_value(
    const nlohmann::json& json_obj) const {
  if (json_obj.is_boolean()) {
    return json_obj.get<bool>();
  } else if (json_obj.is_number_integer()) {
    return json_obj.get<int64_t>();
  } else if (json_obj.is_number_float()) {
    return json_obj.get<double>();
  } else if (json_obj.is_string()) {
    return json_obj.get<std::string>();
  } else if (json_obj.is_array()) {
    if (json_obj.empty()) {
      return std::vector<int64_t>();  // Default to int64_t vector for empty
                                      // arrays
    }

    // Check the type of the first element to determine vector type
    if (json_obj[0].is_number_integer()) {
      return json_obj.get<std::vector<int64_t>>();
    } else if (json_obj[0].is_number_float()) {
      return json_obj.get<std::vector<double>>();
    } else if (json_obj[0].is_string()) {
      return json_obj.get<std::vector<std::string>>();
    } else {
      throw std::runtime_error("Unsupported array element type in JSON");
    }
  } else {
    throw std::runtime_error("Unsupported JSON type");
  }
}

void Settings::lock() const { _locked = true; }

void Settings::save_setting_value_to_hdf5(H5::Group& group,
                                          const std::string& name,
                                          const SettingValue& value) const {
  std::visit(
      [&group, &name](const auto& variant_value) {
        using ValueType = std::decay_t<decltype(variant_value)>;

        if constexpr (std::is_same_v<ValueType, bool>) {
          // Store booleans as integers
          int bool_val = variant_value ? 1 : 0;
          hsize_t dims[1] = {1};
          H5::DataSpace dataspace(1, dims);
          H5::DataSet dataset =
              group.createDataSet(name, H5::PredType::NATIVE_INT, dataspace);
          dataset.write(&bool_val, H5::PredType::NATIVE_INT);

          // Add attribute to indicate this is a boolean
          const char* type_cstr = "bool";
          H5::StrType str_type(H5::PredType::C_S1,
                               5);  // "bool" + null terminator
          H5::DataSpace attr_space(H5S_SCALAR);
          H5::Attribute attr =
              dataset.createAttribute("type", str_type, attr_space);
          attr.write(str_type, type_cstr);

        } else if constexpr (std::is_same_v<ValueType, int64_t>) {
          hsize_t dims[1] = {1};
          H5::DataSpace dataspace(1, dims);
          H5::DataSet dataset =
              group.createDataSet(name, H5::PredType::NATIVE_INT64, dataspace);
          dataset.write(&variant_value, H5::PredType::NATIVE_INT64);

        } else if constexpr (std::is_same_v<ValueType, uint64_t>) {
          hsize_t dims[1] = {1};
          H5::DataSpace dataspace(1, dims);
          H5::DataSet dataset =
              group.createDataSet(name, H5::PredType::NATIVE_UINT64, dataspace);
          dataset.write(&variant_value, H5::PredType::NATIVE_UINT64);

        } else if constexpr (std::is_same_v<ValueType, float>) {
          hsize_t dims[1] = {1};
          H5::DataSpace dataspace(1, dims);
          H5::DataSet dataset =
              group.createDataSet(name, H5::PredType::NATIVE_FLOAT, dataspace);
          dataset.write(&variant_value, H5::PredType::NATIVE_FLOAT);

        } else if constexpr (std::is_same_v<ValueType, double>) {
          hsize_t dims[1] = {1};
          H5::DataSpace dataspace(1, dims);
          H5::DataSet dataset =
              group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
          dataset.write(&variant_value, H5::PredType::NATIVE_DOUBLE);

        } else if constexpr (std::is_same_v<ValueType, std::string>) {
          H5::StrType string_type(H5::PredType::C_S1,
                                  variant_value.length() + 1);
          H5::DataSpace dataspace(H5S_SCALAR);
          H5::DataSet dataset =
              group.createDataSet(name, string_type, dataspace);
          dataset.write(variant_value.c_str(), string_type);

        } else if constexpr (std::is_same_v<ValueType, std::vector<int64_t>>) {
          if (!variant_value.empty()) {
            hsize_t dims[1] = {variant_value.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = group.createDataSet(
                name, H5::PredType::NATIVE_INT64, dataspace);
            dataset.write(variant_value.data(), H5::PredType::NATIVE_INT64);
          } else {
            // Handle empty vector
            hsize_t dims[1] = {0};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = group.createDataSet(
                name, H5::PredType::NATIVE_INT64, dataspace);
          }

        } else if constexpr (std::is_same_v<ValueType, std::vector<uint64_t>>) {
          if (!variant_value.empty()) {
            hsize_t dims[1] = {variant_value.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = group.createDataSet(
                name, H5::PredType::NATIVE_UINT64, dataspace);
            dataset.write(variant_value.data(), H5::PredType::NATIVE_UINT64);
          } else {
            // Handle empty vector
            hsize_t dims[1] = {0};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = group.createDataSet(
                name, H5::PredType::NATIVE_UINT64, dataspace);
          }

        } else if constexpr (std::is_same_v<ValueType, std::vector<double>>) {
          if (!variant_value.empty()) {
            hsize_t dims[1] = {variant_value.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = group.createDataSet(
                name, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(variant_value.data(), H5::PredType::NATIVE_DOUBLE);
          } else {
            hsize_t dims[1] = {0};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = group.createDataSet(
                name, H5::PredType::NATIVE_DOUBLE, dataspace);
          }

        } else if constexpr (std::is_same_v<ValueType,
                                            std::vector<std::string>>) {
          if (!variant_value.empty()) {
            // Create variable-length string type
            H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
            hsize_t dims[1] = {variant_value.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset =
                group.createDataSet(name, string_type, dataspace);

            // Convert to C-style strings
            std::vector<const char*> c_strings;
            for (const auto& str : variant_value) {
              c_strings.push_back(str.c_str());
            }
            dataset.write(c_strings.data(), string_type);
          } else {
            H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
            hsize_t dims[1] = {0};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset =
                group.createDataSet(name, string_type, dataspace);
          }
        }
      },
      value);
}

SettingValue Settings::parse_string_to_setting_value(
    const std::string& key, const std::string& str_value) const {
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    throw SettingNotFound(key);
  }

  const SettingValue& current_value = it->second;

  return std::visit(
      [&str_value, &key,
       this](const auto& current_variant_value) -> SettingValue {
        using CurrentType = std::decay_t<decltype(current_variant_value)>;

        if constexpr (std::is_same_v<CurrentType, bool>) {
          // Handle boolean special cases (case-insensitive)
          std::string lower_str = str_value;
          std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                         ::tolower);

          if (lower_str == "true" || lower_str == "1" || lower_str == "yes" ||
              lower_str == "on") {
            return true;
          } else if (lower_str == "false" || lower_str == "0" ||
                     lower_str == "no" || lower_str == "off") {
            return false;
          } else {
            throw std::runtime_error("Invalid boolean value: '" + str_value +
                                     "'");
          }

        } else if constexpr (std::is_same_v<CurrentType, std::string>) {
          // String is used as-is (no JSON parsing needed)
          return str_value;

        } else {
          // For all other types (numbers and vectors), use JSON parsing
          try {
            nlohmann::json json_obj;

            if constexpr (is_vector_v<CurrentType>) {
              // Vectors must be JSON arrays
              json_obj = nlohmann::json::parse(str_value);
              if (!json_obj.is_array()) {
                throw std::runtime_error(
                    "Vector value must be a JSON array: '" + str_value + "'");
              }
            } else {
              // For numeric types, try to parse as JSON
              // If it's a plain number, this will work directly
              json_obj = nlohmann::json::parse(str_value);
            }

            // Use existing conversion logic
            SettingValue converted = convert_json_to_setting_value(json_obj);

            // Verify the type matches what we expect
            if (!std::holds_alternative<CurrentType>(converted)) {
              throw std::runtime_error("Type mismatch: expected " +
                                       get_type_name(key) + " for key '" + key +
                                       "'");
            }

            return converted;

          } catch (const nlohmann::json::exception& json_exception) {
            throw std::runtime_error("Invalid value format: '" + str_value +
                                     "' - " + json_exception.what());
          }
        }
      },
      current_value);
}

void Settings::to_file(const std::string& filename,
                       const std::string& type) const {
  if (type == "json") {
    _to_json_file(filename);
  } else if (type == "hdf5") {
    _to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

void Settings::to_json_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "settings");

  _to_json_file(validated_filename);
}

std::shared_ptr<Settings> Settings::from_json_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "settings");

  return _from_json_file(validated_filename);
}

void Settings::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  auto json_obj = to_json();
  file << json_obj.dump(2);  // Pretty print with 2-space indentation

  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

std::shared_ptr<Settings> Settings::_from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Settings JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json json_obj;
  file >> json_obj;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(json_obj);
}

void Settings::to_hdf5_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "settings");

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<Settings> Settings::from_hdf5_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "settings");

  return _from_hdf5_file(validated_filename);
}

void Settings::_to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Add version attribute to the root file
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr =
        file.createAttribute("version", string_type, scalar_space);
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);
    version_attr.close();

    H5::Group settings_group = file.createGroup("/settings");

    // Use to_hdf5 to write the settings data with proper version attribute
    to_hdf5(settings_group);

  } catch (const H5::Exception& hdf5_exception) {
    throw std::runtime_error("HDF5 error: " +
                             std::string(hdf5_exception.getCDetailMsg()));
  }
}

std::shared_ptr<Settings> Settings::_from_hdf5_file(
    const std::string& filename) {
  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Settings HDF5 file '" + filename +
                             "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = file.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    if (!Settings::group_exists(file, "/settings")) {
      throw std::runtime_error("Settings group not found in HDF5 file");
    }

    H5::Group settings_group = file.openGroup("/settings");
    return from_hdf5(settings_group);
  } catch (const H5::Exception& hdf5_exception) {
    throw std::runtime_error(
        "Unable to read Settings data from HDF5 file '" + filename + "'. " +
        "HDF5 error: " + std::string(hdf5_exception.getCDetailMsg()));
  }
}

std::shared_ptr<Settings> Settings::from_file(const std::string& filename,
                                              const std::string& type) {
  if (type == "json") {
    return _from_json_file(filename);
  } else if (type == "hdf5") {
    return _from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

bool Settings::group_exists(H5::H5File& file, const std::string& group_name) {
  try {
    H5::Group group = file.openGroup(group_name);
    return true;
  } catch (const H5::Exception&) {
    return false;
  }
}

SettingValue Settings::load_setting_value_from_hdf5(H5::Group& group,
                                                    const std::string& name) {
  H5::DataSet dataset = group.openDataSet(name);
  H5::DataSpace dataspace = dataset.getSpace();
  H5::DataType datatype = dataset.getDataType();

  int ndims = dataspace.getSimpleExtentNdims();
  std::vector<hsize_t> dims(ndims);
  dataspace.getSimpleExtentDims(dims.data(), nullptr);

  // Check if this is a boolean or vector<bool> (stored as int with type
  // attribute)
  if (dataset.attrExists("type")) {
    try {
      H5::Attribute attr = dataset.openAttribute("type");
      H5::StrType attr_type = attr.getStrType();
      size_t attr_size = attr_type.getSize();

      // Create buffer for reading attribute
      std::vector<char> buffer(attr_size + 1, '\0');
      attr.read(attr_type, buffer.data());

      // Convert to string and trim nulls
      std::string type_str(buffer.data());

      if (type_str == "bool") {
        int bool_val;
        dataset.read(&bool_val, H5::PredType::NATIVE_INT);
        return static_cast<bool>(bool_val);
      }
    } catch (const H5::Exception& hdf5_exception) {
      std::cerr << "Warning: Failed to read attribute 'type' for dataset '"
                << name << "': " << hdf5_exception.getCDetailMsg() << std::endl;
      // Continue with regular type detection
    }
  }

  // Determine type based on HDF5 datatype
  H5T_class_t type_class = datatype.getClass();

  if (type_class == H5T_INTEGER) {
    size_t type_size = datatype.getSize();
    H5::IntType int_type = dataset.getIntType();
    bool is_signed = (int_type.getSign() == H5T_SGN_2);

    if (ndims == 0 || (ndims == 1 && dims[0] == 1)) {
      // Scalar value - map to int64_t or uint64_t based on signedness
      if (is_signed) {
        int64_t int64_value;
        dataset.read(&int64_value, H5::PredType::NATIVE_INT64);
        return int64_value;
      } else {
        uint64_t uint64_value;
        dataset.read(&uint64_value, H5::PredType::NATIVE_UINT64);
        return uint64_value;
      }
    } else {
      // Vector of integers - return signed or unsigned vector based on
      // signedness
      if (is_signed) {
        std::vector<int64_t> int_vector(dims[0]);
        if (dims[0] > 0) {
          dataset.read(int_vector.data(), H5::PredType::NATIVE_INT64);
        }
        return int_vector;
      } else {
        std::vector<uint64_t> uint_vector(dims[0]);
        if (dims[0] > 0) {
          dataset.read(uint_vector.data(), H5::PredType::NATIVE_UINT64);
        }
        return uint_vector;
      }
    }

  } else if (type_class == H5T_FLOAT) {
    if (ndims == 0 || (ndims == 1 && dims[0] == 1)) {
      // Scalar value
      size_t type_size = datatype.getSize();
      if (type_size == sizeof(float)) {
        float float_value;
        dataset.read(&float_value, H5::PredType::NATIVE_FLOAT);
        return float_value;
      } else {
        double double_value;
        dataset.read(&double_value, H5::PredType::NATIVE_DOUBLE);
        return double_value;
      }
    } else {
      // Vector of doubles
      std::vector<double> double_vector(dims[0]);
      if (dims[0] > 0) {
        dataset.read(double_vector.data(), H5::PredType::NATIVE_DOUBLE);
      }
      return double_vector;
    }

  } else if (type_class == H5T_STRING) {
    if (ndims == 0) {
      // Scalar string
      if (datatype.isVariableStr()) {
        char* str_data;
        dataset.read(&str_data, datatype);
        std::string result(str_data);
        free(str_data);
        return result;
      } else {
        size_t str_len = datatype.getSize();
        std::string str_data(str_len, '\0');
        dataset.read(&str_data[0], datatype);

        // Remove null terminator if present
        size_t null_pos = str_data.find('\0');
        if (null_pos != std::string::npos) {
          str_data.resize(null_pos);
        }
        return str_data;
      }
    } else {
      // Vector of strings
      std::vector<std::string> string_vector;
      if (dims[0] > 0) {
        if (datatype.isVariableStr()) {
          std::vector<char*> str_data(dims[0]);
          dataset.read(str_data.data(), datatype);

          for (size_t string_index = 0; string_index < dims[0];
               ++string_index) {
            string_vector.emplace_back(str_data[string_index]);
            free(str_data[string_index]);
          }
        } else {
          size_t str_len = datatype.getSize();
          std::vector<char> buffer(dims[0] * str_len);
          dataset.read(buffer.data(), datatype);

          for (size_t string_index = 0; string_index < dims[0];
               ++string_index) {
            const char* str_start = buffer.data() + string_index * str_len;
            std::string str(str_start, str_len);

            // Remove null terminator if present
            size_t null_pos = str.find('\0');
            if (null_pos != std::string::npos) {
              str.resize(null_pos);
            }
            string_vector.push_back(str);
          }
        }
      }
      return string_vector;
    }
  }

  throw std::runtime_error("Unsupported HDF5 datatype for setting: " + name);
}

void Settings::to_hdf5(H5::Group& group) const {
  // Add version attribute to the group
  H5::DataSpace scalar_space(H5S_SCALAR);
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute version_attr =
      group.createAttribute("version", string_type, scalar_space);
  std::string version_str(SERIALIZATION_VERSION);
  version_attr.write(string_type, version_str);
  version_attr.close();

  // Save all settings to the group
  for (const auto& [key, value] : settings_) {
    save_setting_value_to_hdf5(group, key, value);
  }
}

std::shared_ptr<Settings> Settings::from_hdf5(H5::Group& group) {
  auto settings = std::make_shared<Settings>();

  try {
    H5::Attribute version_attr = group.openAttribute("version");
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);
  } catch (const H5::AttributeIException&) {
    // Version attribute not found, continue without validation for backward
    // compatibility
  }

  hsize_t num_objs = group.getNumObjs();

  for (hsize_t i = 0; i < num_objs; ++i) {
    std::string obj_name = group.getObjnameByIdx(i);
    H5G_obj_t obj_type = group.getObjTypeByIdx(i);

    if (obj_type == H5G_DATASET) {
      try {
        SettingValue value =
            Settings::load_setting_value_from_hdf5(group, obj_name);
        // For base Settings class, we create settings dynamically
        settings->settings_[obj_name] = value;
      } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load setting '" + obj_name +
                                 "': " + e.what());
      }
    }
  }

  return settings;
}

}  // namespace qdk::chemistry::data
