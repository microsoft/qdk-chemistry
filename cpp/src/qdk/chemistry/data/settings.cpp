// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <qdk/chemistry/data/settings.hpp>
#include <sstream>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

void Settings::set(const std::string& key, const SettingValue& value) {
  if (_locked) {
    throw SettingsAreLocked();
  }
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    throw SettingNotFound(key);
  }

  // Check if types match
  if (value.index() != settings_[key].index()) {
    throw SettingTypeMismatch(key, "does not match type of given argument");
  }

  // If limits exist, validate against them
  if (limits_.find(key) != limits_.end()) {
    if (std::holds_alternative<std::string>(value)) {
      if (std::holds_alternative<ListConstraint<std::string>>(limits_[key])) {
        auto options = std::get<ListConstraint<std::string>>(limits_[key]);
        if (std::find(
                options.allowed_values.begin(), options.allowed_values.end(),
                std::get<std::string>(value)) == options.allowed_values.end()) {
          std::string options_str = "[";
          for (size_t i = 0; i < options.allowed_values.size(); ++i) {
            if (i > 0) options_str += ", ";
            options_str += "\"" + options.allowed_values[i] + "\"";
          }
          options_str += "]";
          throw std::invalid_argument(
              "Value for setting '" + key +
              "' is out of allowed options. Allowed options: " + options_str);
        }
      }
    } else if (std::holds_alternative<int64_t>(value)) {
      if (std::holds_alternative<ListConstraint<int64_t>>(limits_[key])) {
        auto options = std::get<ListConstraint<int64_t>>(limits_[key]);
        if (std::find(options.allowed_values.begin(),
                      options.allowed_values.end(), std::get<int64_t>(value)) ==
            options.allowed_values.end()) {
          std::string options_str = "[";
          for (size_t i = 0; i < options.allowed_values.size(); ++i) {
            if (i > 0) options_str += ", ";
            options_str +=
                "\"" + std::to_string(options.allowed_values[i]) + "\"";
          }
          options_str += "]";
          throw std::invalid_argument(
              "Value for setting '" + key +
              "' is out of allowed options. Allowed options: " + options_str);
        }
      } else if (std::holds_alternative<BoundConstraint<int64_t>>(
                     limits_[key])) {
        auto options = std::get<BoundConstraint<int64_t>>(limits_[key]);
        if (options.min > std::get<int64_t>(value) ||
            std::get<int64_t>(value) > options.max) {
          std::string options_str = "[" + std::to_string(options.min) + ", " +
                                    std::to_string(options.max) + "]";
          throw std::invalid_argument(
              "Value for setting '" + key +
              "' is out of allowed range. Allowed range: " + options_str);
        }
      }
    } else if (std::holds_alternative<double>(value)) {
      if (std::holds_alternative<BoundConstraint<double>>(limits_[key])) {
        auto options = std::get<BoundConstraint<double>>(limits_[key]);
        if (options.min > std::get<double>(value) ||
            std::get<double>(value) > options.max) {
          std::string options_str = "[" + std::to_string(options.min) + ", " +
                                    std::to_string(options.max) + "]";
          throw std::invalid_argument(
              "Value for setting '" + key +
              "' is out of allowed range. Allowed range: " + options_str);
        }
      }
    } else if (std::holds_alternative<std::vector<std::string>>(value)) {
      if (std::holds_alternative<ListConstraint<std::string>>(limits_[key])) {
        auto options = std::get<ListConstraint<std::string>>(limits_[key]);
        for (auto& test_value : std::get<std::vector<std::string>>(value)) {
          if (std::find(options.allowed_values.begin(),
                        options.allowed_values.end(),
                        test_value) == options.allowed_values.end()) {
            std::string options_str = "[";
            for (size_t i = 0; i < options.allowed_values.size(); ++i) {
              if (i > 0) options_str += ", ";
              options_str += "\"" + options.allowed_values[i] + "\"";
            }
            options_str += "]";
            throw std::invalid_argument(
                "Value for setting '" + key +
                "' is out of allowed options. Allowed options: " + options_str);
          }
        }
      }
    } else if (std::holds_alternative<std::vector<int64_t>>(value)) {
      if (std::holds_alternative<ListConstraint<int64_t>>(limits_[key])) {
        auto options = std::get<ListConstraint<int64_t>>(limits_[key]);
        for (auto& test_value : std::get<std::vector<int64_t>>(value)) {
          if (std::find(options.allowed_values.begin(),
                        options.allowed_values.end(),
                        test_value) == options.allowed_values.end()) {
            std::string options_str = "[";
            for (size_t i = 0; i < options.allowed_values.size(); ++i) {
              if (i > 0) options_str += ", ";
              options_str +=
                  "\"" + std::to_string(options.allowed_values[i]) + "\"";
            }
            options_str += "]";
            throw std::invalid_argument(
                "Value for setting '" + key +
                "' is out of allowed options. Allowed options: " + options_str);
          }
        }
      } else if (std::holds_alternative<BoundConstraint<int64_t>>(
                     limits_[key])) {
        auto options = std::get<BoundConstraint<int64_t>>(limits_[key]);
        for (auto& test_value : std::get<std::vector<int64_t>>(value)) {
          if (options.min > test_value || test_value > options.max) {
            std::string options_str = "[" + std::to_string(options.min) + ", " +
                                      std::to_string(options.max) + "]";
            throw std::invalid_argument(
                "Value for setting '" + key +
                "' is out of allowed range. Allowed range: " + options_str);
          }
        }
      }
    } else if (std::holds_alternative<std::vector<double>>(value)) {
      if (std::holds_alternative<BoundConstraint<double>>(limits_[key])) {
        auto options = std::get<BoundConstraint<double>>(limits_[key]);
        for (auto& test_value : std::get<std::vector<double>>(value)) {
          if (options.min > test_value || test_value > options.max) {
            std::string options_str = "[" + std::to_string(options.min) + ", " +
                                      std::to_string(options.max) + "]";
            throw std::invalid_argument(
                "Value for setting '" + key +
                "' is out of allowed range. Allowed range: " + options_str);
          }
        }
      }
    }
  }

  // Set the value (types already validated)
  settings_[key] = value;
}

void Settings::set(const std::string& key, const char* value) {
  if (_locked) {
    throw SettingsAreLocked();
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

nlohmann::json Settings::to_json() const {
  nlohmann::json json_obj;

  // Store version first
  json_obj["version"] = SERIALIZATION_VERSION;

  for (const auto& [key, value] : settings_) {
    json_obj[key] = convert_setting_value_to_json(value);
  }

  // Store metadata maps
  if (!descriptions_.empty()) {
    json_obj["_descriptions"] = descriptions_;
  }
  if (!limits_.empty()) {
    nlohmann::json limits_json;
    for (const auto& [key, limit_value] : limits_) {
      std::visit(
          [&limits_json, &key](const auto& variant_value) {
            using LimitType = std::decay_t<decltype(variant_value)>;
            if constexpr (std::is_same_v<LimitType, BoundConstraint<int64_t>>) {
              limits_json[key] =
                  nlohmann::json::array({variant_value.min, variant_value.max});
            } else if constexpr (std::is_same_v<LimitType,
                                                BoundConstraint<double>>) {
              limits_json[key] =
                  nlohmann::json::array({variant_value.min, variant_value.max});
            } else if constexpr (std::is_same_v<LimitType,
                                                ListConstraint<int64_t>>) {
              limits_json[key] = variant_value.allowed_values;
            } else if constexpr (std::is_same_v<LimitType,
                                                ListConstraint<std::string>>) {
              limits_json[key] = variant_value.allowed_values;
            }
          },
          limit_value);
    }
    json_obj["_limits"] = limits_json;
  }
  if (!documented_.empty()) {
    json_obj["_documented"] = documented_;
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
    // Skip the version and metadata fields during processing
    if (key == "version" || key == "_descriptions" || key == "_limits" ||
        key == "_documented") {
      continue;
    }
    settings->settings_[key] = settings->convert_json_to_setting_value(value);
  }

  // Load metadata maps
  if (json_obj.contains("_descriptions")) {
    settings->descriptions_ =
        json_obj["_descriptions"].get<std::map<std::string, std::string>>();
  }
  if (json_obj.contains("_limits")) {
    const auto& limits_json = json_obj["_limits"];
    for (const auto& [key, limit_json] : limits_json.items()) {
      if (limit_json.is_array() && limit_json.size() == 2) {
        // Could be a pair of integers or doubles
        if (limit_json[0].is_number_integer()) {
          settings->limits_[key] = data::BoundConstraint<int64_t>{
              limit_json[0].get<int64_t>(), limit_json[1].get<int64_t>()};
        } else if (limit_json[0].is_number_float()) {
          settings->limits_[key] = data::BoundConstraint<double>{
              limit_json[0].get<double>(), limit_json[1].get<double>()};
        }
      } else if (limit_json.is_array()) {
        // Could be a vector of integers or strings
        if (!limit_json.empty() && limit_json[0].is_number_integer()) {
          settings->limits_[key] = data::ListConstraint<int64_t>{
              limit_json.get<std::vector<int64_t>>()};
        } else if (!limit_json.empty() && limit_json[0].is_string()) {
          settings->limits_[key] = data::ListConstraint<std::string>{
              limit_json.get<std::vector<std::string>>()};
        }
      }
    }
  }
  if (json_obj.contains("_documented")) {
    settings->documented_ =
        json_obj["_documented"].get<std::map<std::string, bool>>();
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

bool Settings::has_description(const std::string& key) const {
  return descriptions_.find(key) != descriptions_.end();
}

std::string Settings::get_description(const std::string& key) const {
  auto it = descriptions_.find(key);
  if (it == descriptions_.end()) {
    throw SettingNotFound("No description found for setting: " + key);
  }
  return it->second;
}

bool Settings::has_limits(const std::string& key) const {
  return limits_.find(key) != limits_.end();
}

Constraint Settings::get_limits(const std::string& key) const {
  auto it = limits_.find(key);
  if (it == limits_.end()) {
    throw SettingNotFound("No limits found for setting: " + key);
  }
  return it->second;
}

bool Settings::is_documented(const std::string& key) const {
  if (!has(key)) {
    throw SettingNotFound(key);
  }
  auto it = documented_.find(key);
  if (it == documented_.end()) {
    // If not explicitly set, default to true for backward compatibility
    return true;
  }
  return it->second;
}

std::string Settings::as_table(size_t max_width, bool show_undocumented) const {
  // Helper to wrap text to fit within a column width
  auto wrap_text = [](const std::string& text,
                      size_t width) -> std::vector<std::string> {
    if (text.empty()) return {""};

    std::vector<std::string> lines;
    size_t start = 0;

    while (start < text.length()) {
      if (text.length() - start <= width) {
        lines.push_back(text.substr(start));
        break;
      }

      // Find last space within width
      size_t end = start + width;
      size_t last_space = text.rfind(' ', end);

      if (last_space != std::string::npos && last_space > start) {
        lines.push_back(text.substr(start, last_space - start));
        start = last_space + 1;
      } else {
        // No space found, hard break
        lines.push_back(text.substr(start, width));
        start += width;
      }
    }

    if (lines.empty()) lines.push_back("");
    return lines;
  };

  // Helper to format value with scientific notation for floats
  auto format_value = [](const SettingValue& value) -> std::string {
    return std::visit(
        [](const auto& variant_value) -> std::string {
          using ValueType = std::decay_t<decltype(variant_value)>;

          if constexpr (std::is_same_v<ValueType, bool>) {
            return variant_value ? "true" : "false";
          } else if constexpr (std::is_same_v<ValueType, std::string>) {
            return "\"" + variant_value + "\"";
          } else if constexpr (std::is_same_v<ValueType, float> ||
                               std::is_same_v<ValueType, double>) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(2);
            oss << variant_value;
            return oss.str();
          } else if constexpr (std::is_arithmetic_v<ValueType>) {
            return std::to_string(variant_value);
          } else if constexpr (std::is_same_v<ValueType,
                                              std::vector<int64_t>> ||
                               std::is_same_v<ValueType,
                                              std::vector<uint64_t>>) {
            if (variant_value.empty()) return "[]";
            std::string result = "[";
            for (size_t i = 0; i < variant_value.size() && i < 3; ++i) {
              if (i > 0) result += ", ";
              result += std::to_string(variant_value[i]);
            }
            if (variant_value.size() > 3) result += "...";
            result += "]";
            return result;
          } else if constexpr (std::is_same_v<ValueType, std::vector<double>>) {
            if (variant_value.empty()) return "[]";
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(2);
            oss << "[";
            for (size_t i = 0; i < variant_value.size() && i < 3; ++i) {
              if (i > 0) oss << ", ";
              oss << variant_value[i];
            }
            if (variant_value.size() > 3) oss << "...";
            oss << "]";
            return oss.str();
          } else if constexpr (std::is_same_v<ValueType,
                                              std::vector<std::string>>) {
            if (variant_value.empty()) return "[]";
            std::string result = "[";
            for (size_t i = 0; i < variant_value.size() && i < 3; ++i) {
              if (i > 0) result += ", ";
              result += "\"" + variant_value[i] + "\"";
            }
            if (variant_value.size() > 3) result += "...";
            result += "]";
            return result;
          } else {
            return "?";
          }
        },
        value);
  };

  // Column widths (will be adjusted)
  size_t key_width = 25;
  size_t value_width = 15;
  size_t limits_width = 20;
  size_t desc_width = max_width - key_width - value_width - limits_width -
                      10;  // 10 for separators and padding

  // Ensure minimum widths
  if (desc_width < 20) {
    desc_width = 20;
    key_width = 15;
    value_width = 12;
    limits_width = 15;
  }

  // Helper to format limit value as multi-line if needed
  auto format_limits_multiline =
      [limits_width](const Constraint& limit) -> std::vector<std::string> {
    return std::visit(
        [limits_width](const auto& variant_value) -> std::vector<std::string> {
          using LimitType = std::decay_t<decltype(variant_value)>;
          if constexpr (std::is_same_v<LimitType, BoundConstraint<int64_t>>) {
            std::string single_line =
                std::to_string(variant_value.min) +
                " <= x <= " + std::to_string(variant_value.max);
            if (single_line.length() <= limits_width) {
              return {single_line};
            } else {
              return {std::to_string(variant_value.min) + " <= x",
                      "x <= " + std::to_string(variant_value.max)};
            }
          } else if constexpr (std::is_same_v<LimitType,
                                              BoundConstraint<double>>) {
            std::ostringstream oss1, oss2, oss_combined;
            oss1 << std::scientific << std::setprecision(2);
            oss2 << std::scientific << std::setprecision(2);
            oss_combined << std::scientific << std::setprecision(2);

            oss_combined << variant_value.min
                         << " <= x <= " << variant_value.max;
            std::string single_line = oss_combined.str();

            if (single_line.length() <= limits_width) {
              return {single_line};
            } else {
              oss1 << variant_value.min << " <= x";
              oss2 << "x <= " << variant_value.max;
              return {oss1.str(), oss2.str()};
            }
          } else if constexpr (std::is_same_v<LimitType,
                                              ListConstraint<int64_t>>) {
            if (variant_value.allowed_values.empty()) return {"[]"};
            std::vector<std::string> lines;
            std::string current_line = "[";

            for (size_t i = 0; i < variant_value.allowed_values.size(); ++i) {
              std::string element =
                  std::to_string(variant_value.allowed_values[i]);
              std::string separator =
                  (i < variant_value.allowed_values.size() - 1) ? ", " : "]";

              // Check if adding this element would exceed width
              if (!current_line.empty() && current_line != "[" &&
                  (current_line + element + separator).length() >
                      limits_width) {
                lines.push_back(current_line);
                current_line = " ";
              }
              current_line += element + separator;
            }
            if (!current_line.empty()) lines.push_back(current_line);
            return lines;
          } else if constexpr (std::is_same_v<LimitType,
                                              ListConstraint<std::string>>) {
            if (variant_value.allowed_values.empty()) return {"[]"};
            std::vector<std::string> lines;
            std::string current_line = "[";

            for (size_t i = 0; i < variant_value.allowed_values.size(); ++i) {
              std::string element =
                  "\"" + variant_value.allowed_values[i] + "\"";
              std::string separator =
                  (i < variant_value.allowed_values.size() - 1) ? ", " : "]";

              // Check if adding this element would exceed width
              if (!current_line.empty() && current_line != "[" &&
                  (current_line + element + separator).length() >
                      limits_width) {
                lines.push_back(current_line);
                current_line = " ";
              }
              current_line += element + separator;
            }
            if (!current_line.empty()) lines.push_back(current_line);
            return lines;
          } else {
            return {""};
          }
        },
        limit);
  };

  std::ostringstream oss;

  // Header
  std::string separator(max_width, '-');
  oss << separator << "\n";
  oss << std::left << std::setw(key_width) << "Key" << " | "
      << std::setw(value_width) << "Value" << " | " << std::setw(limits_width)
      << "Allowed" << " | "
      << "Description\n";
  oss << separator << "\n";

  // Collect and sort keys
  std::vector<std::string> keys_to_show;
  for (const auto& key : keys()) {
    bool is_doc = true;
    try {
      is_doc = is_documented(key);
    } catch (...) {
      is_doc = true;  // Default to showing if can't determine
    }

    if (is_doc || show_undocumented) {
      keys_to_show.push_back(key);
    }
  }
  std::sort(keys_to_show.begin(), keys_to_show.end());

  // Print each setting
  for (const auto& key : keys_to_show) {
    std::string value_str = format_value(settings_.at(key));

    std::vector<std::string> limits_lines = {"-"};
    if (has_limits(key)) {
      try {
        limits_lines = format_limits_multiline(get_limits(key));
      } catch (...) {
        limits_lines = {"-"};
      }
    }

    std::string desc_str = "-";
    if (has_description(key)) {
      try {
        desc_str = get_description(key);
      } catch (...) {
        desc_str = "-";
      }
    }

    // Wrap description
    auto desc_lines = wrap_text(desc_str, desc_width);

    // Determine max lines needed
    size_t max_lines =
        std::max({limits_lines.size(), desc_lines.size(), size_t(1)});

    // Print first line with all columns
    oss << std::left << std::setw(key_width)
        << (key.length() > key_width ? key.substr(0, key_width - 3) + "..."
                                     : key)
        << " | " << std::setw(value_width)
        << (value_str.length() > value_width
                ? value_str.substr(0, value_width - 3) + "..."
                : value_str)
        << " | " << std::setw(limits_width)
        << (limits_lines.empty() ? "" : limits_lines[0]) << " | "
        << (desc_lines.empty() ? "" : desc_lines[0]) << "\n";

    // Print remaining lines
    for (size_t i = 1; i < max_lines; ++i) {
      oss << std::string(key_width, ' ') << " | "
          << std::string(value_width, ' ') << " | " << std::setw(limits_width)
          << (i < limits_lines.size() ? limits_lines[i] : "") << " | "
          << (i < desc_lines.size() ? desc_lines[i] : "") << "\n";
    }
  }

  oss << separator << "\n";

  return oss.str();
}

void Settings::update(const std::string& key, const SettingValue& value) {
  if (_locked) {
    throw SettingsAreLocked();
  }
  if (!has(key)) {
    throw SettingNotFound(key);
  }
  set(key, value);
}

void Settings::update(const std::map<std::string, SettingValue>& updates_map) {
  if (_locked) {
    throw SettingsAreLocked();
  }
  for (const auto& [key, value] : updates_map) {
    this->set(key, value);
  }
}

void Settings::update(const std::map<std::string, std::string>& updates_map) {
  if (_locked) {
    throw SettingsAreLocked();
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
    throw SettingsAreLocked();
  }

  // Get all settings from the other object
  const auto& other_map = other_settings.get_all_settings();

  // Use the existing apply method which handles validation and atomicity
  update(other_map);
}

void Settings::set_default(const std::string& key, const SettingValue& value,
                           std::optional<std::string> description,
                           std::optional<Constraint> limit, bool documented) {
  if (!has(key)) {
    settings_[key] = value;  // Direct assignment for set_default - this is
                             // allowed to create new keys
    if (description.has_value()) {
      descriptions_[key] = *description;
    }
    if (std::holds_alternative<bool>(value) && limit.has_value()) {
      throw std::invalid_argument("Limit cannot be set for boolean settings");
    }
    if ((std::holds_alternative<std::string>(value) ||
         std::holds_alternative<std::vector<std::string>>(value)) &&
        limit.has_value()) {
      if (!(std::holds_alternative<ListConstraint<std::string>>(*limit))) {
        throw std::invalid_argument(
            "Type of settings values and limits must match. Value type of "
            "std::string/str, or std::vector<std::string>/list[str]"
            "must have type ListConstraint<std::string>/list[str] options.");
      }
    }
    if ((std::holds_alternative<double>(value) ||
         std::holds_alternative<std::vector<double>>(value)) &&
        limit.has_value()) {
      if (!(std::holds_alternative<BoundConstraint<double>>(*limit))) {
        throw std::invalid_argument(
            "Type of settings values and limits must match. Value type of "
            "double/float, std::vector<double>/list[float]"
            "must have BoundConstraint<double> limits.");
      }
    }
    if ((std::holds_alternative<int64_t>(value) ||
         std::holds_alternative<std::vector<int64_t>>(value)) &&
        limit.has_value()) {
      if (!(std::holds_alternative<ListConstraint<int64_t>>(*limit) ||
            std::holds_alternative<BoundConstraint<int64_t>>(*limit))) {
        throw std::invalid_argument(
            "Type of settings values and limits must match. Value type of "
            "int64_t/int, std::vector<int64_t>/list[int]"
            "must have ListConstraint<int64_t>/list[int] options or "
            "BoundConstraint<int64_t> limits.");
      }
    }
    if (limit.has_value()) {
      limits_[key] = *limit;
    }
    documented_[key] = documented;
  }
}

void Settings::set_default(const std::string& key, const char* value,
                           std::optional<const char*> description,
                           std::optional<std::vector<const char*>> limit,
                           bool documented) {
  std::optional<std::string> desc_str;
  if (description.has_value()) {
    desc_str = std::string(*description);
  }
  std::optional<ListConstraint<std::string>> limit_strs;
  if (limit.has_value()) {
    limit_strs = ListConstraint<std::string>{
        {std::vector<std::string>(limit->begin(), limit->end())}};
  }
  set_default(key, std::string(value), desc_str, limit_strs, documented);
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
          // For empty arrays, add type metadata to preserve type information
          if (json_array.empty()) {
            nlohmann::json typed_array = nlohmann::json::object();
            typed_array["__type__"] = "array";
            if constexpr (std::is_same_v<ValueType, std::vector<int64_t>>) {
              typed_array["__element_type__"] = "int64";
            } else if constexpr (std::is_same_v<ValueType,
                                                std::vector<double>>) {
              typed_array["__element_type__"] = "double";
            } else if constexpr (std::is_same_v<ValueType,
                                                std::vector<std::string>>) {
              typed_array["__element_type__"] = "string";
            }
            typed_array["__value__"] = json_array;
            return typed_array;
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
  } else if (json_obj.is_object() && json_obj.contains("__type__") &&
             json_obj["__type__"] == "array") {
    // Handle typed empty arrays
    std::string elem_type = json_obj["__element_type__"];
    if (elem_type == "int64") {
      return std::vector<int64_t>();
    } else if (elem_type == "double") {
      return std::vector<double>();
    } else if (elem_type == "string") {
      return std::vector<std::string>();
    } else {
      throw std::runtime_error("Unsupported typed array element type: " +
                               elem_type);
    }
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
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

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
      int64_t int64_value;
      dataset.read(&int64_value, H5::PredType::NATIVE_INT64);
      return int64_value;
    } else {
      std::vector<int64_t> int_vector(dims[0]);
      if (dims[0] > 0) {
        dataset.read(int_vector.data(), H5::PredType::NATIVE_INT64);
      }
      return int_vector;
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

  // Save metadata as subgroups
  if (!descriptions_.empty()) {
    H5::Group desc_group = group.createGroup("_descriptions");
    for (const auto& [key, desc] : descriptions_) {
      H5::StrType str_type(H5::PredType::C_S1, desc.length() + 1);
      H5::DataSpace dataspace(H5S_SCALAR);
      H5::DataSet dataset = desc_group.createDataSet(key, str_type, dataspace);
      dataset.write(desc.c_str(), str_type);
    }
  }

  if (!limits_.empty()) {
    H5::Group limits_group = group.createGroup("_limits");
    for (const auto& [key, limit_value] : limits_) {
      std::visit(
          [&limits_group, &key](const auto& variant_value) {
            using LimitType = std::decay_t<decltype(variant_value)>;
            if constexpr (std::is_same_v<LimitType,
                                         std::pair<int64_t, int64_t>>) {
              int64_t pair_data[2] = {variant_value.first,
                                      variant_value.second};
              hsize_t dims[1] = {2};
              H5::DataSpace dataspace(1, dims);
              H5::DataSet dataset = limits_group.createDataSet(
                  key, H5::PredType::NATIVE_INT64, dataspace);
              dataset.write(pair_data, H5::PredType::NATIVE_INT64);
            } else if constexpr (std::is_same_v<LimitType,
                                                std::pair<double, double>>) {
              double pair_data[2] = {variant_value.first, variant_value.second};
              hsize_t dims[1] = {2};
              H5::DataSpace dataspace(1, dims);
              H5::DataSet dataset = limits_group.createDataSet(
                  key, H5::PredType::NATIVE_DOUBLE, dataspace);
              dataset.write(pair_data, H5::PredType::NATIVE_DOUBLE);
            } else if constexpr (std::is_same_v<LimitType,
                                                std::vector<int64_t>>) {
              if (!variant_value.empty()) {
                hsize_t dims[1] = {variant_value.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = limits_group.createDataSet(
                    key, H5::PredType::NATIVE_INT64, dataspace);
                dataset.write(variant_value.data(), H5::PredType::NATIVE_INT64);
              } else {
                hsize_t dims[1] = {0};
                H5::DataSpace dataspace(1, dims);
                limits_group.createDataSet(key, H5::PredType::NATIVE_INT64,
                                           dataspace);
              }
            } else if constexpr (std::is_same_v<LimitType,
                                                std::vector<std::string>>) {
              if (!variant_value.empty()) {
                H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
                hsize_t dims[1] = {variant_value.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset =
                    limits_group.createDataSet(key, str_type, dataspace);
                std::vector<const char*> c_strings;
                for (const auto& str : variant_value) {
                  c_strings.push_back(str.c_str());
                }
                dataset.write(c_strings.data(), str_type);
              } else {
                H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
                hsize_t dims[1] = {0};
                H5::DataSpace dataspace(1, dims);
                limits_group.createDataSet(key, str_type, dataspace);
              }
            }
          },
          limit_value);
    }
  }

  if (!documented_.empty()) {
    H5::Group doc_group = group.createGroup("_documented");
    for (const auto& [key, is_documented] : documented_) {
      int bool_val = is_documented ? 1 : 0;
      hsize_t dims[1] = {1};
      H5::DataSpace dataspace(1, dims);
      H5::DataSet dataset =
          doc_group.createDataSet(key, H5::PredType::NATIVE_INT, dataspace);
      dataset.write(&bool_val, H5::PredType::NATIVE_INT);
    }
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
    } else if (obj_type == H5G_GROUP) {
      // Load metadata groups
      if (obj_name == "_descriptions") {
        H5::Group desc_group = group.openGroup(obj_name);
        hsize_t num_desc = desc_group.getNumObjs();
        for (hsize_t j = 0; j < num_desc; ++j) {
          std::string key = desc_group.getObjnameByIdx(j);
          H5::DataSet dataset = desc_group.openDataSet(key);
          H5::DataType datatype = dataset.getDataType();
          if (datatype.isVariableStr()) {
            char* str_data;
            dataset.read(&str_data, datatype);
            settings->descriptions_[key] = std::string(str_data);
            free(str_data);
          } else {
            size_t str_len = datatype.getSize();
            std::string str_data(str_len, '\0');
            dataset.read(&str_data[0], datatype);
            size_t null_pos = str_data.find('\0');
            if (null_pos != std::string::npos) {
              str_data.resize(null_pos);
            }
            settings->descriptions_[key] = str_data;
          }
        }
      } else if (obj_name == "_limits") {
        H5::Group limits_group = group.openGroup(obj_name);
        hsize_t num_limits = limits_group.getNumObjs();
        for (hsize_t j = 0; j < num_limits; ++j) {
          std::string key = limits_group.getObjnameByIdx(j);
          H5::DataSet dataset = limits_group.openDataSet(key);
          H5::DataSpace dataspace = dataset.getSpace();
          H5::DataType datatype = dataset.getDataType();

          int ndims = dataspace.getSimpleExtentNdims();
          std::vector<hsize_t> dims(ndims);
          dataspace.getSimpleExtentDims(dims.data(), nullptr);

          H5T_class_t type_class = datatype.getClass();

          if (type_class == H5T_INTEGER && dims[0] == 2) {
            // Pair of integers
            int64_t pair_data[2];
            dataset.read(pair_data, H5::PredType::NATIVE_INT64);
            settings->limits_[key] =
                data::BoundConstraint<int64_t>{pair_data[0], pair_data[1]};
          } else if (type_class == H5T_FLOAT && dims[0] == 2) {
            // Pair of doubles
            double pair_data[2];
            dataset.read(pair_data, H5::PredType::NATIVE_DOUBLE);
            settings->limits_[key] =
                data::BoundConstraint<double>{pair_data[0], pair_data[1]};
          } else if (type_class == H5T_INTEGER) {
            // Vector of integers
            std::vector<int64_t> vec(dims[0]);
            if (dims[0] > 0) {
              dataset.read(vec.data(), H5::PredType::NATIVE_INT64);
            }
            settings->limits_[key] = data::ListConstraint<int64_t>{{vec}};
          } else if (type_class == H5T_STRING) {
            // Vector of strings
            std::vector<std::string> vec;
            if (dims[0] > 0) {
              if (datatype.isVariableStr()) {
                std::vector<char*> str_data(dims[0]);
                dataset.read(str_data.data(), datatype);
                for (size_t k = 0; k < dims[0]; ++k) {
                  vec.emplace_back(str_data[k]);
                  free(str_data[k]);
                }
              } else {
                size_t str_len = datatype.getSize();
                std::vector<char> buffer(dims[0] * str_len);
                dataset.read(buffer.data(), datatype);
                for (size_t k = 0; k < dims[0]; ++k) {
                  const char* str_start = buffer.data() + k * str_len;
                  std::string str(str_start, str_len);
                  size_t null_pos = str.find('\0');
                  if (null_pos != std::string::npos) {
                    str.resize(null_pos);
                  }
                  vec.push_back(str);
                }
              }
            }
            settings->limits_[key] = data::ListConstraint<std::string>{{vec}};
          }
        }
      } else if (obj_name == "_documented") {
        H5::Group doc_group = group.openGroup(obj_name);
        hsize_t num_docs = doc_group.getNumObjs();
        for (hsize_t j = 0; j < num_docs; ++j) {
          std::string key = doc_group.getObjnameByIdx(j);
          H5::DataSet dataset = doc_group.openDataSet(key);
          int bool_val;
          dataset.read(&bool_val, H5::PredType::NATIVE_INT);
          settings->documented_[key] = static_cast<bool>(bool_val);
        }
      }
    }
  }

  return settings;
}

}  // namespace qdk::chemistry::data
