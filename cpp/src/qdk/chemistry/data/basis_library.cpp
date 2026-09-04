// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "basis_library.hpp"

#include <qdk/chemistry/scf/config.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <regex>
#include <stdexcept>
#include <utility>

namespace qdk::chemistry::data::detail {

std::string normalize_basis_set_name(const std::string& name) {
  std::string normalized = std::regex_replace(name, std::regex("\\*"), "_st_");
  normalized = std::regex_replace(normalized, std::regex("/"), "_sl_");
  normalized = std::regex_replace(normalized, std::regex("\\+"), "_pl_");
  return normalized;
}

std::string denormalize_basis_set_name(const std::string& normalized) {
  std::string name = std::regex_replace(normalized, std::regex("_st_"), "*");
  name = std::regex_replace(name, std::regex("_sl_"), "/");
  name = std::regex_replace(name, std::regex("_pl_"), "+");
  return name;
}

std::string lowercase_basis_name(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](const unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return name;
}

namespace {

class StagingDirectory {
 public:
  explicit StagingDirectory(std::filesystem::path path)
      : _path(std::move(path)) {}

  ~StagingDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  StagingDirectory(const StagingDirectory&) = delete;
  StagingDirectory& operator=(const StagingDirectory&) = delete;

 private:
  std::filesystem::path _path;
};

std::filesystem::path unpack_basis_set_archive(
    const std::string& basis_set_name) {
  std::string normalized_name = normalize_basis_set_name(basis_set_name);
  std::filesystem::path file_path =
      qdk::chemistry::scf::QDKChemistryConfig::get_resources_dir() /
      "compressed" / (normalized_name + ".tar.gz");

  if (!std::filesystem::exists(file_path)) {
    throw std::invalid_argument("Basis set file does not exist: " +
                                file_path.string());
  }

  const std::filesystem::path temp_root =
      std::filesystem::temp_directory_path() / "qdk" / "chemistry" /
      "basis_loads";
  std::filesystem::create_directories(temp_root);

  std::random_device random;
  std::filesystem::path temp_dir;
  for (size_t attempt = 0; attempt < 100; ++attempt) {
    temp_dir = temp_root / (normalized_name + "-" + std::to_string(random()) +
                            "-" + std::to_string(attempt));
    std::error_code error;
    if (std::filesystem::create_directory(temp_dir, error)) {
      break;
    }
    temp_dir.clear();
  }
  if (temp_dir.empty()) {
    throw std::runtime_error(
        "Unable to create a temporary basis extraction directory");
  }

  // GNU tar needs --force-local so "C:/..." is not read as host:path; BSD tar
  // rejects the flag, so detect support at runtime.
#ifdef _WIN32
  static const bool tar_has_force_local =
      (std::system("tar --force-local --version > nul 2>&1") == 0);
  const std::string tar_cmd =
      tar_has_force_local ? "tar --force-local -xzf " : "tar -xzf ";
#else
  const std::string tar_cmd = "tar -xzf ";
#endif
  auto command = tar_cmd + "\"" + file_path.generic_string() +
                 "\" --directory \"" + temp_dir.generic_string() + "\"";
  if (std::system(command.c_str()) != 0) {
    std::error_code error;
    std::filesystem::remove_all(temp_dir, error);
    throw std::runtime_error("command execution failed: " + command);
  }

  return temp_dir;
}

std::filesystem::path basis_set_json_path(const std::filesystem::path& root,
                                          const std::string& basis_set_name) {
  return root / "basis" / (normalize_basis_set_name(basis_set_name) + ".json");
}

nlohmann::json read_basis_set_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to read basis set file: " + path.string());
  }
  auto data = nlohmann::json::parse(input);
  if (!data.contains("elements") || !data["elements"].is_object()) {
    throw std::runtime_error("Basis set file has no elements object: " +
                             path.string());
  }
  return data;
}

nlohmann::json load_basis_set_json(const std::string& basis_set_name) {
  const std::filesystem::path cache_root =
      std::filesystem::temp_directory_path() / "qdk" / "chemistry";
  const std::filesystem::path cached_file =
      basis_set_json_path(cache_root, basis_set_name);
  if (std::filesystem::exists(cached_file)) {
    return read_basis_set_json(cached_file);
  }

  const std::filesystem::path staging_root =
      unpack_basis_set_archive(basis_set_name);
  const StagingDirectory cleanup(staging_root);
  const std::filesystem::path staged_file =
      basis_set_json_path(staging_root, basis_set_name);
  if (!std::filesystem::exists(staged_file)) {
    throw std::invalid_argument("Basis set JSON file does not exist: " +
                                staged_file.string());
  }

  auto data = read_basis_set_json(staged_file);
  std::filesystem::create_directories(cached_file.parent_path());

  std::error_code error;
  std::filesystem::rename(staged_file, cached_file, error);
  if (error && !std::filesystem::exists(cached_file)) {
    throw std::runtime_error("Unable to publish basis set cache file: " +
                             error.message());
  }
  return data;
}

}  // namespace

BasisLibrary::BasisLibrary(std::string basis_set_name)
    : _basis_set_name(lowercase_basis_name(std::move(basis_set_name))),
      _data(load_basis_set_json(_basis_set_name)) {}

std::tuple<std::vector<Shell>, std::vector<Shell>, size_t>
BasisLibrary::get_basis_for_nuclear_charge(const double nuclear_charge,
                                           const size_t atom_index) const {
  constexpr double integral_charge_tolerance = 1e-12;
  const double rounded_charge = std::round(nuclear_charge);
  if (!std::isfinite(nuclear_charge) || rounded_charge < 0.0 ||
      std::abs(nuclear_charge - rounded_charge) > integral_charge_tolerance ||
      rounded_charge >
          static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument(
        "Nuclear charges must be finite, nonnegative, and integral");
  }

  const auto nuclear_charge_string =
      std::to_string(static_cast<std::uint32_t>(rounded_charge));
  const auto& elements = _data.at("elements");
  const auto element = elements.find(nuclear_charge_string);
  if (element == elements.end() || !element->is_object() ||
      !element->contains("electron_shells") ||
      !element->at("electron_shells").is_array() ||
      element->at("electron_shells").empty()) {
    throw std::invalid_argument("Basis set '" + _basis_set_name +
                                "' is not available for nuclear charge " +
                                nuclear_charge_string);
  }

  const auto& element_data = *element;
  size_t num_ecp_electrons = 0;
  std::vector<Shell> ecp_shells;
  std::vector<Shell> shells;

  for (const auto& shell : element_data.at("electron_shells")) {
    for (size_t contraction = 0; contraction < shell["coefficients"].size();
         ++contraction) {
      size_t angular_momentum_size = shell["angular_momentum"].size();
      size_t momentum = shell["angular_momentum"]
                             [angular_momentum_size > 1 ? contraction : 0];

      std::vector<double> exponents;
      std::vector<double> coefficients;
      for (size_t primitive = 0; primitive < shell["exponents"].size();
           ++primitive) {
        exponents.push_back(
            std::stod(shell["exponents"][primitive].get<std::string>()));
        coefficients.push_back(std::stod(
            shell["coefficients"][contraction][primitive].get<std::string>()));
      }
      shells.emplace_back(atom_index, static_cast<OrbitalType>(momentum),
                          exponents, coefficients,
                          std::vector<int>(exponents.size(), 0));
    }
  }

  if (element_data.contains("ecp_electrons")) {
    num_ecp_electrons =
        static_cast<size_t>(element_data["ecp_electrons"].get<int>());
    for (const auto& ecp_entry : element_data["ecp_potentials"]) {
      if (ecp_entry["ecp_type"].get<std::string>() != "scalar_ecp") {
        throw std::invalid_argument("only scalar_ecp is supported");
      }
      auto angular_momentum = ecp_entry["angular_momentum"];
      if (angular_momentum.size() != 1) {
        throw std::invalid_argument("only one angular momentum is expected");
      }

      std::vector<double> exponents;
      std::vector<double> coefficients;
      std::vector<int> radial_powers;
      for (size_t primitive = 0;
           primitive < ecp_entry["gaussian_exponents"].size(); ++primitive) {
        exponents.push_back(std::stod(
            ecp_entry["gaussian_exponents"][primitive].get<std::string>()));
        coefficients.push_back(std::stod(
            ecp_entry["coefficients"][0][primitive].get<std::string>()));
        radial_powers.push_back(ecp_entry["r_exponents"][primitive].get<int>());
      }
      ecp_shells.emplace_back(atom_index,
                              static_cast<OrbitalType>(angular_momentum[0]),
                              exponents, coefficients, radial_powers);
    }
  }
  return {shells, ecp_shells, num_ecp_electrons};
}

}  // namespace qdk::chemistry::data::detail
