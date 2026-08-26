// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "basis_library.hpp"

#include <qdk/chemistry/scf/config.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>

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

std::filesystem::path unpack_basis_set_archive(std::string& basis_set_name) {
  std::string normalized_name = normalize_basis_set_name(basis_set_name);
  std::filesystem::path file_path =
      qdk::chemistry::scf::QDKChemistryConfig::get_resources_dir() /
      "compressed" / (normalized_name + ".tar.gz");

  if (!std::filesystem::exists(file_path)) {
    throw std::invalid_argument("Basis set file does not exist: " +
                                file_path.string());
  }

  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "qdk" / "chemistry";
  if (!std::filesystem::exists(temp_dir)) {
    std::filesystem::create_directories(temp_dir);
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
    throw std::runtime_error("command execution failed: " + command);
  }

  return temp_dir;
}

std::filesystem::path get_correct_basis_set_file(std::string& basis_set_name) {
  std::filesystem::path temp_dir = unpack_basis_set_archive(basis_set_name);
  std::string normalized_name = normalize_basis_set_name(basis_set_name);
  std::filesystem::path json_file_path =
      temp_dir / "basis" / (normalized_name + ".json");
  if (!std::filesystem::exists(json_file_path)) {
    throw std::invalid_argument("Basis set JSON file does not exist: " +
                                json_file_path.string());
  }
  return json_file_path;
}

}  // namespace

std::tuple<std::vector<Shell>, std::vector<Shell>, size_t>
get_basis_for_nuclear_charge(const double nuclear_charge,
                             std::string basis_set_name,
                             const size_t atom_index) {
  std::filesystem::path json_file_path =
      get_correct_basis_set_file(basis_set_name);

  std::ifstream input(json_file_path);
  auto data = nlohmann::json::parse(input);
  size_t num_ecp_electrons = 0;
  std::vector<Shell> ecp_shells;
  std::vector<Shell> shells;
  auto nuclear_charge_string = std::to_string(static_cast<int>(nuclear_charge));
  auto element_data = data["elements"][nuclear_charge_string];

  for (const auto& shell : element_data["electron_shells"]) {
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
                          exponents, coefficients);
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
