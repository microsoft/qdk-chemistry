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

/**
 * @brief Get basis set shells and ECP information for a given nuclear charge.
 * @param nuclear_charge Nuclear charge of the element.
 * @param basis_set_name Name of the basis set.
 * @param atom_index Index of the atom in the structure.
 * @return A tuple containing:
 *         - vector of Shells for the element,
 *         - vector of ECP Shells for the element,
 *         - number of ECP electrons for the element.
 */
std::tuple<std::vector<Shell>, std::vector<Shell>, size_t>
get_basis_for_nuclear_charge(const double nuclear_charge,
                             std::string basis_set_name,
                             const size_t atom_index) {
  const utils::BasisArchive archive(basis_set_name);
  std::ifstream fin(archive.json());
  auto data = nlohmann::json::parse(fin);

  // extract basis set information for each nuclear charge
  size_t num_ecp_elecs = 0;
  std::vector<Shell> ecp_shells;

  std::vector<Shell> shells;
  // convert nuclear_charge to int and then to string for JSON lookup
  auto nuclear_charge_string = std::to_string(static_cast<int>(nuclear_charge));

  // get element specific data
  auto element_data = data["elements"][nuclear_charge_string];

  // iterate over electron shells
  for (const auto& shell : element_data["electron_shells"]) {
    for (size_t i = 0; i < shell["coefficients"].size(); i++) {
      size_t am_size = shell["angular_momentum"].size();
      size_t momentum = shell["angular_momentum"][am_size > 1 ? i : 0];

      // fill exponents and coefficients
      std::vector<double> exponents;
      std::vector<double> coefficients;
      std::vector<int> rpowers;
      int power = 0;
      for (size_t k = 0; k < shell["exponents"].size(); k++) {
        exponents.push_back(
            std::stod(shell["exponents"][k].get<std::string>()));
        coefficients.push_back(
            std::stod(shell["coefficients"][i][k].get<std::string>()));
        rpowers.push_back(0);
        power++;
      }

      // create shell and add to list
      Shell sh{atom_index, static_cast<OrbitalType>(momentum), exponents,
               coefficients, rpowers};
      shells.push_back(sh);
    }
  }

  // handle ecps
  if (element_data.contains("ecp_electrons")) {
    auto num_core_elec = element_data["ecp_electrons"].get<int>();
    num_ecp_elecs = static_cast<size_t>(num_core_elec);
    auto ecp_data = element_data["ecp_potentials"];
    for (const auto& ecp_entry : ecp_data) {
      // only support scalar_ecp
      if (ecp_entry["ecp_type"].get<std::string>() != "scalar_ecp") {
        throw std::invalid_argument("only scalar_ecp is supported");
      }
      auto am_entry = ecp_entry["angular_momentum"];
      if (am_entry.size() != 1) {
        throw std::invalid_argument("only one angular momentum is expected");
      }
      auto contraction_size = ecp_entry["gaussian_exponents"].size();

      // fill exponents, coefficients, and rpowers
      std::vector<double> exponents;
      std::vector<double> coefficients;
      std::vector<int> rpowers;
      for (size_t i = 0; i < contraction_size; i++) {
        exponents.push_back(
            std::stod(ecp_entry["gaussian_exponents"][i].get<std::string>()));
        coefficients.push_back(
            std::stod(ecp_entry["coefficients"][0][i].get<std::string>()));
        rpowers.push_back(ecp_entry["r_exponents"][i].get<int>());
      }

      // create shell and add to ecp list
      Shell sh{atom_index, static_cast<OrbitalType>(am_entry[0]), exponents,
               coefficients, rpowers};
      ecp_shells.push_back(sh);
    }
    // The basis library stores the local term at the highest angular momentum.
    const auto local_term =
        std::max_element(ecp_shells.begin(), ecp_shells.end(),
                         [](const Shell& lhs, const Shell& rhs) {
                           return lhs.orbital_type < rhs.orbital_type;
                         });
    if (local_term != ecp_shells.end()) {
      local_term->orbital_type = OrbitalType::UL;
    }
  }
  return {shells, ecp_shells, num_ecp_elecs};
}

}  // namespace qdk::chemistry::data::detail
