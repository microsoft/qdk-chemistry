// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <numeric>
#include <qdk/chemistry/data/shell.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::data {

namespace {

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

}  // namespace

Shell::Shell(size_t atom_idx, OrbitalType orb_type,
             const std::vector<double>& exp_list,
             const std::vector<double>& coeff_list)
    : atom_index(atom_idx), orbital_type(orb_type) {
  QDK_LOG_TRACE_ENTERING();

  if (exp_list.size() != coeff_list.size()) {
    throw std::invalid_argument(
        "Exponents and coefficients must have the same size");
  }
  exponents.resize(exp_list.size());
  coefficients.resize(coeff_list.size());
  rpowers.resize(0);
  std::copy(exp_list.begin(), exp_list.end(), exponents.data());
  std::copy(coeff_list.begin(), coeff_list.end(), coefficients.data());
}

Shell::Shell(size_t atom_idx, OrbitalType orb_type,
             const std::vector<double>& exp_list,
             const std::vector<double>& coeff_list,
             const std::vector<int>& rpow_list)
    : atom_index(atom_idx), orbital_type(orb_type) {
  QDK_LOG_TRACE_ENTERING();
  if (exp_list.size() != coeff_list.size()) {
    throw std::invalid_argument(
        "Exponents and coefficients must have the same size");
  }
  if (!rpow_list.empty() && rpow_list.size() != exp_list.size()) {
    throw std::invalid_argument(
        "Radial powers must have the same size as exponents and coefficients");
  }
  exponents.resize(exp_list.size());
  coefficients.resize(coeff_list.size());
  rpowers.resize(rpow_list.size());
  std::copy(exp_list.begin(), exp_list.end(), exponents.data());
  std::copy(coeff_list.begin(), coeff_list.end(), coefficients.data());
  std::copy(rpow_list.begin(), rpow_list.end(), rpowers.data());
}

nlohmann::json Shell::to_json(const bool include_radial_powers) const {
  nlohmann::json json;
  json["orbital_type"] = orbital_type_to_string(orbital_type);
  json["exponents"] = std::vector<double>(exponents.data(),
                                          exponents.data() + exponents.size());
  json["coefficients"] = std::vector<double>(
      coefficients.data(), coefficients.data() + coefficients.size());
  if (include_radial_powers && has_radial_powers()) {
    json["rpowers"] =
        std::vector<int>(rpowers.data(), rpowers.data() + rpowers.size());
  }
  return json;
}

Shell Shell::from_json(const nlohmann::json& json, const size_t atom_index,
                       const bool allow_radial_powers) {
  const auto type =
      string_to_orbital_type(json.at("orbital_type").get<std::string>());
  const auto exponents = json.at("exponents").get<std::vector<double>>();
  const auto coefficients = json.at("coefficients").get<std::vector<double>>();
  if (json.contains("rpowers")) {
    if (!allow_radial_powers) {
      throw std::invalid_argument(
          "Unexpected radial powers in a regular Gaussian shell");
    }
    return Shell(atom_index, type, exponents, coefficients,
                 json.at("rpowers").get<std::vector<int>>());
  }
  return Shell(atom_index, type, exponents, coefficients);
}

void sort_shells_inplace(std::vector<Shell>& shells) {
  auto sort_shell_primitives = [](Shell& shell) {
    std::vector<size_t> indices(shell.get_num_primitives());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&shell](size_t lhs, size_t rhs) {
      return shell.exponents(lhs) > shell.exponents(rhs);
    });

    Eigen::VectorXd sorted_exponents(shell.get_num_primitives());
    Eigen::VectorXd sorted_coefficients(shell.get_num_primitives());
    Eigen::VectorXi sorted_rpowers(shell.get_num_primitives());
    for (size_t index = 0; index < indices.size(); ++index) {
      sorted_exponents(index) = shell.exponents(indices[index]);
      sorted_coefficients(index) = shell.coefficients(indices[index]);
      if (shell.has_radial_powers()) {
        sorted_rpowers(index) = shell.rpowers(indices[index]);
      }
    }
    shell.exponents = sorted_exponents;
    shell.coefficients = sorted_coefficients;
    if (shell.has_radial_powers()) {
      shell.rpowers = sorted_rpowers;
    }
  };

  auto shell_comparator = [](const Shell& lhs, const Shell& rhs) {
    if (lhs.orbital_type != rhs.orbital_type) {
      return lhs.orbital_type < rhs.orbital_type;
    }
    return lhs.exponents(0) > rhs.exponents(0);
  };

  std::for_each(shells.begin(), shells.end(), sort_shell_primitives);
  std::stable_sort(shells.begin(), shells.end(), shell_comparator);
}

ShellsPerAtom group_shells_by_atom(const std::vector<Shell>& shells,
                                   const size_t num_atoms,
                                   const std::string& shell_description) {
  ShellsPerAtom result(num_atoms);
  for (const auto& shell : shells) {
    if (shell.atom_index >= num_atoms) {
      throw std::invalid_argument(shell_description + " atom_index (" +
                                  std::to_string(shell.atom_index) +
                                  ") is out of range for structure with " +
                                  std::to_string(num_atoms) + " atoms");
    }
    result[shell.atom_index].push_back(shell);
  }
  return result;
}

std::vector<Shell> flatten_shells(const ShellsPerAtom& shells_per_atom) {
  std::vector<Shell> result;
  result.reserve(count_shells(shells_per_atom));
  for (const auto& atom_shells : shells_per_atom) {
    result.insert(result.end(), atom_shells.begin(), atom_shells.end());
  }
  return result;
}

size_t count_shells(const ShellsPerAtom& shells_per_atom) {
  size_t result = 0;
  for (const auto& atom_shells : shells_per_atom) {
    result += atom_shells.size();
  }
  return result;
}

size_t count_orbitals(const ShellsPerAtom& shells_per_atom,
                      const AOType atomic_orbital_type) {
  size_t result = 0;
  for (const auto& atom_shells : shells_per_atom) {
    for (const auto& shell : atom_shells) {
      result += shell.get_num_atomic_orbitals(atomic_orbital_type);
    }
  }
  return result;
}

std::string orbital_type_to_string(const OrbitalType orbital_type) {
  switch (orbital_type) {
    case OrbitalType::UL:
      return "ul";
    case OrbitalType::S:
      return "s";
    case OrbitalType::P:
      return "p";
    case OrbitalType::D:
      return "d";
    case OrbitalType::F:
      return "f";
    case OrbitalType::G:
      return "g";
    case OrbitalType::H:
      return "h";
    case OrbitalType::I:
      return "i";
    default:
      return "unknown";
  }
}

OrbitalType l_to_orbital_type(const int l) {
  switch (l) {
    case -1:
      return OrbitalType::UL;
    case 0:
      return OrbitalType::S;
    case 1:
      return OrbitalType::P;
    case 2:
      return OrbitalType::D;
    case 3:
      return OrbitalType::F;
    case 4:
      return OrbitalType::G;
    case 5:
      return OrbitalType::H;
    case 6:
      return OrbitalType::I;
    default:
      throw std::invalid_argument("Unsupported angular momentum l: " +
                                  std::to_string(l));
  }
}

OrbitalType string_to_orbital_type(const std::string& orbital_string) {
  const std::string normalized = lowercase(orbital_string);
  if (normalized == "ul") return OrbitalType::UL;
  if (normalized == "s") return OrbitalType::S;
  if (normalized == "p") return OrbitalType::P;
  if (normalized == "d") return OrbitalType::D;
  if (normalized == "f") return OrbitalType::F;
  if (normalized == "g") return OrbitalType::G;
  if (normalized == "h") return OrbitalType::H;
  if (normalized == "i") return OrbitalType::I;
  throw std::invalid_argument("Unknown orbital type string: " + orbital_string);
}

std::string atomic_orbital_type_to_string(const AOType atomic_orbital_type) {
  switch (atomic_orbital_type) {
    case AOType::Spherical:
      return "spherical";
    case AOType::Cartesian:
      return "cartesian";
    default:
      return "unknown";
  }
}

AOType string_to_atomic_orbital_type(const std::string& basis_string) {
  const std::string normalized = lowercase(basis_string);
  if (normalized == "spherical" || normalized == "sph") {
    return AOType::Spherical;
  }
  if (normalized == "cartesian" || normalized == "cart") {
    return AOType::Cartesian;
  }
  throw std::invalid_argument("Unknown atomic orbital type string: " +
                              basis_string);
}

}  // namespace qdk::chemistry::data
