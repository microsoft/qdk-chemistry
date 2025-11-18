// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/util/int1e.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#include <qdk/chemistry/scf/util/env_helper.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <libint2.hpp>
#include <nlohmann/json.hpp>
#include <regex>

#include "util/macros.h"
#include "util/mpi_vars.h"

namespace qdk::chemistry::scf {

void norm_psi4_mode(std::vector<Shell>& shells);

size_t load_from_database_json(std::filesystem::path bs_path, BasisSet& basis) {
  if (!std::filesystem::exists(bs_path)) {
    throw std::invalid_argument(
        fmt::format("basis set {} is not supported", bs_path.string()));
  }
  std::ifstream fin(bs_path);
  auto data = nlohmann::json::parse(fin);

  auto& mol = *basis.mol;
  auto& shells = basis.shells;
  auto& ecp_shells = basis.ecp_shells;
  auto& element_ecp_electrons = basis.element_ecp_electrons;

  size_t n_ecp_electrons = 0;
  for (uint64_t i = 0; i < mol.n_atoms; ++i) {
    auto atomic_num = std::to_string(mol.atomic_nums[i]);
    VERIFY_INPUT(data["elements"].contains(atomic_num),
                 fmt::format("Element (Z={}) is not supported in basis set {}",
                             atomic_num, bs_path.string()));
    auto elem = data["elements"][atomic_num];
    std::vector<Shell> atom_shells;
    auto elec_shells = elem["electron_shells"];
    for (const auto& entry : elec_shells) {
      for (size_t j = 0; j < entry["coefficients"].size(); j++) {
        Shell sh{0};
        sh.atom_index = i;
        sh.O = mol.coords[i];
        int am_size = entry["angular_momentum"].size();
        sh.angular_momentum = entry["angular_momentum"][am_size > 1 ? j : 0];
        sh.contraction = 0;
        for (size_t k = 0; k < entry["exponents"].size(); k++) {
          auto alpha = std::stod(entry["exponents"][k].get<std::string>());
          auto coeff =
              std::stod(entry["coefficients"][j][k].get<std::string>());
          if (fabs(coeff) > 1e-15) {
            sh.exponents[sh.contraction] = alpha;
            sh.coefficients[sh.contraction] = coeff;
            sh.contraction++;
          }
        }

        atom_shells.push_back(sh);
      }
    }
    stable_sort(atom_shells.begin(), atom_shells.end(),
                [](const auto& x, const auto& y) {
                  return x.angular_momentum == y.angular_momentum
                             ? x.contraction > y.contraction
                             : x.angular_momentum < y.angular_momentum;
                });
    shells.insert(shells.end(), atom_shells.begin(), atom_shells.end());

    if (!elem.contains("ecp_potentials")) continue;

    auto n_core_electrons = elem["ecp_electrons"].get<int>();
    element_ecp_electrons[mol.atomic_nums[i]] = n_core_electrons;
    mol.atomic_charges[i] = mol.atomic_nums[i] - n_core_electrons;
    n_ecp_electrons += n_core_electrons;
    auto ecp = elem["ecp_potentials"];
    for (const auto& entry : ecp) {
      if (entry["ecp_type"].get<std::string>() != "scalar_ecp") {
        spdlog::error("only scalar_ecp is supported");
        exit(EXIT_FAILURE);
      }
      auto am = entry["angular_momentum"];
      if (am.size() != 1) {
        spdlog::error("only one angular momentum is expected");
        exit(EXIT_FAILURE);
      }

      Shell sh{0};
      sh.atom_index = i;
      sh.O = mol.coords[i];
      sh.angular_momentum = am[0];
      sh.contraction = entry["gaussian_exponents"].size();
      for (size_t k = 0; k < sh.contraction; k++) {
        sh.coefficients[k] =
            std::stod(entry["coefficients"][0][k].get<std::string>());
        sh.exponents[k] =
            std::stod(entry["gaussian_exponents"][k].get<std::string>());
        sh.rpowers[k] = entry["r_exponents"][k];
      }
      ecp_shells.push_back(sh);
    }
  }

  return n_ecp_electrons;
}

std::shared_ptr<BasisSet> BasisSet::from_database_json(
    std::shared_ptr<Molecule> mol, const std::string& path, BasisMode mode,
    bool pure, bool sort) {
  return std::shared_ptr<BasisSet>(new BasisSet(mol, path, mode, pure, sort));
}

BasisSet::BasisSet(std::shared_ptr<Molecule> mol, const std::string& path,
                   BasisMode mode, bool pure, bool sort)
    : mol(mol), mode(mode), pure(pure) {
  std::string normalized_path =
      std::regex_replace(path, std::regex("\\*"), "_st_");
  normalized_path =
      std::regex_replace(normalized_path, std::regex("/"), "_sl_");
  std::filesystem::path bs_path(normalized_path);
  if (!std::filesystem::exists(bs_path)) {
    bs_path = std::filesystem::temp_directory_path() / "qdk" / "chemistry" /
              "basis" / (normalized_path + ".json");
    name = normalized_path;
  } else {
    name = bs_path.stem();
  }
  if (!std::filesystem::exists(bs_path)) {
    auto compressed_path = QDKChemistryConfig::get_resources_dir() /
                           "compressed" / (name + ".tar.gz");
    if (mpi::get_local_rank() == 0 &&
        std::filesystem::exists(compressed_path)) {
      auto odir = std::filesystem::temp_directory_path() / "qdk" / "chemistry";
      if (!std::filesystem::exists(odir)) {
        std::filesystem::create_directories(odir);
      }
      auto cmd =
          fmt::format("tar xzf \"{}\" --directory \"{}\"",
                      compressed_path.generic_string(), odir.generic_string());
      spdlog::trace("Execute command: {}", cmd);
      int return_code = std::system(cmd.c_str());
      if (return_code != 0) {
        spdlog::error("command execution failed: {}", cmd);
        exit(EXIT_FAILURE);
      }
    }
  }

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi::get_world_size() > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // Load basis from JSON
  n_ecp_electrons = load_from_database_json(bs_path, *this);

  if (mode == BasisMode::PSI4) {
    norm_psi4_mode(shells);
  }  // RAW branch doesn't normalize

  if (sort) {
    std::stable_sort(shells.begin(), shells.end(),
                     [](const auto& x, const auto& y) {
                       return x.angular_momentum != y.angular_momentum
                                  ? x.angular_momentum < y.angular_momentum
                                  : x.contraction < y.contraction;
                     });
  }

  num_basis_funcs = std::accumulate(
      shells.begin(), shells.end(), 0, [&pure](auto sum, const auto& sh) {
        int sz = 0;
        if (pure) {
          sz = 2 * sh.angular_momentum + 1;
        } else {
          sz = (sh.angular_momentum + 1) * (sh.angular_momentum + 2) / 2;
        }
        return sum + sz;
      });

  // Calculate atom -> bf idx map
  calc_atom2bf();

  if (libint2::initialized()) {
    spdlog::trace("Loaded basis set from {}: n_shells={}, n_basis_funcs={}",
                  path, shells.size(), num_basis_funcs);
    shell_pairs_ = OneBodyIntegral::compute_shell_pairs(shells);
  }
}

void BasisSet::calc_atom2bf() {
  int bf_idx = 0;
  atom2bf_ = std::vector<uint8_t>(mol->n_atoms * num_basis_funcs, 0);
  for (auto& sh : shells) {
    int sz = pure ? 2 * sh.angular_momentum + 1
                  : (sh.angular_momentum + 1) * (sh.angular_momentum + 2) / 2;
    for (int i = 0; i < sz; i++, bf_idx++) {
      atom2bf_[sh.atom_index * num_basis_funcs + bf_idx] = 1;
    }
  }
}

const std::vector<std::pair<int, int>>& BasisSet::get_shell_pairs() const {
  if (shell_pairs_.empty()) {
    throw std::runtime_error(
        "shell_pairs data not available. Call non-const get_shell_pairs() "
        "first to compute it.");
  }
  return shell_pairs_;
}

const std::vector<std::pair<int, int>>& BasisSet::get_shell_pairs() {
  if (shell_pairs_.empty()) {
    shell_pairs_ = OneBodyIntegral::compute_shell_pairs(shells);
  }
  return shell_pairs_;
}

void norm_psi4_mode(std::vector<Shell>& shells) {
  // normalize
  std::vector<double> double_factorial(20, 1);
  for (int i = 3; i < 20; i++) {
    double_factorial[i] = double_factorial[i - 2] * (i - 1);
  }

  constexpr double sqrt_PI_cubed = std::sqrt(std::pow(std::acos(-1), 3));

  for (auto& shell : shells) {
    int am = shell.angular_momentum;
    for (size_t i = 0; i < shell.contraction; i++) {
      if (shell.exponents[i] > 0) {
        const double exp2 = 2 * shell.exponents[i];
        const double exp2_to_am32 = std::pow(exp2, am + 1) * std::sqrt(exp2);
        const double normalization_factor =
            std::sqrt(std::pow(2, am) * exp2_to_am32 /
                      (sqrt_PI_cubed * double_factorial[2 * am]));

        shell.coefficients[i] *= normalization_factor;
      }
    }
    double norm = 0.0;
    for (size_t i = 0; i < shell.contraction; i++) {
      for (size_t j = 0; j <= i; j++) {
        auto gamma = shell.exponents[i] + shell.exponents[j];
        norm += (i == j ? 1 : 2) * double_factorial[2 * am] * sqrt_PI_cubed *
                shell.coefficients[i] * shell.coefficients[j] /
                (pow(2, am) * pow(gamma, am + 1) * sqrt(gamma));
      }
    }
    double normalization_factor = 1 / sqrt(norm);
    for (size_t i = 0; i < shell.contraction; i++) {
      shell.coefficients[i] *= normalization_factor;
    }
  }
}

nlohmann::ordered_json BasisSet::to_json() const {
  std::vector<nlohmann::ordered_json> json_shells;
  for (const auto& sh : shells) {
    json_shells.push_back(sh.to_json());
  }

  std::vector<nlohmann::ordered_json> json_ecp_shells;
  for (const auto& sh : ecp_shells) {
    json_ecp_shells.push_back(sh.to_json(true /*is_ecp*/));
  }

  std::vector<int> json_element_ecp_electrons;
  for (const auto& [k, v] : element_ecp_electrons) {
    json_element_ecp_electrons.push_back(k);
    json_element_ecp_electrons.push_back(v);
  }

  auto basis_set_json = nlohmann::ordered_json(
      {{"name", name},
       {"pure", pure},
       {"mode", mode == BasisMode::PSI4
                    ? "PSI4"
                    : (mode == BasisMode::RAW ? "RAW" : "UNKNOWN")},
       {"atoms", mol->atomic_nums},
       {"num_basis_funcs", num_basis_funcs},
       {"electron_shells", json_shells},
       {"ecp_shells", json_ecp_shells},
       {"element_ecp_electrons", json_element_ecp_electrons}});
  return basis_set_json;
}

Shell Shell::from_json(const nlohmann::ordered_json& rec,
                       const std::shared_ptr<Molecule> mol) {
  Shell sh;
  sh.atom_index = rec["atom"].template get<uint64_t>();
  sh.angular_momentum = rec["am"].template get<uint64_t>();

  std::vector<double> exp = rec["exp"].template get<std::vector<double>>();
  std::vector<double> coeff = rec["coeff"].template get<std::vector<double>>();
  std::vector<int> rpow;
  if (rec.contains("r_exp")) {
    rpow = rec["r_exp"].template get<std::vector<int>>();
  }
  sh.contraction = exp.size();
  VERIFY(sh.contraction == coeff.size());
  sh.O = mol->coords[sh.atom_index];

  std::copy(exp.begin(), exp.end(), sh.exponents);
  std::copy(coeff.begin(), coeff.end(), sh.coefficients);
  if (rpow.size()) {
    std::copy(rpow.begin(), rpow.end(), sh.rpowers);
  }
  return sh;
};

nlohmann::ordered_json Shell::to_json(const bool& is_ecp) const {
  nlohmann::ordered_json record;
  if (is_ecp) {
    record = {{"atom", atom_index},
              {"ecp_type", "scalar"},
              {"am", angular_momentum},
              {"exp", std::vector<double>(exponents, exponents + contraction)},
              {"coeff",
               std::vector<double>(coefficients, coefficients + contraction)},
              {"r_exp", std::vector<int>(rpowers, rpowers + contraction)}};
  } else {
    record = {{"atom", atom_index},
              {"am", angular_momentum},
              {"exp", std::vector<double>(exponents, exponents + contraction)},
              {"coeff",
               std::vector<double>(coefficients, coefficients + contraction)}};
  }
  return record;
}

BasisSet::BasisSet() = default;

std::shared_ptr<BasisSet> BasisSet::from_serialized_json(
    std::shared_ptr<Molecule> mol, std::string _path) {
  // Check path existence
  auto path = std::filesystem::path(_path);
  if (!std::filesystem::exists(path))
    throw std::runtime_error(
        fmt::format("Basis File {} Does Not Exist", path.string()));

  spdlog::trace("Loading basis set from file: {}", _path);

  // Grab JSON from file
  auto json = nlohmann::ordered_json::parse(std::ifstream(path));

  // Call the JSON overload
  return from_serialized_json(mol, json);
}

std::shared_ptr<BasisSet> BasisSet::from_serialized_json(
    std::shared_ptr<Molecule> mol, const nlohmann::ordered_json& json) {
  // Create BasisSet Instance
  auto bs = std::shared_ptr<BasisSet>(new BasisSet());
  bs->mol = mol;

  // Read flat(ish) data
  bs->name = json["name"];
  bs->pure = json["pure"].template get<bool>();
  bs->num_basis_funcs = json["num_basis_funcs"].template get<int>();
  bs->mode = (json["mode"].template get<std::string>() == "PSI4")
                 ? BasisMode::PSI4
                 : BasisMode::RAW;
  std::string mode = json["mode"].template get<std::string>();

  // Read Basis Shells
  {
    nlohmann::ordered_json _shells = json["electron_shells"];
    std::vector<Shell> shells;
    for (auto rec : _shells) {
      Shell sh = Shell::from_json(rec, bs->mol);
      shells.push_back(sh);
    }
    bs->shells = std::move(shells);
  }

  spdlog::trace("Loaded basis set: n_shells={}, n_basis_funcs",
                bs->shells.size(), bs->num_basis_funcs);

  // Read ECP Shells
  nlohmann::ordered_json _ecp_shells = json["ecp_shells"];
  std::vector<Shell> ecp_shells;
  for (auto rec : _ecp_shells) {
    Shell sh = Shell::from_json(rec, bs->mol);
    ecp_shells.push_back(sh);
  }
  bs->ecp_shells = std::move(ecp_shells);

  // Read element_ecp_electrons from flat list format (support both old and new
  // keys)
  std::vector<int> element_ecp_electrons_json;
  if (json.contains("element_ecp_electrons")) {
    element_ecp_electrons_json =
        json["element_ecp_electrons"].get<std::vector<int>>();
  } else if (json.contains("elem2ecpcore")) {
    // Backward compatibility: support old key name
    element_ecp_electrons_json = json["elem2ecpcore"].get<std::vector<int>>();
  }

  if (element_ecp_electrons_json.size() % 2 != 0) {
    throw std::runtime_error(
        "element_ecp_electrons_json expects an even number of elements.");
  }
  for (size_t i = 0; i < element_ecp_electrons_json.size(); i += 2) {
    int atomic_num = element_ecp_electrons_json[i];
    int ecp_electrons = element_ecp_electrons_json[i + 1];
    bs->element_ecp_electrons[atomic_num] = ecp_electrons;
  }

  // Update atomic charges, total nuclear charge, and n_electrons based on ECPs
  bs->n_ecp_electrons = 0;
  for (size_t i = 0; i < bs->mol->n_atoms; ++i) {
    int atomic_num = bs->mol->atomic_nums[i];
    if (bs->element_ecp_electrons.count(atomic_num)) {
      int ecp_electrons = bs->element_ecp_electrons[atomic_num];
      bs->mol->atomic_charges[i] = atomic_num - ecp_electrons;
      bs->n_ecp_electrons += ecp_electrons;
    }
  }
  bs->mol->total_nuclear_charge = std::accumulate(
      bs->mol->atomic_charges.begin(), bs->mol->atomic_charges.end(), 0);
  bs->mol->n_electrons = bs->mol->total_nuclear_charge - bs->mol->charge;

  // Compute derived quantities
  bs->shell_pairs_ = OneBodyIntegral::compute_shell_pairs(bs->shells);
  bs->calc_atom2bf();

  return bs;
}

}  // namespace qdk::chemistry::scf
