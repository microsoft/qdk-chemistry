// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <sstream>
#include <stdexcept>

#include "basis_library.hpp"
#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   AOType atomic_orbital_type)
    : _name(name),
      _atomic_orbital_type(atomic_orbital_type),
      _ecp_name("none") {
  QDK_LOG_TRACE_ENTERING();
  // Organize shells by atom index
  for (const auto& shell : shells) {
    size_t atom_index = shell.atom_index;

    // Ensure we have enough space for this atom
    if (atom_index >= _shells_per_atom.size()) {
      _shells_per_atom.resize(atom_index + 1);
    }

    _shells_per_atom[atom_index].push_back(shell);
  }

  // Initialize ECP electrons vector with zeros for each atom
  _ecp_electrons.resize(_shells_per_atom.size(), 0);

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
  _init_ao_symmetries(nullptr, {});
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const Structure& structure, AOType atomic_orbital_type)
    : BasisSet(name, shells, std::make_shared<Structure>(structure),
               atomic_orbital_type) {
  QDK_LOG_TRACE_ENTERING();
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   std::shared_ptr<Structure> structure,
                   AOType atomic_orbital_type)
    : BasisSet(name, shells, std::move(structure), nullptr, {},
               atomic_orbital_type) {
  QDK_LOG_TRACE_ENTERING();
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   const Structure& structure, AOType atomic_orbital_type)
    : BasisSet(name, shells, "", ecp_shells, ecp_electrons,
               std::make_shared<Structure>(structure), atomic_orbital_type,
               CanonicalConstructionTag{}) {
  QDK_LOG_TRACE_ENTERING();
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   std::shared_ptr<Structure> structure,
                   AOType atomic_orbital_type)
    : BasisSet(name, shells, "", ecp_shells, ecp_electrons,
               std::move(structure), atomic_orbital_type,
               CanonicalConstructionTag{}) {
  QDK_LOG_TRACE_ENTERING();
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::string& ecp_name,
                   const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   const Structure& structure, AOType atomic_orbital_type)
    : BasisSet(name, shells, ecp_name, ecp_shells, ecp_electrons,
               std::make_shared<Structure>(structure), atomic_orbital_type,
               CanonicalConstructionTag{}) {
  QDK_LOG_TRACE_ENTERING();
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::string& ecp_name,
                   const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   std::shared_ptr<Structure> structure,
                   AOType atomic_orbital_type)
    : BasisSet(name, shells, ecp_name, ecp_shells, ecp_electrons,
               std::move(structure), atomic_orbital_type,
               CanonicalConstructionTag{}) {
  QDK_LOG_TRACE_ENTERING();
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   std::string ecp_name, const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   std::shared_ptr<Structure> structure,
                   AOType atomic_orbital_type, CanonicalConstructionTag)
    : _name(name),
      _atomic_orbital_type(atomic_orbital_type),
      _structure(structure),
      _ecp_name(std::move(ecp_name)),
      _ecp_electrons(ecp_electrons) {
  QDK_LOG_TRACE_ENTERING();
  if (_name.empty()) {
    throw std::invalid_argument("BasisSet name cannot be empty");
  }

  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  if ((!ecp_shells.empty() || !ecp_electrons.empty() || !ecp_name.empty()) &&
      ecp_electrons.size() != structure->get_num_atoms()) {
    throw std::invalid_argument(
        "ECP electrons vector size must match number of atoms");
  }

  if (_ecp_electrons.empty()) {
    _ecp_electrons.resize(structure->get_num_atoms(), 0);
  }

  const size_t num_atoms = structure->get_num_atoms();

  // Organize shells by atom index
  for (const auto& shell : shells) {
    size_t atom_index = shell.atom_index;
    if (atom_index >= num_atoms) {
      throw std::invalid_argument("Shell atom_index (" +
                                  std::to_string(atom_index) +
                                  ") is out of range for structure with " +
                                  std::to_string(num_atoms) + " atoms");
    }

    // Ensure we have enough space for this atom
    if (atom_index >= _shells_per_atom.size()) {
      _shells_per_atom.resize(atom_index + 1);
    }

    _shells_per_atom[atom_index].push_back(shell);
  }

  // Organize ECP shells by atom index
  for (const auto& ecp_shell : ecp_shells) {
    size_t atom_index = ecp_shell.atom_index;
    if (atom_index >= num_atoms) {
      throw std::invalid_argument("ECP shell atom_index (" +
                                  std::to_string(atom_index) +
                                  ") is out of range for structure with " +
                                  std::to_string(num_atoms) + " atoms");
    }

    // Ensure we have enough space for this atom
    if (atom_index >= _ecp_shells_per_atom.size()) {
      _ecp_shells_per_atom.resize(atom_index + 1);
    }

    _ecp_shells_per_atom[atom_index].push_back(ecp_shell);
  }

  // "none" is the canonical ECP name for a basis set that carries no ECP.
  if (_ecp_name.empty() && !has_ecp_shells() && !has_ecp_electrons()) {
    _ecp_name = "none";
  }

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
  _init_ao_symmetries(nullptr, {});
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   std::shared_ptr<Structure> structure,
                   std::shared_ptr<const SymmetryProduct> ao_symmetries,
                   std::unordered_map<SymmetryLabel, std::size_t> ao_extents,
                   AOType atomic_orbital_type)
    : _name(name),
      _atomic_orbital_type(atomic_orbital_type),
      _structure(structure),
      _ecp_name("none") {
  QDK_LOG_TRACE_ENTERING();
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  // Organize shells by atom index.
  for (const auto& shell : shells) {
    size_t atom_index = shell.atom_index;
    if (atom_index >= _shells_per_atom.size()) {
      _shells_per_atom.resize(atom_index + 1);
    }
    _shells_per_atom[atom_index].push_back(shell);
  }

  // Initialize ECP electrons vector with zeros for each atom.
  _ecp_electrons.resize(_structure->get_num_atoms(), 0);

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }

  _init_ao_symmetries(std::move(ao_symmetries), std::move(ao_extents));
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const EffectiveCorePotential& ecp,
                   std::shared_ptr<Structure> structure,
                   AOType atomic_orbital_type)
    : BasisSet(name, shells, ecp.get_name(), ecp.get_shells(),
               ecp.get_electrons(), std::move(structure), atomic_orbital_type,
               CanonicalConstructionTag{}) {
  QDK_LOG_TRACE_ENTERING();
}

std::shared_ptr<BasisSet> BasisSet::_from_components(
    const std::string& name, const std::vector<Shell>& shells,
    std::string ecp_name, const std::vector<Shell>& ecp_shells,
    const std::vector<size_t>& ecp_electrons,
    std::shared_ptr<Structure> structure, AOType atomic_orbital_type) {
  return std::shared_ptr<BasisSet>(new BasisSet(
      name, shells, std::move(ecp_name), ecp_shells, ecp_electrons,
      std::move(structure), atomic_orbital_type, CanonicalConstructionTag{}));
}

std::vector<Element> BasisSet::get_supported_elements_for_basis_set(
    std::string basis_set_name) {
  std::vector<Element> elements;

  std::filesystem::path basis_dir =
      qdk::chemistry::scf::QDKChemistryConfig::get_resources_dir();

  if (!std::filesystem::exists(basis_dir) ||
      !std::filesystem::is_directory(basis_dir)) {
    throw std::runtime_error("Basis set resources directory does not exist: " +
                             basis_dir.string());
  }

  // read elements from basis_summary.json
  std::filesystem::path summary_file = basis_dir / "basis_summary.json";
  if (!std::filesystem::exists(summary_file)) {
    throw std::runtime_error("Basis set summary file does not exist: " +
                             summary_file.string());
  }

  std::ifstream fin(summary_file);
  auto data = nlohmann::json::parse(fin);
  for (const auto& basis_entry : data) {
    if (basis_entry["name"].get<std::string>() == basis_set_name) {
      for (const auto& elem_str : basis_entry["supported_elements"]) {
        int atomic_number = std::stoi(elem_str.get<std::string>());
        elements.push_back(static_cast<Element>(atomic_number));
      }
      break;
    }
  }
  if (elements.empty()) {
    throw std::invalid_argument("No supported elements found for basis set: " +
                                basis_set_name);
  }

  return elements;
}

std::vector<std::string> BasisSet::get_supported_basis_set_names() {
  std::vector<std::string> basis_set_names;

  std::filesystem::path basis_dir =
      qdk::chemistry::scf::QDKChemistryConfig::get_resources_dir() /
      "compressed";

  if (!std::filesystem::exists(basis_dir) ||
      !std::filesystem::is_directory(basis_dir)) {
    throw std::runtime_error("Basis set resources directory does not exist: " +
                             basis_dir.string());
  }

  for (const auto& entry : std::filesystem::directory_iterator(basis_dir)) {
    if (entry.is_regular_file()) {
      auto path = entry.path();
      if (path.extension() == ".gz" && path.stem().extension() == ".tar") {
        // extract basis set name from filename and denormalize it
        std::string filename = path.stem().stem().string();
        std::string denormalized_name =
            detail::denormalize_basis_set_name(filename);
        basis_set_names.push_back(denormalized_name);
      }
    }
  }

  return basis_set_names;
}

std::shared_ptr<BasisSet> BasisSet::from_basis_name(
    const std::string& name, const Structure& structure,
    AOType atomic_orbital_type) {
  return BasisSet::from_basis_name(name, std::make_shared<Structure>(structure),
                                   atomic_orbital_type);
}

std::shared_ptr<BasisSet> BasisSet::from_basis_name(
    std::string basis_name, std::shared_ptr<Structure> structure,
    AOType atomic_orbital_type) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }
  basis_name = detail::lowercase_basis_name(std::move(basis_name));

  std::vector<Shell> all_basis_shells;
  std::vector<Shell> all_ecp_shells;
  std::vector<size_t> all_ecp_electrons;
  // loop over each atom in the structure and get basis set shells
  auto nuclear_charges = structure->get_nuclear_charges();
  for (size_t atom_index = 0;
       atom_index < static_cast<size_t>(nuclear_charges.size()); ++atom_index) {
    double nuclear_charge = nuclear_charges[atom_index];

    auto [shells, ecp_shells, ecp_electrons] =
        detail::get_basis_for_nuclear_charge(nuclear_charge, basis_name,
                                             atom_index);

    for (const auto& sh : shells) {
      all_basis_shells.push_back(sh);
    }

    all_ecp_electrons.push_back(ecp_electrons);
    for (const auto& sh : ecp_shells) {
      all_ecp_shells.push_back(sh);
    }
  }

  // sort basis shells
  sort_shells_inplace(all_basis_shells);
  // sort ecp shells
  sort_shells_inplace(all_ecp_shells);

  const bool has_ecp_data =
      !all_ecp_shells.empty() ||
      std::any_of(all_ecp_electrons.begin(), all_ecp_electrons.end(),
                  [](size_t electron_count) { return electron_count > 0; });
  const std::string ecp_name = has_ecp_data ? basis_name : "";
  return _from_components(basis_name, all_basis_shells, ecp_name,
                          all_ecp_shells, all_ecp_electrons, structure,
                          atomic_orbital_type);
}

std::shared_ptr<BasisSet> BasisSet::from_element_map(
    const std::map<std::string, std::string>& element_to_basis_map,
    const Structure& structure, AOType atomic_orbital_type) {
  return BasisSet::from_element_map(element_to_basis_map,
                                    std::make_shared<Structure>(structure),
                                    atomic_orbital_type);
}

std::shared_ptr<BasisSet> BasisSet::from_element_map(
    const std::map<std::string, std::string>& element_to_basis_map,
    std::shared_ptr<Structure> structure, AOType atomic_orbital_type) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  // convert basis and ecp to index_map
  std::map<size_t, std::string> tmp_basis_index_map;
  std::map<size_t, std::string> tmp_ecp_index_map;
  auto elements = structure->get_atomic_symbols();
  for (size_t atom_index = 0; atom_index < elements.size(); ++atom_index) {
    // basis
    auto it_basis = element_to_basis_map.find(elements[atom_index]);
    if (it_basis == element_to_basis_map.end()) {
      throw std::invalid_argument("No basis set specified for element: " +
                                  elements[atom_index]);
    }
    tmp_basis_index_map[atom_index] = it_basis->second;
  }

  return BasisSet::from_index_map(tmp_basis_index_map, structure,
                                  atomic_orbital_type);
}

std::shared_ptr<BasisSet> BasisSet::from_index_map(
    const std::map<size_t, std::string>& index_to_basis_map,
    const Structure& structure, AOType atomic_orbital_type) {
  return BasisSet::from_index_map(index_to_basis_map,
                                  std::make_shared<Structure>(structure),
                                  atomic_orbital_type);
}

std::shared_ptr<BasisSet> BasisSet::from_index_map(
    const std::map<size_t, std::string>& index_to_basis_map,
    std::shared_ptr<Structure> structure, AOType atomic_orbital_type) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  std::vector<Shell> all_basis_shells;
  std::vector<Shell> all_ecp_shells;
  std::vector<size_t> all_ecp_electrons;
  // loop over each atom in the structure and get basis set shells
  auto nuclear_charges = structure->get_nuclear_charges();
  for (size_t atom_index = 0;
       atom_index < static_cast<size_t>(nuclear_charges.size()); ++atom_index) {
    double nuclear_charge = nuclear_charges[atom_index];
    auto it = index_to_basis_map.find(atom_index);
    if (it == index_to_basis_map.end()) {
      throw std::invalid_argument("No basis set specified for atom index: " +
                                  std::to_string(atom_index));
    }
    std::string tmp_basis_set_name = detail::lowercase_basis_name(it->second);

    auto [shells, ecp_shells, ecp_electrons] =
        detail::get_basis_for_nuclear_charge(nuclear_charge, tmp_basis_set_name,
                                             atom_index);

    for (const auto& sh : shells) {
      all_basis_shells.push_back(sh);
    }

    all_ecp_electrons.push_back(ecp_electrons);
    for (const auto& sh : ecp_shells) {
      all_ecp_shells.push_back(sh);
    }
  }

  // sort basis shells
  sort_shells_inplace(all_basis_shells);
  // sort ecp shells
  sort_shells_inplace(all_ecp_shells);

  const bool has_ecp_data =
      !all_ecp_shells.empty() ||
      std::any_of(all_ecp_electrons.begin(), all_ecp_electrons.end(),
                  [](size_t electron_count) { return electron_count > 0; });
  const std::string ecp_name =
      has_ecp_data ? std::string(BasisSet::custom_ecp_name) : std::string();
  return _from_components(std::string(BasisSet::custom_name), all_basis_shells,
                          ecp_name, all_ecp_shells, all_ecp_electrons,
                          structure, atomic_orbital_type);
}

BasisSet::BasisSet(const BasisSet& other)
    : _name(other._name),
      _atomic_orbital_type(other._atomic_orbital_type),
      _structure(other._structure
                     ? std::make_shared<Structure>(*other._structure)
                     : nullptr),
      _shells_per_atom(other._shells_per_atom),
      _ecp_shells_per_atom(other._ecp_shells_per_atom),
      _ecp_name(other._ecp_name),
      _ecp_electrons(other._ecp_electrons),
      _ao_symmetries(other._ao_symmetries),
      _ao_extents(other._ao_extents),
      _auxiliary_bases(other._auxiliary_bases) {
  QDK_LOG_TRACE_ENTERING();
  // Cache will be invalidated by default (_cache_valid = false)
  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
}

BasisSet& BasisSet::operator=(const BasisSet& other) {
  QDK_LOG_TRACE_ENTERING();

  if (this != &other) {
    _name = other._name;
    _atomic_orbital_type = other._atomic_orbital_type;
    _shells_per_atom = other._shells_per_atom;
    _ecp_name = other._ecp_name;
    _ecp_shells_per_atom = other._ecp_shells_per_atom;
    _ecp_electrons = other._ecp_electrons;
    _ao_symmetries = other._ao_symmetries;
    _ao_extents = other._ao_extents;
    _auxiliary_bases = other._auxiliary_bases;
    if (other._structure) {
      _structure = std::make_shared<Structure>(*other._structure);
    } else {
      _structure.reset();
    }
    // Invalidate cache when assigning new basis set
    _cache_valid = false;
  }
  return *this;
}

AOType BasisSet::get_atomic_orbital_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _atomic_orbital_type;
}

std::vector<Shell> BasisSet::get_shells() const {
  QDK_LOG_TRACE_ENTERING();
  return flatten_shells(_shells_per_atom);
}

const std::vector<Shell>& BasisSet::get_shells_for_atom(
    size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();

  _validate_atom_index(atom_index);
  return _shells_per_atom[atom_index];
}

const Shell& BasisSet::get_shell(size_t shell_index) const {
  QDK_LOG_TRACE_ENTERING();

  _validate_shell_index(shell_index);

  size_t current_index = 0;
  for (const auto& atom_shells : _shells_per_atom) {
    if (shell_index < current_index + atom_shells.size()) {
      return atom_shells[shell_index - current_index];
    }
    current_index += atom_shells.size();
  }

  // Should never reach here if _validate_shell_index worked correctly
  throw std::out_of_range("Shell index not found");
}

size_t BasisSet::get_num_shells() const {
  QDK_LOG_TRACE_ENTERING();

  if (!_cache_valid) {
    _compute_mappings();
  }
  return _cached_num_shells;
}

size_t BasisSet::get_num_atoms() const {
  QDK_LOG_TRACE_ENTERING();

  return _shells_per_atom.size();
}

std::vector<Shell> BasisSet::get_ecp_shells() const {
  QDK_LOG_TRACE_ENTERING();
  return flatten_shells(_ecp_shells_per_atom);
}

const std::vector<Shell>& BasisSet::get_ecp_shells_for_atom(
    size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_atom_index(atom_index);
  if (atom_index >= _ecp_shells_per_atom.size()) {
    static const std::vector<Shell> empty_vector;
    return empty_vector;
  }
  return _ecp_shells_per_atom[atom_index];
}

const Shell& BasisSet::get_ecp_shell(size_t shell_index) const {
  QDK_LOG_TRACE_ENTERING();
  size_t total_ecp_shells = get_num_ecp_shells();
  if (shell_index >= total_ecp_shells) {
    throw std::out_of_range("ECP shell index " + std::to_string(shell_index) +
                            " out of range [0, " +
                            std::to_string(total_ecp_shells) + ")");
  }

  size_t current_index = 0;
  for (const auto& atom_ecp_shells : _ecp_shells_per_atom) {
    if (shell_index < current_index + atom_ecp_shells.size()) {
      return atom_ecp_shells[shell_index - current_index];
    }
    current_index += atom_ecp_shells.size();
  }

  // Should never reach here if validation worked correctly
  throw std::out_of_range("ECP shell index not found");
}

size_t BasisSet::get_num_ecp_shells() const {
  QDK_LOG_TRACE_ENTERING();
  return count_shells(_ecp_shells_per_atom);
}

bool BasisSet::has_ecp_shells() const {
  QDK_LOG_TRACE_ENTERING();
  return get_num_ecp_shells() > 0;
}

std::pair<size_t, int> BasisSet::get_atomic_orbital_info(
    size_t atomic_orbital_index) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_atomic_orbital_index(atomic_orbital_index);
  return basis_to_shell_index(atomic_orbital_index);
}

size_t BasisSet::get_num_atomic_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_cache_valid) {
    _compute_mappings();
  }
  return _cached_num_atomic_orbitals;
}

std::shared_ptr<const SymmetryProduct> BasisSet::ao_symmetries() const {
  QDK_LOG_TRACE_ENTERING();
  return _ao_symmetries;
}

const std::unordered_map<SymmetryLabel, std::size_t>& BasisSet::ao_extents()
    const {
  QDK_LOG_TRACE_ENTERING();
  return _ao_extents;
}

void BasisSet::_init_ao_symmetries(
    std::shared_ptr<const SymmetryProduct> ao_symmetries,
    std::unordered_map<SymmetryLabel, std::size_t> ao_extents) {
  if (ao_symmetries) {
    _ao_symmetries = std::move(ao_symmetries);
  } else {
    _ao_symmetries = std::make_shared<const SymmetryProduct>(
        std::vector<SymmetryAxis>{axes::spin(1, true)});
  }

  if (!ao_extents.empty()) {
    _ao_extents = std::move(ao_extents);
  } else {
    const std::size_t num_ao = get_num_atomic_orbitals();
    _ao_extents.clear();
    for (const auto& axis : _ao_symmetries->axes()) {
      for (const auto& label : axis.labels()) {
        _ao_extents.emplace(SymmetryLabel{label}, num_ao);
      }
    }
  }

  _validate_ao_symmetries();
}

void BasisSet::_validate_ao_symmetries() const {
  // Every extent key must be admissible under the AO SymmetryProduct.
  for (const auto& [label, extent] : _ao_extents) {
    (void)extent;
    for (const auto& axis : _ao_symmetries->axes()) {
      if (!label.has(axis.name())) {
        throw std::invalid_argument(
            "AO extent label is missing a value for axis '" +
            to_string(axis.name()) + "'.");
      }
      if (!axis.admits(*label.get(axis.name()))) {
        throw std::invalid_argument(
            "AO extent label is not admissible under the AO symmetries.");
      }
    }
  }

  // Symmetry aliasing: for an equivalent axis, every present label of that
  // axis must carry the same extent.
  for (const auto& axis : _ao_symmetries->axes()) {
    if (!axis.equivalent()) {
      continue;
    }
    bool have_ref = false;
    std::size_t reference = 0;
    for (const auto& value : axis.labels()) {
      auto it = _ao_extents.find(SymmetryLabel{value});
      if (it == _ao_extents.end()) {
        continue;
      }
      if (!have_ref) {
        reference = it->second;
        have_ref = true;
      } else if (it->second != reference) {
        throw std::invalid_argument(
            "Equivalent AO symmetry axis '" + to_string(axis.name()) +
            "' requires symmetry-equivalent labels to share an extent.");
      }
    }
  }
}

size_t BasisSet::get_atom_index_for_atomic_orbital(
    size_t atomic_orbital_index) const {
  QDK_LOG_TRACE_ENTERING();
  if (!_cache_valid) {
    _compute_mappings();
  }

  _validate_atomic_orbital_index(atomic_orbital_index);
  return _basis_to_atom_map[atomic_orbital_index];
}

std::vector<size_t> BasisSet::get_atomic_orbital_indices_for_atom(
    size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();

  _validate_atom_index(atom_index);

  std::vector<size_t> result;
  size_t basis_idx = 0;

  // Count atomic orbitals from atoms before this one
  for (size_t i = 0; i < atom_index; ++i) {
    for (const auto& shell : _shells_per_atom[i]) {
      basis_idx += shell.get_num_atomic_orbitals(_atomic_orbital_type);
    }
  }

  // Add atomic orbitals from this atom
  for (const auto& shell : _shells_per_atom[atom_index]) {
    size_t num_bf = shell.get_num_atomic_orbitals(_atomic_orbital_type);
    for (size_t j = 0; j < num_bf; ++j) {
      result.push_back(basis_idx + j);
    }
    basis_idx += num_bf;
  }

  return result;
}

std::vector<size_t> BasisSet::get_shell_indices_for_atom(
    size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();

  _validate_atom_index(atom_index);

  std::vector<size_t> result;
  size_t shell_idx = 0;

  // Count shells from atoms before this one
  for (size_t i = 0; i < atom_index; ++i) {
    shell_idx += _shells_per_atom[i].size();
  }

  // Add shell indices from this atom
  for (size_t j = 0; j < _shells_per_atom[atom_index].size(); ++j) {
    result.push_back(shell_idx + j);
  }

  return result;
}

size_t BasisSet::get_num_atomic_orbitals_for_atom(size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_atom_index(atom_index);

  size_t total = 0;
  for (const auto& shell : _shells_per_atom[atom_index]) {
    total += shell.get_num_atomic_orbitals(_atomic_orbital_type);
  }
  return total;
}

std::vector<size_t> BasisSet::get_shell_indices_for_orbital_type(
    OrbitalType orbital_type) const {
  QDK_LOG_TRACE_ENTERING();

  std::vector<size_t> result;
  size_t shell_idx = 0;

  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      if (shell.orbital_type == orbital_type) {
        result.push_back(shell_idx);
      }
      shell_idx++;
    }
  }

  return result;
}

size_t BasisSet::get_num_atomic_orbitals_for_orbital_type(
    OrbitalType orbital_type) const {
  QDK_LOG_TRACE_ENTERING();

  size_t total = 0;
  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      if (shell.orbital_type == orbital_type) {
        total += shell.get_num_atomic_orbitals(_atomic_orbital_type);
      }
    }
  }
  return total;
}

std::vector<size_t> BasisSet::get_shell_indices_for_atom_and_orbital_type(
    size_t atom_index, OrbitalType orbital_type) const {
  QDK_LOG_TRACE_ENTERING();

  _validate_atom_index(atom_index);

  std::vector<size_t> result;
  size_t shell_idx = 0;

  // Count shells from atoms before this one
  for (size_t i = 0; i < atom_index; ++i) {
    shell_idx += _shells_per_atom[i].size();
  }

  // Check shells for this atom
  for (const auto& shell : _shells_per_atom[atom_index]) {
    if (shell.orbital_type == orbital_type) {
      result.push_back(shell_idx);
    }
    shell_idx++;
  }

  return result;
}

std::vector<size_t> BasisSet::get_ecp_shell_indices_for_atom(
    size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_atom_index(atom_index);

  std::vector<size_t> result;

  if (atom_index >= _ecp_shells_per_atom.size()) {
    return result;  // No ECP shells for this atom
  }

  size_t ecp_shell_idx = 0;

  // Count ECP shells from atoms before this one
  for (size_t i = 0; i < atom_index && i < _ecp_shells_per_atom.size(); ++i) {
    ecp_shell_idx += _ecp_shells_per_atom[i].size();
  }

  // Add ECP shell indices from this atom
  for (size_t j = 0; j < _ecp_shells_per_atom[atom_index].size(); ++j) {
    result.push_back(ecp_shell_idx + j);
  }

  return result;
}

std::vector<size_t> BasisSet::get_ecp_shell_indices_for_orbital_type(
    OrbitalType orbital_type) const {
  QDK_LOG_TRACE_ENTERING();
  std::vector<size_t> result;
  size_t ecp_shell_idx = 0;

  for (const auto& atom_ecp_shells : _ecp_shells_per_atom) {
    for (const auto& ecp_shell : atom_ecp_shells) {
      if (ecp_shell.orbital_type == orbital_type) {
        result.push_back(ecp_shell_idx);
      }
      ecp_shell_idx++;
    }
  }

  return result;
}

std::vector<size_t> BasisSet::get_ecp_shell_indices_for_atom_and_orbital_type(
    size_t atom_index, OrbitalType orbital_type) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_atom_index(atom_index);

  std::vector<size_t> result;

  if (atom_index >= _ecp_shells_per_atom.size()) {
    return result;  // No ECP shells for this atom
  }

  size_t ecp_shell_idx = 0;

  // Count ECP shells from atoms before this one
  for (size_t i = 0; i < atom_index && i < _ecp_shells_per_atom.size(); ++i) {
    ecp_shell_idx += _ecp_shells_per_atom[i].size();
  }

  // Check ECP shells for this atom
  for (const auto& ecp_shell : _ecp_shells_per_atom[atom_index]) {
    if (ecp_shell.orbital_type == orbital_type) {
      result.push_back(ecp_shell_idx);
    }
    ecp_shell_idx++;
  }

  return result;
}

const std::string& BasisSet::get_name() const {
  QDK_LOG_TRACE_ENTERING();
  return _name;
}

const std::shared_ptr<Structure> BasisSet::get_structure() const {
  QDK_LOG_TRACE_ENTERING();

  if (!_structure) {
    throw std::runtime_error("No structure is associated with this basis set");
  }
  return _structure;
}

bool BasisSet::has_structure() const {
  QDK_LOG_TRACE_ENTERING();

  return _structure != nullptr;
}

bool BasisSet::has_auxiliary_basis(const AuxiliaryBasisRole role) const {
  QDK_LOG_TRACE_ENTERING();
  return _auxiliary_bases.contains(role);
}

std::shared_ptr<AuxiliaryBasis> BasisSet::get_auxiliary_basis(
    const AuxiliaryBasisRole role) const {
  QDK_LOG_TRACE_ENTERING();
  auto it = _auxiliary_bases.find(role);
  if (it == _auxiliary_bases.end()) {
    throw std::out_of_range("No auxiliary basis associated with role '" +
                            to_string(role) + "'");
  }
  return it->second;
}

std::shared_ptr<AuxiliaryBasis> BasisSet::resolve_auxiliary_basis(
    const AuxiliaryBasisRole role) const {
  QDK_LOG_TRACE_ENTERING();
  if (has_auxiliary_basis(role)) {
    return get_auxiliary_basis(role);
  }
  if (role == AuxiliaryBasisRole::JFit &&
      has_auxiliary_basis(AuxiliaryBasisRole::JKFit)) {
    return get_auxiliary_basis(AuxiliaryBasisRole::JKFit);
  }
  throw std::out_of_range("No auxiliary basis compatible with role '" +
                          to_string(role) + "'");
}

const BasisSet::AuxiliaryBasisMap& BasisSet::get_auxiliary_bases() const {
  QDK_LOG_TRACE_ENTERING();
  return _auxiliary_bases;
}

std::shared_ptr<BasisSet> with_auxiliary_basis(
    const BasisSet& basis_set, const AuxiliaryBasisRole role,
    std::shared_ptr<AuxiliaryBasis> auxiliary_basis) {
  return detail::BasisEnrichmentAccess::enrich_basis(
      basis_set, role, std::move(auxiliary_basis));
}

std::shared_ptr<BasisSet> detail::BasisEnrichmentAccess::enrich_basis(
    const BasisSet& basis_set, const AuxiliaryBasisRole role,
    std::shared_ptr<AuxiliaryBasis> auxiliary_basis) {
  QDK_LOG_TRACE_ENTERING();
  if (!auxiliary_basis) {
    throw std::invalid_argument("Auxiliary basis pointer cannot be nullptr");
  }
  if (!basis_set.has_structure()) {
    throw std::invalid_argument(
        "Cannot associate an auxiliary basis with a BasisSet that has no "
        "molecular structure");
  }
  if (basis_set.get_structure()->content_hash() !=
      auxiliary_basis->get_structure()->content_hash()) {
    throw std::invalid_argument(
        "Primary and auxiliary basis sets must describe the same molecular "
        "structure");
  }

  auto enriched = std::make_shared<BasisSet>(basis_set);
  enriched->_auxiliary_bases[role] = std::move(auxiliary_basis);
  return enriched;
}

const std::string& BasisSet::get_ecp_name() const {
  QDK_LOG_TRACE_ENTERING();
  return _ecp_name;
}

const std::vector<size_t>& BasisSet::get_ecp_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  return _ecp_electrons;
}

bool BasisSet::has_ecp_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if any atom has a finite number of ECP electrons
  for (size_t ecp_electrons : _ecp_electrons) {
    if (ecp_electrons > 0) {
      return true;
    }
  }
  return false;
}

bool BasisSet::_is_consistent_with_structure() const {
  QDK_LOG_TRACE_ENTERING();

  if (!has_structure()) {
    return true;  // No structure to validate against
  }

  // Check if we have shells for atoms that don't exist in the structure
  if (_shells_per_atom.size() > _structure->get_num_atoms()) {
    return false;
  }

  if (has_ecp_electrons() &&
      (_ecp_electrons.size() != _structure->get_num_atoms())) {
    return false;
  }

  // Check if any atom has shells but is beyond the structure's atom count
  for (size_t atom_idx = 0; atom_idx < _shells_per_atom.size(); ++atom_idx) {
    if (!_shells_per_atom[atom_idx].empty() &&
        atom_idx >= _structure->get_num_atoms()) {
      return false;
    }
  }

  // Check if any ECP shell references an atom beyond the structure's atom count
  for (size_t atom_idx = 0; atom_idx < _ecp_shells_per_atom.size();
       ++atom_idx) {
    if (!_ecp_shells_per_atom[atom_idx].empty() &&
        atom_idx >= _structure->get_num_atoms()) {
      return false;
    }
  }

  return true;
}

bool BasisSet::_is_valid() const {
  QDK_LOG_TRACE_ENTERING();

  // Check if we have any shells
  bool has_shells = false;
  for (const auto& atom_shells : _shells_per_atom) {
    if (!atom_shells.empty()) {
      has_shells = true;

      // Check if all shells have valid data
      for (const auto& shell : atom_shells) {
        if (shell.exponents.size() == 0 || shell.coefficients.size() == 0) {
          return false;
        }
        if (shell.exponents.size() != shell.coefficients.size()) {
          return false;
        }
      }
    }
  }
  bool has_ecp = false;
  for (const auto& atom_shells : _ecp_shells_per_atom) {
    if (!atom_shells.empty()) {
      has_ecp = true;

      // Check if all shells have valid data
      for (const auto& shell : atom_shells) {
        if (shell.exponents.size() == 0 || shell.coefficients.size() == 0) {
          return false;
        }
        if (shell.exponents.size() != shell.coefficients.size()) {
          return false;
        }
      }
    }
  }
  bool ecp_consistency = (has_ecp_electrons() == has_ecp_shells());

  return has_shells && _is_consistent_with_structure() && ecp_consistency;
}

std::string BasisSet::get_summary() const {
  QDK_LOG_TRACE_ENTERING();

  std::ostringstream oss;
  oss << "BasisSet: " << _name << "\n";
  oss << "Basis type: " << atomic_orbital_type_to_string(_atomic_orbital_type)
      << "\n";
  oss << "Total shells: " << get_num_shells() << "\n";
  oss << "Total atomic orbitals: " << get_num_atomic_orbitals() << "\n";
  oss << "Number of atoms: " << get_num_atoms() << "\n";

  oss << "Contains ECP: " << (has_ecp_electrons() ? "Yes" : "No") << "\n";
  if (has_ecp_electrons()) {
    oss << "ECP name: " << _ecp_name << "\n";
    oss << "Number of ECP shells: " << get_num_ecp_shells() << "\n";
    oss << "ECP electrons: ";
    for (size_t e : _ecp_electrons) {
      oss << e << " ";
    }
    oss << "\n";
  }
  if (has_structure()) {
    oss << "Associated structure: " << _structure->get_num_atoms()
        << " atoms\n";
  }

  // Count shells by orbital type
  std::map<OrbitalType, unsigned> shell_counts;
  std::map<OrbitalType, unsigned> bf_counts;
  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      shell_counts[shell.orbital_type]++;
      bf_counts[shell.orbital_type] +=
          shell.get_num_atomic_orbitals(_atomic_orbital_type);
    }
  }

  oss << "Shell breakdown:\n";
  for (const auto& [type, count] : shell_counts) {
    oss << "  " << orbital_type_to_string(type) << " shells: " << count
        << " (atomic orbitals: " << bf_counts[type] << ")\n";
  }

  return oss.str();
}

void BasisSet::to_file(const std::string& filename,
                       const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

std::shared_ptr<BasisSet> BasisSet::from_file(const std::string& filename,
                                              const std::string& type) {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

void BasisSet::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }

  // Validate filename has correct data type suffix
  std::string validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(BasisSet));

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<BasisSet> BasisSet::from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }

  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "basis_set");

  return _from_hdf5_file(validated_filename);
}

void BasisSet::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }

  // Validate filename has correct data type suffix
  std::string validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(BasisSet));

  _to_json_file(validated_filename);
}

std::shared_ptr<BasisSet> BasisSet::from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }

  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "basis_set");

  return _from_json_file(validated_filename);
}

void BasisSet::_to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  nlohmann::json j = to_json();
  file << j.dump(2);

  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

std::shared_ptr<BasisSet> BasisSet::_from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open BasisSet JSON file '" + filename +
                             "'. Please check that the file exists and you "
                             "have read permissions.");
  }

  nlohmann::json j;
  file >> j;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(j);
}

void BasisSet::_to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group basis_set_group = file.createGroup("/basis_set");
    to_hdf5(basis_set_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void BasisSet::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Save version first
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    H5::Attribute version_attr =
        group.createAttribute("version", string_type, scalar_space);
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);

    // Save metadata
    H5::Group metadata_group = group.createGroup("metadata");

    H5::Attribute name_attr =
        metadata_group.createAttribute("name", string_type, scalar_space);
    name_attr.write(string_type, _name);

    // Save basis type
    H5::Attribute atomic_orbital_type_attr = metadata_group.createAttribute(
        "atomic_orbital_type", string_type, scalar_space);
    std::string atomic_orbital_type_str =
        atomic_orbital_type_to_string(_atomic_orbital_type);
    atomic_orbital_type_attr.write(string_type, atomic_orbital_type_str);

    // Save shell data
    if (get_num_shells() > 0) {
      H5::Group shell_group = group.createGroup("shells");

      // Get all shells in flat format for HDF5 storage
      auto all_shells = get_shells();
      unsigned num_shells = all_shells.size();
      hsize_t dims[1] = {num_shells};
      H5::DataSpace dataspace(1, dims);

      // Create datasets for each shell property
      H5::DataSet atom_indices = shell_group.createDataSet(
          "atom_indices", H5::PredType::NATIVE_UINT, dataspace);
      H5::DataSet orbital_types = shell_group.createDataSet(
          "orbital_types", H5::PredType::NATIVE_INT, dataspace);
      H5::DataSet num_primitives = shell_group.createDataSet(
          "num_primitives", H5::PredType::NATIVE_UINT, dataspace);

      // Prepare data arrays
      std::vector<unsigned> atom_idx_data;
      std::vector<int> orbital_type_data;
      std::vector<unsigned> num_prim_data;
      std::vector<double> all_exponents;
      std::vector<double> all_coefficients;

      atom_idx_data.reserve(num_shells);
      orbital_type_data.reserve(num_shells);
      num_prim_data.reserve(num_shells);
      for (const auto& shell : all_shells) {
        atom_idx_data.push_back(shell.atom_index);
        orbital_type_data.push_back(static_cast<int>(shell.orbital_type));
        num_prim_data.push_back(shell.exponents.size());

        // Flatten primitives
        for (unsigned i = 0; i < shell.exponents.size(); ++i) {
          all_exponents.push_back(shell.exponents(i));
          all_coefficients.push_back(shell.coefficients(i));
        }
      }

      // Write shell data
      atom_indices.write(atom_idx_data.data(), H5::PredType::NATIVE_UINT);
      orbital_types.write(orbital_type_data.data(), H5::PredType::NATIVE_INT);
      num_primitives.write(num_prim_data.data(), H5::PredType::NATIVE_UINT);

      // Write primitive data
      if (!all_exponents.empty()) {
        hsize_t prim_dims[1] = {all_exponents.size()};
        H5::DataSpace prim_dataspace(1, prim_dims);

        H5::DataSet exponents = shell_group.createDataSet(
            "exponents", H5::PredType::NATIVE_DOUBLE, prim_dataspace);
        H5::DataSet coefficients = shell_group.createDataSet(
            "coefficients", H5::PredType::NATIVE_DOUBLE, prim_dataspace);

        exponents.write(all_exponents.data(), H5::PredType::NATIVE_DOUBLE);
        coefficients.write(all_coefficients.data(),
                           H5::PredType::NATIVE_DOUBLE);
      }
    }

    // Save ECP shell data
    if (get_num_ecp_shells() > 0) {
      H5::Group ecp_shell_group = group.createGroup("ecp_shells");

      // Get all ECP shells in flat format for HDF5 storage
      auto all_ecp_shells = get_ecp_shells();
      unsigned num_ecp_shells = all_ecp_shells.size();
      hsize_t ecp_dims[1] = {num_ecp_shells};
      H5::DataSpace ecp_dataspace(1, ecp_dims);

      // Create datasets for each ECP shell property
      H5::DataSet ecp_atom_indices = ecp_shell_group.createDataSet(
          "atom_indices", H5::PredType::NATIVE_UINT, ecp_dataspace);
      H5::DataSet ecp_orbital_types = ecp_shell_group.createDataSet(
          "orbital_types", H5::PredType::NATIVE_INT, ecp_dataspace);
      H5::DataSet ecp_num_primitives = ecp_shell_group.createDataSet(
          "num_primitives", H5::PredType::NATIVE_UINT, ecp_dataspace);

      // Prepare data arrays
      std::vector<unsigned> ecp_atom_idx_data;
      std::vector<int> ecp_orbital_type_data;
      std::vector<unsigned> ecp_num_prim_data;
      std::vector<double> ecp_all_exponents;
      std::vector<double> ecp_all_coefficients;
      std::vector<int> ecp_all_rpowers;

      ecp_atom_idx_data.reserve(num_ecp_shells);
      ecp_orbital_type_data.reserve(num_ecp_shells);
      ecp_num_prim_data.reserve(num_ecp_shells);
      for (const auto& ecp_shell : all_ecp_shells) {
        ecp_atom_idx_data.push_back(ecp_shell.atom_index);
        ecp_orbital_type_data.push_back(
            static_cast<int>(ecp_shell.orbital_type));
        ecp_num_prim_data.push_back(ecp_shell.exponents.size());

        // Flatten primitives
        for (unsigned i = 0; i < ecp_shell.exponents.size(); ++i) {
          ecp_all_exponents.push_back(ecp_shell.exponents(i));
          ecp_all_coefficients.push_back(ecp_shell.coefficients(i));
          if (ecp_shell.rpowers.size() > 0) {
            ecp_all_rpowers.push_back(ecp_shell.rpowers(i));
          } else {
            ecp_all_rpowers.push_back(0);  // Default to 0 if not specified
          }
        }
      }

      // Write ECP shell data
      ecp_atom_indices.write(ecp_atom_idx_data.data(),
                             H5::PredType::NATIVE_UINT);
      ecp_orbital_types.write(ecp_orbital_type_data.data(),
                              H5::PredType::NATIVE_INT);
      ecp_num_primitives.write(ecp_num_prim_data.data(),
                               H5::PredType::NATIVE_UINT);

      // Write ECP primitive data
      if (!ecp_all_exponents.empty()) {
        hsize_t ecp_prim_dims[1] = {ecp_all_exponents.size()};
        H5::DataSpace ecp_prim_dataspace(1, ecp_prim_dims);

        H5::DataSet ecp_exponents = ecp_shell_group.createDataSet(
            "exponents", H5::PredType::NATIVE_DOUBLE, ecp_prim_dataspace);
        H5::DataSet ecp_coefficients = ecp_shell_group.createDataSet(
            "coefficients", H5::PredType::NATIVE_DOUBLE, ecp_prim_dataspace);
        H5::DataSet ecp_rpowers = ecp_shell_group.createDataSet(
            "rpowers", H5::PredType::NATIVE_INT, ecp_prim_dataspace);

        ecp_exponents.write(ecp_all_exponents.data(),
                            H5::PredType::NATIVE_DOUBLE);
        ecp_coefficients.write(ecp_all_coefficients.data(),
                               H5::PredType::NATIVE_DOUBLE);
        ecp_rpowers.write(ecp_all_rpowers.data(), H5::PredType::NATIVE_INT);
      }
    }

    // Save ECP name and electrons if present
    if (!_ecp_name.empty()) {
      // Save ECP name as attribute
      H5::Attribute ecp_name_attr =
          group.createAttribute("ecp_name", string_type, scalar_space);
      ecp_name_attr.write(string_type, _ecp_name);
    }
    if (has_ecp_electrons()) {
      // Save ECP electrons array as dataset
      if (!_ecp_electrons.empty()) {
        hsize_t ecp_dims[1] = {_ecp_electrons.size()};
        H5::DataSpace ecp_dataspace(1, ecp_dims);
        H5::DataSet ecp_electrons = group.createDataSet(
            "ecp_electrons", H5::PredType::NATIVE_UINT64, ecp_dataspace);
        ecp_electrons.write(_ecp_electrons.data(), H5::PredType::NATIVE_UINT64);
      }
    }

    // Save nested structure if present
    if (has_structure()) {
      H5::Group structure_group = group.createGroup("structure");
      _structure->to_hdf5(structure_group);
    }

    if (!_auxiliary_bases.empty()) {
      H5::Group auxiliary_group = group.createGroup("auxiliary_bases");
      for (const auto& [role, auxiliary_basis] : _auxiliary_bases) {
        H5::Group role_group = auxiliary_group.createGroup(to_string(role));
        auxiliary_basis->to_hdf5(role_group);
      }
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<BasisSet> BasisSet::_from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open BasisSet HDF5 file '" + filename +
                             "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    H5::Group basis_set_group = file.openGroup("/basis_set");
    return from_hdf5(basis_set_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to read BasisSet data from HDF5 file '" +
                             filename + "'. " +
                             "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<BasisSet> BasisSet::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Validate version first
    if (!group.attrExists("version")) {
      throw std::runtime_error(
          "HDF5 group missing required 'version' attribute");
    }

    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    if (version_str != LEGACY_SERIALIZATION_VERSION) {
      validate_serialization_version(SERIALIZATION_VERSION, version_str);
    }

    // Load metadata
    H5::Group metadata_group = group.openGroup("metadata");

    H5::Attribute name_attr = metadata_group.openAttribute("name");
    std::string name;
    name_attr.read(string_type, name);

    // Load basis type if present
    AOType atomic_orbital_type = AOType::Spherical;  // Default
    if (metadata_group.attrExists("atomic_orbital_type")) {
      H5::Attribute atomic_orbital_type_attr =
          metadata_group.openAttribute("atomic_orbital_type");
      std::string atomic_orbital_type_str;
      atomic_orbital_type_attr.read(string_type, atomic_orbital_type_str);
      atomic_orbital_type =
          string_to_atomic_orbital_type(atomic_orbital_type_str);
    }

    // Collect shells
    std::vector<Shell> shells;

    // Load shells if present
    if (group.nameExists("shells")) {
      H5::Group shell_group = group.openGroup("shells");

      H5::DataSet atom_indices = shell_group.openDataSet("atom_indices");
      H5::DataSpace dataspace = atom_indices.getSpace();

      hsize_t dims[1];
      dataspace.getSimpleExtentDims(dims);
      unsigned num_shells = dims[0];

      if (num_shells > 0) {
        // Read data arrays
        std::vector<unsigned> atom_idx_data(num_shells);
        std::vector<int> orbital_type_data(num_shells);
        std::vector<unsigned> num_prim_data(num_shells);

        atom_indices.read(atom_idx_data.data(), H5::PredType::NATIVE_UINT);

        H5::DataSet orbital_types = shell_group.openDataSet("orbital_types");
        orbital_types.read(orbital_type_data.data(), H5::PredType::NATIVE_INT);

        H5::DataSet num_primitives = shell_group.openDataSet("num_primitives");
        num_primitives.read(num_prim_data.data(), H5::PredType::NATIVE_UINT);

        // Read primitive data
        std::vector<double> all_exponents;
        std::vector<double> all_coefficients;

        if (shell_group.nameExists("exponents") &&
            shell_group.nameExists("coefficients")) {
          H5::DataSet exponents = shell_group.openDataSet("exponents");
          H5::DataSet coefficients = shell_group.openDataSet("coefficients");

          H5::DataSpace exp_space = exponents.getSpace();
          hsize_t exp_dims[1];
          exp_space.getSimpleExtentDims(exp_dims);

          all_exponents.resize(exp_dims[0]);
          all_coefficients.resize(exp_dims[0]);

          exponents.read(all_exponents.data(), H5::PredType::NATIVE_DOUBLE);
          coefficients.read(all_coefficients.data(),
                            H5::PredType::NATIVE_DOUBLE);
        }

        // Reconstruct shells
        unsigned prim_offset = 0;

        for (unsigned i = 0; i < num_shells; ++i) {
          unsigned num_prims = num_prim_data[i];

          // Prepare primitive data vectors
          Eigen::VectorXd shell_exponents(num_prims);
          Eigen::VectorXd shell_coefficients(num_prims);

          for (unsigned j = 0; j < num_prims; ++j) {
            if (prim_offset + j < all_exponents.size()) {
              shell_exponents(j) = all_exponents[prim_offset + j];
              shell_coefficients(j) = all_coefficients[prim_offset + j];
            }
          }
          prim_offset += num_prims;

          Shell shell(atom_idx_data[i],
                      static_cast<OrbitalType>(orbital_type_data[i]),
                      shell_exponents, shell_coefficients);

          shells.push_back(shell);
        }
      }
    }

    // Collect ECP shells
    std::vector<Shell> ecp_shells;

    // Load ECP shells if present
    if (group.nameExists("ecp_shells")) {
      H5::Group ecp_shell_group = group.openGroup("ecp_shells");

      H5::DataSet ecp_atom_indices =
          ecp_shell_group.openDataSet("atom_indices");
      H5::DataSpace ecp_dataspace = ecp_atom_indices.getSpace();

      hsize_t ecp_dims[1];
      ecp_dataspace.getSimpleExtentDims(ecp_dims);
      unsigned num_ecp_shells = ecp_dims[0];

      if (num_ecp_shells > 0) {
        // Read data arrays
        std::vector<unsigned> ecp_atom_idx_data(num_ecp_shells);
        std::vector<int> ecp_orbital_type_data(num_ecp_shells);
        std::vector<unsigned> ecp_num_prim_data(num_ecp_shells);

        ecp_atom_indices.read(ecp_atom_idx_data.data(),
                              H5::PredType::NATIVE_UINT);

        H5::DataSet ecp_orbital_types =
            ecp_shell_group.openDataSet("orbital_types");
        ecp_orbital_types.read(ecp_orbital_type_data.data(),
                               H5::PredType::NATIVE_INT);

        H5::DataSet ecp_num_primitives =
            ecp_shell_group.openDataSet("num_primitives");
        ecp_num_primitives.read(ecp_num_prim_data.data(),
                                H5::PredType::NATIVE_UINT);

        // Read primitive data
        std::vector<double> ecp_all_exponents;
        std::vector<double> ecp_all_coefficients;
        std::vector<int> ecp_all_rpowers;

        if (ecp_shell_group.nameExists("exponents") &&
            ecp_shell_group.nameExists("coefficients")) {
          H5::DataSet ecp_exponents = ecp_shell_group.openDataSet("exponents");
          H5::DataSet ecp_coefficients =
              ecp_shell_group.openDataSet("coefficients");

          H5::DataSpace ecp_exp_space = ecp_exponents.getSpace();
          hsize_t ecp_exp_dims[1];
          ecp_exp_space.getSimpleExtentDims(ecp_exp_dims);

          ecp_all_exponents.resize(ecp_exp_dims[0]);
          ecp_all_coefficients.resize(ecp_exp_dims[0]);

          ecp_exponents.read(ecp_all_exponents.data(),
                             H5::PredType::NATIVE_DOUBLE);
          ecp_coefficients.read(ecp_all_coefficients.data(),
                                H5::PredType::NATIVE_DOUBLE);

          // Read radial powers if present
          if (ecp_shell_group.nameExists("rpowers")) {
            H5::DataSet ecp_rpowers = ecp_shell_group.openDataSet("rpowers");
            ecp_all_rpowers.resize(ecp_exp_dims[0]);
            ecp_rpowers.read(ecp_all_rpowers.data(), H5::PredType::NATIVE_INT);
          } else {
            ecp_all_rpowers.resize(ecp_exp_dims[0], 0);
          }
        }

        // Reconstruct ECP shells
        unsigned ecp_prim_offset = 0;

        for (unsigned i = 0; i < num_ecp_shells; ++i) {
          unsigned num_prims = ecp_num_prim_data[i];

          // Prepare primitive data vectors
          Eigen::VectorXd shell_exponents(num_prims);
          Eigen::VectorXd shell_coefficients(num_prims);
          Eigen::VectorXi shell_rpowers(num_prims);

          for (unsigned j = 0; j < num_prims; ++j) {
            if (ecp_prim_offset + j < ecp_all_exponents.size()) {
              shell_exponents(j) = ecp_all_exponents[ecp_prim_offset + j];
              shell_coefficients(j) = ecp_all_coefficients[ecp_prim_offset + j];
              shell_rpowers(j) = ecp_all_rpowers[ecp_prim_offset + j];
            }
          }
          ecp_prim_offset += num_prims;

          Shell ecp_shell(ecp_atom_idx_data[i],
                          static_cast<OrbitalType>(ecp_orbital_type_data[i]),
                          shell_exponents, shell_coefficients, shell_rpowers);

          ecp_shells.push_back(ecp_shell);
        }
      }
    }

    // Load ECP name and electrons
    std::string ecp_name;
    std::vector<size_t> ecp_electrons;

    if (group.attrExists("ecp_name")) {
      H5::Attribute ecp_name_attr = group.openAttribute("ecp_name");
      ecp_name_attr.read(string_type, ecp_name);
    }

    if (group.nameExists("ecp_electrons")) {
      H5::DataSet ecp_electrons_ds = group.openDataSet("ecp_electrons");
      H5::DataSpace ecp_dataspace = ecp_electrons_ds.getSpace();

      hsize_t ecp_dims[1];
      ecp_dataspace.getSimpleExtentDims(ecp_dims);
      ecp_electrons.resize(ecp_dims[0]);

      ecp_electrons_ds.read(ecp_electrons.data(), H5::PredType::NATIVE_UINT64);
    }

    if (!group.nameExists("structure")) {
      const bool has_real_ecp_electrons =
          std::any_of(ecp_electrons.begin(), ecp_electrons.end(),
                      [](size_t n) { return n != 0; });
      if (!ecp_shells.empty() || !ecp_name.empty() || has_real_ecp_electrons) {
        throw std::runtime_error(
            "HDF5 BasisSet contains ECP data but no structure; "
            "cannot reconstruct without losing information");
      }
      if (group.nameExists("auxiliary_bases")) {
        throw std::runtime_error(
            "HDF5 BasisSet contains auxiliary bases but no structure");
      }
      return std::make_shared<BasisSet>(name, shells, atomic_orbital_type);
    }

    if (!ecp_shells.empty() && ecp_electrons.empty()) {
      throw std::runtime_error(
          "ECP electrons data is missing but ECP shells are present");
    }

    H5::Group structure_group = group.openGroup("structure");
    auto structure = Structure::from_hdf5(structure_group);
    const bool has_real_ecp_electrons =
        std::any_of(ecp_electrons.begin(), ecp_electrons.end(),
                    [](size_t n) { return n != 0; });
    const std::string effective_ecp_name =
        (!ecp_shells.empty() || has_real_ecp_electrons) ? ecp_name
                                                        : std::string();
    auto basis_set = _from_components(
        name, shells, effective_ecp_name, ecp_shells, ecp_electrons,
        std::move(structure), atomic_orbital_type);

    if (group.nameExists("auxiliary_bases")) {
      H5::Group auxiliary_group = group.openGroup("auxiliary_bases");
      const hsize_t role_count = auxiliary_group.getNumObjs();
      for (hsize_t index = 0; index < role_count; ++index) {
        if (auxiliary_group.getObjTypeByIdx(index) != H5G_GROUP) {
          throw std::runtime_error(
              "Auxiliary-basis HDF5 entries must be groups");
        }
        const std::string role_name = auxiliary_group.getObjnameByIdx(index);
        const auto role = auxiliary_basis_role_from_string(role_name);
        if (basis_set->has_auxiliary_basis(role)) {
          throw std::runtime_error("Duplicate auxiliary-basis role in HDF5: " +
                                   role_name);
        }
        H5::Group role_group = auxiliary_group.openGroup(role_name);
        basis_set = with_auxiliary_basis(*basis_set, role,
                                         AuxiliaryBasis::from_hdf5(role_group));
      }
    }
    return basis_set;

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

nlohmann::json BasisSet::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  j["name"] = _name;
  j["atomic_orbital_type"] =
      atomic_orbital_type_to_string(_atomic_orbital_type);
  j["num_atomic_orbitals"] = get_num_atomic_orbitals();
  j["num_shells"] = get_num_shells();
  j["num_atoms"] = get_num_atoms();

  // Serialize shells organized by atom
  j["atoms"] = nlohmann::json::array();
  for (size_t atom_idx = 0; atom_idx < get_num_atoms(); ++atom_idx) {
    bool has_shells = atom_idx < _shells_per_atom.size() &&
                      !_shells_per_atom[atom_idx].empty();
    bool has_ecp_shells = atom_idx < _ecp_shells_per_atom.size() &&
                          !_ecp_shells_per_atom[atom_idx].empty();

    if (has_shells || has_ecp_shells) {
      nlohmann::json atom_json;
      atom_json["atom_index"] = atom_idx;

      // Serialize regular shells
      if (has_shells) {
        const auto& atom_shells = _shells_per_atom[atom_idx];
        atom_json["shells"] = nlohmann::json::array();

        for (const auto& shell : atom_shells) {
          atom_json["shells"].push_back(shell.to_json());
        }
      }

      // Serialize ECP shells
      if (has_ecp_shells) {
        const auto& atom_ecp_shells = _ecp_shells_per_atom[atom_idx];
        atom_json["ecp_shells"] = nlohmann::json::array();

        for (const auto& ecp_shell : atom_ecp_shells) {
          atom_json["ecp_shells"].push_back(ecp_shell.to_json());
        }
      }

      j["atoms"].push_back(atom_json);
    }
  }

  if (has_ecp_electrons() || !_ecp_name.empty()) {
    j["ecp_name"] = _ecp_name;
    j["ecp_electrons"] = _ecp_electrons;
  }

  if (has_structure()) {
    j["structure"] = _structure->to_json();
  }

  if (!_auxiliary_bases.empty()) {
    j["auxiliary_bases"] = nlohmann::json::object();
    for (const auto& [role, auxiliary_basis] : _auxiliary_bases) {
      j["auxiliary_bases"][to_string(role)] = auxiliary_basis->to_json();
    }
  }

  return j;
}

std::shared_ptr<BasisSet> BasisSet::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    const std::string version = j["version"];
    if (version != LEGACY_SERIALIZATION_VERSION) {
      validate_serialization_version(SERIALIZATION_VERSION, version);
    }

    if (j.contains("auxiliary_bases") && !j["auxiliary_bases"].is_object()) {
      throw std::runtime_error(
          "BasisSet auxiliary_bases must be a JSON object");
    }

    std::string name = j.value("name", "");

    // Load basis type if present, default to spherical
    AOType atomic_orbital_type;
    if (j.contains("atomic_orbital_type")) {
      atomic_orbital_type =
          string_to_atomic_orbital_type(j["atomic_orbital_type"]);
    } else {
      atomic_orbital_type = AOType::Spherical;
    }

    // Collect all shells and ECP shells
    std::vector<Shell> shells;
    std::vector<Shell> ecp_shells;

    // Try to load new per-atom format first
    if (j.contains("atoms") && j["atoms"].is_array()) {
      for (const auto& atom_json : j["atoms"]) {
        size_t atom_index = atom_json["atom_index"];

        // Load regular shells
        if (atom_json.contains("shells") && atom_json["shells"].is_array()) {
          for (const auto& shell_json : atom_json["shells"]) {
            // Load primitive data - try new array format first
            if (shell_json.contains("exponents") &&
                shell_json.contains("coefficients") &&
                shell_json["exponents"].is_array() &&
                shell_json["coefficients"].is_array()) {
              shells.push_back(Shell::from_json(shell_json, atom_index));
            }
            // Legacy support - old primitives format
            else if (shell_json.contains("primitives") &&
                     shell_json["primitives"].is_array()) {
              auto primitives = shell_json["primitives"];
              Eigen::VectorXd shell_exponents(primitives.size());
              Eigen::VectorXd shell_coefficients(primitives.size());

              for (size_t k = 0; k < primitives.size(); ++k) {
                shell_exponents(k) = primitives[k]["exponent"];
                shell_coefficients(k) = primitives[k]["coefficient"];
              }

              Shell shell(atom_index,
                          string_to_orbital_type(shell_json["orbital_type"]),
                          shell_exponents, shell_coefficients);
              shells.push_back(shell);
            }
          }
        }

        // Load ECP shells
        if (atom_json.contains("ecp_shells") &&
            atom_json["ecp_shells"].is_array()) {
          for (const auto& ecp_shell_json : atom_json["ecp_shells"]) {
            // Load primitive data
            if (ecp_shell_json.contains("exponents") &&
                ecp_shell_json.contains("coefficients") &&
                ecp_shell_json["exponents"].is_array() &&
                ecp_shell_json["coefficients"].is_array()) {
              ecp_shells.push_back(
                  Shell::from_json(ecp_shell_json, atom_index, true));
            }
          }
        }
      }
    }
    // Legacy support - flat shell list
    else if (j.contains("shells") && j["shells"].is_array()) {
      for (const auto& shell_json : j["shells"]) {
        size_t atom_index = shell_json["atom_index"];
        OrbitalType orbital_type =
            string_to_orbital_type(shell_json["orbital_type"]);

        // Load primitives - try new format first
        if (shell_json.contains("exponents") &&
            shell_json.contains("coefficients")) {
          auto exp_vec = shell_json["exponents"].get<std::vector<double>>();
          auto coeff_vec =
              shell_json["coefficients"].get<std::vector<double>>();
          Eigen::VectorXd shell_exponents =
              Eigen::Map<const Eigen::VectorXd>(exp_vec.data(), exp_vec.size());
          Eigen::VectorXd shell_coefficients =
              Eigen::Map<const Eigen::VectorXd>(coeff_vec.data(),
                                                coeff_vec.size());

          Shell shell(atom_index, orbital_type, shell_exponents,
                      shell_coefficients);
          shells.push_back(shell);
        }
        // Legacy primitives format
        else if (shell_json.contains("primitives") &&
                 shell_json["primitives"].is_array()) {
          auto primitives = shell_json["primitives"];
          Eigen::VectorXd shell_exponents(primitives.size());
          Eigen::VectorXd shell_coefficients(primitives.size());

          for (size_t k = 0; k < primitives.size(); ++k) {
            shell_exponents(k) = primitives[k]["exponent"];
            shell_coefficients(k) = primitives[k]["coefficient"];
          }

          Shell shell(atom_index, orbital_type, shell_exponents,
                      shell_coefficients);
          shells.push_back(shell);
        }
      }
    }
    // Legacy support - atomic orbitals converted to shells
    else if (j.contains("atomic_orbitals") && j["atomic_orbitals"].is_array()) {
      std::map<std::pair<size_t, OrbitalType>,
               std::vector<std::pair<double, double>>>
          primitive_map;

      for (const auto& bf_json : j["atomic_orbitals"]) {
        size_t atom_index = bf_json["atom_index"];
        OrbitalType orbital_type =
            string_to_orbital_type(bf_json["orbital_type"]);
        auto key = std::make_pair(atom_index, orbital_type);

        // Create primitive list if it doesn't exist
        if (primitive_map.find(key) == primitive_map.end()) {
          primitive_map[key] = std::vector<std::pair<double, double>>();

          // Load primitives (assuming all atomic orbitals in same shell have
          // same primitives)
          if (bf_json.contains("primitives") &&
              bf_json["primitives"].is_array()) {
            for (const auto& prim_json : bf_json["primitives"]) {
              primitive_map[key].emplace_back(prim_json["exponent"],
                                              prim_json["coefficient"]);
            }
          } else {
            // Legacy support - single exponent/coefficient
            double exponent = bf_json.value("exponent", 0.0);
            double coefficient = bf_json.value("coefficient", 1.0);
            primitive_map[key].emplace_back(exponent, coefficient);
          }
        }
      }

      // Convert map to shells
      for (const auto& [key, primitives] : primitive_map) {
        if (!primitives.empty()) {
          Eigen::VectorXd shell_exponents(primitives.size());
          Eigen::VectorXd shell_coefficients(primitives.size());

          for (size_t j = 0; j < primitives.size(); ++j) {
            shell_exponents(j) = primitives[j].first;
            shell_coefficients(j) = primitives[j].second;
          }

          shells.emplace_back(key.first, key.second, shell_exponents,
                              shell_coefficients);
        }
      }
    }

    // Load ECP name and electrons if present
    std::string ecp_name;
    std::vector<size_t> ecp_electrons;
    if (j.contains("ecp_name")) {
      ecp_name = j["ecp_name"];
    }
    if (j.contains("ecp_electrons")) {
      ecp_electrons = j["ecp_electrons"].get<std::vector<size_t>>();
    }

    if (!j.contains("structure")) {
      const bool has_real_ecp_electrons =
          std::any_of(ecp_electrons.begin(), ecp_electrons.end(),
                      [](size_t n) { return n != 0; });
      if (!ecp_shells.empty() || !ecp_name.empty() || has_real_ecp_electrons) {
        throw std::runtime_error(
            "Cannot create BasisSet with ECP data but without structure");
      }
      if (j.contains("auxiliary_bases") && !j["auxiliary_bases"].empty()) {
        throw std::runtime_error(
            "Cannot create BasisSet with auxiliary bases but without "
            "structure");
      }
      return std::make_shared<BasisSet>(name, shells, atomic_orbital_type);
    }

    if (!ecp_shells.empty() && ecp_electrons.empty()) {
      throw std::runtime_error(
          "ECP electrons data is missing but ECP shells are present");
    }

    auto structure = Structure::from_json(j["structure"]);
    const bool has_real_ecp_electrons =
        std::any_of(ecp_electrons.begin(), ecp_electrons.end(),
                    [](size_t n) { return n != 0; });
    const std::string effective_ecp_name =
        (!ecp_shells.empty() || has_real_ecp_electrons) ? ecp_name
                                                        : std::string();
    auto basis_set = _from_components(
        name, shells, effective_ecp_name, ecp_shells, ecp_electrons,
        std::move(structure), atomic_orbital_type);
    if (j.contains("auxiliary_bases")) {
      for (const auto& [role_name, auxiliary_json] :
           j["auxiliary_bases"].items()) {
        const auto role = auxiliary_basis_role_from_string(role_name);
        if (basis_set->has_auxiliary_basis(role)) {
          throw std::runtime_error("Duplicate auxiliary-basis role in JSON: " +
                                   role_name);
        }
        basis_set = with_auxiliary_basis(
            *basis_set, role, AuxiliaryBasis::from_json(auxiliary_json));
      }
    }
    return basis_set;

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse BasisSet from JSON: " +
                             std::string(e.what()));
  }
}

std::string BasisSet::orbital_type_to_string(OrbitalType orbital_type) {
  QDK_LOG_TRACE_ENTERING();
  return qdk::chemistry::data::orbital_type_to_string(orbital_type);
}

OrbitalType BasisSet::l_to_orbital_type(int l) {
  QDK_LOG_TRACE_ENTERING();
  return qdk::chemistry::data::l_to_orbital_type(l);
}

OrbitalType BasisSet::string_to_orbital_type(
    const std::string& orbital_string) {
  QDK_LOG_TRACE_ENTERING();
  return qdk::chemistry::data::string_to_orbital_type(orbital_string);
}

int BasisSet::get_angular_momentum(OrbitalType orbital_type) {
  QDK_LOG_TRACE_ENTERING();

  return static_cast<int>(orbital_type);
}

int BasisSet::get_num_orbitals_for_l(int l, AOType atomic_orbital_type) {
  QDK_LOG_TRACE_ENTERING();
  if (atomic_orbital_type == AOType::Spherical) {
    return 2 * l + 1;  // Spherical harmonics
  } else {
    return (l + 1) * (l + 2) / 2;  // Cartesian coordinates
  }
}

std::string BasisSet::atomic_orbital_type_to_string(
    AOType atomic_orbital_type) {
  QDK_LOG_TRACE_ENTERING();
  return qdk::chemistry::data::atomic_orbital_type_to_string(
      atomic_orbital_type);
}

AOType BasisSet::string_to_atomic_orbital_type(
    const std::string& basis_string) {
  QDK_LOG_TRACE_ENTERING();
  return qdk::chemistry::data::string_to_atomic_orbital_type(basis_string);
}

void BasisSet::_clear_maps() {
  QDK_LOG_TRACE_ENTERING();

  _cache_valid = false;
  _basis_to_atom_map.clear();
  _basis_to_shell_map.clear();
  _cached_num_atomic_orbitals = 0;
  _cached_num_shells = 0;
}

void BasisSet::_compute_mappings() const {
  QDK_LOG_TRACE_ENTERING();

  if (_cache_valid) {
    return;  // Already computed
  }

  // Clear existing mappings
  _basis_to_atom_map.clear();
  _basis_to_shell_map.clear();
  _cached_num_atomic_orbitals = 0;
  _cached_num_shells = 0;

  // Compute total number of shells and atomic orbitals
  for (const auto& atom_shells : _shells_per_atom) {
    _cached_num_shells += atom_shells.size();
    for (const auto& shell : atom_shells) {
      _cached_num_atomic_orbitals +=
          shell.get_num_atomic_orbitals(_atomic_orbital_type);
    }
  }

  // Reserve space for mappings
  _basis_to_atom_map.reserve(_cached_num_atomic_orbitals);
  _basis_to_shell_map.reserve(_cached_num_atomic_orbitals);

  // Build mappings
  size_t current_shell_idx = 0;
  for (size_t atom_idx = 0; atom_idx < _shells_per_atom.size(); ++atom_idx) {
    const auto& atom_shells = _shells_per_atom[atom_idx];

    for (const auto& shell : atom_shells) {
      size_t num_bf = shell.get_num_atomic_orbitals(_atomic_orbital_type);

      // Map each atomic orbital in this shell to the atom and shell
      for (size_t bf_idx = 0; bf_idx < num_bf; ++bf_idx) {
        _basis_to_atom_map.push_back(atom_idx);
        _basis_to_shell_map.push_back(current_shell_idx);
      }

      current_shell_idx++;
    }
  }

  _cache_valid = true;
}

void BasisSet::_validate_atomic_orbital_index(
    size_t atomic_orbital_index) const {
  QDK_LOG_TRACE_ENTERING();
  if (atomic_orbital_index >= get_num_atomic_orbitals()) {
    throw std::out_of_range("atomic orbital index " +
                            std::to_string(atomic_orbital_index) +
                            " is out of range. Maximum index: " +
                            std::to_string(get_num_atomic_orbitals() - 1));
  }
}

void BasisSet::_validate_shell_index(size_t shell_index) const {
  QDK_LOG_TRACE_ENTERING();

  if (shell_index >= get_num_shells()) {
    throw std::out_of_range("Shell index " + std::to_string(shell_index) +
                            " is out of range. Maximum index: " +
                            std::to_string(get_num_shells() - 1));
  }
}

void BasisSet::_validate_atom_index(size_t atom_index) const {
  QDK_LOG_TRACE_ENTERING();

  if (atom_index >= _shells_per_atom.size()) {
    throw std::out_of_range("Atom index " + std::to_string(atom_index) +
                            " is out of range. Maximum index: " +
                            std::to_string(_shells_per_atom.size() - 1));
  }
}

std::pair<size_t, int> BasisSet::basis_to_shell_index(
    size_t atomic_orbital_index) const {
  QDK_LOG_TRACE_ENTERING();
  if (!_cache_valid) {
    _compute_mappings();
  }

  _validate_atomic_orbital_index(atomic_orbital_index);

  size_t shell_index = _basis_to_shell_map[atomic_orbital_index];

  // Find the offset within the shell to compute magnetic quantum number
  size_t basis_offset_in_shell = 0;
  size_t current_basis_idx = 0;
  size_t current_shell_idx = 0;

  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      size_t num_bf = shell.get_num_atomic_orbitals(_atomic_orbital_type);

      if (current_shell_idx == shell_index) {
        // Found the shell, compute the offset
        basis_offset_in_shell = atomic_orbital_index - current_basis_idx;
        int l = shell.get_angular_momentum();
        int m_l = static_cast<int>(basis_offset_in_shell) - l;
        return std::make_pair(shell_index, m_l);
      }

      current_basis_idx += num_bf;
      current_shell_idx++;
    }
  }

  // Should never reach here if mapping is correct
  throw std::out_of_range("Shell index not found in basis set");
}

void BasisSet::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, _name);
  hash_value(ctx, static_cast<int64_t>(_atomic_orbital_type));
  if (_structure) {
    hash_field_presence(ctx, true);
    hash_value(ctx, _structure->content_hash());
  } else {
    hash_field_presence(ctx, false);
  }
  // Hash all shells per atom
  hash_value(ctx, static_cast<uint64_t>(_shells_per_atom.size()));
  for (const auto& atom_shells : _shells_per_atom) {
    hash_value(ctx, static_cast<uint64_t>(atom_shells.size()));
    for (const auto& shell : atom_shells) {
      hash_value(ctx, static_cast<uint64_t>(shell.atom_index));
      hash_value(ctx, static_cast<int64_t>(shell.orbital_type));
      hash_value(ctx, shell.exponents);
      hash_value(ctx, shell.coefficients);
      hash_value(ctx, shell.rpowers);
    }
  }
  // Hash ECP shells
  hash_value(ctx, static_cast<uint64_t>(_ecp_shells_per_atom.size()));
  for (const auto& atom_shells : _ecp_shells_per_atom) {
    hash_value(ctx, static_cast<uint64_t>(atom_shells.size()));
    for (const auto& shell : atom_shells) {
      hash_value(ctx, static_cast<uint64_t>(shell.atom_index));
      hash_value(ctx, static_cast<int64_t>(shell.orbital_type));
      hash_value(ctx, shell.exponents);
      hash_value(ctx, shell.coefficients);
      hash_value(ctx, shell.rpowers);
    }
  }
  hash_value(ctx, _ecp_name);
  hash_value(ctx, static_cast<uint64_t>(_ecp_electrons.size()));
  for (auto e : _ecp_electrons) {
    hash_value(ctx, static_cast<uint64_t>(e));
  }
  hash_value(ctx, static_cast<uint64_t>(_auxiliary_bases.size()));
  for (const auto& [role, auxiliary_basis] : _auxiliary_bases) {
    hash_value(ctx, to_string(role));
    hash_value(ctx, auxiliary_basis->content_hash());
  }
}

EffectiveCorePotential::EffectiveCorePotential(std::vector<Shell> shells,
                                               std::vector<size_t> electrons)
    : EffectiveCorePotential(std::string(custom_name), std::move(shells),
                             std::move(electrons)) {}

EffectiveCorePotential::EffectiveCorePotential(std::string name,
                                               std::vector<Shell> shells,
                                               std::vector<size_t> electrons)
    : _name(std::move(name)),
      _shells(std::move(shells)),
      _electrons(std::move(electrons)) {
  if (_name.empty()) {
    throw std::invalid_argument("EffectiveCorePotential name cannot be empty");
  }
  if (_shells.empty()) {
    throw std::invalid_argument(
        "EffectiveCorePotential shells cannot be empty");
  }
  if (_electrons.empty() ||
      std::none_of(_electrons.begin(), _electrons.end(),
                   [](size_t count) { return count > 0; })) {
    throw std::invalid_argument(
        "EffectiveCorePotential must replace at least one electron");
  }

  for (const auto& shell : _shells) {
    if (!shell.has_radial_powers()) {
      throw std::invalid_argument(
          "EffectiveCorePotential shells must contain radial powers");
    }
    if (shell.exponents.size() == 0 || shell.coefficients.size() == 0 ||
        shell.exponents.size() != shell.coefficients.size() ||
        shell.rpowers.size() != shell.exponents.size()) {
      throw std::invalid_argument(
          "EffectiveCorePotential contains an invalid shell");
    }
    if (shell.atom_index >= _electrons.size()) {
      throw std::invalid_argument(
          "EffectiveCorePotential shell atom_index is out of range");
    }
  }

  std::stable_sort(_shells.begin(), _shells.end(),
                   [](const Shell& lhs, const Shell& rhs) {
                     return lhs.atom_index < rhs.atom_index;
                   });
}

const std::string& EffectiveCorePotential::get_name() const { return _name; }

const std::vector<Shell>& EffectiveCorePotential::get_shells() const {
  return _shells;
}

const std::vector<size_t>& EffectiveCorePotential::get_electrons() const {
  return _electrons;
}

}  // namespace qdk::chemistry::data
