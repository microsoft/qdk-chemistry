// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

Shell::Shell(size_t atom_idx, OrbitalType orb_type,
             const std::vector<double>& exp_list,
             const std::vector<double>& coeff_list)
    : atom_index(atom_idx), orbital_type(orb_type) {
  if (exp_list.size() != coeff_list.size()) {
    throw std::invalid_argument(
        "Exponents and coefficients must have the same size");
  }
  exponents.resize(exp_list.size());
  coefficients.resize(coeff_list.size());
  rpowers.resize(0);  // No radial powers by default
  std::copy(exp_list.begin(), exp_list.end(), exponents.data());
  std::copy(coeff_list.begin(), coeff_list.end(), coefficients.data());
}

Shell::Shell(size_t atom_idx, OrbitalType orb_type,
             const std::vector<double>& exp_list,
             const std::vector<double>& coeff_list,
             const std::vector<int>& rpow_list)
    : atom_index(atom_idx), orbital_type(orb_type) {
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

BasisSet::BasisSet(const std::string& name, const Structure& structure,
                   BasisType basis_type)
    : BasisSet(name, std::make_shared<Structure>(structure), basis_type) {}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   BasisType basis_type)
    : _name(name), _basis_type(basis_type), _ecp_name("none") {
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
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const Structure& structure, BasisType basis_type)
    : BasisSet(name, shells, std::make_shared<Structure>(structure),
               basis_type) {}

BasisSet::BasisSet(const std::string& name,
                   std::shared_ptr<Structure> structure, BasisType basis_type)
    : _name(name),
      _basis_type(basis_type),
      _structure(structure),
      _ecp_name("none") {
  if (_name.empty()) {
    throw std::invalid_argument("BasisSet name cannot be empty");
  }
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  // Initialize ECP electrons vector with zeros for each atom
  _ecp_electrons.resize(structure->get_num_atoms(), 0);

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   std::shared_ptr<Structure> structure, BasisType basis_type)
    : _name(name),
      _basis_type(basis_type),
      _structure(structure),
      _ecp_name("none") {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

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
  _ecp_electrons.resize(structure->get_num_atoms(), 0);

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::vector<Shell>& ecp_shells,
                   const Structure& structure, BasisType basis_type)
    : BasisSet(name, shells, ecp_shells, std::make_shared<Structure>(structure),
               basis_type) {}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::vector<Shell>& ecp_shells,
                   std::shared_ptr<Structure> structure, BasisType basis_type)
    : _name(name),
      _basis_type(basis_type),
      _structure(structure),
      _ecp_name("none") {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  // Organize shells by atom index
  for (const auto& shell : shells) {
    size_t atom_index = shell.atom_index;

    // Ensure we have enough space for this atom
    if (atom_index >= _shells_per_atom.size()) {
      _shells_per_atom.resize(atom_index + 1);
    }

    _shells_per_atom[atom_index].push_back(shell);
  }

  // Organize ECP shells by atom index
  for (const auto& ecp_shell : ecp_shells) {
    size_t atom_index = ecp_shell.atom_index;

    // Ensure we have enough space for this atom
    if (atom_index >= _ecp_shells_per_atom.size()) {
      _ecp_shells_per_atom.resize(atom_index + 1);
    }

    _ecp_shells_per_atom[atom_index].push_back(ecp_shell);
  }

  // Initialize ECP electrons vector with zeros for each atom
  _ecp_electrons.resize(structure->get_num_atoms(), 0);

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::string& ecp_name,
                   const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   const Structure& structure, BasisType basis_type)
    : BasisSet(name, shells, ecp_name, ecp_shells, ecp_electrons,
               std::make_shared<Structure>(structure), basis_type) {}

BasisSet::BasisSet(const std::string& name, const std::vector<Shell>& shells,
                   const std::string& ecp_name,
                   const std::vector<Shell>& ecp_shells,
                   const std::vector<size_t>& ecp_electrons,
                   std::shared_ptr<Structure> structure, BasisType basis_type)
    : _name(name),
      _basis_type(basis_type),
      _structure(structure),
      _ecp_name(ecp_name),
      _ecp_electrons(ecp_electrons) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  if (ecp_electrons.size() != structure->get_num_atoms()) {
    throw std::invalid_argument(
        "ECP electrons vector size must match number of atoms");
  }

  // Organize shells by atom index
  for (const auto& shell : shells) {
    size_t atom_index = shell.atom_index;

    // Ensure we have enough space for this atom
    if (atom_index >= _shells_per_atom.size()) {
      _shells_per_atom.resize(atom_index + 1);
    }

    _shells_per_atom[atom_index].push_back(shell);
  }

  // Organize ECP shells by atom index
  for (const auto& ecp_shell : ecp_shells) {
    size_t atom_index = ecp_shell.atom_index;

    // Ensure we have enough space for this atom
    if (atom_index >= _ecp_shells_per_atom.size()) {
      _ecp_shells_per_atom.resize(atom_index + 1);
    }

    _ecp_shells_per_atom[atom_index].push_back(ecp_shell);
  }

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
}

BasisSet::BasisSet(const BasisSet& other)
    : _name(other._name),
      _basis_type(other._basis_type),
      _shells_per_atom(other._shells_per_atom),
      _ecp_name(other._ecp_name),
      _ecp_shells_per_atom(other._ecp_shells_per_atom),
      _ecp_electrons(other._ecp_electrons) {
  if (other._structure) {
    _structure = std::make_shared<Structure>(*other._structure);
  }
  // Cache will be invalidated by default (_cache_valid = false)
  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid BasisSet");
  }
}

BasisSet& BasisSet::operator=(const BasisSet& other) {
  if (this != &other) {
    _name = other._name;
    _basis_type = other._basis_type;
    _shells_per_atom = other._shells_per_atom;
    _ecp_name = other._ecp_name;
    _ecp_shells_per_atom = other._ecp_shells_per_atom;
    _ecp_electrons = other._ecp_electrons;
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

BasisType BasisSet::get_basis_type() const { return _basis_type; }

std::vector<Shell> BasisSet::get_shells() const {
  std::vector<Shell> all_shells;

  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      all_shells.push_back(shell);
    }
  }

  return all_shells;
}

const std::vector<Shell>& BasisSet::get_shells_for_atom(
    size_t atom_index) const {
  _validate_atom_index(atom_index);
  return _shells_per_atom[atom_index];
}

const Shell& BasisSet::get_shell(size_t shell_index) const {
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
  if (!_cache_valid) {
    _compute_mappings();
  }
  return _cached_num_shells;
}

size_t BasisSet::get_num_atoms() const { return _shells_per_atom.size(); }

std::vector<Shell> BasisSet::get_ecp_shells() const {
  std::vector<Shell> all_ecp_shells;

  for (const auto& atom_ecp_shells : _ecp_shells_per_atom) {
    for (const auto& shell : atom_ecp_shells) {
      all_ecp_shells.push_back(shell);
    }
  }

  return all_ecp_shells;
}

const std::vector<Shell>& BasisSet::get_ecp_shells_for_atom(
    size_t atom_index) const {
  _validate_atom_index(atom_index);
  if (atom_index >= _ecp_shells_per_atom.size()) {
    static const std::vector<Shell> empty_vector;
    return empty_vector;
  }
  return _ecp_shells_per_atom[atom_index];
}

const Shell& BasisSet::get_ecp_shell(size_t shell_index) const {
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
  size_t total = 0;
  for (const auto& atom_ecp_shells : _ecp_shells_per_atom) {
    total += atom_ecp_shells.size();
  }
  return total;
}

bool BasisSet::has_ecp_shells() const { return get_num_ecp_shells() > 0; }

std::pair<size_t, int> BasisSet::get_basis_function_info(
    size_t basis_index) const {
  _validate_basis_index(basis_index);
  return basis_to_shell_index(basis_index);
}

size_t BasisSet::get_num_basis_functions() const {
  if (!_cache_valid) {
    _compute_mappings();
  }
  return _cached_num_basis_functions;
}

size_t BasisSet::get_atom_index_for_basis_function(size_t basis_index) const {
  if (!_cache_valid) {
    _compute_mappings();
  }

  _validate_basis_index(basis_index);
  return _basis_to_atom_map[basis_index];
}

std::vector<size_t> BasisSet::get_basis_function_indices_for_atom(
    size_t atom_index) const {
  _validate_atom_index(atom_index);

  std::vector<size_t> result;
  size_t basis_idx = 0;

  // Count basis functions from atoms before this one
  for (size_t i = 0; i < atom_index; ++i) {
    for (const auto& shell : _shells_per_atom[i]) {
      basis_idx += shell.get_num_basis_functions(_basis_type);
    }
  }

  // Add basis functions from this atom
  for (const auto& shell : _shells_per_atom[atom_index]) {
    size_t num_bf = shell.get_num_basis_functions(_basis_type);
    for (size_t j = 0; j < num_bf; ++j) {
      result.push_back(basis_idx + j);
    }
    basis_idx += num_bf;
  }

  return result;
}

std::vector<size_t> BasisSet::get_shell_indices_for_atom(
    size_t atom_index) const {
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

size_t BasisSet::get_num_basis_functions_for_atom(size_t atom_index) const {
  _validate_atom_index(atom_index);

  size_t total = 0;
  for (const auto& shell : _shells_per_atom[atom_index]) {
    total += shell.get_num_basis_functions(_basis_type);
  }
  return total;
}

std::vector<size_t> BasisSet::get_shell_indices_for_orbital_type(
    OrbitalType orbital_type) const {
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

size_t BasisSet::get_num_basis_functions_for_orbital_type(
    OrbitalType orbital_type) const {
  size_t total = 0;
  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      if (shell.orbital_type == orbital_type) {
        total += shell.get_num_basis_functions(_basis_type);
      }
    }
  }
  return total;
}

std::vector<size_t> BasisSet::get_shell_indices_for_atom_and_orbital_type(
    size_t atom_index, OrbitalType orbital_type) const {
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

const std::string& BasisSet::get_name() const { return _name; }

const std::shared_ptr<Structure> BasisSet::get_structure() const {
  if (!_structure) {
    throw std::runtime_error("No structure is associated with this basis set");
  }
  return _structure;
}

bool BasisSet::has_structure() const { return _structure != nullptr; }

const std::string& BasisSet::get_ecp_name() const { return _ecp_name; }

const std::vector<size_t>& BasisSet::get_ecp_electrons() const {
  return _ecp_electrons;
}

bool BasisSet::has_ecp_electrons() const {
  // Check if any atom has a finite number of ECP electrons
  for (size_t ecp_electrons : _ecp_electrons) {
    if (ecp_electrons > 0) {
      return true;
    }
  }
  return false;
}

bool BasisSet::_is_consistent_with_structure() const {
  if (!has_structure()) {
    return true;  // No structure to validate against
  }

  const Structure& structure = *_structure;

  // Check if we have shells for atoms that don't exist in the structure
  if (_shells_per_atom.size() > structure.get_num_atoms()) {
    return false;
  }

  // Check if any atom has shells but is beyond the structure's atom count
  for (size_t atom_idx = 0; atom_idx < _shells_per_atom.size(); ++atom_idx) {
    if (!_shells_per_atom[atom_idx].empty() &&
        atom_idx >= structure.get_num_atoms()) {
      return false;
    }
  }

  return true;
}

bool BasisSet::_is_valid() const {
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

  return has_shells && _is_consistent_with_structure();
}

std::string BasisSet::get_summary() const {
  std::ostringstream oss;
  oss << "BasisSet: " << _name << "\n";
  oss << "Basis type: " << basis_type_to_string(_basis_type) << "\n";
  oss << "Total shells: " << get_num_shells() << "\n";
  oss << "Total basis functions: " << get_num_basis_functions() << "\n";
  oss << "Number of atoms: " << get_num_atoms() << "\n";

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
          shell.get_num_basis_functions(_basis_type);
    }
  }

  oss << "Shell breakdown:\n";
  for (const auto& [type, count] : shell_counts) {
    oss << "  " << orbital_type_to_string(type) << " shells: " << count
        << " (basis functions: " << bf_counts[type] << ")\n";
  }

  return oss.str();
}

void BasisSet::to_file(const std::string& filename,
                       const std::string& type) const {
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
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "basis_set");

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<BasisSet> BasisSet::from_hdf5_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "basis_set");

  return _from_hdf5_file(validated_filename);
}

void BasisSet::to_json_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "basis_set");

  _to_json_file(validated_filename);
}

std::shared_ptr<BasisSet> BasisSet::from_json_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "basis_set");

  return _from_json_file(validated_filename);
}

void BasisSet::_to_json_file(const std::string& filename) const {
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
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  nlohmann::json j;
  file >> j;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(j);
}

void BasisSet::_to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group basis_set_group = file.createGroup("/basis_set");
    to_hdf5(basis_set_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void BasisSet::to_hdf5(H5::Group& group) const {
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
    H5::Attribute basis_type_attr =
        metadata_group.createAttribute("basis_type", string_type, scalar_space);
    std::string basis_type_str = basis_type_to_string(_basis_type);
    basis_type_attr.write(string_type, basis_type_str);

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
    if (has_ecp_electrons() || !_ecp_name.empty()) {
      // Save ECP name as attribute
      H5::Attribute ecp_name_attr =
          group.createAttribute("ecp_name", string_type, scalar_space);
      ecp_name_attr.write(string_type, _ecp_name);

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

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<BasisSet> BasisSet::_from_hdf5_file(
    const std::string& filename) {
  try {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group basis_set_group = file.openGroup("/basis_set");
    return from_hdf5(basis_set_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<BasisSet> BasisSet::from_hdf5(H5::Group& group) {
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
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    // Load metadata
    H5::Group metadata_group = group.openGroup("metadata");

    H5::Attribute name_attr = metadata_group.openAttribute("name");
    std::string name;
    name_attr.read(string_type, name);

    // Load basis type if present
    BasisType basis_type = BasisType::Spherical;  // Default
    if (metadata_group.attrExists("basis_type")) {
      H5::Attribute basis_type_attr =
          metadata_group.openAttribute("basis_type");
      std::string basis_type_str;
      basis_type_attr.read(string_type, basis_type_str);
      basis_type = string_to_basis_type(basis_type_str);
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

    // Load nested structure if present
    std::shared_ptr<BasisSet> basis_set;
    if (group.nameExists("structure")) {
      H5::Group structure_group = group.openGroup("structure");
      auto structure = Structure::from_hdf5(structure_group);
      if (!ecp_shells.empty()) {
        if (!ecp_name.empty() && !ecp_electrons.empty()) {
          basis_set =
              std::make_shared<BasisSet>(name, shells, ecp_name, ecp_shells,
                                         ecp_electrons, *structure, basis_type);
        } else {
          basis_set = std::make_shared<BasisSet>(name, shells, ecp_shells,
                                                 *structure, basis_type);
        }
      } else {
        basis_set =
            std::make_shared<BasisSet>(name, shells, *structure, basis_type);
      }
    } else {
      basis_set = std::make_shared<BasisSet>(name, shells, basis_type);
    }

    return basis_set;

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

nlohmann::json BasisSet::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  j["name"] = _name;
  j["basis_type"] = basis_type_to_string(_basis_type);
  j["num_basis_functions"] = get_num_basis_functions();
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
          nlohmann::json shell_json;
          shell_json["orbital_type"] =
              orbital_type_to_string(shell.orbital_type);

          // Serialize primitive data as separate arrays
          std::vector<double> exp_vec(
              shell.exponents.data(),
              shell.exponents.data() + shell.exponents.size());
          std::vector<double> coeff_vec(
              shell.coefficients.data(),
              shell.coefficients.data() + shell.coefficients.size());
          shell_json["exponents"] = exp_vec;
          shell_json["coefficients"] = coeff_vec;

          atom_json["shells"].push_back(shell_json);
        }
      }

      // Serialize ECP shells
      if (has_ecp_shells) {
        const auto& atom_ecp_shells = _ecp_shells_per_atom[atom_idx];
        atom_json["ecp_shells"] = nlohmann::json::array();

        for (const auto& ecp_shell : atom_ecp_shells) {
          nlohmann::json ecp_shell_json;
          ecp_shell_json["orbital_type"] =
              orbital_type_to_string(ecp_shell.orbital_type);

          // Serialize primitive data as separate arrays
          std::vector<double> exp_vec(
              ecp_shell.exponents.data(),
              ecp_shell.exponents.data() + ecp_shell.exponents.size());
          std::vector<double> coeff_vec(
              ecp_shell.coefficients.data(),
              ecp_shell.coefficients.data() + ecp_shell.coefficients.size());
          ecp_shell_json["exponents"] = exp_vec;
          ecp_shell_json["coefficients"] = coeff_vec;

          // Serialize radial powers for ECP shells
          if (ecp_shell.rpowers.size() > 0) {
            std::vector<int> rpowers_vec(
                ecp_shell.rpowers.data(),
                ecp_shell.rpowers.data() + ecp_shell.rpowers.size());
            ecp_shell_json["rpowers"] = rpowers_vec;
          }

          atom_json["ecp_shells"].push_back(ecp_shell_json);
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

  return j;
}

std::shared_ptr<BasisSet> BasisSet::from_json(const nlohmann::json& j) {
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    std::string name = j.value("name", "");

    // Load basis type if present, default to spherical
    BasisType basis_type;
    if (j.contains("basis_type")) {
      basis_type = string_to_basis_type(j["basis_type"]);
    } else {
      basis_type = BasisType::Spherical;
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
              auto exp_vec = shell_json["exponents"].get<std::vector<double>>();
              auto coeff_vec =
                  shell_json["coefficients"].get<std::vector<double>>();
              Eigen::VectorXd shell_exponents =
                  Eigen::Map<const Eigen::VectorXd>(exp_vec.data(),
                                                    exp_vec.size());
              Eigen::VectorXd shell_coefficients =
                  Eigen::Map<const Eigen::VectorXd>(coeff_vec.data(),
                                                    coeff_vec.size());

              Shell shell(atom_index,
                          string_to_orbital_type(shell_json["orbital_type"]),
                          shell_exponents, shell_coefficients);
              shells.push_back(shell);
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
              auto exp_vec =
                  ecp_shell_json["exponents"].get<std::vector<double>>();
              auto coeff_vec =
                  ecp_shell_json["coefficients"].get<std::vector<double>>();
              Eigen::VectorXd shell_exponents =
                  Eigen::Map<const Eigen::VectorXd>(exp_vec.data(),
                                                    exp_vec.size());
              Eigen::VectorXd shell_coefficients =
                  Eigen::Map<const Eigen::VectorXd>(coeff_vec.data(),
                                                    coeff_vec.size());

              // Load radial powers if present
              Eigen::VectorXi shell_rpowers;
              if (ecp_shell_json.contains("rpowers") &&
                  ecp_shell_json["rpowers"].is_array()) {
                auto rpowers_vec =
                    ecp_shell_json["rpowers"].get<std::vector<int>>();
                shell_rpowers = Eigen::Map<const Eigen::VectorXi>(
                    rpowers_vec.data(), rpowers_vec.size());
              } else {
                shell_rpowers = Eigen::VectorXi::Zero(0);
              }

              Shell ecp_shell(
                  atom_index,
                  string_to_orbital_type(ecp_shell_json["orbital_type"]),
                  shell_exponents, shell_coefficients, shell_rpowers);
              ecp_shells.push_back(ecp_shell);
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
    // Legacy support - basis functions converted to shells
    else if (j.contains("basis_functions") && j["basis_functions"].is_array()) {
      std::map<std::pair<size_t, OrbitalType>,
               std::vector<std::pair<double, double>>>
          primitive_map;

      for (const auto& bf_json : j["basis_functions"]) {
        size_t atom_index = bf_json["atom_index"];
        OrbitalType orbital_type =
            string_to_orbital_type(bf_json["orbital_type"]);
        auto key = std::make_pair(atom_index, orbital_type);

        // Create primitive list if it doesn't exist
        if (primitive_map.find(key) == primitive_map.end()) {
          primitive_map[key] = std::vector<std::pair<double, double>>();

          // Load primitives (assuming all basis functions in same shell have
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
    if (j.contains("ecp_name") && j.contains("ecp_electrons")) {
      ecp_name = j["ecp_name"];
      ecp_electrons = j["ecp_electrons"].get<std::vector<size_t>>();
    }

    // Create the BasisSet with or without structure
    std::shared_ptr<BasisSet> basis_set;
    if (j.contains("structure")) {
      auto structure = Structure::from_json(j["structure"]);
      if (!ecp_shells.empty()) {
        if (!ecp_name.empty() && !ecp_electrons.empty()) {
          basis_set =
              std::make_shared<BasisSet>(name, shells, ecp_name, ecp_shells,
                                         ecp_electrons, *structure, basis_type);
        } else {
          basis_set = std::make_shared<BasisSet>(name, shells, ecp_shells,
                                                 *structure, basis_type);
        }
      } else {
        basis_set =
            std::make_shared<BasisSet>(name, shells, *structure, basis_type);
      }
    } else {
      if (!ecp_shells.empty()) {
        // Create a minimal structure for ecp_shells constructor
        throw std::runtime_error(
            "Cannot create BasisSet with ECP shells but without structure");
      }
      basis_set = std::make_shared<BasisSet>(name, shells, basis_type);
    }

    return basis_set;

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse BasisSet from JSON: " +
                             std::string(e.what()));
  }
}

std::string BasisSet::orbital_type_to_string(OrbitalType orbital_type) {
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

OrbitalType BasisSet::l_to_orbital_type(int l) {
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

OrbitalType BasisSet::string_to_orbital_type(
    const std::string& orbital_string) {
  std::string lower_str = orbital_string;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 ::tolower);

  if (lower_str == "ul") return OrbitalType::UL;
  if (lower_str == "s") return OrbitalType::S;
  if (lower_str == "p") return OrbitalType::P;
  if (lower_str == "d") return OrbitalType::D;
  if (lower_str == "f") return OrbitalType::F;
  if (lower_str == "g") return OrbitalType::G;
  if (lower_str == "h") return OrbitalType::H;
  if (lower_str == "i") return OrbitalType::I;

  throw std::invalid_argument("Unknown orbital type: " + orbital_string);
}

int BasisSet::get_angular_momentum(OrbitalType orbital_type) {
  return static_cast<int>(orbital_type);
}

int BasisSet::get_num_orbitals_for_l(int l, BasisType basis_type) {
  if (basis_type == BasisType::Spherical) {
    return 2 * l + 1;  // Spherical harmonics
  } else {
    return (l + 1) * (l + 2) / 2;  // Cartesian coordinates
  }
}

std::string BasisSet::basis_type_to_string(BasisType basis_type) {
  switch (basis_type) {
    case BasisType::Spherical:
      return "spherical";
    case BasisType::Cartesian:
      return "cartesian";
    default:
      return "unknown";
  }
}

BasisType BasisSet::string_to_basis_type(const std::string& basis_string) {
  std::string lower_str = basis_string;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 ::tolower);

  if (lower_str == "spherical" || lower_str == "sph")
    return BasisType::Spherical;
  if (lower_str == "cartesian" || lower_str == "cart")
    return BasisType::Cartesian;

  throw std::invalid_argument("Unknown basis type: " + basis_string);
}

void BasisSet::_clear_maps() {
  _cache_valid = false;
  _basis_to_atom_map.clear();
  _basis_to_shell_map.clear();
  _cached_num_basis_functions = 0;
  _cached_num_shells = 0;
}

void BasisSet::_compute_mappings() const {
  if (_cache_valid) {
    return;  // Already computed
  }

  // Clear existing mappings
  _basis_to_atom_map.clear();
  _basis_to_shell_map.clear();
  _cached_num_basis_functions = 0;
  _cached_num_shells = 0;

  // Compute total number of shells and basis functions
  for (const auto& atom_shells : _shells_per_atom) {
    _cached_num_shells += atom_shells.size();
    for (const auto& shell : atom_shells) {
      _cached_num_basis_functions += shell.get_num_basis_functions(_basis_type);
    }
  }

  // Reserve space for mappings
  _basis_to_atom_map.reserve(_cached_num_basis_functions);
  _basis_to_shell_map.reserve(_cached_num_basis_functions);

  // Build mappings
  size_t current_shell_idx = 0;
  for (size_t atom_idx = 0; atom_idx < _shells_per_atom.size(); ++atom_idx) {
    const auto& atom_shells = _shells_per_atom[atom_idx];

    for (const auto& shell : atom_shells) {
      size_t num_bf = shell.get_num_basis_functions(_basis_type);

      // Map each basis function in this shell to the atom and shell
      for (size_t bf_idx = 0; bf_idx < num_bf; ++bf_idx) {
        _basis_to_atom_map.push_back(atom_idx);
        _basis_to_shell_map.push_back(current_shell_idx);
      }

      current_shell_idx++;
    }
  }

  _cache_valid = true;
}

void BasisSet::_validate_basis_index(size_t basis_index) const {
  if (basis_index >= get_num_basis_functions()) {
    throw std::out_of_range("Basis function index " +
                            std::to_string(basis_index) +
                            " is out of range. Maximum index: " +
                            std::to_string(get_num_basis_functions() - 1));
  }
}

void BasisSet::_validate_shell_index(size_t shell_index) const {
  if (shell_index >= get_num_shells()) {
    throw std::out_of_range("Shell index " + std::to_string(shell_index) +
                            " is out of range. Maximum index: " +
                            std::to_string(get_num_shells() - 1));
  }
}

void BasisSet::_validate_atom_index(size_t atom_index) const {
  if (atom_index >= _shells_per_atom.size()) {
    throw std::out_of_range("Atom index " + std::to_string(atom_index) +
                            " is out of range. Maximum index: " +
                            std::to_string(_shells_per_atom.size() - 1));
  }
}

std::pair<size_t, int> BasisSet::basis_to_shell_index(
    size_t basis_index) const {
  if (!_cache_valid) {
    _compute_mappings();
  }

  _validate_basis_index(basis_index);

  size_t shell_index = _basis_to_shell_map[basis_index];

  // Find the offset within the shell to compute magnetic quantum number
  size_t basis_offset_in_shell = 0;
  size_t current_basis_idx = 0;
  size_t current_shell_idx = 0;

  for (const auto& atom_shells : _shells_per_atom) {
    for (const auto& shell : atom_shells) {
      size_t num_bf = shell.get_num_basis_functions(_basis_type);

      if (current_shell_idx == shell_index) {
        // Found the shell, compute the offset
        basis_offset_in_shell = basis_index - current_basis_idx;
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

}  // namespace qdk::chemistry::data
