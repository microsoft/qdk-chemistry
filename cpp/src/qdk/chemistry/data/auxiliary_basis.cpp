// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "basis_library.hpp"
#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

namespace {

void clear_legacy_radial_powers(std::vector<Shell>& shells) {
  for (auto& shell : shells) {
    shell.rpowers.resize(0);
  }
}

}  // namespace

std::string to_string(const AuxiliaryBasisRole role) {
  switch (role) {
    case AuxiliaryBasisRole::JFit:
      return "jfit";
    case AuxiliaryBasisRole::JKFit:
      return "jkfit";
    case AuxiliaryBasisRole::RIFit:
      return "rifit";
    case AuxiliaryBasisRole::CABS:
      return "cabs";
  }
  throw std::invalid_argument("Unknown AuxiliaryBasisRole value");
}

AuxiliaryBasisRole auxiliary_basis_role_from_string(std::string role) {
  if (role == "jfit") {
    return AuxiliaryBasisRole::JFit;
  }
  if (role == "jkfit") {
    return AuxiliaryBasisRole::JKFit;
  }
  if (role == "rifit") {
    return AuxiliaryBasisRole::RIFit;
  }
  if (role == "cabs") {
    return AuxiliaryBasisRole::CABS;
  }
  throw std::invalid_argument("Unknown auxiliary-basis role: " + role);
}

AuxiliaryBasis::AuxiliaryBasis(std::vector<Shell> shells,
                               std::shared_ptr<Structure> structure,
                               const AOType atomic_orbital_type)
    : AuxiliaryBasis(std::string(custom_name), std::move(shells),
                     std::move(structure), atomic_orbital_type) {}

AuxiliaryBasis::AuxiliaryBasis(std::string name, std::vector<Shell> shells,
                               std::shared_ptr<Structure> structure,
                               const AOType atomic_orbital_type)
    : _name(std::move(name)),
      _atomic_orbital_type(atomic_orbital_type),
      _structure(std::move(structure)) {
  if (_name.empty()) {
    throw std::invalid_argument("AuxiliaryBasis name cannot be empty");
  }
  if (!_structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }
  if (shells.empty()) {
    throw std::invalid_argument("AuxiliaryBasis shells cannot be empty");
  }
  for (const auto& shell : shells) {
    if (shell.has_radial_powers()) {
      throw std::invalid_argument(
          "Auxiliary shells cannot contain radial powers");
    }
    if (shell.orbital_type == OrbitalType::UL) {
      throw std::invalid_argument(
          "Auxiliary shells cannot use the ECP local-potential orbital type");
    }
    if (shell.exponents.size() == 0 || shell.coefficients.size() == 0 ||
        shell.exponents.size() != shell.coefficients.size()) {
      throw std::invalid_argument("AuxiliaryBasis contains an invalid shell");
    }
  }

  _shells_per_atom = group_shells_by_atom(shells, _structure->get_num_atoms(),
                                          "Auxiliary shell");
  for (auto& atom_shells : _shells_per_atom) {
    sort_shells_inplace(atom_shells);
  }
  _validate();
}

const std::string& AuxiliaryBasis::get_name() const { return _name; }

AOType AuxiliaryBasis::get_atomic_orbital_type() const {
  return _atomic_orbital_type;
}

std::shared_ptr<Structure> AuxiliaryBasis::get_structure() const {
  return _structure;
}

std::vector<Shell> AuxiliaryBasis::get_shells() const {
  return flatten_shells(_shells_per_atom);
}

const std::vector<Shell>& AuxiliaryBasis::get_shells_for_atom(
    const size_t atom_index) const {
  if (atom_index >= _shells_per_atom.size()) {
    throw std::out_of_range("Atom index " + std::to_string(atom_index) +
                            " is out of range");
  }
  return _shells_per_atom[atom_index];
}

const Shell& AuxiliaryBasis::get_shell(const size_t shell_index) const {
  size_t offset = 0;
  for (const auto& atom_shells : _shells_per_atom) {
    if (shell_index < offset + atom_shells.size()) {
      return atom_shells[shell_index - offset];
    }
    offset += atom_shells.size();
  }
  throw std::out_of_range("Auxiliary shell index " +
                          std::to_string(shell_index) + " is out of range");
}

size_t AuxiliaryBasis::get_num_shells() const {
  return count_shells(_shells_per_atom);
}

size_t AuxiliaryBasis::get_num_atoms() const { return _shells_per_atom.size(); }

size_t AuxiliaryBasis::get_num_auxiliary_orbitals() const {
  return count_orbitals(_shells_per_atom, _atomic_orbital_type);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_basis_name(
    std::string basis_name, std::shared_ptr<Structure> structure,
    const AOType atomic_orbital_type) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }
  basis_name = detail::lowercase_basis_name(std::move(basis_name));

  std::vector<Shell> shells;
  const auto nuclear_charges = structure->get_nuclear_charges();
  for (size_t atom_index = 0; atom_index < nuclear_charges.size();
       ++atom_index) {
    auto [atom_shells, ignored_ecp_shells, ignored_ecp_electrons] =
        detail::get_basis_for_nuclear_charge(nuclear_charges[atom_index],
                                             basis_name, atom_index);
    clear_legacy_radial_powers(atom_shells);
    shells.insert(shells.end(), atom_shells.begin(), atom_shells.end());
  }
  return std::make_shared<AuxiliaryBasis>(
      basis_name, std::move(shells), std::move(structure), atomic_orbital_type);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_element_map(
    const std::map<std::string, std::string>& element_to_basis_map,
    std::shared_ptr<Structure> structure, const AOType atomic_orbital_type) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }
  std::map<size_t, std::string> index_map;
  const auto symbols = structure->get_atomic_symbols();
  for (size_t atom_index = 0; atom_index < symbols.size(); ++atom_index) {
    auto basis = element_to_basis_map.find(symbols[atom_index]);
    if (basis == element_to_basis_map.end()) {
      throw std::invalid_argument("No auxiliary basis specified for element: " +
                                  symbols[atom_index]);
    }
    index_map.emplace(atom_index, basis->second);
  }
  return from_index_map(index_map, std::move(structure), atomic_orbital_type);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_index_map(
    const std::map<size_t, std::string>& index_to_basis_map,
    std::shared_ptr<Structure> structure, const AOType atomic_orbital_type) {
  if (!structure) {
    throw std::invalid_argument("Structure shared_ptr cannot be nullptr");
  }

  std::vector<Shell> shells;
  const auto nuclear_charges = structure->get_nuclear_charges();
  for (size_t atom_index = 0; atom_index < nuclear_charges.size();
       ++atom_index) {
    auto basis = index_to_basis_map.find(atom_index);
    if (basis == index_to_basis_map.end()) {
      throw std::invalid_argument(
          "No auxiliary basis specified for atom index: " +
          std::to_string(atom_index));
    }
    std::string basis_name = detail::lowercase_basis_name(basis->second);
    auto [atom_shells, ignored_ecp_shells, ignored_ecp_electrons] =
        detail::get_basis_for_nuclear_charge(nuclear_charges[atom_index],
                                             basis_name, atom_index);
    clear_legacy_radial_powers(atom_shells);
    shells.insert(shells.end(), atom_shells.begin(), atom_shells.end());
  }

  return std::make_shared<AuxiliaryBasis>(
      std::string(custom_name), std::move(shells), std::move(structure),
      atomic_orbital_type);
}

std::string AuxiliaryBasis::get_summary() const {
  std::ostringstream output;
  output << "AuxiliaryBasis: " << _name << "\n"
         << "Basis type: "
         << atomic_orbital_type_to_string(_atomic_orbital_type) << "\n"
         << "Number of atoms: " << get_num_atoms() << "\n"
         << "Number of shells: " << get_num_shells() << "\n"
         << "Number of auxiliary orbitals: " << get_num_auxiliary_orbitals();
  return output.str();
}

void AuxiliaryBasis::to_file(const std::string& filename,
                             const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type);
  }
}

nlohmann::json AuxiliaryBasis::to_json() const {
  nlohmann::json json;
  json["version"] = SERIALIZATION_VERSION;
  json["name"] = _name;
  json["atomic_orbital_type"] =
      atomic_orbital_type_to_string(_atomic_orbital_type);
  json["structure"] = _structure->to_json();
  json["atoms"] = nlohmann::json::array();
  for (size_t atom_index = 0; atom_index < _shells_per_atom.size();
       ++atom_index) {
    if (_shells_per_atom[atom_index].empty()) {
      continue;
    }
    nlohmann::json atom_json;
    atom_json["atom_index"] = atom_index;
    atom_json["shells"] = nlohmann::json::array();
    for (const auto& shell : _shells_per_atom[atom_index]) {
      atom_json["shells"].push_back(shell.to_json());
    }
    json["atoms"].push_back(std::move(atom_json));
  }
  return json;
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_json(
    const nlohmann::json& json) {
  if (!json.contains("version")) {
    throw std::runtime_error("Invalid JSON: missing version field");
  }
  validate_serialization_version(SERIALIZATION_VERSION, json["version"]);
  if (!json.contains("structure")) {
    throw std::runtime_error("AuxiliaryBasis JSON is missing structure data");
  }

  auto structure = Structure::from_json(json["structure"]);
  std::vector<Shell> shells;
  for (const auto& atom_json : json.at("atoms")) {
    size_t atom_index = atom_json.at("atom_index");
    for (const auto& shell_json : atom_json.at("shells")) {
      shells.push_back(Shell::from_json(shell_json, atom_index));
    }
  }
  return std::make_shared<AuxiliaryBasis>(
      json.at("name").get<std::string>(), std::move(shells),
      std::move(structure),
      string_to_atomic_orbital_type(
          json.value("atomic_orbital_type", "spherical")));
}

void AuxiliaryBasis::to_json_file(const std::string& filename) const {
  std::string validated = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasis));
  std::ofstream output(validated);
  if (!output) {
    throw std::runtime_error("Unable to open file for writing: " + validated);
  }
  output << to_json().dump(2);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_json_file(
    const std::string& filename) {
  std::string validated = DataTypeFilename::validate_read_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasis));
  std::ifstream input(validated);
  if (!input) {
    throw std::runtime_error("Unable to open file for reading: " + validated);
  }
  nlohmann::json json;
  input >> json;
  return from_json(json);
}

void AuxiliaryBasis::to_hdf5(H5::Group& group) const {
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  H5::Attribute attribute =
      group.createAttribute("json", string_type, scalar_space);
  std::string json = to_json().dump();
  attribute.write(string_type, json);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_hdf5(H5::Group& group) {
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute attribute = group.openAttribute("json");
  std::string json;
  attribute.read(string_type, json);
  return from_json(nlohmann::json::parse(json));
}

void AuxiliaryBasis::to_hdf5_file(const std::string& filename) const {
  std::string validated = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasis));
  H5::H5File file(validated, H5F_ACC_TRUNC);
  H5::Group group = file.createGroup("/auxiliary_basis");
  to_hdf5(group);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_hdf5_file(
    const std::string& filename) {
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }
  std::string validated = DataTypeFilename::validate_read_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasis));
  H5::H5File file(validated, H5F_ACC_RDONLY);
  H5::Group group = file.openGroup("/auxiliary_basis");
  return from_hdf5(group);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasis::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::runtime_error("Unsupported file type: " + type);
}

void AuxiliaryBasis::_validate() const {
  if (!_structure || _name.empty() || get_num_shells() == 0) {
    throw std::invalid_argument("Tried to generate invalid AuxiliaryBasis");
  }
}

void AuxiliaryBasis::hash_update(
    qdk::chemistry::utils::HashContext& context) const {
  using qdk::chemistry::utils::hash_value;
  hash_value(context, get_data_type_name());
  hash_value(context, _name);
  hash_value(context, static_cast<int64_t>(_atomic_orbital_type));
  hash_value(context, _structure->content_hash());
  hash_value(context, static_cast<uint64_t>(_shells_per_atom.size()));
  for (const auto& atom_shells : _shells_per_atom) {
    hash_value(context, static_cast<uint64_t>(atom_shells.size()));
    for (const auto& shell : atom_shells) {
      hash_value(context, static_cast<uint64_t>(shell.atom_index));
      hash_value(context, static_cast<int64_t>(shell.orbital_type));
      hash_value(context, shell.exponents);
      hash_value(context, shell.coefficients);
    }
  }
}

AuxiliaryBasisCollection::AuxiliaryBasisCollection(Map auxiliary_bases)
    : _auxiliary_bases(std::move(auxiliary_bases)) {
  _validate();
}

bool AuxiliaryBasisCollection::has_auxiliary_basis(
    const AuxiliaryBasisRole role) const {
  return _auxiliary_bases.contains(role);
}

std::shared_ptr<AuxiliaryBasis> AuxiliaryBasisCollection::get_auxiliary_basis(
    const AuxiliaryBasisRole role) const {
  auto basis = _auxiliary_bases.find(role);
  if (basis == _auxiliary_bases.end()) {
    throw std::out_of_range("No auxiliary basis is associated with role " +
                            to_string(role));
  }
  return basis->second;
}

std::shared_ptr<AuxiliaryBasis>
AuxiliaryBasisCollection::resolve_auxiliary_basis(
    const AuxiliaryBasisRole role) const {
  if (has_auxiliary_basis(role)) {
    return get_auxiliary_basis(role);
  }
  if (role == AuxiliaryBasisRole::JFit &&
      has_auxiliary_basis(AuxiliaryBasisRole::JKFit)) {
    return get_auxiliary_basis(AuxiliaryBasisRole::JKFit);
  }
  throw std::out_of_range("No auxiliary basis can satisfy role " +
                          to_string(role));
}

const AuxiliaryBasisCollection::Map&
AuxiliaryBasisCollection::get_auxiliary_bases() const {
  return _auxiliary_bases;
}

std::string AuxiliaryBasisCollection::get_summary() const {
  std::ostringstream output;
  output << "AuxiliaryBasisCollection: " << _auxiliary_bases.size()
         << " role(s)";
  for (const auto& [role, basis] : _auxiliary_bases) {
    output << "\n" << to_string(role) << ": " << basis->get_name();
  }
  return output.str();
}

void AuxiliaryBasisCollection::to_file(const std::string& filename,
                                       const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type);
  }
}

nlohmann::json AuxiliaryBasisCollection::to_json() const {
  nlohmann::json json;
  json["version"] = SERIALIZATION_VERSION;
  json["auxiliary_bases"] = nlohmann::json::object();
  for (const auto& [role, basis] : _auxiliary_bases) {
    json["auxiliary_bases"][to_string(role)] = basis->to_json();
  }
  return json;
}

std::shared_ptr<AuxiliaryBasisCollection> AuxiliaryBasisCollection::from_json(
    const nlohmann::json& json) {
  if (!json.contains("version")) {
    throw std::runtime_error("Invalid JSON: missing version field");
  }
  validate_serialization_version(SERIALIZATION_VERSION, json["version"]);
  if (!json.contains("auxiliary_bases") ||
      !json["auxiliary_bases"].is_object()) {
    throw std::runtime_error(
        "AuxiliaryBasisCollection JSON must contain an auxiliary_bases "
        "object");
  }

  Map auxiliary_bases;
  for (const auto& [role_name, basis_json] : json["auxiliary_bases"].items()) {
    const auto role = auxiliary_basis_role_from_string(role_name);
    auxiliary_bases.emplace(role, AuxiliaryBasis::from_json(basis_json));
  }
  return std::make_shared<AuxiliaryBasisCollection>(std::move(auxiliary_bases));
}

void AuxiliaryBasisCollection::to_json_file(const std::string& filename) const {
  std::string validated = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasisCollection));
  std::ofstream output(validated);
  if (!output) {
    throw std::runtime_error("Unable to open file for writing: " + validated);
  }
  output << to_json().dump(2);
}

std::shared_ptr<AuxiliaryBasisCollection>
AuxiliaryBasisCollection::from_json_file(const std::string& filename) {
  std::string validated = DataTypeFilename::validate_read_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasisCollection));
  std::ifstream input(validated);
  if (!input) {
    throw std::runtime_error("Unable to open file for reading: " + validated);
  }
  nlohmann::json json;
  input >> json;
  return from_json(json);
}

void AuxiliaryBasisCollection::to_hdf5(H5::Group& group) const {
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  H5::Attribute attribute =
      group.createAttribute("json", string_type, scalar_space);
  std::string json = to_json().dump();
  attribute.write(string_type, json);
}

std::shared_ptr<AuxiliaryBasisCollection> AuxiliaryBasisCollection::from_hdf5(
    H5::Group& group) {
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute attribute = group.openAttribute("json");
  std::string json;
  attribute.read(string_type, json);
  return from_json(nlohmann::json::parse(json));
}

void AuxiliaryBasisCollection::to_hdf5_file(const std::string& filename) const {
  std::string validated = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasisCollection));
  H5::H5File file(validated, H5F_ACC_TRUNC);
  H5::Group group = file.createGroup("/auxiliary_basis_collection");
  to_hdf5(group);
}

std::shared_ptr<AuxiliaryBasisCollection>
AuxiliaryBasisCollection::from_hdf5_file(const std::string& filename) {
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }
  std::string validated = DataTypeFilename::validate_read_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(AuxiliaryBasisCollection));
  H5::H5File file(validated, H5F_ACC_RDONLY);
  H5::Group group = file.openGroup("/auxiliary_basis_collection");
  return from_hdf5(group);
}

std::shared_ptr<AuxiliaryBasisCollection> AuxiliaryBasisCollection::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::runtime_error("Unsupported file type: " + type);
}

void AuxiliaryBasisCollection::_validate() const {
  std::string structure_hash;
  for (const auto& [role, basis] : _auxiliary_bases) {
    if (!basis) {
      throw std::invalid_argument("Auxiliary basis for role " +
                                  to_string(role) + " cannot be null");
    }
    const auto basis_structure_hash = basis->get_structure()->content_hash();
    if (structure_hash.empty()) {
      structure_hash = basis_structure_hash;
    } else if (basis_structure_hash != structure_hash) {
      throw std::invalid_argument(
          "All auxiliary bases in a collection must describe the same "
          "molecular structure");
    }
  }
}

void AuxiliaryBasisCollection::hash_update(
    qdk::chemistry::utils::HashContext& context) const {
  using qdk::chemistry::utils::hash_value;
  hash_value(context, get_data_type_name());
  hash_value(context, static_cast<uint64_t>(_auxiliary_bases.size()));
  for (const auto& [role, basis] : _auxiliary_bases) {
    hash_value(context, to_string(role));
    hash_value(context, basis->content_hash());
  }
}

std::shared_ptr<AuxiliaryBasisCollection> with_auxiliary_basis(
    const AuxiliaryBasisCollection& collection, const AuxiliaryBasisRole role,
    std::shared_ptr<AuxiliaryBasis> auxiliary_basis) {
  auto auxiliary_bases = collection.get_auxiliary_bases();
  auxiliary_bases[role] = std::move(auxiliary_basis);
  return std::make_shared<AuxiliaryBasisCollection>(std::move(auxiliary_bases));
}

}  // namespace qdk::chemistry::data
