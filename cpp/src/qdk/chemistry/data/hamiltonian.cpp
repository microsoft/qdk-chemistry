// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

HamiltonianContainer::HamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix, HamiltonianType type)
    : _one_body_integrals(
          make_restricted_one_body_integrals(one_body_integrals)),
      _inactive_fock_matrix(
          make_restricted_inactive_fock_matrix(inactive_fock_matrix)),
      _orbitals(orbitals),
      _core_energy(core_energy),
      _type(type) {
  QDK_LOG_TRACE_ENTERING();
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  // Validate that orbitals have the necessary data
  if (!orbitals->has_active_space()) {
    throw std::runtime_error(
        "Orbitals must have an active space set for HamiltonianContainer");
  }
}

HamiltonianContainer::HamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals_alpha,
    const Eigen::MatrixXd& one_body_integrals_beta,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix_alpha,
    const Eigen::MatrixXd& inactive_fock_matrix_beta, HamiltonianType type)
    : _one_body_integrals(
          std::make_unique<Eigen::MatrixXd>(one_body_integrals_alpha),
          std::make_unique<Eigen::MatrixXd>(one_body_integrals_beta)),
      _inactive_fock_matrix(
          std::make_unique<Eigen::MatrixXd>(inactive_fock_matrix_alpha),
          std::make_unique<Eigen::MatrixXd>(inactive_fock_matrix_beta)),
      _orbitals(orbitals),
      _core_energy(core_energy),
      _type(type) {
  QDK_LOG_TRACE_ENTERING();
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  // Validate that orbitals have the necessary data
  if (!orbitals->has_active_space()) {
    throw std::runtime_error(
        "Orbitals must have an active space set for HamiltonianContainer");
  }
}

std::tuple<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
HamiltonianContainer::get_one_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_one_body_integrals()) {
    throw std::runtime_error("One-body integrals are not set");
  }
  return std::make_tuple(std::cref(*_one_body_integrals.first),
                         std::cref(*_one_body_integrals.second));
}

bool HamiltonianContainer::has_one_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _one_body_integrals.first != nullptr &&
         _one_body_integrals.first->rows() > 0 &&
         _one_body_integrals.first->cols() > 0;
}

double HamiltonianContainer::get_one_body_element(unsigned i, unsigned j,
                                                  SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_one_body_integrals()) {
    throw std::runtime_error("One-body integrals are not set");
  }

  size_t norb = _orbitals->get_active_space_indices().first.size();
  if (i >= norb || j >= norb) {
    throw std::out_of_range("Orbital index out of range");
  }

  // Select the appropriate integral based on spin channel
  switch (channel) {
    case SpinChannel::aa:
      return (*_one_body_integrals.first)(i, j);
    case SpinChannel::bb:
      return (*_one_body_integrals.second)(i, j);
    default:
      throw std::invalid_argument(
          "Invalid spin channel for one-body integrals");
  }
}

bool HamiltonianContainer::has_inactive_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  bool has_alpha = _inactive_fock_matrix.first != nullptr &&
                   _inactive_fock_matrix.first->size() > 0;
  bool has_beta = _inactive_fock_matrix.second != nullptr &&
                  _inactive_fock_matrix.second->size() > 0;
  return has_alpha && has_beta;
}

std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
HamiltonianContainer::get_inactive_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_inactive_fock_matrix()) {
    throw std::runtime_error("Inactive Fock matrix is not set");
  }
  return {*_inactive_fock_matrix.first, *_inactive_fock_matrix.second};
}

const std::shared_ptr<Orbitals> HamiltonianContainer::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_orbitals()) {
    throw std::runtime_error("Orbitals are not set");
  }
  return _orbitals;
}

bool HamiltonianContainer::has_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _orbitals != nullptr;
}

double HamiltonianContainer::get_core_energy() const {
  QDK_LOG_TRACE_ENTERING();
  return _core_energy;
}

HamiltonianType HamiltonianContainer::get_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _type;
}

bool HamiltonianContainer::is_hermitian() const {
  QDK_LOG_TRACE_ENTERING();
  return _type == HamiltonianType::Hermitian;
}

bool HamiltonianContainer::is_unrestricted() const {
  QDK_LOG_TRACE_ENTERING();
  return !is_restricted();
}

void HamiltonianContainer::validate_integral_dimensions() const {
  // assume the container base has one-body integrals only, expect derived
  // classes to work on the two-body integrals

  QDK_LOG_TRACE_ENTERING();
  if (!has_one_body_integrals()) {
    return;
  }

  // Check alpha one-body integrals
  size_t norb_alpha = _one_body_integrals.first->rows();
  size_t norb_alpha_cols = _one_body_integrals.first->cols();

  if (norb_alpha != norb_alpha_cols) {
    throw std::invalid_argument(
        "Alpha one-body integrals matrix must be square");
  }

  // Check beta one-body integrals (if different from alpha)
  if (_one_body_integrals.second != _one_body_integrals.first) {
    size_t norb_beta = _one_body_integrals.second->rows();
    size_t norb_beta_cols = _one_body_integrals.second->cols();

    if (norb_beta != norb_beta_cols) {
      throw std::invalid_argument(
          "Beta one-body integrals matrix must be square");
    }

    if (norb_beta != norb_alpha) {
      throw std::invalid_argument(
          "Alpha and beta one-body integrals must have same dimensions");
    }
  }
}

void HamiltonianContainer::validate_restrictedness_consistency() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_orbitals) return;

  bool orbitals_restricted = _orbitals->is_restricted();
  bool hamiltonian_restricted = is_restricted();

  if (orbitals_restricted != hamiltonian_restricted) {
    throw std::invalid_argument(
        "Hamiltonian restrictedness (" +
        std::string(hamiltonian_restricted ? "restricted" : "unrestricted") +
        ") must match orbitals restrictedness (" +
        std::string(orbitals_restricted ? "restricted" : "unrestricted") + ")");
  }
}

void HamiltonianContainer::validate_active_space_dimensions() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_orbitals || !_orbitals->has_active_space()) return;

  auto active_indices = _orbitals->get_active_space_indices();
  size_t n_active_alpha = active_indices.first.size();
  size_t n_active_beta = active_indices.second.size();

  // Check one-body integrals dimensions match active space
  if (has_one_body_integrals()) {
    if (_one_body_integrals.first->rows() != n_active_alpha) {
      throw std::invalid_argument(
          "Alpha one-body integrals dimension (" +
          std::to_string(_one_body_integrals.first->rows()) +
          ") does not match number of alpha active orbitals (" +
          std::to_string(n_active_alpha) + ")");
    }

    if (_one_body_integrals.second != _one_body_integrals.first &&
        _one_body_integrals.second->rows() != n_active_beta) {
      throw std::invalid_argument(
          "Beta one-body integrals dimension does not match number of beta "
          "active orbitals");
    }
  }

  // For restricted case, alpha and beta active spaces should be the same
  if (is_restricted() && n_active_alpha != n_active_beta) {
    throw std::invalid_argument(
        "For restricted Hamiltonian, alpha and beta active spaces must have "
        "same size");
  }
}

std::pair<std::shared_ptr<Eigen::MatrixXd>, std::shared_ptr<Eigen::MatrixXd>>
HamiltonianContainer::make_restricted_one_body_integrals(
    const Eigen::MatrixXd& integrals) {
  QDK_LOG_TRACE_ENTERING();
  auto shared_integrals = std::make_shared<Eigen::MatrixXd>(integrals);
  return std::make_pair(
      shared_integrals,
      shared_integrals);  // Both alpha and beta point to same data
}

std::pair<std::shared_ptr<Eigen::MatrixXd>, std::shared_ptr<Eigen::MatrixXd>>
HamiltonianContainer::make_restricted_inactive_fock_matrix(
    const Eigen::MatrixXd& matrix) {
  QDK_LOG_TRACE_ENTERING();
  auto shared_matrix = std::make_shared<Eigen::MatrixXd>(matrix);
  return std::make_pair(
      shared_matrix, shared_matrix);  // Both alpha and beta point to same data
}

std::string Hamiltonian::get_summary() const {
  QDK_LOG_TRACE_ENTERING();
  std::string summary = "Hamiltonian Summary:\n";
  size_t num_molecular_orbitals = get_orbitals()->get_num_molecular_orbitals();
  size_t norb = get_orbitals()->get_active_space_indices().first.size();
  summary += "  Type: ";
  summary += (is_hermitian() ? "Hermitian" : "NonHermitian");
  summary += "\n";
  summary += "  Restrictedness: ";
  summary += (is_restricted() ? "Restricted" : "Unrestricted");
  summary += "\n";
  summary += "  Active Orbitals: " + std::to_string(norb) + "\n";
  summary +=
      "  Total Orbitals: " + std::to_string(num_molecular_orbitals) + "\n";

  const double threshold = 1e-6;  // Threshold for determining negligible
                                  // integrals in summary statistics

  summary += "  Core Energy: " + std::to_string(get_core_energy()) + "\n";
  summary += "  Integral Statistics:\n";

  // One-body integrals - alpha
  auto [one_body_alpha, one_body_beta] = get_one_body_integrals();
  const size_t non_negligible_one_body_alpha = std::count_if(
      one_body_alpha.data(), one_body_alpha.data() + one_body_alpha.size(),
      [threshold](double val) { return std::abs(val) > threshold; });

  summary += "    One-body Integrals (alpha): " +
             std::to_string(one_body_alpha.size()) + " (larger than " +
             std::to_string(threshold) + ": " +
             std::to_string(non_negligible_one_body_alpha) + ")\n";

  // One-body integrals - beta (if unrestricted)
  if (is_unrestricted()) {
    const size_t non_negligible_one_body_beta = std::count_if(
        one_body_beta.data(), one_body_beta.data() + one_body_beta.size(),
        [threshold](double val) { return std::abs(val) > threshold; });

    summary += "    One-body Integrals (beta): " +
               std::to_string(one_body_beta.size()) + " (larger than " +
               std::to_string(threshold) + ": " +
               std::to_string(non_negligible_one_body_beta) + ")\n";
  }

  // Two-body integrals - aaaa
  const auto& two_body_aaaa = std::get<0>(get_two_body_integrals());
  const size_t non_negligible_two_body_aaaa = std::count_if(
      two_body_aaaa.data(), two_body_aaaa.data() + two_body_aaaa.size(),
      [threshold](double val) { return std::abs(val) > threshold; });

  summary +=
      "    Two-body Integrals (aaaa): " + std::to_string(two_body_aaaa.size()) +
      " (larger than " + std::to_string(threshold) + ": " +
      std::to_string(non_negligible_two_body_aaaa) + ")\n";

  // Two-body integrals - aabb and bbbb (if unrestricted)
  if (is_unrestricted()) {
    const auto& two_body_aabb = std::get<1>(get_two_body_integrals());
    const size_t non_negligible_two_body_aabb = std::count_if(
        two_body_aabb.data(), two_body_aabb.data() + two_body_aabb.size(),
        [threshold](double val) { return std::abs(val) > threshold; });

    summary += "    Two-body Integrals (aabb): " +
               std::to_string(two_body_aabb.size()) + " (larger than " +
               std::to_string(threshold) + ": " +
               std::to_string(non_negligible_two_body_aabb) + ")\n";

    const auto& two_body_bbbb = std::get<2>(get_two_body_integrals());
    const size_t non_negligible_two_body_bbbb = std::count_if(
        two_body_bbbb.data(), two_body_bbbb.data() + two_body_bbbb.size(),
        [threshold](double val) { return std::abs(val) > threshold; });

    summary += "    Two-body Integrals (bbbb): " +
               std::to_string(two_body_bbbb.size()) + " (larger than " +
               std::to_string(threshold) + ": " +
               std::to_string(non_negligible_two_body_bbbb) + ")\n";
  }

  return summary;
}

std::unique_ptr<HamiltonianContainer> HamiltonianContainer::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  if (!j.contains("container_type")) {
    throw std::runtime_error("JSON missing required 'container_type' field");
  }

  std::string container_type = j["container_type"];

  // Forward to appropriate container implementation
  if (container_type == "canonical_four_center") {
    return CanonicalFourCenterHamiltonianContainer::from_json(j);
  } else {
    throw std::runtime_error("Unknown container type: " + container_type);
  }
}

std::unique_ptr<HamiltonianContainer> HamiltonianContainer::from_hdf5(
    H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    // Read container type identifier
    if (!group.attrExists("container_type")) {
      throw std::runtime_error(
          "HDF5 group missing required 'container_type' attribute");
    }

    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute type_attr = group.openAttribute("container_type");
    std::string container_type;
    type_attr.read(string_type, container_type);

    // Forward to appropriate container implementation
    if (container_type == "canonical_four_center") {
      return CanonicalFourCenterHamiltonianContainer::from_hdf5(group);
    } else {
      throw std::runtime_error("Unknown container type: " + container_type);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Hamiltonian::to_file(const std::string& filename,
                          const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();
  if (type == "json") {
    _to_json_file(filename);
  } else if (type == "hdf5") {
    _to_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5, fcidump");
  }
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_file(const std::string& filename,
                                                    const std::string& type) {
  QDK_LOG_TRACE_ENTERING();
  if (type == "json") {
    return _from_json_file(filename);
  } else if (type == "hdf5") {
    return _from_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

void Hamiltonian::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  // Validate filename has correct data type suffix
  std::string validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(Hamiltonian));

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "hamiltonian");

  return _from_hdf5_file(validated_filename);
}

void Hamiltonian::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  // Validate filename has correct data type suffix
  std::string validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(Hamiltonian));

  _to_json_file(validated_filename);
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "hamiltonian");

  return _from_json_file(validated_filename);
}

void Hamiltonian::to_fcidump_file(const std::string& filename, size_t nalpha,
                                  size_t nbeta) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  // Validate filename has correct data type suffix
  std::string validated_filename = DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(Hamiltonian));

  _container->to_fcidump_file(validated_filename, nalpha, nbeta);
}

void Hamiltonian::_to_json_file(const std::string& filename) const {
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

std::shared_ptr<Hamiltonian> Hamiltonian::_from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Hamiltonian JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json j;
  file >> j;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(j);
}

nlohmann::json Hamiltonian::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Delegate to container serialization (orbitals are included within the
  // container)
  j["container"] = _container->to_json();

  return j;
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load container using factory method (orbitals are loaded internally by
    // the container)
    if (!j.contains("container")) {
      throw std::runtime_error("JSON missing required 'container' field");
    }

    auto container = HamiltonianContainer::from_json(j["container"]);
    return std::make_shared<Hamiltonian>(std::move(container));

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Hamiltonian from JSON: " +
                             std::string(e.what()));
  }
}

void Hamiltonian::_to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (!_container->is_valid()) {
    throw std::runtime_error("Cannot save invalid Hamiltonian data to HDF5");
  }

  try {
    // Create HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Use the group-based serialization method
    to_hdf5(file);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Hamiltonian::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Add version attribute
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);

    // Delegate to container serialization (orbitals are included within the
    // container)
    H5::Group container_group = group.createGroup("container");
    _container->to_hdf5(container_group);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Load container using factory method (orbitals are loaded internally by
    // the container)
    if (!group.nameExists("container")) {
      throw std::runtime_error(
          "HDF5 group missing required 'container' subgroup");
    }
    H5::Group container_group = group.openGroup("container");
    auto container = HamiltonianContainer::from_hdf5(container_group);

    return std::make_shared<Hamiltonian>(std::move(container));

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Hamiltonian> Hamiltonian::_from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

  H5::H5File file;
  try {
    // Open HDF5 file
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Hamiltonian HDF5 file '" +
                             filename + "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    // Use the group-based deserialization method
    return from_hdf5(file);
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "Unable to read Hamiltonian data from HDF5 file '" + filename + "'. " +
        "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

Hamiltonian::Hamiltonian(std::unique_ptr<HamiltonianContainer> container)
    : _container(std::move(container)) {
  QDK_LOG_TRACE_ENTERING();
}

// Copy constructor
Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : _container(other._container->clone()) {
  QDK_LOG_TRACE_ENTERING();
}

// Copy assignment operator
Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
  QDK_LOG_TRACE_ENTERING();
  if (this != &other) {
    _container = other._container->clone();
  }
  return *this;
}

std::tuple<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
Hamiltonian::get_one_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_one_body_integrals();
}

double Hamiltonian::get_one_body_element(unsigned i, unsigned j,
                                         SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_one_body_element(i, j, channel);
}

bool Hamiltonian::has_one_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_one_body_integrals();
}

std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&>
Hamiltonian::get_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_two_body_integrals();
}

double Hamiltonian::get_two_body_element(unsigned i, unsigned j, unsigned k,
                                         unsigned l,
                                         SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_two_body_element(i, j, k, l, channel);
}

bool Hamiltonian::has_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_two_body_integrals();
}

std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
Hamiltonian::get_inactive_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_inactive_fock_matrix();
}

bool Hamiltonian::has_inactive_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_inactive_fock_matrix();
}

const std::shared_ptr<Orbitals> Hamiltonian::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_orbitals();
}

bool Hamiltonian::has_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_orbitals();
}

double Hamiltonian::get_core_energy() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_core_energy();
}

HamiltonianType Hamiltonian::get_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_type();
}

std::string Hamiltonian::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_container_type();
}

bool Hamiltonian::is_hermitian() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->is_hermitian();
}

bool Hamiltonian::is_restricted() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->is_restricted();
}

bool Hamiltonian::is_unrestricted() const {
  QDK_LOG_TRACE_ENTERING();
  return !_container->is_restricted();
}

}  // namespace qdk::chemistry::data
