// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : _one_body_integrals(other._one_body_integrals),
      _two_body_integrals(other._two_body_integrals),
      _inactive_fock_matrix(other._inactive_fock_matrix),
      _orbitals(other._orbitals),
      _core_energy(other._core_energy),
      _type(other._type) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

Hamiltonian::Hamiltonian(const Eigen::MatrixXd& one_body_integrals,
                         const Eigen::VectorXd& two_body_integrals,
                         std::shared_ptr<Orbitals> orbitals, double core_energy,
                         const Eigen::MatrixXd& inactive_fock_matrix,
                         HamiltonianType type)
    : _one_body_integrals(
          make_restricted_one_body_integrals(one_body_integrals)),
      _two_body_integrals(
          make_restricted_two_body_integrals(two_body_integrals)),
      _inactive_fock_matrix(
          make_restricted_inactive_fock_matrix(inactive_fock_matrix)),
      _orbitals(orbitals),
      _core_energy(core_energy),
      _type(type) {
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  // Validate that orbitals have the necessary data
  if (!orbitals->has_active_space()) {
    throw std::runtime_error(
        "Orbitals must have an active space set for Hamiltonian");
  }
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

Hamiltonian::Hamiltonian(const Eigen::MatrixXd& one_body_integrals_alpha,
                         const Eigen::MatrixXd& one_body_integrals_beta,
                         const Eigen::VectorXd& two_body_integrals_aaaa,
                         const Eigen::VectorXd& two_body_integrals_aabb,
                         const Eigen::VectorXd& two_body_integrals_bbbb,
                         std::shared_ptr<Orbitals> orbitals, double core_energy,
                         const Eigen::MatrixXd& inactive_fock_matrix_alpha,
                         const Eigen::MatrixXd& inactive_fock_matrix_beta,
                         HamiltonianType type)
    : _one_body_integrals(
          std::make_unique<Eigen::MatrixXd>(one_body_integrals_alpha),
          std::make_unique<Eigen::MatrixXd>(one_body_integrals_beta)),
      _two_body_integrals(
          std::make_unique<Eigen::VectorXd>(two_body_integrals_aaaa),
          std::make_unique<Eigen::VectorXd>(two_body_integrals_aabb),
          std::make_unique<Eigen::VectorXd>(two_body_integrals_bbbb)),
      _inactive_fock_matrix(
          std::make_unique<Eigen::MatrixXd>(inactive_fock_matrix_alpha),
          std::make_unique<Eigen::MatrixXd>(inactive_fock_matrix_beta)),
      _orbitals(orbitals),
      _core_energy(core_energy),
      _type(type) {
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  // Validate that orbitals have the necessary data
  if (!orbitals->has_active_space()) {
    throw std::runtime_error(
        "Orbitals must have an active space set for Hamiltonian");
  }
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
  if (this != &other) {
    // Since all members are const, we need to use placement new
    this->~Hamiltonian();
    new (this) Hamiltonian(other);
  }
  return *this;
}

const Eigen::MatrixXd& Hamiltonian::get_one_body_integrals() const {
  if (!has_one_body_integrals()) {
    throw std::runtime_error("One-body integrals are not set");
  }
  return *_one_body_integrals
              .first;  // Return alpha integrals for backward compatibility
}

bool Hamiltonian::has_one_body_integrals() const {
  return _one_body_integrals.first != nullptr &&
         _one_body_integrals.first->rows() > 0 &&
         _one_body_integrals.first->cols() > 0;
}

const Eigen::MatrixXd& Hamiltonian::get_one_body_integrals_alpha() const {
  if (!has_one_body_integrals()) {
    throw std::runtime_error("One-body integrals are not set");
  }
  return *_one_body_integrals.first;
}

const Eigen::MatrixXd& Hamiltonian::get_one_body_integrals_beta() const {
  if (!has_one_body_integrals()) {
    throw std::runtime_error("One-body integrals are not set");
  }
  return *_one_body_integrals.second;
}

const Eigen::VectorXd& Hamiltonian::get_two_body_integrals() const {
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }
  return *std::get<0>(_two_body_integrals);  // Return alpha-alpha integrals for
                                             // backward compatibility
}

bool Hamiltonian::has_inactive_fock_matrix() const {
  return _inactive_fock_matrix.first != nullptr &&
         _inactive_fock_matrix.first->size() > 0;
}

double Hamiltonian::get_two_body_element(unsigned i, unsigned j, unsigned k,
                                         unsigned l,
                                         SpinChannel channel) const {
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }

  size_t norb = _orbitals->get_active_space_indices().first.size();
  if (i >= norb || j >= norb || k >= norb || l >= norb) {
    throw std::out_of_range("Orbital index out of range");
  }

  size_t index = get_two_body_index(i, j, k, l);

  // Select the appropriate integral based on spin channel
  switch (channel) {
    case SpinChannel::aaaa:
      return (*std::get<0>(_two_body_integrals))[index];
    case SpinChannel::aabb:
      return (*std::get<1>(_two_body_integrals))[index];
    case SpinChannel::bbbb:
      return (*std::get<2>(_two_body_integrals))[index];
    default:
      throw std::invalid_argument("Invalid spin channel");
  }
}

bool Hamiltonian::has_two_body_integrals() const {
  return std::get<0>(_two_body_integrals) != nullptr &&
         std::get<0>(_two_body_integrals)->size() > 0;
}

const Eigen::VectorXd& Hamiltonian::get_two_body_integrals_aaaa() const {
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }
  return *std::get<0>(_two_body_integrals);
}

const Eigen::VectorXd& Hamiltonian::get_two_body_integrals_aabb() const {
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }
  return *std::get<1>(_two_body_integrals);
}

const Eigen::VectorXd& Hamiltonian::get_two_body_integrals_bbbb() const {
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }
  return *std::get<2>(_two_body_integrals);
}

std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
Hamiltonian::get_inactive_fock_matrix() const {
  if (!has_inactive_fock_matrix()) {
    throw std::runtime_error("Inactive Fock matrix is not set");
  }
  return {*_inactive_fock_matrix.first, *_inactive_fock_matrix.second};
}

const Eigen::MatrixXd& Hamiltonian::get_inactive_fock_matrix_alpha() const {
  if (!has_inactive_fock_matrix()) {
    throw std::runtime_error("Inactive Fock matrix is not set");
  }
  return *_inactive_fock_matrix.first;
}

const Eigen::MatrixXd& Hamiltonian::get_inactive_fock_matrix_beta() const {
  if (!has_inactive_fock_matrix()) {
    throw std::runtime_error("Inactive Fock matrix is not set");
  }
  return *_inactive_fock_matrix.second;
}

const std::shared_ptr<Orbitals> Hamiltonian::get_orbitals() const {
  if (!has_orbitals()) {
    throw std::runtime_error("Orbitals are not set");
  }
  return _orbitals;
}

bool Hamiltonian::has_orbitals() const { return _orbitals != nullptr; }

double Hamiltonian::get_core_energy() const { return _core_energy; }

HamiltonianType Hamiltonian::get_type() const { return _type; }

bool Hamiltonian::is_hermitian() const {
  return _type == HamiltonianType::Hermitian;
}

bool Hamiltonian::is_restricted() const {
  // Hamiltonian is restricted if alpha and beta components point to the same
  // data
  return (_one_body_integrals.first == _one_body_integrals.second) &&
         (std::get<0>(_two_body_integrals) ==
          std::get<1>(_two_body_integrals)) &&
         (std::get<0>(_two_body_integrals) ==
          std::get<2>(_two_body_integrals)) &&
         (_inactive_fock_matrix.first == _inactive_fock_matrix.second ||
          (!_inactive_fock_matrix.first && !_inactive_fock_matrix.second));
}

bool Hamiltonian::is_unrestricted() const { return !is_restricted(); }

bool Hamiltonian::_is_valid() const {
  // Check if essential data is present
  if (!has_one_body_integrals() || !has_two_body_integrals()) {
    return false;
  }

  // Check dimension consistency
  try {
    validate_integral_dimensions();
  } catch (const std::exception&) {
    return false;
  }

  return true;
}

void Hamiltonian::validate_integral_dimensions() const {
  if (!has_one_body_integrals() || !has_two_body_integrals()) {
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

  // Check two-body integrals dimensions
  unsigned expected_size = norb_alpha * norb_alpha * norb_alpha * norb_alpha;

  // Check alpha-alpha integrals
  if (static_cast<unsigned>(std::get<0>(_two_body_integrals)->size()) !=
      expected_size) {
    throw std::invalid_argument(
        "Alpha-alpha two-body integrals size (" +
        std::to_string(std::get<0>(_two_body_integrals)->size()) +
        ") does not match expected size (" + std::to_string(expected_size) +
        ") for " + std::to_string(norb_alpha) + " orbitals");
  }

  // Check alpha-beta integrals (if different from alpha-alpha)
  if (std::get<1>(_two_body_integrals) != std::get<0>(_two_body_integrals)) {
    if (static_cast<unsigned>(std::get<1>(_two_body_integrals)->size()) !=
        expected_size) {
      throw std::invalid_argument(
          "Alpha-beta two-body integrals size mismatch");
    }
  }

  // Check beta-beta integrals (if different from alpha-alpha)
  if (std::get<2>(_two_body_integrals) != std::get<0>(_two_body_integrals)) {
    if (static_cast<unsigned>(std::get<2>(_two_body_integrals)->size()) !=
        expected_size) {
      throw std::invalid_argument("Beta-beta two-body integrals size mismatch");
    }
  }
}

void Hamiltonian::validate_restrictedness_consistency() const {
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

void Hamiltonian::validate_active_space_dimensions() const {
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

size_t Hamiltonian::get_two_body_index(size_t i, size_t j, size_t k,
                                       size_t l) const {
  size_t norb = _orbitals->get_active_space_indices().first.size();
  return i * norb * norb * norb + j * norb * norb + k * norb + l;
}

std::pair<std::shared_ptr<Eigen::MatrixXd>, std::shared_ptr<Eigen::MatrixXd>>
Hamiltonian::make_restricted_one_body_integrals(
    const Eigen::MatrixXd& integrals) {
  auto shared_integrals = std::make_shared<Eigen::MatrixXd>(integrals);
  return std::make_pair(
      shared_integrals,
      shared_integrals);  // Both alpha and beta point to same data
}

std::tuple<std::shared_ptr<Eigen::VectorXd>, std::shared_ptr<Eigen::VectorXd>,
           std::shared_ptr<Eigen::VectorXd>>
Hamiltonian::make_restricted_two_body_integrals(
    const Eigen::VectorXd& integrals) {
  auto shared_integrals = std::make_shared<Eigen::VectorXd>(integrals);
  return std::make_tuple(
      shared_integrals, shared_integrals,
      shared_integrals);  // aaaa, aabb, bbbb all point to same data
}

std::pair<std::shared_ptr<Eigen::MatrixXd>, std::shared_ptr<Eigen::MatrixXd>>
Hamiltonian::make_restricted_inactive_fock_matrix(
    const Eigen::MatrixXd& matrix) {
  auto shared_matrix = std::make_shared<Eigen::MatrixXd>(matrix);
  return std::make_pair(
      shared_matrix, shared_matrix);  // Both alpha and beta point to same data
}

std::string Hamiltonian::get_summary() const {
  std::string summary = "Hamiltonian Summary:\n";
  size_t num_molecular_orbitals = _orbitals->get_num_molecular_orbitals();
  size_t norb = _orbitals->get_active_space_indices().first.size();
  summary += "  Type: ";
  summary += (is_hermitian() ? "Hermitian" : "NonHermitian");
  summary += "\n";
  summary += "  Active Orbitals: " + std::to_string(norb) + "\n";
  summary +=
      "  Total Orbitals: " + std::to_string(num_molecular_orbitals) + "\n";

  const double threshold = 1e-6;  // Threshold for determining negligible
                                  // integrals in summary statistics
  const size_t non_negligible_one_body_ints = std::count_if(
      get_one_body_integrals().data(),
      get_one_body_integrals().data() + get_one_body_integrals().size(),
      [threshold](double val) { return std::abs(val) > threshold; });
  const size_t non_negligible_two_body_ints = std::count_if(
      get_two_body_integrals().data(),
      get_two_body_integrals().data() + get_two_body_integrals().size(),
      [threshold](double val) { return std::abs(val) > threshold; });

  summary += "  Core Energy: " + std::to_string(get_core_energy()) + "\n";
  summary += "  Integral Statistics:\n";
  summary += "    One-body Integrals: " +
             std::to_string(get_one_body_integrals().size()) +
             " (larger than " + std::to_string(threshold) + ": " +
             std::to_string(non_negligible_one_body_ints) + ")\n";
  summary += "    Two-body Integrals: " +
             std::to_string(get_two_body_integrals().size()) +
             " (larger than " + std::to_string(threshold) + ": " +
             std::to_string(non_negligible_two_body_ints) + ")\n";

  return summary;
}

void Hamiltonian::to_file(const std::string& filename,
                          const std::string& type) const {
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
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "hamiltonian");

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_hdf5_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "hamiltonian");

  return _from_hdf5_file(validated_filename);
}

void Hamiltonian::to_json_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "hamiltonian");

  _to_json_file(validated_filename);
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_json_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "hamiltonian");

  return _from_json_file(validated_filename);
}

void Hamiltonian::to_fcidump_file(const std::string& filename, size_t nalpha,
                                  size_t nbeta) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "hamiltonian");

  _to_fcidump_file(validated_filename, nalpha, nbeta);
}

void Hamiltonian::_to_json_file(const std::string& filename) const {
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

nlohmann::json Hamiltonian::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Store metadata
  j["core_energy"] = _core_energy;
  j["type"] =
      (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";

  // Store restrictedness information
  j["is_restricted"] = is_restricted();

  // Store one-body integrals
  if (has_one_body_integrals()) {
    j["has_one_body_integrals"] = true;
    std::vector<std::vector<double>> one_body_alpha_vec;
    for (int i = 0; i < _one_body_integrals.first->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _one_body_integrals.first->cols(); ++j_idx) {
        row.push_back((*_one_body_integrals.first)(i, j_idx));
      }
      one_body_alpha_vec.push_back(row);
    }
    j["one_body_integrals_alpha"] = one_body_alpha_vec;

    // For unrestricted case, store beta integrals separately
    if (is_unrestricted()) {
      std::vector<std::vector<double>> one_body_beta_vec;
      for (int i = 0; i < _one_body_integrals.second->rows(); ++i) {
        std::vector<double> row;
        for (int j_idx = 0; j_idx < _one_body_integrals.second->cols();
             ++j_idx) {
          row.push_back((*_one_body_integrals.second)(i, j_idx));
        }
        one_body_beta_vec.push_back(row);
      }
      j["one_body_integrals_beta"] = one_body_beta_vec;
    }
  } else {
    j["has_one_body_integrals"] = false;
  }

  // Store two-body integrals
  if (has_two_body_integrals()) {
    j["has_two_body_integrals"] = true;
    std::vector<double> two_body_aaaa_vec;
    for (int i = 0; i < std::get<0>(_two_body_integrals)->size(); ++i) {
      two_body_aaaa_vec.push_back((*std::get<0>(_two_body_integrals))(i));
    }
    j["two_body_integrals_aaaa"] = two_body_aaaa_vec;

    // For unrestricted case, store aabb and bbbb integrals separately
    if (is_unrestricted()) {
      std::vector<double> two_body_aabb_vec;
      for (int i = 0; i < std::get<1>(_two_body_integrals)->size(); ++i) {
        two_body_aabb_vec.push_back((*std::get<1>(_two_body_integrals))(i));
      }
      j["two_body_integrals_aabb"] = two_body_aabb_vec;

      std::vector<double> two_body_bbbb_vec;
      for (int i = 0; i < std::get<2>(_two_body_integrals)->size(); ++i) {
        two_body_bbbb_vec.push_back((*std::get<2>(_two_body_integrals))(i));
      }
      j["two_body_integrals_bbbb"] = two_body_bbbb_vec;
    }
  } else {
    j["has_two_body_integrals"] = false;
  }

  // Store inactive Fock matrix
  if (has_inactive_fock_matrix()) {
    j["has_inactive_fock_matrix"] = true;
    std::vector<std::vector<double>> inactive_fock_alpha_vec;
    for (int i = 0; i < _inactive_fock_matrix.first->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _inactive_fock_matrix.first->cols();
           ++j_idx) {
        row.push_back((*_inactive_fock_matrix.first)(i, j_idx));
      }
      inactive_fock_alpha_vec.push_back(row);
    }
    j["inactive_fock_matrix_alpha"] = inactive_fock_alpha_vec;

    // For unrestricted case, store beta Fock matrix separately
    if (is_unrestricted()) {
      std::vector<std::vector<double>> inactive_fock_beta_vec;
      for (int i = 0; i < _inactive_fock_matrix.second->rows(); ++i) {
        std::vector<double> row;
        for (int j_idx = 0; j_idx < _inactive_fock_matrix.second->cols();
             ++j_idx) {
          row.push_back((*_inactive_fock_matrix.second)(i, j_idx));
        }
        inactive_fock_beta_vec.push_back(row);
      }
      j["inactive_fock_matrix_beta"] = inactive_fock_beta_vec;
    }
  } else {
    j["has_inactive_fock_matrix"] = false;
  }

  // Store orbital data
  if (has_orbitals()) {
    j["has_orbitals"] = true;
    j["orbitals"] = _orbitals->to_json();
  } else {
    j["has_orbitals"] = false;
  }

  return j;
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_json(const nlohmann::json& j) {
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load metadata
    double core_energy = j.value("core_energy", 0.0);

    // Load Hamiltonian type (default to Hermitian for backward compatibility)
    HamiltonianType type = HamiltonianType::Hermitian;
    if (j.contains("type")) {
      std::string type_str = j["type"].get<std::string>();
      if (type_str == "NonHermitian") {
        type = HamiltonianType::NonHermitian;
      }
    }

    // Load restrictedness information (default to true for backward
    // compatibility)
    bool is_restricted_data = j.value("is_restricted", true);

    // Load orbital data
    if (!j.value("has_orbitals", false)) {
      throw std::runtime_error("Hamiltonian JSON must include orbitals data");
    }
    auto orbitals = Orbitals::from_json(j["orbitals"]);

    // Load integral data based on restrictedness
    Eigen::MatrixXd one_body_alpha, one_body_beta;
    Eigen::VectorXd two_body_aaaa, two_body_aabb, two_body_bbbb;
    Eigen::MatrixXd inactive_fock_alpha, inactive_fock_beta;

    // Load one-body integrals
    if (j.value("has_one_body_integrals", false)) {
      // Try new format first (with alpha/beta labels)
      if (j.contains("one_body_integrals_alpha")) {
        auto one_body_alpha_vec = j["one_body_integrals_alpha"]
                                      .get<std::vector<std::vector<double>>>();
        int rows = one_body_alpha_vec.size();
        int cols = rows > 0 ? one_body_alpha_vec[0].size() : 0;
        one_body_alpha.resize(rows, cols);
        for (int i = 0; i < rows; ++i) {
          for (int j_idx = 0; j_idx < cols; ++j_idx) {
            one_body_alpha(i, j_idx) = one_body_alpha_vec[i][j_idx];
          }
        }

        // For unrestricted, load beta separately
        if (!is_restricted_data && j.contains("one_body_integrals_beta")) {
          auto one_body_beta_vec = j["one_body_integrals_beta"]
                                       .get<std::vector<std::vector<double>>>();
          int rows_beta = one_body_beta_vec.size();
          int cols_beta = rows_beta > 0 ? one_body_beta_vec[0].size() : 0;
          one_body_beta.resize(rows_beta, cols_beta);
          for (int i = 0; i < rows_beta; ++i) {
            for (int j_idx = 0; j_idx < cols_beta; ++j_idx) {
              one_body_beta(i, j_idx) = one_body_beta_vec[i][j_idx];
            }
          }
        }
      } else if (j.contains("one_body_integrals")) {
        // Backward compatibility: old format without alpha/beta labels
        // This is always restricted format
        auto one_body_vec =
            j["one_body_integrals"].get<std::vector<std::vector<double>>>();
        int rows = one_body_vec.size();
        int cols = rows > 0 ? one_body_vec[0].size() : 0;
        one_body_alpha.resize(rows, cols);
        for (int i = 0; i < rows; ++i) {
          for (int j_idx = 0; j_idx < cols; ++j_idx) {
            one_body_alpha(i, j_idx) = one_body_vec[i][j_idx];
          }
        }
      }
    }

    // Load two-body integrals
    if (j.value("has_two_body_integrals", false)) {
      // Try new format first (with aaaa/aabb/bbbb labels)
      if (j.contains("two_body_integrals_aaaa")) {
        auto two_body_aaaa_vec =
            j["two_body_integrals_aaaa"].get<std::vector<double>>();
        two_body_aaaa.resize(two_body_aaaa_vec.size());
        for (size_t i = 0; i < two_body_aaaa_vec.size(); ++i) {
          two_body_aaaa(i) = two_body_aaaa_vec[i];
        }

        // For unrestricted, load aabb and bbbb separately
        if (!is_restricted_data) {
          if (j.contains("two_body_integrals_aabb")) {
            auto two_body_aabb_vec =
                j["two_body_integrals_aabb"].get<std::vector<double>>();
            two_body_aabb.resize(two_body_aabb_vec.size());
            for (size_t i = 0; i < two_body_aabb_vec.size(); ++i) {
              two_body_aabb(i) = two_body_aabb_vec[i];
            }
          }
          if (j.contains("two_body_integrals_bbbb")) {
            auto two_body_bbbb_vec =
                j["two_body_integrals_bbbb"].get<std::vector<double>>();
            two_body_bbbb.resize(two_body_bbbb_vec.size());
            for (size_t i = 0; i < two_body_bbbb_vec.size(); ++i) {
              two_body_bbbb(i) = two_body_bbbb_vec[i];
            }
          }
        }
      } else if (j.contains("two_body_integrals")) {
        // Backward compatibility: old format without spin labels
        // This is always restricted format
        auto two_body_vec = j["two_body_integrals"].get<std::vector<double>>();
        two_body_aaaa.resize(two_body_vec.size());
        for (size_t i = 0; i < two_body_vec.size(); ++i) {
          two_body_aaaa(i) = two_body_vec[i];
        }
      }
    }

    // Load inactive Fock matrix
    if (j.value("has_inactive_fock_matrix", false)) {
      // Try new format first (with alpha/beta labels)
      if (j.contains("inactive_fock_matrix_alpha")) {
        auto inactive_fock_alpha_vec =
            j["inactive_fock_matrix_alpha"]
                .get<std::vector<std::vector<double>>>();
        int rows = inactive_fock_alpha_vec.size();
        int cols = rows > 0 ? inactive_fock_alpha_vec[0].size() : 0;
        inactive_fock_alpha.resize(rows, cols);
        for (int i = 0; i < rows; ++i) {
          for (int j_idx = 0; j_idx < cols; ++j_idx) {
            inactive_fock_alpha(i, j_idx) = inactive_fock_alpha_vec[i][j_idx];
          }
        }

        // For unrestricted, load beta separately
        if (!is_restricted_data && j.contains("inactive_fock_matrix_beta")) {
          auto inactive_fock_beta_vec =
              j["inactive_fock_matrix_beta"]
                  .get<std::vector<std::vector<double>>>();
          int rows_beta = inactive_fock_beta_vec.size();
          int cols_beta = rows_beta > 0 ? inactive_fock_beta_vec[0].size() : 0;
          inactive_fock_beta.resize(rows_beta, cols_beta);
          for (int i = 0; i < rows_beta; ++i) {
            for (int j_idx = 0; j_idx < cols_beta; ++j_idx) {
              inactive_fock_beta(i, j_idx) = inactive_fock_beta_vec[i][j_idx];
            }
          }
        }
      } else if (j.contains("inactive_fock_matrix")) {
        // Backward compatibility: old format without alpha/beta labels
        // This is always restricted format
        auto inactive_fock_vec =
            j["inactive_fock_matrix"].get<std::vector<std::vector<double>>>();
        int rows = inactive_fock_vec.size();
        int cols = rows > 0 ? inactive_fock_vec[0].size() : 0;
        inactive_fock_alpha.resize(rows, cols);
        for (int i = 0; i < rows; ++i) {
          for (int j_idx = 0; j_idx < cols; ++j_idx) {
            inactive_fock_alpha(i, j_idx) = inactive_fock_vec[i][j_idx];
          }
        }
      }
    }

    // Validate consistency: if orbitals have inactive indices,
    // then inactive fock matrix must be present
    if (orbitals->has_inactive_space()) {
      if (inactive_fock_alpha.size() == 0) {
        auto inactive_indices = orbitals->get_inactive_space_indices();
        size_t total_inactive =
            inactive_indices.first.size() + inactive_indices.second.size();
        throw std::runtime_error(
            "Hamiltonian JSON: orbitals have " +
            std::to_string(total_inactive) +
            " inactive indices but no inactive Fock matrix is provided");
      }
      // Core energy should be explicitly set when there are inactive orbitals
      if (!j.contains("core_energy")) {
        auto inactive_indices = orbitals->get_inactive_space_indices();
        size_t total_inactive =
            inactive_indices.first.size() + inactive_indices.second.size();
        throw std::runtime_error(
            "Hamiltonian JSON: orbitals have " +
            std::to_string(total_inactive) +
            " inactive indices but no core energy is provided");
      }
    }

    // Create Hamiltonian with loaded data using appropriate constructor
    if (is_restricted_data) {
      // Use restricted constructor - it will create shared pointers internally
      // so alpha and beta point to the same data
      return std::make_shared<Hamiltonian>(one_body_alpha, two_body_aaaa,
                                           orbitals, core_energy,
                                           inactive_fock_alpha, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_shared<Hamiltonian>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, orbitals, core_energy, inactive_fock_alpha,
          inactive_fock_beta, type);
    }

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Hamiltonian from JSON: " +
                             std::string(e.what()));
  }
}

void Hamiltonian::_to_hdf5_file(const std::string& filename) const {
  if (!_is_valid()) {
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

    // Save core energy
    H5::Attribute core_energy_attr = metadata_group.createAttribute(
        "core_energy", H5::PredType::NATIVE_DOUBLE, scalar_space);
    core_energy_attr.write(H5::PredType::NATIVE_DOUBLE, &_core_energy);

    // Save Hamiltonian type
    std::string type_str =
        (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";
    H5::StrType type_string_type(H5::PredType::C_S1, type_str.length() + 1);
    H5::Attribute type_attr =
        metadata_group.createAttribute("type", type_string_type, scalar_space);
    type_attr.write(type_string_type, type_str.c_str());

    // Save restrictedness information
    hbool_t is_restricted_flag = is_restricted() ? 1 : 0;
    H5::Attribute restricted_attr = metadata_group.createAttribute(
        "is_restricted", H5::PredType::NATIVE_HBOOL, scalar_space);
    restricted_attr.write(H5::PredType::NATIVE_HBOOL, &is_restricted_flag);

    // Save integrals data
    if (has_one_body_integrals()) {
      save_matrix_to_group(group, "one_body_integrals_alpha",
                           *_one_body_integrals.first);
      if (is_unrestricted()) {
        save_matrix_to_group(group, "one_body_integrals_beta",
                             *_one_body_integrals.second);
      }
    }

    if (has_two_body_integrals()) {
      save_vector_to_group(group, "two_body_integrals_aaaa",
                           *std::get<0>(_two_body_integrals));
      if (is_unrestricted()) {
        save_vector_to_group(group, "two_body_integrals_aabb",
                             *std::get<1>(_two_body_integrals));
        save_vector_to_group(group, "two_body_integrals_bbbb",
                             *std::get<2>(_two_body_integrals));
      }
    }

    // Save inactive Fock matrix
    if (has_inactive_fock_matrix()) {
      save_matrix_to_group(group, "inactive_fock_matrix_alpha",
                           *_inactive_fock_matrix.first);
      if (is_unrestricted()) {
        save_matrix_to_group(group, "inactive_fock_matrix_beta",
                             *_inactive_fock_matrix.second);
      }
    }

    // Save nested orbitals data using HDF5 group
    if (_orbitals) {
      H5::Group orbitals_group = group.createGroup("orbitals");
      _orbitals->to_hdf5(orbitals_group);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Hamiltonian> Hamiltonian::_from_hdf5_file(
    const std::string& filename) {
  try {
    // Open HDF5 file
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // Use the group-based deserialization method
    return from_hdf5(file);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Hamiltonian> Hamiltonian::from_hdf5(H5::Group& group) {
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

    // Load core energy
    double core_energy;
    H5::Attribute core_energy_attr =
        metadata_group.openAttribute("core_energy");
    core_energy_attr.read(H5::PredType::NATIVE_DOUBLE, &core_energy);

    // Load Hamiltonian type
    HamiltonianType type = HamiltonianType::Hermitian;
    if (metadata_group.attrExists("type")) {
      H5::Attribute type_attr = metadata_group.openAttribute("type");
      H5::StrType string_type = type_attr.getStrType();
      std::string type_str;
      type_attr.read(string_type, type_str);
      if (type_str == "NonHermitian") {
        type = HamiltonianType::NonHermitian;
      }
    }

    // Load restrictedness information
    bool is_restricted_data = true;  // default to restricted
    if (metadata_group.attrExists("is_restricted")) {
      H5::Attribute restricted_attr =
          metadata_group.openAttribute("is_restricted");
      hbool_t is_restricted_flag;
      restricted_attr.read(H5::PredType::NATIVE_HBOOL, &is_restricted_flag);
      is_restricted_data = (is_restricted_flag != 0);
    }

    // Load orbitals data from nested group
    std::shared_ptr<Orbitals> orbitals;
    if (group.nameExists("orbitals")) {
      H5::Group orbitals_group = group.openGroup("orbitals");
      orbitals = Orbitals::from_hdf5(orbitals_group);
    }

    if (!orbitals) {
      throw std::runtime_error("Hamiltonian HDF5 must include orbitals data");
    }

    // Load integral data based on restrictedness
    Eigen::MatrixXd one_body_alpha, one_body_beta;
    Eigen::VectorXd two_body_aaaa, two_body_aabb, two_body_bbbb;
    Eigen::MatrixXd inactive_fock_alpha, inactive_fock_beta;

    // Load one-body integrals
    if (dataset_exists_in_group(group, "one_body_integrals_alpha")) {
      one_body_alpha =
          load_matrix_from_group(group, "one_body_integrals_alpha");
    }

    // For unrestricted, load beta separately
    if (!is_restricted_data &&
        dataset_exists_in_group(group, "one_body_integrals_beta")) {
      one_body_beta = load_matrix_from_group(group, "one_body_integrals_beta");
    }

    // Load two-body integrals
    if (dataset_exists_in_group(group, "two_body_integrals_aaaa")) {
      two_body_aaaa = load_vector_from_group(group, "two_body_integrals_aaaa");
    }

    // For unrestricted, load aabb and bbbb separately
    if (!is_restricted_data) {
      if (dataset_exists_in_group(group, "two_body_integrals_aabb")) {
        two_body_aabb =
            load_vector_from_group(group, "two_body_integrals_aabb");
      }
      if (dataset_exists_in_group(group, "two_body_integrals_bbbb")) {
        two_body_bbbb =
            load_vector_from_group(group, "two_body_integrals_bbbb");
      }
    }

    // Load inactive Fock matrix
    if (dataset_exists_in_group(group, "inactive_fock_matrix_alpha")) {
      inactive_fock_alpha =
          load_matrix_from_group(group, "inactive_fock_matrix_alpha");
    }

    // For unrestricted, load beta separately
    if (!is_restricted_data &&
        dataset_exists_in_group(group, "inactive_fock_matrix_beta")) {
      inactive_fock_beta =
          load_matrix_from_group(group, "inactive_fock_matrix_beta");
    }

    // Create and return appropriate Hamiltonian using the correct constructor
    if (is_restricted_data) {
      // Use restricted constructor - it will create shared pointers internally
      return std::make_shared<Hamiltonian>(one_body_alpha, two_body_aaaa,
                                           orbitals, core_energy,
                                           inactive_fock_alpha, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_shared<Hamiltonian>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, orbitals, core_energy, inactive_fock_alpha,
          inactive_fock_beta, type);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Hamiltonian::_to_fcidump_file(const std::string& filename, size_t nalpha,
                                   size_t nbeta) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  size_t num_molecular_orbitals;
  if (has_orbitals()) {
    if (_orbitals->has_active_space()) {
      num_molecular_orbitals =
          _orbitals->get_active_space_indices()
              .first.size();  // TODO: Assumes the same indices for alpha/beta
    } else {
      num_molecular_orbitals = _orbitals->get_num_molecular_orbitals();
    }
  } else {
    throw std::runtime_error("Orbitals are not set");
  }

  const size_t nelec = nalpha + nbeta;
  const size_t num_molecular_orbitals2 =
      num_molecular_orbitals * num_molecular_orbitals;
  const size_t num_molecular_orbitals3 =
      num_molecular_orbitals2 * num_molecular_orbitals;
  const double print_thresh =
      std::numeric_limits<double>::epsilon();  // TODO: Make configurable?

  // We don't use symmetry, so populate with C1 data
  std::string orb_string;
  for (auto i = 0ul; i < num_molecular_orbitals - 1; ++i) {
    orb_string += "1,";
  }
  orb_string += "1";

  // Write the header of the FCIDUMP file
  file << "&FCI ";
  file << "NORB=" << num_molecular_orbitals << ", ";
  file << "NELEC=" << nelec << ", ";
  file << "MS2=" << (nalpha - nbeta) << ",\n";
  file << "ORBSYM=" << orb_string << ",\n";
  file << "ISYM=1,\n";
  file << "&END\n";

  auto formatted_line = [&](size_t i, size_t j, size_t k, size_t l,
                            double val) {
    if (std::abs(val) < print_thresh) return;

    file << std::setw(28) << std::scientific << std::setprecision(16)
         << std::right << val << " ";
    file << std::setw(4) << i << " ";
    file << std::setw(4) << j << " ";
    file << std::setw(4) << k << " ";
    file << std::setw(4) << l;
  };

  auto write_eri = [&](size_t i, size_t j, size_t k, size_t l) {
    auto eri = (*std::get<0>(_two_body_integrals))(
        i * num_molecular_orbitals3 + j * num_molecular_orbitals2 +
        k * num_molecular_orbitals + l);

    formatted_line(i + 1, j + 1, k + 1, l + 1, eri);
    file << "\n";
  };

  auto write_1body = [&](size_t i, size_t j) {
    auto hel = (*_one_body_integrals.first)(i, j);

    formatted_line(i + 1, j + 1, 0, 0, hel);
    file << "\n";
  };

  // Write permutationally unique MO ERIs
  // TODO: This is only valid for integrals with 8 fold symmetry
  // TODO (NAB):  will this TODO be resolved before the release?
  for (size_t i = 0, ij = 0; i < num_molecular_orbitals; ++i)
    for (size_t j = i; j < num_molecular_orbitals; ++j, ij++) {
      for (size_t k = 0, kl = 0; k < num_molecular_orbitals; ++k)
        for (size_t l = k; l < num_molecular_orbitals; ++l, kl++) {
          if (ij <= kl) {
            write_eri(i, j, k, l);
          }
        }  // kl loop
    }  // ij loop

  // Write permutationally unique MO 1-body integrals
  for (size_t i = 0; i < num_molecular_orbitals; ++i)
    for (size_t j = 0; j <= i; ++j) {
      write_1body(i, j);
    }

  // Write core energy
  formatted_line(0, 0, 0, 0, _core_energy);
}

}  // namespace qdk::chemistry::data
