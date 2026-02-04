// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "../filename_utils.hpp"
#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

CanonicalFourCenterHamiltonianContainer::
    CanonicalFourCenterHamiltonianContainer(
        const Eigen::MatrixXd& one_body_integrals,
        const Eigen::VectorXd& two_body_integrals,
        std::shared_ptr<Orbitals> orbitals, double core_energy,
        const Eigen::MatrixXd& inactive_fock_matrix, HamiltonianType type)
    : HamiltonianContainer(one_body_integrals, orbitals, core_energy,
                           inactive_fock_matrix, type),
      _two_body_integrals(
          make_restricted_two_body_integrals(two_body_integrals)) {
  QDK_LOG_TRACE_ENTERING();

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  if (!is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

CanonicalFourCenterHamiltonianContainer::
    CanonicalFourCenterHamiltonianContainer(
        const Eigen::MatrixXd& one_body_integrals_alpha,
        const Eigen::MatrixXd& one_body_integrals_beta,
        const Eigen::VectorXd& two_body_integrals_aaaa,
        const Eigen::VectorXd& two_body_integrals_aabb,
        const Eigen::VectorXd& two_body_integrals_bbbb,
        std::shared_ptr<Orbitals> orbitals, double core_energy,
        const Eigen::MatrixXd& inactive_fock_matrix_alpha,
        const Eigen::MatrixXd& inactive_fock_matrix_beta, HamiltonianType type)
    : HamiltonianContainer(one_body_integrals_alpha, one_body_integrals_beta,
                           orbitals, core_energy, inactive_fock_matrix_alpha,
                           inactive_fock_matrix_beta, type),
      _two_body_integrals(
          std::make_unique<Eigen::VectorXd>(two_body_integrals_aaaa),
          std::make_unique<Eigen::VectorXd>(two_body_integrals_aabb),
          std::make_unique<Eigen::VectorXd>(two_body_integrals_bbbb)) {
  QDK_LOG_TRACE_ENTERING();

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  if (!is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

std::unique_ptr<HamiltonianContainer>
CanonicalFourCenterHamiltonianContainer::clone() const {
  QDK_LOG_TRACE_ENTERING();
  if (is_restricted()) {
    return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
        *_one_body_integrals.first, *std::get<0>(_two_body_integrals),
        _orbitals, _core_energy, *_inactive_fock_matrix.first, _type);
  }
  return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      *_one_body_integrals.first, *_one_body_integrals.second,
      *std::get<0>(_two_body_integrals), *std::get<1>(_two_body_integrals),
      *std::get<2>(_two_body_integrals), _orbitals, _core_energy,
      *_inactive_fock_matrix.first, *_inactive_fock_matrix.second, _type);
}

std::string CanonicalFourCenterHamiltonianContainer::get_container_type()
    const {
  QDK_LOG_TRACE_ENTERING();
  return "canonical_four_center";
}

std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&>
CanonicalFourCenterHamiltonianContainer::get_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }
  return std::make_tuple(std::cref(*std::get<0>(_two_body_integrals)),
                         std::cref(*std::get<1>(_two_body_integrals)),
                         std::cref(*std::get<2>(_two_body_integrals)));
}

double CanonicalFourCenterHamiltonianContainer::get_two_body_element(
    unsigned i, unsigned j, unsigned k, unsigned l, SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();

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

size_t CanonicalFourCenterHamiltonianContainer::get_two_body_index(
    size_t i, size_t j, size_t k, size_t l) const {
  QDK_LOG_TRACE_ENTERING();
  size_t norb = _orbitals->get_active_space_indices().first.size();
  return i * norb * norb * norb + j * norb * norb + k * norb + l;
}

bool CanonicalFourCenterHamiltonianContainer::has_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return std::get<0>(_two_body_integrals) != nullptr &&
         std::get<0>(_two_body_integrals)->size() > 0;
}

bool CanonicalFourCenterHamiltonianContainer::is_restricted() const {
  QDK_LOG_TRACE_ENTERING();
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

bool CanonicalFourCenterHamiltonianContainer::is_valid() const {
  QDK_LOG_TRACE_ENTERING();
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

void CanonicalFourCenterHamiltonianContainer::validate_integral_dimensions()
    const {
  QDK_LOG_TRACE_ENTERING();
  // Check alpha one-body integrals
  HamiltonianContainer::validate_integral_dimensions();

  if (!has_two_body_integrals()) {
    return;
  }

  // Check two-body integrals dimensions
  size_t norb_alpha = _one_body_integrals.first->rows();
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

std::tuple<std::shared_ptr<Eigen::VectorXd>, std::shared_ptr<Eigen::VectorXd>,
           std::shared_ptr<Eigen::VectorXd>>
CanonicalFourCenterHamiltonianContainer::make_restricted_two_body_integrals(
    const Eigen::VectorXd& integrals) {
  QDK_LOG_TRACE_ENTERING();
  auto shared_integrals = std::make_shared<Eigen::VectorXd>(integrals);
  return std::make_tuple(
      shared_integrals, shared_integrals,
      shared_integrals);  // aaaa, aabb, bbbb all point to same data
}

void CanonicalFourCenterHamiltonianContainer::to_fcidump_file(
    const std::string& filename, size_t nalpha, size_t nbeta) const {
  QDK_LOG_TRACE_ENTERING();
  _to_fcidump_file(filename, nalpha, nbeta);
}

nlohmann::json CanonicalFourCenterHamiltonianContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Store container type
  j["container_type"] = get_container_type();

  // Store metadata
  j["core_energy"] = _core_energy;
  j["type"] =
      (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";

  // Store restrictedness information
  j["is_restricted"] = is_restricted();

  // Store one-body integrals
  if (has_one_body_integrals()) {
    j["has_one_body_integrals"] = true;

    // Store alpha one-body integrals
    std::vector<std::vector<double>> one_body_alpha_vec;
    for (int i = 0; i < _one_body_integrals.first->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _one_body_integrals.first->cols(); ++j_idx) {
        row.push_back((*_one_body_integrals.first)(i, j_idx));
      }
      one_body_alpha_vec.push_back(row);
    }
    j["one_body_integrals_alpha"] = one_body_alpha_vec;

    // Store beta one-body integrals (only if unrestricted)
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

    // Store as object {"aaaa": [...], "aabb": [...], "bbbb": [...]}
    nlohmann::json two_body_obj;

    // Store aaaa
    std::vector<double> two_body_aaaa_vec;
    for (int i = 0; i < std::get<0>(_two_body_integrals)->size(); ++i) {
      two_body_aaaa_vec.push_back((*std::get<0>(_two_body_integrals))(i));
    }
    two_body_obj["aaaa"] = two_body_aaaa_vec;

    // Store aabb
    std::vector<double> two_body_aabb_vec;
    for (int i = 0; i < std::get<1>(_two_body_integrals)->size(); ++i) {
      two_body_aabb_vec.push_back((*std::get<1>(_two_body_integrals))(i));
    }
    two_body_obj["aabb"] = two_body_aabb_vec;

    // Store bbbb
    std::vector<double> two_body_bbbb_vec;
    for (int i = 0; i < std::get<2>(_two_body_integrals)->size(); ++i) {
      two_body_bbbb_vec.push_back((*std::get<2>(_two_body_integrals))(i));
    }
    two_body_obj["bbbb"] = two_body_bbbb_vec;

    j["two_body_integrals"] = two_body_obj;
  } else {
    j["has_two_body_integrals"] = false;
  }

  // Store inactive Fock matrix
  if (has_inactive_fock_matrix()) {
    j["has_inactive_fock_matrix"] = true;
    // Store alpha inactive Fock matrix
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

    // Store beta inactive Fock matrix (only if unrestricted)
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

std::unique_ptr<CanonicalFourCenterHamiltonianContainer>
CanonicalFourCenterHamiltonianContainer::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load metadata
    double core_energy = j.value("core_energy", 0.0);

    // Load Hamiltonian type
    HamiltonianType type = HamiltonianType::Hermitian;
    if (j.contains("type")) {
      std::string type_str = j["type"].get<std::string>();
      if (type_str == "NonHermitian") {
        type = HamiltonianType::NonHermitian;
      }
    }

    // Determine if the saved Hamiltonian was restricted or unrestricted
    bool is_restricted_data = j.value("is_restricted", true);

    // Helper function to load matrix from JSON
    auto load_matrix =
        [](const nlohmann::json& matrix_json) -> Eigen::MatrixXd {
      auto matrix_vec = matrix_json.get<std::vector<std::vector<double>>>();
      if (matrix_vec.empty()) {
        return Eigen::MatrixXd(0, 0);
      }

      Eigen::MatrixXd matrix(matrix_vec.size(), matrix_vec[0].size());
      for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        if (static_cast<Eigen::Index>(matrix_vec[i].size()) != matrix.cols()) {
          throw std::runtime_error(
              "Matrix rows have inconsistent column counts");
        }
        matrix.row(i) =
            Eigen::VectorXd::Map(matrix_vec[i].data(), matrix.cols());
      }
      return matrix;
    };

    // Helper function to load vector from JSON
    auto load_vector =
        [](const nlohmann::json& vector_json) -> Eigen::VectorXd {
      auto vector_vec = vector_json.get<std::vector<double>>();
      Eigen::VectorXd vector(vector_vec.size());
      for (size_t i = 0; i < vector_vec.size(); ++i) {
        vector(i) = vector_vec[i];
      }
      return vector;
    };

    // Load one-body integrals
    Eigen::MatrixXd one_body_alpha, one_body_beta;
    if (j.value("has_one_body_integrals", false)) {
      if (j.contains("one_body_integrals_alpha")) {
        one_body_alpha = load_matrix(j["one_body_integrals_alpha"]);
      }

      if (is_restricted_data) {
        one_body_beta = one_body_alpha;
      } else if (j.contains("one_body_integrals_beta")) {
        one_body_beta = load_matrix(j["one_body_integrals_beta"]);
      } else {
        throw std::runtime_error("Should have beta integrals, if unrestricted");
      }
    }

    // Load two-body integrals
    Eigen::VectorXd two_body_aaaa, two_body_aabb, two_body_bbbb;
    bool has_two_body = j.value("has_two_body_integrals", false);
    if (has_two_body) {
      if (!j.contains("two_body_integrals")) {
        throw std::runtime_error("Two-body integrals data not found in JSON");
      }

      auto two_body_obj = j["two_body_integrals"];
      if (!two_body_obj.is_object()) {
        throw std::runtime_error(
            "two_body_integrals must be an object with aaaa, aabb, bbbb keys");
      }

      if (!two_body_obj.contains("aaaa") || !two_body_obj.contains("aabb") ||
          !two_body_obj.contains("bbbb")) {
        throw std::runtime_error(
            "two_body_integrals must contain aaaa, aabb, and bbbb keys");
      }

      two_body_aaaa = load_vector(two_body_obj["aaaa"]);
      two_body_aabb = load_vector(two_body_obj["aabb"]);
      two_body_bbbb = load_vector(two_body_obj["bbbb"]);
    }

    // Load inactive Fock matrix

    Eigen::MatrixXd inactive_fock_alpha, inactive_fock_beta;
    bool has_inactive_fock = j.value("has_inactive_fock_matrix", false);
    if (has_inactive_fock) {
      if (j.contains("inactive_fock_matrix_alpha")) {
        inactive_fock_alpha = load_matrix(j["inactive_fock_matrix_alpha"]);
      }

      if (is_restricted_data) {
        inactive_fock_beta = inactive_fock_alpha;
      } else if (j.contains("inactive_fock_matrix_beta")) {
        inactive_fock_beta = load_matrix(j["inactive_fock_matrix_beta"]);
      }
    }

    // Load orbital data
    if (!j.value("has_orbitals", false)) {
      throw std::runtime_error("Hamiltonian JSON must include orbitals data");
    }
    auto orbitals = Orbitals::from_json(j["orbitals"]);

    // Validate consistency: if orbitals have inactive indices,
    // then inactive fock matrix must be present
    if (orbitals->has_inactive_space()) {
      if (!has_inactive_fock) {
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

    // Create and return appropriate Hamiltonian using the correct constructor
    if (is_restricted_data) {
      // Use restricted constructor - it will create shared pointers internally
      // so alpha and beta point to the same data
      return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, two_body_aaaa, orbitals, core_energy,
          inactive_fock_alpha, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, orbitals, core_energy, inactive_fock_alpha,
          inactive_fock_beta, type);
    }

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Hamiltonian from JSON: " +
                             std::string(e.what()));
  }
}

void CanonicalFourCenterHamiltonianContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    // Save version first
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    H5::Attribute version_attr =
        group.createAttribute("version", string_type, scalar_space);
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);

    // Add container type attribute
    H5::Attribute container_type_attr =
        group.createAttribute("container_type", string_type, scalar_space);
    std::string container_type_str = get_container_type();
    container_type_attr.write(string_type, container_type_str);

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

std::unique_ptr<CanonicalFourCenterHamiltonianContainer>
CanonicalFourCenterHamiltonianContainer::from_hdf5(H5::Group& group) {
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
      return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, two_body_aaaa, orbitals, core_energy,
          inactive_fock_alpha, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, orbitals, core_energy, inactive_fock_alpha,
          inactive_fock_beta, type);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void CanonicalFourCenterHamiltonianContainer::_to_fcidump_file(
    const std::string& filename, size_t nalpha, size_t nbeta) const {
  QDK_LOG_TRACE_ENTERING();
  // Check if this is an unrestricted Hamiltonian and throw error
  if (is_unrestricted()) {
    throw std::runtime_error(
        "FCIDUMP format is not supported for unrestricted Hamiltonians.");
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  size_t num_molecular_orbitals;
  if (has_orbitals()) {
    if (_orbitals->has_active_space()) {
      auto active_indices = _orbitals->get_active_space_indices();
      size_t n_active_alpha = active_indices.first.size();
      size_t n_active_beta = active_indices.second.size();

      // For restricted case, alpha and beta should be the same
      if (n_active_alpha != n_active_beta) {
        throw std::invalid_argument(
            "For restricted Hamiltonian, alpha and beta active spaces must "
            "have "
            "same size");
      }
      num_molecular_orbitals =
          n_active_alpha;  // Can use either alpha or beta since they're equal
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
  const double print_thresh = std::numeric_limits<double>::epsilon();

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
