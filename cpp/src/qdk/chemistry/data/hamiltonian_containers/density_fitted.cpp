// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/density_fitted.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "../filename_utils.hpp"
#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

DensityFittedHamiltonianContainer::DensityFittedHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::MatrixXd& three_center_integrals,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix, HamiltonianType type)
    : HamiltonianContainer(one_body_integrals, orbitals, core_energy,
                           inactive_fock_matrix, type),
      _three_center_integrals(
          make_restricted_three_center_integrals(three_center_integrals)) {
  QDK_LOG_TRACE_ENTERING();

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  if (!is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

DensityFittedHamiltonianContainer::DensityFittedHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals_alpha,
    const Eigen::MatrixXd& one_body_integrals_beta,
    const Eigen::MatrixXd& three_center_integrals_aa,
    const Eigen::MatrixXd& three_center_integrals_bb,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix_alpha,
    const Eigen::MatrixXd& inactive_fock_matrix_beta, HamiltonianType type)
    : HamiltonianContainer(one_body_integrals_alpha, one_body_integrals_beta,
                           orbitals, core_energy, inactive_fock_matrix_alpha,
                           inactive_fock_matrix_beta, type),
      _three_center_integrals(
          std::make_unique<Eigen::MatrixXd>(three_center_integrals_aa),
          std::make_unique<Eigen::MatrixXd>(three_center_integrals_bb)) {
  QDK_LOG_TRACE_ENTERING();

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  if (!is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

std::unique_ptr<HamiltonianContainer> DensityFittedHamiltonianContainer::clone()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (is_restricted()) {
    return std::make_unique<DensityFittedHamiltonianContainer>(
        *_one_body_integrals.first, *_three_center_integrals.first, _orbitals,
        _core_energy, *_inactive_fock_matrix.first, _type);
  }
  return std::make_unique<DensityFittedHamiltonianContainer>(
      *_one_body_integrals.first, *_one_body_integrals.second,
      *_three_center_integrals.first, *_three_center_integrals.second,
      _orbitals, _core_energy, *_inactive_fock_matrix.first,
      *_inactive_fock_matrix.second, _type);
}

std::string DensityFittedHamiltonianContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "density_fitted";
}

std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&>
DensityFittedHamiltonianContainer::get_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Three-center integrals are not set");
  }

  // Lazily build and cache the four-center integrals on first access
  if (!_cached_four_center_integrals) {
    _build_four_center_cache();
  }

  return std::make_tuple(
      std::cref(*std::get<0>(*_cached_four_center_integrals)),
      std::cref(*std::get<1>(*_cached_four_center_integrals)),
      std::cref(*std::get<2>(*_cached_four_center_integrals)));
}

void DensityFittedHamiltonianContainer::_build_four_center_cache() const {
  QDK_LOG_TRACE_ENTERING();

  size_t norb = _orbitals->get_active_space_indices().first.size();
  size_t norb2 = norb * norb;
  size_t norb4 = norb2 * norb2;

  // Helper lambda to build 4-center from 3-center: (ij|kl) = sum_P A_P,ij *
  // A_P,kl This is a Gram matrix computation: G = A^T * A, using optimized BLAS
  // GEMM
  auto build_four_center = [&](const Eigen::MatrixXd& three_center_left,
                               const Eigen::MatrixXd& three_center_right)
      -> std::shared_ptr<Eigen::VectorXd> {
    // Allocate output vector
    auto four_center = std::make_shared<Eigen::VectorXd>(norb4);

    // Map the VectorXd memory as a row-major matrix view
    // Layout: V[ij*norb2 + kl] = G(ij,kl) where G = A^T * A
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        gram_view(four_center->data(), norb2, norb2);

    gram_view.noalias() = three_center_left.transpose() * three_center_right;

    return four_center;
  };

  // Build four-center integrals from three-center
  auto aaaa = build_four_center(*_three_center_integrals.first,
                                *_three_center_integrals.first);

  if (is_restricted()) {
    _cached_four_center_integrals.emplace(aaaa, aaaa, aaaa);
    return;
  } else {
    auto aabb = build_four_center(*_three_center_integrals.first,
                                  *_three_center_integrals.second);
    auto bbbb = build_four_center(*_three_center_integrals.second,
                                  *_three_center_integrals.second);
    _cached_four_center_integrals.emplace(std::move(aaaa), std::move(aabb),
                                          std::move(bbbb));
  }
}

std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
DensityFittedHamiltonianContainer::get_three_center_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Three-center two-body integrals are not set");
  }
  return std::make_pair(std::cref(*_three_center_integrals.first),
                        std::cref(*_three_center_integrals.second));
}

double DensityFittedHamiltonianContainer::get_two_body_element(
    unsigned i, unsigned j, unsigned k, unsigned l, SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();

  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }

  size_t norb = _orbitals->get_active_space_indices().first.size();
  if (i >= norb || j >= norb || k >= norb || l >= norb) {
    throw std::out_of_range("Orbital index out of range");
  }

  size_t ij = _get_geminal_index(i, j);
  size_t kl = _get_geminal_index(k, l);

  // Select the appropriate integral based on spin channel
  switch (channel) {
    case SpinChannel::aaaa:
      return _get_two_body_element(*_three_center_integrals.first, ij,
                                   *_three_center_integrals.first, kl);
    case SpinChannel::aabb:
      return _get_two_body_element(*_three_center_integrals.first, ij,
                                   *_three_center_integrals.second, kl);
    case SpinChannel::bbbb:
      return _get_two_body_element(*_three_center_integrals.second, ij,
                                   *_three_center_integrals.second, kl);
    default:
      throw std::invalid_argument("Invalid spin channel");
  }
}

double DensityFittedHamiltonianContainer::_get_two_body_element(
    const Eigen::MatrixXd& A, unsigned ij, const Eigen::MatrixXd& B,
    unsigned kl) const {
  QDK_LOG_TRACE_ENTERING();
  // Note three-center integral stores each geminal in a column
  return A.col(ij).dot(B.col(kl));
}

size_t DensityFittedHamiltonianContainer::_get_geminal_index(size_t i,
                                                             size_t j) const {
  QDK_LOG_TRACE_ENTERING();
  size_t norb = _orbitals->get_active_space_indices().first.size();
  return i * norb + j;
}

bool DensityFittedHamiltonianContainer::has_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _three_center_integrals.first != nullptr &&
         _three_center_integrals.first->size() > 0;
}

bool DensityFittedHamiltonianContainer::is_restricted() const {
  QDK_LOG_TRACE_ENTERING();
  // Hamiltonian is restricted if alpha and beta components point to the same
  // data
  return (_one_body_integrals.first == _one_body_integrals.second) &&
         (_three_center_integrals.first == _three_center_integrals.second) &&
         (_inactive_fock_matrix.first == _inactive_fock_matrix.second ||
          (!_inactive_fock_matrix.first && !_inactive_fock_matrix.second));
}

bool DensityFittedHamiltonianContainer::is_valid() const {
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

void DensityFittedHamiltonianContainer::validate_integral_dimensions() const {
  QDK_LOG_TRACE_ENTERING();
  // Check alpha one-body integrals
  HamiltonianContainer::validate_integral_dimensions();

  if (!has_two_body_integrals()) {
    return;
  }

  // Check two-body integrals dimensions
  // Three-center integrals have shape [n_aux x n_geminals] where n_geminals =
  // norb^2
  size_t norb_alpha = _one_body_integrals.first->rows();
  unsigned geminal_size = norb_alpha * norb_alpha;

  // Check alpha-alpha integrals - cols should equal geminal_size
  if (static_cast<unsigned>(_three_center_integrals.first->cols()) !=
      geminal_size) {
    throw std::invalid_argument(
        "Alpha-alpha three-center integrals columns (" +
        std::to_string(_three_center_integrals.first->cols()) +
        ") does not match expected geminal size (" +
        std::to_string(geminal_size) + " for " + std::to_string(norb_alpha) +
        " orbitals)");
  }

  // Check beta-beta integrals (if different from alpha-alpha)
  if (_three_center_integrals.second != _three_center_integrals.first) {
    if (static_cast<unsigned>(_three_center_integrals.second->cols()) !=
            geminal_size or
        static_cast<unsigned>(_three_center_integrals.second->rows()) !=
            static_cast<unsigned>(_three_center_integrals.first->rows())) {
      throw std::invalid_argument(
          "Alpha-beta three-center integrals size mismatch");
    }
  }
}

std::pair<std::shared_ptr<Eigen::MatrixXd>, std::shared_ptr<Eigen::MatrixXd>>
DensityFittedHamiltonianContainer::make_restricted_three_center_integrals(
    const Eigen::MatrixXd& integrals) {
  QDK_LOG_TRACE_ENTERING();
  auto shared_integrals = std::make_shared<Eigen::MatrixXd>(integrals);
  return std::make_pair(shared_integrals, shared_integrals);
}

void DensityFittedHamiltonianContainer::to_fcidump_file(
    const std::string& filename, size_t nalpha, size_t nbeta) const {
  QDK_LOG_TRACE_ENTERING();
  _to_fcidump_file(filename, nalpha, nbeta);
}

nlohmann::json DensityFittedHamiltonianContainer::to_json() const {
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

    // Store as object {"aa": [...], "ab": [...], "bb": [...]}
    nlohmann::json two_body_obj;

    // Store aa
    std::vector<std::vector<double>> three_center_aa_vec;
    for (int i = 0; i < _three_center_integrals.first->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _three_center_integrals.first->cols();
           ++j_idx) {
        row.push_back((*_three_center_integrals.first)(i, j_idx));
      }
      three_center_aa_vec.push_back(row);
    }
    two_body_obj["aa"] = three_center_aa_vec;
    // Store bb
    std::vector<std::vector<double>> three_center_bb_vec;
    for (int i = 0; i < _three_center_integrals.second->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _three_center_integrals.second->cols();
           ++j_idx) {
        row.push_back((*_three_center_integrals.second)(i, j_idx));
      }
      three_center_bb_vec.push_back(row);
    }
    two_body_obj["bb"] = three_center_bb_vec;

    j["three_center_integrals"] = two_body_obj;
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

std::unique_ptr<DensityFittedHamiltonianContainer>
DensityFittedHamiltonianContainer::from_json(const nlohmann::json& j) {
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
      int rows = matrix_vec.size();
      int cols = rows > 0 ? matrix_vec[0].size() : 0;
      Eigen::MatrixXd matrix(rows, cols);
      for (int i = 0; i < rows; ++i) {
        if (static_cast<int>(matrix_vec[i].size()) != cols) {
          throw std::runtime_error(
              "Matrix rows have inconsistent column counts");
        }
        for (int j_idx = 0; j_idx < cols; ++j_idx) {
          matrix(i, j_idx) = matrix_vec[i][j_idx];
        }
      }
      return matrix;
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
    Eigen::MatrixXd three_center_aa, three_center_bb;
    bool has_two_body = j.value("has_two_body_integrals", false);
    if (has_two_body) {
      if (!j.contains("three_center_integrals")) {
        throw std::runtime_error("Two-body integrals data not found in JSON");
      }

      auto two_body_obj = j["three_center_integrals"];
      if (!two_body_obj.is_object()) {
        throw std::runtime_error(
            "three_center_integrals must be an object with aa, bb "
            "keys");
      }

      if (!two_body_obj.contains("aa") || !two_body_obj.contains("bb")) {
        throw std::runtime_error(
            "three_center_integrals must contain aa and bb keys");
      }

      three_center_aa = load_matrix(two_body_obj["aa"]);
      three_center_bb = load_matrix(two_body_obj["bb"]);
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
      return std::make_unique<DensityFittedHamiltonianContainer>(
          one_body_alpha, three_center_aa, orbitals, core_energy,
          inactive_fock_alpha, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_unique<DensityFittedHamiltonianContainer>(
          one_body_alpha, one_body_beta, three_center_aa, three_center_bb,
          orbitals, core_energy, inactive_fock_alpha, inactive_fock_beta, type);
    }

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Hamiltonian from JSON: " +
                             std::string(e.what()));
  }
}

void DensityFittedHamiltonianContainer::to_hdf5(H5::Group& group) const {
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
      save_matrix_to_group(group, "three_center_integrals_aa",
                           *_three_center_integrals.first);
      if (is_unrestricted()) {
        save_matrix_to_group(group, "three_center_integrals_bb",
                             *_three_center_integrals.second);
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

std::unique_ptr<DensityFittedHamiltonianContainer>
DensityFittedHamiltonianContainer::from_hdf5(H5::Group& group) {
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
    Eigen::MatrixXd three_center_aa, three_center_bb;
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
    if (dataset_exists_in_group(group, "three_center_integrals_aa")) {
      three_center_aa =
          load_matrix_from_group(group, "three_center_integrals_aa");
    }

    // For unrestricted, load bb separately
    if (!is_restricted_data) {
      if (dataset_exists_in_group(group, "three_center_integrals_bb")) {
        three_center_bb =
            load_matrix_from_group(group, "three_center_integrals_bb");
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
      return std::make_unique<DensityFittedHamiltonianContainer>(
          one_body_alpha, three_center_aa, orbitals, core_energy,
          inactive_fock_alpha, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_unique<DensityFittedHamiltonianContainer>(
          one_body_alpha, one_body_beta, three_center_aa, three_center_bb,
          orbitals, core_energy, inactive_fock_alpha, inactive_fock_beta, type);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void DensityFittedHamiltonianContainer::_to_fcidump_file(
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
    auto eri = get_two_body_element(i, j, k, l, SpinChannel::aaaa);
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
