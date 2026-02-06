// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "../filename_utils.hpp"
#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

CholeskyHamiltonianContainer::CholeskyHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::VectorXd& two_body_integrals,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix, const Eigen::MatrixXd& L_ao,
    HamiltonianType type)
    : CanonicalFourCenterHamiltonianContainer(
          one_body_integrals, two_body_integrals, orbitals, core_energy,
          inactive_fock_matrix, type),
      _ao_cholesky_vectors(std::make_shared<Eigen::MatrixXd>(L_ao)) {
  QDK_LOG_TRACE_ENTERING();
}

CholeskyHamiltonianContainer::CholeskyHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals_alpha,
    const Eigen::MatrixXd& one_body_integrals_beta,
    const Eigen::VectorXd& two_body_integrals_aaaa,
    const Eigen::VectorXd& two_body_integrals_aabb,
    const Eigen::VectorXd& two_body_integrals_bbbb,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix_alpha,
    const Eigen::MatrixXd& inactive_fock_matrix_beta,
    const Eigen::MatrixXd& L_ao, HamiltonianType type)
    : CanonicalFourCenterHamiltonianContainer(
          one_body_integrals_alpha, one_body_integrals_beta,
          two_body_integrals_aaaa, two_body_integrals_aabb,
          two_body_integrals_bbbb, orbitals, core_energy,
          inactive_fock_matrix_alpha, inactive_fock_matrix_beta, type),
      _ao_cholesky_vectors(std::make_shared<Eigen::MatrixXd>(L_ao)) {
  QDK_LOG_TRACE_ENTERING();
}

std::unique_ptr<HamiltonianContainer> CholeskyHamiltonianContainer::clone()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (is_restricted()) {
    return std::make_unique<CholeskyHamiltonianContainer>(
        *_one_body_integrals.first, *std::get<0>(_two_body_integrals),
        _orbitals, _core_energy, *_inactive_fock_matrix.first,
        *_ao_cholesky_vectors, _type);
  }
  return std::make_unique<CholeskyHamiltonianContainer>(
      *_one_body_integrals.first, *_one_body_integrals.second,
      *std::get<0>(_two_body_integrals), *std::get<1>(_two_body_integrals),
      *std::get<2>(_two_body_integrals), _orbitals, _core_energy,
      *_inactive_fock_matrix.first, *_inactive_fock_matrix.second,
      *_ao_cholesky_vectors, _type);
}

std::string CholeskyHamiltonianContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "cholesky";
}

nlohmann::json CholeskyHamiltonianContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  // Start with base class serialization
  nlohmann::json j = CanonicalFourCenterHamiltonianContainer::to_json();

  // Override container type
  j["container_type"] = get_container_type();

  // Store ao cholesky vectors (Cholesky-specific data)
  if (_ao_cholesky_vectors != nullptr && _ao_cholesky_vectors->size() > 0) {
    j["has_ao_cholesky_vectors"] = true;
    // Store ao cholesky vectors
    std::vector<std::vector<double>> L_ao_vec;
    for (int i = 0; i < _ao_cholesky_vectors->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _ao_cholesky_vectors->cols(); ++j_idx) {
        row.push_back((*_ao_cholesky_vectors)(i, j_idx));
      }
      L_ao_vec.push_back(row);
    }
    j["ao_cholesky_vectors"] = L_ao_vec;
  } else {
    j["has_ao_cholesky_vectors"] = false;
  }

  return j;
}

std::unique_ptr<CholeskyHamiltonianContainer>
CholeskyHamiltonianContainer::from_json(const nlohmann::json& j) {
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

    // Load ao cholesky vectors
    Eigen::MatrixXd L_ao;
    bool has_ao_cholesky_vectors = j.value("has_ao_cholesky_vectors", false);
    if (has_ao_cholesky_vectors) {
      if (j.contains("ao_cholesky_vectors")) {
        L_ao = load_matrix(j["ao_cholesky_vectors"]);
      }
    }

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
      return std::make_unique<CholeskyHamiltonianContainer>(
          one_body_alpha, two_body_aaaa, orbitals, core_energy,
          inactive_fock_alpha, L_ao, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_unique<CholeskyHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, orbitals, core_energy, inactive_fock_alpha,
          inactive_fock_beta, L_ao, type);
    }

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Hamiltonian from JSON: " +
                             std::string(e.what()));
  }
}

void CholeskyHamiltonianContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    // Start with base class serialization
    CanonicalFourCenterHamiltonianContainer::to_hdf5(group);

    // Override container type attribute
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Remove and recreate container_type attribute with correct value
    if (group.attrExists("container_type")) {
      group.removeAttr("container_type");
    }
    H5::Attribute container_type_attr =
        group.createAttribute("container_type", string_type, scalar_space);
    std::string container_type_str = get_container_type();
    container_type_attr.write(string_type, container_type_str);

    // Save ao cholesky vectors (Cholesky-specific data)
    if (_ao_cholesky_vectors != nullptr && _ao_cholesky_vectors->size() > 0) {
      save_matrix_to_group(group, "ao_cholesky_vectors", *_ao_cholesky_vectors);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<CholeskyHamiltonianContainer>
CholeskyHamiltonianContainer::from_hdf5(H5::Group& group) {
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

    // load ao cholesky vectors
    Eigen::MatrixXd L_ao;
    if (dataset_exists_in_group(group, "ao_cholesky_vectors")) {
      L_ao = load_matrix_from_group(group, "ao_cholesky_vectors");
    }

    // Create and return appropriate Hamiltonian using the correct constructor
    if (is_restricted_data) {
      // Use restricted constructor - it will create shared pointers internally
      return std::make_unique<CholeskyHamiltonianContainer>(
          one_body_alpha, two_body_aaaa, orbitals, core_energy,
          inactive_fock_alpha, L_ao, type);
    } else {
      // Use unrestricted constructor with separate alpha and beta data
      return std::make_unique<CholeskyHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, orbitals, core_energy, inactive_fock_alpha,
          inactive_fock_beta, L_ao, type);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
