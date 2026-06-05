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
    : CanonicalFourCenterHamiltonianContainer(
          make_spin_diagonal_rank2_sbt(one_body_integrals, one_body_integrals,
                                       /*restricted=*/true),
          make_spin_diagonal_rank4_sbt(two_body_integrals), orbitals,
          core_energy,
          make_spin_diagonal_rank2_sbt(inactive_fock_matrix, Eigen::MatrixXd{}),
          type) {
  QDK_LOG_TRACE_ENTERING();
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
    : CanonicalFourCenterHamiltonianContainer(
          make_spin_diagonal_rank2_sbt(one_body_integrals_alpha,
                                       one_body_integrals_beta,
                                       /*restricted=*/false),
          make_spin_diagonal_rank4_sbt(two_body_integrals_aaaa,
                                       two_body_integrals_aabb,
                                       two_body_integrals_bbbb,
                                       /*restricted=*/false),
          orbitals, core_energy,
          make_spin_diagonal_rank2_sbt(inactive_fock_matrix_alpha,
                                       inactive_fock_matrix_beta),
          type) {
  QDK_LOG_TRACE_ENTERING();
}

CanonicalFourCenterHamiltonianContainer::
    CanonicalFourCenterHamiltonianContainer(
        SymmetryBlockedTensor<2> one_body, SymmetryBlockedTensor<4> two_body,
        std::shared_ptr<Orbitals> orbitals, double core_energy,
        std::shared_ptr<const SymmetryBlockedTensor<2>> inactive_fock,
        HamiltonianType type)
    : HamiltonianContainer(std::move(one_body), orbitals, core_energy,
                           std::move(inactive_fock), type),
      _two_body(std::make_shared<const SymmetryBlockedTensor<4>>(
          std::move(two_body))) {
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
  // SBT is immutable and shared via shared_ptr; pass the existing containers
  // straight through (no per-block copy or v1 round-trip needed).
  return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      *_one_body, *_two_body, _orbitals, _core_energy, _inactive_fock, _type);
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

  return std::make_tuple(
      std::cref(_two_body->block(
          {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()})),
      std::cref(_two_body->block(
          {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()})),
      std::cref(_two_body->block(
          {axes::beta(), axes::beta(), axes::beta(), axes::beta()})));
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

  switch (channel) {
    case SpinChannel::aaaa:
      return _two_body->block(
          {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()})[index];
    case SpinChannel::aabb:
      return _two_body->block(
          {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()})[index];
    case SpinChannel::bbbb:
      return _two_body->block(
          {axes::beta(), axes::beta(), axes::beta(), axes::beta()})[index];
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
  return _two_body != nullptr;
}

bool CanonicalFourCenterHamiltonianContainer::is_restricted() const {
  QDK_LOG_TRACE_ENTERING();

  bool h1_restricted =
      !_one_body || _one_body->all_aliased({{{axes::alpha(), axes::alpha()},
                                             {axes::beta(), axes::beta()}}});
  bool fock_restricted =
      !_inactive_fock ||
      _inactive_fock->all_aliased(
          {{{axes::alpha(), axes::alpha()}, {axes::beta(), axes::beta()}}});
  bool h2_restricted =
      !_two_body ||
      _two_body->all_aliased(
          {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()},
            {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()},
            {axes::beta(), axes::beta(), axes::beta(), axes::beta()}}});

  return h1_restricted && h2_restricted && fock_restricted;
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
  HamiltonianContainer::validate_integral_dimensions();

  if (!has_two_body_integrals()) {
    return;
  }

  size_t norb_alpha = static_cast<size_t>(
      _one_body->block({axes::alpha(), axes::alpha()}).rows());
  size_t expected_size = norb_alpha * norb_alpha * norb_alpha * norb_alpha;

  const auto& aaaa = _two_body->block(
      {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()});
  if (static_cast<size_t>(aaaa.size()) != expected_size) {
    throw std::invalid_argument(
        "Alpha-alpha two-body integrals size (" + std::to_string(aaaa.size()) +
        ") does not match expected size (" + std::to_string(expected_size) +
        ") for " + std::to_string(norb_alpha) + " orbitals");
  }

  // Check alpha-beta integrals (if different from alpha-alpha)
  if (!_two_body->all_aliased(
          {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()},
            {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}}})) {
    const auto& aabb = _two_body->block(
        {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()});
    if (static_cast<size_t>(aabb.size()) != expected_size) {
      throw std::invalid_argument(
          "Alpha-beta two-body integrals size mismatch");
    }
  }

  // Check beta-beta integrals (if different from alpha-alpha)
  if (!_two_body->all_aliased(
          {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()},
            {axes::beta(), axes::beta(), axes::beta(), axes::beta()}}})) {
    const auto& bbbb = _two_body->block(
        {axes::beta(), axes::beta(), axes::beta(), axes::beta()});
    if (static_cast<size_t>(bbbb.size()) != expected_size) {
      throw std::invalid_argument("Beta-beta two-body integrals size mismatch");
    }
  }
}

const SymmetryBlockedTensor<4>&
CanonicalFourCenterHamiltonianContainer::two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_two_body) {
    throw std::runtime_error("Two-body symmetry-blocked tensor is not set.");
  }
  return *_two_body;
}

const Eigen::VectorXd&
CanonicalFourCenterHamiltonianContainer::two_body_integrals_block(
    const SymmetryLabel& p, const SymmetryLabel& q, const SymmetryLabel& r,
    const SymmetryLabel& s) const {
  return two_body_integrals().block({p, q, r, s});
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
  j["is_restricted"] = is_restricted();

  // Store integrals via SBT-direct serialization
  if (_one_body) {
    j["one_body_integrals"] = _one_body->to_json();
  }
  if (_two_body) {
    j["two_body_integrals"] = _two_body->to_json();
  }
  if (_inactive_fock) {
    j["inactive_fock_matrix"] = _inactive_fock->to_json();
  }

  // Store orbital data
  if (has_orbitals()) {
    j["orbitals"] = _orbitals->to_json();
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
    if (j.contains("type") && j["type"].get<std::string>() == "NonHermitian") {
      type = HamiltonianType::NonHermitian;
    }

    // Load orbital data
    if (!j.contains("orbitals")) {
      throw std::runtime_error("Hamiltonian JSON must include orbitals data");
    }
    auto orbitals = Orbitals::from_json(j["orbitals"]);

    // Load integrals via SBT-direct deserialization
    if (!j.contains("one_body_integrals")) {
      throw std::runtime_error(
          "Hamiltonian JSON must include one_body_integrals");
    }
    auto one_body =
        SymmetryBlockedTensor<2>::from_json(j["one_body_integrals"]);

    if (!j.contains("two_body_integrals")) {
      throw std::runtime_error(
          "Hamiltonian JSON must include two_body_integrals");
    }
    auto two_body =
        SymmetryBlockedTensor<4>::from_json(j["two_body_integrals"]);

    if (orbitals->has_inactive_space()) {
      if (!j.contains("inactive_fock_matrix")) {
        throw std::runtime_error(
            "Hamiltonian JSON: orbitals have inactive indices but no "
            "inactive Fock matrix is provided");
      }
      if (!j.contains("core_energy")) {
        throw std::runtime_error(
            "Hamiltonian JSON: orbitals have inactive indices but no core "
            "energy is provided");
      }
    }

    std::shared_ptr<const SymmetryBlockedTensor<2>> inactive_fock =
        j.contains("inactive_fock_matrix")
            ? SymmetryBlockedTensor<2>::from_json(j["inactive_fock_matrix"])
            : nullptr;

    return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
        std::move(*one_body), std::move(*two_body), orbitals, core_energy,
        std::move(inactive_fock), type);

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

    // Save integrals via SBT-direct serialization
    if (_one_body) {
      H5::Group sub = group.createGroup("one_body_integrals");
      _one_body->to_hdf5(sub);
    }
    if (_two_body) {
      H5::Group sub = group.createGroup("two_body_integrals");
      _two_body->to_hdf5(sub);
    }
    if (_inactive_fock) {
      H5::Group sub = group.createGroup("inactive_fock_matrix");
      _inactive_fock->to_hdf5(sub);
    }

    // Save nested orbitals data
    if (has_orbitals()) {
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
    metadata_group.openAttribute("core_energy")
        .read(H5::PredType::NATIVE_DOUBLE, &core_energy);

    // Load Hamiltonian type
    HamiltonianType type = HamiltonianType::Hermitian;
    if (metadata_group.attrExists("type")) {
      H5::Attribute type_attr = metadata_group.openAttribute("type");
      std::string type_str;
      type_attr.read(type_attr.getStrType(), type_str);
      if (type_str == "NonHermitian") {
        type = HamiltonianType::NonHermitian;
      }
    }

    // Load orbital data
    if (!group.nameExists("orbitals")) {
      throw std::runtime_error("Hamiltonian HDF5 must include orbitals data");
    }
    H5::Group orbitals_group = group.openGroup("orbitals");
    auto orbitals = Orbitals::from_hdf5(orbitals_group);

    // Load integrals via SBT-direct deserialization
    if (!group.nameExists("one_body_integrals")) {
      throw std::runtime_error(
          "Hamiltonian HDF5 must include one_body_integrals");
    }
    H5::Group one_body_group = group.openGroup("one_body_integrals");
    auto one_body = SymmetryBlockedTensor<2>::from_hdf5(one_body_group);

    if (!group.nameExists("two_body_integrals")) {
      throw std::runtime_error(
          "Hamiltonian HDF5 must include two_body_integrals");
    }
    H5::Group two_body_group = group.openGroup("two_body_integrals");
    auto two_body = SymmetryBlockedTensor<4>::from_hdf5(two_body_group);

    std::shared_ptr<const SymmetryBlockedTensor<2>> inactive_fock;
    if (group.nameExists("inactive_fock_matrix")) {
      H5::Group fock_group = group.openGroup("inactive_fock_matrix");
      inactive_fock = SymmetryBlockedTensor<2>::from_hdf5(fock_group);
    }

    return std::make_unique<CanonicalFourCenterHamiltonianContainer>(
        std::move(*one_body), std::move(*two_body), orbitals, core_energy,
        std::move(inactive_fock), type);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
