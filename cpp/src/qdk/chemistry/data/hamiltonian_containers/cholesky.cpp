// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <blas.hh>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <macis/util/fcidump.hpp>
#include <memory>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "../filename_utils.hpp"
#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

// Forward declaration of the file-local three-center SBT builder; defined
// after the constructors that delegate to the SBT-native overload.
static std::shared_ptr<const SymmetryBlockedTensor<3>> make_three_center_sbt(
    const Eigen::MatrixXd& aa, const Eigen::MatrixXd& bb,
    const Orbitals& orbitals);

CholeskyHamiltonianContainer::CholeskyHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::MatrixXd& three_center_integrals,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix,
    std::optional<Eigen::MatrixXd> ao_cholesky_vectors, HamiltonianType type)
    : CholeskyHamiltonianContainer(
          make_spin_diagonal_rank2_sbt(one_body_integrals, one_body_integrals,
                                       /*restricted=*/true),
          *make_three_center_sbt(three_center_integrals, Eigen::MatrixXd{},
                                 *orbitals),
          orbitals, core_energy,
          make_spin_diagonal_rank2_sbt(inactive_fock_matrix, Eigen::MatrixXd{}),
          std::move(ao_cholesky_vectors), type) {
  QDK_LOG_TRACE_ENTERING();
}

CholeskyHamiltonianContainer::CholeskyHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals_alpha,
    const Eigen::MatrixXd& one_body_integrals_beta,
    const Eigen::MatrixXd& three_center_integrals_aa,
    const Eigen::MatrixXd& three_center_integrals_bb,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix_alpha,
    const Eigen::MatrixXd& inactive_fock_matrix_beta,
    std::optional<Eigen::MatrixXd> ao_cholesky_vectors, HamiltonianType type)
    : CholeskyHamiltonianContainer(
          make_spin_diagonal_rank2_sbt(one_body_integrals_alpha,
                                       one_body_integrals_beta,
                                       /*restricted=*/false),
          *make_three_center_sbt(three_center_integrals_aa,
                                 three_center_integrals_bb, *orbitals),
          orbitals, core_energy,
          make_spin_diagonal_rank2_sbt(inactive_fock_matrix_alpha,
                                       inactive_fock_matrix_beta),
          std::move(ao_cholesky_vectors), type) {
  QDK_LOG_TRACE_ENTERING();
}

CholeskyHamiltonianContainer::CholeskyHamiltonianContainer(
    SymmetryBlockedTensor<2> one_body, SymmetryBlockedTensor<3> three_center,
    std::shared_ptr<Orbitals> orbitals, double core_energy,
    std::shared_ptr<const SymmetryBlockedTensor<2>> inactive_fock,
    std::optional<Eigen::MatrixXd> ao_cholesky_vectors, HamiltonianType type)
    : HamiltonianContainer(std::move(one_body), orbitals, core_energy,
                           std::move(inactive_fock), type),
      _three_center(std::make_shared<const SymmetryBlockedTensor<3>>(
          std::move(three_center))),
      _ao_cholesky_vectors(std::move(ao_cholesky_vectors)) {
  QDK_LOG_TRACE_ENTERING();

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  if (!is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid Hamiltonian object.");
  }
}

std::unique_ptr<HamiltonianContainer> CholeskyHamiltonianContainer::clone()
    const {
  QDK_LOG_TRACE_ENTERING();
  // SBT is immutable and shared via shared_ptr; pass the existing containers
  // straight through (no per-block copy or v1 round-trip needed).
  return std::make_unique<CholeskyHamiltonianContainer>(
      *_one_body, *_three_center, _orbitals, _core_energy, _inactive_fock,
      _ao_cholesky_vectors, _type);
}

std::string CholeskyHamiltonianContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "cholesky";
}

std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&>
CholeskyHamiltonianContainer::get_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Three-center integrals are not set");
  }

  // Lazily build and cache the four-center integrals on first access
  if (!std::get<0>(_cached_four_center_integrals)) {
    _build_four_center_cache();
  }

  return std::make_tuple(
      std::cref(*std::get<0>(_cached_four_center_integrals)),
      std::cref(*std::get<1>(_cached_four_center_integrals)),
      std::cref(*std::get<2>(_cached_four_center_integrals)));
}

void CholeskyHamiltonianContainer::_build_four_center_cache() const {
  QDK_LOG_TRACE_ENTERING();

  size_t norb = _orbitals->get_active_space_indices().first.size();
  size_t norb2 = norb * norb;
  size_t norb4 = norb2 * norb2;

  // 4-center build from 3-center: (ij|kl) = sum_Q L_ij,Q * R_Q,kl.
  // The two reshaped dense matrices have shape [norb*norb, naux] in
  // column-major order; the resulting 4-center has shape [norb2, norb2] in
  // column-major (= row-major (ij|kl)).
  auto build_four_center = [&](const Eigen::MatrixXd& three_left,
                               const Eigen::MatrixXd& three_right)
      -> std::shared_ptr<Eigen::VectorXd> {
    auto four_center = std::make_shared<Eigen::VectorXd>(norb4);
    size_t naux = three_left.cols();
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               norb2, norb2, naux, 1.0, three_right.data(), norb2,
               three_left.data(), norb2, 0.0, four_center->data(), norb2);
    return four_center;
  };

  const auto& tc = three_center();
  const auto& aa = tc.block({axes::alpha(), axes::alpha(), SymmetryLabel{}});
  const auto& bb = tc.has_block({axes::beta(), axes::beta(), SymmetryLabel{}})
                       ? tc.block({axes::beta(), axes::beta(), SymmetryLabel{}})
                       : aa;
  auto aaaa = build_four_center(aa, aa);

  if (is_restricted()) {
    _cached_four_center_integrals = std::make_tuple(aaaa, aaaa, aaaa);
  } else {
    auto aabb = build_four_center(aa, bb);
    auto bbbb = build_four_center(bb, bb);
    _cached_four_center_integrals =
        std::make_tuple(std::move(aaaa), std::move(aabb), std::move(bbbb));
  }
}

std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
CholeskyHamiltonianContainer::get_three_center_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error("Three-center two-body integrals are not set");
  }
  const auto& alpha =
      _three_center->block({axes::alpha(), axes::alpha(), SymmetryLabel{}});
  // Beta partner may not be stored (restricted case: aux axis is trivial-
  // symmetry so SBT orbit-aliasing does not fire). Fall back to alpha.
  if (!_three_center->has_block(
          {axes::beta(), axes::beta(), SymmetryLabel{}})) {
    return {alpha, alpha};
  }
  const auto& beta =
      _three_center->block({axes::beta(), axes::beta(), SymmetryLabel{}});
  return {alpha, beta};
}

const std::optional<Eigen::MatrixXd>&
CholeskyHamiltonianContainer::get_ao_cholesky_vectors() const {
  QDK_LOG_TRACE_ENTERING();
  return _ao_cholesky_vectors;
}

double CholeskyHamiltonianContainer::get_two_body_element(
    unsigned i, unsigned j, unsigned k, unsigned l, SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();

  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }

  size_t norb = _orbitals->get_active_space_indices().first.size();
  if (i >= norb || j >= norb || k >= norb || l >= norb) {
    throw std::out_of_range("Orbital index out of range");
  }

  if (!std::get<0>(_cached_four_center_integrals)) {
    _build_four_center_cache();
  }

  size_t ij = i * norb + j;
  size_t kl = k * norb + l;

  // Select the appropriate integral based on spin channel
  switch (channel) {
    case SpinChannel::aaaa:
      return (*std::get<0>(_cached_four_center_integrals))(ij * norb * norb +
                                                           kl);
    case SpinChannel::aabb:
      return (*std::get<1>(_cached_four_center_integrals))(ij * norb * norb +
                                                           kl);
    case SpinChannel::bbbb:
      return (*std::get<2>(_cached_four_center_integrals))(ij * norb * norb +
                                                           kl);

    default:
      throw std::invalid_argument("Invalid spin channel");
  }
}

bool CholeskyHamiltonianContainer::has_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _three_center != nullptr;
}

bool CholeskyHamiltonianContainer::is_restricted() const {
  QDK_LOG_TRACE_ENTERING();
  bool h1_restricted =
      !_one_body || _one_body->all_aliased({{{axes::alpha(), axes::alpha()},
                                             {axes::beta(), axes::beta()}}});
  bool three_center_restricted =
      !_three_center || _three_center->all_aliased(
                            {{{axes::alpha(), axes::alpha(), SymmetryLabel{}},
                              {axes::beta(), axes::beta(), SymmetryLabel{}}}});
  bool fock_restricted =
      !_inactive_fock ||
      _inactive_fock->all_aliased(
          {{{axes::alpha(), axes::alpha()}, {axes::beta(), axes::beta()}}});

  return h1_restricted && three_center_restricted && fock_restricted;
}

bool CholeskyHamiltonianContainer::is_valid() const {
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

void CholeskyHamiltonianContainer::validate_integral_dimensions() const {
  QDK_LOG_TRACE_ENTERING();
  HamiltonianContainer::validate_integral_dimensions();

  if (!has_two_body_integrals()) {
    return;
  }

  auto norb_alpha = _one_body->block({axes::alpha(), axes::alpha()}).rows();
  auto naux = _three_center->extents()[2].at(SymmetryLabel{});
  auto expected_size = static_cast<size_t>(norb_alpha * norb_alpha) * naux;

  const auto& aa =
      _three_center->block({axes::alpha(), axes::alpha(), SymmetryLabel{}});
  if (static_cast<size_t>(aa.size()) != expected_size) {
    throw std::invalid_argument("Alpha-alpha three-center integrals size (" +
                                std::to_string(aa.size()) +
                                ") does not match expected norb^2 * naux (" +
                                std::to_string(expected_size) + " for " +
                                std::to_string(norb_alpha) + " orbitals and " +
                                std::to_string(naux) + " auxiliaries)");
  }

  if (!_three_center->all_aliased(
          {{{axes::alpha(), axes::alpha(), SymmetryLabel{}},
            {axes::beta(), axes::beta(), SymmetryLabel{}}}})) {
    const auto& bb =
        _three_center->block({axes::beta(), axes::beta(), SymmetryLabel{}});
    if (static_cast<size_t>(bb.size()) != expected_size) {
      throw std::invalid_argument(
          "Beta three-center integrals size does not match Alpha");
    }
  }
}

// ---- SBT-canonical container builders --------------------------------------

// Build the canonical rank-3 three-center SBT from dense alpha (and optional
// beta) blocks, sharing MO symmetry/extents with @p orbitals' active space.
// Returns @c nullptr when @p aa is empty (no data supplied). When @p bb is
// empty the spin axis is restricted and the alpha block is aliased into the
// beta slot via partner-block aliasing in @ref SymmetryBlockedTensor.
static std::shared_ptr<const SymmetryBlockedTensor<3>> make_three_center_sbt(
    const Eigen::MatrixXd& aa, const Eigen::MatrixXd& bb,
    const Orbitals& orbitals) {
  if (aa.size() == 0) {
    return nullptr;
  }
  auto mo_sym = orbitals.symmetries();
  auto active_indices = orbitals.get_active_space_indices();
  std::size_t n_active_alpha = active_indices.first.size();
  std::size_t n_active_beta = active_indices.second.size();
  std::size_t naux = static_cast<std::size_t>(aa.cols());

  std::unordered_map<SymmetryLabel, std::size_t> mo_ext;
  mo_ext[axes::alpha()] = n_active_alpha;
  mo_ext[axes::beta()] = n_active_beta;

  auto aux_sym =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  std::unordered_map<SymmetryLabel, std::size_t> aux_ext;
  aux_ext[SymmetryLabel{}] = naux;

  SymmetryBlockedTensor<3>::SymmetriesArray symmetries = {mo_sym, mo_sym,
                                                          aux_sym};
  SymmetryBlockedTensor<3>::ExtentsArray extents = {mo_ext, mo_ext, aux_ext};

  if (static_cast<std::size_t>(aa.rows()) != n_active_alpha * n_active_alpha) {
    throw std::invalid_argument(
        "Alpha three-center rows does not match n_active_alpha^2");
  }

  // Rank-3 SBT block is the dense [orb_pair, aux] MatrixXd verbatim — no
  // copy or reshape needed.
  auto aa_block = std::make_shared<const Eigen::MatrixXd>(aa);
  SymmetryBlockedTensor<3>::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha(), SymmetryLabel{}}] = aa_block;

  if (bb.size() != 0) {
    if (static_cast<std::size_t>(bb.rows()) != n_active_beta * n_active_beta) {
      throw std::invalid_argument(
          "Beta three-center rows does not match n_active_beta^2");
    }
    if (static_cast<std::size_t>(bb.cols()) != naux) {
      throw std::invalid_argument(
          "Beta three-center cols does not match alpha naux");
    }
    auto bb_block = std::make_shared<const Eigen::MatrixXd>(bb);
    blocks[{axes::beta(), axes::beta(), SymmetryLabel{}}] = bb_block;
  }

  return std::make_shared<const SymmetryBlockedTensor<3>>(
      std::move(symmetries), std::move(extents), std::move(blocks));
}

const SymmetryBlockedTensor<3>& CholeskyHamiltonianContainer::three_center()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (!_three_center) {
    throw std::runtime_error(
        "Three-center symmetry-blocked tensor is not set.");
  }
  return *_three_center;
}

nlohmann::json CholeskyHamiltonianContainer::to_json() const {
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
  if (_three_center) {
    j["three_center_integrals"] = _three_center->to_json();
  }
  if (_inactive_fock) {
    j["inactive_fock_matrix"] = _inactive_fock->to_json();
  }

  // Store orbital data
  if (has_orbitals()) {
    j["orbitals"] = _orbitals->to_json();
  }

  // Store AO Cholesky vectors (if available)
  if (_ao_cholesky_vectors) {
    std::vector<std::vector<double>> ao_cholesky_vectors_vec;
    for (int i = 0; i < _ao_cholesky_vectors->rows(); ++i) {
      std::vector<double> row;
      for (int j_idx = 0; j_idx < _ao_cholesky_vectors->cols(); ++j_idx) {
        row.push_back((*_ao_cholesky_vectors)(i, j_idx));
      }
      ao_cholesky_vectors_vec.push_back(row);
    }
    j["ao_cholesky_vectors"] = ao_cholesky_vectors_vec;
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

    if (!j.contains("three_center_integrals")) {
      throw std::runtime_error(
          "Hamiltonian JSON must include three_center_integrals");
    }
    auto three_center =
        SymmetryBlockedTensor<3>::from_json(j["three_center_integrals"]);

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

    std::optional<Eigen::MatrixXd> ao_cholesky_vectors;
    if (j.contains("ao_cholesky_vectors")) {
      auto matrix_vec =
          j["ao_cholesky_vectors"].get<std::vector<std::vector<double>>>();
      int rows = matrix_vec.size();
      int cols = rows > 0 ? matrix_vec[0].size() : 0;
      Eigen::MatrixXd matrix(rows, cols);
      for (int i = 0; i < rows; ++i) {
        for (int jj = 0; jj < cols; ++jj) {
          matrix(i, jj) = matrix_vec[i][jj];
        }
      }
      ao_cholesky_vectors = std::move(matrix);
    }

    return std::make_unique<CholeskyHamiltonianContainer>(
        std::move(*one_body), std::move(*three_center), orbitals, core_energy,
        std::move(inactive_fock), std::move(ao_cholesky_vectors), type);

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Hamiltonian from JSON: " +
                             std::string(e.what()));
  }
}

void CholeskyHamiltonianContainer::to_hdf5(H5::Group& group) const {
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
    if (_three_center) {
      H5::Group sub = group.createGroup("three_center_integrals");
      _three_center->to_hdf5(sub);
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

    // Save AO Cholesky vectors (if available)
    if (_ao_cholesky_vectors) {
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

    if (!group.nameExists("three_center_integrals")) {
      throw std::runtime_error(
          "Hamiltonian HDF5 must include three_center_integrals");
    }
    H5::Group tc_group = group.openGroup("three_center_integrals");
    auto three_center = SymmetryBlockedTensor<3>::from_hdf5(tc_group);

    std::shared_ptr<const SymmetryBlockedTensor<2>> inactive_fock;
    if (group.nameExists("inactive_fock_matrix")) {
      H5::Group fock_group = group.openGroup("inactive_fock_matrix");
      inactive_fock = SymmetryBlockedTensor<2>::from_hdf5(fock_group);
    }

    // Load AO Cholesky vectors (if available)
    std::optional<Eigen::MatrixXd> ao_cholesky_vectors;
    if (dataset_exists_in_group(group, "ao_cholesky_vectors")) {
      ao_cholesky_vectors =
          load_matrix_from_group(group, "ao_cholesky_vectors");
    }

    return std::make_unique<CholeskyHamiltonianContainer>(
        std::move(*one_body), std::move(*three_center), orbitals, core_energy,
        std::move(inactive_fock), std::move(ao_cholesky_vectors), type);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
