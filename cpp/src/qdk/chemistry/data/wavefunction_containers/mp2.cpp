/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include "../../algorithms/microsoft/mp2.hpp"

#include <optional>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <qdk/chemistry/utils/tensor.hpp>
#include <qdk/chemistry/utils/tensor_span.hpp>
#include <stdexcept>

#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

MP2Container::MP2Container(std::shared_ptr<Hamiltonian> hamiltonian,
                           std::shared_ptr<Wavefunction> wavefunction,
                           const std::string& partitioning)
    // Mp2 is always not self dual
    : WavefunctionContainer(WavefunctionType::NotSelfDual),
      _wavefunction(wavefunction),
      _hamiltonian(hamiltonian) {
  QDK_LOG_TRACE_ENTERING();
  if (!hamiltonian) {
    throw std::invalid_argument("Hamiltonian cannot be null");
  }

  if (wavefunction->get_total_determinants().empty()) {
    throw std::invalid_argument("Reference determinants cannot be empty");
  }

  if (partitioning != "mp") {
    throw std::runtime_error(
        "Only Moeller-Plesset Hamiltonian partitioning is implemented");
  }
}

std::unique_ptr<WavefunctionContainer> MP2Container::clone() const {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<MP2Container>(_hamiltonian, _wavefunction);
}

std::shared_ptr<Orbitals> MP2Container::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _hamiltonian->get_orbitals();
}

std::shared_ptr<Hamiltonian> MP2Container::get_hamiltonian() const {
  QDK_LOG_TRACE_ENTERING();
  return _hamiltonian;
}

std::shared_ptr<Wavefunction> MP2Container::get_wavefunction() const {
  QDK_LOG_TRACE_ENTERING();
  return _wavefunction;
}

const MP2Container::VectorVariant& MP2Container::get_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_coefficients() is not implemented for MP2 wavefunctions.");
}

MP2Container::ScalarVariant MP2Container::get_coefficient(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_coefficient() is not implemented for MP2 wavefunctions.");
}

const MP2Container::DeterminantVector& MP2Container::get_active_determinants()
    const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_active_determinants() is not implemented for MP2 wavefunctions.");
}

void MP2Container::_compute_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if T1 amplitudes are already computed
  if (_t1_amplitudes_aa && _t1_amplitudes_bb) {
    return;
  }

  // Get electron counts
  auto [n_alpha, n_beta] = get_active_num_electrons();

  // Get active space size
  int active_space_size = get_orbitals()->get_num_molecular_orbitals();
  if (get_orbitals()->has_active_space()) {
    const auto& [active_space_ind_alpha, active_space_ind_beta] =
        get_orbitals()->get_active_space_indices();
    active_space_size = active_space_ind_alpha.size();
  }

  const size_t n_vir_alpha = active_space_size - n_alpha;
  const size_t n_vir_beta = active_space_size - n_beta;

  // For MP2, T1 amplitudes are always zero
  Eigen::MatrixXd t1_aa = Eigen::MatrixXd::Zero(n_alpha, n_vir_alpha);
  Eigen::MatrixXd t1_bb = Eigen::MatrixXd::Zero(n_beta, n_vir_beta);
  _t1_amplitudes_aa = std::make_shared<MatrixVariant>(std::move(t1_aa));
  _t1_amplitudes_bb = std::make_shared<MatrixVariant>(std::move(t1_bb));
}

void MP2Container::_compute_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if amplitudes are already computed
  if (_t2_amplitudes_abab && _t2_amplitudes_aaaa && _t2_amplitudes_bbbb) {
    return;
  }

  // Get electron counts
  auto [n_alpha, n_beta] = get_active_num_electrons();

  // Check if we need unrestricted calculation
  bool use_unrestricted =
      _hamiltonian->is_unrestricted() || (n_alpha != n_beta);

  if (!get_orbitals()->has_energies()) {
    throw std::runtime_error(
        "Orbital energies are required for MP2 amplitude calculation");
  }

  // Get active space size
  int active_space_size = get_orbitals()->get_num_molecular_orbitals();
  if (get_orbitals()->has_active_space()) {
    const auto& [active_space_ind_alpha, active_space_ind_beta] =
        get_orbitals()->get_active_space_indices();
    active_space_size = active_space_ind_alpha.size();
  }

  if (use_unrestricted) {
    // Unrestricted MP2
    const auto& [eps_alpha, eps_beta] = get_orbitals()->get_energies();
    const size_t n_vir_alpha = active_space_size - n_alpha;
    const size_t n_vir_beta = active_space_size - n_beta;

    // Get two-electron integrals as 4D tensor views
    auto [mo_aaaa, mo_aabb, mo_bbbb] = _hamiltonian->get_two_body_integrals();

    // Initialize T2 amplitudes as 4D tensors
    auto t2_aa = std::make_shared<TensorVariant>(
        qdk::chemistry::make_rank4_tensor<double>(n_alpha, n_alpha, n_vir_alpha,
                                                  n_vir_alpha));
    auto t2_ab = std::make_shared<TensorVariant>(
        qdk::chemistry::make_rank4_tensor<double>(n_alpha, n_beta, n_vir_alpha,
                                                  n_vir_beta));
    auto t2_bb = std::make_shared<TensorVariant>(
        qdk::chemistry::make_rank4_tensor<double>(n_beta, n_beta, n_vir_beta,
                                                  n_vir_beta));

    // Alpha-Alpha contribution
    algorithms::microsoft::MP2Calculator::compute_same_spin_t2(
        eps_alpha, mo_aaaa,
        std::get<qdk::chemistry::rank4_tensor<double>>(*t2_aa));

    // Alpha-Beta contribution
    algorithms::microsoft::MP2Calculator::compute_opposite_spin_t2(
        eps_alpha, eps_beta, mo_aabb,
        std::get<qdk::chemistry::rank4_tensor<double>>(*t2_ab));

    // Beta-Beta contribution
    algorithms::microsoft::MP2Calculator::compute_same_spin_t2(
        eps_beta, mo_bbbb,
        std::get<qdk::chemistry::rank4_tensor<double>>(*t2_bb));

    _t2_amplitudes_abab = t2_ab;
    _t2_amplitudes_aaaa = t2_aa;
    _t2_amplitudes_bbbb = t2_bb;
  } else {
    // Restricted MP2
    if (n_alpha != n_beta) {
      throw std::runtime_error(
          "Restricted MP2 requires equal alpha and beta electrons");
    }

    const auto& [eps_alpha, eps_beta] = get_orbitals()->get_energies();
    const size_t n_occ = n_alpha;
    const size_t n_vir = active_space_size - n_occ;

    // Get two-electron integrals as 4D tensor view
    auto [mo_aaaa, mo_aabb, mo_bbbb] = _hamiltonian->get_two_body_integrals();

    // Initialize T2 amplitudes as 4D tensor wrapped in TensorVariant
    auto t2_amplitudes = std::make_shared<TensorVariant>(
        qdk::chemistry::make_rank4_tensor<double>(n_occ, n_occ, n_vir, n_vir));

    // Compute T2 amplitudes
    algorithms::microsoft::MP2Calculator::compute_restricted_t2(
        eps_alpha, mo_aaaa,
        std::get<qdk::chemistry::rank4_tensor<double>>(*t2_amplitudes));

    _t2_amplitudes_abab = t2_amplitudes;
    _t2_amplitudes_aaaa = t2_amplitudes;
    _t2_amplitudes_bbbb = t2_amplitudes;
  }
}

std::pair<const MP2Container::MatrixVariant&,
          const MP2Container::MatrixVariant&>
MP2Container::get_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_t1_amplitudes_aa) {
    _compute_t1_amplitudes();
  }

  return std::make_pair(std::cref(*_t1_amplitudes_aa),
                        std::cref(*_t1_amplitudes_bb));
}

std::tuple<const MP2Container::TensorVariant&,
           const MP2Container::TensorVariant&,
           const MP2Container::TensorVariant&>
MP2Container::get_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_t2_amplitudes_abab) {
    _compute_t2_amplitudes();
  }

  return std::make_tuple(std::cref(*_t2_amplitudes_abab),
                         std::cref(*_t2_amplitudes_aaaa),
                         std::cref(*_t2_amplitudes_bbbb));
}

bool MP2Container::has_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t1_amplitudes_aa != nullptr;
}

bool MP2Container::has_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t2_amplitudes_abab != nullptr;
}

size_t MP2Container::size() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error("size() is not meaningful for MP2 wavefunctions.");
}

MP2Container::ScalarVariant MP2Container::overlap(
    const WavefunctionContainer& other) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "overlap() is not implemented for MP2 wavefunctions.");
}

double MP2Container::norm() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error("norm() is not implemented for MP2 wavefunctions.");
}

bool MP2Container::contains_determinant(const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "contains_determinant() is not implemented for MP2 wavefunctions.");
}

bool MP2Container::contains_reference(const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& references = _wavefunction->get_total_determinants();
  return std::find(references.begin(), references.end(), det) !=
         references.end();
}

void MP2Container::clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  _t1_amplitudes_aa = nullptr;
  _t1_amplitudes_bb = nullptr;
  _t2_amplitudes_abab = nullptr;
  _t2_amplitudes_aaaa = nullptr;
  _t2_amplitudes_bbbb = nullptr;

  _determinant_vector_cache = nullptr;
}

nlohmann::json MP2Container::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["type"] = "mp2";
  j["version"] = SERIALIZATION_VERSION;

  // Serialize orbitals
  j["orbitals"] = get_orbitals()->to_json();

  // Serialize Hamiltonian
  j["hamiltonian"] = _hamiltonian->to_json();

  // Serialize wavefunction
  j["wavefunction"] = _wavefunction->to_json();

  // Note: We don't serialize amplitudes in JSON - they are computed on
  // demand when Hamiltonian is available

  return j;
}

std::unique_ptr<MP2Container> MP2Container::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  // Deserialize orbitals
  auto orbitals = Orbitals::from_json(j.at("orbitals"));

  // Deserialize Hamiltonian (can be null)
  std::shared_ptr<Hamiltonian> hamiltonian = nullptr;
  if (!j.at("hamiltonian").is_null()) {
    hamiltonian = Hamiltonian::from_json(j.at("hamiltonian"));
  }

  // Deserialize Wavefunction (can be null)
  std::shared_ptr<Wavefunction> wavefunction = nullptr;
  if (!j.at("wavefunction").is_null()) {
    wavefunction = Wavefunction::from_json(j.at("wavefunction"));
  }

  return std::make_unique<MP2Container>(hamiltonian, wavefunction);
}

void MP2Container::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // version
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);
    version_attr.close();

    // container type
    std::string container_type = get_container_type();
    H5::Attribute container_type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    container_type_attr.write(string_type, container_type);

    // complex flag
    bool is_complex_flag = this->is_complex();
    H5::Attribute is_complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    hbool_t is_complex_hbool = is_complex_flag ? 1 : 0;
    is_complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex_hbool);

    // Store wfn if available
    if (_wavefunction) {
      H5::Group wavefunction_group = group.createGroup("wavefunction");
      _wavefunction->to_hdf5(wavefunction_group);
    }

    // Store Hamiltonian if available
    if (_hamiltonian) {
      H5::Group hamiltonian_group = group.createGroup("hamiltonian");
      _hamiltonian->to_hdf5(hamiltonian_group);
    }

    // Note: We don't serialize amplitudes - they are computed on demand
    // when Hamiltonian is available

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<MP2Container> MP2Container::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // version
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    // Load Hamiltonian (if available)
    std::shared_ptr<Hamiltonian> hamiltonian = nullptr;
    if (group.nameExists("hamiltonian")) {
      H5::Group hamiltonian_group = group.openGroup("hamiltonian");
      hamiltonian = Hamiltonian::from_hdf5(hamiltonian_group);
    }

    // Load wavefunction (if available)
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (group.nameExists("wavefunction")) {
      H5::Group wavefunction_group = group.openGroup("wavefunction");
      wavefunction = Wavefunction::from_hdf5(wavefunction_group);
    }

    return std::make_unique<MP2Container>(hamiltonian, wavefunction);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::pair<size_t, size_t> MP2Container::get_total_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  // Get from first reference determinant
  const auto& references = _wavefunction->get_total_determinants();
  if (references.size() == 0) {
    throw std::runtime_error("No reference determinants available");
  }
  const auto& first_ref = references[0];
  auto [n_alpha, n_beta] = first_ref.get_n_electrons();
  return std::make_pair(n_alpha, n_beta);
}

std::pair<size_t, size_t> MP2Container::get_active_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  return get_total_num_electrons();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MP2Container::get_total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_total_orbital_occupations() is not implemented for MP2 "
      "wavefunctions.");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MP2Container::get_active_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_active_orbital_occupations() is not implemented for MP2 "
      "wavefunctions.");
}

std::string MP2Container::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "mp2";
}

bool MP2Container::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  // MP2 amplitudes are always real (Eigen::MatrixXd and rank4_tensor<double>)
  return false;
}

}  // namespace qdk::chemistry::data
