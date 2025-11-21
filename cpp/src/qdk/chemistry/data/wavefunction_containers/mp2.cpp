/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <optional>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>
#include <stdexcept>
#include <variant>

#include "../../algorithms/microsoft/mp2_amplitude_helpers.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

MP2Container::MP2Container(std::shared_ptr<Hamiltonian> hamiltonian,
                           const DeterminantVector& references,
                           WavefunctionType type)
    : WavefunctionContainer(type),
      _references(references, hamiltonian->get_orbitals()),
      _hamiltonian(hamiltonian) {
  if (!hamiltonian) {
    throw std::invalid_argument("Hamiltonian cannot be null");
  }

  if (references.empty()) {
    throw std::invalid_argument("Reference determinants cannot be empty");
  }
}

std::unique_ptr<WavefunctionContainer> MP2Container::clone() const {
  return std::make_unique<MP2Container>(
      _hamiltonian, _references.get_configurations(), _type);
}

std::shared_ptr<Orbitals> MP2Container::get_orbitals() const {
  return MP2Container::get_hamiltonian()->get_orbitals();
}

std::shared_ptr<Hamiltonian> MP2Container::get_hamiltonian() const {
  return _hamiltonian;
}

const MP2Container::VectorVariant& MP2Container::get_coefficients() const {
  throw std::runtime_error(
      "get_coefficients() is not implemented for MP2 wavefunctions.");
}

MP2Container::ScalarVariant MP2Container::get_coefficient(
    const Configuration& det) const {
  throw std::runtime_error(
      "get_coefficient() is not implemented for MP2 wavefunctions.");
}

const MP2Container::DeterminantVector& MP2Container::get_references() const {
  return _references.get_configurations();
}

const MP2Container::DeterminantVector& MP2Container::get_active_determinants()
    const {
  throw std::runtime_error(
      "get_active_determinants() is not implemented for MP2 wavefunctions.");
}

void MP2Container::_compute_t1_amplitudes() const {
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
  Eigen::VectorXd t1_aa_vec = Eigen::VectorXd::Zero(n_alpha * n_vir_alpha);
  Eigen::VectorXd t1_bb_vec = Eigen::VectorXd::Zero(n_beta * n_vir_beta);

  _t1_amplitudes_aa = std::make_shared<VectorVariant>(t1_aa_vec);
  _t1_amplitudes_bb = std::make_shared<VectorVariant>(t1_bb_vec);
}

void MP2Container::_compute_t2_amplitudes() const {
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

    const auto& [moeri_aaaa, moeri_aabb, moeri_bbbb] =
        _hamiltonian->get_two_body_integrals();

    // Initialize T2 amplitudes storage
    size_t t2_aa_size = n_alpha * n_alpha * n_vir_alpha * n_vir_alpha;
    size_t t2_ab_size = n_alpha * n_beta * n_vir_alpha * n_vir_beta;
    size_t t2_bb_size = n_beta * n_beta * n_vir_beta * n_vir_beta;

    Eigen::VectorXd t2_aa(t2_aa_size);
    Eigen::VectorXd t2_ab(t2_ab_size);
    Eigen::VectorXd t2_bb(t2_bb_size);
    t2_aa.setZero();
    t2_ab.setZero();
    t2_bb.setZero();

    // Pre-compute strides for tensor indexing
    const size_t stride_k = active_space_size;
    const size_t stride_j = active_space_size * active_space_size;
    const size_t stride_i =
        active_space_size * active_space_size * active_space_size;

    // Alpha-Alpha contribution
    algorithms::microsoft::mp2_helpers::compute_same_spin_t2(
        eps_alpha, moeri_aaaa, n_alpha, n_vir_alpha, stride_i, stride_j,
        stride_k, t2_aa);

    // Alpha-Beta contribution
    algorithms::microsoft::mp2_helpers::compute_opposite_spin_t2(
        eps_alpha, eps_beta, moeri_aabb, n_alpha, n_beta, n_vir_alpha,
        n_vir_beta, stride_i, stride_j, stride_k, t2_ab);

    // Beta-Beta contribution
    algorithms::microsoft::mp2_helpers::compute_same_spin_t2(
        eps_beta, moeri_bbbb, n_beta, n_vir_beta, stride_i, stride_j, stride_k,
        t2_bb);

    _t2_amplitudes_abab = std::make_shared<VectorVariant>(t2_ab);
    _t2_amplitudes_aaaa = std::make_shared<VectorVariant>(t2_aa);
    _t2_amplitudes_bbbb = std::make_shared<VectorVariant>(t2_bb);
  } else {
    // Restricted MP2
    if (n_alpha != n_beta) {
      throw std::runtime_error(
          "Restricted MP2 requires equal alpha and beta electrons");
    }

    const auto& [eps_alpha, eps_beta] = get_orbitals()->get_energies();
    const size_t n_occ = n_alpha;
    const size_t n_vir = active_space_size - n_occ;

    const auto& [moeri_aaaa, moeri_aabb, moeri_bbbb] =
        _hamiltonian->get_two_body_integrals();
    // For restricted case, all components are the same; use aaaa
    const auto& moeri = moeri_aaaa;

    size_t t2_size = n_occ * n_occ * n_vir * n_vir;
    Eigen::VectorXd t2_amplitudes(t2_size);
    t2_amplitudes.setZero();

    // Pre-compute strides for tensor indexing
    const size_t stride_k = active_space_size;
    const size_t stride_j = active_space_size * active_space_size;
    const size_t stride_i =
        active_space_size * active_space_size * active_space_size;

    // Compute T2 amplitudes
    algorithms::microsoft::mp2_helpers::compute_restricted_t2(
        eps_alpha, moeri, n_occ, n_vir, stride_i, stride_j, stride_k,
        t2_amplitudes);

    _t2_amplitudes_abab = std::make_shared<VectorVariant>(t2_amplitudes);
    _t2_amplitudes_aaaa = std::make_shared<VectorVariant>(t2_amplitudes);
    _t2_amplitudes_bbbb = std::make_shared<VectorVariant>(t2_amplitudes);
  }
}

std::pair<const MP2Container::VectorVariant&,
          const MP2Container::VectorVariant&>
MP2Container::get_t1_amplitudes() const {
  if (!_t1_amplitudes_aa) {
    _compute_t1_amplitudes();
  }

  return std::make_pair(std::cref(*_t1_amplitudes_aa),
                        std::cref(*_t1_amplitudes_bb));
}

std::tuple<const MP2Container::VectorVariant&,
           const MP2Container::VectorVariant&,
           const MP2Container::VectorVariant&>
MP2Container::get_t2_amplitudes() const {
  if (!_t2_amplitudes_abab) {
    _compute_t2_amplitudes();
  }

  return std::make_tuple(std::cref(*_t2_amplitudes_abab),
                         std::cref(*_t2_amplitudes_aaaa),
                         std::cref(*_t2_amplitudes_bbbb));
}

bool MP2Container::has_t1_amplitudes() const {
  return _t1_amplitudes_aa != nullptr;
}

bool MP2Container::has_t2_amplitudes() const {
  return _t2_amplitudes_abab != nullptr;
}

size_t MP2Container::size() const {
  throw std::runtime_error("size() is not meaningful for MP2 wavefunctions.");
}

MP2Container::ScalarVariant MP2Container::overlap(
    const WavefunctionContainer& other) const {
  throw std::runtime_error(
      "overlap() is not implemented for MP2 wavefunctions.");
}

double MP2Container::norm() const {
  throw std::runtime_error("norm() is not implemented for MP2 wavefunctions.");
}

bool MP2Container::contains_determinant(const Configuration& det) const {
  throw std::runtime_error(
      "contains_determinant() is not implemented for MP2 wavefunctions.");
}

bool MP2Container::contains_reference(const Configuration& det) const {
  if (std::find(_references.get_configurations().begin(),
                _references.get_configurations().end(),
                det) != _references.get_configurations().end()) {
    return true;
  }
  return false;
}

void MP2Container::clear_caches() const {
  _t1_amplitudes_aa = nullptr;
  _t1_amplitudes_bb = nullptr;
  _t2_amplitudes_abab = nullptr;
  _t2_amplitudes_aaaa = nullptr;
  _t2_amplitudes_bbbb = nullptr;

  _determinant_vector_cache = nullptr;
}

nlohmann::json MP2Container::to_json() const {
  nlohmann::json j;
  j["type"] = "mp2";
  j["version"] = SERIALIZATION_VERSION;
  j["wavefunction_type"] = (get_type() == WavefunctionType::SelfDual)
                               ? "self_dual"
                               : "not_self_dual";

  // Serialize references
  j["references"] = nlohmann::json::array();
  for (const auto& ref : _references.get_configurations()) {
    j["references"].push_back(ref.to_json());
  }

  // Serialize orbitals
  j["orbitals"] = get_orbitals()->to_json();

  // Serialize Hamiltonian
  j["hamiltonian"] = _hamiltonian->to_json();

  // Note: We don't serialize amplitudes in JSON - they are computed on
  // demand when Hamiltonian is available

  return j;
}

std::unique_ptr<MP2Container> MP2Container::from_json(const nlohmann::json& j) {
  // Deserialize basic fields
  auto wf_type =
      static_cast<WavefunctionType>(j.at("wavefunction_type").get<int>());

  // Deserialize references
  DeterminantVector references;
  for (const auto& ref_json : j.at("references")) {
    references.push_back(Configuration::from_json(ref_json));
  }

  // Deserialize orbitals
  auto orbitals = Orbitals::from_json(j.at("orbitals"));

  // Deserialize Hamiltonian (can be null)
  std::shared_ptr<Hamiltonian> hamiltonian = nullptr;
  if (!j.at("hamiltonian").is_null()) {
    hamiltonian = Hamiltonian::from_json(j.at("hamiltonian"));
  }

  return std::make_unique<MP2Container>(hamiltonian, references, wf_type);
}

void MP2Container::to_hdf5(H5::Group& group) const {
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

    // wavefunction type
    std::string wf_type_str = (get_type() == WavefunctionType::SelfDual)
                                  ? "self_dual"
                                  : "not_self_dual";
    H5::Attribute wf_type_attr = group.createAttribute(
        "wavefunction_type", string_type, H5::DataSpace(H5S_SCALAR));
    wf_type_attr.write(string_type, wf_type_str);

    // complex flag
    bool is_complex_flag = this->is_complex();
    H5::Attribute is_complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    hbool_t is_complex_hbool = is_complex_flag ? 1 : 0;
    is_complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex_hbool);

    // Store configuration set
    H5::Group reference_configs_group =
        group.createGroup("reference_configurations");
    _references.to_hdf5(reference_configs_group);

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
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // version
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    // wavefunction type
    WavefunctionType wf_type = WavefunctionType::NotSelfDual;
    if (group.attrExists("wavefunction_type")) {
      H5::Attribute wf_type_attr = group.openAttribute("wavefunction_type");
      std::string wf_type_str;
      wf_type_attr.read(string_type, wf_type_str);
      wf_type = (wf_type_str == "self_dual") ? WavefunctionType::SelfDual
                                             : WavefunctionType::NotSelfDual;
    }

    // Load configuration set
    if (!group.nameExists("reference_configurations")) {
      throw std::runtime_error(
          "HDF5 group missing required 'reference_configurations' subgroup");
    }
    H5::Group reference_configs_group =
        group.openGroup("reference_configurations");
    auto reference_configs =
        ConfigurationSet::from_hdf5(reference_configs_group);
    const auto& determinants = reference_configs.get_configurations();
    auto orbitals = reference_configs.get_orbitals();

    // Load Hamiltonian (if available)
    std::shared_ptr<Hamiltonian> hamiltonian = nullptr;
    if (group.nameExists("hamiltonian")) {
      H5::Group hamiltonian_group = group.openGroup("hamiltonian");
      hamiltonian = Hamiltonian::from_hdf5(hamiltonian_group);
    }

    return std::make_unique<MP2Container>(hamiltonian, determinants, wf_type);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::pair<size_t, size_t> MP2Container::get_total_num_electrons() const {
  // Get from first reference determinant
  if (_references.size() == 0) {
    throw std::runtime_error("No reference determinants available");
  }
  const auto& first_ref = _references.get_configurations()[0];
  auto [n_alpha, n_beta] = first_ref.get_n_electrons();
  return std::make_pair(n_alpha, n_beta);
}

std::pair<size_t, size_t> MP2Container::get_active_num_electrons() const {
  return get_total_num_electrons();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MP2Container::get_total_orbital_occupations() const {
  throw std::runtime_error(
      "get_total_orbital_occupations() is not implemented for MP2 "
      "wavefunctions.");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MP2Container::get_active_orbital_occupations() const {
  throw std::runtime_error(
      "get_active_orbital_occupations() is not implemented for MP2 "
      "wavefunctions.");
}

std::string MP2Container::get_container_type() const { return "mp2"; }

bool MP2Container::is_complex() const {
  if (_t1_amplitudes_aa) {
    if (std::holds_alternative<Eigen::VectorXcd>(*_t1_amplitudes_aa)) {
      return true;
    }
  }
  if (_t2_amplitudes_abab) {
    if (std::holds_alternative<Eigen::VectorXcd>(*_t2_amplitudes_abab)) {
      return true;
    }
  }
  return false;
}

}  // namespace qdk::chemistry::data
