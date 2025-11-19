// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <stdexcept>

#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

SlaterDeterminantContainer::SlaterDeterminantContainer(
    const Configuration& det, std::shared_ptr<Orbitals> orbitals,
    WavefunctionType type)
    : WavefunctionContainer(type),
      _determinant(det),
      _orbitals(orbitals),
      _coefficient_vector(Eigen::VectorXd(Eigen::VectorXd::Ones(1))) {
  // Validate that the configuration represents the active space correctly
  // Note: Configurations only represent the active space, not the full orbital
  // space (inactive and virtual orbitals are not included in the configuration
  // representation)

  const std::string config_str = det.to_string();

  // Get active space indices for validation
  auto [alpha_active, beta_active] = orbitals->get_active_space_indices();

  // For restricted calculations, use alpha indices (they should be the same)
  const auto& active_indices = alpha_active;

  // Validate that configuration has sufficient orbital capacity for the active
  // space
  if (!active_indices.empty()) {
    size_t active_space_size = active_indices.size();

    // The configuration must have at least as many orbitals as the active space
    // size
    if (det.get_orbital_capacity() < active_space_size) {
      throw std::invalid_argument(
          "SlaterDeterminantContainer: configuration has orbital capacity " +
          std::to_string(det.get_orbital_capacity()) +
          " which is insufficient for active space (requires at least " +
          std::to_string(active_space_size) + " orbitals).");
    }

    // Validate that any orbitals beyond the active space size are unoccupied
    // (no "overhanging" electrons)
    for (size_t orbital_idx = active_space_size;
         orbital_idx < det.get_orbital_capacity(); ++orbital_idx) {
      if (orbital_idx < config_str.length() && config_str[orbital_idx] != '0') {
        throw std::invalid_argument(
            "SlaterDeterminantContainer: configuration has occupied orbital at "
            "index " +
            std::to_string(orbital_idx) +
            " which is beyond the active space size (" +
            std::to_string(active_space_size) +
            "). Only orbitals within the active space can be occupied.");
      }
    }
  }
}

std::unique_ptr<WavefunctionContainer> SlaterDeterminantContainer::clone()
    const {
  return std::make_unique<SlaterDeterminantContainer>(_determinant, _orbitals,
                                                      _type);
}

std::shared_ptr<Orbitals> SlaterDeterminantContainer::get_orbitals() const {
  return _orbitals;
}

const ContainerTypes::VectorVariant&
SlaterDeterminantContainer::get_coefficients() const {
  return _coefficient_vector;
}

ContainerTypes::ScalarVariant SlaterDeterminantContainer::get_coefficient(
    const Configuration& det) const {
  return (_determinant == det) ? 1.0 : 0.0;
}

const ContainerTypes::DeterminantVector&
SlaterDeterminantContainer::get_active_determinants() const {
  if (!_determinant_vector_cache) {
    _determinant_vector_cache = std::make_unique<DeterminantVector>();
    _determinant_vector_cache->push_back(_determinant);
  }
  return *_determinant_vector_cache;
}

size_t SlaterDeterminantContainer::size() const { return 1; }

ContainerTypes::ScalarVariant SlaterDeterminantContainer::overlap(
    const WavefunctionContainer& other) const {
  // For single determinant, overlap is coefficient of this determinant in other
  // wavefunction if the bases are identical
  // return other.get_coefficient(_determinant);
  throw std::runtime_error(
      "overlap not yet implemented for "
      "SlaterDeterminantContainer");
}

double SlaterDeterminantContainer::norm() const {
  return 1.0;  // Single normalized determinant always has norm 1
}

bool SlaterDeterminantContainer::contains_determinant(
    const Configuration& det) const {
  return _determinant == det;
}

void SlaterDeterminantContainer::clear_caches() const {
  // Clear the cached determinant vector
  _determinant_vector_cache.reset();

  // Clear all cached RDMs
  _clear_rdms();
}

std::tuple<const ContainerTypes::MatrixVariant&,
           const ContainerTypes::MatrixVariant&>
SlaterDeterminantContainer::get_active_one_rdm_spin_dependent() const {
  // TODO: Implement RDM calculation for single determinant
  throw std::runtime_error(
      "get_active_one_rdm_spin_dependent not yet implemented for "
      "SlaterDeterminantContainer");
}

std::tuple<const ContainerTypes::VectorVariant&,
           const ContainerTypes::VectorVariant&,
           const ContainerTypes::VectorVariant&>
SlaterDeterminantContainer::get_active_two_rdm_spin_dependent() const {
  // TODO: Implement RDM calculation for single determinant
  throw std::runtime_error(
      "get_active_two_rdm_spin_dependent not yet implemented for "
      "SlaterDeterminantContainer");
}

const ContainerTypes::MatrixVariant&
SlaterDeterminantContainer::get_active_one_rdm_spin_traced() const {
  if (!_one_rdm_spin_traced) {
    auto [alpha_occupations, beta_occupations] =
        get_active_orbital_occupations();
    size_t n_orbs = _orbitals->get_active_space_indices().first.size();
    Eigen::MatrixXd tmp_one_rdm = Eigen::MatrixXd::Zero(n_orbs, n_orbs);
    for (size_t i = 0; i < alpha_occupations.size(); ++i) {
      tmp_one_rdm(i, i) += alpha_occupations(i);
    }
    for (size_t i = 0; i < beta_occupations.size(); ++i) {
      tmp_one_rdm(i, i) += beta_occupations(i);
    }
    _one_rdm_spin_traced =
        std::make_unique<ContainerTypes::MatrixVariant>(std::move(tmp_one_rdm));
  }
  return *_one_rdm_spin_traced;
}

const ContainerTypes::VectorVariant&
SlaterDeterminantContainer::get_active_two_rdm_spin_traced() const {
  if (!_two_rdm_spin_traced) {
    auto [alpha_occupations, beta_occupations] =
        get_active_orbital_occupations();
    const auto& one_rdm_var = get_active_one_rdm_spin_traced();
    const Eigen::MatrixXd& one_rdm = std::get<Eigen::MatrixXd>(one_rdm_var);
    size_t norbs = one_rdm.rows();
    Eigen::VectorXd tmp_two_rdm =
        Eigen::VectorXd::Zero(norbs * norbs * norbs * norbs);
    size_t norb2 = norbs * norbs;
    size_t norb3 = norbs * norb2;
    for (size_t i = 0; i < alpha_occupations.size(); ++i) {
      if (alpha_occupations(i) != 1.0) continue;
      for (size_t j = 0; j < beta_occupations.size(); ++j) {
        if (beta_occupations(j) != 1.0 || i == j) continue;
        size_t index1 = i * norb3 + j * norb2 + j * norbs + i;
        tmp_two_rdm(index1) = -2.0;
        size_t index2 = j * norb3 + i * norb2 + i * norbs + j;
        tmp_two_rdm(index2) = -2.0;
        size_t index3 = i * norb3 + i * norb2 + j * norbs + j;
        tmp_two_rdm(index3) = 4.0;
        size_t index4 = j * norb3 + j * norb2 + i * norbs + i;
        tmp_two_rdm(index4) = 4.0;
      }
      size_t index_diag = i * norb3 + i * norb2 + i * norbs + i;
      tmp_two_rdm(index_diag) = 2.0;
    }
    _two_rdm_spin_traced =
        std::make_unique<ContainerTypes::VectorVariant>(std::move(tmp_two_rdm));
  }
  return *_two_rdm_spin_traced;
}

Eigen::VectorXd SlaterDeterminantContainer::get_single_orbital_entropies()
    const {
  // TODO: Implement entropy calculation for single determinant
  throw std::runtime_error(
      "get_single_orbital_entropies not yet implemented for "
      "SlaterDeterminantContainer");
}

std::pair<size_t, size_t> SlaterDeterminantContainer::get_total_num_electrons()
    const {
  // Get active space electrons from the determinant
  auto [n_alpha_active, n_beta_active] = get_active_num_electrons();

  // Add electrons from inactive space (doubly occupied orbitals)
  auto [alpha_inactive_indices, beta_inactive_indices] =
      _orbitals->get_inactive_space_indices();

  size_t n_alpha_total = n_alpha_active + alpha_inactive_indices.size();
  size_t n_beta_total = n_beta_active + beta_inactive_indices.size();

  return {n_alpha_total, n_beta_total};
}

std::pair<size_t, size_t> SlaterDeterminantContainer::get_active_num_electrons()
    const {
  auto [n_alpha, n_beta] = _determinant.get_n_electrons();
  return {n_alpha, n_beta};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
SlaterDeterminantContainer::get_total_orbital_occupations() const {
  // Get the total number of orbitals from the orbital basis set
  const int num_orbitals =
      static_cast<int>(_orbitals->get_num_molecular_orbitals());

  Eigen::VectorXd alpha_occupations = Eigen::VectorXd::Zero(num_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_orbitals);

  // Get inactive space indices for doubly occupied orbitals
  auto [alpha_inactive_indices, beta_inactive_indices] =
      _orbitals->get_inactive_space_indices();

  // Set inactive orbitals as doubly occupied
  for (size_t inactive_idx : alpha_inactive_indices) {
    if (inactive_idx < static_cast<size_t>(num_orbitals)) {
      alpha_occupations(inactive_idx) = 1.0;
    }
  }
  for (size_t inactive_idx : beta_inactive_indices) {
    if (inactive_idx < static_cast<size_t>(num_orbitals)) {
      beta_occupations(inactive_idx) = 1.0;
    }
  }

  // Get active space occupations and insert them at the correct positions
  auto [alpha_active_occs, beta_active_occs] = get_active_orbital_occupations();
  auto [alpha_active_indices, beta_active_indices] =
      _orbitals->get_active_space_indices();

  // Map active space occupations to their global orbital indices
  for (size_t active_idx = 0; active_idx < alpha_active_indices.size() &&
                              active_idx < alpha_active_occs.size();
       ++active_idx) {
    size_t orbital_idx = alpha_active_indices[active_idx];
    if (orbital_idx < static_cast<size_t>(num_orbitals)) {
      alpha_occupations(orbital_idx) = alpha_active_occs(active_idx);
    }
  }

  for (size_t active_idx = 0; active_idx < beta_active_indices.size() &&
                              active_idx < beta_active_occs.size();
       ++active_idx) {
    size_t orbital_idx = beta_active_indices[active_idx];
    if (orbital_idx < static_cast<size_t>(num_orbitals)) {
      beta_occupations(orbital_idx) = beta_active_occs(active_idx);
    }
  }

  return {alpha_occupations, beta_occupations};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
SlaterDeterminantContainer::get_active_orbital_occupations() const {
  // Get the active space indices and size
  auto [alpha_active_indices, beta_active_indices] =
      _orbitals->get_active_space_indices();

  // If no active space is defined, return the same as total occupations
  if (alpha_active_indices.empty()) {
    return get_total_orbital_occupations();
  }

  // Get the active space size
  const size_t num_active_orbitals = alpha_active_indices.size();

  Eigen::VectorXd alpha_occupations =
      Eigen::VectorXd::Zero(num_active_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_active_orbitals);

  // Convert determinant to string representation to parse occupations
  std::string config_str = _determinant.to_string();

  // Parse only the active space orbitals
  for (int active_idx = 0;
       active_idx < num_active_orbitals && active_idx < config_str.length();
       ++active_idx) {
    char state =
        config_str[active_idx];  // Read directly from active space position
    if (state == 'u' || state == '2') {  // Alpha or doubly occupied
      alpha_occupations(active_idx) = 1.0;
    }
    if (state == 'd' || state == '2') {  // Beta or doubly occupied
      beta_occupations(active_idx) = 1.0;
    }
  }

  return {alpha_occupations, beta_occupations};
}

bool SlaterDeterminantContainer::has_one_rdm_spin_dependent() const {
  // Always available for Slater determinants (can be computed on-the-fly)
  return true;
}

bool SlaterDeterminantContainer::has_one_rdm_spin_traced() const {
  // Always available for Slater determinants (can be computed on-the-fly)
  return true;
}

bool SlaterDeterminantContainer::has_two_rdm_spin_dependent() const {
  // Always available for Slater determinants (can be computed on-the-fly)
  return true;
}

bool SlaterDeterminantContainer::has_two_rdm_spin_traced() const {
  // Always available for Slater determinants (can be computed on-the-fly)
  return true;
}

std::string SlaterDeterminantContainer::get_container_type() const {
  return "sd";
}

bool SlaterDeterminantContainer::is_complex() const {
  return false;  // Slater determinants always use real coefficients (unity)
}

nlohmann::json SlaterDeterminantContainer::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Store container type
  j["container_type"] = get_container_type();

  // Store wavefunction type
  j["wavefunction_type"] =
      (_type == WavefunctionType::SelfDual) ? "self_dual" : "not_self_dual";

  // Store orbitals
  j["orbitals"] = _orbitals->to_json();

  // Store single determinant
  j["determinant"] = _determinant.to_json();

  // SD containers are always real with coefficient 1.0
  j["is_complex"] = false;

  return j;
}

std::unique_ptr<SlaterDeterminantContainer>
SlaterDeterminantContainer::from_json(const nlohmann::json& j) {
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load orbitals
    if (!j.contains("orbitals")) {
      throw std::runtime_error("JSON missing required 'orbitals' field");
    }
    auto orbitals = Orbitals::from_json(j["orbitals"]);

    // Load wavefunction type
    WavefunctionType type = WavefunctionType::SelfDual;
    if (j.contains("wavefunction_type")) {
      std::string type_str = j["wavefunction_type"];
      type = (type_str == "self_dual") ? WavefunctionType::SelfDual
                                       : WavefunctionType::NotSelfDual;
    }

    // Load determinant
    if (!j.contains("determinant")) {
      throw std::runtime_error("JSON missing required 'determinant' field");
    }
    Configuration determinant = Configuration::from_json(j["determinant"]);

    return std::make_unique<SlaterDeterminantContainer>(determinant, orbitals,
                                                        type);

  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to parse SlaterDeterminantContainer from JSON: " +
        std::string(e.what()));
  }
}

void SlaterDeterminantContainer::to_hdf5(H5::Group& group) const {
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Add version attribute
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);
    version_attr.close();

    // Store container type
    std::string container_type = get_container_type();
    H5::Attribute type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    type_attr.write(string_type, container_type);

    // Store wavefunction type
    std::string wf_type =
        (_type == WavefunctionType::SelfDual) ? "self_dual" : "not_self_dual";
    H5::Attribute wf_type_attr = group.createAttribute(
        "wavefunction_type", string_type, H5::DataSpace(H5S_SCALAR));
    wf_type_attr.write(string_type, wf_type);

    // Store orbitals
    H5::Group orbitals_group = group.createGroup("orbitals");
    _orbitals->to_hdf5(orbitals_group);

    // Store complexity flag (always false for SD)
    H5::Attribute complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    hbool_t is_complex_flag = 0;  // Always false for SlaterDeterminant
    complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex_flag);

    // Store single determinant
    H5::Group det_group = group.createGroup("determinant");
    _determinant.to_hdf5(det_group);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<SlaterDeterminantContainer>
SlaterDeterminantContainer::from_hdf5(H5::Group& group) {
  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Load orbitals
    if (!group.nameExists("orbitals")) {
      throw std::runtime_error(
          "HDF5 group missing required 'orbitals' subgroup");
    }
    H5::Group orbitals_group = group.openGroup("orbitals");
    auto orbitals = Orbitals::from_hdf5(orbitals_group);

    // Load wavefunction type
    WavefunctionType type = WavefunctionType::SelfDual;
    if (group.attrExists("wavefunction_type")) {
      H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
      H5::Attribute wf_type_attr = group.openAttribute("wavefunction_type");
      std::string type_str;
      wf_type_attr.read(string_type, type_str);
      type = (type_str == "self_dual") ? WavefunctionType::SelfDual
                                       : WavefunctionType::NotSelfDual;
    }

    // Load determinant
    if (!group.nameExists("determinant")) {
      throw std::runtime_error(
          "HDF5 group missing required 'determinant' dataset");
    }
    H5::Group det_group = group.openGroup("determinant");
    Configuration determinant = Configuration::from_hdf5(det_group);
    det_group.close();

    return std::make_unique<SlaterDeterminantContainer>(determinant, orbitals,
                                                        type);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
