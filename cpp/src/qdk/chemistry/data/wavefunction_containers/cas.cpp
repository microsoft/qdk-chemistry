// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <stdexcept>
#include <tuple>
#include <variant>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {
using MatrixVariant = ContainerTypes::MatrixVariant;
using VectorVariant = ContainerTypes::VectorVariant;
using ScalarVariant = ContainerTypes::ScalarVariant;

CasWavefunctionContainer::CasWavefunctionContainer(
    const VectorVariant& coeffs, const DeterminantVector& dets,
    std::shared_ptr<Orbitals> orbitals, WavefunctionType type)
    : CasWavefunctionContainer(coeffs, dets, orbitals,
                               std::nullopt,  // one_rdm_spin_traced
                               std::nullopt,  // one_rdm_aa
                               std::nullopt,  // one_rdm_bb
                               std::nullopt,  // two_rdm_spin_traced
                               std::nullopt,  // two_rdm_aabb
                               std::nullopt,  // two_rdm_aaaa
                               std::nullopt,  // two_rdm_bbbb
                               type) {}

CasWavefunctionContainer::CasWavefunctionContainer(
    const VectorVariant& coeffs, const DeterminantVector& dets,
    std::shared_ptr<Orbitals> orbitals,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    WavefunctionType type)
    : CasWavefunctionContainer(coeffs, dets, orbitals, one_rdm_spin_traced,
                               std::nullopt,  // one_rdm_aa
                               std::nullopt,  // one_rdm_bb
                               two_rdm_spin_traced,
                               std::nullopt,  // two_rdm_aabb
                               std::nullopt,  // two_rdm_aaaa
                               std::nullopt,  // two_rdm_bbbb
                               type) {}

CasWavefunctionContainer::CasWavefunctionContainer(
    const VectorVariant& coeffs, const DeterminantVector& dets,
    std::shared_ptr<Orbitals> orbitals,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<MatrixVariant>& one_rdm_aa,
    const std::optional<MatrixVariant>& one_rdm_bb,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_aabb,
    const std::optional<VectorVariant>& two_rdm_aaaa,
    const std::optional<VectorVariant>& two_rdm_bbbb, WavefunctionType type)
    : WavefunctionContainer(one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
                            two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
                            two_rdm_bbbb, type),
      _coefficients(coeffs),
      _configuration_set(dets, orbitals) {}

std::unique_ptr<WavefunctionContainer> CasWavefunctionContainer::clone() const {
  return std::make_unique<CasWavefunctionContainer>(
      _coefficients, _configuration_set.get_configurations(),
      this->get_orbitals(),
      _one_rdm_spin_traced ? std::optional<MatrixVariant>(*_one_rdm_spin_traced)
                           : std::nullopt,
      _one_rdm_spin_dependent_aa
          ? std::optional<MatrixVariant>(*_one_rdm_spin_dependent_aa)
          : std::nullopt,
      _one_rdm_spin_dependent_bb
          ? std::optional<MatrixVariant>(*_one_rdm_spin_dependent_bb)
          : std::nullopt,
      _two_rdm_spin_traced ? std::optional<VectorVariant>(*_two_rdm_spin_traced)
                           : std::nullopt,
      _two_rdm_spin_dependent_aabb
          ? std::optional<VectorVariant>(*_two_rdm_spin_dependent_aabb)
          : std::nullopt,
      _two_rdm_spin_dependent_aaaa
          ? std::optional<VectorVariant>(*_two_rdm_spin_dependent_aaaa)
          : std::nullopt,
      _two_rdm_spin_dependent_bbbb
          ? std::optional<VectorVariant>(*_two_rdm_spin_dependent_bbbb)
          : std::nullopt,
      this->get_type());
}

ScalarVariant CasWavefunctionContainer::get_coefficient(
    const Configuration& det) const {
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  auto it = std::find(determinants.begin(), determinants.end(), det);
  if (it != determinants.end()) {
    size_t index = std::distance(determinants.begin(), it);
    if (detail::is_vector_variant_complex(_coefficients)) {
      return std::get<Eigen::VectorXcd>(_coefficients)(index);
    }
    return std::get<Eigen::VectorXd>(_coefficients)(index);
  }
  throw std::runtime_error("Determinant not found in wavefunction");
}

const CasWavefunctionContainer::VectorVariant&
CasWavefunctionContainer::get_coefficients() const {
  return _coefficients;
}

std::shared_ptr<Orbitals> CasWavefunctionContainer::get_orbitals() const {
  return _configuration_set.get_orbitals();
}

const CasWavefunctionContainer::DeterminantVector&
CasWavefunctionContainer::get_active_determinants() const {
  return _configuration_set.get_configurations();
}

size_t CasWavefunctionContainer::size() const {
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    return 0;  // Empty wavefunction has size 0
  }
  if (detail::is_vector_variant_complex(_coefficients)) {
    return std::get<Eigen::VectorXcd>(_coefficients).size();
  }
  return std::get<Eigen::VectorXd>(_coefficients).size();
}

CasWavefunctionContainer::ScalarVariant CasWavefunctionContainer::overlap(
    const WavefunctionContainer& other) const {
  // Check type of other.  If not CasWavefunctionContainer, throw error.
  const auto* other_cas = dynamic_cast<const CasWavefunctionContainer*>(&other);
  if (!other_cas) {
    throw std::runtime_error(
        "Overlap only implemented between two CasWavefunctionContainer");
  }
  // both are CasWavefunctionContainers
  if (this->size() != other_cas->size()) {
    throw std::runtime_error(
        "Overlap only implemented for wavefunctions with same number of "
        "determinants");
  }
  if (this->get_active_num_electrons() !=
      other_cas->get_active_num_electrons()) {
    throw std::runtime_error(
        "Overlap only implemented for wavefunctions with same number of "
        "electrons");
  }
  // TODO: implement proper overlap calculation, workitem: 41338
  if (this->get_orbitals() != other_cas->get_orbitals()) {
    throw std::runtime_error(
        "Overlap only implemented for wavefunctions with same orbitals");
  }

  // Calculate overlap using helper functions to check types
  const auto& coeffs1 = this->get_coefficients();
  const auto& coeffs2 = other_cas->get_coefficients();

  bool coeffs1_complex = detail::is_vector_variant_complex(coeffs1);
  bool coeffs2_complex = detail::is_vector_variant_complex(coeffs2);

  if (!coeffs1_complex && !coeffs2_complex) {
    // Both real
    const auto& real_coeffs1 = std::get<Eigen::VectorXd>(coeffs1);
    const auto& real_coeffs2 = std::get<Eigen::VectorXd>(coeffs2);
    return real_coeffs1.dot(real_coeffs2);
  } else if (coeffs1_complex && coeffs2_complex) {
    // Both complex
    const auto& complex_coeffs1 = std::get<Eigen::VectorXcd>(coeffs1);
    const auto& complex_coeffs2 = std::get<Eigen::VectorXcd>(coeffs2);
    return complex_coeffs1.adjoint() * complex_coeffs2;
  } else if (coeffs1_complex && !coeffs2_complex) {
    // First complex, second real
    const auto& complex_coeffs1 = std::get<Eigen::VectorXcd>(coeffs1);
    const auto& real_coeffs2 = std::get<Eigen::VectorXd>(coeffs2);
    return complex_coeffs1.adjoint() *
           real_coeffs2.cast<std::complex<double>>();
  } else {
    // First real, second complex
    const auto& real_coeffs1 = std::get<Eigen::VectorXd>(coeffs1);
    const auto& complex_coeffs2 = std::get<Eigen::VectorXcd>(coeffs2);
    return real_coeffs1.cast<std::complex<double>>().adjoint() *
           complex_coeffs2;
  }
}

double CasWavefunctionContainer::norm() const {
  const auto& coeffs = this->get_coefficients();
  if (detail::is_vector_variant_complex(coeffs)) {
    const auto& complex_coeffs = std::get<Eigen::VectorXcd>(coeffs);
    return std::sqrt((complex_coeffs.adjoint() * complex_coeffs)(0).real());
  } else {
    const auto& real_coeffs = std::get<Eigen::VectorXd>(coeffs);
    return real_coeffs.norm();
  }
}

void CasWavefunctionContainer::clear_caches() const {
  // Clear all cached RDMs
  _clear_rdms();
}

std::pair<size_t, size_t> CasWavefunctionContainer::get_total_num_electrons()
    const {
  // Get active space electrons using the dedicated method
  auto [n_alpha_active, n_beta_active] = get_active_num_electrons();

  // Add electrons from inactive space (doubly occupied orbitals)
  auto [alpha_inactive_indices, beta_inactive_indices] =
      get_orbitals()->get_inactive_space_indices();

  size_t n_alpha_total = n_alpha_active + alpha_inactive_indices.size();
  size_t n_beta_total = n_beta_active + beta_inactive_indices.size();

  return {n_alpha_total, n_beta_total};
}

std::pair<size_t, size_t> CasWavefunctionContainer::get_active_num_electrons()
    const {
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }
  auto [n_alpha, n_beta] = determinants[0].get_n_electrons();
  return {n_alpha, n_beta};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CasWavefunctionContainer::get_total_orbital_occupations() const {
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  // Get the total number of orbitals from the orbital basis set
  const int num_orbitals =
      static_cast<int>(get_orbitals()->get_num_molecular_orbitals());

  Eigen::VectorXd alpha_occupations = Eigen::VectorXd::Zero(num_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_orbitals);

  // Get inactive space indices and mark them as doubly occupied
  auto [alpha_inactive_indices, beta_inactive_indices] =
      get_orbitals()->get_inactive_space_indices();

  // Set inactive orbitals as doubly occupied (occupation = 1.0)
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

  // For active space orbitals, get occupations from 1RDM eigenvalues
  if (has_one_rdm_spin_dependent()) {
    // Get active space occupations using the dedicated method
    auto [alpha_active_occs, beta_active_occs] =
        get_active_orbital_occupations();

    // Get active space indices to map back to total orbital indices
    auto [alpha_active_indices, beta_active_indices] =
        get_orbitals()->get_active_space_indices();

    // Map active space occupations to total orbital indices
    for (size_t active_idx = 0; active_idx < alpha_active_indices.size();
         ++active_idx) {
      size_t orbital_idx = alpha_active_indices[active_idx];
      if (orbital_idx < static_cast<size_t>(num_orbitals) &&
          active_idx < alpha_active_occs.size()) {
        alpha_occupations(orbital_idx) = alpha_active_occs(active_idx);
      }
    }

    for (size_t active_idx = 0; active_idx < beta_active_indices.size();
         ++active_idx) {
      size_t orbital_idx = beta_active_indices[active_idx];
      if (orbital_idx < static_cast<size_t>(num_orbitals) &&
          active_idx < beta_active_occs.size()) {
        beta_occupations(orbital_idx) = beta_active_occs(active_idx);
      }
    }
  } else {
    throw std::runtime_error(
        "1RDM must be available to compute orbital occupations");
  }

  return {alpha_occupations, beta_occupations};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CasWavefunctionContainer::get_active_orbital_occupations() const {
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  // Get the active space indices
  auto [alpha_active_indices, beta_active_indices] =
      get_orbitals()->get_active_space_indices();

  // If no active space is defined, return empty vectors
  if (alpha_active_indices.empty()) {
    return {Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0)};
  }

  const int num_active_orbitals = static_cast<int>(alpha_active_indices.size());

  Eigen::VectorXd alpha_occupations =
      Eigen::VectorXd::Zero(num_active_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_active_orbitals);

  // For active space orbitals, get occupations from 1RDM eigenvalues
  if (has_one_rdm_spin_dependent()) {
    const auto& rdm_tuple = get_active_one_rdm_spin_dependent();
    const auto& alpha_rdm_var = std::get<0>(rdm_tuple);
    const auto& beta_rdm_var = std::get<1>(rdm_tuple);

    // Extract real matrices (assuming real for now)
    if (detail::is_matrix_variant_complex(alpha_rdm_var) ||
        detail::is_matrix_variant_complex(beta_rdm_var)) {
      throw std::runtime_error(
          "Complex 1RDM diagonalization not yet implemented");
    }

    const Eigen::MatrixXd& alpha_rdm = std::get<Eigen::MatrixXd>(alpha_rdm_var);
    const Eigen::MatrixXd& beta_rdm = std::get<Eigen::MatrixXd>(beta_rdm_var);

    // Diagonalize alpha 1RDM to get occupations
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> alpha_solver(alpha_rdm);
    if (alpha_solver.info() != Eigen::Success) {
      throw std::runtime_error("Failed to diagonalize alpha 1RDM");
    }
    Eigen::VectorXd alpha_eigenvalues = alpha_solver.eigenvalues();

    // reverse to have descending order
    std::reverse(alpha_eigenvalues.data(),
                 alpha_eigenvalues.data() + alpha_eigenvalues.size());

    // Diagonalize beta 1RDM to get occupations
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> beta_solver(beta_rdm);
    if (beta_solver.info() != Eigen::Success) {
      throw std::runtime_error("Failed to diagonalize beta 1RDM");
    }
    Eigen::VectorXd beta_eigenvalues = beta_solver.eigenvalues();

    // reverse to have descending order
    std::reverse(beta_eigenvalues.data(),
                 beta_eigenvalues.data() + beta_eigenvalues.size());

    // Copy eigenvalues directly as active space occupations
    for (int active_idx = 0;
         active_idx < std::min(num_active_orbitals,
                               static_cast<int>(alpha_eigenvalues.size()));
         ++active_idx) {
      alpha_occupations(active_idx) = alpha_eigenvalues(active_idx);
    }

    for (int active_idx = 0;
         active_idx < std::min(num_active_orbitals,
                               static_cast<int>(beta_eigenvalues.size()));
         ++active_idx) {
      beta_occupations(active_idx) = beta_eigenvalues(active_idx);
    }
  } else {
    throw std::runtime_error(
        "1RDM must be available to compute orbital occupations");
  }

  return {alpha_occupations, beta_occupations};
}

std::string CasWavefunctionContainer::get_container_type() const {
  return "cas";
}

bool CasWavefunctionContainer::is_complex() const {
  return detail::is_vector_variant_complex(_coefficients);
}

nlohmann::json CasWavefunctionContainer::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Store container type
  j["container_type"] = get_container_type();

  // Store wavefunction type
  j["wavefunction_type"] =
      (_type == WavefunctionType::SelfDual) ? "self_dual" : "not_self_dual";

  // Store coefficients
  bool is_complex = detail::is_vector_variant_complex(_coefficients);
  j["is_complex"] = is_complex;
  if (is_complex) {
    const auto& coeffs_complex = std::get<Eigen::VectorXcd>(_coefficients);
    // Use NumPy's format: array of [real, imag] pairs
    nlohmann::json coeffs_array = nlohmann::json::array();
    for (int i = 0; i < coeffs_complex.size(); ++i) {
      coeffs_array.push_back(
          {coeffs_complex(i).real(), coeffs_complex(i).imag()});
    }
    j["coefficients"] = coeffs_array;
  } else {
    const auto& coeffs_real = std::get<Eigen::VectorXd>(_coefficients);
    // No copying - use data pointer directly
    j["coefficients"] = std::vector<double>(
        coeffs_real.data(), coeffs_real.data() + coeffs_real.size());
  }

  // Store configuration set (delegates to ConfigurationSet serialization)
  j["configuration_set"] = _configuration_set.to_json();

  return j;
}

std::unique_ptr<CasWavefunctionContainer> CasWavefunctionContainer::from_json(
    const nlohmann::json& j) {
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load wavefunction type
    WavefunctionType type = WavefunctionType::SelfDual;
    if (j.contains("wavefunction_type")) {
      std::string type_str = j["wavefunction_type"];
      type = (type_str == "self_dual") ? WavefunctionType::SelfDual
                                       : WavefunctionType::NotSelfDual;
    }

    // Load coefficients
    VectorVariant coefficients;
    bool is_complex = j.value("is_complex", false);
    if (is_complex) {
      if (!j.contains("coefficients")) {
        throw std::runtime_error("JSON missing required 'coefficients' field");
      }
      const auto& coeffs_json = j["coefficients"];

      // NumPy format: array of [real, imag] pairs
      if (!coeffs_json.is_array() || coeffs_json.empty() ||
          !coeffs_json[0].is_array()) {
        throw std::runtime_error(
            "Invalid complex coefficient format: expected array of [real, "
            "imag] pairs");
      }

      Eigen::VectorXcd coeffs_complex(coeffs_json.size());
      for (size_t i = 0; i < coeffs_json.size(); ++i) {
        if (coeffs_json[i].size() != 2) {
          throw std::runtime_error(
              "Invalid complex coefficient format: expected [real, imag] "
              "pairs");
        }
        coeffs_complex(i) =
            std::complex<double>(coeffs_json[i][0], coeffs_json[i][1]);
      }
      coefficients = coeffs_complex;
    } else {
      if (!j.contains("coefficients")) {
        throw std::runtime_error("JSON missing required 'coefficients' field");
      }
      std::vector<double> coeff_data = j["coefficients"];
      Eigen::VectorXd coeffs_real =
          Eigen::Map<Eigen::VectorXd>(coeff_data.data(), coeff_data.size());
      coefficients = coeffs_real;
    }

    // Load configuration set (delegates to ConfigurationSet deserialization)
    // ConfigurationSet now deserializes orbitals internally
    if (!j.contains("configuration_set")) {
      throw std::runtime_error(
          "JSON missing required 'configuration_set' field");
    }
    auto config_set = ConfigurationSet::from_json(j["configuration_set"]);
    const auto& determinants = config_set.get_configurations();
    auto orbitals = config_set.get_orbitals();

    return std::make_unique<CasWavefunctionContainer>(
        coefficients, determinants, orbitals, type);

  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to parse CasWavefunctionContainer from JSON: " +
        std::string(e.what()));
  }
}

void CasWavefunctionContainer::to_hdf5(H5::Group& group) const {
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

    // Store restrictedness flag
    bool is_restricted = get_orbitals()->is_restricted();
    H5::Attribute restricted_attr = group.createAttribute(
        "is_restricted", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));

    // Store complexity flag for coefficients
    bool is_complex = detail::is_vector_variant_complex(_coefficients);
    H5::Attribute complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    hbool_t is_complex_flag = is_complex ? 1 : 0;
    complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex_flag);

    // Store coefficients
    if (is_complex) {
      const auto& coeffs_complex = std::get<Eigen::VectorXcd>(_coefficients);
      hsize_t coeff_dims = coeffs_complex.size();
      H5::DataSpace coeff_space(1, &coeff_dims);

      // Use HDF5's native complex number support - no data copying
      // Create compound type for complex numbers (real, imag)
      H5::CompType complex_type(sizeof(std::complex<double>));
      complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
      complex_type.insertMember("imag", sizeof(double),
                                H5::PredType::NATIVE_DOUBLE);

      H5::DataSet complex_dataset =
          group.createDataSet("coefficients", complex_type, coeff_space);
      // Write directly from Eigen's memory layout without copying
      complex_dataset.write(coeffs_complex.data(), complex_type);
    } else {
      const auto& coeffs_real = std::get<Eigen::VectorXd>(_coefficients);
      hsize_t coeff_dims = coeffs_real.size();
      H5::DataSpace coeff_space(1, &coeff_dims);
      H5::DataSet coeff_dataset = group.createDataSet(
          "coefficients", H5::PredType::NATIVE_DOUBLE, coeff_space);
      // Write directly from Eigen's memory without copying
      coeff_dataset.write(coeffs_real.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Store configuration set (delegates to ConfigurationSet serialization)
    H5::Group config_set_group = group.createGroup("configuration_set");
    _configuration_set.to_hdf5(config_set_group);

    // If rdms are available, store
    if (has_one_rdm_spin_dependent() || has_two_rdm_spin_dependent()) {
      H5::Group rdm_group = group.createGroup("rdms");

      if (has_one_rdm_spin_dependent()) {
        // restricted only
        if (get_orbitals()->is_restricted()) {
          std::string storage_name = "one_rdm_aa";
          H5::Attribute one_rdm_aa_complex_attr = group.createAttribute(
              "is_one_rdm_aa_complex", H5::PredType::NATIVE_HBOOL,
              H5::DataSpace(H5S_SCALAR));

          if (_one_rdm_spin_dependent_aa != nullptr) {
            // check if real or complex
            bool is_one_rdm_complex =
                detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
            save_matrix_variant_to_group(is_one_rdm_complex,
                                         *_one_rdm_spin_dependent_aa, rdm_group,
                                         storage_name);

            // store complexity flag
            hbool_t is_one_rdm_aa_complex_flag = is_one_rdm_complex ? 1 : 0;
            one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                          &is_one_rdm_aa_complex_flag);

            // if we dont have aa, save bb
          } else if (_one_rdm_spin_dependent_bb != nullptr &&
                     _one_rdm_spin_dependent_aa == nullptr) {
            // check if real or complex
            bool is_one_rdm_complex =
                detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_bb);
            save_matrix_variant_to_group(is_one_rdm_complex,
                                         *_one_rdm_spin_dependent_bb, rdm_group,
                                         storage_name);

            // store complexity flag
            hbool_t is_one_rdm_aa_complex_flag = is_one_rdm_complex ? 1 : 0;
            one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                          &is_one_rdm_aa_complex_flag);
          } else if (_one_rdm_spin_traced != nullptr &&
                     get_orbitals()->is_restricted()) {
            // only spin traced
            _one_rdm_spin_dependent_aa =
                detail::multiply_matrix_variant(*_one_rdm_spin_traced, 0.5);
            // check if real or complex
            bool is_one_rdm_complex =
                detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
            save_matrix_variant_to_group(is_one_rdm_complex,
                                         *_one_rdm_spin_dependent_aa, rdm_group,
                                         storage_name);

            // store complexity flag
            hbool_t is_one_rdm_aa_complex_flag = is_one_rdm_complex ? 1 : 0;
            one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                          &is_one_rdm_aa_complex_flag);

          } else {
            throw std::runtime_error(
                "Supposedly we have one-rmds available, but they are not "
                "available as _one_rdm_spin_traced, _one_rdm_spin_dependent_aa "
                "or _one_rdm_spin_dependent_bb.");
          }
        } else {
          // unrestricted - want to store both
          std::string storage_name_aa = "one_rdm_aa";
          H5::Attribute one_rdm_aa_complex_attr = group.createAttribute(
              "is_one_rdm_aa_complex", H5::PredType::NATIVE_HBOOL,
              H5::DataSpace(H5S_SCALAR));
          H5::Attribute one_rdm_bb_complex_attr = group.createAttribute(
              "is_one_rdm_bb_complex", H5::PredType::NATIVE_HBOOL,
              H5::DataSpace(H5S_SCALAR));

          bool is_aa_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
          save_matrix_variant_to_group(is_aa_rdm_complex,
                                       *_one_rdm_spin_dependent_aa, rdm_group,
                                       storage_name_aa);
          std::string storage_name_bb = "one_rdm_bb";
          bool is_bb_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_bb);
          save_matrix_variant_to_group(is_bb_rdm_complex,
                                       *_one_rdm_spin_dependent_bb, rdm_group,
                                       storage_name_bb);

          // store complexity flags
          hbool_t is_one_rdm_aa_complex_flag = is_aa_rdm_complex ? 1 : 0;
          one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_aa_complex_flag);
          hbool_t is_one_rdm_bb_complex_flag = is_bb_rdm_complex ? 1 : 0;
          one_rdm_bb_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_bb_complex_flag);
        }
      }

      if (has_two_rdm_spin_dependent()) {
        std::string storage_name_aabb = "two_rdm_aabb";
        std::string storage_name_aaaa = "two_rdm_aaaa";
        H5::Attribute two_rdm_aabb_complex_attr = group.createAttribute(
            "is_two_rdm_aabb_complex", H5::PredType::NATIVE_HBOOL,
            H5::DataSpace(H5S_SCALAR));
        H5::Attribute two_rdm_aaaa_complex_attr = group.createAttribute(
            "is_two_rdm_aaaa_complex", H5::PredType::NATIVE_HBOOL,
            H5::DataSpace(H5S_SCALAR));
        // we need aabb and aaaa for both restricted and unrestricted
        bool is_aabb_rdm_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_aabb);
        save_vector_variant_to_group(is_aabb_rdm_complex,
                                     *_two_rdm_spin_dependent_aabb, rdm_group,
                                     storage_name_aabb);
        bool is_aaaa_rdm_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_aaaa);
        save_vector_variant_to_group(is_aaaa_rdm_complex,
                                     *_two_rdm_spin_dependent_aaaa, rdm_group,
                                     storage_name_aaaa);

        // store complexity flags
        hbool_t is_two_rdm_aabb_complex_flag = is_aabb_rdm_complex ? 1 : 0;
        two_rdm_aabb_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_two_rdm_aabb_complex_flag);
        hbool_t is_two_rdm_aaaa_complex_flag = is_aaaa_rdm_complex ? 1 : 0;
        two_rdm_aaaa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_two_rdm_aaaa_complex_flag);
        if (get_orbitals()->is_unrestricted()) {
          // also save bbbb
          std::string storage_name_bbbb = "two_rdm_bbbb";
          H5::Attribute two_rdm_bbbb_complex_attr = group.createAttribute(
              "is_two_rdm_bbbb_complex", H5::PredType::NATIVE_HBOOL,
              H5::DataSpace(H5S_SCALAR));
          bool is_bbbb_rdm_complex =
              detail::is_vector_variant_complex(*_two_rdm_spin_dependent_bbbb);
          save_vector_variant_to_group(is_bbbb_rdm_complex,
                                       *_two_rdm_spin_dependent_bbbb, rdm_group,
                                       storage_name_bbbb);
          hbool_t is_two_rdm_bbbb_complex_flag = is_bbbb_rdm_complex ? 1 : 0;
          two_rdm_bbbb_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                          &is_two_rdm_bbbb_complex_flag);
        }
      }
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<CasWavefunctionContainer> CasWavefunctionContainer::from_hdf5(
    H5::Group& group) {
  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

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

    // Load coefficients restrictedness flag
    bool is_restricted = false;
    if (group.attrExists("is_restricted")) {
      H5::Attribute restrictedness_attr = group.openAttribute("is_restricted");
      hbool_t is_restricted_flag;
      restrictedness_attr.read(H5::PredType::NATIVE_HBOOL, &is_restricted_flag);
      is_restricted = (is_restricted_flag != 0);
    }
    // Load complexity flag
    bool is_complex = false;
    if (group.attrExists("is_complex")) {
      H5::Attribute complex_attr = group.openAttribute("is_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_complex = (is_complex_flag != 0);
    }
    // Load coefficients
    VectorVariant coefficients;
    if (is_complex) {
      if (!group.nameExists("coefficients")) {
        throw std::runtime_error(
            "HDF5 group missing required 'coefficients' dataset");
      }

      H5::DataSet coeff_dataset = group.openDataSet("coefficients");
      H5::DataSpace coeff_space = coeff_dataset.getSpace();
      hsize_t coeff_size = coeff_space.getSimpleExtentNpoints();

      // Check if it's complex compound type
      H5::DataType dtype = coeff_dataset.getDataType();
      if (dtype.getClass() != H5T_COMPOUND) {
        throw std::runtime_error(
            "Expected complex compound type in HDF5 coefficients dataset");
      }

      // Native complex compound type
      H5::CompType complex_type(sizeof(std::complex<double>));
      complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
      complex_type.insertMember("imag", sizeof(double),
                                H5::PredType::NATIVE_DOUBLE);

      Eigen::VectorXcd coeffs_complex(coeff_size);
      // Read directly into Eigen's memory without intermediate copying
      coeff_dataset.read(coeffs_complex.data(), complex_type);
      coefficients = coeffs_complex;
    } else {
      if (!group.nameExists("coefficients")) {
        throw std::runtime_error(
            "HDF5 group missing required 'coefficients' dataset");
      }
      H5::DataSet coeff_dataset = group.openDataSet("coefficients");
      H5::DataSpace coeff_space = coeff_dataset.getSpace();
      hsize_t coeff_size = coeff_space.getSimpleExtentNpoints();

      Eigen::VectorXd coeffs_real(coeff_size);
      // Read directly into Eigen's memory without copying
      coeff_dataset.read(coeffs_real.data(), H5::PredType::NATIVE_DOUBLE);
      coefficients = coeffs_real;
    }

    // Load configuration set (delegates to ConfigurationSet deserialization)
    // ConfigurationSet now deserializes orbitals internally
    if (!group.nameExists("configuration_set")) {
      throw std::runtime_error(
          "HDF5 group missing required 'configuration_set' subgroup");
    }
    H5::Group config_set_group = group.openGroup("configuration_set");
    auto config_set = ConfigurationSet::from_hdf5(config_set_group);
    const auto& determinants = config_set.get_configurations();
    auto orbitals = config_set.get_orbitals();

    // load rdms if they are available
    if (group.nameExists("rdms")) {
      // initialize variables
      std::optional<MatrixVariant> one_rdm_aa;
      std::optional<MatrixVariant> one_rdm_bb;
      std::optional<VectorVariant> two_rdm_aabb;
      std::optional<VectorVariant> two_rdm_aaaa;
      std::optional<VectorVariant> two_rdm_bbbb;
      std::optional<MatrixVariant> one_rdm_spin_traced;
      std::optional<VectorVariant> two_rdm_spin_traced;

      H5::Group rdm_group = group.openGroup("rdms");

      // check if any one rdms were saved
      if (rdm_group.nameExists("one_rdm_aa")) {
        // check complexity
        bool is_one_rdm_aa_complex = false;
        if (group.attrExists("is_one_rdm_aa_complex")) {
          H5::Attribute complex_attr =
              group.openAttribute("is_one_rdm_aa_complex");
          hbool_t is_complex_flag;
          complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
          is_one_rdm_aa_complex = (is_complex_flag != 0);
        }
        one_rdm_aa = load_matrix_variant_from_group(rdm_group, "one_rdm_aa",
                                                    is_one_rdm_aa_complex);
      } else {
        throw std::runtime_error(
            "One rdms should be available but none were found in hdf5.");
      }

      // check if any two rdms were saved
      if (rdm_group.nameExists("two_rdm_aabb") &&
          rdm_group.nameExists("two_rdm_aaaa")) {
        // check complexity
        bool is_two_rdm_aabb_complex = false;
        if (group.attrExists("is_two_rdm_aabb_complex")) {
          H5::Attribute complex_attr =
              group.openAttribute("is_two_rdm_aabb_complex");
          hbool_t is_complex_flag;
          complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
          is_two_rdm_aabb_complex = (is_complex_flag != 0);
        }
        bool is_two_rdm_aaaa_complex = false;
        if (group.attrExists("is_two_rdm_aaaa_complex")) {
          H5::Attribute complex_attr =
              group.openAttribute("is_two_rdm_aaaa_complex");
          hbool_t is_complex_flag;
          complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
          is_two_rdm_aaaa_complex = (is_complex_flag != 0);
        }
        two_rdm_aabb = load_vector_variant_from_group(rdm_group, "two_rdm_aabb",
                                                      is_two_rdm_aabb_complex);
        two_rdm_aaaa = load_vector_variant_from_group(rdm_group, "two_rdm_aaaa",
                                                      is_two_rdm_aaaa_complex);
      }

      // Determine if we're dealing with restricted or unrestricted orbitals
      is_restricted = orbitals->is_restricted();

      // return restricted object with rdms
      if (is_restricted) {
        if (one_rdm_aa.has_value()) {
          // get one rdm spin traced
          auto spin_traced_result =
              detail::multiply_matrix_variant(*one_rdm_aa, 2.0);
          if (spin_traced_result != nullptr) {
            one_rdm_spin_traced = *spin_traced_result;
          }
        }

        if (two_rdm_aabb.has_value() && two_rdm_aaaa.has_value()) {
          // get two rdm spin traced
          auto two_rdm_ss_result =
              detail::multiply_vector_variant(*two_rdm_aaaa, 2.0);
          auto two_rdm_bbaa_result = detail::transpose_ijkl_klij_vector_variant(
              *two_rdm_aabb, orbitals->get_active_space_indices().first.size());

          if (two_rdm_ss_result != nullptr && two_rdm_bbaa_result != nullptr) {
            auto two_rdm_os_result = detail::add_vector_variants(
                *two_rdm_aabb, *two_rdm_bbaa_result);
            if (two_rdm_os_result != nullptr) {
              auto final_result = detail::add_vector_variants(
                  *two_rdm_ss_result, *two_rdm_os_result);
              if (final_result != nullptr) {
                two_rdm_spin_traced = *final_result;
              }
            }
          }
        }

        // return based on whether stuff is available
        // only one rdms
        if (one_rdm_aa.has_value() && !two_rdm_aabb.has_value()) {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa,
              one_rdm_aa,  // bb is aa
              std::nullopt, std::nullopt, std::nullopt, std::nullopt, type);
        }
        // only two rdms
        else if (!one_rdm_aa.has_value() && two_rdm_aabb.has_value()) {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, std::nullopt,
              std::nullopt, two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_aaaa,  // two_rdm_bbbb is two_rdm_aaaa
              type);
        }
        // both
        else if (one_rdm_aa.has_value() && two_rdm_aabb.has_value()) {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa,
              one_rdm_aa,  // bb is aa
              two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_aaaa,  // two_rdm_bbbb is two_rdm_aaaa
              type);
        } else {
          throw std::runtime_error(
              "Unexpected combination of rdms are available.");
        }
      } else {
        // Unrestricted
        // check if one rdms are available
        if (rdm_group.nameExists("one_rdm_bb")) {
          // check complexity
          bool is_one_rdm_bb_complex = false;
          if (group.attrExists("is_one_rdm_bb_complex")) {
            H5::Attribute complex_attr =
                group.openAttribute("is_one_rdm_bb_complex");
            hbool_t is_complex_flag;
            complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
            is_one_rdm_bb_complex = (is_complex_flag != 0);
          }
          one_rdm_bb = load_matrix_variant_from_group(rdm_group, "one_rdm_bb",
                                                      is_one_rdm_bb_complex);

          // get one rdm spin traced
          auto spin_traced_result =
              detail::add_matrix_variants(*one_rdm_aa, *one_rdm_bb);
          if (spin_traced_result != nullptr) {
            one_rdm_spin_traced = *spin_traced_result;
          }
        } else {
          throw std::runtime_error("Expected aa and bb rdms for unrestricted.");
        }
        // also get two rdms bbbb
        if (rdm_group.nameExists("two_rdm_bbbb")) {
          bool is_two_rdm_bbbb_complex = false;
          if (group.attrExists("is_two_rdm_bbbb_complex")) {
            H5::Attribute complex_attr =
                group.openAttribute("is_two_rdm_bbbb_complex");
            hbool_t is_complex_flag;
            complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
            is_two_rdm_bbbb_complex = (is_complex_flag != 0);
          }
          two_rdm_bbbb = load_vector_variant_from_group(
              rdm_group, "two_rdm_bbbb", is_two_rdm_bbbb_complex);

          // get two rdm spin traced
          auto two_rdm_ss_result =
              detail::add_vector_variants(*two_rdm_aaaa, *two_rdm_bbbb);
          auto two_rdm_bbaa_result = detail::transpose_ijkl_klij_vector_variant(
              *two_rdm_aabb, orbitals->get_active_space_indices().first.size());

          if (two_rdm_ss_result != nullptr && two_rdm_bbaa_result != nullptr) {
            auto two_rdm_os_result = detail::add_vector_variants(
                *two_rdm_aabb, *two_rdm_bbaa_result);
            if (two_rdm_os_result != nullptr) {
              auto final_result = detail::add_vector_variants(
                  *two_rdm_ss_result, *two_rdm_os_result);
              if (final_result != nullptr) {
                two_rdm_spin_traced = *final_result;
              }
            }
          }
        }

        // return based on whether stuff is available
        // only one rdms
        if (one_rdm_aa.has_value() && one_rdm_bb.has_value() &&
            !two_rdm_aabb.has_value()) {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa, one_rdm_bb, std::nullopt, std::nullopt, std::nullopt,
              std::nullopt, type);
        }
        // only two rdms
        else if (!one_rdm_aa.has_value() && two_rdm_aabb.has_value() &&
                 two_rdm_aaaa.has_value() && two_rdm_bbbb.has_value()) {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, std::nullopt,
              std::nullopt, two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        }
        // both
        else if (one_rdm_aa.has_value() && one_rdm_bb.has_value() &&
                 two_rdm_aabb.has_value() && two_rdm_aaaa.has_value() &&
                 two_rdm_bbbb.has_value()) {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa, one_rdm_bb, two_rdm_spin_traced, two_rdm_aabb,
              two_rdm_aaaa, two_rdm_bbbb, type);
        } else {
          throw std::runtime_error(
              "Unexpected combination of rdms are available.");
        }
      }
    }

    return std::make_unique<CasWavefunctionContainer>(
        coefficients, determinants, orbitals, type);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
