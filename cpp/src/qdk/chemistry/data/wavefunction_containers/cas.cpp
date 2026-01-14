// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/utils/logger.hpp>
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
                               type) {
  QDK_LOG_TRACE_ENTERING();
}

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
                               type) {
  QDK_LOG_TRACE_ENTERING();
}

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
      _configuration_set(dets, orbitals) {
  QDK_LOG_TRACE_ENTERING();
}

std::unique_ptr<WavefunctionContainer> CasWavefunctionContainer::clone() const {
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();

  return _coefficients;
}

std::shared_ptr<Orbitals> CasWavefunctionContainer::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();

  return _configuration_set.get_orbitals();
}

const CasWavefunctionContainer::DeterminantVector&
CasWavefunctionContainer::get_active_determinants() const {
  QDK_LOG_TRACE_ENTERING();
  return _configuration_set.get_configurations();
}

size_t CasWavefunctionContainer::size() const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();

  // Clear all cached RDMs
  _clear_rdms();
}

std::pair<size_t, size_t> CasWavefunctionContainer::get_total_num_electrons()
    const {
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }
  auto [n_alpha, n_beta] = determinants[0].get_n_electrons();
  return {n_alpha, n_beta};
}

bool CasWavefunctionContainer::has_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  return !_coefficients.valueless_by_exception();
}

bool CasWavefunctionContainer::has_configuration_set() const {
  QDK_LOG_TRACE_ENTERING();
  return true;
}

const ConfigurationSet& CasWavefunctionContainer::get_configuration_set()
    const {
  QDK_LOG_TRACE_ENTERING();
  return _configuration_set;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CasWavefunctionContainer::get_total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();

  return "cas";
}

bool CasWavefunctionContainer::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  return detail::is_vector_variant_complex(_coefficients);
}

nlohmann::json CasWavefunctionContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();

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

  // Serialize RDMs if available
  _serialize_rdms_to_json(j);

  return j;
}

}  // namespace qdk::chemistry::data
