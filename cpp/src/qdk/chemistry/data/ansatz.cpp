// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

Ansatz::Ansatz(const Hamiltonian& hamiltonian, const Wavefunction& wavefunction)
    : _hamiltonian(std::make_shared<Hamiltonian>(hamiltonian)),
      _wavefunction(std::make_shared<Wavefunction>(wavefunction)) {
  QDK_LOG_TRACE_ENTERING();

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid Ansatz object.");
  }
}

Ansatz::Ansatz(std::shared_ptr<Hamiltonian> hamiltonian,
               std::shared_ptr<Wavefunction> wavefunction)
    : _hamiltonian(hamiltonian), _wavefunction(wavefunction) {
  QDK_LOG_TRACE_ENTERING();

  if (!_hamiltonian) {
    throw std::invalid_argument("Hamiltonian pointer cannot be nullptr");
  }
  if (!_wavefunction) {
    throw std::invalid_argument("Wavefunction pointer cannot be nullptr");
  }
  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid Ansatz object.");
  }
}

Ansatz::Ansatz(const Ansatz& other)
    : _hamiltonian(other._hamiltonian), _wavefunction(other._wavefunction) {
  QDK_LOG_TRACE_ENTERING();

  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid Ansatz object.");
  }
}

Ansatz& Ansatz::operator=(const Ansatz& other) {
  QDK_LOG_TRACE_ENTERING();

  if (this != &other) {
    _hamiltonian = other._hamiltonian;
    _wavefunction = other._wavefunction;
  }
  return *this;
}

std::shared_ptr<Hamiltonian> Ansatz::get_hamiltonian() const {
  QDK_LOG_TRACE_ENTERING();

  if (!_hamiltonian) {
    throw std::runtime_error("Hamiltonian is not set");
  }
  return _hamiltonian;
}

bool Ansatz::has_hamiltonian() const {
  QDK_LOG_TRACE_ENTERING();

  return _hamiltonian != nullptr;
}

std::shared_ptr<Wavefunction> Ansatz::get_wavefunction() const {
  QDK_LOG_TRACE_ENTERING();

  if (!_wavefunction) {
    throw std::runtime_error("Wavefunction is not set");
  }
  return _wavefunction;
}

bool Ansatz::has_wavefunction() const {
  QDK_LOG_TRACE_ENTERING();

  return _wavefunction != nullptr;
}

std::shared_ptr<Orbitals> Ansatz::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();

  if (!has_hamiltonian()) {
    throw std::runtime_error("Hamiltonian is not available");
  }
  return _hamiltonian->get_orbitals();
}

bool Ansatz::has_orbitals() const {
  QDK_LOG_TRACE_ENTERING();

  return has_hamiltonian() && _hamiltonian->has_orbitals() &&
         has_wavefunction();
}

double Ansatz::calculate_energy() const {
  QDK_LOG_TRACE_ENTERING();

  if (!_is_valid()) {
    throw std::runtime_error("Cannot calculate energy for invalid Ansatz");
  }

  if (!_wavefunction->has_one_rdm_spin_traced() ||
      !_wavefunction->has_two_rdm_spin_traced()) {
    throw std::runtime_error(
        "Wavefunction does not have spin-traced RDMs available for energy "
        "calculation");
  }

  double energy = 0.0;

  if (_hamiltonian->is_restricted()) {
    // get 2 rdm from wavefunction
    const auto& rdm2 = std::get<Eigen::VectorXd>(
        _wavefunction->get_active_two_rdm_spin_traced());
    const auto& rdm1 = std::get<Eigen::MatrixXd>(
        _wavefunction->get_active_one_rdm_spin_traced());

    // get integrals from hamiltonian
    const auto& [h1_a, h1_b] = _hamiltonian->get_one_body_integrals();
    // get_two_body_integrals returns (aaaa, aabb, bbbb) tuple
    const auto& [h2_aaaa, h2_aabb, h2_bbbb] =
        _hamiltonian->get_two_body_integrals();
    // For restricted case, all components are the same; use a and aaaa
    const auto& h1 = h1_a;
    const auto& h2 = h2_aaaa;

    // check that active space indices are consistent
    auto active_space_indices =
        _wavefunction->get_orbitals()->get_active_space_indices();
    if (active_space_indices.first.size() !=
        active_space_indices.second.size()) {
      throw std::runtime_error(
          "Active space indices are inconsistent between Hamiltonian and "
          "wavefunction");
    }
    size_t norb = active_space_indices.first.size();

    // One-body contribution
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        energy += h1(p, q) * rdm1(q, p);
      }
    }

    // Two-body contribution
    const size_t norb2 = norb * norb;
    const size_t norb3 = norb * norb2;
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        for (int r = 0; r < norb; ++r) {
          for (int s = 0; s < norb; ++s) {
            size_t index_co = p * norb3 + q * norb2 + r * norb + s;
            size_t index_dm = r * norb3 + s * norb2 + p * norb + q;
            energy += 0.5 * h2(index_co) * rdm2(index_dm);
          }
        }
      }
    }

    // core energy contribution
    energy += _hamiltonian->get_core_energy();
  }

  // Unrestricted case
  else {
    // Use spin-dependent RDMs for unrestricted case
    const auto& [rdm1_aa_var, rdm1_bb_var] =
        _wavefunction->get_active_one_rdm_spin_dependent();
    const auto& [rdm2_aabb_var, rdm2_aaaa_var, rdm2_bbbb_var] =
        _wavefunction->get_active_two_rdm_spin_dependent();

    // Extract RDM matrices/vectors from variants
    const auto& rdm1_aa = std::get<Eigen::MatrixXd>(rdm1_aa_var);
    const auto& rdm1_bb = std::get<Eigen::MatrixXd>(rdm1_bb_var);
    const auto& rdm2_aabb = std::get<Eigen::VectorXd>(rdm2_aabb_var);
    const auto& rdm2_aaaa = std::get<Eigen::VectorXd>(rdm2_aaaa_var);
    const auto& rdm2_bbbb = std::get<Eigen::VectorXd>(rdm2_bbbb_var);

    // get one body integrals from hamiltonian
    const auto& [h1_alpha, h1_beta] = _hamiltonian->get_one_body_integrals();

    // get_two_body_integrals returns (aaaa, aabb, bbbb) tuple
    const auto& [h2_aaaa, h2_aabb, h2_bbbb] =
        _hamiltonian->get_two_body_integrals();

    // check that active space indices are consistent
    auto active_space_indices =
        _wavefunction->get_orbitals()->get_active_space_indices();
    if (active_space_indices.first.size() !=
        active_space_indices.second.size()) {
      throw std::runtime_error(
          "Active space indices are inconsistent between Hamiltonian and "
          "wavefunction");
    }
    size_t norb = active_space_indices.first.size();

    // One-body contribution (alpha + beta)
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        // Alpha contribution
        energy += h1_alpha(p, q) * rdm1_aa(q, p);
        // Beta contribution
        energy += h1_beta(p, q) * rdm1_bb(q, p);
      }
    }

    // Two-body contribution - loop through spin channels
    const size_t norb2 = norb * norb;
    const size_t norb3 = norb * norb2;

    // Alpha-alpha contribution (h2_aaaa)
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        for (int r = 0; r < norb; ++r) {
          for (int s = 0; s < norb; ++s) {
            size_t index_co = p * norb3 + q * norb2 + r * norb + s;
            size_t index_dm = r * norb3 + s * norb2 + p * norb + q;
            energy += 0.5 * h2_aaaa(index_co) * rdm2_aaaa(index_dm);
          }
        }
      }
    }

    // Alpha-beta contribution (h2_aabb)
    // Note: h2_aabb is stored as (ββ|αα), so indices are swapped
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        for (int r = 0; r < norb; ++r) {
          for (int s = 0; s < norb; ++s) {
            // h2_aabb is (ββ|αα) so access as (r,s,p,q) not (p,q,r,s)
            size_t index_co = r * norb3 + s * norb2 + p * norb + q;
            size_t index_dm = r * norb3 + s * norb2 + p * norb + q;
            energy += h2_aabb(index_co) * rdm2_aabb(index_dm);
          }
        }
      }
    }

    // Beta-beta contribution (h2_bbbb)
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        for (int r = 0; r < norb; ++r) {
          for (int s = 0; s < norb; ++s) {
            size_t index_co = p * norb3 + q * norb2 + r * norb + s;
            size_t index_dm = r * norb3 + s * norb2 + p * norb + q;
            energy += 0.5 * h2_bbbb(index_co) * rdm2_bbbb(index_dm);
          }
        }
      }
    }

    // core energy contribution
    energy += _hamiltonian->get_core_energy();
  }

  return energy;
}

bool Ansatz::_is_valid() const {
  QDK_LOG_TRACE_ENTERING();

  try {
    if (!has_hamiltonian() || !has_wavefunction()) {
      return false;
    }

    // Check orbital consistency
    validate_orbital_consistency();
    return true;
  } catch (...) {
    return false;
  }
}

void Ansatz::validate_orbital_consistency() const {
  QDK_LOG_TRACE_ENTERING();

  if (!has_hamiltonian() || !has_wavefunction()) {
    throw std::runtime_error(
        "Cannot validate orbital consistency: missing Hamiltonian or "
        "wavefunction");
  }

  // Check that both Hamiltonian and wavefunction have orbital information
  if (!_hamiltonian->has_orbitals()) {
    throw std::runtime_error(
        "Hamiltonian does not contain orbital information");
  }

  // Get orbital pointers from both components
  const auto ham_orbitals_ptr = _hamiltonian->get_orbitals();
  const auto wf_orbitals_ptr = _wavefunction->get_orbitals();

  // First compare pointers - if they point to the same object, they're
  // consistent
  if (ham_orbitals_ptr.get() == wf_orbitals_ptr.get()) {
    return;  // Same object, orbital consistency is guaranteed
  }

  // If pointers are different, compare the orbital properties for structural
  // equivalence.
  // This is necessary when Hamiltonian and Wavefunction are loaded
  // from separate files.
  const auto& ham_orbitals = *ham_orbitals_ptr;
  const auto& wf_orbitals = *wf_orbitals_ptr;

  // Check number of molecular orbitals
  if (ham_orbitals.get_num_molecular_orbitals() !=
      wf_orbitals.get_num_molecular_orbitals()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian has " +
        std::to_string(ham_orbitals.get_num_molecular_orbitals()) +
        " molecular orbitals but Wavefunction has " +
        std::to_string(wf_orbitals.get_num_molecular_orbitals()));
  }

  // Check number of atomic orbitals
  if (ham_orbitals.get_num_atomic_orbitals() !=
      wf_orbitals.get_num_atomic_orbitals()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian has " +
        std::to_string(ham_orbitals.get_num_atomic_orbitals()) +
        " atomic orbitals but Wavefunction has " +
        std::to_string(wf_orbitals.get_num_atomic_orbitals()));
  }

  // Check restricted/unrestricted consistency
  if (ham_orbitals.is_restricted() != wf_orbitals.is_restricted()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "spin restriction (restricted vs unrestricted)");
  }

  // Check active space indices consistency
  const auto& [ham_active_alpha, ham_active_beta] =
      ham_orbitals.get_active_space_indices();
  const auto& [wf_active_alpha, wf_active_beta] =
      wf_orbitals.get_active_space_indices();

  if (ham_active_alpha != wf_active_alpha) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "alpha active space indices");
  }
  if (ham_active_beta != wf_active_beta) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "beta active space indices");
  }

  // Check inactive space indices consistency
  const auto& [ham_inactive_alpha, ham_inactive_beta] =
      ham_orbitals.get_inactive_space_indices();
  const auto& [wf_inactive_alpha, wf_inactive_beta] =
      wf_orbitals.get_inactive_space_indices();

  if (ham_inactive_alpha != wf_inactive_alpha) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "alpha inactive space indices");
  }
  if (ham_inactive_beta != wf_inactive_beta) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "beta inactive space indices");
  }

  // Compare orbital coefficients numerically

  const auto& [ham_coeffs_alpha, ham_coeffs_beta] =
      ham_orbitals.get_coefficients();
  const auto& [wf_coeffs_alpha, wf_coeffs_beta] =
      wf_orbitals.get_coefficients();

  // Check alpha coefficients
  if (ham_coeffs_alpha.rows() != wf_coeffs_alpha.rows() ||
      ham_coeffs_alpha.cols() != wf_coeffs_alpha.cols()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "alpha coefficient matrix dimensions");
  }

  double alpha_diff = (ham_coeffs_alpha - wf_coeffs_alpha).norm();
  if (alpha_diff > std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "alpha orbital coefficients (norm difference: " +
        std::to_string(alpha_diff) + ")");
  }

  // Check beta coefficients
  if (ham_coeffs_beta.rows() != wf_coeffs_beta.rows() ||
      ham_coeffs_beta.cols() != wf_coeffs_beta.cols()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "beta coefficient matrix dimensions");
  }

  double beta_diff = (ham_coeffs_beta - wf_coeffs_beta).norm();
  if (beta_diff > std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "beta orbital coefficients (norm difference: " +
        std::to_string(beta_diff) + ")");
  }

  // Compare orbital energies if both have them
  if (ham_orbitals.has_energies() && wf_orbitals.has_energies()) {
    constexpr double energy_tolerance = 1e-12;

    const auto& [ham_energies_alpha, ham_energies_beta] =
        ham_orbitals.get_energies();
    const auto& [wf_energies_alpha, wf_energies_beta] =
        wf_orbitals.get_energies();

    // Check alpha energies
    if (ham_energies_alpha.size() != wf_energies_alpha.size()) {
      throw std::runtime_error(
          "Orbital inconsistency: Hamiltonian and Wavefunction have different "
          "number of alpha orbital energies");
    }

    double alpha_energy_diff = (ham_energies_alpha - wf_energies_alpha).norm();
    if (alpha_energy_diff > energy_tolerance) {
      throw std::runtime_error(
          "Orbital inconsistency: Hamiltonian and Wavefunction have different "
          "alpha orbital energies (norm difference: " +
          std::to_string(alpha_energy_diff) + ")");
    }

    // Check beta energies
    if (ham_energies_beta.size() != wf_energies_beta.size()) {
      throw std::runtime_error(
          "Orbital inconsistency: Hamiltonian and Wavefunction have different "
          "number of beta orbital energies");
    }

    double beta_energy_diff = (ham_energies_beta - wf_energies_beta).norm();
    if (beta_energy_diff > energy_tolerance) {
      throw std::runtime_error(
          "Orbital inconsistency: Hamiltonian and Wavefunction have different "
          "beta orbital energies (norm difference: " +
          std::to_string(beta_energy_diff) + ")");
    }
  } else if (ham_orbitals.has_energies() != wf_orbitals.has_energies()) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction have different "
        "orbital energy availability (one has energies, the other does not)");
  }
}

void Ansatz::_validate_construction() const {
  QDK_LOG_TRACE_ENTERING();

  validate_orbital_consistency();
}

std::string Ansatz::get_summary() const {
  QDK_LOG_TRACE_ENTERING();

  std::ostringstream oss;
  oss << "=== Ansatz Summary ===\n";

  if (has_hamiltonian()) {
    oss << "Hamiltonian: Available\n";
    oss << "  Type: "
        << (_hamiltonian->is_hermitian() ? "Hermitian" : "Non-Hermitian")
        << "\n";

    if (_hamiltonian->has_orbitals()) {
      oss << "  Orbitals: Available\n";
    } else {
      oss << "  Orbitals: Not available\n";
    }
  } else {
    oss << "Hamiltonian: Not available\n";
  }

  if (has_wavefunction()) {
    oss << "Wavefunction: Available\n";
    oss << "  Type: ";
    switch (_wavefunction->get_type()) {
      case WavefunctionType::SelfDual:
        oss << "SelfDual";
        break;
      case WavefunctionType::NotSelfDual:
        oss << "NotSelfDual";
        break;
    }
    oss << "\n";
    oss << "  Size: " << _wavefunction->size() << " determinants\n";
    oss << "  Norm: " << _wavefunction->norm() << "\n";
  } else {
    oss << "Wavefunction: Not available\n";
  }

  oss << "Valid: " << (_is_valid() ? "Yes" : "No") << "\n";

  return oss.str();
}

void Ansatz::to_file(const std::string& filename,
                     const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

std::shared_ptr<Ansatz> Ansatz::from_file(const std::string& filename,
                                          const std::string& type) {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

void Ansatz::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  _to_hdf5_file(filename);
}

std::shared_ptr<Ansatz> Ansatz::from_hdf5_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  return _from_hdf5_file(filename);
}

void Ansatz::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  _to_json_file(filename);
}

std::shared_ptr<Ansatz> Ansatz::from_json_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  return _from_json_file(filename);
}

nlohmann::json Ansatz::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Store metadata
  j["type"] = "Ansatz";

  // Delegate to Hamiltonian serialization
  if (has_hamiltonian()) {
    j["hamiltonian"] = _hamiltonian->to_json();
  } else {
    j["hamiltonian"] = nullptr;
  }

  // Delegate to Wavefunction serialization
  if (has_wavefunction()) {
    j["wavefunction"] = _wavefunction->to_json();
  } else {
    j["wavefunction"] = nullptr;
  }

  return j;
}

std::shared_ptr<Ansatz> Ansatz::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Validate JSON structure
    if (!j.contains("type") || j["type"] != "Ansatz") {
      throw std::runtime_error("Invalid JSON: missing or incorrect type field");
    }

    if (!j.contains("hamiltonian") || !j.contains("wavefunction")) {
      throw std::runtime_error(
          "Invalid JSON: missing hamiltonian or wavefunction field");
    }

    // 1. Read the hamiltonian
    std::shared_ptr<Hamiltonian> original_hamiltonian = nullptr;
    if (!j["hamiltonian"].is_null()) {
      original_hamiltonian = Hamiltonian::from_json(j["hamiltonian"]);
    }

    if (!original_hamiltonian) {
      throw std::runtime_error("Cannot create Ansatz: Hamiltonian is required");
    }

    // 2. Read the wavefunction
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (!j["wavefunction"].is_null()) {
      wavefunction = Wavefunction::from_json(j["wavefunction"]);
    }

    if (!wavefunction) {
      throw std::runtime_error(
          "Cannot create Ansatz: Wavefunction is required");
    }

    // 3. Make a new hamiltonian from the original hamiltonian's data plus the
    // wavefunction's orbitals
    auto wavefunction_orbitals = wavefunction->get_orbitals();
    if (!wavefunction_orbitals) {
      throw std::runtime_error("Wavefunction must contain orbital information");
    }

    std::shared_ptr<Hamiltonian> new_hamiltonian;
    if (original_hamiltonian->is_restricted()) {
      // Create restricted hamiltonian with wavefunction's orbitals
      Eigen::MatrixXd fock_matrix;
      if (original_hamiltonian->has_inactive_fock_matrix()) {
        auto [fock_matrix, fock_matrix_beta] =
            original_hamiltonian->get_inactive_fock_matrix();
      } else {
        // Use empty matrix if fock matrix is not set
        fock_matrix = Eigen::MatrixXd(0, 0);
      }

      const auto& [h2_aaaa, h2_aabb, h2_bbbb] =
          original_hamiltonian->get_two_body_integrals();
      const auto& [h_aa, h_bb] = original_hamiltonian->get_one_body_integrals();
      new_hamiltonian = std::make_shared<Hamiltonian>(
          std::make_unique<CanonicalFourCenterHamiltonianContainer>(
              h_aa, h2_aaaa, wavefunction_orbitals,
              original_hamiltonian->get_core_energy(), fock_matrix,
              original_hamiltonian->get_type()));
    } else {
      // Create unrestricted hamiltonian with wavefunction's orbitals
      Eigen::MatrixXd fock_matrix_alpha, fock_matrix_beta;
      if (original_hamiltonian->has_inactive_fock_matrix()) {
        auto [fock_matrix_alpha, fock_matrix_beta] =
            original_hamiltonian->get_inactive_fock_matrix();
      } else {
        // Use empty matrices if fock matrix is not set
        fock_matrix_alpha = Eigen::MatrixXd(0, 0);
        fock_matrix_beta = Eigen::MatrixXd(0, 0);
      }

      const auto& [h2_aaaa, h2_aabb, h2_bbbb] =
          original_hamiltonian->get_two_body_integrals();
      const auto& [h_aa, h_bb] = original_hamiltonian->get_one_body_integrals();

      new_hamiltonian = std::make_shared<Hamiltonian>(
          std::make_unique<CanonicalFourCenterHamiltonianContainer>(
              h_aa, h_bb, h2_aaaa, h2_aabb, h2_bbbb, wavefunction_orbitals,
              original_hamiltonian->get_core_energy(), fock_matrix_alpha,
              fock_matrix_beta, original_hamiltonian->get_type()));
    }

    // 4. Make ansatz with the new hamiltonian and the wavefunction
    return std::make_shared<Ansatz>(new_hamiltonian, wavefunction);

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Ansatz from JSON: " +
                             std::string(e.what()));
  }
}

void Ansatz::_to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  nlohmann::json j = to_json();
  file << j.dump(2);
  file.close();

  if (file.fail()) {
    throw std::runtime_error("Failed to write to file: " + filename);
  }
}

std::shared_ptr<Ansatz> Ansatz::_from_json_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Ansatz JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse JSON file " + filename + ": " +
                             e.what());
  }

  return from_json(j);
}

void Ansatz::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Write metadata
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    std::string type_str = "Ansatz";
    H5::Attribute type_attr =
        group.createAttribute("type", string_type, H5::DataSpace(H5S_SCALAR));
    type_attr.write(string_type, type_str);

    std::string version_str = SERIALIZATION_VERSION;
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    version_attr.write(string_type, version_str);

    // Delegate to Hamiltonian serialization
    if (has_hamiltonian()) {
      H5::Group hamiltonian_group = group.createGroup("hamiltonian");
      _hamiltonian->to_hdf5(hamiltonian_group);
    }

    // Delegate to Wavefunction serialization
    if (has_wavefunction()) {
      H5::Group wavefunction_group = group.createGroup("wavefunction");
      _wavefunction->to_hdf5(wavefunction_group);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Ansatz> Ansatz::from_hdf5(H5::Group& group) {
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

    // Validate HDF5 structure
    if (!group.attrExists("type")) {
      throw std::runtime_error("HDF5 group missing required 'type' attribute");
    }

    H5::Attribute type_attr = group.openAttribute("type");
    std::string type_str;
    type_attr.read(string_type, type_str);

    if (type_str != "Ansatz") {
      throw std::runtime_error("Invalid HDF5 group: incorrect type '" +
                               type_str + "', expected 'Ansatz'");
    }

    // 1. Read the Hamiltonian
    std::shared_ptr<Hamiltonian> original_hamiltonian = nullptr;
    if (group.nameExists("hamiltonian")) {
      H5::Group hamiltonian_group = group.openGroup("hamiltonian");
      original_hamiltonian = Hamiltonian::from_hdf5(hamiltonian_group);
    }

    if (!original_hamiltonian) {
      throw std::runtime_error("Cannot create Ansatz: Hamiltonian is required");
    }

    // 2. Read the wavefunction
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (group.nameExists("wavefunction")) {
      H5::Group wavefunction_group = group.openGroup("wavefunction");
      wavefunction = Wavefunction::from_hdf5(wavefunction_group);
    }

    if (!wavefunction) {
      throw std::runtime_error(
          "Cannot create Ansatz: Wavefunction is required");
    }

    // 3. Make a new Hamiltonian from the original Hamiltonian's data plus the
    // wavefunction's orbitals
    auto wavefunction_orbitals = wavefunction->get_orbitals();
    if (!wavefunction_orbitals) {
      throw std::runtime_error("Wavefunction must contain orbital information");
    }

    std::shared_ptr<Hamiltonian> new_hamiltonian;
    if (original_hamiltonian->is_restricted()) {
      // Create restricted Hamiltonian with wavefunction's orbitals
      Eigen::MatrixXd fock_matrix;
      if (original_hamiltonian->has_inactive_fock_matrix()) {
        auto [fock_matrix, fock_matrix_beta] =
            original_hamiltonian->get_inactive_fock_matrix();
      } else {
        // Use empty matrix if Fock matrix is not set
        fock_matrix = Eigen::MatrixXd(0, 0);
      }

      const auto& [h2_aaaa, h2_aabb, h2_bbbb] =
          original_hamiltonian->get_two_body_integrals();
      const auto& [h_aa, h_bb] = original_hamiltonian->get_one_body_integrals();
      auto container =
          std::make_unique<CanonicalFourCenterHamiltonianContainer>(
              h_aa, h2_aaaa, wavefunction_orbitals,
              original_hamiltonian->get_core_energy(), fock_matrix,
              original_hamiltonian->get_type());
      new_hamiltonian = std::make_shared<Hamiltonian>(std::move(container));
    } else {
      // Create unrestricted Hamiltonian with wavefunction's orbitals
      Eigen::MatrixXd fock_matrix_alpha, fock_matrix_beta;
      if (original_hamiltonian->has_inactive_fock_matrix()) {
        auto [fock_matrix_alpha, fock_matrix_beta] =
            original_hamiltonian->get_inactive_fock_matrix();
      } else {
        // Use empty matrices if Fock matrix is not set
        fock_matrix_alpha = Eigen::MatrixXd(0, 0);
        fock_matrix_beta = Eigen::MatrixXd(0, 0);
      }

      const auto& [h2_aaaa, h2_aabb, h2_bbbb] =
          original_hamiltonian->get_two_body_integrals();
      const auto& [h_aa, h_bb] = original_hamiltonian->get_one_body_integrals();

      new_hamiltonian = std::make_shared<Hamiltonian>(
          std::make_unique<CanonicalFourCenterHamiltonianContainer>(
              h_aa, h_bb, h2_aaaa, h2_aabb, h2_bbbb, wavefunction_orbitals,
              original_hamiltonian->get_core_energy(), fock_matrix_alpha,
              fock_matrix_beta, original_hamiltonian->get_type()));
    }

    // 4. Make ansatz with the new Hamiltonian and the wavefunction
    return std::make_shared<Ansatz>(new_hamiltonian, wavefunction);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Ansatz::_to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group ansatz_group = file.createGroup("/ansatz");
    to_hdf5(ansatz_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Ansatz> Ansatz::_from_hdf5_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Ansatz HDF5 file '" + filename +
                             "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    H5::Group ansatz_group = file.openGroup("/ansatz");
    return from_hdf5(ansatz_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to read Ansatz data from HDF5 file '" +
                             filename + "'. " +
                             "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
