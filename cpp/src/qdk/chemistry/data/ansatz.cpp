// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <sstream>
#include <stdexcept>

#include "filename_utils.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {

Ansatz::Ansatz(const Hamiltonian& hamiltonian, const Wavefunction& wavefunction)
    : _hamiltonian(std::make_shared<Hamiltonian>(hamiltonian)),
      _wavefunction(std::make_shared<Wavefunction>(wavefunction)) {
  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid Ansatz object.");
  }
}

Ansatz::Ansatz(std::shared_ptr<Hamiltonian> hamiltonian,
               std::shared_ptr<Wavefunction> wavefunction)
    : _hamiltonian(hamiltonian), _wavefunction(wavefunction) {
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
  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid Ansatz object.");
  }
}

Ansatz& Ansatz::operator=(const Ansatz& other) {
  if (this != &other) {
    _hamiltonian = other._hamiltonian;
    _wavefunction = other._wavefunction;
  }
  return *this;
}

std::shared_ptr<Hamiltonian> Ansatz::get_hamiltonian() const {
  if (!_hamiltonian) {
    throw std::runtime_error("Hamiltonian is not set");
  }
  return _hamiltonian;
}

bool Ansatz::has_hamiltonian() const { return _hamiltonian != nullptr; }

std::shared_ptr<Wavefunction> Ansatz::get_wavefunction() const {
  if (!_wavefunction) {
    throw std::runtime_error("Wavefunction is not set");
  }
  return _wavefunction;
}

bool Ansatz::has_wavefunction() const { return _wavefunction != nullptr; }

std::shared_ptr<Orbitals> Ansatz::get_orbitals() const {
  if (!has_hamiltonian()) {
    throw std::runtime_error("Hamiltonian is not available");
  }
  return _hamiltonian->get_orbitals();
}

bool Ansatz::has_orbitals() const {
  return has_hamiltonian() && _hamiltonian->has_orbitals() &&
         has_wavefunction();
}

double Ansatz::calculate_energy() const {
  if (!_is_valid()) {
    throw std::runtime_error("Cannot calculate energy for invalid Ansatz");
  }

  if (!_wavefunction->has_one_rdm_spin_traced() ||
      !_wavefunction->has_two_rdm_spin_traced()) {
    throw std::runtime_error(
        "Wavefunction does not have spin-traced RDMs available for energy "
        "calculation");
  }

  // get 2 rdm from wavefunction
  const auto& rdm2 = std::get<Eigen::VectorXd>(
      _wavefunction->get_active_two_rdm_spin_traced());
  const auto& rdm1 = std::get<Eigen::MatrixXd>(
      _wavefunction->get_active_one_rdm_spin_traced());

  // get integrals from hamiltonian
  const auto& h1 = _hamiltonian->get_one_body_integrals();
  const auto& h2 = _hamiltonian->get_two_body_integrals();

  // check that active space indices are consistent
  auto active_space_indices =
      _wavefunction->get_orbitals()->get_active_space_indices();
  if (active_space_indices.first.size() != active_space_indices.second.size()) {
    throw std::runtime_error(
        "Active space indices are inconsistent between Hamiltonian and "
        "wavefunction");
  }
  size_t norb = active_space_indices.first.size();

  // Compute energy expectation value
  double energy = 0.0;

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
  std::cout << "Core energy: " << _hamiltonian->get_core_energy() << std::endl;

  return energy;
}

bool Ansatz::_is_valid() const {
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

  // If pointers are different, get the actual objects and compare them
  const auto& ham_orbitals = _hamiltonian->get_orbitals();
  const auto& wf_orbitals = _wavefunction->get_orbitals();

  // If they're not the same object instance, throw an error
  if (&ham_orbitals != &wf_orbitals) {
    throw std::runtime_error(
        "Orbital inconsistency: Hamiltonian and Wavefunction must use the "
        "same "
        "Orbitals object instance");
  }
}

void Ansatz::_validate_construction() const { validate_orbital_consistency(); }

std::string Ansatz::get_summary() const {
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
  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  _to_hdf5_file(filename);
}

std::shared_ptr<Ansatz> Ansatz::from_hdf5_file(const std::string& filename) {
  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  return _from_hdf5_file(filename);
}

void Ansatz::to_json_file(const std::string& filename) const {
  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  _to_json_file(filename);
}

std::shared_ptr<Ansatz> Ansatz::from_json_file(const std::string& filename) {
  // Validate filename
  if (filename.empty()) {
    throw std::runtime_error("Filename cannot be empty");
  }
  return _from_json_file(filename);
}

nlohmann::json Ansatz::to_json() const {
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
        fock_matrix = original_hamiltonian->get_inactive_fock_matrix_alpha();
      } else {
        // Use empty matrix if fock matrix is not set
        fock_matrix = Eigen::MatrixXd(0, 0);
      }

      new_hamiltonian = std::make_shared<Hamiltonian>(
          original_hamiltonian->get_one_body_integrals(),
          original_hamiltonian->get_two_body_integrals(), wavefunction_orbitals,
          original_hamiltonian->get_core_energy(), fock_matrix,
          original_hamiltonian->get_type());
    } else {
      // Create unrestricted hamiltonian with wavefunction's orbitals
      Eigen::MatrixXd fock_matrix_alpha, fock_matrix_beta;
      if (original_hamiltonian->has_inactive_fock_matrix()) {
        fock_matrix_alpha =
            original_hamiltonian->get_inactive_fock_matrix_alpha();
        fock_matrix_beta =
            original_hamiltonian->get_inactive_fock_matrix_beta();
      } else {
        // Use empty matrices if fock matrix is not set
        fock_matrix_alpha = Eigen::MatrixXd(0, 0);
        fock_matrix_beta = Eigen::MatrixXd(0, 0);
      }

      new_hamiltonian = std::make_shared<Hamiltonian>(
          original_hamiltonian->get_one_body_integrals_alpha(),
          original_hamiltonian->get_one_body_integrals_beta(),
          original_hamiltonian->get_two_body_integrals_aaaa(),
          original_hamiltonian->get_two_body_integrals_aabb(),
          original_hamiltonian->get_two_body_integrals_bbbb(),
          wavefunction_orbitals, original_hamiltonian->get_core_energy(),
          fock_matrix_alpha, fock_matrix_beta,
          original_hamiltonian->get_type());
    }

    // 4. Make ansatz with the new hamiltonian and the wavefunction
    return std::make_shared<Ansatz>(new_hamiltonian, wavefunction);

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Ansatz from JSON: " +
                             std::string(e.what()));
  }
}

void Ansatz::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  nlohmann::json j = to_json();
  file << j.dump(2);  // Pretty print with 2-space indentation
  file.close();

  if (file.fail()) {
    throw std::runtime_error("Failed to write to file: " + filename);
  }
}

std::shared_ptr<Ansatz> Ansatz::_from_json_file(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
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
        fock_matrix = original_hamiltonian->get_inactive_fock_matrix_alpha();
      } else {
        // Use empty matrix if Fock matrix is not set
        fock_matrix = Eigen::MatrixXd(0, 0);
      }

      new_hamiltonian = std::make_shared<Hamiltonian>(
          original_hamiltonian->get_one_body_integrals(),
          original_hamiltonian->get_two_body_integrals(), wavefunction_orbitals,
          original_hamiltonian->get_core_energy(), fock_matrix,
          original_hamiltonian->get_type());
    } else {
      // Create unrestricted Hamiltonian with wavefunction's orbitals
      Eigen::MatrixXd fock_matrix_alpha, fock_matrix_beta;
      if (original_hamiltonian->has_inactive_fock_matrix()) {
        fock_matrix_alpha =
            original_hamiltonian->get_inactive_fock_matrix_alpha();
        fock_matrix_beta =
            original_hamiltonian->get_inactive_fock_matrix_beta();
      } else {
        // Use empty matrices if Fock matrix is not set
        fock_matrix_alpha = Eigen::MatrixXd(0, 0);
        fock_matrix_beta = Eigen::MatrixXd(0, 0);
      }

      new_hamiltonian = std::make_shared<Hamiltonian>(
          original_hamiltonian->get_one_body_integrals_alpha(),
          original_hamiltonian->get_one_body_integrals_beta(),
          original_hamiltonian->get_two_body_integrals_aaaa(),
          original_hamiltonian->get_two_body_integrals_aabb(),
          original_hamiltonian->get_two_body_integrals_bbbb(),
          wavefunction_orbitals, original_hamiltonian->get_core_energy(),
          fock_matrix_alpha, fock_matrix_beta,
          original_hamiltonian->get_type());
    }

    // 4. Make ansatz with the new Hamiltonian and the wavefunction
    return std::make_shared<Ansatz>(new_hamiltonian, wavefunction);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Ansatz::_to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group ansatz_group = file.createGroup("/ansatz");
    to_hdf5(ansatz_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Ansatz> Ansatz::_from_hdf5_file(const std::string& filename) {
  try {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group ansatz_group = file.openGroup("/ansatz");
    return from_hdf5(ansatz_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data
