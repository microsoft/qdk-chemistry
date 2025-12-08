// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <set>
#include <stdexcept>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry {
namespace data {

// TODO (NAB):  this is a huge file.  Is there a way to break it up into smaller
// pieces? Workitem: 41350

Orbitals::Orbitals(
    const Eigen::MatrixXd& coefficients,
    const std::optional<Eigen::VectorXd>& energies,
    const std::optional<Eigen::MatrixXd>& ao_overlap,
    const std::shared_ptr<BasisSet> basis_set,
    const std::optional<std::tuple<std::vector<size_t>, std::vector<size_t>>>&
        indices)
    : _basis_set(basis_set) {
  // Validate input data
  if (!_basis_set) {
    throw std::runtime_error("Basis set cannot be null");
  }

  if (coefficients.size() == 0) {
    throw std::runtime_error("Coefficient matrix cannot be empty");
  }

  // Validate energies dimensions if provided
  if (energies.has_value() && energies.value().size() != coefficients.cols()) {
    throw std::runtime_error(
        "Energy vector size must match number of orbitals");
  }

  // Set coefficients (restricted)
  _coefficients.first = std::make_shared<Eigen::MatrixXd>(coefficients);
  _coefficients.second = _coefficients.first;

  // Set energies if provided
  if (energies.has_value()) {
    _energies.first = std::make_shared<Eigen::VectorXd>(energies.value());
    _energies.second = _energies.first;  // Restricted: alpha = beta
  }

  // Set AO overlap if provided
  if (ao_overlap.has_value()) {
    // Validate AO overlap matrix dimensions
    const auto& overlap_matrix = ao_overlap.value();
    if (overlap_matrix.rows() != coefficients.rows() ||
        overlap_matrix.cols() != coefficients.rows()) {
      throw std::runtime_error(
          "AO overlap matrix dimensions must match number of atomic orbitals");
    }
    if (overlap_matrix.rows() != overlap_matrix.cols()) {
      throw std::runtime_error("AO overlap matrix must be square");
    }
    _ao_overlap = std::make_unique<Eigen::MatrixXd>(ao_overlap.value());
  }

  // lambda to generate all indices
  auto generate_all_indices = [&coefficients]() {
    const size_t num_cols = static_cast<size_t>(coefficients.cols());
    std::vector<size_t> all_indices(num_cols);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    return all_indices;
  };

  // Set active space indices - default to all indices if not provided
  if (indices.has_value()) {
    _active_space_indices.first = std::get<0>(indices.value());
    _active_space_indices.second =
        std::get<0>(indices.value());  // Restricted: alpha = beta
    _inactive_space_indices.first = std::get<1>(indices.value());
    _inactive_space_indices.second =
        std::get<1>(indices.value());  // Restricted: alpha = beta
  } else {
    _active_space_indices.first = generate_all_indices();
    _active_space_indices.second =
        generate_all_indices();  // Restricted: alpha = beta
  }

  // Validate that active and inactive spaces do not overlap
  std::set<size_t> active_set_alpha(_active_space_indices.first.begin(),
                                    _active_space_indices.first.end());
  std::set<size_t> inactive_set_alpha(_inactive_space_indices.first.begin(),
                                      _inactive_space_indices.first.end());
  std::set<size_t> intersection_alpha;
  std::set_intersection(
      active_set_alpha.begin(), active_set_alpha.end(),
      inactive_set_alpha.begin(), inactive_set_alpha.end(),
      std::inserter(intersection_alpha, intersection_alpha.begin()));
  if (!intersection_alpha.empty()) {
    throw std::runtime_error(
        "Active and inactive space indices overlap for alpha orbitals");
  }

  // Validate after construction is complete to ensure virtual dispatch works
  _post_construction_validate();
}

Orbitals::Orbitals(
    const Eigen::MatrixXd& coefficients_alpha,
    const Eigen::MatrixXd& coefficients_beta,
    const std::optional<Eigen::VectorXd>& energies_alpha,
    const std::optional<Eigen::VectorXd>& energies_beta,
    const std::optional<Eigen::MatrixXd>& ao_overlap,
    const std::shared_ptr<BasisSet> basis_set,
    const std::optional<std::tuple<std::vector<size_t>, std::vector<size_t>,
                                   std::vector<size_t>, std::vector<size_t>>>&
        indices)
    : _basis_set(basis_set) {
  // Validate input data
  if (!_basis_set) {
    throw std::runtime_error("Basis set cannot be null");
  }

  if (coefficients_alpha.size() == 0 || coefficients_beta.size() == 0) {
    throw std::runtime_error("Coefficient matrices cannot be empty");
  }

  // Validate that alpha and beta have consistent dimensions
  if (coefficients_alpha.rows() != coefficients_beta.rows() ||
      coefficients_alpha.cols() != coefficients_beta.cols()) {
    throw std::runtime_error(
        "Alpha and beta coefficient matrices must have the same dimensions");
  }

  // Validate energies are both given or not at all
  if (energies_beta.has_value() != energies_alpha.has_value()) {
    throw std::runtime_error(
        "Either both or neither of alpha and beta energies must be provided");
  }

  // Validate energies dimensions if provided
  if (energies_alpha.has_value() &&
      energies_alpha.value().size() != coefficients_alpha.cols()) {
    throw std::runtime_error(
        "Alpha energy vector size must match number of orbitals");
  }
  if (energies_beta.has_value() &&
      energies_beta.value().size() != coefficients_beta.cols()) {
    throw std::runtime_error(
        "Beta energy vector size must match number of orbitals");
  }

  // Set coefficients (unrestricted)
  _coefficients.first = std::make_shared<Eigen::MatrixXd>(coefficients_alpha);
  _coefficients.second = std::make_shared<Eigen::MatrixXd>(coefficients_beta);

  // Set energies if provided
  if (energies_alpha.has_value()) {
    _energies.first = std::make_shared<Eigen::VectorXd>(energies_alpha.value());
  }
  if (energies_beta.has_value()) {
    _energies.second = std::make_shared<Eigen::VectorXd>(energies_beta.value());
  }

  // Set AO overlap if provided
  if (ao_overlap.has_value()) {
    // Validate AO overlap matrix dimensions
    const auto& overlap_matrix = ao_overlap.value();
    if (overlap_matrix.rows() != coefficients_alpha.rows() ||
        overlap_matrix.cols() != coefficients_alpha.rows()) {
      throw std::runtime_error(
          "AO overlap matrix dimensions must match number of atomic "
          "orbitals");
    }
    if (overlap_matrix.rows() != overlap_matrix.cols()) {
      throw std::runtime_error("AO overlap matrix must be square");
    }
    _ao_overlap = std::make_unique<Eigen::MatrixXd>(ao_overlap.value());
  }

  // Lambda function to generate all indices
  auto generate_all_indices = [&coefficients_alpha]() {
    const size_t num_cols = static_cast<size_t>(coefficients_alpha.cols());
    std::vector<size_t> all_indices(num_cols);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    return all_indices;
  };

  // Set active space indices - default to all indices if not provided
  if (indices.has_value()) {
    _active_space_indices.first = std::get<0>(indices.value());
    _active_space_indices.second = std::get<1>(indices.value());
    _inactive_space_indices.first = std::get<2>(indices.value());
    _inactive_space_indices.second = std::get<3>(indices.value());
  } else {
    // Default to all orbital indices for alpha
    _active_space_indices.first = generate_all_indices();
    // Default to all orbital indices for beta
    _active_space_indices.second = generate_all_indices();
  }

  // Validate that active and inactive spaces do not overlap
  std::set<size_t> active_set_alpha(_active_space_indices.first.begin(),
                                    _active_space_indices.first.end());
  std::set<size_t> inactive_set_alpha(_inactive_space_indices.first.begin(),
                                      _inactive_space_indices.first.end());
  std::set<size_t> intersection_alpha;
  std::set_intersection(
      active_set_alpha.begin(), active_set_alpha.end(),
      inactive_set_alpha.begin(), inactive_set_alpha.end(),
      std::inserter(intersection_alpha, intersection_alpha.begin()));
  if (!intersection_alpha.empty()) {
    throw std::runtime_error(
        "Active and inactive space indices overlap for alpha orbitals");
  }

  // Validate after construction is complete to ensure virtual dispatch works
  _post_construction_validate();
}

Orbitals::~Orbitals() = default;

Orbitals::Orbitals(const Orbitals& other) {
  // Copy coefficients
  if (other._coefficients.first) {
    _coefficients.first =
        std::make_shared<Eigen::MatrixXd>(*other._coefficients.first);
  } else {
    _coefficients.first = nullptr;
  }

  // For restricted calculation, share the same pointer for alpha and beta
  if (other._coefficients.second) {
    if (other._coefficients.first == other._coefficients.second) {
      // If restricted in source, maintain restriction in copy
      _coefficients.second = _coefficients.first;
    } else {
      // If unrestricted in source, create separate beta coefficients
      _coefficients.second =
          std::make_shared<Eigen::MatrixXd>(*other._coefficients.second);
    }
  } else {
    _coefficients.second = nullptr;
  }

  // Copy energies
  if (other._energies.first) {
    _energies.first = std::make_shared<Eigen::VectorXd>(*other._energies.first);
  } else {
    _energies.first = nullptr;
  }

  // For restricted calculation, share the same pointer for alpha and beta
  if (other._energies.second) {
    if (other._energies.first == other._energies.second) {
      // If restricted in source, maintain restriction in copy
      _energies.second = _energies.first;
    } else {
      // If unrestricted in source, create separate beta energies
      _energies.second =
          std::make_shared<Eigen::VectorXd>(*other._energies.second);
    }
  } else {
    _energies.second = nullptr;
  }

  // Copy active space information
  _active_space_indices = other._active_space_indices;

  // Copy inactive space information
  _inactive_space_indices = other._inactive_space_indices;

  // Copy AO overlap
  if (other._ao_overlap) {
    _ao_overlap = std::make_unique<Eigen::MatrixXd>(*other._ao_overlap);
  } else {
    _ao_overlap = nullptr;
  }

  // Copy basis set
  if (other._basis_set) {
    _basis_set = other._basis_set;  // Shared pointer can be safely copied
  } else {
    _basis_set = nullptr;
  }

  // Validate after construction is complete to ensure virtual dispatch works
  _post_construction_validate();
}

Orbitals& Orbitals::operator=(const Orbitals& other) {
  if (this != &other) {  // Self-assignment check
    // Copy coefficients
    if (other._coefficients.first) {
      _coefficients.first =
          std::make_shared<Eigen::MatrixXd>(*other._coefficients.first);
    } else {
      _coefficients.first = nullptr;
    }

    // For restricted calculation, share the same pointer for alpha and beta
    if (other._coefficients.second) {
      if (other._coefficients.first == other._coefficients.second) {
        // If restricted in source, maintain restriction in copy
        _coefficients.second = _coefficients.first;
      } else {
        // If unrestricted in source, create separate beta coefficients
        _coefficients.second =
            std::make_shared<Eigen::MatrixXd>(*other._coefficients.second);
      }
    } else {
      _coefficients.second = nullptr;
    }

    // Copy energies
    if (other._energies.first) {
      _energies.first =
          std::make_shared<Eigen::VectorXd>(*other._energies.first);
    } else {
      _energies.first = nullptr;
    }

    // For restricted calculation, share the same pointer for alpha and beta
    if (other._energies.second) {
      if (other._energies.first == other._energies.second) {
        // If restricted in source, maintain restriction in copy
        _energies.second = _energies.first;
      } else {
        // If unrestricted in source, create separate beta energies
        _energies.second =
            std::make_shared<Eigen::VectorXd>(*other._energies.second);
      }
    } else {
      _energies.second = nullptr;
    }

    // Copy active space
    _active_space_indices = other._active_space_indices;

    // Copy inactive space
    _inactive_space_indices = other._inactive_space_indices;

    // Copy AO overlap
    if (other._ao_overlap) {
      _ao_overlap = std::make_unique<Eigen::MatrixXd>(*other._ao_overlap);
    } else {
      _ao_overlap = nullptr;
    }

    // Copy basis set
    if (other._basis_set) {
      _basis_set = other._basis_set;  // Shared pointer can be safely copied
    } else {
      _basis_set = nullptr;
    }
  }
  return *this;
}

std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
Orbitals::get_coefficients() const {
  if (!_coefficients.first || !_coefficients.second) {
    throw std::runtime_error("Orbital coefficients not set");
  }
  // Check for empty matrices
  if (_coefficients.first->size() == 0 || _coefficients.second->size() == 0) {
    throw std::runtime_error("Orbital coefficients are empty");
  }
  return {*_coefficients.first, *_coefficients.second};
}

std::pair<const Eigen::VectorXd&, const Eigen::VectorXd&>
Orbitals::get_energies() const {
  if (!_energies.first || !_energies.second) {
    throw std::runtime_error("Orbital energies not set");
  }
  return {*_energies.first, *_energies.second};
}

bool Orbitals::has_energies() const {
  if (_energies.first == nullptr || _energies.second == nullptr) {
    return false;
  }
  return true;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
Orbitals::calculate_ao_density_matrix(
    const Eigen::VectorXd& occupations_alpha,
    const Eigen::VectorXd& occupations_beta) const {
  // Validate inputs
  if (!_coefficients.first || !_coefficients.second) {
    throw std::runtime_error("Orbital coefficients not set");
  }

  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  if (occupations_alpha.size() != num_molecular_orbitals ||
      occupations_beta.size() != num_molecular_orbitals) {
    throw std::runtime_error(
        "Occupation vector size must match number of molecular orbitals");
  }

  const auto& C_alpha = *_coefficients.first;
  const auto& C_beta = *_coefficients.second;

  // Calculate density matrices: P = C * n * C^T
  // Where n is the diagonal matrix of occupations
  Eigen::MatrixXd P_alpha =
      C_alpha * occupations_alpha.asDiagonal() * C_alpha.transpose();
  Eigen::MatrixXd P_beta =
      C_beta * occupations_beta.asDiagonal() * C_beta.transpose();

  return {P_alpha, P_beta};
}

Eigen::MatrixXd Orbitals::calculate_ao_density_matrix(
    const Eigen::VectorXd& occupations) const {
  // Validate inputs
  if (!_coefficients.first) {
    throw std::runtime_error("Orbital coefficients not set");
  }

  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  if (occupations.size() != num_molecular_orbitals) {
    throw std::runtime_error(
        "Occupation vector size must match number of molecular orbitals");
  }

  const auto& C = *_coefficients.first;

  // Calculate total density matrix: P = C * n * C^T
  // Where n is the diagonal matrix of total occupations
  Eigen::MatrixXd P = C * occupations.asDiagonal() * C.transpose();

  return P;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
Orbitals::calculate_ao_density_matrix_from_rdm(
    const Eigen::MatrixXd& rdm_alpha, const Eigen::MatrixXd& rdm_beta) const {
  // Validate inputs
  if (!_coefficients.first || !_coefficients.second) {
    throw std::runtime_error("Orbital coefficients not set");
  }

  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  if (rdm_alpha.rows() != num_molecular_orbitals ||
      rdm_alpha.cols() != num_molecular_orbitals ||
      rdm_beta.rows() != num_molecular_orbitals ||
      rdm_beta.cols() != num_molecular_orbitals) {
    throw std::runtime_error(
        "1RDM matrix size must match number of molecular orbitals");
  }

  const auto& C_alpha = *_coefficients.first;
  const auto& C_beta = *_coefficients.second;

  // Transform 1RDM from MO to AO basis: P_AO = C * P_MO * C^T
  Eigen::MatrixXd P_alpha = C_alpha * rdm_alpha * C_alpha.transpose();
  Eigen::MatrixXd P_beta = C_beta * rdm_beta * C_beta.transpose();

  return {P_alpha, P_beta};
}

Eigen::MatrixXd Orbitals::calculate_ao_density_matrix_from_rdm(
    const Eigen::MatrixXd& rdm) const {
  // Validate inputs
  if (!_coefficients.first) {
    throw std::runtime_error("Orbital coefficients not set");
  }

  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  if (rdm.rows() != num_molecular_orbitals ||
      rdm.cols() != num_molecular_orbitals) {
    throw std::runtime_error(
        "1RDM matrix size must match number of molecular orbitals");
  }

  const auto& C = *_coefficients.first;

  // Transform 1RDM from MO to AO basis: P_AO = C * P_MO * C^T
  Eigen::MatrixXd P = C * rdm * C.transpose();

  return P;
}

std::pair<const std::vector<size_t>&, const std::vector<size_t>&>
Orbitals::get_active_space_indices() const {
  return _active_space_indices;
}

std::pair<const std::vector<size_t>&, const std::vector<size_t>&>
Orbitals::get_inactive_space_indices() const {
  return _inactive_space_indices;
}

std::pair<std::vector<size_t>, std::vector<size_t>>
Orbitals::get_virtual_space_indices() const {
  const size_t num_molecular_orbitals = get_num_molecular_orbitals();

  // Create sets for efficient lookup
  std::set<size_t> active_alpha(_active_space_indices.first.begin(),
                                _active_space_indices.first.end());
  std::set<size_t> inactive_alpha(_inactive_space_indices.first.begin(),
                                  _inactive_space_indices.first.end());
  std::set<size_t> active_beta(_active_space_indices.second.begin(),
                               _active_space_indices.second.end());
  std::set<size_t> inactive_beta(_inactive_space_indices.second.begin(),
                                 _inactive_space_indices.second.end());

  // Find virtual orbitals (those not in active or inactive)
  std::vector<size_t> virtual_alpha, virtual_beta;

  for (size_t i = 0; i < num_molecular_orbitals; ++i) {
    if (active_alpha.find(i) == active_alpha.end() &&
        inactive_alpha.find(i) == inactive_alpha.end()) {
      virtual_alpha.push_back(i);
    }
    if (active_beta.find(i) == active_beta.end() &&
        inactive_beta.find(i) == inactive_beta.end()) {
      virtual_beta.push_back(i);
    }
  }

  return {virtual_alpha, virtual_beta};
}

// === AO overlap matrix ===

const Eigen::MatrixXd& Orbitals::get_overlap_matrix() const {
  if (!_ao_overlap) {
    throw std::runtime_error("AO overlap matrix not set");
  }
  return *_ao_overlap;
}

bool Orbitals::has_overlap_matrix() const { return _ao_overlap != nullptr; }

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
Orbitals::get_mo_overlap() const {
  return std::make_tuple(get_mo_overlap_alpha_alpha(),
                         get_mo_overlap_alpha_beta(),
                         get_mo_overlap_beta_beta());
}

Eigen::MatrixXd Orbitals::get_mo_overlap_alpha_alpha() const {
  // Validate inputs
  if (!_ao_overlap) {
    throw std::runtime_error("AO overlap matrix not set");
  }
  if (!_coefficients.first) {
    throw std::runtime_error("Alpha orbital coefficients not set");
  }

  const auto& S_AO = *_ao_overlap;
  const auto& C_alpha = *_coefficients.first;

  // Calculate MO overlap: S_MO^αα = C_α^T * S_AO * C_α
  Eigen::MatrixXd S_MO_alpha_alpha = C_alpha.transpose() * S_AO * C_alpha;

  return S_MO_alpha_alpha;
}

Eigen::MatrixXd Orbitals::get_mo_overlap_alpha_beta() const {
  // Validate inputs
  if (!_ao_overlap) {
    throw std::runtime_error("AO overlap matrix not set");
  }
  if (!_coefficients.first || !_coefficients.second) {
    throw std::runtime_error("Orbital coefficients not set");
  }

  const auto& S_AO = *_ao_overlap;
  const auto& C_alpha = *_coefficients.first;
  const auto& C_beta = *_coefficients.second;

  // Calculate MO overlap: S_MO^αβ = C_α^T * S_AO * C_β
  Eigen::MatrixXd S_MO_alpha_beta = C_alpha.transpose() * S_AO * C_beta;

  return S_MO_alpha_beta;
}

Eigen::MatrixXd Orbitals::get_mo_overlap_beta_beta() const {
  // Validate inputs
  if (!_ao_overlap) {
    throw std::runtime_error("AO overlap matrix not set");
  }
  if (!_coefficients.second) {
    throw std::runtime_error("Beta orbital coefficients not set");
  }

  const auto& S_AO = *_ao_overlap;
  const auto& C_beta = *_coefficients.second;

  // Calculate MO overlap: S_MO^ββ = C_β^T * S_AO * C_β
  Eigen::MatrixXd S_MO_beta_beta = C_beta.transpose() * S_AO * C_beta;

  return S_MO_beta_beta;
}

std::shared_ptr<BasisSet> Orbitals::get_basis_set() const { return _basis_set; }

bool Orbitals::has_basis_set() const { return _basis_set != nullptr; }

size_t Orbitals::get_num_molecular_orbitals() const {
  if (_coefficients.first) {
    return _coefficients.first->cols();
  }
  return 0;
}

size_t Orbitals::get_num_atomic_orbitals() const {
  if (_coefficients.first) {
    return _coefficients.first->rows();
  }
  return 0;
}

std::vector<size_t> Orbitals::get_all_mo_indices() const {
  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  std::vector<size_t> indices(num_molecular_orbitals);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

bool Orbitals::is_restricted() const {
  if (!_coefficients.first || !_coefficients.second) {
    throw std::runtime_error(
        "Cannot determine if orbitals are restricted: orbital coefficients "
        "not "
        "set");
  }
  // Compare coefficient pointers first for efficiency
  if (_coefficients.first == _coefficients.second) {
    return true;
  }
  // If pointers are different, check if the data is the same
  if (!_coefficients.first->isApprox(*_coefficients.second)) {
    return false;
  } else {
    // Also check energies if both are set
    if (_energies.first && _energies.second) {
      if (!_energies.first->isApprox(*_energies.second)) {
        return false;
      }
    }
  }
  return true;
}

bool Orbitals::has_active_space() const {
  return !_active_space_indices.first.empty() ||
         !_active_space_indices.second.empty();
}

bool Orbitals::has_inactive_space() const {
  return !_inactive_space_indices.first.empty() ||
         !_inactive_space_indices.second.empty();
}

bool Orbitals::_is_valid() const {
  // Check if coefficients are set
  if (!_coefficients.first || !_coefficients.second) {
    return false;
  }

  // Check if coefficients are not empty
  if (_coefficients.first->rows() == 0 || _coefficients.first->cols() == 0 ||
      _coefficients.second->rows() == 0 || _coefficients.second->cols() == 0) {
    return false;
  }

  // Check if dimensions are consistent
  if (_coefficients.first->rows() != _coefficients.second->rows() ||
      _coefficients.first->cols() != _coefficients.second->cols()) {
    return false;
  }

  // Check energies if set
  if (_energies.first && _energies.second) {
    if (_energies.first->size() != _coefficients.first->cols() ||
        _energies.second->size() != _coefficients.second->cols()) {
      return false;
    }
  }

  // Check active space indices
  if (!_active_space_indices.first.empty() ||
      !_active_space_indices.second.empty()) {
    // Check that the number of active orbitals is less than the
    // total number of orbitals
    if (_active_space_indices.first.size() > _coefficients.first->cols() ||
        _active_space_indices.second.size() > _coefficients.second->cols()) {
      return false;
    }

    // Check that all the indices are valid
    for (const auto& index : _active_space_indices.first) {
      if (index >= _coefficients.first->cols()) {
        return false;
      }
    }
    for (const auto& index : _active_space_indices.second) {
      if (index >= _coefficients.second->cols()) {
        return false;
      }
    }

    // Check that there are no repeated entries in the active indices
    std::set<size_t> unique_indices_first(_active_space_indices.first.begin(),
                                          _active_space_indices.first.end());
    if (unique_indices_first.size() != _active_space_indices.first.size()) {
      return false;
    }

    std::set<size_t> unique_indices_second(_active_space_indices.second.begin(),
                                           _active_space_indices.second.end());
    if (unique_indices_second.size() != _active_space_indices.second.size()) {
      return false;
    }
  }

  // Check inactive space indices
  if (!_inactive_space_indices.first.empty() ||
      !_inactive_space_indices.second.empty()) {
    // Check that the number of inactive orbitals is less than the
    // total number of orbitals
    if (_inactive_space_indices.first.size() > _coefficients.first->cols() ||
        _inactive_space_indices.second.size() > _coefficients.second->cols()) {
      return false;
    }

    // Check that all the indices are valid
    for (const auto& index : _inactive_space_indices.first) {
      if (index >= _coefficients.first->cols()) {
        return false;
      }
    }
    for (const auto& index : _inactive_space_indices.second) {
      if (index >= _coefficients.second->cols()) {
        return false;
      }
    }

    // Check that there are no repeated entries in the inactive indices
    std::set<size_t> unique_inactive_first(
        _inactive_space_indices.first.begin(),
        _inactive_space_indices.first.end());
    if (unique_inactive_first.size() != _inactive_space_indices.first.size()) {
      return false;
    }

    std::set<size_t> unique_inactive_second(
        _inactive_space_indices.second.begin(),
        _inactive_space_indices.second.end());
    if (unique_inactive_second.size() !=
        _inactive_space_indices.second.size()) {
      return false;
    }

    // Check that active and inactive spaces don't overlap
    std::set<size_t> active_set_alpha(_active_space_indices.first.begin(),
                                      _active_space_indices.first.end());
    std::set<size_t> inactive_set_alpha(_inactive_space_indices.first.begin(),
                                        _inactive_space_indices.first.end());
    for (const auto& idx : inactive_set_alpha) {
      if (active_set_alpha.find(idx) != active_set_alpha.end()) {
        return false;  // Overlap found
      }
    }

    std::set<size_t> active_set_beta(_active_space_indices.second.begin(),
                                     _active_space_indices.second.end());
    std::set<size_t> inactive_set_beta(_inactive_space_indices.second.begin(),
                                       _inactive_space_indices.second.end());
    for (const auto& idx : inactive_set_beta) {
      if (active_set_beta.find(idx) != active_set_beta.end()) {
        return false;  // Overlap found
      }
    }
  }

  // Check AO overlap if set
  if (_ao_overlap) {
    if (_ao_overlap->rows() != _coefficients.first->rows() ||
        _ao_overlap->cols() != _coefficients.first->rows()) {
      return false;
    }
  }

  return true;
}

std::string Orbitals::get_summary() const {
  std::string summary = "Orbitals Summary:\n";
  summary += "  AOs: " + std::to_string(get_num_atomic_orbitals()) + "\n";
  summary += "  MOs: " + std::to_string(get_num_molecular_orbitals()) + "\n";

  summary += "  Type: " +
             std::string(is_restricted() ? "Restricted" : "Unrestricted") +
             "\n";
  summary +=
      "  Has overlap: " + std::string(has_overlap_matrix() ? "Yes" : "No") +
      "\n";
  summary +=
      "  Has basis set: " + std::string(has_basis_set() ? "Yes" : "No") + "\n";
  summary += "  Valid: " + std::string(_is_valid() ? "Yes" : "No") + "\n";

  summary +=
      "  Has active space: " + std::string(has_active_space() ? "Yes" : "No") +
      "\n";
  if (has_active_space()) {
    auto [act_orbitals_alpha, act_orbitals_beta] = get_active_space_indices();
    summary +=
        "  Active Orbitals: α=" + std::to_string(act_orbitals_alpha.size()) +
        ", β=" + std::to_string(act_orbitals_beta.size()) + "\n";
  }
  summary += "  Has inactive space: " +
             std::string(has_inactive_space() ? "Yes" : "No") + "\n";
  if (has_inactive_space()) {
    auto [inact_orbitals_alpha, inact_orbitals_beta] =
        get_inactive_space_indices();
    summary += "  Inactive Orbitals: α=" +
               std::to_string(inact_orbitals_alpha.size()) +
               ", β=" + std::to_string(inact_orbitals_beta.size()) + "\n";
  }
  auto [virt_orbitals_alpha, virt_orbitals_beta] = get_virtual_space_indices();
  summary +=
      "  Virtual Orbitals: α=" + std::to_string(virt_orbitals_alpha.size()) +
      ", β=" + std::to_string(virt_orbitals_beta.size()) + "\n";
  return summary;
}

void Orbitals::to_file(const std::string& filename,
                       const std::string& type) const {
  if (type == "json") {
    _to_json_file(filename);
  } else if (type == "hdf5") {
    _to_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

std::shared_ptr<Orbitals> Orbitals::from_file(const std::string& filename,
                                              const std::string& type) {
  if (type == "json") {
    return _from_json_file(filename);
  } else if (type == "hdf5") {
    return _from_hdf5_file(filename);
  } else {
    throw std::runtime_error("Unsupported file type: " + type +
                             ". Supported types: json, hdf5");
  }
}

void Orbitals::to_hdf5_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "orbitals");

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<Orbitals> Orbitals::from_hdf5_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "orbitals");

  return _from_hdf5_file(validated_filename);
}

void Orbitals::to_json_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "orbitals");

  _to_json_file(validated_filename);
}

std::shared_ptr<Orbitals> Orbitals::from_json_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "orbitals");

  return _from_json_file(validated_filename);
}

void Orbitals::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  auto j = to_json();
  file << j.dump(2);  // Pretty print with 2-space indentation

  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

std::shared_ptr<Orbitals> Orbitals::_from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Orbitals JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json j;
  file >> j;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(j);
}

void Orbitals::_to_hdf5_file(const std::string& filename) const {
  if (!_is_valid()) {
    throw std::runtime_error("Cannot save invalid orbital data to HDF5");
  }

  try {
    // Create HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Use the group-based serialization method
    to_hdf5(file);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Orbitals::to_hdf5(H5::Group& group) const {
  try {
    // Save version first
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    H5::Attribute version_attr =
        group.createAttribute("version", string_type, scalar_space);
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);

    // Save metadata first - only type and essential info, not computed flags
    H5::Group metadata_group = group.createGroup("metadata");

    // Save type information for polymorphism
    H5::Attribute type_attr =
        metadata_group.createAttribute("type", string_type, scalar_space);
    std::string type_name = "Orbitals";
    type_attr.write(string_type, type_name);

    // Save essential metadata that can't be computed from data
    unsigned num_atomic_orbitals = get_num_atomic_orbitals();
    unsigned num_molecular_orbitals = get_num_molecular_orbitals();
    bool restricted = is_restricted();

    H5::DataSet aos_dataset = metadata_group.createDataSet(
        "num_atomic_orbitals", H5::PredType::NATIVE_UINT, scalar_space);
    aos_dataset.write(&num_atomic_orbitals, H5::PredType::NATIVE_UINT);

    H5::DataSet mos_dataset = metadata_group.createDataSet(
        "num_molecular_orbitals", H5::PredType::NATIVE_UINT, scalar_space);
    mos_dataset.write(&num_molecular_orbitals, H5::PredType::NATIVE_UINT);

    H5::DataSet restricted_dataset = metadata_group.createDataSet(
        "is_restricted", H5::PredType::NATIVE_HBOOL, scalar_space);
    restricted_dataset.write(&restricted, H5::PredType::NATIVE_HBOOL);

    // Save coefficient matrices using helper functions
    if (_coefficients.first) {
      save_matrix_to_group(group, "coefficients_alpha", *_coefficients.first);
    }
    if (_coefficients.second) {
      save_matrix_to_group(group, "coefficients_beta", *_coefficients.second);
    }
    // Save energies if available
    if (_energies.first && _energies.second) {
      save_vector_to_group(group, "energies_alpha", *_energies.first);
      save_vector_to_group(group, "energies_beta", *_energies.second);
    }

    // Save active space indices if available
    if (has_active_space()) {
      save_vector_to_group(group, "active_space_indices_alpha",
                           _active_space_indices.first);
      save_vector_to_group(group, "active_space_indices_beta",
                           _active_space_indices.second);
    }

    // Save inactive space indices if available
    if (has_inactive_space()) {
      save_vector_to_group(group, "inactive_space_indices_alpha",
                           _inactive_space_indices.first);
      save_vector_to_group(group, "inactive_space_indices_beta",
                           _inactive_space_indices.second);
    }

    // Save AO overlap if available
    if (_ao_overlap) {
      save_matrix_to_group(group, "ao_overlap", *_ao_overlap);
    }

    // Save nested basis set if available
    if (_basis_set) {
      H5::Group basis_set_group = group.createGroup("basis_set");
      _basis_set->to_hdf5(basis_set_group);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Orbitals> Orbitals::_from_hdf5_file(
    const std::string& filename) {
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

  H5::H5File file;
  try {
    // Open HDF5 file
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Orbitals HDF5 file '" + filename +
                             "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    // Use the group-based deserialization method
    return from_hdf5(file);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to read Orbitals data from HDF5 file '" +
                             filename + "'. " +
                             "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Orbitals> Orbitals::from_hdf5(H5::Group& group) {
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

    // Check type information in metadata (if present) to handle ModelOrbitals
    std::string type_name = "Orbitals";  // Default
    if (group.nameExists("metadata")) {
      H5::Group metadata_group = group.openGroup("metadata");
      if (metadata_group.attrExists("type")) {
        H5::Attribute type_attr = metadata_group.openAttribute("type");
        type_attr.read(string_type, type_name);
      }
    }

    // Handle ModelOrbitals case
    if (type_name == "ModelOrbitals") {
      return ModelOrbitals::from_hdf5(group);
    }

    // Handle regular Orbitals case
    // Load coefficients (required)
    auto coeffs_alpha = load_matrix_from_group(group, "coefficients_alpha");

    // Check if beta coefficients exist
    std::optional<Eigen::MatrixXd> coeffs_beta_opt;
    try {
      auto coeffs_beta = load_matrix_from_group(group, "coefficients_beta");
      coeffs_beta_opt = coeffs_beta;
    } catch (const std::exception&) {
      // Beta coefficients don't exist - this is a restricted calculation
    }

    // Load optional energies
    std::optional<Eigen::VectorXd> energies_alpha, energies_beta;
    if (dataset_exists_in_group(group, "energies_alpha")) {
      energies_alpha = load_vector_from_group(group, "energies_alpha");
    }
    if (dataset_exists_in_group(group, "energies_beta")) {
      energies_beta = load_vector_from_group(group, "energies_beta");
    }

    // Load optional AO overlap
    std::optional<Eigen::MatrixXd> ao_overlap;
    if (dataset_exists_in_group(group, "ao_overlap")) {
      ao_overlap = load_matrix_from_group(group, "ao_overlap");
    }

    // Load optional basis set using nested group method
    std::shared_ptr<BasisSet> basis_set;
    try {
      if (group.nameExists("basis_set")) {
        H5::Group basis_set_group = group.openGroup("basis_set");
        basis_set = BasisSet::from_hdf5(basis_set_group);
      }
    } catch (const std::exception&) { /* optional */
    }

    // Load optional active space information
    std::vector<size_t> active_indices_alpha, active_indices_beta;
    try {
      if (group.nameExists("active_space_indices_alpha")) {
        active_indices_alpha =
            load_size_vector_from_group(group, "active_space_indices_alpha");
      }
    } catch (const std::exception&) { /* optional */
    }
    try {
      if (group.nameExists("active_space_indices_beta")) {
        active_indices_beta =
            load_size_vector_from_group(group, "active_space_indices_beta");
      }
    } catch (const std::exception&) { /* optional */
    }

    // Load optional inactive space information
    std::vector<size_t> inactive_indices_alpha, inactive_indices_beta;
    try {
      if (group.nameExists("inactive_space_indices_alpha")) {
        inactive_indices_alpha =
            load_size_vector_from_group(group, "inactive_space_indices_alpha");
      }
    } catch (const std::exception&) { /* optional */
    }
    try {
      if (group.nameExists("inactive_space_indices_beta")) {
        inactive_indices_beta =
            load_size_vector_from_group(group, "inactive_space_indices_beta");
      }
    } catch (const std::exception&) { /* optional */
    }

    if (coeffs_beta_opt) {
      // Unrestricted case
      return std::make_shared<Orbitals>(
          coeffs_alpha, *coeffs_beta_opt, energies_alpha, energies_beta,
          ao_overlap, basis_set,
          std::make_tuple(std::move(active_indices_alpha),
                          std::move(active_indices_beta),
                          std::move(inactive_indices_alpha),
                          std::move(inactive_indices_beta)));
    } else {
      // Restricted case
      return std::make_shared<Orbitals>(
          coeffs_alpha, energies_alpha, ao_overlap, basis_set,
          std::make_tuple(std::move(active_indices_alpha),
                          std::move(inactive_indices_alpha)));
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

nlohmann::json Orbitals::to_json() const {
  if (!_is_valid()) {
    throw std::runtime_error("Cannot serialize invalid orbital data to JSON");
  }

  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Save type information
  j["type"] = "Orbitals";

  // Save metadata
  j["num_atomic_orbitals"] = get_num_atomic_orbitals();
  j["num_molecular_orbitals"] = get_num_molecular_orbitals();
  j["is_restricted"] = is_restricted();
  j["has_overlap_matrix"] = has_overlap_matrix();

  // Save coefficients as nested arrays
  j["coefficients"] = {{"alpha", matrix_to_json(*_coefficients.first)},
                       {"beta", matrix_to_json(*_coefficients.second)}};

  // Save energies if available
  if (_energies.first && _energies.second) {
    j["energies"] = {{"alpha", vector_to_json(*_energies.first)},
                     {"beta", vector_to_json(*_energies.second)}};
  }

  // Save AO overlap if available
  if (_ao_overlap) {
    j["ao_overlap"] = matrix_to_json(*_ao_overlap);
  }

  // Save active space information if available
  if (has_active_space()) {
    // Save active space indices
    j["active_space_indices"] = {{"alpha", _active_space_indices.first},
                                 {"beta", _active_space_indices.second}};
  }

  // Save inactive space information if available
  if (has_inactive_space()) {
    // Save inactive space indices
    j["inactive_space_indices"] = {{"alpha", _inactive_space_indices.first},
                                   {"beta", _inactive_space_indices.second}};
  }

  // Save basis set if available
  if (_basis_set) {
    j["basis_set"] = _basis_set->to_json();
  }

  return j;
}

std::shared_ptr<Orbitals> Orbitals::from_json(const nlohmann::json& j) {
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Check if this is a ModelOrbitals type (both new and old formats)
    if (j.contains("type") && j["type"] == "ModelOrbitals") {
      return ModelOrbitals::from_json(j);
    }

    // Backward compatibility: detect ModelOrbitals by checking for missing
    // coefficients but present num_orbitals field
    if (j.contains("num_orbitals") && !j.contains("coefficients")) {
      return ModelOrbitals::from_json(j);
    }

    // Load coefficients (required for regular Orbitals)
    if (!j.contains("coefficients") || !j["coefficients"].contains("alpha") ||
        !j["coefficients"].contains("beta")) {
      throw std::runtime_error("JSON missing required coefficient data");
    }

    const auto is_restricted = j.value("is_restricted", false);
    auto coeffs_alpha = json_to_matrix(j["coefficients"]["alpha"]);

    // Load optional energies
    std::optional<Eigen::VectorXd> energies_alpha, energies_beta;
    if (j.contains("energies")) {
      if (j["energies"].contains("alpha")) {
        energies_alpha = json_to_vector(j["energies"]["alpha"]);
      }
      if (j["energies"].contains("beta")) {
        energies_beta = json_to_vector(j["energies"]["beta"]);
      }
    }

    // Load optional AO overlap
    std::optional<Eigen::MatrixXd> ao_overlap;
    if (j.contains("ao_overlap")) {
      ao_overlap = json_to_matrix(j["ao_overlap"]);
    }

    // Load optional basis set
    std::shared_ptr<BasisSet> basis_set;
    if (j.contains("basis_set")) {
      basis_set = BasisSet::from_json(j["basis_set"]);
    }

    // Load optional active space information
    std::vector<size_t> active_indices_alpha, active_indices_beta;
    if (j.contains("active_space_indices") &&
        j["active_space_indices"].contains("alpha") &&
        j["active_space_indices"].contains("beta")) {
      active_indices_alpha =
          j["active_space_indices"]["alpha"].get<std::vector<size_t>>();
      active_indices_beta =
          j["active_space_indices"]["beta"].get<std::vector<size_t>>();
    }

    // Load optional inactive space information
    std::vector<size_t> inactive_indices_alpha, inactive_indices_beta;
    if (j.contains("inactive_space_indices") &&
        j["inactive_space_indices"].contains("alpha") &&
        j["inactive_space_indices"].contains("beta")) {
      inactive_indices_alpha =
          j["inactive_space_indices"]["alpha"].get<std::vector<size_t>>();
      inactive_indices_beta =
          j["inactive_space_indices"]["beta"].get<std::vector<size_t>>();
    }

    if (is_restricted) {
      // For restricted, use single values for energies, treat as restricted
      std::optional<Eigen::VectorXd> energies = energies_alpha;

      return std::make_shared<Orbitals>(
          coeffs_alpha, energies, ao_overlap, basis_set,
          std::make_tuple(std::move(active_indices_alpha),
                          std::move(inactive_indices_alpha)));
    } else {
      auto coeffs_beta = json_to_matrix(j["coefficients"]["beta"]);

      return std::make_shared<Orbitals>(
          coeffs_alpha, coeffs_beta, energies_alpha, energies_beta, ao_overlap,
          basis_set,
          std::make_tuple(std::move(active_indices_alpha),
                          std::move(active_indices_beta),
                          std::move(inactive_indices_alpha),
                          std::move(inactive_indices_beta)));
    }

  } catch (const std::exception& e) {
    throw std::runtime_error("Error parsing JSON: " + std::string(e.what()));
  }
}

void Orbitals::_save_metadata_to_hdf5(H5::H5File& file) const {
  // Extract metadata
  unsigned num_atomic_orbitals = get_num_atomic_orbitals();
  unsigned num_molecular_orbitals = get_num_molecular_orbitals();
  bool restricted = is_restricted();
  bool has_overlap = has_overlap_matrix();
  bool has_basis = has_basis_set();

  // Call the generic function
  _save_orbital_metadata_to_hdf5(file, num_atomic_orbitals,
                                 num_molecular_orbitals, restricted,
                                 has_overlap, has_basis);
}

void Orbitals::_save_orbital_metadata_to_hdf5(
    H5::H5File& file, size_t num_atomic_orbitals, size_t num_molecular_orbitals,
    bool is_restricted, bool has_overlap_matrix, bool has_basis_set) {
  // Create metadata group
  H5::Group metadata_group = file.createGroup("metadata");

  // Save scalar metadata
  H5::DataSpace scalar_space;

  H5::DataSet aos_dataset = metadata_group.createDataSet(
      "num_atomic_orbitals", H5::PredType::NATIVE_UINT, scalar_space);
  aos_dataset.write(&num_atomic_orbitals, H5::PredType::NATIVE_UINT);

  H5::DataSet mos_dataset = metadata_group.createDataSet(
      "num_molecular_orbitals", H5::PredType::NATIVE_UINT, scalar_space);
  mos_dataset.write(&num_molecular_orbitals, H5::PredType::NATIVE_UINT);

  // Save boolean flags
  H5::DataSet restricted_dataset = metadata_group.createDataSet(
      "is_restricted", H5::PredType::NATIVE_HBOOL, scalar_space);
  restricted_dataset.write(&is_restricted, H5::PredType::NATIVE_HBOOL);

  H5::DataSet overlap_dataset = metadata_group.createDataSet(
      "has_overlap_matrix", H5::PredType::NATIVE_HBOOL, scalar_space);
  overlap_dataset.write(&has_overlap_matrix, H5::PredType::NATIVE_HBOOL);

  H5::DataSet basis_dataset = metadata_group.createDataSet(
      "has_basis_set", H5::PredType::NATIVE_HBOOL, scalar_space);
  basis_dataset.write(&has_basis_set, H5::PredType::NATIVE_HBOOL);
}

bool Orbitals::is_unrestricted() const { return !is_restricted(); }

void Orbitals::_post_construction_validate() {
  if (!_is_valid()) {
    throw std::invalid_argument("Tried to generate invalid Orbitals object.");
  }
}

const Eigen::MatrixXd& Orbitals::get_coefficients_alpha() const {
  return get_coefficients().first;
}

const Eigen::MatrixXd& Orbitals::get_coefficients_beta() const {
  return get_coefficients().second;
}

const Eigen::VectorXd& Orbitals::get_energies_alpha() const {
  return get_energies().first;
}

const Eigen::VectorXd& Orbitals::get_energies_beta() const {
  return get_energies().second;
}

// === ModelOrbitals Implementation ===

ModelOrbitals::ModelOrbitals(size_t basis_size, bool restricted)
    : Orbitals(), _num_orbitals(basis_size), _is_restricted(restricted) {
  // Set active space indices to all orbitals by default
  std::vector<size_t> all_indices(basis_size);
  std::iota(all_indices.begin(), all_indices.end(), 0);

  if (restricted) {
    _active_space_indices = {all_indices, all_indices};
  } else {
    // For unrestricted, alpha and beta have the same indices but are
    // independent
    _active_space_indices = {all_indices, all_indices};
  }
  // Inactive space remains empty by default
}

ModelOrbitals::ModelOrbitals(
    size_t basis_size,
    const std::tuple<std::vector<size_t>, std::vector<size_t>>& indices)
    : Orbitals(), _num_orbitals(basis_size), _is_restricted(true) {
  const auto& [active_space_indices, inactive_space_indices] = indices;
  // Validate that all indices are within bounds
  for (size_t idx : active_space_indices) {
    if (idx >= basis_size) {
      throw std::invalid_argument("Active space index " + std::to_string(idx) +
                                  " is >= basis size " +
                                  std::to_string(basis_size));
    }
  }

  for (size_t idx : inactive_space_indices) {
    if (idx >= basis_size) {
      throw std::invalid_argument("Inactive space index " +
                                  std::to_string(idx) + " is >= basis size " +
                                  std::to_string(basis_size));
    }
  }

  // Validate that active + inactive count doesn't exceed basis size
  if (active_space_indices.size() + inactive_space_indices.size() >
      basis_size) {
    throw std::invalid_argument(
        "Total active (" + std::to_string(active_space_indices.size()) +
        ") + inactive (" + std::to_string(inactive_space_indices.size()) +
        ") orbitals exceeds basis size (" + std::to_string(basis_size) + ")");
  }

  // Validate no overlap between active and inactive indices
  std::set<size_t> active_set(active_space_indices.begin(),
                              active_space_indices.end());
  std::set<size_t> inactive_set(inactive_space_indices.begin(),
                                inactive_space_indices.end());

  for (size_t idx : active_set) {
    if (inactive_set.count(idx) > 0) {
      throw std::invalid_argument("Orbital index " + std::to_string(idx) +
                                  " appears in both active and inactive space");
    }
  }

  // Set active and inactive space indices (restricted case)
  _active_space_indices = {active_space_indices, active_space_indices};
  _inactive_space_indices = {inactive_space_indices, inactive_space_indices};
}

ModelOrbitals::ModelOrbitals(
    size_t basis_size,
    const std::tuple<std::vector<size_t>, std::vector<size_t>,
                     std::vector<size_t>, std::vector<size_t>>& indices)
    : Orbitals(), _num_orbitals(basis_size), _is_restricted(false) {
  const auto& [active_space_indices_alpha, active_space_indices_beta,
               inactive_space_indices_alpha, inactive_space_indices_beta] =
      indices;
  // Validate alpha indices
  for (size_t idx : active_space_indices_alpha) {
    if (idx >= basis_size) {
      throw std::invalid_argument("Active alpha space index " +
                                  std::to_string(idx) + " is >= basis size " +
                                  std::to_string(basis_size));
    }
  }

  for (size_t idx : inactive_space_indices_alpha) {
    if (idx >= basis_size) {
      throw std::invalid_argument("Inactive alpha space index " +
                                  std::to_string(idx) + " is >= basis size " +
                                  std::to_string(basis_size));
    }
  }

  // Validate beta indices
  for (size_t idx : active_space_indices_beta) {
    if (idx >= basis_size) {
      throw std::invalid_argument("Active beta space index " +
                                  std::to_string(idx) + " is >= basis size " +
                                  std::to_string(basis_size));
    }
  }

  for (size_t idx : inactive_space_indices_beta) {
    if (idx >= basis_size) {
      throw std::invalid_argument("Inactive beta space index " +
                                  std::to_string(idx) + " is >= basis size " +
                                  std::to_string(basis_size));
    }
  }

  // Validate that active + inactive count doesn't exceed basis size for each
  // spin
  if (active_space_indices_alpha.size() + inactive_space_indices_alpha.size() >
      basis_size) {
    throw std::invalid_argument(
        "Total active alpha (" +
        std::to_string(active_space_indices_alpha.size()) +
        ") + inactive alpha (" +
        std::to_string(inactive_space_indices_alpha.size()) +
        ") orbitals exceeds basis size (" + std::to_string(basis_size) + ")");
  }

  if (active_space_indices_beta.size() + inactive_space_indices_beta.size() >
      basis_size) {
    throw std::invalid_argument(
        "Total active beta (" +
        std::to_string(active_space_indices_beta.size()) +
        ") + inactive beta (" +
        std::to_string(inactive_space_indices_beta.size()) +
        ") orbitals exceeds basis size (" + std::to_string(basis_size) + ")");
  }

  // Validate no overlap between active and inactive indices for alpha
  std::set<size_t> active_alpha_set(active_space_indices_alpha.begin(),
                                    active_space_indices_alpha.end());
  std::set<size_t> inactive_alpha_set(inactive_space_indices_alpha.begin(),
                                      inactive_space_indices_alpha.end());

  for (size_t idx : active_alpha_set) {
    if (inactive_alpha_set.count(idx) > 0) {
      throw std::invalid_argument("Alpha orbital index " + std::to_string(idx) +
                                  " appears in both active and inactive space");
    }
  }

  // Validate no overlap between active and inactive indices for beta
  std::set<size_t> active_beta_set(active_space_indices_beta.begin(),
                                   active_space_indices_beta.end());
  std::set<size_t> inactive_beta_set(inactive_space_indices_beta.begin(),
                                     inactive_space_indices_beta.end());

  for (size_t idx : active_beta_set) {
    if (inactive_beta_set.count(idx) > 0) {
      throw std::invalid_argument("Beta orbital index " + std::to_string(idx) +
                                  " appears in both active and inactive space");
    }
  }

  // Set active and inactive space indices (unrestricted case)
  _active_space_indices = {active_space_indices_alpha,
                           active_space_indices_beta};
  _inactive_space_indices = {inactive_space_indices_alpha,
                             inactive_space_indices_beta};
}

// Copy constructor for ModelOrbitals
ModelOrbitals::ModelOrbitals(const ModelOrbitals& other)
    : Orbitals(),  // Call base class default constructor
      _num_orbitals(other._num_orbitals),
      _is_restricted(other._is_restricted) {
  // Copy the active/inactive space indices from the base class
  _active_space_indices = other._active_space_indices;
  _inactive_space_indices = other._inactive_space_indices;

  // No need to call _post_construction_validate() since ModelOrbitals are
  // always valid
}

// Assignment operator for ModelOrbitals
ModelOrbitals& ModelOrbitals::operator=(const ModelOrbitals& other) {
  if (this != &other) {
    _num_orbitals = other._num_orbitals;
    _is_restricted = other._is_restricted;
    _active_space_indices = other._active_space_indices;
    _inactive_space_indices = other._inactive_space_indices;
  }
  return *this;
}

// Override methods to throw errors for model systems
std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
ModelOrbitals::get_coefficients() const {
  throw std::runtime_error(
      "ModelOrbitals: get_coefficients() not available for model systems");
}

std::pair<const Eigen::VectorXd&, const Eigen::VectorXd&>
ModelOrbitals::get_energies() const {
  throw std::runtime_error(
      "ModelOrbitals: get_energies() not available for model systems");
}

const Eigen::MatrixXd& ModelOrbitals::get_overlap_matrix() const {
  throw std::runtime_error(
      "ModelOrbitals: get_overlap_matrix() not available for model systems");
}

std::shared_ptr<BasisSet> ModelOrbitals::get_basis_set() const {
  throw std::runtime_error(
      "ModelOrbitals: get_basis_set() not available for model systems");
}

const Eigen::MatrixXd& ModelOrbitals::get_coefficients_alpha() const {
  throw std::runtime_error(
      "ModelOrbitals: get_coefficients_alpha() not available for model "
      "systems");
}

const Eigen::MatrixXd& ModelOrbitals::get_coefficients_beta() const {
  throw std::runtime_error(
      "ModelOrbitals: get_coefficients_beta() not available for model "
      "systems");
}

const Eigen::VectorXd& ModelOrbitals::get_energies_alpha() const {
  throw std::runtime_error(
      "ModelOrbitals: get_energies_alpha() not available for model systems");
}

const Eigen::VectorXd& ModelOrbitals::get_energies_beta() const {
  throw std::runtime_error(
      "ModelOrbitals: get_energies_beta() not available for model systems");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
ModelOrbitals::calculate_ao_density_matrix(
    const Eigen::VectorXd& occupations_alpha,
    const Eigen::VectorXd& occupations_beta) const {
  throw std::runtime_error(
      "ModelOrbitals: calculate_ao_density_matrix() not available for model "
      "systems");
}

Eigen::MatrixXd ModelOrbitals::calculate_ao_density_matrix(
    const Eigen::VectorXd& occupations) const {
  throw std::runtime_error(
      "ModelOrbitals: calculate_ao_density_matrix() not available for model "
      "systems");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
ModelOrbitals::calculate_ao_density_matrix_from_rdm(
    const Eigen::MatrixXd& rdm_alpha, const Eigen::MatrixXd& rdm_beta) const {
  throw std::runtime_error(
      "ModelOrbitals: calculate_ao_density_matrix_from_rdm() not available "
      "for "
      "model systems");
}

Eigen::MatrixXd ModelOrbitals::calculate_ao_density_matrix_from_rdm(
    const Eigen::MatrixXd& rdm) const {
  throw std::runtime_error(
      "ModelOrbitals: calculate_ao_density_matrix_from_rdm() not available "
      "for "
      "model systems");
}

// MO overlap methods return identity matrices for model systems
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
ModelOrbitals::get_mo_overlap() const {
  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(num_molecular_orbitals, num_molecular_orbitals);
  return std::make_tuple(identity, identity, identity);
}

Eigen::MatrixXd ModelOrbitals::get_mo_overlap_alpha_alpha() const {
  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  return Eigen::MatrixXd::Identity(num_molecular_orbitals,
                                   num_molecular_orbitals);
}

Eigen::MatrixXd ModelOrbitals::get_mo_overlap_alpha_beta() const {
  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  return Eigen::MatrixXd::Identity(num_molecular_orbitals,
                                   num_molecular_orbitals);
}

Eigen::MatrixXd ModelOrbitals::get_mo_overlap_beta_beta() const {
  const size_t num_molecular_orbitals = get_num_molecular_orbitals();
  return Eigen::MatrixXd::Identity(num_molecular_orbitals,
                                   num_molecular_orbitals);
}

std::vector<size_t> ModelOrbitals::get_all_mo_indices() const {
  std::vector<size_t> indices(_num_orbitals);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

bool ModelOrbitals::is_restricted() const { return _is_restricted; }

bool ModelOrbitals::is_unrestricted() const { return !_is_restricted; }

bool ModelOrbitals::_is_valid() const {
  // ModelOrbitals are always valid - they represent model systems
  // without requiring actual coefficient data
  return true;
}

void ModelOrbitals::_post_construction_validate() {
  // For ModelOrbitals, we call our own _is_valid method which always returns
  // true
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid ModelOrbitals object.");
  }
}

nlohmann::json ModelOrbitals::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Save type information
  j["type"] = "ModelOrbitals";

  // Save metadata
  j["num_orbitals"] = _num_orbitals;
  j["is_restricted"] = _is_restricted;

  // Save active space indices
  j["active_space_indices"] = {{"alpha", _active_space_indices.first},
                               {"beta", _active_space_indices.second}};

  // Save inactive space indices
  j["inactive_space_indices"] = {{"alpha", _inactive_space_indices.first},
                                 {"beta", _inactive_space_indices.second}};

  return j;
}

std::shared_ptr<ModelOrbitals> ModelOrbitals::from_json(
    const nlohmann::json& j) {
  try {
    // Validate version first (only if version field exists, for backward
    // compatibility)
    if (j.contains("version")) {
      validate_serialization_version(SERIALIZATION_VERSION, j["version"]);
    }

    // Load required data
    if (!j.contains("num_orbitals")) {
      throw std::runtime_error("JSON missing required num_orbitals field");
    }

    size_t num_orbitals = j["num_orbitals"];
    bool is_restricted = j.value("is_restricted", true);

    // Load active space indices if available
    std::vector<size_t> active_alpha, active_beta;
    if (j.contains("active_space_indices")) {
      if (j["active_space_indices"].contains("alpha")) {
        active_alpha =
            j["active_space_indices"]["alpha"].get<std::vector<size_t>>();
      }
      if (j["active_space_indices"].contains("beta")) {
        active_beta =
            j["active_space_indices"]["beta"].get<std::vector<size_t>>();
      }
    }

    // Load inactive space indices if available
    std::vector<size_t> inactive_alpha, inactive_beta;
    if (j.contains("inactive_space_indices")) {
      if (j["inactive_space_indices"].contains("alpha")) {
        inactive_alpha =
            j["inactive_space_indices"]["alpha"].get<std::vector<size_t>>();
      }
      if (j["inactive_space_indices"].contains("beta")) {
        inactive_beta =
            j["inactive_space_indices"]["beta"].get<std::vector<size_t>>();
      }
    }

    // Create the appropriate ModelOrbitals object
    if (is_restricted && active_alpha == active_beta &&
        inactive_alpha == inactive_beta) {
      // Use the restricted constructor
      if (active_alpha.empty() && inactive_alpha.empty()) {
        return std::make_shared<ModelOrbitals>(num_orbitals, is_restricted);
      } else {
        return std::make_shared<ModelOrbitals>(
            num_orbitals, std::make_tuple(std::move(active_alpha),
                                          std::move(inactive_alpha)));
      }
    } else {
      // Use the unrestricted constructor
      return std::make_shared<ModelOrbitals>(
          num_orbitals,
          std::make_tuple(std::move(active_alpha), std::move(active_beta),
                          std::move(inactive_alpha), std::move(inactive_beta)));
    }

  } catch (const std::exception& e) {
    throw std::runtime_error("Error parsing ModelOrbitals JSON: " +
                             std::string(e.what()));
  }
}

void ModelOrbitals::to_hdf5(H5::Group& group) const {
  try {
    // Add version attribute
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr =
        group.createAttribute("version", string_type, scalar_space);
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);
    version_attr.close();

    // Save metadata
    H5::Group metadata_group = group.createGroup("metadata");

    // Save type information
    H5::Attribute type_attr =
        metadata_group.createAttribute("type", string_type, scalar_space);
    std::string type_name = "ModelOrbitals";
    type_attr.write(string_type, type_name);

    // Save ModelOrbitals metadata
    unsigned num_orbitals = _num_orbitals;
    bool is_restricted = _is_restricted;

    H5::DataSet orbitals_dataset = metadata_group.createDataSet(
        "num_orbitals", H5::PredType::NATIVE_UINT, scalar_space);
    orbitals_dataset.write(&num_orbitals, H5::PredType::NATIVE_UINT);

    H5::DataSet restricted_dataset = metadata_group.createDataSet(
        "is_restricted", H5::PredType::NATIVE_HBOOL, scalar_space);
    restricted_dataset.write(&is_restricted, H5::PredType::NATIVE_HBOOL);

    // Save active space indices
    save_vector_to_group(group, "active_space_indices_alpha",
                         _active_space_indices.first);
    save_vector_to_group(group, "active_space_indices_beta",
                         _active_space_indices.second);

    // Save inactive space indices
    save_vector_to_group(group, "inactive_space_indices_alpha",
                         _inactive_space_indices.first);
    save_vector_to_group(group, "inactive_space_indices_beta",
                         _inactive_space_indices.second);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<ModelOrbitals> ModelOrbitals::from_hdf5(H5::Group& group) {
  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Load metadata
    H5::Group metadata_group = group.openGroup("metadata");

    H5::DataSpace scalar_space(H5S_SCALAR);

    // Load required data
    H5::DataSet orbitals_dataset = metadata_group.openDataSet("num_orbitals");
    unsigned num_orbitals;
    orbitals_dataset.read(&num_orbitals, H5::PredType::NATIVE_UINT);

    H5::DataSet restricted_dataset =
        metadata_group.openDataSet("is_restricted");
    bool is_restricted;
    restricted_dataset.read(&is_restricted, H5::PredType::NATIVE_HBOOL);

    // Load active space indices
    std::vector<size_t> active_indices_alpha, active_indices_beta;
    try {
      if (group.nameExists("active_space_indices_alpha")) {
        active_indices_alpha =
            load_size_vector_from_group(group, "active_space_indices_alpha");
      }
    } catch (const std::exception&) { /* optional */
    }
    try {
      if (group.nameExists("active_space_indices_beta")) {
        active_indices_beta =
            load_size_vector_from_group(group, "active_space_indices_beta");
      }
    } catch (const std::exception&) { /* optional */
    }

    // Load inactive space indices
    std::vector<size_t> inactive_indices_alpha, inactive_indices_beta;
    try {
      if (group.nameExists("inactive_space_indices_alpha")) {
        inactive_indices_alpha =
            load_size_vector_from_group(group, "inactive_space_indices_alpha");
      }
    } catch (const std::exception&) { /* optional */
    }
    try {
      if (group.nameExists("inactive_space_indices_beta")) {
        inactive_indices_beta =
            load_size_vector_from_group(group, "inactive_space_indices_beta");
      }
    } catch (const std::exception&) { /* optional */
    }

    // Create the appropriate ModelOrbitals object
    if (is_restricted && active_indices_alpha == active_indices_beta &&
        inactive_indices_alpha == inactive_indices_beta) {
      // Use the restricted constructor
      if (active_indices_alpha.empty() && inactive_indices_alpha.empty()) {
        return std::make_shared<ModelOrbitals>(num_orbitals, is_restricted);
      } else {
        return std::make_shared<ModelOrbitals>(
            num_orbitals, std::make_tuple(std::move(active_indices_alpha),
                                          std::move(inactive_indices_alpha)));
      }
    } else {
      // Use the unrestricted constructor
      return std::make_shared<ModelOrbitals>(
          num_orbitals, std::make_tuple(std::move(active_indices_alpha),
                                        std::move(active_indices_beta),
                                        std::move(inactive_indices_alpha),
                                        std::move(inactive_indices_beta)));
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace data
}  // namespace qdk::chemistry
