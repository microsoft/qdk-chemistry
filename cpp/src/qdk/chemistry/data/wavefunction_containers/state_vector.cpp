// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
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

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

StateVectorContainer::StateVectorContainer(const VectorVariant& coeffs,
                                           const DeterminantVector& dets,
                                           std::shared_ptr<Orbitals> orbitals,
                                           const std::string& sector,
                                           WavefunctionType type)
    : StateVectorContainer(coeffs, dets, orbitals,
                           std::nullopt,  // one_rdm_spin_traced
                           std::nullopt,  // one_rdm_aa
                           std::nullopt,  // one_rdm_bb
                           std::nullopt,  // two_rdm_spin_traced
                           std::nullopt,  // two_rdm_aaaa
                           std::nullopt,  // two_rdm_aabb
                           std::nullopt,  // two_rdm_bbbb
                           sector, OrbitalEntropies{}, type) {
  QDK_LOG_TRACE_ENTERING();
}

StateVectorContainer::StateVectorContainer(const Configuration& det,
                                           std::shared_ptr<Orbitals> orbitals,
                                           const std::string& sector,
                                           WavefunctionType type)
    : WavefunctionContainer(type),
      _coefficients(Eigen::VectorXd(Eigen::VectorXd::Ones(1))),
      _configuration_set(DeterminantVector{det}, orbitals, sector) {
  QDK_LOG_TRACE_ENTERING();

  // Validate that the configuration represents the active space correctly.
  // Configurations only represent the active space, not the full orbital space
  // (inactive and virtual orbitals are not included).
  const std::string config_str = det.to_string();
  auto [alpha_active, beta_active] = orbitals->get_active_space_indices();
  const auto& active_indices = alpha_active;

  if (!active_indices.empty()) {
    size_t active_space_size = active_indices.size();

    if (det.get_orbital_capacity() < active_space_size) {
      throw std::invalid_argument(
          "StateVectorContainer: configuration has orbital capacity " +
          std::to_string(det.get_orbital_capacity()) +
          " which is insufficient for active space (requires at least " +
          std::to_string(active_space_size) + " orbitals).");
    }

    for (size_t orbital_idx = active_space_size;
         orbital_idx < det.get_orbital_capacity(); ++orbital_idx) {
      if (orbital_idx < config_str.length() && config_str[orbital_idx] != '0') {
        throw std::invalid_argument(
            "StateVectorContainer: configuration has occupied orbital at "
            "index " +
            std::to_string(orbital_idx) +
            " which is beyond the active space size (" +
            std::to_string(active_space_size) +
            "). Only orbitals within the active space can be occupied.");
      }
    }
  }
}

StateVectorContainer::StateVectorContainer(
    const VectorVariant& coeffs, const DeterminantVector& dets,
    std::shared_ptr<Orbitals> orbitals,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::string& sector, const OrbitalEntropies& entropies,
    WavefunctionType type)
    : StateVectorContainer(coeffs, dets, orbitals, one_rdm_spin_traced,
                           std::nullopt,  // one_rdm_aa
                           std::nullopt,  // one_rdm_bb
                           two_rdm_spin_traced,
                           std::nullopt,  // two_rdm_aaaa
                           std::nullopt,  // two_rdm_aabb
                           std::nullopt,  // two_rdm_bbbb
                           sector, entropies, type) {
  QDK_LOG_TRACE_ENTERING();
}

StateVectorContainer::StateVectorContainer(
    const VectorVariant& coeffs, const DeterminantVector& dets,
    std::shared_ptr<Orbitals> orbitals,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<MatrixVariant>& one_rdm_aa,
    const std::optional<MatrixVariant>& one_rdm_bb,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_aaaa,
    const std::optional<VectorVariant>& two_rdm_aabb,
    const std::optional<VectorVariant>& two_rdm_bbbb, const std::string& sector,
    const OrbitalEntropies& entropies, WavefunctionType type)
    : WavefunctionContainer(one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
                            two_rdm_spin_traced, two_rdm_aaaa, two_rdm_aabb,
                            two_rdm_bbbb, entropies, type),
      _coefficients(coeffs),
      _configuration_set(dets, orbitals, sector) {
  QDK_LOG_TRACE_ENTERING();
  auto n_coeffs =
      detail::is_vector_variant_complex(coeffs)
          ? static_cast<size_t>(std::get<Eigen::VectorXcd>(coeffs).size())
          : static_cast<size_t>(std::get<Eigen::VectorXd>(coeffs).size());
  if (n_coeffs != dets.size()) {
    throw std::invalid_argument(
        "StateVectorContainer: coefficient vector size (" +
        std::to_string(n_coeffs) +
        ") does not match the number of determinants (" +
        std::to_string(dets.size()) + ").");
  }
}

StateVectorContainer::StateVectorContainer(
    const VectorVariant& coeffs, const DeterminantVector& dets,
    std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<MatrixVariant> one_rdm_spin_traced,
    std::shared_ptr<VectorVariant> two_rdm_spin_traced,
    std::shared_ptr<const SymmetryBlockedTensorVariant<2>> active_one_rdm,
    std::shared_ptr<const SymmetryBlockedTensorVariant<4>> active_two_rdm,
    const std::string& sector, const OrbitalEntropies& entropies,
    WavefunctionType type)
    : WavefunctionContainer(std::move(one_rdm_spin_traced),
                            std::move(two_rdm_spin_traced),
                            std::move(active_one_rdm),
                            std::move(active_two_rdm), entropies, type),
      _coefficients(coeffs),
      _configuration_set(dets, orbitals, sector) {
  QDK_LOG_TRACE_ENTERING();
  auto n_coeffs =
      detail::is_vector_variant_complex(coeffs)
          ? static_cast<size_t>(std::get<Eigen::VectorXcd>(coeffs).size())
          : static_cast<size_t>(std::get<Eigen::VectorXd>(coeffs).size());
  if (n_coeffs != dets.size()) {
    throw std::invalid_argument(
        "StateVectorContainer: coefficient vector size (" +
        std::to_string(n_coeffs) +
        ") does not match the number of determinants (" +
        std::to_string(dets.size()) + ").");
  }
}

// ---------------------------------------------------------------------------
// Basic accessors
// ---------------------------------------------------------------------------

bool StateVectorContainer::_is_single_determinant() const {
  return size() == 1;
}

std::unique_ptr<WavefunctionContainer> StateVectorContainer::clone() const {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<StateVectorContainer>(
      _coefficients, _configuration_set.get_configurations(),
      this->get_orbitals(), _one_rdm_spin_traced, _two_rdm_spin_traced,
      _active_one_rdm, _active_two_rdm,
      _configuration_set.sector_layout().front().first, _entropies,
      this->get_type());
}

std::shared_ptr<Orbitals> StateVectorContainer::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _configuration_set.get_orbitals();
}

std::vector<std::string> StateVectorContainer::sectors() const {
  QDK_LOG_TRACE_ENTERING();
  std::vector<std::string> names;
  for (const auto& [name, basis] : _configuration_set.sector_layout()) {
    names.push_back(name);
  }
  return names;
}

std::shared_ptr<const Orbitals> StateVectorContainer::sector_basis(
    const std::string& name) const {
  QDK_LOG_TRACE_ENTERING();
  for (const auto& [sector_name, basis] : _configuration_set.sector_layout()) {
    if (sector_name == name) {
      return basis;
    }
  }
  throw std::out_of_range("Container has no sector named '" + name + "'");
}

ScalarVariant StateVectorContainer::get_coefficient(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  bool complex = detail::is_vector_variant_complex(_coefficients);
  auto it = std::find(determinants.begin(), determinants.end(), det);
  if (it != determinants.end()) {
    size_t index = std::distance(determinants.begin(), it);
    if (complex) {
      return std::get<Eigen::VectorXcd>(_coefficients)(index);
    }
    return std::get<Eigen::VectorXd>(_coefficients)(index);
  }
  // A determinant absent from the expansion has zero amplitude.
  if (complex) {
    return std::complex<double>(0.0, 0.0);
  }
  return 0.0;
}

const StateVectorContainer::VectorVariant&
StateVectorContainer::get_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  return _coefficients;
}

const StateVectorContainer::DeterminantVector&
StateVectorContainer::get_active_determinants() const {
  QDK_LOG_TRACE_ENTERING();
  return _configuration_set.get_configurations();
}

const ConfigurationSet& StateVectorContainer::get_configuration_set() const {
  QDK_LOG_TRACE_ENTERING();
  return _configuration_set;
}

size_t StateVectorContainer::size() const {
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

ScalarVariant StateVectorContainer::overlap(
    const WavefunctionContainer& other) const {
  QDK_LOG_TRACE_ENTERING();

  const auto* other_sv = dynamic_cast<const StateVectorContainer*>(&other);
  if (!other_sv) {
    throw std::runtime_error(
        "Overlap only implemented between two StateVectorContainer");
  }
  if (this->size() != other_sv->size()) {
    throw std::runtime_error(
        "Overlap only implemented for wavefunctions with same number of "
        "determinants");
  }
  if (this->get_active_determinants()[0].bits_per_mode() == 2) {
    if (this->get_active_determinants()[0].get_n_electrons() !=
        other_sv->get_active_determinants()[0].get_n_electrons()) {
      throw std::runtime_error(
          "Overlap only implemented for wavefunctions with same number of "
          "electrons");
    }
  } else {
    if (this->get_active_determinants()[0].total_occupation() !=
        other_sv->get_active_determinants()[0].total_occupation()) {
      throw std::runtime_error(
          "Overlap only implemented for wavefunctions with same number of "
          "particles");
    }
  }
  if (this->get_orbitals() != other_sv->get_orbitals()) {
    throw std::runtime_error(
        "Overlap only implemented for wavefunctions with same orbitals");
  }

  const auto& coeffs1 = this->get_coefficients();
  const auto& coeffs2 = other_sv->get_coefficients();

  bool coeffs1_complex = detail::is_vector_variant_complex(coeffs1);
  bool coeffs2_complex = detail::is_vector_variant_complex(coeffs2);

  if (!coeffs1_complex && !coeffs2_complex) {
    const auto& real_coeffs1 = std::get<Eigen::VectorXd>(coeffs1);
    const auto& real_coeffs2 = std::get<Eigen::VectorXd>(coeffs2);
    return real_coeffs1.dot(real_coeffs2);
  } else if (coeffs1_complex && coeffs2_complex) {
    const auto& complex_coeffs1 = std::get<Eigen::VectorXcd>(coeffs1);
    const auto& complex_coeffs2 = std::get<Eigen::VectorXcd>(coeffs2);
    return complex_coeffs1.adjoint() * complex_coeffs2;
  } else if (coeffs1_complex && !coeffs2_complex) {
    const auto& complex_coeffs1 = std::get<Eigen::VectorXcd>(coeffs1);
    const auto& real_coeffs2 = std::get<Eigen::VectorXd>(coeffs2);
    return complex_coeffs1.adjoint() *
           real_coeffs2.cast<std::complex<double>>();
  } else {
    const auto& real_coeffs1 = std::get<Eigen::VectorXd>(coeffs1);
    const auto& complex_coeffs2 = std::get<Eigen::VectorXcd>(coeffs2);
    return real_coeffs1.cast<std::complex<double>>().adjoint() *
           complex_coeffs2;
  }
}

double StateVectorContainer::norm() const {
  QDK_LOG_TRACE_ENTERING();
  return std::visit([](const auto& vec) -> double { return vec.norm(); },
                    _coefficients);
}

bool StateVectorContainer::contains_determinant(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = get_active_determinants();
  return std::find(determinants.begin(), determinants.end(), det) !=
         determinants.end();
}

void StateVectorContainer::clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  _clear_rdms();
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
StateVectorContainer::total_num_particles() const {
  QDK_LOG_TRACE_ENTERING();
  const auto& dets = get_active_determinants();
  if (dets.empty()) {
    throw std::runtime_error("No determinants available");
  }
  if (dets[0].bits_per_mode() != 2) {
    // Generic (non-spin-½): aggregate count, no spin decomposition.
    // Use only one channel of inactive indices — for spinless bases
    // v1_indices_from_index_set duplicates the trivial-label indices into
    // both alpha and beta, so summing both would double-count.
    std::size_t active = dets[0].total_occupation();
    auto [alpha_inactive, _] = get_orbitals()->get_inactive_space_indices();
    return _make_particle_count(active + alpha_inactive.size(), 0);
  }
  auto [n_alpha, n_beta] = dets[0].get_n_electrons();
  auto [alpha_inactive, beta_inactive] =
      get_orbitals()->get_inactive_space_indices();
  return _make_particle_count(n_alpha + alpha_inactive.size(),
                              n_beta + beta_inactive.size());
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
StateVectorContainer::active_num_particles() const {
  QDK_LOG_TRACE_ENTERING();
  const auto& dets = get_active_determinants();
  if (dets.empty()) {
    throw std::runtime_error("No determinants available");
  }
  if (dets[0].bits_per_mode() != 2) {
    return _make_particle_count(dets[0].total_occupation(), 0);
  }
  auto [n_alpha, n_beta] = dets[0].get_n_electrons();
  return _make_particle_count(n_alpha, n_beta);
}

bool StateVectorContainer::has_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  return !_coefficients.valueless_by_exception();
}

bool StateVectorContainer::has_configuration_set() const {
  QDK_LOG_TRACE_ENTERING();
  return true;
}

std::string StateVectorContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "state_vector";
}

bool StateVectorContainer::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  return detail::is_vector_variant_complex(_coefficients);
}

// ---------------------------------------------------------------------------
// Reduced density matrices and occupations
//
// Single-determinant expansions generate active-space RDMs, orbital
// occupations, and single-orbital entropies on the fly from the determinant
// occupations. Multi-determinant expansions use the stored RDMs (via the base
// class), diagonalizing the 1-RDM for orbital occupations.
// ---------------------------------------------------------------------------

bool StateVectorContainer::has_one_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
  // A single determinant can generate its spin-dependent RDM on the fly only
  // when the basis declares a spin (S_z) axis to block it by.
  if (_is_single_determinant()) {
    auto sym = get_orbitals()->symmetries();
    if (sym && sym->has_axis(AxisName::Spin)) {
      return true;
    }
  }
  return WavefunctionContainer::has_one_rdm_spin_dependent();
}

bool StateVectorContainer::has_one_rdm_spin_traced() const {
  QDK_LOG_TRACE_ENTERING();
  return _is_single_determinant() ||
         WavefunctionContainer::has_one_rdm_spin_traced();
}

bool StateVectorContainer::has_two_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
  // See has_one_rdm_spin_dependent: lazy generation needs a spin axis.
  if (_is_single_determinant()) {
    auto sym = get_orbitals()->symmetries();
    if (sym && sym->has_axis(AxisName::Spin)) {
      return true;
    }
  }
  return WavefunctionContainer::has_two_rdm_spin_dependent();
}

bool StateVectorContainer::has_two_rdm_spin_traced() const {
  QDK_LOG_TRACE_ENTERING();
  return _is_single_determinant() ||
         WavefunctionContainer::has_two_rdm_spin_traced();
}

const SymmetryBlockedTensorVariant<2>& StateVectorContainer::active_one_rdm()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (!_active_one_rdm && _is_single_determinant()) {
    auto sym = get_orbitals()->symmetries();
    if (!sym || !sym->has_axis(AxisName::Spin)) {
      throw std::runtime_error(
          "Active 1-RDM is unavailable: the orbital basis declares no spin "
          "(S_z) axis, so a spin-blocked active-space 1-RDM cannot be "
          "generated "
          "on the fly. Attach an explicit spin symmetry to the orbitals to "
          "compute spin-resolved RDMs.");
    }
    auto [alpha_occupations, beta_occupations] = _active_occupations_pair();
    if (get_orbitals()->get_active_space_indices().first.size() !=
        get_orbitals()->get_active_space_indices().second.size()) {
      throw std::runtime_error(
          "Spin dependent 1-RDMs not implemented for different alpha and beta "
          "active space sizes");
    }
    if (alpha_occupations.size() != beta_occupations.size()) {
      throw std::runtime_error(
          "Mismatched sizes in active orbital occupations for alpha and beta");
    }
    size_t n_orbs = get_orbitals()->get_active_space_indices().first.size();
    Eigen::MatrixXd tmp_one_rdm_aa = Eigen::MatrixXd::Zero(n_orbs, n_orbs);
    Eigen::MatrixXd tmp_one_rdm_bb = Eigen::MatrixXd::Zero(n_orbs, n_orbs);

    for (size_t i = 0; i < alpha_occupations.size(); ++i) {
      if (alpha_occupations(i) > 0.0) tmp_one_rdm_aa(i, i) = 1.0;
    }
    for (size_t i = 0; i < beta_occupations.size(); ++i) {
      if (beta_occupations(i) > 0.0) tmp_one_rdm_bb(i, i) = 1.0;
    }

    // Restrictedness requires both a closed-shell determinant (alpha/beta
    // occupation patterns coincide) AND restricted orbitals (alpha/beta MO
    // coefficients share storage). Either being open-shell forces distinct
    // alpha/beta blocks.
    const bool restricted = get_active_determinants()[0].is_closed_shell() &&
                            get_orbitals()->is_restricted();
    _active_one_rdm = std::make_shared<const SymmetryBlockedTensorVariant<2>>(
        std::in_place_type<SymmetryBlockedTensor<2, double>>,
        make_spin_diagonal_rank2_sbt(tmp_one_rdm_aa, tmp_one_rdm_bb,
                                     restricted));
  }
  return WavefunctionContainer::active_one_rdm();
}

const SymmetryBlockedTensorVariant<4>& StateVectorContainer::active_two_rdm()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (!_active_two_rdm && _is_single_determinant()) {
    auto sym = get_orbitals()->symmetries();
    if (!sym || !sym->has_axis(AxisName::Spin)) {
      throw std::runtime_error(
          "Active 2-RDM is unavailable: the orbital basis declares no spin "
          "(S_z) axis, so a spin-blocked active-space 2-RDM cannot be "
          "generated "
          "on the fly. Attach an explicit spin symmetry to the orbitals to "
          "compute spin-resolved RDMs.");
    }
    auto [alpha_occupations, beta_occupations] = _active_occupations_pair();
    if (get_orbitals()->get_active_space_indices().first.size() !=
        get_orbitals()->get_active_space_indices().second.size()) {
      throw std::runtime_error(
          "Spin dependent 2-RDMs not implemented for different alpha and beta "
          "active space sizes");
    }
    if (alpha_occupations.size() != beta_occupations.size()) {
      throw std::runtime_error(
          "Mismatched sizes in active orbital occupations for alpha and beta");
    }

    size_t norbs = alpha_occupations.size();
    size_t norb2 = norbs * norbs;
    size_t norb3 = norbs * norb2;
    Eigen::VectorXd tmp_two_rdm_aabb = Eigen::VectorXd::Zero(norb2 * norb2);
    Eigen::VectorXd tmp_two_rdm_aaaa = Eigen::VectorXd::Zero(norb2 * norb2);
    Eigen::VectorXd tmp_two_rdm_bbbb = Eigen::VectorXd::Zero(norb2 * norb2);

    auto build_same_spin_block = [&](const Eigen::VectorXd& occupations,
                                     Eigen::VectorXd& target) {
      for (size_t i = 0; i < norbs; ++i) {
        for (size_t j = i + 1; j < norbs; ++j) {
          if (occupations(i) > 0.0 && occupations(j) > 0.0) {
            size_t index_iijj = i * norb3 + i * norb2 + j * norbs + j;
            size_t index_jjii = j * norb3 + j * norb2 + i * norbs + i;
            target(index_iijj) = 1.0;
            target(index_jjii) = 1.0;
            size_t index_ijji = i * norb3 + j * norb2 + j * norbs + i;
            size_t index_jiij = j * norb3 + i * norb2 + i * norbs + j;
            target(index_ijji) = -1.0;
            target(index_jiij) = -1.0;
          }
        }
      }
    };

    build_same_spin_block(alpha_occupations, tmp_two_rdm_aaaa);
    build_same_spin_block(beta_occupations, tmp_two_rdm_bbbb);

    for (size_t i = 0; i < norbs; ++i) {
      for (size_t j = i; j < norbs; ++j) {
        if (alpha_occupations(i) > 0.0 && beta_occupations(j) > 0.0) {
          size_t index_iijj = i * norb3 + i * norb2 + j * norbs + j;
          tmp_two_rdm_aabb(index_iijj) = 1.0;
        }
        if (alpha_occupations(j) > 0.0 && beta_occupations(i) > 0.0) {
          size_t index_jjii = j * norb3 + j * norb2 + i * norbs + i;
          tmp_two_rdm_aabb(index_jjii) = 1.0;
        }
      }
    }
    const bool restricted = get_active_determinants()[0].is_closed_shell() &&
                            get_orbitals()->is_restricted();
    _active_two_rdm = std::make_shared<const SymmetryBlockedTensorVariant<4>>(
        std::in_place_type<SymmetryBlockedTensor<4, double>>,
        make_spin_diagonal_rank4_sbt(tmp_two_rdm_aaaa, tmp_two_rdm_aabb,
                                     tmp_two_rdm_bbbb, restricted));
  }
  return WavefunctionContainer::active_two_rdm();
}

const MatrixVariant& StateVectorContainer::get_active_one_rdm_spin_traced()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (_is_single_determinant() && !_one_rdm_spin_traced && !_active_one_rdm) {
    if (get_orbitals()->get_active_space_indices().first.size() !=
        get_orbitals()->get_active_space_indices().second.size()) {
      throw std::runtime_error(
          "Spin traced 1-RDM not implemented for different alpha and beta "
          "active space sizes");
    }
    auto [alpha_occupations, beta_occupations] = _active_occupations_pair();
    size_t n_orbs = get_orbitals()->get_active_space_indices().first.size();
    Eigen::MatrixXd tmp_one_rdm = Eigen::MatrixXd::Zero(n_orbs, n_orbs);
    for (size_t i = 0; i < alpha_occupations.size(); ++i) {
      if (alpha_occupations(i) > 0.0) tmp_one_rdm(i, i) += 1.0;
    }
    for (size_t i = 0; i < beta_occupations.size(); ++i) {
      if (beta_occupations(i) > 0.0) tmp_one_rdm(i, i) += 1.0;
    }
    _one_rdm_spin_traced =
        std::make_shared<MatrixVariant>(std::move(tmp_one_rdm));
    return *_one_rdm_spin_traced;
  }
  return WavefunctionContainer::get_active_one_rdm_spin_traced();
}

const VectorVariant& StateVectorContainer::get_active_two_rdm_spin_traced()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (_is_single_determinant() && !_two_rdm_spin_traced && !_active_two_rdm) {
    auto [alpha_occupations, beta_occupations] = _active_occupations_pair();
    if (get_orbitals()->get_active_space_indices().first.size() !=
        get_orbitals()->get_active_space_indices().second.size()) {
      throw std::runtime_error(
          "Spin-traced 2-RDM not implemented for different alpha and beta "
          "active space sizes");
    }
    if (alpha_occupations.size() != beta_occupations.size()) {
      throw std::runtime_error(
          "Mismatched sizes in active orbital occupations for alpha and beta");
    }
    size_t norbs = alpha_occupations.size();
    size_t norb2 = norbs * norbs;
    size_t norb3 = norbs * norb2;
    Eigen::VectorXd tmp_two_rdm = Eigen::VectorXd::Zero(norb2 * norb2);

    for (size_t i = 0; i < norbs; ++i) {
      double occ_alpha_i = alpha_occupations(i);
      double occ_beta_i = beta_occupations(i);
      double occ_sum_i = occ_alpha_i + occ_beta_i;

      if (occ_alpha_i > 0.0 && occ_beta_i > 0.0) {
        size_t index_iiii = i * norb3 + i * norb2 + i * norbs + i;
        tmp_two_rdm(index_iiii) = 2.0;
      }

      for (size_t j = i + 1; j < norbs; ++j) {
        double occ_alpha_j = alpha_occupations(j);
        double occ_beta_j = beta_occupations(j);
        double occ_sum_j = occ_alpha_j + occ_beta_j;
        if (occ_sum_j > 0.0 || occ_sum_i > 0.0) {
          size_t index_iijj = i * norb3 + i * norb2 + j * norbs + j;
          size_t index_jjii = j * norb3 + j * norb2 + i * norbs + i;
          tmp_two_rdm(index_iijj) = occ_sum_i * occ_sum_j;
          tmp_two_rdm(index_jjii) = occ_sum_i * occ_sum_j;

          size_t index_ijji = i * norb3 + j * norb2 + j * norbs + i;
          size_t index_jiij = j * norb3 + i * norb2 + i * norbs + j;
          tmp_two_rdm(index_ijji) =
              -(occ_alpha_i * occ_alpha_j + occ_beta_i * occ_beta_j);
          tmp_two_rdm(index_jiij) =
              -(occ_alpha_i * occ_alpha_j + occ_beta_i * occ_beta_j);
        }
      }
    }
    _two_rdm_spin_traced =
        std::make_shared<VectorVariant>(std::move(tmp_two_rdm));
    return *_two_rdm_spin_traced;
  }
  return WavefunctionContainer::get_active_two_rdm_spin_traced();
}

Eigen::VectorXd StateVectorContainer::get_single_orbital_entropies() const {
  QDK_LOG_TRACE_ENTERING();
  if (_is_single_determinant() && !_entropies.single_orbital) {
    // For a single Slater determinant with no provided entropies, all orbitals
    // are either fully occupied or unoccupied, giving zero entropy each.
    if (get_orbitals()->get_active_space_indices().first.size() !=
        get_orbitals()->get_active_space_indices().second.size()) {
      throw std::runtime_error(
          "Single orbital entropies not implemented for different alpha and "
          "beta active space sizes");
    }
    size_t num_active_orbitals =
        get_orbitals()->get_active_space_indices().first.size();
    return Eigen::VectorXd::Zero(num_active_orbitals);
  }
  return WavefunctionContainer::get_single_orbital_entropies();
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
StateVectorContainer::active_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  auto [alpha, beta] = _active_occupations_pair();
  return _make_orbital_occupations(alpha, beta);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
StateVectorContainer::_active_occupations_pair() const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  auto [alpha_active_indices, beta_active_indices] =
      get_orbitals()->get_active_space_indices();

  if (alpha_active_indices.empty()) {
    if (_is_single_determinant()) {
      // No active space partition: all orbitals are active.
      // Compute directly from the determinant to avoid mutual recursion
      // with _total_occupations_pair().
      const size_t num_orbitals = get_orbitals()->get_num_molecular_orbitals();
      Eigen::VectorXd alpha_occ = Eigen::VectorXd::Zero(num_orbitals);
      Eigen::VectorXd beta_occ = Eigen::VectorXd::Zero(num_orbitals);
      const auto& det = determinants[0];
      for (size_t i = 0; i < num_orbitals && i < det.capacity(); ++i) {
        if (det.bits_per_mode() == 2) {
          if (det.has_alpha_electron(i)) alpha_occ(i) = 1.0;
          if (det.has_beta_electron(i)) beta_occ(i) = 1.0;
        } else {
          alpha_occ(i) = det.get_mode_state(i) ? 1.0 : 0.0;
        }
      }
      return {alpha_occ, beta_occ};
    }
    return {Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0)};
  }

  const size_t num_active_orbitals = alpha_active_indices.size();

  if (_is_single_determinant()) {
    // Read occupations directly from the single determinant.
    Eigen::VectorXd alpha_occupations =
        Eigen::VectorXd::Zero(num_active_orbitals);
    Eigen::VectorXd beta_occupations =
        Eigen::VectorXd::Zero(num_active_orbitals);

    const auto& det = determinants[0];
    if (det.bits_per_mode() == 2) {
      for (size_t active_idx = 0;
           active_idx < num_active_orbitals && active_idx < det.capacity();
           ++active_idx) {
        if (det.has_alpha_electron(active_idx))
          alpha_occupations(active_idx) = 1.0;
        if (det.has_beta_electron(active_idx))
          beta_occupations(active_idx) = 1.0;
      }
    } else {
      for (size_t active_idx = 0;
           active_idx < num_active_orbitals && active_idx < det.capacity();
           ++active_idx) {
        alpha_occupations(active_idx) =
            det.get_mode_state(active_idx) ? 1.0 : 0.0;
      }
    }
    return {alpha_occupations, beta_occupations};
  }

  // Multi-determinant: occupations come from diagonalizing the 1-RDM.
  Eigen::VectorXd alpha_occupations =
      Eigen::VectorXd::Zero(num_active_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_active_orbitals);

  if (!has_one_rdm_spin_dependent()) {
    throw std::runtime_error(
        "1RDM must be available to compute orbital occupations");
  }

  const auto& rdm = active_one_rdm();
  if (std::holds_alternative<SymmetryBlockedTensor<2, std::complex<double>>>(
          rdm)) {
    throw std::runtime_error(
        "Complex 1RDM diagonalization not yet implemented");
  }

  const auto& rdm_sbt = std::get<SymmetryBlockedTensor<2, double>>(rdm);
  const Eigen::MatrixXd& alpha_rdm =
      rdm_sbt.block({axes::alpha(), axes::alpha()});
  const Eigen::MatrixXd& beta_rdm = rdm_sbt.block({axes::beta(), axes::beta()});

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> alpha_solver(alpha_rdm);
  if (alpha_solver.info() != Eigen::Success) {
    throw std::runtime_error("Failed to diagonalize alpha 1RDM");
  }
  Eigen::VectorXd alpha_eigenvalues = alpha_solver.eigenvalues();
  std::reverse(alpha_eigenvalues.data(),
               alpha_eigenvalues.data() + alpha_eigenvalues.size());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> beta_solver(beta_rdm);
  if (beta_solver.info() != Eigen::Success) {
    throw std::runtime_error("Failed to diagonalize beta 1RDM");
  }
  Eigen::VectorXd beta_eigenvalues = beta_solver.eigenvalues();
  std::reverse(beta_eigenvalues.data(),
               beta_eigenvalues.data() + beta_eigenvalues.size());

  for (int active_idx = 0;
       active_idx < std::min(static_cast<int>(num_active_orbitals),
                             static_cast<int>(alpha_eigenvalues.size()));
       ++active_idx) {
    alpha_occupations(active_idx) = alpha_eigenvalues(active_idx);
  }
  for (int active_idx = 0;
       active_idx < std::min(static_cast<int>(num_active_orbitals),
                             static_cast<int>(beta_eigenvalues.size()));
       ++active_idx) {
    beta_occupations(active_idx) = beta_eigenvalues(active_idx);
  }

  return {alpha_occupations, beta_occupations};
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
StateVectorContainer::total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  auto [alpha, beta] = _total_occupations_pair();
  return _make_orbital_occupations(alpha, beta);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
StateVectorContainer::_total_occupations_pair() const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  const int num_orbitals =
      static_cast<int>(get_orbitals()->get_num_molecular_orbitals());

  Eigen::VectorXd alpha_occupations = Eigen::VectorXd::Zero(num_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_orbitals);

  // Inactive orbitals are doubly occupied (for spin-½) or singly occupied
  // (for spinless). For spinless bases v1_indices_from_index_set duplicates
  // trivial-label indices into both alpha and beta channels; filling both
  // would double-count after _make_orbital_occupations sums them.
  auto sym = get_orbitals()->symmetries();
  bool has_spin = sym && sym->has_axis(AxisName::Spin);
  auto [alpha_inactive_indices, beta_inactive_indices] =
      get_orbitals()->get_inactive_space_indices();
  for (size_t inactive_idx : alpha_inactive_indices) {
    if (inactive_idx < static_cast<size_t>(num_orbitals)) {
      alpha_occupations(inactive_idx) = 1.0;
    }
  }
  if (has_spin) {
    for (size_t inactive_idx : beta_inactive_indices) {
      if (inactive_idx < static_cast<size_t>(num_orbitals)) {
        beta_occupations(inactive_idx) = 1.0;
      }
    }
  }

  if (!_is_single_determinant() && !has_one_rdm_spin_dependent()) {
    throw std::runtime_error(
        "1RDM must be available to compute orbital occupations");
  }

  auto [alpha_active_occs, beta_active_occs] = _active_occupations_pair();
  auto [alpha_active_indices, beta_active_indices] =
      get_orbitals()->get_active_space_indices();

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

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

nlohmann::json StateVectorContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  nlohmann::json j;
  j["version"] = SERIALIZATION_VERSION;
  j["container_type"] = get_container_type();
  j["wavefunction_type"] =
      (_type == WavefunctionType::SelfDual) ? "self_dual" : "not_self_dual";

  bool is_complex = detail::is_vector_variant_complex(_coefficients);
  j["is_complex"] = is_complex;
  if (is_complex) {
    const auto& coeffs_complex = std::get<Eigen::VectorXcd>(_coefficients);
    nlohmann::json coeffs_array = nlohmann::json::array();
    for (int i = 0; i < coeffs_complex.size(); ++i) {
      coeffs_array.push_back(
          {coeffs_complex(i).real(), coeffs_complex(i).imag()});
    }
    j["coefficients"] = coeffs_array;
  } else {
    const auto& coeffs_real = std::get<Eigen::VectorXd>(_coefficients);
    j["coefficients"] = std::vector<double>(
        coeffs_real.data(), coeffs_real.data() + coeffs_real.size());
  }

  j["configuration_set"] = _configuration_set.to_json();

  {
    bool has_any_rdm = _one_rdm_spin_traced != nullptr ||
                       _two_rdm_spin_traced != nullptr ||
                       _active_one_rdm != nullptr || _active_two_rdm != nullptr;
    if (has_any_rdm) {
      nlohmann::json rdm_json;
      if (_one_rdm_spin_traced != nullptr) {
        bool rdm_is_complex =
            detail::is_matrix_variant_complex(*_one_rdm_spin_traced);
        rdm_json["is_one_rdm_spin_traced_complex"] = rdm_is_complex;
        rdm_json["one_rdm_spin_traced"] =
            matrix_variant_to_json(*_one_rdm_spin_traced, rdm_is_complex);
      }
      if (_two_rdm_spin_traced != nullptr) {
        bool rdm_is_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_traced);
        rdm_json["is_two_rdm_spin_traced_complex"] = rdm_is_complex;
        rdm_json["two_rdm_spin_traced"] =
            vector_variant_to_json(*_two_rdm_spin_traced, rdm_is_complex);
      }
      if (_active_one_rdm != nullptr) {
        rdm_json["active_one_rdm"] = std::visit(
            [](const auto& t) { return t.to_json(); }, *_active_one_rdm);
      }
      if (_active_two_rdm != nullptr) {
        rdm_json["active_two_rdm"] = std::visit(
            [](const auto& t) { return t.to_json(); }, *_active_two_rdm);
      }
      j["rdms"] = std::move(rdm_json);
    }
  }

  _serialize_entropies_to_json(j);

  return j;
}

std::unique_ptr<WavefunctionContainer> StateVectorContainer::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  try {
    if (!j.contains("container_type")) {
      throw std::runtime_error("JSON missing required 'container_type' field");
    }

    // Only the current "state_vector" schema is accepted. Files written by an
    // older release must be migrated with python -m qdk_chemistry.migrate.
    return WavefunctionContainer::from_json(j);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to parse StateVectorContainer from JSON: " +
        std::string(e.what()));
  }
}

std::unique_ptr<WavefunctionContainer> StateVectorContainer::from_hdf5(
    H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();

  try {
    if (!group.attrExists("container_type")) {
      throw std::runtime_error("HDF5 group missing 'container_type' attribute");
    }

    // Only the current "state_vector" schema is accepted. Files written by an
    // older release must be migrated with python -m qdk_chemistry.migrate.
    return WavefunctionContainer::from_hdf5(group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void StateVectorContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  WavefunctionContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, _coefficients);
  hash_value(ctx, _configuration_set.content_hash());
}

}  // namespace qdk::chemistry::data
