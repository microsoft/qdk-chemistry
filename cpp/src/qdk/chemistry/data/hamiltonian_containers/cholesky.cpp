// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <macis/util/fcidump.hpp>
#include <memory>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

#include "../filename_utils.hpp"
#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

namespace {

template <class Derived>
double max_abs(const Eigen::MatrixBase<Derived>& matrix) {
  return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

template <class LhsDerived, class RhsDerived>
void require_close(const Eigen::MatrixBase<LhsDerived>& lhs,
                   const Eigen::MatrixBase<RhsDerived>& rhs, double tolerance,
                   const std::string& description) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    throw std::invalid_argument(description + " dimensions do not match");
  }
  if (!lhs.allFinite() || !rhs.allFinite()) {
    throw std::invalid_argument(description + " contains non-finite values");
  }
  const double error = max_abs(lhs - rhs);
  if (error > tolerance) {
    std::ostringstream message;
    message << description << " differs by " << error
            << ", exceeding tolerance " << tolerance;
    throw std::invalid_argument(message.str());
  }
}

SymmetryBlockedTensor<2> make_restricted_rank2(Eigen::MatrixXd block) {
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/true)}));
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents[axes::alpha()] = static_cast<std::size_t>(block.rows());
  extents[axes::beta()] = static_cast<std::size_t>(block.rows());
  auto shared_block = std::make_shared<const Eigen::MatrixXd>(std::move(block));
  SymmetryBlockedTensor<2>::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha()}] = shared_block;
  blocks[{axes::beta(), axes::beta()}] = shared_block;
  return SymmetryBlockedTensor<2>(
      SymmetryBlockedTensor<2>::SymmetriesArray{symmetry, symmetry},
      SymmetryBlockedTensor<2>::ExtentsArray{extents, extents},
      std::move(blocks));
}

std::shared_ptr<const SymmetryBlockedTensor<2>> make_restricted_rank2_ptr(
    Eigen::MatrixXd block) {
  return std::make_shared<const SymmetryBlockedTensor<2>>(
      make_restricted_rank2(std::move(block)));
}

SymmetryBlockedTensor<3> make_restricted_rank3(
    const Orbitals& target_orbitals, const SymmetryBlockedTensor<3>& source,
    Eigen::MatrixXd block) {
  const auto orbital_symmetry = target_orbitals.symmetries();
  const auto auxiliary_symmetry = source.symmetries()[2];
  std::unordered_map<SymmetryLabel, std::size_t> orbital_extents;
  const auto active_indices = target_orbitals.active_indices();
  orbital_extents[axes::alpha()] =
      spin_channel_indices(active_indices, axes::alpha()).size();
  orbital_extents[axes::beta()] =
      spin_channel_indices(active_indices, axes::beta()).size();
  auto shared_block = std::make_shared<const Eigen::MatrixXd>(std::move(block));
  SymmetryBlockedTensor<3>::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha(), SymmetryLabel{}}] = shared_block;
  blocks[{axes::beta(), axes::beta(), SymmetryLabel{}}] = shared_block;
  return SymmetryBlockedTensor<3>(
      SymmetryBlockedTensor<3>::SymmetriesArray{
          orbital_symmetry, orbital_symmetry, auxiliary_symmetry},
      SymmetryBlockedTensor<3>::ExtentsArray{orbital_extents, orbital_extents,
                                             source.extents()[2]},
      std::move(blocks));
}

}  // namespace

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

  size_t norb =
      spin_channel_indices(_orbitals->active_indices(), axes::alpha()).size();
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

std::unique_ptr<CholeskyHamiltonianContainer>
CholeskyHamiltonianContainer::transform_active_orbital_basis(
    std::shared_ptr<Orbitals> target_orbitals,
    double validation_tolerance) const {
  QDK_LOG_TRACE_ENTERING();
  if (!target_orbitals) {
    throw std::invalid_argument("Target orbitals pointer cannot be nullptr");
  }
  if (!std::isfinite(validation_tolerance) || validation_tolerance < 0.0) {
    throw std::invalid_argument(
        "Validation tolerance must be finite and non-negative");
  }
  if (!is_restricted() || !_orbitals->is_restricted() ||
      !target_orbitals->is_restricted()) {
    throw std::runtime_error(
        "Active-orbital basis transformation currently supports only "
        "restricted Cholesky Hamiltonians and orbitals");
  }
  if (!_orbitals->has_overlap_matrix() ||
      !target_orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "Source and target orbitals must contain AO overlap matrices");
  }
  if (_orbitals->get_num_atomic_orbitals() !=
          target_orbitals->get_num_atomic_orbitals() ||
      _orbitals->get_num_molecular_orbitals() !=
          target_orbitals->get_num_molecular_orbitals()) {
    throw std::invalid_argument(
        "Source and target orbital dimensions do not match");
  }
  if (_orbitals->has_basis_set() != target_orbitals->has_basis_set()) {
    throw std::invalid_argument(
        "Source and target orbitals must have matching basis metadata");
  }
  if (_orbitals->has_basis_set() &&
      _orbitals->get_basis_set()->content_hash() !=
          target_orbitals->get_basis_set()->content_hash()) {
    throw std::invalid_argument(
        "Source and target orbitals use different AO basis sets");
  }
  require_close(_orbitals->get_overlap_matrix(),
                target_orbitals->get_overlap_matrix(), validation_tolerance,
                "Source and target AO overlap matrices");

  const auto expected_symmetry =
      SymmetryProduct({axes::spin(1, /*equivalent=*/true)});
  const auto source_symmetry = _orbitals->symmetries();
  const auto target_symmetry = target_orbitals->symmetries();
  if (!source_symmetry || !target_symmetry ||
      *source_symmetry != expected_symmetry ||
      *target_symmetry != expected_symmetry ||
      _orbitals->mo_extents() != target_orbitals->mo_extents()) {
    throw std::invalid_argument(
        "Source and target orbitals must have matching restricted spin-only "
        "MO symmetry metadata");
  }

  const auto validate_restricted_rank2 = [&](const auto& tensor,
                                             const std::string& description) {
    const SymmetryBlockedTensor<2>::Labels alpha_labels{axes::alpha(),
                                                        axes::alpha()};
    const SymmetryBlockedTensor<2>::Labels beta_labels{axes::beta(),
                                                       axes::beta()};
    if (*tensor.symmetries()[0] != expected_symmetry ||
        *tensor.symmetries()[1] != expected_symmetry ||
        !tensor.has_block(alpha_labels) || !tensor.has_block(beta_labels) ||
        tensor.block_ptr(alpha_labels).get() !=
            tensor.block_ptr(beta_labels).get()) {
      throw std::runtime_error(
          description +
          " must use explicit shared restricted-spin block storage");
    }
  };
  validate_restricted_rank2(*_one_body, "Source one-body integrals");
  if (_inactive_fock) {
    validate_restricted_rank2(*_inactive_fock, "Source inactive Fock matrix");
  }

  if (*_three_center->symmetries()[0] != expected_symmetry ||
      *_three_center->symmetries()[1] != expected_symmetry ||
      *_three_center->symmetries()[2] != SymmetryProduct::trivial() ||
      !_three_center->has_block(
          {axes::alpha(), axes::alpha(), SymmetryLabel{}})) {
    throw std::runtime_error(
        "Source three-center factors must use spin-diagonal MO slots and a "
        "trivial auxiliary slot");
  }

  const auto source_coefficient_tensor = _orbitals->coefficients();
  const auto target_coefficient_tensor = target_orbitals->coefficients();
  for (std::size_t slot = 0; slot < 2; ++slot) {
    if (*source_coefficient_tensor->symmetries()[slot] !=
            *target_coefficient_tensor->symmetries()[slot] ||
        source_coefficient_tensor->extents()[slot] !=
            target_coefficient_tensor->extents()[slot]) {
      throw std::invalid_argument(
          "Source and target orbital coefficient symmetry metadata do not "
          "match");
    }
  }

  const auto source_active = _orbitals->active_indices();
  const auto target_active = target_orbitals->active_indices();
  const auto source_inactive = _orbitals->inactive_indices();
  const auto target_inactive = target_orbitals->inactive_indices();
  if (!source_active || !target_active) {
    throw std::invalid_argument(
        "Source and target orbitals must define active spaces");
  }
  if (static_cast<bool>(source_inactive) !=
      static_cast<bool>(target_inactive)) {
    throw std::invalid_argument(
        "Source and target inactive-space metadata presence must match");
  }

  const auto require_matching_index_metadata =
      [&](const std::shared_ptr<const SymmetryBlockedIndexSet>& source,
          const std::shared_ptr<const SymmetryBlockedIndexSet>& target,
          const std::string& description) {
        if (!source || !target || *source->symmetries() != expected_symmetry ||
            *target->symmetries() != expected_symmetry ||
            source->extents() != target->extents() ||
            source->extents() != _orbitals->mo_extents() ||
            target->extents() != target_orbitals->mo_extents()) {
          throw std::invalid_argument(
              "Source and target " + description +
              " must have matching restricted spin-only symmetry metadata");
        }
      };
  require_matching_index_metadata(source_active, target_active,
                                  "active-space index sets");
  if (source_inactive) {
    require_matching_index_metadata(source_inactive, target_inactive,
                                    "inactive-space index sets");
  }

  const auto source_active_alpha =
      spin_channel_indices(source_active, axes::alpha());
  const auto source_active_beta =
      spin_channel_indices(source_active, axes::beta());
  const auto target_active_alpha =
      spin_channel_indices(target_active, axes::alpha());
  const auto target_active_beta =
      spin_channel_indices(target_active, axes::beta());
  if (source_active_alpha.empty()) {
    throw std::invalid_argument("The active orbital space cannot be empty");
  }
  if (source_active_alpha != source_active_beta ||
      source_active_alpha != target_active_alpha ||
      source_active_alpha != target_active_beta) {
    throw std::invalid_argument(
        "Source and target active-space indices must match for both spins");
  }

  const auto source_inactive_alpha =
      source_inactive ? spin_channel_indices(source_inactive, axes::alpha())
                      : std::vector<size_t>{};
  const auto source_inactive_beta =
      source_inactive ? spin_channel_indices(source_inactive, axes::beta())
                      : std::vector<size_t>{};
  const auto target_inactive_alpha =
      target_inactive ? spin_channel_indices(target_inactive, axes::alpha())
                      : std::vector<size_t>{};
  const auto target_inactive_beta =
      target_inactive ? spin_channel_indices(target_inactive, axes::beta())
                      : std::vector<size_t>{};
  if (source_inactive_alpha != source_inactive_beta ||
      source_inactive_alpha != target_inactive_alpha ||
      source_inactive_alpha != target_inactive_beta) {
    throw std::invalid_argument(
        "Source and target inactive-space indices must match for both spins");
  }

  const auto& source_coefficients =
      _orbitals->coefficients()->block({axes::alpha(), axes::alpha()});
  const auto& target_coefficients =
      target_orbitals->coefficients()->block({axes::alpha(), axes::alpha()});
  if (source_coefficients.rows() != target_coefficients.rows() ||
      source_coefficients.cols() != target_coefficients.cols()) {
    throw std::invalid_argument(
        "Source and target coefficient dimensions do not match");
  }
  if (!source_coefficients.allFinite() || !target_coefficients.allFinite()) {
    throw std::invalid_argument(
        "Source and target orbital coefficients must be finite");
  }

  std::vector<bool> is_active(source_coefficients.cols(), false);
  for (const size_t index : source_active_alpha) {
    if (index >= is_active.size()) {
      throw std::invalid_argument("Active orbital index is out of range");
    }
    is_active[index] = true;
  }
  for (Eigen::Index index = 0; index < source_coefficients.cols(); ++index) {
    if (!is_active[index]) {
      require_close(source_coefficients.col(index),
                    target_coefficients.col(index), validation_tolerance,
                    "Orbital coefficients outside the active space");
    }
  }

  const Eigen::Index num_active =
      static_cast<Eigen::Index>(source_active_alpha.size());
  Eigen::MatrixXd source_active_coefficients(source_coefficients.rows(),
                                             num_active);
  Eigen::MatrixXd target_active_coefficients(target_coefficients.rows(),
                                             num_active);
  for (Eigen::Index column = 0; column < num_active; ++column) {
    const size_t index = source_active_alpha[column];
    source_active_coefficients.col(column) = source_coefficients.col(index);
    target_active_coefficients.col(column) = target_coefficients.col(index);
  }

  const auto& overlap = _orbitals->get_overlap_matrix();
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(num_active, num_active);
  require_close(source_active_coefficients.transpose() * overlap *
                    source_active_coefficients,
                identity, validation_tolerance,
                "Source active orbital overlap");
  require_close(target_active_coefficients.transpose() * overlap *
                    target_active_coefficients,
                identity, validation_tolerance,
                "Target active orbital overlap");

  const Eigen::MatrixXd rotation = source_active_coefficients.transpose() *
                                   overlap * target_active_coefficients;
  require_close(rotation.transpose() * rotation, identity, validation_tolerance,
                "Recovered active-space rotation");
  require_close(source_active_coefficients * rotation,
                target_active_coefficients, validation_tolerance,
                "Target active orbital reconstruction");

  const auto& source_one_body =
      _one_body->block({axes::alpha(), axes::alpha()});
  Eigen::MatrixXd transformed_one_body =
      rotation.transpose() * source_one_body * rotation;

  std::shared_ptr<const SymmetryBlockedTensor<2>> transformed_inactive_fock;
  if (_inactive_fock) {
    const auto& source_fock =
        _inactive_fock->block({axes::alpha(), axes::alpha()});
    Eigen::MatrixXd full_rotation =
        Eigen::MatrixXd::Identity(source_fock.rows(), source_fock.cols());
    for (Eigen::Index row = 0; row < num_active; ++row) {
      for (Eigen::Index column = 0; column < num_active; ++column) {
        full_rotation(source_active_alpha[row], source_active_alpha[column]) =
            rotation(row, column);
      }
    }
    Eigen::MatrixXd transformed_fock =
        full_rotation.transpose() * source_fock * full_rotation;
    transformed_inactive_fock =
        make_restricted_rank2_ptr(std::move(transformed_fock));
  }

  const auto& source_three_center =
      _three_center->block({axes::alpha(), axes::alpha(), SymmetryLabel{}});
  Eigen::MatrixXd transformed_three_center(source_three_center.rows(),
                                           source_three_center.cols());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (Eigen::Index factor = 0; factor < source_three_center.cols(); ++factor) {
    Eigen::Map<const Eigen::MatrixXd> source_matrix(
        source_three_center.col(factor).data(), num_active, num_active);
    Eigen::Map<Eigen::MatrixXd> transformed_matrix(
        transformed_three_center.col(factor).data(), num_active, num_active);
    transformed_matrix.noalias() =
        rotation.transpose() * source_matrix * rotation;
  }

  auto output_one_body = make_restricted_rank2(std::move(transformed_one_body));
  auto output_three_center = make_restricted_rank3(
      *target_orbitals, *_three_center, std::move(transformed_three_center));
  return std::make_unique<CholeskyHamiltonianContainer>(
      std::move(output_one_body), std::move(output_three_center),
      std::move(target_orbitals), _core_energy,
      std::move(transformed_inactive_fock), _ao_cholesky_vectors, _type);
}

double CholeskyHamiltonianContainer::get_two_body_element(
    unsigned i, unsigned j, unsigned k, unsigned l, SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();

  if (!has_two_body_integrals()) {
    throw std::runtime_error("Two-body integrals are not set");
  }

  size_t norb =
      spin_channel_indices(_orbitals->active_indices(), axes::alpha()).size();
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
  auto norb_beta = _one_body->block({axes::beta(), axes::beta()}).rows();
  const auto& row_extents = _three_center->extents()[0];
  const auto& column_extents = _three_center->extents()[1];
  const auto extent_matches = [](const auto& extents,
                                 const SymmetryLabel& label, size_t expected) {
    const auto iterator = extents.find(label);
    return iterator != extents.end() && iterator->second == expected;
  };
  if (!extent_matches(row_extents, axes::alpha(), norb_alpha) ||
      !extent_matches(row_extents, axes::beta(), norb_beta) ||
      !extent_matches(column_extents, axes::alpha(), norb_alpha) ||
      !extent_matches(column_extents, axes::beta(), norb_beta)) {
    throw std::invalid_argument(
        "Three-center MO-slot extents do not match the active one-body "
        "dimensions");
  }
  auto naux = _three_center->extents()[2].at(SymmetryLabel{});
  auto expected_rows = static_cast<size_t>(norb_alpha * norb_alpha);

  const auto& aa =
      _three_center->block({axes::alpha(), axes::alpha(), SymmetryLabel{}});
  if (static_cast<size_t>(aa.rows()) != expected_rows ||
      static_cast<size_t>(aa.cols()) != naux) {
    throw std::invalid_argument(
        "Alpha-alpha three-center integrals shape (" +
        std::to_string(aa.rows()) + ", " + std::to_string(aa.cols()) +
        ") does not match expected (norb^2, naux) = (" +
        std::to_string(expected_rows) + ", " + std::to_string(naux) + ")");
  }

  if (!_three_center->all_aliased(
          {{{axes::alpha(), axes::alpha(), SymmetryLabel{}},
            {axes::beta(), axes::beta(), SymmetryLabel{}}}})) {
    const auto& bb =
        _three_center->block({axes::beta(), axes::beta(), SymmetryLabel{}});
    if (static_cast<size_t>(bb.rows()) != expected_rows ||
        static_cast<size_t>(bb.cols()) != naux) {
      throw std::invalid_argument(
          "Beta-beta three-center integrals shape does not match expected "
          "(norb^2, naux)");
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
  const auto active_ai = orbitals.active_indices();
  std::size_t n_active_alpha =
      spin_channel_indices(active_ai, axes::alpha()).size();
  std::size_t n_active_beta =
      spin_channel_indices(active_ai, axes::beta()).size();
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

void CholeskyHamiltonianContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  HamiltonianContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  if (_three_center) {
    hash_field_presence(ctx, true);
    hash_value(ctx, _three_center->content_hash());
  } else {
    hash_field_presence(ctx, false);
  }
  hash_value(ctx, _ao_cholesky_vectors);
}

}  // namespace qdk::chemistry::data
