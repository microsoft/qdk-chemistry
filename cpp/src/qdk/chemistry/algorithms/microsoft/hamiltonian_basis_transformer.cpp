// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "hamiltonian_basis_transformer.hpp"

#include <cmath>
#include <optional>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>

namespace qdk::chemistry::algorithms::microsoft {
namespace {

using data::CholeskyHamiltonianContainer;
using data::Orbitals;
using data::SymmetryBlockedIndexSet;
using data::SymmetryBlockedTensor;
using data::SymmetryLabel;
using data::SymmetryProduct;
using data::axes::alpha;
using data::axes::beta;

class HamiltonianBasisTransformerSettings : public data::Settings {
 public:
  HamiltonianBasisTransformerSettings() {
    set_default("validation_tolerance", 1.0e-10,
                "Absolute tolerance for validating the orbital basis change",
                data::BoundConstraint<double>{0.0, 1.0});
  }
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

template <class Lhs, class Rhs>
void require_close(const Eigen::MatrixBase<Lhs>& lhs,
                   const Eigen::MatrixBase<Rhs>& rhs, double tolerance,
                   const std::string& description) {
  require(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols() &&
              lhs.allFinite() && rhs.allFinite(),
          description + " is incompatible");
  const double error =
      lhs.size() == 0 ? 0.0 : (lhs - rhs).cwiseAbs().maxCoeff();
  if (error > tolerance) {
    std::ostringstream message;
    message << description << " differs by " << error
            << ", exceeding tolerance " << tolerance;
    throw std::invalid_argument(message.str());
  }
}

void require_restricted(const SymmetryBlockedTensor<2>& tensor,
                        const SymmetryProduct& symmetry,
                        const std::string& description) {
  const SymmetryBlockedTensor<2>::Labels alpha_block{alpha(), alpha()};
  const SymmetryBlockedTensor<2>::Labels beta_block{beta(), beta()};
  require(
      *tensor.symmetries()[0] == symmetry &&
          *tensor.symmetries()[1] == symmetry &&
          tensor.has_block(alpha_block) && tensor.has_block(beta_block) &&
          tensor.block_ptr(alpha_block) == tensor.block_ptr(beta_block),
      description + " must use explicit shared restricted-spin block storage");
}

bool matching_metadata(const SymmetryBlockedTensor<2>& lhs,
                       const SymmetryBlockedTensor<2>& rhs) {
  for (std::size_t slot = 0; slot < 2; ++slot) {
    if (*lhs.symmetries()[slot] != *rhs.symmetries()[slot] ||
        lhs.extents()[slot] != rhs.extents()[slot]) {
      return false;
    }
  }
  return true;
}

std::vector<std::size_t> restricted_indices(
    const std::shared_ptr<const SymmetryBlockedIndexSet>& indices,
    const Orbitals& orbitals, const SymmetryProduct& symmetry,
    const std::string& description) {
  require(indices && *indices->symmetries() == symmetry &&
              indices->extents() == orbitals.mo_extents(),
          description + " must use restricted spin-only symmetry metadata");
  auto alpha_indices = data::spin_channel_indices(indices, alpha());
  require(alpha_indices == data::spin_channel_indices(indices, beta()),
          description + " must contain the same indices for both spins");
  return alpha_indices;
}

SymmetryBlockedTensor<2> restricted_rank2(Eigen::MatrixXd block) {
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({data::axes::spin(1, true)}));
  std::unordered_map<SymmetryLabel, std::size_t> extents{
      {alpha(), static_cast<std::size_t>(block.rows())},
      {beta(), static_cast<std::size_t>(block.rows())}};
  auto storage = std::make_shared<const Eigen::MatrixXd>(std::move(block));
  SymmetryBlockedTensor<2>::BlockMap blocks{{{{alpha(), alpha()}}, storage},
                                            {{{beta(), beta()}}, storage}};
  return {{symmetry, symmetry}, {extents, extents}, std::move(blocks)};
}

SymmetryBlockedTensor<3> restricted_rank3(
    const Orbitals& orbitals, const SymmetryBlockedTensor<3>& source,
    Eigen::MatrixXd block) {
  const auto symmetry = orbitals.symmetries();
  const auto active = orbitals.active_indices();
  std::unordered_map<SymmetryLabel, std::size_t> extents{
      {alpha(), data::spin_channel_indices(active, alpha()).size()},
      {beta(), data::spin_channel_indices(active, beta()).size()}};
  auto storage = std::make_shared<const Eigen::MatrixXd>(std::move(block));
  SymmetryBlockedTensor<3>::BlockMap blocks{
      {{{alpha(), alpha(), SymmetryLabel{}}}, storage},
      {{{beta(), beta(), SymmetryLabel{}}}, storage}};
  return {{symmetry, symmetry, source.symmetries()[2]},
          {extents, extents, source.extents()[2]},
          std::move(blocks)};
}

Eigen::MatrixXd active_coefficients(const Orbitals& orbitals,
                                    const std::vector<std::size_t>& indices) {
  const auto& coefficients = orbitals.coefficients()->block({alpha(), alpha()});
  Eigen::MatrixXd active(coefficients.rows(), indices.size());
  for (std::size_t column = 0; column < indices.size(); ++column) {
    active.col(column) = coefficients.col(indices[column]);
  }
  return active;
}

}  // namespace

QdkHamiltonianBasisTransformer::QdkHamiltonianBasisTransformer() {
  _settings = std::make_unique<HamiltonianBasisTransformerSettings>();
}

std::shared_ptr<data::Hamiltonian> QdkHamiltonianBasisTransformer::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<data::Orbitals> target_orbitals) const {
  QDK_LOG_TRACE_ENTERING();
  require(hamiltonian && target_orbitals,
          "Hamiltonian and target orbitals are required");
  require(hamiltonian->has_container_type<CholeskyHamiltonianContainer>(),
          "QDK Hamiltonian basis transformation requires a Cholesky "
          "Hamiltonian");

  const auto& source =
      hamiltonian->get_container<CholeskyHamiltonianContainer>();
  const auto source_orbitals = source.get_orbitals();
  require(source.is_restricted() && source_orbitals->is_restricted() &&
              target_orbitals->is_restricted(),
          "QDK Hamiltonian basis transformation currently requires "
          "restricted orbitals");

  const double tolerance = _settings->get<double>("validation_tolerance");
  require(std::isfinite(tolerance) && tolerance >= 0.0,
          "Validation tolerance must be finite and non-negative");
  require(source_orbitals->has_overlap_matrix() &&
              target_orbitals->has_overlap_matrix(),
          "Source and target AO overlap matrices are required");
  require(
      source_orbitals->has_basis_set() == target_orbitals->has_basis_set() &&
          (!source_orbitals->has_basis_set() ||
           source_orbitals->get_basis_set()->content_hash() ==
               target_orbitals->get_basis_set()->content_hash()),
      "Source and target AO bases do not match");
  require_close(source_orbitals->get_overlap_matrix(),
                target_orbitals->get_overlap_matrix(), tolerance,
                "Source and target AO overlap matrices");

  const SymmetryProduct expected_symmetry({data::axes::spin(1, true)});
  require(*source_orbitals->symmetries() == expected_symmetry &&
              *target_orbitals->symmetries() == expected_symmetry &&
              source_orbitals->mo_extents() == target_orbitals->mo_extents(),
          "Source and target orbitals must use matching restricted spin "
          "symmetry");
  require_restricted(source.one_body_integrals(), expected_symmetry,
                     "Source one-body integrals");
  if (source.has_inactive_fock_matrix()) {
    require_restricted(source.inactive_fock(), expected_symmetry,
                       "Source inactive Fock matrix");
  }
  const auto& source_factors = source.three_center();
  require(*source_factors.symmetries()[0] == expected_symmetry &&
              *source_factors.symmetries()[1] == expected_symmetry &&
              *source_factors.symmetries()[2] == SymmetryProduct::trivial() &&
              source_factors.has_block({alpha(), alpha(), SymmetryLabel{}}),
          "Source three-center factors must use spin-diagonal MO slots and a "
          "trivial auxiliary slot");

  const auto source_coefficient_tensor = source_orbitals->coefficients();
  const auto target_coefficient_tensor = target_orbitals->coefficients();
  require(
      matching_metadata(*source_coefficient_tensor, *target_coefficient_tensor),
      "Source and target coefficient symmetry metadata do not match");

  const auto source_indices =
      restricted_indices(source_orbitals->active_indices(), *source_orbitals,
                         expected_symmetry, "Source active-space index set");
  const auto target_indices =
      restricted_indices(target_orbitals->active_indices(), *target_orbitals,
                         expected_symmetry, "Target active-space index set");
  require(!source_indices.empty() && source_indices == target_indices,
          "Source and target active spaces do not match");

  const auto source_inactive = source_orbitals->inactive_indices();
  const auto target_inactive = target_orbitals->inactive_indices();
  require(
      static_cast<bool>(source_inactive) == static_cast<bool>(target_inactive),
      "Source and target inactive-space metadata presence must match");
  if (source_inactive) {
    require(
        restricted_indices(source_inactive, *source_orbitals, expected_symmetry,
                           "Source inactive-space index set") ==
            restricted_indices(target_inactive, *target_orbitals,
                               expected_symmetry,
                               "Target inactive-space index set"),
        "Source and target inactive spaces do not match");
  }

  const auto& source_coefficients =
      source_orbitals->coefficients()->block({alpha(), alpha()});
  const auto& target_coefficients =
      target_orbitals->coefficients()->block({alpha(), alpha()});
  require(source_coefficients.rows() == target_coefficients.rows() &&
              source_coefficients.cols() == target_coefficients.cols(),
          "Source and target coefficient dimensions do not match");
  require(source_coefficients.allFinite() && target_coefficients.allFinite(),
          "Source and target orbital coefficients must be finite");
  std::vector<bool> is_active(source_coefficients.cols(), false);
  for (const auto index : source_indices) {
    require(index < is_active.size(), "Active orbital index is out of range");
    is_active[index] = true;
  }
  for (Eigen::Index column = 0; column < source_coefficients.cols(); ++column) {
    if (!is_active[column]) {
      require_close(source_coefficients.col(column),
                    target_coefficients.col(column), tolerance,
                    "Orbitals outside the active space");
    }
  }

  const auto source_active_coefficients =
      active_coefficients(*source_orbitals, source_indices);
  const auto target_active_coefficients =
      active_coefficients(*target_orbitals, source_indices);
  const auto& overlap = source_orbitals->get_overlap_matrix();
  const Eigen::Index nactive = source_indices.size();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(nactive, nactive);
  require_close(source_active_coefficients.transpose() * overlap *
                    source_active_coefficients,
                identity, tolerance, "Source active-orbital overlap");
  require_close(target_active_coefficients.transpose() * overlap *
                    target_active_coefficients,
                identity, tolerance, "Target active-orbital overlap");
  const Eigen::MatrixXd rotation = source_active_coefficients.transpose() *
                                   overlap * target_active_coefficients;
  require_close(rotation.transpose() * rotation, identity, tolerance,
                "Recovered active-space rotation");
  require_close(source_active_coefficients * rotation,
                target_active_coefficients, tolerance,
                "Target active orbitals");

  const auto& one_body = std::get<0>(source.get_one_body_integrals());
  Eigen::MatrixXd transformed_one_body =
      rotation.transpose() * one_body * rotation;

  std::shared_ptr<const SymmetryBlockedTensor<2>> transformed_fock;
  if (source.has_inactive_fock_matrix()) {
    const auto& fock = source.get_inactive_fock_matrix().first;
    Eigen::MatrixXd full_rotation =
        Eigen::MatrixXd::Identity(fock.rows(), fock.cols());
    for (Eigen::Index row = 0; row < nactive; ++row) {
      for (Eigen::Index column = 0; column < nactive; ++column) {
        full_rotation(source_indices[row], source_indices[column]) =
            rotation(row, column);
      }
    }
    Eigen::MatrixXd output = full_rotation.transpose() * fock * full_rotation;
    transformed_fock = std::make_shared<const SymmetryBlockedTensor<2>>(
        restricted_rank2(std::move(output)));
  }

  const auto& factors = source.get_three_center_integrals().first;
  require(factors.rows() == nactive * nactive,
          "Cholesky factors must use [nactive^2, naux] storage");
  Eigen::MatrixXd transformed_factors(factors.rows(), factors.cols());
  const auto transform_factor = [&](Eigen::Index factor,
                                    Eigen::MatrixXd& scratch) {
    Eigen::Map<const Eigen::MatrixXd> input(factors.col(factor).data(), nactive,
                                            nactive);
    Eigen::Map<Eigen::MatrixXd> output(transformed_factors.col(factor).data(),
                                       nactive, nactive);
    scratch.noalias() = input * rotation;
    output.noalias() = rotation.transpose() * scratch;
  };
#ifdef _OPENMP
#pragma omp parallel
  {
    Eigen::MatrixXd scratch(nactive, nactive);
#pragma omp for schedule(static)
    for (Eigen::Index factor = 0; factor < factors.cols(); ++factor) {
      transform_factor(factor, scratch);
    }
  }
#else
  Eigen::MatrixXd scratch(nactive, nactive);
  for (Eigen::Index factor = 0; factor < factors.cols(); ++factor) {
    transform_factor(factor, scratch);
  }
#endif

  auto container = std::make_unique<CholeskyHamiltonianContainer>(
      restricted_rank2(std::move(transformed_one_body)),
      restricted_rank3(*target_orbitals, source.three_center(),
                       std::move(transformed_factors)),
      std::move(target_orbitals), source.get_core_energy(),
      std::move(transformed_fock), std::nullopt, source.get_type());
  return std::make_shared<data::Hamiltonian>(std::move(container));
}

}  // namespace qdk::chemistry::algorithms::microsoft
