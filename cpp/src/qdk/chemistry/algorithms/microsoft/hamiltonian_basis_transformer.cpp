// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "hamiltonian_basis_transformer.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
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

constexpr double kMaximumValidationTolerance = 1.0e-2;

class HamiltonianBasisTransformerSettings : public data::Settings {
 public:
  HamiltonianBasisTransformerSettings() {
    set_default(
        "validation_tolerance", 1.0e-10,
        "Tolerance for validating the orbital basis change",
        data::BoundConstraint<double>{0.0, kMaximumValidationTolerance});
  }
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

template <class Matrix>
void require_finite(const Eigen::MatrixBase<Matrix>& matrix,
                    const std::string& description) {
  require(matrix.allFinite(), description + " must be finite");
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

void require_full_column_rank(const Eigen::MatrixXd& matrix,
                              const std::string& description) {
  Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(matrix);
  require(decomposition.info() == Eigen::Success &&
              decomposition.singularValues().allFinite() &&
              decomposition.singularValues().size() == matrix.cols(),
          description + " rank could not be determined");
  const double largest = decomposition.singularValues()(0);
  const double threshold =
      100.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(matrix.rows(), matrix.cols())) *
      std::max(1.0, largest);
  require(decomposition.singularValues().tail(1)(0) > threshold,
          description + " must have full column rank");
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

std::vector<std::size_t> optional_restricted_indices(
    const std::shared_ptr<const SymmetryBlockedIndexSet>& indices,
    const Orbitals& orbitals, const SymmetryProduct& symmetry,
    const std::string& description) {
  if (!indices) return {};
  return restricted_indices(indices, orbitals, symmetry, description);
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
      {{{alpha(), alpha(), SymmetryLabel{}}}, std::move(storage)}};
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

std::shared_ptr<data::Hamiltonian> QdkHamiltonianBasisTransformer::run(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<data::Orbitals> target_orbitals) const {
  const std::scoped_lock lock(_run_mutex);
  return HamiltonianBasisTransformer::run(std::move(hamiltonian),
                                          std::move(target_orbitals));
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
  require_close(source_orbitals->get_overlap_matrix(),
                source_orbitals->get_overlap_matrix().transpose(), tolerance,
                "Source AO overlap matrix symmetry");

  const SymmetryProduct expected_symmetry({data::axes::spin(1, true)});
  require(*source_orbitals->symmetries() == expected_symmetry &&
              *target_orbitals->symmetries() == expected_symmetry &&
              source_orbitals->mo_extents() == target_orbitals->mo_extents(),
          "Source and target orbitals must use matching restricted spin "
          "symmetry");
  require_restricted(source.one_body_integrals(), expected_symmetry,
                     "Source one-body integrals");
  require(std::isfinite(source.get_core_energy()),
          "Source core energy must be finite");
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
  require(optional_restricted_indices(source_inactive, *source_orbitals,
                                      expected_symmetry,
                                      "Source inactive-space index set") ==
              optional_restricted_indices(target_inactive, *target_orbitals,
                                          expected_symmetry,
                                          "Target inactive-space index set"),
          "Source and target inactive spaces do not match");

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
  const Eigen::Index nactive = source_indices.size();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(nactive, nactive);
  Eigen::MatrixXd rotation = [&]() -> Eigen::MatrixXd {
    const auto& stored_overlap = source_orbitals->get_overlap_matrix();
    const Eigen::MatrixXd overlap =
        0.5 * (stored_overlap + stored_overlap.transpose());
    Eigen::MatrixXd source_metric_coefficients;
    Eigen::MatrixXd target_metric_coefficients;
    std::optional<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>
        numerical_null_mode_coefficients;
    const double relative_eigenvalue_tolerance =
        100.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(overlap.rows());
    double null_mode_residual_tolerance = tolerance;
    Eigen::LLT<Eigen::MatrixXd> overlap_cholesky(overlap);
    if (overlap_cholesky.info() == Eigen::Success &&
        overlap_cholesky.rcond() > relative_eigenvalue_tolerance) {
      source_metric_coefficients =
          overlap_cholesky.matrixU() * source_active_coefficients;
      target_metric_coefficients =
          overlap_cholesky.matrixU() * target_active_coefficients;
    } else {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_eigensolver(
          overlap);
      require(overlap_eigensolver.info() == Eigen::Success &&
                  overlap_eigensolver.eigenvalues().allFinite(),
              "AO overlap metric eigendecomposition failed");
      const double spectral_scale =
          overlap_eigensolver.eigenvalues().cwiseAbs().maxCoeff();
      require(spectral_scale > 0.0,
              "AO overlap metric must have positive spectral scale");
      const double negative_eigenvalue_tolerance =
          relative_eigenvalue_tolerance * spectral_scale;
      require(overlap_eigensolver.eigenvalues().minCoeff() >=
                  -negative_eigenvalue_tolerance,
              "AO overlap metric must be positive semidefinite");
      const Eigen::MatrixXd source_eigen_coefficients =
          overlap_eigensolver.eigenvectors().transpose() *
          source_active_coefficients;
      const Eigen::MatrixXd target_eigen_coefficients =
          overlap_eigensolver.eigenvectors().transpose() *
          target_active_coefficients;
      const Eigen::VectorXd square_root_eigenvalues =
          overlap_eigensolver.eigenvalues().unaryExpr(
              [negative_eigenvalue_tolerance](double eigenvalue) {
                return eigenvalue > negative_eigenvalue_tolerance
                           ? std::sqrt(eigenvalue)
                           : 0.0;
              });
      source_metric_coefficients =
          square_root_eigenvalues.asDiagonal() * source_eigen_coefficients;
      target_metric_coefficients =
          square_root_eigenvalues.asDiagonal() * target_eigen_coefficients;
      const Eigen::VectorXd numerical_null_mode_scale =
          overlap_eigensolver.eigenvalues().unaryExpr(
              [negative_eigenvalue_tolerance](double eigenvalue) {
                return std::abs(eigenvalue) <= negative_eigenvalue_tolerance
                           ? std::sqrt(std::abs(eigenvalue))
                           : 0.0;
              });
      numerical_null_mode_coefficients.emplace(
          numerical_null_mode_scale.asDiagonal() * source_eigen_coefficients,
          numerical_null_mode_scale.asDiagonal() * target_eigen_coefficients);
      null_mode_residual_tolerance =
          std::max(tolerance, std::sqrt(relative_eigenvalue_tolerance));
    }
    require_full_column_rank(source_metric_coefficients,
                             "Source active orbitals");
    require_full_column_rank(target_metric_coefficients,
                             "Target active orbitals");
    require_close(
        source_metric_coefficients.transpose() * source_metric_coefficients,
        identity, tolerance, "Source active-orbital overlap");
    require_close(
        target_metric_coefficients.transpose() * target_metric_coefficients,
        identity, tolerance, "Target active-orbital overlap");
    if (numerical_null_mode_coefficients) {
      const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(nactive, nactive);
      require_close(numerical_null_mode_coefficients->first.transpose() *
                        numerical_null_mode_coefficients->first,
                    zero, tolerance,
                    "Source active-orbital numerical-null contribution");
      require_close(numerical_null_mode_coefficients->second.transpose() *
                        numerical_null_mode_coefficients->second,
                    zero, tolerance,
                    "Target active-orbital numerical-null contribution");
    }
    Eigen::MatrixXd recovered_rotation =
        source_metric_coefficients.transpose() * target_metric_coefficients;
    require_full_column_rank(recovered_rotation,
                             "Recovered active-space rotation");
    require_close(source_metric_coefficients * recovered_rotation,
                  target_metric_coefficients, tolerance,
                  "Target active orbitals");
    if (numerical_null_mode_coefficients) {
      require_close(
          numerical_null_mode_coefficients->first * recovered_rotation,
          numerical_null_mode_coefficients->second,
          null_mode_residual_tolerance,
          "Target active orbitals in numerical null modes");
    }
    require_close(recovered_rotation.transpose() * recovered_rotation, identity,
                  tolerance, "Recovered active-space rotation");
    return recovered_rotation;
  }();

  const auto& one_body = std::get<0>(source.get_one_body_integrals());
  require_finite(one_body, "Source one-body integrals");
  Eigen::MatrixXd transformed_one_body =
      rotation.transpose() * one_body * rotation;
  require_finite(transformed_one_body, "Transformed one-body integrals");

  std::shared_ptr<const SymmetryBlockedTensor<2>> transformed_fock;
  if (source.has_inactive_fock_matrix()) {
    const auto& fock = source.get_inactive_fock_matrix().first;
    require_finite(fock, "Source inactive Fock matrix");
    Eigen::MatrixXd full_rotation =
        Eigen::MatrixXd::Identity(fock.rows(), fock.cols());
    for (Eigen::Index row = 0; row < nactive; ++row) {
      for (Eigen::Index column = 0; column < nactive; ++column) {
        full_rotation(source_indices[row], source_indices[column]) =
            rotation(row, column);
      }
    }
    Eigen::MatrixXd output = full_rotation.transpose() * fock * full_rotation;
    require_finite(output, "Transformed inactive Fock matrix");
    transformed_fock = std::make_shared<const SymmetryBlockedTensor<2>>(
        restricted_rank2(std::move(output)));
  }

  const auto& factors = source.get_three_center_integrals().first;
  require_finite(factors, "Source Cholesky factors");
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
  require_finite(transformed_factors, "Transformed Cholesky factors");

  auto transformed_three_center = restricted_rank3(
      *target_orbitals, source.three_center(), std::move(transformed_factors));
  auto container = std::make_unique<CholeskyHamiltonianContainer>(
      restricted_rank2(std::move(transformed_one_body)),
      std::move(transformed_three_center), std::move(target_orbitals),
      source.get_core_energy(), std::move(transformed_fock), std::nullopt,
      source.get_type());
  return std::make_shared<data::Hamiltonian>(std::move(container));
}

}  // namespace qdk::chemistry::algorithms::microsoft
