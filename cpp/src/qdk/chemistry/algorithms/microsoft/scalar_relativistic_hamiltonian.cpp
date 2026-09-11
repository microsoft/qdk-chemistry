// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scalar_relativistic_hamiltonian.hpp"

#include <qdk/chemistry/scf/util/int1e.h>

#include <array>
#include <blas.hh>
#include <cmath>
#include <cstdint>
#include <functional>
#include <lapack.hh>
#include <map>
#include <qdk/chemistry/constants.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail {

namespace {

// Canonical-orthogonalization cutoff for the modified Dirac metric.
constexpr double metric_linear_dependence_threshold = 1e-9;
// Eigenvalue cutoff for the projected-overlap inverse square root.
constexpr double overlap_linear_dependence_threshold = 1e-14;

struct DiracEigensystem {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
  Eigen::Index large_component_metric_rank;
};

/** @brief Return the number of AO components represented by a shell. */
size_t shell_size(const qcs::Shell& shell, bool pure) {
  const size_t angular_momentum = shell.angular_momentum;
  return pure ? 2 * angular_momentum + 1
              : (angular_momentum + 1) * (angular_momentum + 2) / 2;
}

/** @brief Compute the first AO offset of every shell. */
std::vector<size_t> shell_offsets(const std::vector<qcs::Shell>& shells,
                                  bool pure) {
  std::vector<size_t> offsets(shells.size());
  size_t offset = 0;
  for (size_t shell_index = 0; shell_index < shells.size(); ++shell_index) {
    offsets[shell_index] = offset;
    offset += shell_size(shells[shell_index], pure);
  }
  return offsets;
}

/** @brief Solve the modified Dirac problem with a screened-metric fallback. */
DiracEigensystem solve_modified_dirac(const Eigen::MatrixXd& overlap,
                                      const Eigen::MatrixXd& kinetic,
                                      const Eigen::MatrixXd& potential,
                                      const Eigen::MatrixXd& pvp) {
  const Eigen::Index dimension = overlap.rows();
  const Eigen::Index dirac_dimension = 2 * dimension;
  const double inverse_speed_of_light =
      qdk::chemistry::constants::fine_structure_constant;
  const double inverse_speed_of_light_squared =
      inverse_speed_of_light * inverse_speed_of_light;

  Eigen::MatrixXd dirac =
      Eigen::MatrixXd::Zero(dirac_dimension, dirac_dimension);
  dirac.topLeftCorner(dimension, dimension) = potential;
  dirac.topRightCorner(dimension, dimension) = kinetic;
  dirac.bottomLeftCorner(dimension, dimension) = kinetic;
  dirac.bottomRightCorner(dimension, dimension) =
      pvp * (inverse_speed_of_light_squared / 4.0) - kinetic;

  Eigen::MatrixXd metric =
      Eigen::MatrixXd::Zero(dirac_dimension, dirac_dimension);
  metric.topLeftCorner(dimension, dimension) = overlap;
  metric.bottomRightCorner(dimension, dimension) =
      kinetic * (inverse_speed_of_light_squared / 2.0);

  Eigen::MatrixXd generalized_dirac = dirac;
  Eigen::MatrixXd generalized_metric = metric;
  Eigen::VectorXd eigenvalues(dirac_dimension);
  const int64_t generalized_info = lapack::sygvd(
      1, lapack::Job::Vec, lapack::Uplo::Lower, dirac_dimension,
      generalized_dirac.data(), dirac_dimension, generalized_metric.data(),
      dirac_dimension, eigenvalues.data());
  if (generalized_info == 0) {
    return {std::move(eigenvalues), std::move(generalized_dirac), dimension};
  }
  if (generalized_info <= dirac_dimension) {
    throw std::runtime_error(
        "X2C generalized eigendecomposition failed (info=" +
        std::to_string(generalized_info) + ")");
  }

  Eigen::VectorXd metric_eigenvalues(dirac_dimension);
  const int64_t metric_info =
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, dirac_dimension,
                   metric.data(), dirac_dimension, metric_eigenvalues.data());
  if (metric_info != 0) {
    throw std::runtime_error(
        "Symmetric eigendecomposition failed for X2C Dirac metric (info=" +
        std::to_string(metric_info) + ")");
  }
  std::vector<Eigen::Index> retained_metric_indices;
  for (Eigen::Index index = 0; index < metric_eigenvalues.size(); ++index) {
    if (metric_eigenvalues(index) > metric_linear_dependence_threshold) {
      retained_metric_indices.push_back(index);
    }
  }
  if (retained_metric_indices.empty()) {
    throw std::runtime_error(
        "X2C Dirac metric has no linearly independent "
        "modes");
  }

  // The retained metric projector is block diagonal, so the trace of its
  // large-component block is the retained overlap rank.
  double large_component_metric_rank_trace = 0.0;
  for (const Eigen::Index index : retained_metric_indices) {
    large_component_metric_rank_trace +=
        metric.col(index).head(dimension).squaredNorm();
  }
  const Eigen::Index large_component_metric_rank = static_cast<Eigen::Index>(
      std::llround(large_component_metric_rank_trace));

  const Eigen::Index retained_metric_dimension =
      static_cast<Eigen::Index>(retained_metric_indices.size());
  Eigen::MatrixXd orthogonalizer(dirac_dimension, retained_metric_dimension);
  for (size_t column = 0; column < retained_metric_indices.size(); ++column) {
    const Eigen::Index index = retained_metric_indices[column];
    orthogonalizer.col(column) =
        metric.col(index) / std::sqrt(metric_eigenvalues(index));
  }

  Eigen::MatrixXd dirac_times_orthogonalizer(dirac_dimension,
                                             retained_metric_dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             dirac_dimension, retained_metric_dimension, dirac_dimension, 1.0,
             dirac.data(), dirac_dimension, orthogonalizer.data(),
             dirac_dimension, 0.0, dirac_times_orthogonalizer.data(),
             dirac_dimension);
  Eigen::MatrixXd orthogonal_dirac(retained_metric_dimension,
                                   retained_metric_dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             retained_metric_dimension, retained_metric_dimension,
             dirac_dimension, 1.0, orthogonalizer.data(), dirac_dimension,
             dirac_times_orthogonalizer.data(), dirac_dimension, 0.0,
             orthogonal_dirac.data(), retained_metric_dimension);
  orthogonal_dirac =
      0.5 * (orthogonal_dirac + orthogonal_dirac.transpose()).eval();

  eigenvalues.resize(retained_metric_dimension);
  const int64_t info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, retained_metric_dimension,
      orthogonal_dirac.data(), retained_metric_dimension, eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error(
        "Symmetric eigendecomposition failed for orthogonalized X2C Dirac "
        "Hamiltonian "
        "(info=" +
        std::to_string(info) + ")");
  }

  Eigen::MatrixXd eigenvectors(dirac_dimension, retained_metric_dimension);
  blas::gemm(
      blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      dirac_dimension, retained_metric_dimension, retained_metric_dimension,
      1.0, orthogonalizer.data(), dirac_dimension, orthogonal_dirac.data(),
      retained_metric_dimension, 0.0, eigenvectors.data(), dirac_dimension);
  return {std::move(eigenvalues), std::move(eigenvectors),
          large_component_metric_rank};
}

/** @brief Construct the spin-free X2C-1e Hamiltonian from AO integrals. */
Eigen::MatrixXd compute_x2c_hamiltonian(const Eigen::MatrixXd& overlap,
                                        const Eigen::MatrixXd& kinetic,
                                        const Eigen::MatrixXd& potential,
                                        const Eigen::MatrixXd& pvp) {
  if (!overlap.allFinite() || !kinetic.allFinite() || !potential.allFinite() ||
      !pvp.allFinite()) {
    throw std::invalid_argument(
        "X2C input matrices must contain finite values");
  }

  const Eigen::Index dimension = overlap.rows();
  const double inverse_speed_of_light =
      qdk::chemistry::constants::fine_structure_constant;
  const double speed_of_light_squared =
      1.0 / (inverse_speed_of_light * inverse_speed_of_light);
  auto dirac_eigensystem =
      solve_modified_dirac(overlap, kinetic, potential, pvp);

  std::vector<Eigen::Index> electronic_indices;
  for (Eigen::Index index = 0; index < dirac_eigensystem.eigenvalues.size();
       ++index) {
    if (dirac_eigensystem.eigenvalues(index) > -speed_of_light_squared) {
      electronic_indices.push_back(index);
    }
  }
  if (electronic_indices.empty()) {
    throw std::runtime_error("X2C found no positive-energy electronic states");
  }
  if (static_cast<Eigen::Index>(electronic_indices.size()) !=
      dirac_eigensystem.large_component_metric_rank) {
    throw std::runtime_error(
        "X2C electronic subspace is incomplete (expected=" +
        std::to_string(dirac_eigensystem.large_component_metric_rank) +
        ", actual=" + std::to_string(electronic_indices.size()) + ")");
  }

  const Eigen::Index electronic_dimension =
      static_cast<Eigen::Index>(electronic_indices.size());
  Eigen::MatrixXd large_components(dimension, electronic_dimension);
  Eigen::VectorXd electronic_energies(electronic_dimension);
  for (size_t column = 0; column < electronic_indices.size(); ++column) {
    const Eigen::Index index = electronic_indices[column];
    large_components.col(column) =
        dirac_eigensystem.eigenvectors.block(0, index, dimension, 1);
    electronic_energies(column) = dirac_eigensystem.eigenvalues(index);
  }

  Eigen::MatrixXd overlap_times_large_components(dimension,
                                                 electronic_dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             dimension, electronic_dimension, dimension, 1.0, overlap.data(),
             dimension, large_components.data(), dimension, 0.0,
             overlap_times_large_components.data(), dimension);
  Eigen::MatrixXd projected_overlap(electronic_dimension, electronic_dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             electronic_dimension, electronic_dimension, dimension, 1.0,
             large_components.data(), dimension,
             overlap_times_large_components.data(), dimension, 0.0,
             projected_overlap.data(), electronic_dimension);
  projected_overlap =
      0.5 * (projected_overlap + projected_overlap.transpose()).eval();
  Eigen::VectorXd projected_overlap_eigenvalues(electronic_dimension);
  const int64_t projected_overlap_info =
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, electronic_dimension,
                   projected_overlap.data(), electronic_dimension,
                   projected_overlap_eigenvalues.data());
  if (projected_overlap_info != 0) {
    throw std::runtime_error(
        "Symmetric eigendecomposition failed for X2C projected electronic "
        "overlap (info=" +
        std::to_string(projected_overlap_info) + ")");
  }
  const Eigen::Index retained_overlap_dimension =
      (projected_overlap_eigenvalues.array() >
       overlap_linear_dependence_threshold)
          .count();
  if (retained_overlap_dimension != electronic_dimension) {
    throw std::runtime_error(
        "X2C projected electronic overlap lost rank (expected=" +
        std::to_string(electronic_dimension) +
        ", actual=" + std::to_string(retained_overlap_dimension) + ")");
  }

  Eigen::MatrixXd projected_overlap_inverse_sqrt =
      Eigen::MatrixXd::Zero(electronic_dimension, electronic_dimension);
  for (Eigen::Index index = 0; index < electronic_dimension; ++index) {
    projected_overlap_inverse_sqrt.noalias() +=
        projected_overlap.col(index) *
        projected_overlap.col(index).transpose() /
        std::sqrt(projected_overlap_eigenvalues(index));
  }

  Eigen::MatrixXd overlap_projection(electronic_dimension, dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             electronic_dimension, dimension, dimension, 1.0,
             large_components.data(), dimension, overlap.data(), dimension, 0.0,
             overlap_projection.data(), electronic_dimension);
  Eigen::MatrixXd back_transform(electronic_dimension, dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             electronic_dimension, dimension, electronic_dimension, 1.0,
             projected_overlap_inverse_sqrt.data(), electronic_dimension,
             overlap_projection.data(), electronic_dimension, 0.0,
             back_transform.data(), electronic_dimension);
  Eigen::MatrixXd weighted_back_transform = back_transform;
  weighted_back_transform.array().colwise() *= electronic_energies.array();
  Eigen::MatrixXd hamiltonian(dimension, dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             dimension, dimension, electronic_dimension, 1.0,
             back_transform.data(), electronic_dimension,
             weighted_back_transform.data(), electronic_dimension, 0.0,
             hamiltonian.data(), dimension);
  hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose()).eval();
  return hamiltonian;
}

}  // namespace

DecontractedBasis decontract_basis(
    const std::shared_ptr<qcs::BasisSet>& contracted_basis) {
  using AtomAngularMomentum = std::pair<uint64_t, uint64_t>;
  using ExponentSet = std::set<double, std::greater<double>>;

  std::map<AtomAngularMomentum, ExponentSet> grouped_exponents;
  std::map<AtomAngularMomentum, std::array<double, 3>> origins;
  for (const auto& shell : contracted_basis->shells) {
    const AtomAngularMomentum key{shell.atom_index, shell.angular_momentum};
    origins[key] = shell.O;
    for (size_t primitive = 0; primitive < shell.contraction; ++primitive) {
      grouped_exponents[key].insert(shell.exponents[primitive]);
    }
  }

  std::vector<qcs::Shell> uncontracted_shells;
  std::map<std::tuple<uint64_t, uint64_t, double>, size_t>
      primitive_shell_indices;
  for (const auto& [key, exponents] : grouped_exponents) {
    const auto [atom_index, angular_momentum] = key;
    for (const double exponent : exponents) {
      qcs::Shell shell{};
      shell.atom_index = atom_index;
      shell.O = origins.at(key);
      shell.angular_momentum = angular_momentum;
      shell.contraction = 1;
      shell.exponents[0] = exponent;
      shell.coefficients[0] = 1.0;
      primitive_shell_indices.emplace(
          std::make_tuple(atom_index, angular_momentum, exponent),
          uncontracted_shells.size());
      uncontracted_shells.push_back(shell);
    }
  }

  auto uncontracted_basis = std::make_shared<qcs::BasisSet>(
      contracted_basis->mol, uncontracted_shells, contracted_basis->mode,
      contracted_basis->pure, false);
  const auto contracted_offsets =
      shell_offsets(contracted_basis->shells, contracted_basis->pure);
  const auto uncontracted_offsets =
      shell_offsets(uncontracted_basis->shells, uncontracted_basis->pure);
  Eigen::MatrixXd contraction =
      Eigen::MatrixXd::Zero(uncontracted_basis->num_atomic_orbitals,
                            contracted_basis->num_atomic_orbitals);

  for (size_t contracted_shell_index = 0;
       contracted_shell_index < contracted_basis->shells.size();
       ++contracted_shell_index) {
    const auto& contracted_shell =
        contracted_basis->shells[contracted_shell_index];
    const size_t components =
        shell_size(contracted_shell, contracted_basis->pure);
    for (size_t primitive = 0; primitive < contracted_shell.contraction;
         ++primitive) {
      const auto key = std::make_tuple(contracted_shell.atom_index,
                                       contracted_shell.angular_momentum,
                                       contracted_shell.exponents[primitive]);
      const size_t uncontracted_shell_index = primitive_shell_indices.at(key);
      const double primitive_coefficient =
          uncontracted_basis->shells[uncontracted_shell_index].coefficients[0];
      const double coefficient =
          contracted_shell.coefficients[primitive] / primitive_coefficient;
      for (size_t component = 0; component < components; ++component) {
        contraction(uncontracted_offsets[uncontracted_shell_index] + component,
                    contracted_offsets[contracted_shell_index] + component) +=
            coefficient;
      }
    }
  }

  return {std::move(uncontracted_basis), std::move(contraction)};
}

Eigen::MatrixXd build_x2c_one_body_ao(
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set, bool decontract) {
  if (!internal_basis_set->ecp_shells.empty() ||
      internal_basis_set->get_n_ecp_electrons() != 0) {
    throw std::invalid_argument(
        "The X2C-1e approximation does not support effective core potentials; "
        "use an all-electron basis set");
  }

  std::shared_ptr<qcs::BasisSet> working_basis = internal_basis_set;
  Eigen::MatrixXd contraction;
  if (decontract) {
    auto decontracted = decontract_basis(internal_basis_set);
    working_basis = std::move(decontracted.basis);
    contraction = std::move(decontracted.contraction);
  }

  const size_t dimension = working_basis->num_atomic_orbitals;
  const auto mpi = qcs::mpi_default_input();
  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      working_basis.get(), working_basis->mol.get(), mpi);
  Eigen::MatrixXd overlap(dimension, dimension);
  Eigen::MatrixXd kinetic(dimension, dimension);
  Eigen::MatrixXd potential(dimension, dimension);
  Eigen::MatrixXd pvp(dimension, dimension);
  int1e->overlap_integral(overlap.data());
  int1e->kinetic_integral(kinetic.data());
  int1e->nuclear_integral(potential.data());
  int1e->pvp_integral(pvp.data());

  Eigen::MatrixXd hamiltonian =
      compute_x2c_hamiltonian(overlap, kinetic, potential, pvp);
  if (decontract) {
    const Eigen::Index contracted_dimension = contraction.cols();
    Eigen::MatrixXd hamiltonian_times_contraction(hamiltonian.rows(),
                                                  contracted_dimension);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               hamiltonian.rows(), contracted_dimension, hamiltonian.cols(),
               1.0, hamiltonian.data(), hamiltonian.rows(), contraction.data(),
               contraction.rows(), 0.0, hamiltonian_times_contraction.data(),
               hamiltonian_times_contraction.rows());
    Eigen::MatrixXd recontracted_hamiltonian(contracted_dimension,
                                             contracted_dimension);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               contracted_dimension, contracted_dimension, contraction.rows(),
               1.0, contraction.data(), contraction.rows(),
               hamiltonian_times_contraction.data(),
               hamiltonian_times_contraction.rows(), 0.0,
               recontracted_hamiltonian.data(), contracted_dimension);
    hamiltonian = std::move(recontracted_hamiltonian);
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose()).eval();
  }
  if (!hamiltonian.allFinite()) {
    throw std::runtime_error("X2C produced non-finite one-electron integrals");
  }
  return hamiltonian;
}

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
