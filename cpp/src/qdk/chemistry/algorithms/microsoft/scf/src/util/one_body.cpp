// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "util/one_body.h"

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

namespace qdk::chemistry::scf {

namespace qcs = qdk::chemistry::scf;

namespace {

// Relative overlap cutoff for the common large/small AO space.
constexpr double overlap_relative_linear_dependence_threshold = 1e-9;
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

/** @brief Solve the modified Dirac problem in a common RKB-preserving space. */
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

  // A successful generalized solve does not establish numerical overlap rank.
  Eigen::MatrixXd overlap_vectors = overlap;
  Eigen::VectorXd overlap_eigenvalues(dimension);
  const int64_t overlap_info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, dimension, overlap_vectors.data(),
      dimension, overlap_eigenvalues.data());
  if (overlap_info != 0) {
    throw std::runtime_error("X2C overlap eigendecomposition failed (info=" +
                             std::to_string(overlap_info) + ")");
  }
  const double cutoff = overlap_relative_linear_dependence_threshold *
                        overlap_eigenvalues(dimension - 1);
  Eigen::Index rank = 0;
  for (Eigen::Index index = 0; index < dimension; ++index) {
    if (overlap_eigenvalues(index) > cutoff) ++rank;
  }
  if (rank == 0) {
    throw std::runtime_error("X2C overlap has no linearly independent modes");
  }

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

  Eigen::MatrixXd reduction;
  if (rank < dimension) {
    reduction = Eigen::MatrixXd::Zero(dirac_dimension, 2 * rank);
    for (Eigen::Index column = 0; column < rank; ++column) {
      const Eigen::Index index = dimension - rank + column;
      double* large_column = reduction.data() + column * dirac_dimension;
      blas::copy(dimension, overlap_vectors.data() + index * dimension, 1,
                 large_column, 1);
      blas::scal(dimension, 1.0 / std::sqrt(overlap_eigenvalues(index)),
                 large_column, 1);
      // The same AO combinations generate the large basis and its RKB partners.
      blas::copy(
          dimension, large_column, 1,
          reduction.data() + (rank + column) * dirac_dimension + dimension, 1);
    }
    for (Eigen::MatrixXd* matrix : {&dirac, &metric}) {
      Eigen::MatrixXd product(dirac_dimension, 2 * rank);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 dirac_dimension, 2 * rank, dirac_dimension, 1.0,
                 matrix->data(), dirac_dimension, reduction.data(),
                 dirac_dimension, 0.0, product.data(), dirac_dimension);
      Eigen::MatrixXd reduced(2 * rank, 2 * rank);
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 2 * rank, 2 * rank, dirac_dimension, 1.0, reduction.data(),
                 dirac_dimension, product.data(), dirac_dimension, 0.0,
                 reduced.data(), 2 * rank);
      *matrix = 0.5 * (reduced + reduced.transpose()).eval();
    }
  }

  const Eigen::Index solve_dimension = 2 * rank;
  Eigen::VectorXd eigenvalues(solve_dimension);
  const int64_t info = lapack::sygvd(
      1, lapack::Job::Vec, lapack::Uplo::Lower, solve_dimension, dirac.data(),
      solve_dimension, metric.data(), solve_dimension, eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error(
        "X2C generalized eigendecomposition failed (info=" +
        std::to_string(info) + ")");
  }
  if (rank == dimension) {
    return {std::move(eigenvalues), std::move(dirac), rank};
  }

  Eigen::MatrixXd eigenvectors(dirac_dimension, solve_dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             dirac_dimension, solve_dimension, solve_dimension, 1.0,
             reduction.data(), dirac_dimension, dirac.data(), solve_dimension,
             0.0, eigenvectors.data(), dirac_dimension);
  return {std::move(eigenvalues), std::move(eigenvectors), rank};
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

  Eigen::MatrixXd scaled_overlap_vectors = projected_overlap;
  for (Eigen::Index index = 0; index < electronic_dimension; ++index) {
    blas::scal(electronic_dimension,
               1.0 / std::sqrt(projected_overlap_eigenvalues(index)),
               scaled_overlap_vectors.data() + index * electronic_dimension, 1);
  }
  Eigen::MatrixXd projected_overlap_inverse_sqrt(electronic_dimension,
                                                 electronic_dimension);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
             electronic_dimension, electronic_dimension, electronic_dimension,
             1.0, scaled_overlap_vectors.data(), electronic_dimension,
             projected_overlap.data(), electronic_dimension, 0.0,
             projected_overlap_inverse_sqrt.data(), electronic_dimension);

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

detail::DecontractedBasis detail::decontract_basis(
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
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set,
    const qcs::ParallelConfig& mpi, bool decontract) {
  if (mpi.world_size > 1) {
    throw std::runtime_error(
        "X2C construction is not supported with MPI world_size > 1.");
  }
  if (!internal_basis_set->pure) {
    throw std::invalid_argument("X2C-1e currently supports spherical AOs only");
  }
  if (!internal_basis_set->ecp_shells.empty() ||
      internal_basis_set->get_n_ecp_electrons() != 0) {
    throw std::invalid_argument(
        "The X2C-1e approximation does not support effective core potentials; "
        "use an all-electron basis set");
  }

  std::shared_ptr<qcs::BasisSet> working_basis = internal_basis_set;
  Eigen::MatrixXd contraction;
  if (decontract) {
    auto decontracted = detail::decontract_basis(internal_basis_set);
    working_basis = std::move(decontracted.basis);
    contraction = std::move(decontracted.contraction);
  }

  const size_t dimension = working_basis->num_atomic_orbitals;
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

}  // namespace qdk::chemistry::scf
