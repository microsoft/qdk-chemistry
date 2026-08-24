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

#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif

namespace qdk::chemistry::scf {

namespace qcs = qdk::chemistry::scf;

namespace {

// Canonical-orthogonalization cutoff for the modified Dirac
// metric.
constexpr double metric_linear_dependence_threshold = 1e-9;
// Eigenvalue cutoff for the projected-overlap pseudoinverse.
constexpr double overlap_linear_dependence_threshold = 1e-14;

/** @brief Validate the dimensions, finiteness, and symmetry of X2C inputs. */
void validate_x2c_inputs(const Eigen::MatrixXd& overlap,
                         const Eigen::MatrixXd& kinetic,
                         const Eigen::MatrixXd& potential,
                         const Eigen::MatrixXd& pvp) {
  const Eigen::Index dimension = overlap.rows();
  if (dimension == 0 || overlap.cols() != dimension ||
      kinetic.rows() != dimension || kinetic.cols() != dimension ||
      potential.rows() != dimension || potential.cols() != dimension ||
      pvp.rows() != dimension || pvp.cols() != dimension) {
    throw std::invalid_argument(
        "X2C input matrices must be non-empty square matrices of equal size");
  }
  if (!overlap.allFinite() || !kinetic.allFinite() || !potential.allFinite() ||
      !pvp.allFinite()) {
    throw std::invalid_argument(
        "X2C input matrices must contain finite values");
  }
  if (!overlap.isApprox(overlap.transpose(), 1e-10) ||
      !kinetic.isApprox(kinetic.transpose(), 1e-10) ||
      !potential.isApprox(potential.transpose(), 1e-10) ||
      !pvp.isApprox(pvp.transpose(), 1e-10)) {
    throw std::invalid_argument("X2C input matrices must be symmetric");
  }
}

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

/** @brief Construct the spin-free X2C-1e Hamiltonian from AO integrals. */
Eigen::MatrixXd compute_x2c_hamiltonian(const Eigen::MatrixXd& overlap,
                                        const Eigen::MatrixXd& kinetic,
                                        const Eigen::MatrixXd& potential,
                                        const Eigen::MatrixXd& pvp) {
  validate_x2c_inputs(overlap, kinetic, potential, pvp);

  const Eigen::Index dimension = overlap.rows();
  const Eigen::Index dirac_dimension = 2 * dimension;
  const double inverse_speed_of_light =
      qdk::chemistry::constants::fine_structure_constant;
  const double inverse_speed_of_light_squared =
      inverse_speed_of_light * inverse_speed_of_light;
  const double speed_of_light_squared = 1.0 / inverse_speed_of_light_squared;

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
  Eigen::VectorXd dirac_eigenvalues(dirac_dimension);
  const int64_t generalized_info = lapack::sygvd(
      1, lapack::Job::Vec, lapack::Uplo::Lower, dirac_dimension,
      generalized_dirac.data(), dirac_dimension, generalized_metric.data(),
      dirac_dimension, dirac_eigenvalues.data());

  Eigen::MatrixXd dirac_eigenvectors;
  if (generalized_info == 0) {
    dirac_eigenvectors = std::move(generalized_dirac);
  } else if (generalized_info > dirac_dimension) {
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
          "X2C Dirac metric has no linearly independent modes");
    }

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
    dirac_eigenvalues.resize(retained_metric_dimension);
    const int64_t orthogonal_dirac_info =
        lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower,
                     retained_metric_dimension, orthogonal_dirac.data(),
                     retained_metric_dimension, dirac_eigenvalues.data());
    if (orthogonal_dirac_info != 0) {
      throw std::runtime_error(
          "Symmetric eigendecomposition failed for orthogonalized X2C Dirac "
          "Hamiltonian (info=" +
          std::to_string(orthogonal_dirac_info) + ")");
    }
    dirac_eigenvectors.resize(dirac_dimension, retained_metric_dimension);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               dirac_dimension, retained_metric_dimension,
               retained_metric_dimension, 1.0, orthogonalizer.data(),
               dirac_dimension, orthogonal_dirac.data(),
               retained_metric_dimension, 0.0, dirac_eigenvectors.data(),
               dirac_dimension);
  } else {
    throw std::runtime_error(
        "X2C generalized eigendecomposition failed (info=" +
        std::to_string(generalized_info) + ")");
  }

  std::vector<Eigen::Index> electronic_indices;
  for (Eigen::Index index = 0; index < dirac_eigenvalues.size(); ++index) {
    if (dirac_eigenvalues(index) > -speed_of_light_squared) {
      electronic_indices.push_back(index);
    }
  }
  if (electronic_indices.empty()) {
    throw std::runtime_error("X2C found no positive-energy electronic states");
  }

  const Eigen::Index electronic_dimension =
      static_cast<Eigen::Index>(electronic_indices.size());
  Eigen::MatrixXd large_components(dimension, electronic_dimension);
  Eigen::VectorXd electronic_energies(electronic_dimension);
  for (size_t column = 0; column < electronic_indices.size(); ++column) {
    const Eigen::Index index = electronic_indices[column];
    large_components.col(column) =
        dirac_eigenvectors.block(0, index, dimension, 1);
    electronic_energies(column) = dirac_eigenvalues(index);
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
  std::vector<Eigen::Index> retained_overlap_indices;
  for (Eigen::Index index = 0; index < projected_overlap_eigenvalues.size();
       ++index) {
    if (projected_overlap_eigenvalues(index) >
        overlap_linear_dependence_threshold) {
      retained_overlap_indices.push_back(index);
    }
  }
  if (retained_overlap_indices.empty()) {
    throw std::runtime_error(
        "X2C electronic overlap has no linearly independent modes");
  }

  Eigen::MatrixXd projected_overlap_inverse_sqrt =
      Eigen::MatrixXd::Zero(electronic_dimension, electronic_dimension);
  for (const Eigen::Index index : retained_overlap_indices) {
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
  if (!hamiltonian.allFinite()) {
    throw std::runtime_error("X2C produced non-finite one-electron integrals");
  }
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

namespace {

/** @brief Compute X2C-1e integrals, optionally in a decontracted basis. */
Eigen::MatrixXd compute_x2c_one_electron(
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set,
    const qcs::ParallelConfig& mpi, bool decontract) {
  if (!internal_basis_set->pure) {
    throw std::invalid_argument("X2C-1e currently supports spherical AOs only");
  }
  if (!internal_basis_set->ecp_shells.empty() ||
      internal_basis_set->n_ecp_electrons != 0) {
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
      Eigen::MatrixXd::Zero(internal_basis_set->num_atomic_orbitals,
                            internal_basis_set->num_atomic_orbitals);
  if (mpi.world_rank != 0) {
    return hamiltonian;
  }

  hamiltonian = compute_x2c_hamiltonian(overlap, kinetic, potential, pvp);
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

}  // namespace

RowMajorMatrix build_nonrelativistic_one_body_ao(const BasisSet& basis_set,
                                                 OneBodyIntegral& integrals) {
  const size_t dimension = basis_set.num_atomic_orbitals;
  RowMajorMatrix kinetic(dimension, dimension);
  RowMajorMatrix potential(dimension, dimension);
  integrals.kinetic_integral(kinetic.data());
  integrals.nuclear_integral(potential.data());
  RowMajorMatrix one_body_ao = kinetic + potential;

  if (!basis_set.ecp_shells.empty()) {
    RowMajorMatrix ecp = RowMajorMatrix::Zero(dimension, dimension);
    integrals.ecp_integral(ecp.data());
    one_body_ao += ecp;
  }
  return one_body_ao;
}

RowMajorMatrix build_x2c_one_body_ao(
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set,
    const qcs::ParallelConfig& mpi, bool decontract) {
  Eigen::MatrixXd hamiltonian;
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi.world_size == 1) {
    return compute_x2c_one_electron(internal_basis_set, mpi, decontract);
  }
  std::string local_error;
  try {
    hamiltonian = compute_x2c_one_electron(internal_basis_set, mpi, decontract);
  } catch (const std::exception& error) {
    local_error = error.what();
  } catch (...) {
    local_error = "unknown error";
  }
  const int local_succeeded = local_error.empty() ? 1 : 0;
  int succeeded = 0;
  MPI_Allreduce(&local_succeeded, &succeeded, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if (succeeded == 0) {
    const int local_failed_rank =
        local_error.empty() ? mpi.world_size : mpi.world_rank;
    int failed_rank = 0;
    MPI_Allreduce(&local_failed_rank, &failed_rank, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    int message_size = mpi.world_rank == failed_rank
                           ? static_cast<int>(local_error.size())
                           : 0;
    MPI_Bcast(&message_size, 1, MPI_INT, failed_rank, MPI_COMM_WORLD);
    if (mpi.world_rank != failed_rank) {
      local_error.resize(message_size);
    }
    MPI_Bcast(local_error.data(), message_size, MPI_CHAR, failed_rank,
              MPI_COMM_WORLD);
    throw std::runtime_error("X2C construction failed on MPI rank " +
                             std::to_string(failed_rank) + ": " + local_error);
  }
  MPI_Bcast(hamiltonian.data(), static_cast<int>(hamiltonian.size()),
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
  hamiltonian = compute_x2c_one_electron(internal_basis_set, mpi, decontract);
#endif
  return hamiltonian;
}

}  // namespace qdk::chemistry::scf
