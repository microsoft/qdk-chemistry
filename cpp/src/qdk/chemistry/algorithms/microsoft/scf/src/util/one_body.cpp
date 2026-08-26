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

// Relative canonical-orthogonalization cutoff, applied independently to the
// unscaled overlap and kinetic metric blocks.
constexpr double metric_relative_linear_dependence_threshold = 1e-9;
// Eigenvalue cutoff for the projected-overlap pseudoinverse.
constexpr double overlap_linear_dependence_threshold = 1e-14;

struct MetricSubspace {
  Eigen::MatrixXd eigenvectors;
  Eigen::VectorXd eigenvalues;
  std::vector<Eigen::Index> retained_indices;
};

struct DiracEigensystem {
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;
  Eigen::Index large_component_rank;
};

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

/** @brief Diagonalize and relatively screen one unscaled metric block. */
MetricSubspace metric_subspace(const Eigen::MatrixXd& block,
                               const std::string& context) {
  MetricSubspace result{block, Eigen::VectorXd(block.rows()), {}};
  const int64_t info = lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower,
                                    block.rows(), result.eigenvectors.data(),
                                    block.rows(), result.eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error(
        "Symmetric eigendecomposition failed for X2C " + context +
        " metric block (info=" + std::to_string(info) + ")");
  }

  const double largest_eigenvalue = result.eigenvalues.maxCoeff();
  if (!std::isfinite(largest_eigenvalue) || largest_eigenvalue <= 0.0) {
    throw std::runtime_error("X2C " + context +
                             " metric block has no positive modes");
  }
  const double cutoff =
      metric_relative_linear_dependence_threshold * largest_eigenvalue;
  for (Eigen::Index index = 0; index < result.eigenvalues.size(); ++index) {
    if (result.eigenvalues(index) > cutoff) {
      result.retained_indices.push_back(index);
    }
  }
  if (result.retained_indices.empty()) {
    throw std::runtime_error("X2C " + context +
                             " metric block has no linearly independent "
                             "modes");
  }
  return result;
}

/** @brief Solve the modified Dirac problem in the screened metric space. */
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

  const auto overlap_subspace = metric_subspace(overlap, "overlap");
  const auto kinetic_subspace = metric_subspace(kinetic, "kinetic");
  const Eigen::Index large_component_rank =
      static_cast<Eigen::Index>(overlap_subspace.retained_indices.size());
  const Eigen::Index small_component_rank =
      static_cast<Eigen::Index>(kinetic_subspace.retained_indices.size());
  if (small_component_rank != large_component_rank) {
    throw std::runtime_error(
        "X2C overlap and kinetic metric blocks retained different ranks "
        "(overlap=" +
        std::to_string(large_component_rank) +
        ", kinetic=" + std::to_string(small_component_rank) + ")");
  }

  Eigen::MatrixXd dirac =
      Eigen::MatrixXd::Zero(dirac_dimension, dirac_dimension);
  dirac.topLeftCorner(dimension, dimension) = potential;
  dirac.topRightCorner(dimension, dimension) = kinetic;
  dirac.bottomLeftCorner(dimension, dimension) = kinetic;
  dirac.bottomRightCorner(dimension, dimension) =
      pvp * (inverse_speed_of_light_squared / 4.0) - kinetic;

  if (large_component_rank == dimension) {
    Eigen::MatrixXd metric =
        Eigen::MatrixXd::Zero(dirac_dimension, dirac_dimension);
    metric.topLeftCorner(dimension, dimension) = overlap;
    metric.bottomRightCorner(dimension, dimension) =
        kinetic * (inverse_speed_of_light_squared / 2.0);

    Eigen::VectorXd eigenvalues(dirac_dimension);
    const int64_t info = lapack::sygvd(
        1, lapack::Job::Vec, lapack::Uplo::Lower, dirac_dimension, dirac.data(),
        dirac_dimension, metric.data(), dirac_dimension, eigenvalues.data());
    if (info != 0) {
      throw std::runtime_error(
          "X2C generalized eigendecomposition failed after metric screening "
          "(info=" +
          std::to_string(info) + ")");
    }
    return {std::move(eigenvalues), std::move(dirac), large_component_rank};
  }

  const Eigen::Index retained_metric_dimension =
      large_component_rank + small_component_rank;
  Eigen::MatrixXd orthogonalizer =
      Eigen::MatrixXd::Zero(dirac_dimension, retained_metric_dimension);
  for (Eigen::Index column = 0; column < large_component_rank; ++column) {
    const Eigen::Index index = overlap_subspace.retained_indices[column];
    orthogonalizer.block(0, column, dimension, 1) =
        overlap_subspace.eigenvectors.col(index) /
        std::sqrt(overlap_subspace.eigenvalues(index));
  }
  const double small_component_scale = inverse_speed_of_light_squared / 2.0;
  for (Eigen::Index column = 0; column < small_component_rank; ++column) {
    const Eigen::Index index = kinetic_subspace.retained_indices[column];
    orthogonalizer.block(dimension, large_component_rank + column, dimension,
                         1) =
        kinetic_subspace.eigenvectors.col(index) /
        std::sqrt(small_component_scale * kinetic_subspace.eigenvalues(index));
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

  Eigen::VectorXd eigenvalues(retained_metric_dimension);
  const int64_t info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, retained_metric_dimension,
      orthogonal_dirac.data(), retained_metric_dimension, eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error(
        "Symmetric eigendecomposition failed for orthogonalized X2C Dirac "
        "Hamiltonian (info=" +
        std::to_string(info) + ")");
  }

  Eigen::MatrixXd eigenvectors(dirac_dimension, retained_metric_dimension);
  blas::gemm(
      blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      dirac_dimension, retained_metric_dimension, retained_metric_dimension,
      1.0, orthogonalizer.data(), dirac_dimension, orthogonal_dirac.data(),
      retained_metric_dimension, 0.0, eigenvectors.data(), dirac_dimension);
  return {std::move(eigenvalues), std::move(eigenvectors),
          large_component_rank};
}

/** @brief Construct the spin-free X2C-1e Hamiltonian from AO integrals. */
Eigen::MatrixXd compute_x2c_hamiltonian(const Eigen::MatrixXd& overlap,
                                        const Eigen::MatrixXd& kinetic,
                                        const Eigen::MatrixXd& potential,
                                        const Eigen::MatrixXd& pvp) {
  validate_x2c_inputs(overlap, kinetic, potential, pvp);

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
  if (static_cast<Eigen::Index>(electronic_indices.size()) !=
      dirac_eigensystem.large_component_rank) {
    throw std::runtime_error(
        "X2C selected an unexpected number of electronic states (expected=" +
        std::to_string(dirac_eigensystem.large_component_rank) +
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
  std::vector<Eigen::Index> retained_overlap_indices;
  for (Eigen::Index index = 0; index < projected_overlap_eigenvalues.size();
       ++index) {
    if (projected_overlap_eigenvalues(index) >
        overlap_linear_dependence_threshold) {
      retained_overlap_indices.push_back(index);
    }
  }
  if (retained_overlap_indices.size() != electronic_indices.size()) {
    throw std::runtime_error(
        "X2C projected electronic overlap lost rank (expected=" +
        std::to_string(electronic_indices.size()) +
        ", actual=" + std::to_string(retained_overlap_indices.size()) + ")");
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
    const qcs::ParallelConfig& mpi, bool decontract,
    const qcs::RowMajorMatrix* contracted_overlap) {
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
  if (!decontract && contracted_overlap != nullptr) {
    if (contracted_overlap->rows() != static_cast<Eigen::Index>(dimension) ||
        contracted_overlap->cols() != static_cast<Eigen::Index>(dimension)) {
      throw std::invalid_argument(
          "Precomputed X2C overlap dimension does not match the basis");
    }
    overlap = *contracted_overlap;
  } else {
    int1e->overlap_integral(overlap.data());
  }
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
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  MPI_Bcast(one_body_ao.data(), static_cast<int>(one_body_ao.size()),
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return one_body_ao;
}

RowMajorMatrix build_x2c_one_body_ao(
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set,
    const qcs::ParallelConfig& mpi, bool decontract,
    const qcs::RowMajorMatrix* contracted_overlap) {
  Eigen::MatrixXd hamiltonian;
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi.world_size == 1) {
    return compute_x2c_one_electron(internal_basis_set, mpi, decontract,
                                    contracted_overlap);
  }
  std::string local_error;
  try {
    hamiltonian = compute_x2c_one_electron(internal_basis_set, mpi, decontract,
                                           contracted_overlap);
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
  hamiltonian = compute_x2c_one_electron(internal_basis_set, mpi, decontract,
                                         contracted_overlap);
#endif
  return hamiltonian;
}

}  // namespace qdk::chemistry::scf
