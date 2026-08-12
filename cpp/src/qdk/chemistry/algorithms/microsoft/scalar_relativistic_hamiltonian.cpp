// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scalar_relativistic_hamiltonian.hpp"

#include <qdk/chemistry/scf/util/int1e.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <lapack.hh>
#include <map>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail_x2c {

namespace {

constexpr double metric_linear_dependence_threshold = 1e-9;
constexpr double overlap_linear_dependence_threshold = 1e-14;
constexpr double exponent_rounding_factor = 1e9;

void symmetric_eigendecomposition(Eigen::MatrixXd& matrix,
                                  Eigen::VectorXd& eigenvalues,
                                  const std::string& context) {
  const int64_t dimension = matrix.rows();
  eigenvalues.resize(dimension);
  const int64_t info =
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, dimension,
                   matrix.data(), dimension, eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error("X2C eigendecomposition failed for " + context +
                             " (info=" + std::to_string(info) + ")");
  }
}

std::vector<Eigen::Index> indices_above(const Eigen::VectorXd& values,
                                        double threshold) {
  std::vector<Eigen::Index> indices;
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    if (values(index) > threshold) indices.push_back(index);
  }
  return indices;
}

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

size_t shell_size(const qcs::Shell& shell, bool pure) {
  const size_t angular_momentum = shell.angular_momentum;
  return pure ? 2 * angular_momentum + 1
              : (angular_momentum + 1) * (angular_momentum + 2) / 2;
}

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

double rounded_exponent(double exponent) {
  return std::round(exponent * exponent_rounding_factor) /
         exponent_rounding_factor;
}

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
    Eigen::VectorXd metric_eigenvalues;
    symmetric_eigendecomposition(metric, metric_eigenvalues, "Dirac metric");
    const auto retained_metric_indices =
        indices_above(metric_eigenvalues, metric_linear_dependence_threshold);
    if (retained_metric_indices.empty()) {
      throw std::runtime_error(
          "X2C Dirac metric has no linearly independent modes");
    }

    Eigen::MatrixXd orthogonalizer(dirac_dimension,
                                   retained_metric_indices.size());
    for (size_t column = 0; column < retained_metric_indices.size(); ++column) {
      const Eigen::Index index = retained_metric_indices[column];
      orthogonalizer.col(column) =
          metric.col(index) / std::sqrt(metric_eigenvalues(index));
    }

    Eigen::MatrixXd orthogonal_dirac =
        orthogonalizer.transpose() * dirac * orthogonalizer;
    orthogonal_dirac =
        0.5 * (orthogonal_dirac + orthogonal_dirac.transpose()).eval();
    symmetric_eigendecomposition(orthogonal_dirac, dirac_eigenvalues,
                                 "orthogonalized Dirac Hamiltonian");
    dirac_eigenvectors = orthogonalizer * orthogonal_dirac;
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

  Eigen::MatrixXd large_components(dimension, electronic_indices.size());
  Eigen::VectorXd electronic_energies(electronic_indices.size());
  for (size_t column = 0; column < electronic_indices.size(); ++column) {
    const Eigen::Index index = electronic_indices[column];
    large_components.col(column) =
        dirac_eigenvectors.block(0, index, dimension, 1);
    electronic_energies(column) = dirac_eigenvalues(index);
  }

  Eigen::MatrixXd projected_overlap =
      large_components.transpose() * overlap * large_components;
  projected_overlap =
      0.5 * (projected_overlap + projected_overlap.transpose()).eval();
  Eigen::VectorXd projected_overlap_eigenvalues;
  symmetric_eigendecomposition(projected_overlap, projected_overlap_eigenvalues,
                               "projected electronic overlap");
  const auto retained_overlap_indices = indices_above(
      projected_overlap_eigenvalues, overlap_linear_dependence_threshold);
  if (retained_overlap_indices.empty()) {
    throw std::runtime_error(
        "X2C electronic overlap has no linearly independent modes");
  }

  Eigen::MatrixXd projected_overlap_inverse_sqrt = Eigen::MatrixXd::Zero(
      electronic_indices.size(), electronic_indices.size());
  for (const Eigen::Index index : retained_overlap_indices) {
    projected_overlap_inverse_sqrt.noalias() +=
        projected_overlap.col(index) *
        projected_overlap.col(index).transpose() /
        std::sqrt(projected_overlap_eigenvalues(index));
  }

  const Eigen::MatrixXd back_transform =
      projected_overlap_inverse_sqrt * large_components.transpose() * overlap;
  Eigen::MatrixXd hamiltonian = back_transform.transpose() *
                                electronic_energies.asDiagonal() *
                                back_transform;
  hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose()).eval();
  if (!hamiltonian.allFinite()) {
    throw std::runtime_error("X2C produced non-finite one-electron integrals");
  }
  return hamiltonian;
}

struct DecontractedBasis {
  std::shared_ptr<qcs::BasisSet> basis;
  Eigen::MatrixXd contraction;
};

DecontractedBasis decontract_basis(
    const std::shared_ptr<qcs::BasisSet>& contracted_basis) {
  using AtomAngularMomentum = std::pair<uint64_t, uint64_t>;
  using ExponentMap = std::map<double, double, std::greater<double>>;

  std::map<AtomAngularMomentum, ExponentMap> grouped_exponents;
  std::map<AtomAngularMomentum, std::array<double, 3>> origins;
  for (const auto& shell : contracted_basis->shells) {
    const AtomAngularMomentum key{shell.atom_index, shell.angular_momentum};
    origins[key] = shell.O;
    for (size_t primitive = 0; primitive < shell.contraction; ++primitive) {
      grouped_exponents[key].try_emplace(
          rounded_exponent(shell.exponents[primitive]),
          shell.exponents[primitive]);
    }
  }

  std::vector<qcs::Shell> uncontracted_shells;
  std::map<std::tuple<uint64_t, uint64_t, double>, size_t>
      primitive_shell_indices;
  for (const auto& [key, exponents] : grouped_exponents) {
    const auto [atom_index, angular_momentum] = key;
    for (const auto& [rounded, exponent] : exponents) {
      qcs::Shell shell{};
      shell.atom_index = atom_index;
      shell.O = origins.at(key);
      shell.angular_momentum = angular_momentum;
      shell.contraction = 1;
      shell.exponents[0] = exponent;
      shell.coefficients[0] = 1.0;
      primitive_shell_indices.emplace(
          std::make_tuple(atom_index, angular_momentum, rounded),
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
      const auto key = std::make_tuple(
          contracted_shell.atom_index, contracted_shell.angular_momentum,
          rounded_exponent(contracted_shell.exponents[primitive]));
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

Eigen::MatrixXd compute_x2c_one_electron(
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set,
    const qcs::ParallelConfig& mpi, bool xuncontract) {
  if (!internal_basis_set->ecp_shells.empty() ||
      internal_basis_set->n_ecp_electrons != 0) {
    throw std::invalid_argument(
        "The X2C Hamiltonian constructor does not support effective core "
        "potentials; use an all-electron basis set");
  }

  std::shared_ptr<qcs::BasisSet> working_basis = internal_basis_set;
  Eigen::MatrixXd contraction;
  if (xuncontract) {
    auto decontracted = decontract_basis(internal_basis_set);
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
  if (xuncontract) {
    hamiltonian = contraction.transpose() * hamiltonian * contraction;
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose()).eval();
  }
  if (!hamiltonian.allFinite()) {
    throw std::runtime_error("X2C produced non-finite one-electron integrals");
  }
  return hamiltonian;
}

}  // namespace

}  // namespace detail_x2c

std::shared_ptr<data::Hamiltonian>
ScalarRelativisticHamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals) const {
  QDK_LOG_TRACE_ENTERING();
  utils::microsoft::initialize_backend();

  auto basis_set = orbitals->get_basis_set();
  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  const auto mpi = qcs::mpi_default_input();
  Eigen::MatrixXd one_body_ao = detail_x2c::compute_x2c_one_electron(
      internal_basis_set, mpi, _settings->get<bool>("xuncontract"));
  return detail::construct_canonical_hamiltonian(
      std::move(orbitals), internal_basis_set, one_body_ao,
      _settings->get<std::string>("eri_method"));
}

}  // namespace qdk::chemistry::algorithms::microsoft
