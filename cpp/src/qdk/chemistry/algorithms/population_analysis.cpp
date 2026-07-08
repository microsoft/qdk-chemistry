// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/population_analysis.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace qdk::chemistry::algorithms {
namespace {

std::vector<double> structure_only_population(
    const std::shared_ptr<data::Structure>& structure, double total_charge) {
  if (!structure) {
    throw std::invalid_argument(
        "Population analysis requires a non-null structure.");
  }
  if (structure->get_num_atoms() == 0) {
    return {};
  }
  return std::vector<double>(structure->get_num_atoms(),
                             total_charge / structure->get_num_atoms());
}

Eigen::MatrixXd density_from_coefficients_and_occupations(
    const Eigen::MatrixXd& coefficients, const Eigen::VectorXd& occupations) {
  if (coefficients.cols() != occupations.size()) {
    throw std::runtime_error(
        "Cannot compute Mulliken populations: coefficient and occupation "
        "dimensions are inconsistent.");
  }
  return coefficients * occupations.asDiagonal() * coefficients.transpose();
}

Eigen::MatrixXd density_from_symmetry_blocks(
    const data::SymmetryBlockedTensor<2>& coefficients,
    const data::SymmetryBlockedTensor<1>& occupations) {
  Eigen::MatrixXd density;
  bool initialized = false;

  for (const auto& [label, _] : occupations.extents()[0]) {
    const data::SymmetryBlockedTensor<2>::Labels coefficient_labels{label,
                                                                    label};
    if (!coefficients.has_block(coefficient_labels)) {
      continue;
    }

    const auto& coefficient_block = coefficients.block(coefficient_labels);
    const auto& occupation_block = occupations.block({label});
    auto block_density = density_from_coefficients_and_occupations(
        coefficient_block, occupation_block);

    if (!initialized) {
      density =
          Eigen::MatrixXd::Zero(block_density.rows(), block_density.cols());
      initialized = true;
    }
    if (density.rows() != block_density.rows() ||
        density.cols() != block_density.cols()) {
      throw std::runtime_error(
          "Cannot compute Mulliken populations: symmetry-blocked coefficient "
          "blocks produce inconsistent AO density dimensions.");
    }
    density += block_density;
  }

  if (!initialized) {
    throw std::runtime_error(
        "Cannot compute Mulliken populations: no matching coefficient and "
        "occupation symmetry blocks were found.");
  }

  return density;
}

std::vector<double> mulliken_population(
    const std::shared_ptr<data::Wavefunction>& wavefunction) {
  if (!wavefunction) {
    throw std::invalid_argument(
        "Population analysis requires a non-null wavefunction.");
  }

  auto orbitals = wavefunction->get_orbitals();
  if (!orbitals || !orbitals->has_basis_set() ||
      !orbitals->has_overlap_matrix()) {
    throw std::runtime_error(
        "QDK population analysis from a wavefunction requires orbitals with a "
        "basis set and AO overlap matrix.");
  }

  auto basis = orbitals->get_basis_set();
  if (!basis || !basis->has_structure()) {
    throw std::runtime_error(
        "QDK population analysis from a wavefunction requires the orbital "
        "basis set to carry the molecular structure.");
  }

  auto structure = basis->get_structure();
  const auto n_atoms = structure->get_num_atoms();
  std::vector<double> electron_population(n_atoms, 0.0);

  const auto& overlap = orbitals->get_overlap_matrix();
  auto coefficients = orbitals->coefficients();
  auto occupations = wavefunction->total_orbital_occupations();
  if (!coefficients || !occupations) {
    throw std::runtime_error(
        "QDK population analysis from a wavefunction requires symmetry-blocked "
        "orbital coefficients and occupations.");
  }

  Eigen::MatrixXd density =
      density_from_symmetry_blocks(*coefficients, *occupations);
  Eigen::MatrixXd population_matrix = density * overlap;

  for (int ao = 0; ao < population_matrix.rows(); ++ao) {
    const auto atom_index =
        basis->get_atom_index_for_atomic_orbital(static_cast<size_t>(ao));
    if (atom_index < electron_population.size()) {
      electron_population[atom_index] += population_matrix(ao, ao);
    }
  }

  std::vector<double> populations(n_atoms, 0.0);
  const auto& nuclear_charges = structure->get_nuclear_charges();
  for (size_t atom = 0; atom < n_atoms; ++atom) {
    populations[atom] = nuclear_charges(static_cast<Eigen::Index>(atom)) -
                        electron_population[atom];
  }
  return populations;
}

std::unique_ptr<PopulationAnalyzer> make_qdk_population_analyzer() {
  return std::make_unique<QdkPopulationAnalyzer>();
}

}  // namespace

std::vector<double> QdkPopulationAnalyzer::_run_impl(
    PopulationAnalysisInput input) const {
  const auto total_charge =
      static_cast<double>(this->_settings->get<int64_t>("charge"));
  return std::visit(
      [total_charge](const auto& value) -> std::vector<double> {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<ValueType,
                                     std::shared_ptr<data::Structure>>) {
          return structure_only_population(value, total_charge);
        } else {
          return mulliken_population(value);
        }
      },
      input);
}

void PopulationAnalyzerFactory::register_default_instances() {
  PopulationAnalyzerFactory::register_instance(&make_qdk_population_analyzer);
}

}  // namespace qdk::chemistry::algorithms
