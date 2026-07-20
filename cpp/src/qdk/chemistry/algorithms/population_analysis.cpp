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

std::vector<double> model_population(
    const std::shared_ptr<const data::SymmetryBlockedTensor<1>>& occupations,
    size_t n_sites) {
  std::vector<double> populations(n_sites, 0.0);
  for (const auto& [label, _] : occupations->extents()[0]) {
    const auto& block = occupations->block({label});
    if (block.size() != static_cast<Eigen::Index>(n_sites)) {
      throw std::runtime_error(
          "Cannot compute model populations: occupation dimensions do not "
          "match the number of sites.");
    }
    for (size_t site = 0; site < n_sites; ++site) {
      populations[site] += block(static_cast<Eigen::Index>(site));
    }
  }
  return populations;
}

std::vector<double> mulliken_population(
    const std::shared_ptr<data::Wavefunction>& wavefunction) {
  if (!wavefunction) {
    throw std::invalid_argument(
        "Population analysis requires a non-null wavefunction.");
  }

  auto orbitals = wavefunction->get_orbitals();
  if (!orbitals) {
    throw std::runtime_error(
        "QDK population analysis requires a wavefunction with orbitals.");
  }

  auto occupations = wavefunction->total_orbital_occupations();
  if (!occupations) {
    throw std::runtime_error(
        "QDK population analysis requires total orbital occupations.");
  }

  if (std::dynamic_pointer_cast<data::ModelOrbitals>(orbitals)) {
    return model_population(occupations,
                            orbitals->get_num_molecular_orbitals());
  }

  if (!orbitals->has_basis_set() || !orbitals->has_overlap_matrix()) {
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
  if (!coefficients) {
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

  return electron_population;
}

std::unique_ptr<PopulationAnalyzer> make_qdk_population_analyzer() {
  return std::make_unique<QdkPopulationAnalyzer>();
}

}  // namespace

std::vector<double> QdkPopulationAnalyzer::_run_impl(
    PopulationAnalysisInput input, int charge, int spin_multiplicity,
    unsigned int n_inactive_orbitals) const {
  (void)charge;
  (void)spin_multiplicity;
  (void)n_inactive_orbitals;
  return std::visit(
      [](const auto& value) -> std::vector<double> {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<ValueType,
                                     std::shared_ptr<data::Structure>>) {
          throw std::invalid_argument(
              "QDK population analysis requires a wavefunction; use a backend "
              "that can solve structure inputs first.");
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
