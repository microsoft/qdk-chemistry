// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/population_analysis.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace qdk::chemistry::algorithms {
namespace {

const Eigen::MatrixXd& real_one_rdm(
    const data::ContainerTypes::MatrixVariant& one_rdm) {
  const auto* real_rdm = std::get_if<Eigen::MatrixXd>(&one_rdm);
  if (!real_rdm) {
    throw std::runtime_error(
        "QDK population analysis requires a real-valued 1-RDM.");
  }
  return *real_rdm;
}

Eigen::MatrixXd embed_active_one_rdm(
    const Eigen::MatrixXd& active_one_rdm,
    const std::vector<size_t>& active_indices,
    const std::vector<size_t>& inactive_indices, size_t n_orbitals,
    double inactive_occupation) {
  if (active_one_rdm.rows() !=
          static_cast<Eigen::Index>(active_indices.size()) ||
      active_one_rdm.cols() !=
          static_cast<Eigen::Index>(active_indices.size())) {
    throw std::runtime_error(
        "QDK population analysis requires 1-RDM dimensions to match the "
        "active space.");
  }

  Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  for (size_t row = 0; row < active_indices.size(); ++row) {
    if (active_indices[row] >= n_orbitals) {
      throw std::runtime_error(
          "QDK population analysis encountered an invalid active-orbital "
          "index.");
    }
    for (size_t column = 0; column < active_indices.size(); ++column) {
      one_rdm(static_cast<Eigen::Index>(active_indices[row]),
              static_cast<Eigen::Index>(active_indices[column])) =
          active_one_rdm(static_cast<Eigen::Index>(row),
                         static_cast<Eigen::Index>(column));
    }
  }
  for (size_t index : inactive_indices) {
    if (index >= n_orbitals) {
      throw std::runtime_error(
          "QDK population analysis encountered an invalid inactive-orbital "
          "index.");
    }
    one_rdm(static_cast<Eigen::Index>(index),
            static_cast<Eigen::Index>(index)) = inactive_occupation;
  }
  return one_rdm;
}

bool has_spin_axis(const std::shared_ptr<data::Orbitals>& orbitals) {
  const auto symmetries = orbitals->symmetries();
  return symmetries && symmetries->has_axis(data::AxisName::Spin);
}

Eigen::MatrixXd total_one_rdm_spin_traced(
    const std::shared_ptr<data::Wavefunction>& wavefunction,
    const std::shared_ptr<data::Orbitals>& orbitals) {
  if (!wavefunction->has_one_rdm_spin_traced()) {
    throw std::runtime_error(
        "QDK population analysis requires a spin-traced active-space 1-RDM.");
  }

  const auto active_indices = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto inactive_indices = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::alpha());
  const double inactive_occupation = has_spin_axis(orbitals) ? 2.0 : 1.0;
  return embed_active_one_rdm(
      real_one_rdm(wavefunction->get_active_one_rdm_spin_traced()),
      active_indices, inactive_indices, orbitals->get_num_molecular_orbitals(),
      inactive_occupation);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> total_one_rdm_spin_dependent(
    const std::shared_ptr<data::Wavefunction>& wavefunction,
    const std::shared_ptr<data::Orbitals>& orbitals) {
  if (!wavefunction->has_one_rdm_spin_dependent()) {
    throw std::runtime_error(
        "QDK population analysis requires spin-dependent active-space 1-RDM "
        "blocks for unrestricted orbitals.");
  }

  auto [active_alpha_variant, active_beta_variant] =
      wavefunction->get_active_one_rdm_spin_dependent();
  const auto active_indices = orbitals->active_indices();
  const auto inactive_indices = orbitals->inactive_indices();
  auto alpha = embed_active_one_rdm(
      real_one_rdm(active_alpha_variant),
      data::spin_channel_indices(active_indices, data::axes::alpha()),
      data::spin_channel_indices(inactive_indices, data::axes::alpha()),
      orbitals->get_num_molecular_orbitals(), 1.0);
  auto beta = embed_active_one_rdm(
      real_one_rdm(active_beta_variant),
      data::spin_channel_indices(active_indices, data::axes::beta()),
      data::spin_channel_indices(inactive_indices, data::axes::beta()),
      orbitals->get_num_molecular_orbitals(), 1.0);
  return {std::move(alpha), std::move(beta)};
}

std::vector<double> model_population(
    const std::shared_ptr<data::Wavefunction>& wavefunction,
    const std::shared_ptr<data::Orbitals>& orbitals) {
  const size_t n_sites = orbitals->get_num_molecular_orbitals();
  Eigen::MatrixXd one_rdm;
  if (!has_spin_axis(orbitals) || orbitals->is_restricted()) {
    one_rdm = total_one_rdm_spin_traced(wavefunction, orbitals);
  } else {
    auto [one_rdm_alpha, one_rdm_beta] =
        total_one_rdm_spin_dependent(wavefunction, orbitals);
    one_rdm = one_rdm_alpha + one_rdm_beta;
  }

  std::vector<double> populations(n_sites, 0.0);
  for (size_t site = 0; site < n_sites; ++site) {
    populations[site] = one_rdm(static_cast<Eigen::Index>(site),
                                static_cast<Eigen::Index>(site));
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

  if (std::dynamic_pointer_cast<data::ModelOrbitals>(orbitals)) {
    return model_population(wavefunction, orbitals);
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
  Eigen::MatrixXd density;
  if (orbitals->is_restricted()) {
    density = orbitals->calculate_ao_density_matrix_from_rdm(
        total_one_rdm_spin_traced(wavefunction, orbitals));
  } else {
    auto [one_rdm_alpha, one_rdm_beta] =
        total_one_rdm_spin_dependent(wavefunction, orbitals);
    auto [density_alpha, density_beta] =
        orbitals->calculate_ao_density_matrix_from_rdm(one_rdm_alpha,
                                                       one_rdm_beta);
    density = density_alpha + density_beta;
  }
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
  const auto method = _settings->get<std::string>("method");
  if (method == "mulliken") {
    return std::visit(
        [](const auto& value) -> std::vector<double> {
          using ValueType = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<ValueType,
                                       std::shared_ptr<data::Structure>>) {
            throw std::invalid_argument(
                "QDK population analysis requires a wavefunction; use a "
                "backend that can solve structure inputs first.");
          } else {
            return mulliken_population(value);
          }
        },
        input);
  }
  throw std::invalid_argument("Unsupported QDK population-analysis method: " +
                              method);
}

void PopulationAnalyzerFactory::register_default_instances() {
  PopulationAnalyzerFactory::register_instance(&make_qdk_population_analyzer);
}

}  // namespace qdk::chemistry::algorithms
