// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/population_analysis.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::algorithms {
namespace {

std::unique_ptr<PopulationAnalyzer> make_qdk_population_analyzer() {
  return std::make_unique<QdkPopulationAnalyzer>();
}

}  // namespace

void PopulationAnalyzerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();
  PopulationAnalyzerFactory::register_instance(make_qdk_population_analyzer);
}

std::vector<double> QdkPopulationAnalyzer::_run_impl(
    PopulationAnalysisInput input, int, int, unsigned int) const {
  QDK_LOG_TRACE_ENTERING();

  const auto* wavefunction =
      std::get_if<std::shared_ptr<data::Wavefunction>>(&input);
  if (wavefunction == nullptr || *wavefunction == nullptr) {
    throw std::invalid_argument(
        "QDK population analysis requires a wavefunction input");
  }

  const auto orbitals = (*wavefunction)->get_orbitals();
  if (!orbitals->has_basis_set()) {
    const auto occupations = (*wavefunction)->total_orbital_occupations();
    Eigen::VectorXd populations;
    if (occupations->symmetries()[0]->has_axis(data::AxisName::Spin)) {
      populations = occupations->block({data::axes::alpha()}) +
                    occupations->block({data::axes::beta()});
    } else {
      populations = occupations->block({data::SymmetryLabel{}});
    }
    return {populations.data(), populations.data() + populations.size()};
  }

  const auto [occupations_alpha, occupations_beta] =
      (*wavefunction)->get_total_orbital_occupations();
  const auto [density_alpha, density_beta] =
      orbitals->calculate_ao_density_matrix(occupations_alpha,
                                            occupations_beta);
  const Eigen::VectorXd ao_populations =
      ((density_alpha + density_beta) * orbitals->get_overlap_matrix())
          .diagonal();
  const auto basis_set = orbitals->get_basis_set();
  std::vector<double> populations(basis_set->get_structure()->get_num_atoms(),
                                  0.0);
  for (Eigen::Index atomic_orbital = 0; atomic_orbital < ao_populations.size();
       ++atomic_orbital) {
    const auto atom = basis_set->get_atom_index_for_atomic_orbital(
        static_cast<size_t>(atomic_orbital));
    populations[atom] += ao_populations[atomic_orbital];
  }

  return populations;
}

}  // namespace qdk::chemistry::algorithms
