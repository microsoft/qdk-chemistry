// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_pmc.hpp"

#include <macis/asci/determinant_search.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/mcscf/cas.hpp>
#include <macis/sd_operations.hpp>
#include <macis/solvers/selected_ci_diag.hpp>
#include <macis/util/mpi.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Helper struct for ProjectedMultiConfiguration (PMC) calculation
 * dispatch
 */
struct pmc_helper {
  using return_type = std::pair<double, data::Wavefunction>;

  /**
   * @brief Template implementation of PMC calculation
   * @tparam N Number of bits for wavefunction representation
   * @param hamiltonian Hamiltonian object containing molecular integrals
   * @param configurations Set of configurations to project onto
   * @param settings_ PMC calculation settings
   * @return std::pair containing energy and wavefunction
   */
  template <size_t N>
  static return_type impl(
      const data::Hamiltonian& hamiltonian,
      const std::vector<data::Configuration>& configurations,
      const data::Settings& settings_) {
    // Create MacisPmcSettings instance for accessing PMC-specific settings
    MacisPmcSettings macis_pmc_settings;
    double h_el_tol = macis_pmc_settings.get<double>("h_el_tol");
    double H_thresh = macis_pmc_settings.get<double>("H_thresh");
    double davidson_res_tol =
        macis_pmc_settings.get<double>("davidson_res_tol");
    int64_t iterative_solver_dimension_cutoff =
        macis_pmc_settings.get<int64_t>("iterative_solver_dimension_cutoff");
    int64_t davidson_max_m = macis_pmc_settings.get<int64_t>("davidson_max_m");

    using wfn_type = macis::wfn_t<N>;
    using wfn_traits = macis::wavefunction_traits<wfn_type>;
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

    auto orbitals = hamiltonian.get_orbitals();
    std::vector<size_t> active_indices =
        orbitals->get_active_space_indices().first;
    const size_t num_molecular_orbitals = active_indices.size();

    const auto& [T_a, T_b] = hamiltonian.get_one_body_integrals();
    const auto& [V_aaaa, V_aabb, V_bbbb] = hamiltonian.get_two_body_integrals();

    // Check that the orbitals are consistent with the Hamiltonian
    if (!configurations.empty()) {
      // Get the orbital count from the configuration's string representation
      size_t config_num_orbitals = configurations[0].get_orbital_capacity();
      if (config_num_orbitals < num_molecular_orbitals) {
        throw std::runtime_error(
            "Configuration orbital capacity does not match Hamiltonian active "
            "space.");
      }
    }

    // Convert data::Configuration to wfn_type
    std::vector<wfn_type> dets;
    dets.reserve(configurations.size());
    for (const auto& config : configurations) {
      dets.emplace_back(config.to_bitset<N>());
    }

    if (dets.empty()) {
      throw std::runtime_error("Configuration basis cannot be empty");
    }

    // Create Hamiltonian Generator
    generator_t ham_gen(macis::matrix_span<double>(
                            const_cast<double*>(T_a.data()),
                            num_molecular_orbitals, num_molecular_orbitals),
                        macis::rank4_span<double>(
                            const_cast<double*>(V_aaaa.data()),
                            num_molecular_orbitals, num_molecular_orbitals,
                            num_molecular_orbitals, num_molecular_orbitals));

    // Perform projected CI diagonalization
    std::vector<double> C_pmc;
    double E_pmc = 0.0;
    if (dets.size() == 1) {
      E_pmc = ham_gen.matrix_element(dets[0], dets[0]);
      C_pmc = {1.0};
    } else if (dets.size() < iterative_solver_dimension_cutoff) {
      auto H = macis::make_csr_hamiltonian<int32_t>(dets.begin(), dets.end(),
                                                    ham_gen, H_thresh);
      std::vector<double> H_dense(dets.size() * dets.size(), 0.0);
      std::vector<double> evals(dets.size(), 0.0);

      sparsexx::convert_to_dense(H, H_dense.data(), dets.size());
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Upper, dets.size(),
                   H_dense.data(), dets.size(), evals.data());

      E_pmc = evals[0];

      C_pmc.resize(dets.size());
      std::copy(H_dense.begin(), H_dense.begin() + dets.size(), C_pmc.begin());
    } else {
      E_pmc = macis::selected_ci_diag<int64_t, wfn_type>(
          dets.begin(), dets.end(), ham_gen, h_el_tol, davidson_max_m,
          davidson_res_tol, C_pmc);
    }

    // Copy-back data to return struct
    Eigen::VectorXd C_vector(C_pmc.size());
    std::vector<data::Configuration> dets_configs;
    for (auto det : dets) {
      // Convert macis::wfn_t to data::Configuration
      dets_configs.emplace_back(det, num_molecular_orbitals);
    }
    std::copy(C_pmc.begin(), C_pmc.end(), C_vector.data());

    data::Wavefunction wfn = [&]() {
      if (settings_.get<bool>("calculate_one_rdm") ||
          settings_.get<bool>("calculate_two_rdm")) {
        // Calculate RDMs from CI coefficients
        std::vector<double> active_ordm(
            num_molecular_orbitals * num_molecular_orbitals, 0.0);
        std::vector<double> active_trdm(
            num_molecular_orbitals * num_molecular_orbitals *
                num_molecular_orbitals * num_molecular_orbitals,
            0.0);

        // Calculate RDMs using the Hamiltonian generator
        ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                          C_pmc.data(),
                          macis::matrix_span<double>(active_ordm.data(),
                                                     num_molecular_orbitals,
                                                     num_molecular_orbitals),
                          macis::rank4_span<double>(
                              active_trdm.data(), num_molecular_orbitals,
                              num_molecular_orbitals, num_molecular_orbitals,
                              num_molecular_orbitals));

        // Convert to Eigen format
        Eigen::MatrixXd one_rdm = Eigen::Map<Eigen::MatrixXd>(
            active_ordm.data(), num_molecular_orbitals, num_molecular_orbitals);
        Eigen::VectorXd two_rdm = Eigen::Map<Eigen::VectorXd>(
            active_trdm.data(),
            num_molecular_orbitals * num_molecular_orbitals *
                num_molecular_orbitals * num_molecular_orbitals);

        // Create wavefunction with RDMs
        return data::Wavefunction(
            std::make_unique<data::SciWavefunctionContainer>(
                std::move(C_vector), std::move(dets_configs),
                hamiltonian.get_orbitals(), std::move(one_rdm),
                std::move(two_rdm)));
      } else {
        // Create wavefunction without RDMs
        return data::Wavefunction(
            std::make_unique<data::SciWavefunctionContainer>(
                std::move(C_vector), std::move(dets_configs),
                hamiltonian.get_orbitals()));
      }
    }();

    // Add core energy to get total energy
    double final_energy = E_pmc + hamiltonian.get_core_energy();

    return std::make_pair<double, data::Wavefunction>(std::move(final_energy),
                                                      std::move(wfn));
  }
};

std::pair<double, std::shared_ptr<data::Wavefunction>> MacisPmc::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    const std::vector<data::Configuration>& configurations) const {
  const auto& orbitals = hamiltonian->get_orbitals();
  std::vector<size_t> active_indices =
      orbitals->get_active_space_indices().first;
  auto result = dispatch_by_norb<pmc_helper>(
      active_indices.size(), *hamiltonian, configurations, *_settings);
  return std::make_pair(result.first, std::make_shared<data::Wavefunction>(
                                          std::move(result.second)));
}

}  // namespace qdk::chemistry::algorithms::microsoft
