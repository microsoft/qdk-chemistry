// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_cas.hpp"

#include <macis/asci/determinant_search.hpp>
#include <macis/asci/grow.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/mcscf/cas.hpp>
#include <macis/util/mpi.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Helper struct for CASCI calculation dispatch
 */
struct cas_helper {
  using return_type = std::pair<double, data::Wavefunction>;

  /**
   * @brief Template implementation of CASCI calculation
   * @tparam N Number of bits for wavefunction representation
   * @param hamiltonian Hamiltonian object containing molecular integrals
   * @param settings_ Settings object storing all CAS specific settings
   * @param nalpha Number of active alpha electrons
   * @param nbeta Number of active beta electrons
   * @return std::pair containing energy and wavefunction
   */
  template <size_t N>
  static return_type impl(const data::Hamiltonian& hamiltonian,
                          const data::Settings& settings_, unsigned int nalpha,
                          unsigned int nbeta) {
    using wfn_type = macis::wfn_t<N>;
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

    auto orbitals = hamiltonian.get_orbitals();
    std::vector<size_t> active_indices = detail::get_active_indices(*orbitals);
    const size_t nmo = active_indices.size();

    const auto& T = hamiltonian.get_one_body_integrals();
    const auto& V = hamiltonian.get_two_body_integrals();

    // get settings
    macis::MCSCFSettings mcscf_settings = get_mcscf_settings_(settings_);
    macis::ASCISettings asci_settings = get_asci_settings_(settings_);

    std::vector<double> C_casci;
    std::vector<wfn_type> dets;
    double E_casci = 0.0;

    E_casci = macis::CASRDMFunctor<generator_t>::rdms(
        mcscf_settings, macis::NumOrbital(nmo), nalpha, nbeta,
        const_cast<double*>(T.data()), const_cast<double*>(V.data()), nullptr,
        nullptr, C_casci);
    // Generate determinant basis for RDM calculation
    dets = macis::generate_hilbert_space<typename generator_t::full_det_t>(
        nmo, nalpha, nbeta);

    // Copy-back data to return struct
    Eigen::VectorXd C_vector(C_casci.size());
    std::copy(C_casci.begin(), C_casci.end(), C_vector.data());

    std::vector<data::Configuration> dets_configs;
    for (auto det : dets) {
      // Convert macis::wfn_t to data::Configuration
      dets_configs.emplace_back(det, nmo);
    }

    data::Wavefunction wfn = [&]() {
      if (settings_.get<bool>("calculate_one_rdm") ||
          settings_.get<bool>("calculate_two_rdm")) {
        // Calculate RDMs from CI coefficients
        std::vector<double> active_ordm(nmo * nmo, 0.0);
        std::vector<double> active_trdm(nmo * nmo * nmo * nmo, 0.0);

        // Calculate RDMs using the Hamiltonian generator
        generator_t ham_gen(
            macis::matrix_span<double>(const_cast<double*>(T.data()), nmo, nmo),
            macis::rank4_span<double>(const_cast<double*>(V.data()), nmo, nmo,
                                      nmo, nmo));

        ham_gen.form_rdms(
            dets.begin(), dets.end(), dets.begin(), dets.end(), C_casci.data(),
            macis::matrix_span<double>(active_ordm.data(), nmo, nmo),
            macis::rank4_span<double>(active_trdm.data(), nmo, nmo, nmo, nmo));

        // Convert to Eigen format
        Eigen::MatrixXd one_rdm =
            Eigen::Map<Eigen::MatrixXd>(active_ordm.data(), nmo, nmo);
        Eigen::VectorXd two_rdm = Eigen::Map<Eigen::VectorXd>(
            active_trdm.data(), nmo * nmo * nmo * nmo);

        // Create wavefunction with RDMs
        return data::Wavefunction(
            std::make_unique<data::CasWavefunctionContainer>(
                std::move(C_vector), std::move(dets_configs),
                hamiltonian.get_orbitals(), std::move(one_rdm),
                std::move(two_rdm)));
      } else {
        // Create wavefunction without RDMs
        return data::Wavefunction(
            std::make_unique<data::CasWavefunctionContainer>(
                std::move(C_vector), std::move(dets_configs),
                hamiltonian.get_orbitals()));
      }
    }();

    // Add core energy to get total energy
    double final_energy = E_casci + hamiltonian.get_core_energy();

    return std::make_pair<double, data::Wavefunction>(std::move(final_energy),
                                                      std::move(wfn));
  }
};

std::pair<double, std::shared_ptr<data::Wavefunction>> MacisCas::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    unsigned int n_active_alpha_electrons,
    unsigned int n_active_beta_electrons) const {
  const auto& orbitals = hamiltonian->get_orbitals();
  std::vector<size_t> active_indices = detail::get_active_indices(*orbitals);
  auto result = dispatch_by_norb<cas_helper>(
      active_indices.size(), *hamiltonian, *_settings, n_active_alpha_electrons,
      n_active_beta_electrons);
  return std::make_pair(result.first, std::make_shared<data::Wavefunction>(
                                          std::move(result.second)));
}

}  // namespace qdk::chemistry::algorithms::microsoft
