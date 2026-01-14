// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_asci.hpp"

#include <macis/asci/determinant_search.hpp>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/mcscf/cas.hpp>
#include <macis/util/mpi.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/utils/logger.hpp>

// Local implementation details
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Helper struct for CASCI calculation dispatch
 */
struct asci_helper {
  using return_type = std::pair<double, data::Wavefunction>;

  /**
   * @brief Template implementation of CASCI calculation
   * @tparam N Number of bits for wavefunction representation
   * @param hamiltonian Hamiltonian object containing molecular integrals
   * @param settings_ Settings object storing asci specific settings
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @return std::pair containing energy and wavefunction
   */
  template <size_t N>
  static return_type impl(const data::Hamiltonian& hamiltonian,
                          const data::Settings& settings_, unsigned int nalpha,
                          unsigned int nbeta) {
    QDK_LOG_TRACE_ENTERING();

    using wfn_type = macis::wfn_t<N>;
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

    auto orbitals = hamiltonian.get_orbitals();
    const auto& [active_indices, active_indices_beta] =
        orbitals->get_active_space_indices();
    // check that alpha and beta active space indices are the same
    if (active_indices != active_indices_beta) {
      throw std::runtime_error(
          "MacisAsci only supports identical alpha and beta active "
          "space indices.");
    }

    const size_t num_molecular_orbitals = active_indices.size();

    const auto& [T_a, T_b] = hamiltonian.get_one_body_integrals();
    const auto& [V_aaaa, V_aabb, V_bbbb] = hamiltonian.get_two_body_integrals();

    // get settings
    macis::MCSCFSettings mcscf_settings = get_mcscf_settings_(settings_);
    macis::ASCISettings asci_settings = get_asci_settings_(settings_);

    std::vector<double> C_casci;
    std::vector<wfn_type> dets;
    double E_casci = 0.0;

    generator_t ham_gen(macis::matrix_span<double>(
                            const_cast<double*>(T_a.data()),
                            num_molecular_orbitals, num_molecular_orbitals),
                        macis::rank4_span<double>(
                            const_cast<double*>(V_aaaa.data()),
                            num_molecular_orbitals, num_molecular_orbitals,
                            num_molecular_orbitals, num_molecular_orbitals));

    size_t fci_dimension =
        qdk::chemistry::utils::microsoft::binomial_coefficient(
            num_molecular_orbitals, nalpha) *
        qdk::chemistry::utils::microsoft::binomial_coefficient(
            num_molecular_orbitals, nbeta);

    if (asci_settings.ntdets_max > fci_dimension) {
      QDK_LOGGER().info(
          "Requested number of determinants ({}) exceeds FCI dimension ({}).",
          asci_settings.ntdets_max, fci_dimension);
      E_casci = macis::CASRDMFunctor<generator_t>::rdms(
          mcscf_settings, macis::NumOrbital(num_molecular_orbitals), nalpha,
          nbeta, const_cast<double*>(T_a.data()),
          const_cast<double*>(V_aaaa.data()), nullptr, nullptr, C_casci);
      // Generate determinant basis for RDM calculation
      dets = macis::generate_hilbert_space<typename generator_t::full_det_t>(
          num_molecular_orbitals, nalpha, nbeta);
    } else {
      // HF Guess
      dets = {macis::wavefunction_traits<wfn_type>::canonical_hf_determinant(
          nalpha, nbeta)};
      E_casci = ham_gen.matrix_element(dets[0], dets[0]);
      C_casci = {1.0};

      // Growth phase
      std::tie(E_casci, dets, C_casci) = macis::asci_grow<N, int64_t>(
          asci_settings, mcscf_settings, E_casci, std::move(dets),
          std::move(C_casci), ham_gen,
          num_molecular_orbitals MACIS_MPI_CODE(, MPI_COMM_WORLD));

      // Refinement phase
      if (asci_settings.max_refine_iter) {
        std::tie(E_casci, dets, C_casci) = macis::asci_refine<N, int64_t>(
            asci_settings, mcscf_settings, E_casci, std::move(dets),
            std::move(C_casci), ham_gen,
            num_molecular_orbitals MACIS_MPI_CODE(, MPI_COMM_WORLD));
      }
    }

    // Build wavefunction with unified builder (supports spin-dependent RDMs
    // when requested)
    data::Wavefunction wfn = build_wavefunction<data::SciWavefunctionContainer>(
        settings_, hamiltonian, ham_gen, num_molecular_orbitals, C_casci, dets);

    // Add core energy to get total energy
    double final_energy = E_casci + hamiltonian.get_core_energy();

    return std::make_pair<double, data::Wavefunction>(std::move(final_energy),
                                                      std::move(wfn));
  }
};

std::pair<double, std::shared_ptr<data::Wavefunction>> MacisAsci::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian, unsigned int nalpha,
    unsigned int nbeta) const {
  QDK_LOG_TRACE_ENTERING();

  const auto& orbitals = hamiltonian->get_orbitals();
  const auto& [active_indices, active_indices_beta] =
      orbitals->get_active_space_indices();
  // check that alpha and beta active space indices are the same
  if (active_indices != active_indices_beta) {
    throw std::runtime_error(
        "MacisAsci only supports identical alpha and beta active "
        "space indices.");
  }

  auto result = dispatch_by_norb<asci_helper>(
      active_indices.size(), *hamiltonian, *_settings, nalpha, nbeta);
  return std::make_pair(result.first, std::make_shared<data::Wavefunction>(
                                          std::move(result.second)));
}

}  // namespace qdk::chemistry::algorithms::microsoft
