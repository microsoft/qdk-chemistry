// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_asci.hpp"

#include <macis/asci/determinant_search.hpp>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/hamiltonian_generator/residue_arrays.hpp>
#include <macis/hamiltonian_generator/dynamic_bit_masking.hpp>
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
    using sdl_gen_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;
    using ra_gen_t = macis::ResidueArrayHamiltonianGenerator<wfn_type>;
    using dbm_gen_t = macis::DynamicBitMaskHamiltonianGenerator<wfn_type>;

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

    QDK_LOGGER().debug(
        "MACIS ASCI helper prepared: norb={}, nalpha={}, nbeta={}, "
        "ntdets_max={}, "
        "ntdets_min={}, max_refine_iter={}, grow_factor={}, rv_prune_tol={}",
        num_molecular_orbitals, nalpha, nbeta, asci_settings.ntdets_max,
        asci_settings.ntdets_min, asci_settings.max_refine_iter,
        asci_settings.grow_factor, asci_settings.rv_prune_tol);

    macis::matrix_span<double> T_span(
        const_cast<double*>(T_a.data()),
        num_molecular_orbitals, num_molecular_orbitals);
    macis::rank4_span<double> V_span(
        const_cast<double*>(V_aaaa.data()),
        num_molecular_orbitals, num_molecular_orbitals,
        num_molecular_orbitals, num_molecular_orbitals);

    // Select Hamiltonian generator based on h_build_algo
    QDK_LOGGER().debug("Constructing MACIS Hamiltonian generator.");
    std::unique_ptr<macis::HamiltonianGenerator<wfn_type>> ham_gen_ptr;
    const auto& algo = asci_settings.h_build_algo;
    if (algo == "residue_arrays") {
      ham_gen_ptr = std::make_unique<ra_gen_t>(T_span, V_span);
    } else if (algo == "dynamic_bit_masking") {
      ham_gen_ptr = std::make_unique<dbm_gen_t>(T_span, V_span);
    } else if (algo == "dynamic_bit_masking_8") {
      auto p = std::make_unique<dbm_gen_t>(T_span, V_span);
      p->set_num_masks(8);
      ham_gen_ptr = std::move(p);
    } else {
      ham_gen_ptr = std::make_unique<sdl_gen_t>(T_span, V_span);
    }
    auto& ham_gen = *ham_gen_ptr;
    QDK_LOGGER().debug("MACIS Hamiltonian generator constructed.");

    std::vector<double> C_casci;
    std::vector<wfn_type> dets;
    double E_casci = 0.0;

    size_t fci_dimension =
        qdk::chemistry::utils::microsoft::binomial_coefficient(
            num_molecular_orbitals, nalpha) *
        qdk::chemistry::utils::microsoft::binomial_coefficient(
            num_molecular_orbitals, nbeta);

    QDK_LOGGER().debug("MACIS ASCI FCI dimension estimate: {}", fci_dimension);

    if (asci_settings.ntdets_max > fci_dimension) {
      QDK_LOGGER().info(
          "Requested number of determinants ({}) exceeds FCI dimension ({}).",
          asci_settings.ntdets_max, fci_dimension);
      // IS SDL the best idea for FCI?
      E_casci = macis::CASRDMFunctor<sdl_gen_t>::rdms(
          mcscf_settings, macis::NumOrbital(num_molecular_orbitals), nalpha,
          nbeta, const_cast<double*>(T_a.data()),
          const_cast<double*>(V_aaaa.data()), nullptr, nullptr, C_casci);
      // Generate determinant basis for RDM calculation
      dets = macis::generate_hilbert_space<typename sdl_gen_t::full_det_t>(
          num_molecular_orbitals, nalpha, nbeta);
    } else {
      // HF Guess
      dets = {macis::wavefunction_traits<wfn_type>::canonical_hf_determinant(
          nalpha, nbeta)};
      E_casci = ham_gen.matrix_element(dets[0], dets[0]);
      C_casci = {1.0};

      // Growth phase
      QDK_LOGGER().debug("Starting MACIS ASCI growth phase.");
      std::tie(E_casci, dets, C_casci) = macis::asci_grow<N, int64_t>(
          asci_settings, mcscf_settings, E_casci, std::move(dets),
          std::move(C_casci), ham_gen,
          num_molecular_orbitals MACIS_MPI_CODE(, MPI_COMM_WORLD));
      QDK_LOGGER().debug(
          "Completed MACIS ASCI growth phase with {} determinants.",
          dets.size());

      // Refinement phase
      if (asci_settings.max_refine_iter) {
        QDK_LOGGER().debug("Starting MACIS ASCI refinement phase.");
        std::tie(E_casci, dets, C_casci) = macis::asci_refine<N, int64_t>(
            asci_settings, mcscf_settings, E_casci, std::move(dets),
            std::move(C_casci), ham_gen,
            num_molecular_orbitals MACIS_MPI_CODE(, MPI_COMM_WORLD));
        QDK_LOGGER().debug(
            "Completed MACIS ASCI refinement phase with {} determinants.",
            dets.size());
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

  QDK_LOGGER().debug("MacisAsci::_run_impl starting: nalpha={}, nbeta={}",
                     nalpha, nbeta);

  const auto& orbitals = hamiltonian->get_orbitals();
  if (hamiltonian->is_unrestricted()) {
    throw std::runtime_error(
        "MacisAsci does not support unrestricted orbitals. "
        "Only restricted orbitals are supported.");
  }
  const auto& [active_indices, active_indices_beta] =
      orbitals->get_active_space_indices();
  // check that alpha and beta active space indices are the same
  if (active_indices != active_indices_beta) {
    throw std::runtime_error(
        "MacisAsci only supports identical alpha and beta active "
        "space indices.");
  }

  QDK_LOGGER().debug("MacisAsci::_run_impl dispatching for {} active orbitals.",
                     active_indices.size());

  auto result = dispatch_by_norb<asci_helper>(
      active_indices.size(), *hamiltonian, *_settings, nalpha, nbeta);
  return std::make_pair(result.first, std::make_shared<data::Wavefunction>(
                                          std::move(result.second)));
}

}  // namespace qdk::chemistry::algorithms::microsoft
