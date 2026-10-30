// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_cas.hpp"

#include <macis/fci/fci_hamiltonian.hpp>
#include <macis/fci/fci_rdms.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/sd_operations.hpp>
#include <numeric>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas_state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

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
    QDK_LOG_TRACE_ENTERING();

    using wfn_type = macis::wfn_t<N>;
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

    auto orbitals = hamiltonian.get_orbitals();
    const auto active_ai = orbitals->active_indices();
    const auto active_indices =
        data::spin_channel_indices(active_ai, data::axes::alpha());
    const auto active_indices_beta =
        data::spin_channel_indices(active_ai, data::axes::beta());
    // check that alpha and beta active space indices are the same
    if (active_indices != active_indices_beta) {
      throw std::runtime_error(
          "MacisCas only supports identical alpha and beta active "
          "space indices.");
    }
    const size_t num_molecular_orbitals = active_indices.size();

    const auto& [T_a, T_b] = hamiltonian.get_one_body_integrals();
    Eigen::VectorXd V_storage;
    const auto& V_aaaa =
        macis_two_body_integrals(hamiltonian, "MacisCas", V_storage);

    // +1 (-1) restricts the CI vector to C^T = C (C^T = -C), the states of
    // even (odd) total spin
    const auto symmetry = settings_.get<std::string>("spin_flip_symmetry");
    const int spin_flip = symmetry == "even" ? 1 : symmetry == "odd" ? -1 : 0;
    if (spin_flip != 0 and nalpha != nbeta) {
      throw std::invalid_argument(
          "MacisCas: spin_flip_symmetry requires equal numbers of active "
          "alpha and beta electrons.");
    }

    // Direct (string-driven) FCI: H is applied from the integrals, never
    // stored over determinants
    macis::FCIHamiltonian fci_ham(num_molecular_orbitals, nalpha, nbeta,
                                  T_a.data(), num_molecular_orbitals,
                                  V_aaaa.data(), num_molecular_orbitals);
    const size_t ndets = fci_ham.size();

    std::vector<double> C_casci;
    double E_casci = 0.0;
    if (ndets <= static_cast<size_t>(settings_.get<int64_t>(
                     "iterative_solver_dimension_cutoff"))) {
      std::vector<size_t> all_dets(ndets);
      std::iota(all_dets.begin(), all_dets.end(), size_t{0});
      E_casci = macis::fci_subspace_ground_state(fci_ham, all_dets, C_casci,
                                                 spin_flip);
    } else {
      E_casci = macis::fci_ground_state(
          fci_ham, settings_.get<int64_t>("max_solver_iterations"),
          settings_.get<double>("ci_residual_tolerance"), C_casci, spin_flip);
    }

    data::Wavefunction wfn = build_wavefunction(
        settings_, num_molecular_orbitals, C_casci,
        [&](auto... rdms) {
          macis::fci_rdms_spin_dep(fci_ham.alpha_strings(),
                                   fci_ham.beta_strings(), C_casci.data(),
                                   rdms..., spin_flip);
        },
        [&](auto&&... entropies) {
          // Determinant-list entropies, determinants in CI-vector order
          auto dets = macis::generate_hilbert_space<wfn_type>(
              num_molecular_orbitals, nalpha, nbeta);
          generator_t ham_gen(
              macis::matrix_span<double>(const_cast<double*>(T_a.data()),
                                         num_molecular_orbitals,
                                         num_molecular_orbitals),
              macis::rank4_span<double>(
                  const_cast<double*>(V_aaaa.data()), num_molecular_orbitals,
                  num_molecular_orbitals, num_molecular_orbitals,
                  num_molecular_orbitals));
          ham_gen.form_entropies(
              dets.begin(), dets.end(), dets.begin(), dets.end(),
              C_casci.data(), std::forward<decltype(entropies)>(entropies)...);
        },
        [&](Eigen::VectorXd coeffs, const auto& one_aa, const auto& one_bb,
            const auto& two_aaaa, const auto& two_aabb, const auto& two_bbbb,
            const data::OrbitalEntropies& entropies) {
          return std::make_unique<data::CasStateVectorContainer>(
              std::move(coeffs), nalpha, nbeta, orbitals, std::nullopt, one_aa,
              one_bb, std::nullopt, two_aaaa, two_aabb, two_bbbb, "electrons",
              entropies, data::WavefunctionType::SelfDual);
        });

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
  QDK_LOG_TRACE_ENTERING();

  const auto& orbitals = hamiltonian->get_orbitals();
  if (hamiltonian->is_unrestricted()) {
    throw std::runtime_error(
        "MacisCas does not support unrestricted orbitals. "
        "Only restricted orbitals are supported.");
  }
  const auto active_ai = orbitals->active_indices();
  const auto active_indices =
      data::spin_channel_indices(active_ai, data::axes::alpha());
  const auto active_indices_beta =
      data::spin_channel_indices(active_ai, data::axes::beta());
  // check that alpha and beta active space indices are the same
  if (active_indices != active_indices_beta) {
    throw std::runtime_error(
        "MacisCas only supports identical alpha and beta active "
        "space indices.");
  }
  auto result = dispatch_by_norb<cas_helper>(
      active_indices.size(), *hamiltonian, *_settings, n_active_alpha_electrons,
      n_active_beta_electrons);
  return std::make_pair(result.first, std::make_shared<data::Wavefunction>(
                                          std::move(result.second)));
}

}  // namespace qdk::chemistry::algorithms::microsoft
