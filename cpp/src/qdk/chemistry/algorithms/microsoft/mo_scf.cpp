// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "mo_scf.hpp"

#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/scf/scf_solver.h>

#include <algorithm>
#include <array>
#include <blas.hh>
#include <cctype>
#include <macis/util/transform.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <unordered_map>

#include "scf/src/eri/INCORE/incore.h"
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

std::pair<double, std::shared_ptr<data::Ansatz>> MoScfSolver::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    unsigned int n_active_alpha_electrons,
    unsigned int n_active_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();
  if (!hamiltonian || !hamiltonian->is_hermitian() ||
      !hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        "MO SCF requires a Hermitian Hamiltonian in a common restricted input "
        "basis. "
        "Spin-dependent input integrals are not supported.");
  }
  auto method = _settings->get<std::string>("method");
  std::transform(method.begin(), method.end(), method.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (method != "hf") {
    throw std::invalid_argument("MO-basis SCF supports Hartree-Fock only.");
  }
  const auto input_orbitals = hamiltonian->get_orbitals();
  const auto active = data::spin_channel_indices(
      input_orbitals->active_indices(), data::axes::alpha());
  if (active != data::spin_channel_indices(input_orbitals->active_indices(),
                                           data::axes::beta())) {
    throw std::invalid_argument(
        "MO SCF requires identical alpha and beta active spaces.");
  }
  const size_t n = active.size();
  if (n == 0 || n_active_alpha_electrons > n || n_active_beta_electrons > n) {
    throw std::invalid_argument(
        "Active electron counts must lie within a nonempty active space.");
  }

  utils::microsoft::initialize_backend();
  qcs::SCFConfig config;
  config.mpi = qcs::mpi_default_input();
  config.require_gradient = false;
  config.require_polarizability = false;
  config.exc.xc_name = "HF";
  config.density_init_method = qcs::DensityInitializationMethod::UserProvided;
  const auto scf_type = _settings->get<std::string>("scf_type");
  const bool open_shell = n_active_alpha_electrons != n_active_beta_electrons;
  const bool unrestricted =
      scf_type == "unrestricted" || (scf_type == "auto" && open_shell);
  config.scf_orbital_type =
      unrestricted ? qcs::SCFOrbitalType::Unrestricted
                   : (open_shell ? qcs::SCFOrbitalType::RestrictedOpenShell
                                 : qcs::SCFOrbitalType::Restricted);
  configure_scf_iterations(*_settings, config);
  const auto& [h1, h1_beta] = hamiltonian->get_one_body_integrals();
  const auto& [g, g_ab, g_bb] = hamiltonian->get_two_body_integrals();
  auto eri = std::make_shared<qcs::ERIINCORE>(config.scf_orbital_type, n, g,
                                              config.mpi);
  auto solver = qcs::SCF::make_mo_hf_solver(
      h1, eri, n_active_alpha_electrons, n_active_beta_electrons,
      hamiltonian->get_core_energy(), config);
  const auto& result = solver->run().result;

  // The engine's coefficient matrices are rotations in the input active basis.
  // Compose them with any AO coefficients, preserving the spectator columns.
  const size_t nmo = input_orbitals->get_num_molecular_orbitals();
  const bool model =
      bool(std::dynamic_pointer_cast<data::ModelOrbitals>(input_orbitals));
  const Eigen::MatrixXd original =
      model ? Eigen::MatrixXd(Eigen::MatrixXd::Identity(nmo, nmo))
            : input_orbitals->coefficients()->block(
                  {data::axes::alpha(), data::axes::alpha()});
  const size_t nbasis = original.rows();
  Eigen::MatrixXd active_coefficients(nbasis, n);
  for (size_t i = 0; i < n; ++i) {
    active_coefficients.col(i) = original.col(active[i]);
  }
  std::array<qcs::RowMajorMatrix, 2> rotations;
  std::array<Eigen::MatrixXd, 2> coefficients, one_body, inactive_fock;
  std::array<std::optional<Eigen::VectorXd>, 2> energies;
  const int nspin = unrestricted ? 2 : 1;
  for (int spin = 0; spin < nspin; ++spin) {
    rotations[spin] = solver->get_orbitals_matrix().middleRows(spin * n, n);
    const Eigen::MatrixXd rotation = rotations[spin];
    Eigen::MatrixXd rotated_active(nbasis, n);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               nbasis, n, n, 1.0, active_coefficients.data(), nbasis,
               rotation.data(), n, 0.0, rotated_active.data(), nbasis);
    coefficients[spin] = original;
    for (size_t i = 0; i < n; ++i) {
      coefficients[spin].col(active[i]) = rotated_active.col(i);
    }
    if (n == nmo) {
      energies[spin] = Eigen::VectorXd::Zero(nmo);
    }
    if (energies[spin]) {
      for (size_t i = 0; i < n; ++i) {
        (*energies[spin])(active[i]) = solver->get_eigenvalues()(spin, i);
      }
    }
    one_body[spin].resize(n, n);
    macis::two_index_transform(n, n, h1.data(), n, rotation.data(), n,
                               one_body[spin].data(), n);
    if (hamiltonian->has_inactive_fock_matrix()) {
      Eigen::MatrixXd full_rotation = Eigen::MatrixXd::Identity(nmo, nmo);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          full_rotation(active[i], active[j]) = rotation(i, j);
        }
      }
      const auto& [fock, fock_beta] = hamiltonian->get_inactive_fock_matrix();
      inactive_fock[spin].resize(nmo, nmo);
      macis::two_index_transform(nmo, nmo, fock.data(), nmo,
                                 full_rotation.data(), nmo,
                                 inactive_fock[spin].data(), nmo);
    }
  }
  std::optional<Eigen::MatrixXd> overlap;
  if (model) {
    overlap = Eigen::MatrixXd::Identity(nmo, nmo);
  } else if (input_orbitals->has_overlap_matrix()) {
    overlap = input_orbitals->get_overlap_matrix();
  }
  auto basis = input_orbitals->has_basis_set() ? input_orbitals->get_basis_set()
                                               : nullptr;
  using Coefficients = data::SymmetryBlockedTensor<2>;
  using Energies = data::SymmetryBlockedTensor<1>;
  const auto symmetry = std::make_shared<const data::SymmetryProduct>(
      data::SymmetryProduct({data::axes::spin(1, !unrestricted)}));
  const auto basis_symmetry =
      basis && basis->ao_symmetries() ? basis->ao_symmetries() : symmetry;
  const std::unordered_map<data::SymmetryLabel, size_t> basis_extents{
      {data::axes::alpha(), nbasis}, {data::axes::beta(), nbasis}};
  const std::unordered_map<data::SymmetryLabel, size_t> orbital_extents{
      {data::axes::alpha(), nmo}, {data::axes::beta(), nmo}};
  Coefficients::BlockMap coefficient_blocks;
  Energies::BlockMap energy_blocks;
  for (int spin = 0; spin < nspin; ++spin) {
    const auto label = spin == 0 ? data::axes::alpha() : data::axes::beta();
    coefficient_blocks[{label, label}] =
        std::make_shared<const Eigen::MatrixXd>(std::move(coefficients[spin]));
    if (energies[spin]) {
      energy_blocks[{label}] =
          std::make_shared<const Eigen::VectorXd>(std::move(*energies[spin]));
    }
  }
  auto coefficient_tensor = std::make_shared<const Coefficients>(
      Coefficients::SymmetriesArray{basis_symmetry, symmetry},
      Coefficients::ExtentsArray{basis_extents, orbital_extents},
      std::move(coefficient_blocks));
  std::shared_ptr<const Energies> energy_tensor;
  if (!energy_blocks.empty()) {
    energy_tensor = std::make_shared<const Energies>(
        Energies::SymmetriesArray{symmetry},
        Energies::ExtentsArray{orbital_extents}, std::move(energy_blocks));
  }
  std::array<std::shared_ptr<const data::SymmetryBlockedIndexSet>, 2> spaces;
  const std::array input_spaces{input_orbitals->active_indices(),
                                input_orbitals->inactive_indices()};
  for (size_t space = 0; space < spaces.size(); ++space) {
    std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> indices;
    for (const auto& spin : {data::axes::alpha(), data::axes::beta()}) {
      const auto values = data::spin_channel_indices(input_spaces[space], spin);
      indices[spin] = std::vector<std::uint32_t>(values.begin(), values.end());
    }
    spaces[space] = std::make_shared<const data::SymmetryBlockedIndexSet>(
        symmetry, orbital_extents, std::move(indices));
  }
  auto orbitals = std::make_shared<data::Orbitals>(
      std::move(coefficient_tensor), std::move(energy_tensor), overlap, basis,
      spaces[0], spaces[1]);

  qcs::MOERI transform(eri);
  Eigen::VectorXd g_aa(g.size()), g_alpha_beta, g_beta_beta;
  transform.compute(n, n, rotations[0].data(), g_aa.data());
  std::unique_ptr<data::HamiltonianContainer> container;
  if (unrestricted) {
    g_alpha_beta.resize(g.size());
    g_beta_beta.resize(g.size());
    // MOERI emits column-major output; reverse spin pairs for QDK's row-major
    // layout.
    transform.compute(n, n, rotations[1].data(), rotations[1].data(),
                      rotations[0].data(), rotations[0].data(),
                      g_alpha_beta.data());
    transform.compute(n, n, rotations[1].data(), g_beta_beta.data());
    container = std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
        one_body[0], one_body[1], g_aa, g_alpha_beta, g_beta_beta, orbitals,
        hamiltonian->get_core_energy(), inactive_fock[0], inactive_fock[1]);
  } else {
    container = std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
        one_body[0], g_aa, orbitals, hamiltonian->get_core_energy(),
        inactive_fock[0]);
  }
  auto rotated_hamiltonian =
      std::make_shared<data::Hamiltonian>(std::move(container));
  auto determinant = data::Configuration::canonical_hf_configuration(
      n_active_alpha_electrons, n_active_beta_electrons, n);
  auto wavefunction = std::make_shared<data::Wavefunction>(
      std::make_unique<data::StateVectorContainer>(determinant, orbitals,
                                                   "electrons"));
  return {result.scf_total_energy,
          std::make_shared<data::Ansatz>(rotated_hamiltonian, wavefunction)};
}

}  // namespace qdk::chemistry::algorithms::microsoft
