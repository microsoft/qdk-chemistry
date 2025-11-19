// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "hamiltonian.hpp"

// STL Headers
#include <set>

// MACIS Headers
#include <macis/mcscf/fock_matrices.hpp>

// QDK/Chemistry SCF headers
#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/util/int1e.h>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

std::shared_ptr<data::Hamiltonian> HamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals) const {
  // Sanity Checks
  if (not orbitals->is_restricted()) {
    throw std::runtime_error(
        "Hamiltonian construction only works for restricted orbitals");
  }

  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  auto basis_set = orbitals->get_basis_set();
  const auto& [Ca, Cb] = orbitals->get_coefficients();
  const size_t num_basis_funcs = basis_set->get_num_basis_functions();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Determine whether we're doing an active space calculation
  auto indices = orbitals->has_active_space()
                     ? orbitals->get_active_space_indices().first
                     : std::vector<size_t>();

  bool active_space_is_contiguous = true;
  const size_t nactive =
      indices.size() ? indices.size() : num_molecular_orbitals;
  if (indices.size()) {
    // Sanity checks on the active orbitals

    // Cannot contain more than the total number of MOs
    if (indices.size() > num_molecular_orbitals) {
      throw std::runtime_error(
          "Number of requested active orbitals exceeds total number of MOs");
    }

    // Make sure that the indices are within bounds
    for (const auto& idx : indices) {
      if (static_cast<size_t>(idx) >= num_molecular_orbitals) {
        throw std::runtime_error("Active orbital index out of bounds: " +
                                 std::to_string(idx));
      }
    }

    // Make sure that the indices are unique
    std::set<size_t> unique_indices(indices.begin(), indices.end());
    if (unique_indices.size() != indices.size()) {
      throw std::runtime_error("Active orbital indices must be unique");
    }

    // Make sure that the indices are sorted
    std::vector<size_t> sorted_indices(indices.begin(), indices.end());
    std::sort(sorted_indices.begin(), sorted_indices.end());
    if (indices != sorted_indices) {
      throw std::runtime_error("Active orbital indices must be sorted");
    }

    // Determine if active space is contiguous
    for (size_t i = 0; i < indices.size() - 1; ++i) {
      if (indices[i + 1] - indices[i] != 1) {
        active_space_is_contiguous = false;
        break;
      }
    }
  }
  // Create internal Molecule
  auto structure = basis_set->get_structure();
  auto mol = utils::microsoft::convert_to_molecule(*structure, 0, 1);

  // Create internal BasisSet
  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  // Create dummy SCFConfig
  auto scf_config = std::make_unique<qcs::SCFConfig>();

  // Use the default MPI configuration (fallback to serial if MPI not enabled)
  scf_config->mpi = qcs::mpi_default_input();
  scf_config->require_gradient = false;
  scf_config->basis = internal_basis_set->name;
  scf_config->cartesian = !internal_basis_set->pure;
  scf_config->unrestricted = false;
  // TODO: Handle unrestricted, workitems: 41317

  // Set ERI method based on settings
  std::string method_name = _settings->get<std::string>("eri_method");
  if (!method_name.compare("incore")) {
    scf_config->eri.method = qcs::ERIMethod::Incore;
    scf_config->k_eri.method = qcs::ERIMethod::Incore;
  } else if (!method_name.compare("direct")) {
    scf_config->eri.method = qcs::ERIMethod::Libint2Direct;
    scf_config->k_eri.method = qcs::ERIMethod::Libint2Direct;
  } else {
    throw std::runtime_error("Unsupported ERI method '" + method_name +
                             "'. Only CPU ERI methods are supported now");
  }

  // Create Integral Instance
  auto eri = qcs::ERIMultiplexer::create(*internal_basis_set, *scf_config, 0.0);
  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      internal_basis_set.get(), mol.get(), scf_config->mpi);

  // Compute Core Hamiltonian in AO basis
  Eigen::MatrixXd T_full(num_basis_funcs, num_basis_funcs),
      V_full(num_basis_funcs, num_basis_funcs);
  int1e->kinetic_integral(T_full.data());
  int1e->nuclear_integral(V_full.data());
  Eigen::MatrixXd H_full = T_full + V_full;

  Eigen::MatrixXd C_active(num_basis_funcs, nactive);
  if (indices.empty()) {
    // If no active orbitals are specified, use all orbitals
    C_active = Ca;
  } else if (active_space_is_contiguous) {
    C_active = Ca.block(0, indices.front(), num_basis_funcs, nactive);
  } else {
    for (auto i = 0; i < nactive; i++) {
      C_active.col(i) = Ca.col(indices[i]);
    }
  }
  // Transform ERIs into active MO basis
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C_rm =
      C_active;
  Eigen::VectorXd moeri_active(nactive * nactive * nactive * nactive);
  qcs::MOERI moeri_c(eri);
  moeri_c.compute(num_basis_funcs, nactive, C_rm.data(), moeri_active.data());

  // Early exit
  if (indices.empty()) {
    // If no active orbitals are specified, use all orbitals
    Eigen::MatrixXd H_active(nactive, nactive);
    H_active = Ca.transpose() * H_full * Ca;
    Eigen::MatrixXd dummy_fock =
        Eigen::MatrixXd::Zero(0, 0);  // No inactive orbitals
    return std::make_shared<data::Hamiltonian>(
        H_active, moeri_active, orbitals,
        structure->calculate_nuclear_repulsion_energy(), dummy_fock);
  }

  auto inactive_indices =
      orbitals->get_inactive_space_indices().first;  // TODO: restricted only
  auto active_indices =
      orbitals->get_active_space_indices().first;  // TODO: restricted only

  // all occupied orbitals specified as active
  if (inactive_indices.empty()) {
    Eigen::MatrixXd H_active(nactive, nactive);
    H_active = C_active.transpose() * H_full * C_active;
    Eigen::MatrixXd dummy_fock =
        Eigen::MatrixXd::Zero(0, 0);  // No inactive orbitals
    return std::make_shared<data::Hamiltonian>(
        H_active, moeri_active, orbitals,
        structure->calculate_nuclear_repulsion_energy(), dummy_fock);
  }

  // Determine whether the inactive space is contiguous
  bool inactive_space_is_contiguous = true;
  // Only test for contiguity if there is more than one inactive index
  for (size_t i = 0; i < inactive_indices.size() - 1; ++i) {
    if (inactive_indices[i + 1] - inactive_indices[i] != 1) {
      inactive_space_is_contiguous = false;
      break;
    }
  }

  // Compute the inactive density matrix
  Eigen::MatrixXd D_inactive =
      Eigen::MatrixXd::Zero(num_basis_funcs, num_basis_funcs);
  if (inactive_space_is_contiguous) {
    auto C_inactive = Ca.block(0, inactive_indices.front(), num_basis_funcs,
                               inactive_indices.size());
    D_inactive = C_inactive * C_inactive.transpose();
  } else {
    for (size_t i : inactive_indices) {
      D_inactive += Ca.col(i) * Ca.col(i).transpose();
    }
  }

  // Compute the two electron part of the inactive fock matrix
  Eigen::MatrixXd J_inactive_ao(num_basis_funcs, num_basis_funcs),
      K_inactive_ao(num_basis_funcs, num_basis_funcs);
  eri->build_JK(D_inactive.data(), J_inactive_ao.data(), K_inactive_ao.data(),
                1.0, 0.0, 0.0);
  Eigen::MatrixXd G_inactive_ao = 2 * J_inactive_ao - K_inactive_ao;

  // Compute the inactive Fock matrix
  Eigen::MatrixXd F_inactive_ao = G_inactive_ao + H_full;
  Eigen::MatrixXd F_inactive(num_molecular_orbitals, num_molecular_orbitals);
  F_inactive = Ca.transpose() * F_inactive_ao * Ca;

  // Compute active one body operator
  Eigen::MatrixXd H_active(nactive, nactive);
  for (size_t i = 0; i < nactive; i++) {
    for (size_t j = 0; j < nactive; j++) {
      H_active(i, j) = F_inactive(indices[i], indices[j]);
    }
  }

  // Compute the inactive energy
  double E_inactive = 0.0;
  Eigen::MatrixXd H_mo = Ca.transpose() * H_full * Ca;
  for (auto i : inactive_indices) {
    E_inactive += H_mo(i, i) + F_inactive(i, i);
  }

  // Return the Hamiltonian instance
  data::Hamiltonian H(
      H_active, moeri_active, orbitals,
      E_inactive + structure->calculate_nuclear_repulsion_energy(), F_inactive);
  return std::make_shared<data::Hamiltonian>(std::move(H));
}
}  // namespace qdk::chemistry::algorithms::microsoft
