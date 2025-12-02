// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "hamiltonian.hpp"

// STL Headers
#include <filesystem>
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

namespace detail {
/**
 * @brief Validate active orbital indices
 * @param indices The indices to validate
 * @param spin_label Label for error messages (e.g., "Alpha", "Beta")
 * @param num_molecular_orbitals Total number of molecular orbitals
 * @return true if the indices are contiguous, false otherwise
 */
bool validate_active_contiguous_indices(const std::vector<size_t>& indices,
                                        const std::string& spin_label,
                                        size_t num_molecular_orbitals) {
  if (indices.empty()) return true;

  // Cannot contain more than the total number of MOs
  if (indices.size() > num_molecular_orbitals) {
    throw std::runtime_error("Number of requested " + spin_label +
                             " active orbitals exceeds total number of MOs");
  }

  // Make sure that the indices are within bounds
  for (const auto& idx : indices) {
    if (static_cast<size_t>(idx) >= num_molecular_orbitals) {
      throw std::runtime_error(
          spin_label +
          " active orbital index out of bounds: " + std::to_string(idx));
    }
  }

  // Make sure that the indices are unique
  std::set<size_t> unique_indices(indices.begin(), indices.end());
  if (unique_indices.size() != indices.size()) {
    throw std::runtime_error(spin_label +
                             " active orbital indices must be unique");
  }

  // Make sure that the indices are sorted
  std::vector<size_t> sorted_indices(indices.begin(), indices.end());
  std::sort(sorted_indices.begin(), sorted_indices.end());
  if (indices != sorted_indices) {
    throw std::runtime_error(spin_label +
                             " active orbital indices must be sorted");
  }

  // Check if indices are contiguous
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    if (indices[i + 1] - indices[i] != 1) {
      return false;
    }
  }

  return true;
}
}  // namespace detail

std::shared_ptr<data::Hamiltonian> HamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals) const {
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  auto basis_set = orbitals->get_basis_set();
  const auto& [Ca, Cb] = orbitals->get_coefficients();
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Get alpha and beta active space indices
  auto active_space_indices = orbitals->get_active_space_indices();
  auto active_indices_alpha = active_space_indices.first;
  auto active_indices_beta = active_space_indices.second;

  if (orbitals->is_restricted() && active_indices_alpha.empty()) {
    throw std::runtime_error("Need to specify an active space.");
  } else if (orbitals->is_unrestricted() &&
             (active_indices_alpha.empty() || active_indices_beta.empty())) {
    throw std::runtime_error(
        "Need to specify an active space for alpha and beta.");
  }

  const size_t nactive_alpha = active_indices_alpha.size();
  const size_t nactive_beta = active_indices_beta.size();

  // Validate alpha active orbitals and check contiguity
  bool alpha_space_is_contiguous = detail::validate_active_contiguous_indices(
      active_indices_alpha, "Alpha", num_molecular_orbitals);

  // Validate beta active orbitals (if different from alpha) and check
  // contiguity
  bool beta_space_is_contiguous = true;
  if (active_indices_beta != active_indices_alpha) {
    beta_space_is_contiguous = detail::validate_active_contiguous_indices(
        active_indices_beta, "Beta", num_molecular_orbitals);
  } else {
    beta_space_is_contiguous = alpha_space_is_contiguous;
  }

  // Overall contiguity requires both alpha and beta to be contiguous
  bool active_space_is_contiguous =
      alpha_space_is_contiguous && beta_space_is_contiguous;

  // Ensure alpha and beta active spaces have the same size
  if (nactive_alpha != nactive_beta) {
    throw std::runtime_error(
        "Alpha and beta active spaces must have the same size. "
        "Alpha: " +
        std::to_string(nactive_alpha) +
        ", Beta: " + std::to_string(nactive_beta));
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
  Eigen::MatrixXd T_full(num_atomic_orbitals, num_atomic_orbitals),
      V_full(num_atomic_orbitals, num_atomic_orbitals);
  int1e->kinetic_integral(T_full.data());
  int1e->nuclear_integral(V_full.data());
  Eigen::MatrixXd H_full = T_full + V_full;

  // Build active coefficient matrices for alpha and beta (can have different
  // sizes)
  Eigen::MatrixXd Ca_active(num_atomic_orbitals, nactive_alpha);
  Eigen::MatrixXd Cb_active(num_atomic_orbitals, nactive_beta);

  if (alpha_space_is_contiguous) {
    // Contiguous alpha indices
    Ca_active = Ca.block(0, active_indices_alpha.front(), num_atomic_orbitals,
                         nactive_alpha);
  } else {
    // Non-contiguous alpha indices
    for (size_t i = 0; i < nactive_alpha; i++) {
      Ca_active.col(i) = Ca.col(active_indices_alpha[i]);
    }
  }

  if (beta_space_is_contiguous) {
    // Contiguous beta indices
    Cb_active = Cb.block(0, active_indices_beta.front(), num_atomic_orbitals,
                         nactive_beta);
  } else {
    // Non-contiguous beta indices
    for (size_t i = 0; i < nactive_beta; i++) {
      Cb_active.col(i) = Cb.col(active_indices_beta[i]);
    }
  }

  // Convert to row-major for MOERI
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ca_active_rm = Ca_active;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Cb_active_rm = Cb_active;

  // Initialize MOERI
  qcs::MOERI moeri_c(eri);

  // Determine SCF type from settings
  std::string scf_type = _settings->get<std::string>("scf_type");

  bool is_restricted_calc;
  if (scf_type == "restricted") {
    is_restricted_calc = true;
  } else if (scf_type == "unrestricted") {
    is_restricted_calc = false;
  } else {  // "auto"
    is_restricted_calc = (active_indices_alpha == active_indices_beta) &&
                         orbitals->is_restricted();
  }

  scf_config->unrestricted = !is_restricted_calc;

  // Compute integrals (same size for alpha and beta)
  const size_t nactive = nactive_alpha;

  // Declare MOERI vectors
  Eigen::VectorXd moeri_aaaa;
  Eigen::VectorXd moeri_aabb;
  Eigen::VectorXd moeri_bbbb;

  const size_t moeri_size = nactive * nactive * nactive * nactive;

  if (is_restricted_calc) {
    // Only allocate and compute (αα|αα) integrals - the others are identical
    moeri_aaaa.resize(moeri_size);
    moeri_c.compute(num_atomic_orbitals, nactive, Ca_active_rm.data(),
                    moeri_aaaa.data());
  } else {
    // Unrestricted case - allocate and compute all three types of integrals
    moeri_aaaa.resize(moeri_size);
    moeri_aabb.resize(moeri_size);
    moeri_bbbb.resize(moeri_size);

    // (αα|αα) integrals
    moeri_c.compute(num_atomic_orbitals, nactive,
                    Ca_active_rm.data(),  // 1st quarter: alpha
                    Ca_active_rm.data(),  // 2nd quarter: alpha
                    Ca_active_rm.data(),  // 3rd quarter: alpha
                    Ca_active_rm.data(),  // 4th quarter: alpha
                    moeri_aaaa.data());

    // (αα|ββ) integrals
    // Here, the C's are accessed like beta, beta, alpha, alpha, but results in
    // saving alpha, alpha, beta beta integrals that can be indexed "as usual"
    // in αα|ββ order.
    moeri_c.compute(num_atomic_orbitals, nactive,
                    Cb_active_rm.data(),  // 1st quarter: beta
                    Cb_active_rm.data(),  // 2nd quarter: beta
                    Ca_active_rm.data(),  // 3rd quarter: alpha
                    Ca_active_rm.data(),  // 4th quarter: alpha
                    moeri_aabb.data());

    // (ββ|ββ) integrals
    moeri_c.compute(num_atomic_orbitals, nactive,
                    Cb_active_rm.data(),  // 1st quarter: beta
                    Cb_active_rm.data(),  // 2nd quarter: beta
                    Cb_active_rm.data(),  // 3rd quarter: beta
                    Cb_active_rm.data(),  // 4th quarter: beta
                    moeri_bbbb.data());
  }

  // Get inactive space indices for both alpha and beta
  auto [inactive_indices_alpha, inactive_indices_beta] =
      orbitals->get_inactive_space_indices();

  // For restricted calculations, alpha and beta inactive spaces should be
  // identical
  if (orbitals->is_restricted() &&
      inactive_indices_alpha != inactive_indices_beta) {
    throw std::runtime_error(
        "For restricted orbitals, alpha and beta inactive spaces must be "
        "identical");
  }

  // all occupied orbitals specified as active
  if (inactive_indices_alpha.empty() && inactive_indices_beta.empty()) {
    if (is_restricted_calc) {
      // Use restricted constructor
      Eigen::MatrixXd H_active(nactive, nactive);
      H_active = Ca_active.transpose() * H_full * Ca_active;
      Eigen::MatrixXd dummy_fock = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          H_active, moeri_aaaa, orbitals,
          structure->calculate_nuclear_repulsion_energy(), dummy_fock);
    } else {
      // Use unrestricted constructor
      Eigen::MatrixXd H_active_alpha(nactive, nactive);
      Eigen::MatrixXd H_active_beta(nactive, nactive);
      H_active_alpha = Ca_active.transpose() * H_full * Ca_active;
      H_active_beta = Cb_active.transpose() * H_full * Cb_active;
      Eigen::MatrixXd dummy_fock_alpha = Eigen::MatrixXd::Zero(0, 0);
      Eigen::MatrixXd dummy_fock_beta = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          H_active_alpha, H_active_beta, moeri_aaaa, moeri_aabb, moeri_bbbb,
          orbitals, structure->calculate_nuclear_repulsion_energy(),
          dummy_fock_alpha, dummy_fock_beta);
    }
  }

  if (is_restricted_calc) {
    // Restricted case
    auto inactive_indices = inactive_indices_alpha;

    // Determine whether the inactive space is contiguous
    bool inactive_space_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices.size() - 1; ++i) {
      if (inactive_indices[i + 1] - inactive_indices[i] != 1) {
        inactive_space_is_contiguous = false;
        break;
      }
    }

    // Compute the inactive density matrix
    Eigen::MatrixXd D_inactive =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    if (inactive_space_is_contiguous) {
      auto C_inactive = Ca.block(0, inactive_indices.front(),
                                 num_atomic_orbitals, inactive_indices.size());
      D_inactive = C_inactive * C_inactive.transpose();
    } else {
      for (size_t i : inactive_indices) {
        D_inactive += Ca.col(i) * Ca.col(i).transpose();
      }
    }

    // Compute the two electron part of the inactive fock matrix
    Eigen::MatrixXd J_inactive_ao(num_atomic_orbitals, num_atomic_orbitals),
        K_inactive_ao(num_atomic_orbitals, num_atomic_orbitals);
    eri->build_JK(D_inactive.data(), J_inactive_ao.data(), K_inactive_ao.data(),
                  1.0, 0.0, 0.0);
    Eigen::MatrixXd G_inactive_ao = 2 * J_inactive_ao - K_inactive_ao;

    // Compute the inactive Fock matrix
    Eigen::MatrixXd F_inactive_ao = G_inactive_ao + H_full;
    Eigen::MatrixXd F_inactive(num_molecular_orbitals, num_molecular_orbitals);
    F_inactive = Ca.transpose() * F_inactive_ao * Ca;

    // Compute the inactive energy
    double E_inactive = 0.0;
    Eigen::MatrixXd H_mo = Ca.transpose() * H_full * Ca;
    for (auto i : inactive_indices) {
      E_inactive += H_mo(i, i) + F_inactive(i, i);
    }

    // Extract active space Hamiltonian
    Eigen::MatrixXd H_active(nactive, nactive);
    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active(i, j) =
            F_inactive(active_indices_alpha[i], active_indices_alpha[j]);
      }
    }

    return std::make_shared<data::Hamiltonian>(
        H_active, moeri_aaaa, orbitals,
        E_inactive + structure->calculate_nuclear_repulsion_energy(),
        F_inactive);

  } else {
    // Unrestricted case

    // Determine whether the alpha inactive space is contiguous
    bool alpha_inactive_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices_alpha.size() - 1; ++i) {
      if (inactive_indices_alpha[i + 1] - inactive_indices_alpha[i] != 1) {
        alpha_inactive_is_contiguous = false;
        break;
      }
    }

    // Determine whether the beta inactive space is contiguous
    bool beta_inactive_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices_beta.size() - 1; ++i) {
      if (inactive_indices_beta[i + 1] - inactive_indices_beta[i] != 1) {
        beta_inactive_is_contiguous = false;
        break;
      }
    }

    // Compute separate alpha and beta inactive density matrices
    Eigen::MatrixXd D_inactive_alpha =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    Eigen::MatrixXd D_inactive_beta =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);

    // Build alpha inactive density
    if (alpha_inactive_is_contiguous && !inactive_indices_alpha.empty()) {
      auto C_inactive_alpha =
          Ca.block(0, inactive_indices_alpha.front(), num_atomic_orbitals,
                   inactive_indices_alpha.size());
      D_inactive_alpha = C_inactive_alpha * C_inactive_alpha.transpose();
    } else {
      for (size_t i : inactive_indices_alpha) {
        D_inactive_alpha += Ca.col(i) * Ca.col(i).transpose();
      }
    }

    // Build beta inactive density
    if (beta_inactive_is_contiguous && !inactive_indices_beta.empty()) {
      auto C_inactive_beta =
          Cb.block(0, inactive_indices_beta.front(), num_atomic_orbitals,
                   inactive_indices_beta.size());
      D_inactive_beta = C_inactive_beta * C_inactive_beta.transpose();
    } else {
      for (size_t i : inactive_indices_beta) {
        D_inactive_beta += Cb.col(i) * Cb.col(i).transpose();
      }
    }

    // Compute J and K matrices for alpha and beta densities
    Eigen::MatrixXd J_alpha_ao(num_atomic_orbitals, num_atomic_orbitals),
        K_alpha_ao(num_atomic_orbitals, num_atomic_orbitals);
    Eigen::MatrixXd J_beta_ao(num_atomic_orbitals, num_atomic_orbitals),
        K_beta_ao(num_atomic_orbitals, num_atomic_orbitals);

    eri->build_JK(D_inactive_alpha.data(), J_alpha_ao.data(), K_alpha_ao.data(),
                  1.0, 0.0, 0.0);
    eri->build_JK(D_inactive_beta.data(), J_beta_ao.data(), K_beta_ao.data(),
                  1.0, 0.0, 0.0);

    Eigen::MatrixXd F_inactive_alpha_ao =
        H_full + J_alpha_ao + J_beta_ao - K_alpha_ao;
    Eigen::MatrixXd F_inactive_beta_ao =
        H_full + J_alpha_ao + J_beta_ao - K_beta_ao;

    // Transform to MO basis
    Eigen::MatrixXd F_inactive_alpha(num_molecular_orbitals,
                                     num_molecular_orbitals);
    Eigen::MatrixXd F_inactive_beta(num_molecular_orbitals,
                                    num_molecular_orbitals);
    F_inactive_alpha = Ca.transpose() * F_inactive_alpha_ao * Ca;
    F_inactive_beta = Cb.transpose() * F_inactive_beta_ao * Cb;

    // Compute inactive energy
    Eigen::MatrixXd H_mo_alpha = Ca.transpose() * H_full * Ca;
    Eigen::MatrixXd H_mo_beta = Cb.transpose() * H_full * Cb;

    double E_inactive = 0.0;
    for (auto i : inactive_indices_alpha) {
      E_inactive += H_mo_alpha(i, i) + F_inactive_alpha(i, i);
    }
    for (auto i : inactive_indices_beta) {
      E_inactive += H_mo_beta(i, i) + F_inactive_beta(i, i);
    }
    // Avoid double counting of two-electron interactions
    E_inactive *= 0.5;

    // Extract active space Hamiltonians
    Eigen::MatrixXd H_active_alpha(nactive, nactive);
    Eigen::MatrixXd H_active_beta(nactive, nactive);

    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active_alpha(i, j) =
            F_inactive_alpha(active_indices_alpha[i], active_indices_alpha[j]);
        H_active_beta(i, j) =
            F_inactive_beta(active_indices_beta[i], active_indices_beta[j]);
      }
    }

    return std::make_shared<data::Hamiltonian>(
        H_active_alpha, H_active_beta, moeri_aaaa, moeri_aabb, moeri_bbbb,
        orbitals, E_inactive + structure->calculate_nuclear_repulsion_energy(),
        F_inactive_alpha, F_inactive_beta);
  }
}
}  // namespace qdk::chemistry::algorithms::microsoft
