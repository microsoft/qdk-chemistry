// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "density_fitted_hamiltonian.hpp"

// STL Headers
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <variant>

// MACIS Headers
#include <macis/mcscf/fock_matrices.hpp>

// QDK/Chemistry SCF headers
#include <Eigen/src/Core/Matrix.h>
#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/util/int1e.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

// QDK/Chemistry data::Hamiltonian headers
#include <blas.hh>
#include <lapack.hh>
#include <qdk/chemistry/data/hamiltonian_containers/density_fitted.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "qdk/chemistry/scf/util/libint2_util.h"
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail_df {
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
  QDK_LOG_TRACE_ENTERING();
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

void transform_dferi_ao_to_mo(
    size_t num_atomic_orbitals, size_t nactive, size_t naux,
    std::unique_ptr<double[]>& df_eri, std::unique_ptr<double[]>& df_metric,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& C_active_row_maj,
    Eigen::MatrixXd& df_mo_eri_cm) {
  size_t nao = num_atomic_orbitals;
  size_t nmo = nactive;
  size_t nao2 = nao * nao;

  // 1. Cholesky factorization of metric:  df_metric = L L^{T}
  lapack::potrf(lapack::Uplo::Upper, naux, df_metric.get(), naux);

  // 2. Solve L B = eri_df  => B = L^{-1} eri_df = (metric)^(-1/2) eri_df
  // save result in df_eri.
  blas::trsm(blas::Layout::RowMajor, blas::Side::Left, blas::Uplo::Lower,
             blas::Op::NoTrans, blas::Diag::NonUnit, naux, nao2, 1.0,
             df_metric.get(), naux, df_eri.get(), nao2);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      df_eri_mo_rm(df_mo_eri_cm.rows(), df_mo_eri_cm.cols());

  std::vector<double> tmp(naux * nao * nmo);

  // TMP(Q,i,q) = C(p,i) * B_ao(Q,p,q)
  for (size_t Q = 0; Q < naux; ++Q) {
    auto TMP = tmp.data() + Q * nao * nmo;
    auto B_pt = df_eri.get() + Q * nao2;
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, nmo,
               nao, nao, 1.0, C_active_row_maj.data(), nmo, B_pt, nao, 0.0, TMP,
               nao);
  }

  // B(Q,i,j) = C(q,j) * TMP(Q,i,q)
  // B(Qi,j) = TMP(Qi,q) * C(q,j)
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             naux * nmo, nmo, nao, 1.0, tmp.data(), nao,
             C_active_row_maj.data(), nmo, 0.0, df_eri_mo_rm.data(), nmo);

  df_mo_eri_cm = df_eri_mo_rm;
}
}  // namespace detail_df

std::shared_ptr<data::Hamiltonian>
DensityFittedHamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals,
    OptionalAuxBasis aux_basis) const {
  QDK_LOG_TRACE_ENTERING();
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  if (!aux_basis) {
    throw std::runtime_error(
        "An auxiliary basis set must be provided for density-fitted "
        "Hamiltonian construction.");
  }

  enum class BasisSetType { Explicit, FromString };
  BasisSetType aux_basis_set_type;

  std::string aux_basis_set_name;
  // aux_basis is std::optional<std::variant<...>>, so dereference it first
  const auto& aux_basis_value = *aux_basis;
  if (std::holds_alternative<std::shared_ptr<data::BasisSet>>(
          aux_basis_value)) {
    aux_basis_set_name =
        std::get<std::shared_ptr<data::BasisSet>>(aux_basis_value)->get_name();
    aux_basis_set_type = BasisSetType::Explicit;
  } else if (std::holds_alternative<std::string>(aux_basis_value)) {
    aux_basis_set_name = std::get<std::string>(aux_basis_value);
    aux_basis_set_type = BasisSetType::FromString;
  }
  std::transform(aux_basis_set_name.begin(), aux_basis_set_name.end(),
                 aux_basis_set_name.begin(), ::tolower);

  std::shared_ptr<data::BasisSet> qdk_raw_aux_basis_set = nullptr;
  // TODO: (RL) test the custom_name path here!
  if (aux_basis_set_name == data::BasisSet::custom_name ||
      aux_basis_set_type == BasisSetType::Explicit) {
    qdk_raw_aux_basis_set =
        std::get<std::shared_ptr<data::BasisSet>>(aux_basis_value);
  } else {
    // Create auxiliary basis set from standard name
    auto structure = orbitals->get_basis_set()->get_structure();
    qdk_raw_aux_basis_set =
        data::BasisSet::from_basis_name(aux_basis_set_name, structure);
  }

  auto basis_set = orbitals->get_basis_set();
  auto aux_basis_set = qdk_raw_aux_basis_set;
  const auto& [Ca, Cb] = orbitals->get_coefficients();
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  const size_t num_auxiliary_orbitals =
      aux_basis_set->get_num_atomic_orbitals();
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
  bool alpha_space_is_contiguous =
      detail_df::validate_active_contiguous_indices(
          active_indices_alpha, "Alpha", num_molecular_orbitals);

  // Validate beta active orbitals (if different from alpha) and check
  // contiguity
  bool beta_space_is_contiguous = true;
  if (active_indices_beta != active_indices_alpha) {
    beta_space_is_contiguous = detail_df::validate_active_contiguous_indices(
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
  auto internal_aux_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*aux_basis_set);
  // Create dummy SCFConfig
  auto scf_config = std::make_unique<qcs::SCFConfig>();

  // Create Integral Instance
  auto eri = qcs::ERIMultiplexer::create(*internal_basis_set, *scf_config, 0.0);
  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      internal_basis_set.get(), mol.get(), qcs::mpi_default_input());

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

  // Compute integrals (same size for alpha and beta)
  const size_t nactive = nactive_alpha;

  // Declare MOERI vectors
  Eigen::MatrixXd dfmoeri_aa;
  Eigen::MatrixXd dfmoeri_bb;

  const size_t df_orb_pair_size = nactive * nactive;

  auto basis_libint2 =
      qcs::libint2_util::convert_to_libint_basisset(*internal_basis_set);
  auto aux_basis_libint2 =
      qcs::libint2_util::convert_to_libint_basisset(*internal_aux_basis_set);

  auto h_eri =
      qcs::libint2_util::eri_df(internal_basis_set->mode, basis_libint2,
                                aux_basis_libint2, 0, num_auxiliary_orbitals);
  auto h_metric =
      qcs::libint2_util::metric_df(internal_basis_set->mode, aux_basis_libint2);

  bool is_restricted_calc = orbitals->is_restricted();

  if (is_restricted_calc) {
    // Only allocate and compute (αα|αα) integrals - the others are identical
    dfmoeri_aa.resize(num_auxiliary_orbitals, df_orb_pair_size);
    detail_df::transform_dferi_ao_to_mo(num_atomic_orbitals, nactive,
                                        num_auxiliary_orbitals, h_eri, h_metric,
                                        Ca_active_rm, dfmoeri_aa);
  } else {
    // Unrestricted case - allocate and compute all three types of integrals
    dfmoeri_aa.resize(num_auxiliary_orbitals, df_orb_pair_size);
    dfmoeri_bb.resize(num_auxiliary_orbitals, df_orb_pair_size);

    // (X|αα) integrals
    detail_df::transform_dferi_ao_to_mo(num_atomic_orbitals, nactive,
                                        num_auxiliary_orbitals, h_eri, h_metric,
                                        Ca_active_rm, dfmoeri_aa);

    // (X|ββ) integrals
    detail_df::transform_dferi_ao_to_mo(num_atomic_orbitals, nactive,
                                        num_auxiliary_orbitals, h_eri, h_metric,
                                        Cb_active_rm, dfmoeri_bb);
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
      Eigen::MatrixXd dummy_inactive_fock = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::DensityFittedHamiltonianContainer>(
              H_active, dfmoeri_aa, orbitals,
              structure->calculate_nuclear_repulsion_energy(),
              dummy_inactive_fock));
    } else {
      // Use unrestricted constructor
      Eigen::MatrixXd H_active_alpha(nactive, nactive);
      Eigen::MatrixXd H_active_beta(nactive, nactive);
      H_active_alpha = Ca_active.transpose() * H_full * Ca_active;
      H_active_beta = Cb_active.transpose() * H_full * Cb_active;
      Eigen::MatrixXd dummy_fock_alpha = Eigen::MatrixXd::Zero(0, 0);
      Eigen::MatrixXd dummy_fock_beta = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::DensityFittedHamiltonianContainer>(
              H_active_alpha, H_active_beta, dfmoeri_aa, dfmoeri_bb, orbitals,
              structure->calculate_nuclear_repulsion_energy(), dummy_fock_alpha,
              dummy_fock_beta));
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
        std::make_unique<data::DensityFittedHamiltonianContainer>(
            H_active, dfmoeri_aa, orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive));

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
        std::make_unique<data::DensityFittedHamiltonianContainer>(
            H_active_alpha, H_active_beta, dfmoeri_aa, dfmoeri_bb, orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive_alpha, F_inactive_beta));
  }
}
}  // namespace qdk::chemistry::algorithms::microsoft
