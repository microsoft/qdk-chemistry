// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "density_fitted_hamiltonian.hpp"

#include "hamiltonian_util.hpp"

// STL Headers
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>

// QDK/Chemistry SCF headers
#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/util/int1e.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <Eigen/Core>

// QDK/Chemistry data::Hamiltonian headers
#include <blas.hh>
#include <lapack.hh>
#include <qdk/chemistry/data/hamiltonian_containers/three_center.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail_df {

// Fold the Coulomb metric into raw three-center integrals E. For M = L L^T,
// the right-side solve B L^T = E produces B = E L^{-T}, so
// B B^T = E M^{-1} E^T. The resulting metric-orthonormalized factors B can be
// contracted directly over one auxiliary index. Both E and B use the dense
// [nao^2, naux] layout; df_eri is overwritten with B.
void fold_metric_to_three_center(size_t num_atomic_orbitals, size_t naux,
                                 std::unique_ptr<double[]>& df_eri,
                                 std::unique_ptr<double[]>& df_metric) {
  const auto nao = num_atomic_orbitals;
  const auto nao2 = nao * nao;

  // Factor the auxiliary Coulomb metric M = L L^T.
  const auto info =
      lapack::potrf(lapack::Uplo::Lower, naux, df_metric.get(), naux);
  if (info != 0) {
    throw std::runtime_error(
        info > 0 ? "Density-fitting metric is not positive definite; remove "
                   "linearly dependent auxiliary functions"
                 : "Density-fitting metric factorization received an invalid "
                   "argument");
  }

  // Solve B L^T = E in place.
  blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Lower,
             blas::Op::Trans, blas::Diag::NonUnit, nao2, naux, 1.0,
             df_metric.get(), naux, df_eri.get(), nao2);
  if (!std::all_of(df_eri.get(), df_eri.get() + nao2 * naux,
                   [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error(
        "Density-fitting metric solve produced non-finite factors");
  }
}
}  // namespace detail_df

std::shared_ptr<data::Hamiltonian>
DensityFittedHamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals,
    std::shared_ptr<data::AuxiliaryBasisCollection> auxiliary_bases) const {
  QDK_LOG_TRACE_ENTERING();
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  auto basis_set = orbitals->get_basis_set();
  if (!auxiliary_bases ||
      !auxiliary_bases->has_auxiliary_basis(data::AuxiliaryBasisRole::RIFit)) {
    throw std::runtime_error(
        "Density-fitted Hamiltonian construction requires an auxiliary basis "
        "with the RIFit role.");
  }

  auto auxiliary_basis =
      auxiliary_bases->get_auxiliary_basis(data::AuxiliaryBasisRole::RIFit);
  if (auxiliary_basis->get_structure()->content_hash() !=
      basis_set->get_structure()->content_hash()) {
    throw std::invalid_argument(
        "The RIFit auxiliary basis must describe the Hamiltonian's molecular "
        "structure.");
  }

  const auto& Ca = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  const auto& Cb =
      orbitals->coefficients()->block({data::axes::beta(), data::axes::beta()});
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  const size_t num_auxiliary_orbitals =
      auxiliary_basis->get_num_auxiliary_orbitals();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  if (static_cast<size_t>(Ca.rows()) != num_atomic_orbitals ||
      static_cast<size_t>(Cb.rows()) != num_atomic_orbitals) {
    throw std::invalid_argument(
        "Orbital coefficient row count must match the primary basis AO count");
  }

  // Get alpha and beta active space indices
  const auto active_ai = orbitals->active_indices();
  auto active_indices_alpha =
      data::spin_channel_indices(active_ai, data::axes::alpha());
  auto active_indices_beta =
      data::spin_channel_indices(active_ai, data::axes::beta());

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

  // Ensure alpha and beta active spaces have the same size
  if (nactive_alpha != nactive_beta) {
    throw std::runtime_error(
        "Alpha and beta active spaces must have the same size. "
        "Alpha: " +
        std::to_string(nactive_alpha) +
        ", Beta: " + std::to_string(nactive_beta));
  }

  const double effective_nuclear_repulsion =
      basis_set->calculate_effective_nuclear_repulsion_energy();

  // Create internal primary and auxiliary basis sets.
  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  auto internal_aux_basis_set =
      utils::microsoft::convert_auxiliary_basis_from_qdk(*auxiliary_basis);

  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      internal_basis_set.get(), internal_basis_set->mol.get(),
      qcs::mpi_default_input());

  // Compute Core Hamiltonian in AO basis
  Eigen::MatrixXd T_full(num_atomic_orbitals, num_atomic_orbitals),
      V_full(num_atomic_orbitals, num_atomic_orbitals);
  int1e->kinetic_integral(T_full.data());
  int1e->nuclear_integral(V_full.data());
  Eigen::MatrixXd H_full = T_full + V_full;

  if (!internal_basis_set->ecp_shells.empty()) {
    Eigen::MatrixXd ecp_full =
        Eigen::MatrixXd::Zero(num_atomic_orbitals, num_atomic_orbitals);
    int1e->ecp_integral(ecp_full.data());
    H_full += ecp_full;
  }

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

  // Compute integrals (same size for alpha and beta)
  const size_t nactive = nactive_alpha;

  // Declare MOERI vectors
  Eigen::MatrixXd dfmoeri_aa;
  Eigen::MatrixXd dfmoeri_bb;

  auto basis_libint2 =
      qcs::libint2_util::convert_to_libint_basisset(*internal_basis_set);
  auto aux_basis_libint2 =
      qcs::libint2_util::convert_to_libint_basisset(*internal_aux_basis_set);

  auto h_eri =
      qcs::libint2_util::eri_df(internal_basis_set->mode, basis_libint2,
                                aux_basis_libint2, 0, num_auxiliary_orbitals);
  auto h_metric =
      qcs::libint2_util::metric_df(internal_basis_set->mode, aux_basis_libint2);

  bool is_restricted_calc = (active_indices_alpha == active_indices_beta) &&
                            orbitals->is_restricted();

  detail_df::fold_metric_to_three_center(
      num_atomic_orbitals, num_auxiliary_orbitals, h_eri, h_metric);
  Eigen::Map<Eigen::MatrixXd> B_ao(h_eri.get(),
                                   num_atomic_orbitals * num_atomic_orbitals,
                                   num_auxiliary_orbitals);
  dfmoeri_aa = detail::transform_three_center_ao_to_mo(B_ao, Ca_active);

  if (!is_restricted_calc) {
    dfmoeri_bb = detail::transform_three_center_ao_to_mo(B_ao, Cb_active);
  }

  // Get inactive space indices for both alpha and beta
  const auto inactive_ai = orbitals->inactive_indices();
  auto inactive_indices_alpha =
      data::spin_channel_indices(inactive_ai, data::axes::alpha());
  auto inactive_indices_beta =
      data::spin_channel_indices(inactive_ai, data::axes::beta());

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
          std::make_unique<data::ThreeCenterHamiltonianContainer>(
              H_active, dfmoeri_aa, orbitals, effective_nuclear_repulsion,
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
          std::make_unique<data::ThreeCenterHamiltonianContainer>(
              H_active_alpha, H_active_beta, dfmoeri_aa, dfmoeri_bb, orbitals,
              effective_nuclear_repulsion, dummy_fock_alpha, dummy_fock_beta));
    }
  }

  if (is_restricted_calc) {
    // Restricted case
    auto inactive_indices = inactive_indices_alpha;

    // Compute inactive density and J/K from 3-center vectors
    Eigen::MatrixXd D_inactive = detail::build_inactive_density(
        Ca, inactive_indices, num_atomic_orbitals);
    Eigen::MatrixXd J_inactive_ao =
        detail::build_J_from_three_center(B_ao, D_inactive);
    Eigen::MatrixXd K_inactive_ao =
        detail::build_K_from_three_center(B_ao, Ca, inactive_indices);

    auto result = detail::compute_restricted_inactive(
        J_inactive_ao, K_inactive_ao, H_full, Ca, inactive_indices,
        active_indices_alpha);

    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::ThreeCenterHamiltonianContainer>(
            result.H_active, dfmoeri_aa, orbitals,
            result.E_inactive + effective_nuclear_repulsion,
            result.F_inactive));

  } else {
    // Unrestricted case

    // Compute inactive densities and J/K from 3-center vectors
    Eigen::MatrixXd D_inactive_alpha = detail::build_inactive_density(
        Ca, inactive_indices_alpha, num_atomic_orbitals);
    Eigen::MatrixXd D_inactive_beta = detail::build_inactive_density(
        Cb, inactive_indices_beta, num_atomic_orbitals);

    Eigen::MatrixXd J_alpha_ao =
        detail::build_J_from_three_center(B_ao, D_inactive_alpha);
    Eigen::MatrixXd K_alpha_ao =
        detail::build_K_from_three_center(B_ao, Ca, inactive_indices_alpha);
    Eigen::MatrixXd J_beta_ao =
        detail::build_J_from_three_center(B_ao, D_inactive_beta);
    Eigen::MatrixXd K_beta_ao =
        detail::build_K_from_three_center(B_ao, Cb, inactive_indices_beta);

    auto result = detail::compute_unrestricted_inactive(
        J_alpha_ao, K_alpha_ao, J_beta_ao, K_beta_ao, H_full, Ca, Cb,
        inactive_indices_alpha, inactive_indices_beta, active_indices_alpha,
        active_indices_beta);

    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::ThreeCenterHamiltonianContainer>(
            result.H_active_alpha, result.H_active_beta, dfmoeri_aa, dfmoeri_bb,
            orbitals, result.E_inactive + effective_nuclear_repulsion,
            result.F_inactive_alpha, result.F_inactive_beta));
  }
}
}  // namespace qdk::chemistry::algorithms::microsoft
