// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cstddef>
#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian_regularizer.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>

#include "microsoft/flr_bliss/flr_bliss_regularizer.hpp"

namespace qdk::chemistry::algorithms {

// ---------------------------------------------------------------------------
// rebuild_hamiltonian: apply a BlissShift to the dense integrals and rebuild.
// Independent of how `shift` was computed (BlissRegularizer or external).
// ---------------------------------------------------------------------------

std::shared_ptr<data::Hamiltonian> rebuild_hamiltonian(
    const data::Hamiltonian& original, const BlissShift& shift,
    double num_electrons) {
  using qdk::chemistry::data::CanonicalFourCenterHamiltonianContainer;
  using qdk::chemistry::data::Hamiltonian;

  if (!original.is_restricted()) {
    throw std::invalid_argument(
        "rebuild_hamiltonian currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  auto [h_alpha, h_beta] = original.get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = original.get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  if (shift.xi.rows() != static_cast<Eigen::Index>(norb) ||
      shift.xi.cols() != static_cast<Eigen::Index>(norb)) {
    throw std::invalid_argument(
        "rebuild_hamiltonian: shift.xi must be norb x norb.");
  }

  const double mu1 = shift.mu1;
  const double mu2 = shift.mu2;
  const Eigen::MatrixXd& xi = shift.xi;

  // One-body part: h_tilde_ij = h_ij + (Ne-1)*xi_ij - (mu1+mu2)*delta_ij.
  Eigen::MatrixXd h_tilde = h_alpha + (num_electrons - 1.0) * xi;
  h_tilde.diagonal().array() -= (mu1 + mu2);

  // Two-body part: g_tilde = g + dg, folded in place (single source of truth
  // in TwoBodyBlissCorrection) to avoid a separate O(norb^4) allocation.
  const TwoBodyBlissCorrection correction{mu2, xi};
  Eigen::VectorXd g_tilde = g_aaaa;
  correction.add_two_body_correction(g_tilde, static_cast<Eigen::Index>(norb));

  // Constant part of -K in the Ne-electron sector: +mu1*Ne + mu2*Ne^2.
  const double core_energy_new = original.get_core_energy() +
                                 mu1 * num_electrons +
                                 mu2 * num_electrons * num_electrons;

  const Eigen::MatrixXd inactive_fock =
      original.has_inactive_fock_matrix()
          ? original.get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto container = std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      h_tilde, g_tilde, original.get_orbitals(), core_energy_new, inactive_fock,
      original.get_type());

  return std::make_shared<Hamiltonian>(std::move(container));
}

// ---------------------------------------------------------------------------
// BlissRegularizer: compute the shift (dispatch on shift_method), then rebuild.
// ---------------------------------------------------------------------------

BlissShift BlissRegularizer::compute_shift(
    const data::Hamiltonian& hamiltonian, unsigned int n_alpha_electrons,
    unsigned int n_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian.is_restricted()) {
    throw std::invalid_argument(
        "BlissRegularizer currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  const std::string shift_method = _settings->get<std::string>("shift_method");

  if (shift_method == "flr_bliss") {
    const double df_truncation_threshold =
        _settings->get<double>("df_truncation_threshold");
    return microsoft::flr_bliss::compute_flr_bliss_shift(
        hamiltonian, n_alpha_electrons, n_beta_electrons,
        df_truncation_threshold);
  }

  throw std::invalid_argument("BlissRegularizer: unknown shift_method '" +
                              shift_method + "'");
}

std::shared_ptr<data::Hamiltonian> BlissRegularizer::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian) {
    throw std::invalid_argument("BlissRegularizer: hamiltonian is null");
  }

  const BlissShift shift =
      compute_shift(*hamiltonian, n_alpha_electrons, n_beta_electrons);
  const double num_electrons = static_cast<double>(n_alpha_electrons) +
                               static_cast<double>(n_beta_electrons);

  return rebuild_hamiltonian(*hamiltonian, shift, num_electrons);
}

// ---------------------------------------------------------------------------
// Factory registration.
// ---------------------------------------------------------------------------

namespace {

std::unique_ptr<BlissRegularizer> make_flr_bliss_regularizer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<BlissRegularizer>();
}

}  // namespace

void HamiltonianRegularizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  HamiltonianRegularizerFactory::register_instance(&make_flr_bliss_regularizer);
}

}  // namespace qdk::chemistry::algorithms
