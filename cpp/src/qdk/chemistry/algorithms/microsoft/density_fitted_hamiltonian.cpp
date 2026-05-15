// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "density_fitted_hamiltonian.hpp"

#include "hamiltonian_util.hpp"

// STL Headers
#include <cstddef>
#include <memory>

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
#include <qdk/chemistry/utils/logger.hpp>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail_df {

// Helper function that takes in DF integrals (ij|P) and metric integral (P|Q),
// fold (P|Q)^(-1/2) into (ij|P), such that the resulting integrals (ij|P), also
// written as B^Q_ij, can be used directly in the DF expression for four-center
// integrals: (ij|kl) ≈ Σ_Q B^Q_ij B^Q_kl. This assumes everything is in the
// atomic orbital basis. The variable df_eri is over-written upon output.
void fold_metric_to_three_center(size_t num_atomic_orbitals, size_t naux,
                                 std::unique_ptr<double[]>& df_eri,
                                 std::unique_ptr<double[]>& df_metric) {
  size_t nao = num_atomic_orbitals;

  size_t nao2 = nao * nao;

  // 1. Use cholesky factorization on metric:  df_metric = L L^{T}
  lapack::potrf(lapack::Uplo::Lower, naux, df_metric.get(), naux);

  // 2. Solve L B = eri_df  => B = L^{-1} eri_df = (metric)^(-1/2) eri_df
  // save result in df_eri.
  blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Lower,
             blas::Op::Trans, blas::Diag::NonUnit, nao2, naux, 1.0,
             df_metric.get(), naux, df_eri.get(), nao2);
}
}  // namespace detail_df

std::shared_ptr<data::Hamiltonian>
DensityFittedHamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals) const {
  QDK_LOG_TRACE_ENTERING();
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  auto basis_set = orbitals->get_basis_set();
  if (!basis_set->has_aux_basis()) {
    throw std::runtime_error(
        "An auxiliary basis set must be provided for density-fitted "
        "Hamiltonian construction.");
  }

  const auto& [Ca, Cb] = orbitals->get_coefficients();
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  const size_t num_auxiliary_orbitals = basis_set->get_num_auxiliary_orbitals();

  // Get alpha and beta active space indices
  auto [active_indices_alpha, active_indices_beta] =
      orbitals->get_active_space_indices();

  if (orbitals->is_restricted() && active_indices_alpha.empty()) {
    throw std::runtime_error("Need to specify an active space.");
  } else if (orbitals->is_unrestricted() &&
             (active_indices_alpha.empty() || active_indices_beta.empty())) {
    throw std::runtime_error(
        "Need to specify an active space for alpha and beta.");
  }

  if (active_indices_alpha.size() != active_indices_beta.size()) {
    throw std::runtime_error(
        "Alpha and beta active spaces must have the same size. "
        "Alpha: " +
        std::to_string(active_indices_alpha.size()) +
        ", Beta: " + std::to_string(active_indices_beta.size()));
  }

  // Create internal Molecule
  auto structure = basis_set->get_structure();
  auto mol = utils::microsoft::convert_to_molecule(*structure, 0, 1);

  // Create internal BasisSet
  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  auto internal_aux_basis_set =
      utils::microsoft::convert_aux_basis_set_from_qdk(*basis_set);

  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      internal_basis_set.get(), mol.get(), qcs::mpi_default_input());

  // Compute Core Hamiltonian in AO basis
  Eigen::MatrixXd T_full(num_atomic_orbitals, num_atomic_orbitals),
      V_full(num_atomic_orbitals, num_atomic_orbitals);
  int1e->kinetic_integral(T_full.data());
  int1e->nuclear_integral(V_full.data());
  Eigen::MatrixXd H_full = T_full + V_full;

  // Compute DF integrals and fold metric
  auto basis_libint2 =
      qcs::libint2_util::convert_to_libint_basisset(*internal_basis_set);
  auto aux_basis_libint2 =
      qcs::libint2_util::convert_to_libint_basisset(*internal_aux_basis_set);

  auto h_eri =
      qcs::libint2_util::eri_df(internal_basis_set->mode, basis_libint2,
                                aux_basis_libint2, 0, num_auxiliary_orbitals);
  auto h_metric =
      qcs::libint2_util::metric_df(internal_basis_set->mode, aux_basis_libint2);

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

  detail_df::fold_metric_to_three_center(
      num_atomic_orbitals, num_auxiliary_orbitals, h_eri, h_metric);
  Eigen::Map<Eigen::MatrixXd> B_ao(h_eri.get(),
                                   num_atomic_orbitals * num_atomic_orbitals,
                                   num_auxiliary_orbitals);

  return detail::build_active_space_hamiltonian_from_three_center(
      B_ao, H_full, Ca, Cb, orbitals, structure, is_restricted_calc,
      /*store_ao_vectors=*/false);
}
}  // namespace qdk::chemistry::algorithms::microsoft
