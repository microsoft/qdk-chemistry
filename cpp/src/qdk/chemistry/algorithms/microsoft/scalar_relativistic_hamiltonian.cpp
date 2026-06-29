// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scalar_relativistic_hamiltonian.hpp"

#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/util/int1e.h>

#include <blas.hh>
#include <cmath>
#include <lapack.hh>
#include <map>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail_x2c {

/**
 * @brief Spin-free X2C-1e one-electron Hamiltonian via BLAS/LAPACK.
 *
 * Solves the modified Dirac eigenproblem, extracts the X (small/large
 * component ratio) and R (renormalisation) matrices, and returns the
 * Foldy–Wouthuysen Hamiltonian h^{X2C} = R^T h_FW R in the AO basis.
 *
 * @param S_ao  Overlap (nao × nao)
 * @param T_ao  Kinetic energy (nao × nao)
 * @param V_ao  Nuclear attraction + ECP if present (nao × nao)
 * @param W_ao  Spin-free pVp integrals (nao × nao)
 * @param nao   Number of AOs
 * @return      X2C one-electron Hamiltonian (nao × nao)
 */
Eigen::MatrixXd compute_x2c_hamiltonian(const Eigen::MatrixXd& S_ao,
                                        const Eigen::MatrixXd& T_ao,
                                        const Eigen::MatrixXd& V_ao,
                                        const Eigen::MatrixXd& W_ao,
                                        size_t nao) {
  // 1/c = α (fine-structure constant) in atomic units
  const double c_inv = qdk::chemistry::constants::fine_structure_constant;
  const double c2_inv = c_inv * c_inv;
  const int64_t n = static_cast<int64_t>(nao);

  using L = blas::Layout;
  using O = blas::Op;

  // C = α op(A) op(B) + β C
  auto gemm = [](O opA, O opB, int64_t m, int64_t nn, int64_t k, double alpha,
                 const double* A, int64_t lda, const double* B, int64_t ldb,
                 double beta, double* C, int64_t ldc) {
    blas::gemm(L::ColMajor, opA, opB, m, nn, k, alpha, A, lda, B, ldb, beta, C,
               ldc);
  };

  // Symmetric eigendecomposition in-place (eigenvectors in mat, eigenvalues in
  // eigvals)
  auto syev = [](Eigen::MatrixXd& mat, Eigen::VectorXd& eigvals,
                 const char* context) {
    int64_t dim = mat.rows();
    eigvals.resize(dim);
    int64_t info = lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, dim,
                                mat.data(), dim, eigvals.data());
    if (info != 0) {
      throw std::runtime_error(std::string("X2C: LAPACK dsyev failed for ") +
                               context + " (info=" + std::to_string(info) +
                               ")");
    }
  };

  // ── Step 1: Eigendecompose S for S^{±1/2} ──
  Eigen::MatrixXd S_work = S_ao;
  Eigen::VectorXd s_eigvals;
  syev(S_work, s_eigvals, "overlap S");

  // ── Step 2: Build 4-component Dirac Hamiltonian D and metric M ──
  const int64_t dim2 = 2 * n;
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(dim2, dim2);
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dim2, dim2);

  D.block(0, 0, n, n) = V_ao;
  D.block(0, n, n, n) = T_ao;
  D.block(n, 0, n, n) = T_ao;
  D.block(n, n, n, n) = W_ao * c2_inv / 4.0 - T_ao;

  M.block(0, 0, n, n) = S_ao;
  M.block(n, n, n, n) = T_ao * c2_inv / 2.0;

  // ── Step 3: Solve generalised eigenproblem D C = M C ε ──

  Eigen::VectorXd m_eigvals;
  syev(M, m_eigvals, "metric M");

  // Discard near-zero eigenvalues
  int64_t n_kept = 0;
  for (int64_t i = 0; i < dim2; ++i) {
    if (m_eigvals(i) > x2c_metric_lindep_threshold) ++n_kept;
  }
  int64_t first_kept = dim2 - n_kept;

  // Orthogonaliser X_orth = U_kept · diag(1/√σ)
  Eigen::MatrixXd X_orth(dim2, n_kept);
  for (int64_t i = 0; i < n_kept; ++i) {
    double s = 1.0 / std::sqrt(m_eigvals(first_kept + i));
    X_orth.col(i) = M.col(first_kept + i) * s;
  }

  // D' = X_orth^T D X_orth
  Eigen::MatrixXd tmp(n_kept, dim2);
  gemm(O::Trans, O::NoTrans, n_kept, dim2, dim2, 1.0, X_orth.data(), dim2,
       D.data(), dim2, 0.0, tmp.data(), n_kept);
  Eigen::MatrixXd D_orth(n_kept, n_kept);
  gemm(O::NoTrans, O::NoTrans, n_kept, n_kept, dim2, 1.0, tmp.data(), n_kept,
       X_orth.data(), dim2, 0.0, D_orth.data(), n_kept);

  Eigen::VectorXd eigvals;
  syev(D_orth, eigvals, "orthogonalised Dirac Hamiltonian");

  // Back-transform eigenvectors
  if (n_kept < n) {
    throw std::runtime_error(
        "X2C: too many linearly dependent basis functions (n_kept=" +
        std::to_string(n_kept) + " < nao=" + std::to_string(n) + ")");
  }
  Eigen::MatrixXd eigvecs(dim2, n_kept);
  gemm(O::NoTrans, O::NoTrans, dim2, n_kept, n_kept, 1.0, X_orth.data(), dim2,
       D_orth.data(), n_kept, 0.0, eigvecs.data(), dim2);

  // Positive-energy solutions occupy the last n columns
  int64_t pos_start = n_kept - n;
  Eigen::MatrixXd C_L = eigvecs.block(0, pos_start, n, n);
  Eigen::MatrixXd C_S = eigvecs.block(n, pos_start, n, n);

  // ── Step 4: X = C_S C_L^{-1}  via dgesv ──
  Eigen::MatrixXd CL_T = C_L.transpose();
  Eigen::MatrixXd X = C_S.transpose();
  std::vector<int64_t> ipiv(n);
  int64_t info_gesv =
      lapack::gesv(n, n, CL_T.data(), n, ipiv.data(), X.data(), n);
  if (info_gesv != 0) {
    throw std::runtime_error("X2C: LAPACK dgesv failed for X matrix (info=" +
                             std::to_string(info_gesv) + ")");
  }
  X.transposeInPlace();  // X was stored as X^T

  // ── Step 5: Renormalisation R = S^{-1/2} (S^{-1/2} R̃ S^{-1/2})^{-1/2}
  // S^{1/2} ──

  // R̃ = S + X^T (T/(2c²)) X
  Eigen::MatrixXd T_scaled = T_ao * (c2_inv / 2.0);
  Eigen::MatrixXd TX(n, n);  // TX = T_scaled * X
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, T_scaled.data(), n, X.data(), n,
       0.0, TX.data(), n);
  Eigen::MatrixXd R_tilde = S_ao;  // start with S
  gemm(O::Trans, O::NoTrans, n, n, n, 1.0, X.data(), n, TX.data(), n, 1.0,
       R_tilde.data(), n);  // R_tilde += X^T * TX

  // S^{±1/2} = U diag(λ^{±1/2}) U^T
  Eigen::MatrixXd U_scaled(n, n);
  // S^{-1/2}
  for (int64_t j = 0; j < n; ++j)
    U_scaled.col(j) = S_work.col(j) / std::sqrt(s_eigvals(j));
  Eigen::MatrixXd S_mhalf(n, n);
  gemm(O::NoTrans, O::Trans, n, n, n, 1.0, U_scaled.data(), n, S_work.data(), n,
       0.0, S_mhalf.data(), n);

  // S^{1/2}
  for (int64_t j = 0; j < n; ++j)
    U_scaled.col(j) = S_work.col(j) * std::sqrt(s_eigvals(j));
  Eigen::MatrixXd S_half(n, n);
  gemm(O::NoTrans, O::Trans, n, n, n, 1.0, U_scaled.data(), n, S_work.data(), n,
       0.0, S_half.data(), n);

  // R̃_orth = S^{-1/2} R̃ S^{-1/2}
  Eigen::MatrixXd tmp2(n, n);
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, R_tilde.data(), n, S_mhalf.data(),
       n, 0.0, tmp2.data(), n);
  Eigen::MatrixXd Rtilde_orth(n, n);
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, S_mhalf.data(), n, tmp2.data(), n,
       0.0, Rtilde_orth.data(), n);

  Eigen::VectorXd rt_eigvals;
  syev(Rtilde_orth, rt_eigvals, "R_tilde");

  // R̃_orth^{-1/2} = W diag(d^{-1/2}) W^T
  Eigen::MatrixXd Wrt_scaled(n, n);
  for (int64_t j = 0; j < n; ++j)
    Wrt_scaled.col(j) = Rtilde_orth.col(j) / std::sqrt(rt_eigvals(j));
  Eigen::MatrixXd Rt_mhalf(n, n);
  gemm(O::NoTrans, O::Trans, n, n, n, 1.0, Wrt_scaled.data(), n,
       Rtilde_orth.data(), n, 0.0, Rt_mhalf.data(), n);

  // R = S^{-1/2} R̃_orth^{-1/2} S^{1/2}
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, Rt_mhalf.data(), n, S_half.data(),
       n, 0.0, tmp2.data(), n);
  Eigen::MatrixXd R(n, n);
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, S_mhalf.data(), n, tmp2.data(), n,
       0.0, R.data(), n);

  // ── Step 6: h_FW = V + X^T T + T X + X^T (W/(4c²) - T) X;  H_x2c = R^T h_FW
  // R ──

  Eigen::MatrixXd W_scaled = W_ao * (c2_inv / 4.0) - T_ao;
  Eigen::MatrixXd WX(n, n);
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, W_scaled.data(), n, X.data(), n,
       0.0, WX.data(), n);

  Eigen::MatrixXd h_FW = V_ao;
  gemm(O::Trans, O::NoTrans, n, n, n, 1.0, X.data(), n, T_ao.data(), n, 1.0,
       h_FW.data(), n);
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, T_ao.data(), n, X.data(), n, 1.0,
       h_FW.data(), n);
  gemm(O::Trans, O::NoTrans, n, n, n, 1.0, X.data(), n, WX.data(), n, 1.0,
       h_FW.data(), n);

  // H_x2c = R^T h_FW R
  gemm(O::NoTrans, O::NoTrans, n, n, n, 1.0, h_FW.data(), n, R.data(), n, 0.0,
       tmp2.data(), n);
  Eigen::MatrixXd H_x2c(n, n);
  gemm(O::Trans, O::NoTrans, n, n, n, 1.0, R.data(), n, tmp2.data(), n, 0.0,
       H_x2c.data(), n);

  // Symmetrise for numerical safety
  H_x2c = 0.5 * (H_x2c + H_x2c.transpose());

  return H_x2c;
}

/**
 * @brief Decontract a basis, merging duplicate exponents per (atom, l).
 *
 * Matches PySCF's ``decontract_basis(aggregate=True)``.
 *
 * @param contracted_basis  The contracted basis set
 * @return Uncontracted basis set
 */
std::shared_ptr<qcs::BasisSet> decontract_basis(
    const std::shared_ptr<qcs::BasisSet>& contracted_basis) {
  const bool pure = contracted_basis->pure;

  using AtomL = std::pair<uint64_t, int>;
  std::map<AtomL, std::vector<double>> atoml_exps;
  std::map<AtomL, std::array<double, 3>> atoml_origin;

  for (const auto& sh : contracted_basis->shells) {
    AtomL key{sh.atom_index, sh.angular_momentum};
    auto& exps = atoml_exps[key];
    atoml_origin[key] = sh.O;
    for (uint64_t p = 0; p < sh.contraction; ++p) {
      double exp_rounded =
          std::round(sh.exponents[p] * x2c_exponent_rounding_factor) /
          x2c_exponent_rounding_factor;
      bool found = false;
      for (double e : exps) {
        if (std::abs(e - exp_rounded) < x2c_exponent_duplicate_tolerance) {
          found = true;
          break;
        }
      }
      if (!found) exps.push_back(exp_rounded);
    }
  }

  for (auto& [key, exps] : atoml_exps) {
    std::sort(exps.begin(), exps.end(), std::greater<double>());
  }

  std::vector<qcs::Shell> unc_shells;
  for (const auto& [key, exps] : atoml_exps) {
    auto [atom_idx, l] = key;
    const auto& origin = atoml_origin[key];
    for (double exp_val : exps) {
      qcs::Shell unc_sh;
      unc_sh.atom_index = atom_idx;
      unc_sh.O = origin;
      unc_sh.angular_momentum = l;
      unc_sh.contraction = 1;
      unc_sh.exponents[0] = exp_val;
      unc_sh.coefficients[0] = 1.0;
      unc_shells.push_back(unc_sh);
    }
  }

  return std::make_shared<qcs::BasisSet>(contracted_basis->mol, unc_shells,
                                         contracted_basis->mode,
                                         contracted_basis->pure, false);
}

/**
 * @brief Project a matrix from uncontracted to contracted basis.
 *
 * Uses C_proj = S_unc^{-1} <unc|contr>, then H_contr = C_proj^T H_unc C_proj.
 *
 * @param H_unc             Matrix in uncontracted AO basis
 * @param unc_basis         Uncontracted basis
 * @param contracted_basis  Contracted basis
 * @return H in contracted AO basis
 */
Eigen::MatrixXd recontract(const Eigen::MatrixXd& H_unc,
                           const qcs::BasisSet& unc_basis,
                           const qcs::BasisSet& contracted_basis) {
  const size_t n_unc = unc_basis.num_atomic_orbitals;
  const size_t n_contr = contracted_basis.num_atomic_orbitals;

  // Compute cross-overlap ⟨unc|contr⟩
  Eigen::MatrixXd S_cross = Eigen::MatrixXd::Zero(n_unc, n_contr);
  qcs::OneBodyIntegral::cross_overlap(unc_basis, contracted_basis,
                                      unc_basis.mol.get(),
                                      qcs::mpi_default_input(), S_cross.data());

  // Compute S_unc for the projection
  auto int1e_unc = std::make_unique<qcs::OneBodyIntegral>(
      &unc_basis, unc_basis.mol.get(), qcs::mpi_default_input());
  Eigen::MatrixXd S_unc(n_unc, n_unc);
  int1e_unc->overlap_integral(S_unc.data());

  // C_proj = S_unc^{-1} S_cross
  Eigen::MatrixXd C_proj = S_unc.llt().solve(S_cross);
  Eigen::MatrixXd H = C_proj.transpose() * H_unc * C_proj;
  return 0.5 * (H + H.transpose());
}

/**
 * @brief X2C one-electron Hamiltonian in the contracted AO basis.
 *
 * Computes S, T, V(+ECP), W integrals, runs X2C decoupling, and
 * optionally decontracts/recontracts the basis.
 *
 * @param internal_basis_set  Contracted basis set
 * @param mpi                 MPI configuration
 * @param xuncontract         Decontract before X2C, recontract after
 * @return X2C Hamiltonian (nao × nao)
 */
Eigen::MatrixXd compute_x2c_one_electron(
    const std::shared_ptr<qcs::BasisSet>& internal_basis_set,
    const qcs::ParallelConfig& mpi, bool xuncontract) {
  auto working_basis =
      xuncontract ? decontract_basis(internal_basis_set) : internal_basis_set;

  const size_t nao = working_basis->num_atomic_orbitals;

  // One-electron integrals in working basis
  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      working_basis.get(), working_basis->mol.get(), mpi);
  Eigen::MatrixXd S(nao, nao), T(nao, nao), V(nao, nao);
  int1e->overlap_integral(S.data());
  int1e->kinetic_integral(T.data());
  int1e->nuclear_integral(V.data());

  if (working_basis->ecp_shells.size() > 0) {
    Eigen::MatrixXd ECP = Eigen::MatrixXd::Zero(nao, nao);
    int1e->ecp_integral(ECP.data());
    V += ECP;
  }

  // pVp integrals
  Eigen::MatrixXd W(nao, nao);
  int1e->pvp_integral(W.data());

  // X2C decoupling + symmetrise
  Eigen::MatrixXd H_x2c = compute_x2c_hamiltonian(S, T, V, W, nao);
  H_x2c = 0.5 * (H_x2c + H_x2c.transpose());

  // Recontract if we decontracted
  if (xuncontract) {
    return recontract(H_x2c, *working_basis, *internal_basis_set);
  }
  return H_x2c;
}

}  // namespace detail_x2c

std::shared_ptr<data::Hamiltonian>
ScalarRelativisticHamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Orbitals> orbitals) const {
  QDK_LOG_TRACE_ENTERING();
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
  bool alpha_space_is_contiguous =
      utils::microsoft::validate_active_contiguous_indices(
          active_indices_alpha, "Alpha", num_molecular_orbitals);

  // Validate beta active orbitals (if different from alpha) and check
  // contiguity
  bool beta_space_is_contiguous = true;
  if (active_indices_beta != active_indices_alpha) {
    beta_space_is_contiguous =
        utils::microsoft::validate_active_contiguous_indices(
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

  auto structure = basis_set->get_structure();

  auto internal_basis_set =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  // Create dummy SCFConfig
  auto scf_config = std::make_unique<qcs::SCFConfig>();
  scf_config->mpi = qcs::mpi_default_input();
  scf_config->require_gradient = false;
  scf_config->basis = internal_basis_set->name;
  scf_config->cartesian = !internal_basis_set->pure;
  scf_config->scf_orbital_type = qcs::SCFOrbitalType::Restricted;

  // ERI method
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

  // Create ERI engine
  auto eri = qcs::ERIMultiplexer::create(*internal_basis_set, *scf_config, 0.0);

  // X2C one-electron Hamiltonian
  bool xuncontract = _settings->get<bool>("xuncontract");

  Eigen::MatrixXd H_full = detail_x2c::compute_x2c_one_electron(
      internal_basis_set, scf_config->mpi, xuncontract);

  auto int1e = std::make_unique<qcs::OneBodyIntegral>(
      internal_basis_set.get(), internal_basis_set->mol.get(), scf_config->mpi);

  // ── Active-space Hamiltonian construction (mirrors NR constructor) ──

  // Build active coefficient matrices
  Eigen::MatrixXd Ca_active(num_atomic_orbitals, nactive_alpha);
  Eigen::MatrixXd Cb_active(num_atomic_orbitals, nactive_beta);

  if (alpha_space_is_contiguous) {
    // Contiguous alpha
    Ca_active = Ca.block(0, active_indices_alpha.front(), num_atomic_orbitals,
                         nactive_alpha);
  } else {
    // Non-contiguous alpha
    for (size_t i = 0; i < nactive_alpha; i++) {
      Ca_active.col(i) = Ca.col(active_indices_alpha[i]);
    }
  }

  if (beta_space_is_contiguous) {
    // Contiguous beta
    Cb_active = Cb.block(0, active_indices_beta.front(), num_atomic_orbitals,
                         nactive_beta);
  } else {
    // Non-contiguous beta
    for (size_t i = 0; i < nactive_beta; i++) {
      Cb_active.col(i) = Cb.col(active_indices_beta[i]);
    }
  }

  // Convert to row-major for MOERI and compute MO integrals
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ca_active_rm = Ca_active;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Cb_active_rm = Cb_active;

  qcs::MOERI moeri_c(eri);

  // SCF type
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

  // SCFOrbitalType for the ERI engine
  scf_config->scf_orbital_type = is_restricted_calc
                                     ? qcs::SCFOrbitalType::Restricted
                                     : qcs::SCFOrbitalType::Unrestricted;

  // Compute MO-basis ERIs
  const size_t nactive = nactive_alpha;

  // Declare MOERI vectors
  Eigen::VectorXd moeri_aaaa, moeri_aabb, moeri_bbbb;

  const size_t moeri_size = nactive * nactive * nactive * nactive;

  if (is_restricted_calc) {
    moeri_aaaa.resize(moeri_size);
    moeri_c.compute(num_atomic_orbitals, nactive, Ca_active_rm.data(),
                    moeri_aaaa.data());
  } else {
    moeri_aaaa.resize(moeri_size);
    moeri_aabb.resize(moeri_size);
    moeri_bbbb.resize(moeri_size);

    moeri_c.compute(num_atomic_orbitals, nactive, Ca_active_rm.data(),
                    moeri_aaaa.data());

    moeri_c.compute(num_atomic_orbitals, nactive, Cb_active_rm.data(),
                    Cb_active_rm.data(), Ca_active_rm.data(),
                    Ca_active_rm.data(), moeri_aabb.data());

    moeri_c.compute(num_atomic_orbitals, nactive, Cb_active_rm.data(),
                    Cb_active_rm.data(), Cb_active_rm.data(),
                    Cb_active_rm.data(), moeri_bbbb.data());
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

  // all occupied orbitals active
  if (inactive_indices_alpha.empty() && inactive_indices_beta.empty()) {
    if (is_restricted_calc) {
      Eigen::MatrixXd H_active(nactive, nactive);
      H_active = Ca_active.transpose() * H_full * Ca_active;
      Eigen::MatrixXd dummy_fock = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
              H_active, moeri_aaaa, orbitals,
              structure->calculate_nuclear_repulsion_energy(), dummy_fock));
    } else {
      Eigen::MatrixXd H_active_alpha(nactive, nactive);
      Eigen::MatrixXd H_active_beta(nactive, nactive);
      H_active_alpha = Ca_active.transpose() * H_full * Ca_active;
      H_active_beta = Cb_active.transpose() * H_full * Cb_active;
      Eigen::MatrixXd dummy_fock_alpha = Eigen::MatrixXd::Zero(0, 0);
      Eigen::MatrixXd dummy_fock_beta = Eigen::MatrixXd::Zero(0, 0);
      return std::make_shared<data::Hamiltonian>(
          std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
              H_active_alpha, H_active_beta, moeri_aaaa, moeri_aabb, moeri_bbbb,
              orbitals, structure->calculate_nuclear_repulsion_energy(),
              dummy_fock_alpha, dummy_fock_beta));
    }
  }

  if (is_restricted_calc) {
    auto inactive_indices = inactive_indices_alpha;

    bool inactive_space_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices.size() - 1; ++i) {
      if (inactive_indices[i + 1] - inactive_indices[i] != 1) {
        inactive_space_is_contiguous = false;
        break;
      }
    }

    // Inactive density matrix
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

    // Two-electron part of inactive Fock matrix
    Eigen::MatrixXd J_inactive_ao(num_atomic_orbitals, num_atomic_orbitals),
        K_inactive_ao(num_atomic_orbitals, num_atomic_orbitals);
    eri->build_JK(D_inactive.data(), J_inactive_ao.data(), K_inactive_ao.data(),
                  1.0, 0.0, 0.0);
    Eigen::MatrixXd G_inactive_ao = 2 * J_inactive_ao - K_inactive_ao;

    // Inactive Fock matrix
    Eigen::MatrixXd F_inactive_ao = G_inactive_ao + H_full;
    Eigen::MatrixXd F_inactive(num_molecular_orbitals, num_molecular_orbitals);
    F_inactive = Ca.transpose() * F_inactive_ao * Ca;

    // Inactive energy
    double E_inactive = 0.0;
    Eigen::MatrixXd H_mo = Ca.transpose() * H_full * Ca;
    for (auto i : inactive_indices) {
      E_inactive += H_mo(i, i) + F_inactive(i, i);
    }

    // Active-space Hamiltonian
    Eigen::MatrixXd H_active(nactive, nactive);
    for (size_t i = 0; i < nactive; i++) {
      for (size_t j = 0; j < nactive; j++) {
        H_active(i, j) =
            F_inactive(active_indices_alpha[i], active_indices_alpha[j]);
      }
    }

    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
            H_active, moeri_aaaa, orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive));

  } else {
    // Unrestricted
    bool alpha_inactive_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices_alpha.size() - 1; ++i) {
      if (inactive_indices_alpha[i + 1] - inactive_indices_alpha[i] != 1) {
        alpha_inactive_is_contiguous = false;
        break;
      }
    }

    bool beta_inactive_is_contiguous = true;
    for (size_t i = 0; i < inactive_indices_beta.size() - 1; ++i) {
      if (inactive_indices_beta[i + 1] - inactive_indices_beta[i] != 1) {
        beta_inactive_is_contiguous = false;
        break;
      }
    }

    // Separate alpha/beta inactive densities
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

    // J and K matrices
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

    // Inactive energy
    Eigen::MatrixXd H_mo_alpha = Ca.transpose() * H_full * Ca;
    Eigen::MatrixXd H_mo_beta = Cb.transpose() * H_full * Cb;

    double E_inactive = 0.0;
    for (auto i : inactive_indices_alpha) {
      E_inactive += H_mo_alpha(i, i) + F_inactive_alpha(i, i);
    }
    for (auto i : inactive_indices_beta) {
      E_inactive += H_mo_beta(i, i) + F_inactive_beta(i, i);
    }
    // Avoid double-counting
    E_inactive *= 0.5;

    // Active-space Hamiltonians
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
        std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
            H_active_alpha, H_active_beta, moeri_aaaa, moeri_aabb, moeri_bbbb,
            orbitals,
            E_inactive + structure->calculate_nuclear_repulsion_energy(),
            F_inactive_alpha, F_inactive_beta));
  }
}

}  // namespace qdk::chemistry::algorithms::microsoft
