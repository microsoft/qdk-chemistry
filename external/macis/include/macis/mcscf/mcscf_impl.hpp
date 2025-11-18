/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/mcscf/diis.hpp>
#include <macis/mcscf/fock_matrices.hpp>
#include <macis/mcscf/mcscf.hpp>
#include <macis/mcscf/orbital_gradient.hpp>
#include <macis/mcscf/orbital_hessian.hpp>
#include <macis/mcscf/orbital_rotation_utilities.hpp>
#include <macis/mcscf/orbital_steps.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/transform.hpp>

namespace macis {

/**
 * @brief Multi-Configuration Self-Consistent Field (MCSCF) implementation
 *
 * This function performs a complete MCSCF calculation using the two-step
 * algorithm, optimizing both CI coefficients and orbital rotations to minimize
 * the total energy. The implementation includes DIIS extrapolation for
 * accelerated convergence and supports various active space configurations.
 *
 * @tparam Functor Type of the RDM operator functor used for CI calculations
 *
 * @param rdm_op Functor object that computes reduced density matrices (RDMs)
 * and CI energy
 * @param settings MCSCF configuration settings including convergence criteria
 * and iteration limits
 * @param nalpha Number of alpha electrons in the active space
 * @param nbeta Number of beta electrons in the active space
 * @param norb Total number of molecular orbitals
 * @param ninact Number of inactive (doubly occupied) orbitals
 * @param nact Number of active orbitals
 * @param nvirt Number of virtual (unoccupied) orbitals
 * @param E_core Core nuclear repulsion energy and frozen orbital contributions
 * @param T Pointer to one-electron integral matrix, column-major
 * @param LDT Leading dimension of the T matrix
 * @param V Pointer to two-electron integral tensor in chemist notation,
 * column-major
 * @param LDV Leading dimension of the V tensor
 * @param A1RDM Pointer to active space 1-RDM matrix (input/output),
 * column-major
 * @param LDD1 Leading dimension of the A1RDM matrix
 * @param A2RDM Pointer to active space 2-RDM tensor (input/output),
 * column-major
 * @param LDD2 Leading dimension of the A2RDM tensor
 * @param comm MPI communicator for parallel execution (if MPI enabled)
 *
 * @return Final converged MCSCF energy
 *
 * @note The input RDMs serve as initial guesses if their trace matches the
 * number of active electrons, otherwise fresh RDMs are computed from the CI
 * solver
 * @note All matrices use column-major storage following BLAS/LAPACK conventions
 * @note The algorithm alternates between CI optimization (fixed orbitals) and
 *       orbital optimization (fixed CI coefficients) until convergence
 */
template <typename Functor>
double mcscf_impl(const Functor& rdm_op, MCSCFSettings settings,
                  NumElectron nalpha, NumElectron nbeta, NumOrbital norb,
                  NumInactive ninact, NumActive nact,
                  NumVirtual num_virtual_orbitals, double E_core, double* T,
                  size_t LDT, double* V, size_t LDV, double* A1RDM, size_t LDD1,
                  double* A2RDM, size_t LDD2 MACIS_MPI_CODE(, MPI_Comm comm)) {
  /******************************************************************
   *  Top of MCSCF Routine - Setup and print header info to logger  *
   ******************************************************************/

  auto logger = spdlog::get("mcscf");
  if (!logger) logger = spdlog::stdout_color_mt("mcscf");

  logger->info("[MCSCF Settings]:");
  logger->info("  {:13} = {:4}, {:13} = {:3}, {:13} = {:3}", "NACTIVE_ALPHA",
               nalpha.get(), "NACTIVE_BETA", nbeta.get(), "NORB_TOTAL",
               norb.get());
  logger->info("  {:13} = {:4}, {:13} = {:3}, {:13} = {:3}", "NINACTIVE",
               ninact.get(), "NACTIVE", nact.get(), "NVIRTUAL",
               num_virtual_orbitals.get());
  logger->info("  {:13} = {:4}, {:13} = {:3}, {:13} = {:3}", "ENABLE_DIIS",
               settings.enable_diis, "DIIS_START", settings.diis_start_iter,
               "DIIS_NKEEP", settings.diis_nkeep);
  logger->info("  {:13} = {:.6f}", "E_CORE", E_core);
  logger->info("  {:13} = {:.6e}, {:13} = {:.6e}", "MAX_ORB_STEP",
               settings.max_orbital_step, "ORBGRAD_TOL",
               settings.orb_grad_tol_mcscf
               //"BFGS_TOL",    settings.orb_grad_tol_bfgs,
               //"BFGS_MAX_ITER", settings.max_bfgs_iter
  );
  logger->info("  {:13} = {:.6e}, {:13} = {:.6e}, {:13} = {:3}", "CI_RES_TOL",
               settings.ci_res_tol, "CI_MATEL_TOL", settings.ci_matel_tol,
               "CI_MAX_SUB", settings.ci_max_subspace);

  // MCSCF Iteration format string
  constexpr const char* fmt_string =
      "iter = {:4} E(CI) = {:.10f}, dE = {:18.10e}, |orb_rms| = {:18.10e}";

  /*********************************************************
   *  Calculate persistant derived dimensions to be reused *
   *  throughout this routine                              *
   *********************************************************/

  const size_t no = norb.get(), ni = ninact.get(), na = nact.get(),
               nv = num_virtual_orbitals.get();

  const size_t no2 = no * no;
  const size_t no4 = no2 * no2;
  const size_t na2 = na * na;
  const size_t na4 = na2 * na2;

  const size_t orb_rot_sz = nv * (ni + na) + na * ni;
  const double rms_factor = std::sqrt(orb_rot_sz);
  logger->info("  {:13} = {}", "ORB_ROT_SZ", orb_rot_sz);

  /********************************************************
   *               Allocate persistant data               *
   ********************************************************/

  // Energies
  double E_inactive, E0;

  // Convergence data
  double grad_nrm;
  bool converged = false;

  // Storage for active space Hamitonian
  std::vector<double> T_active(na2), V_active(na4);

  // CI vector - will be resized on first CI call
  std::vector<double> X_CI;

  // Orbital Gradient and Generalized Fock Matrix
  std::vector<double> F(no2), OG(orb_rot_sz), F_inactive(no2), F_active(no2),
      Q(na * no);

  // Storage for transformed integrals
  std::vector<double> transT(T, T + no2), transV(V, V + no4);

  // Storage for total transformation
  std::vector<double> U_total(no2, 0.0), K_total(no2, 0.0);

  // DIIS Object
  DIIS<std::vector<double>> diis(settings.diis_nkeep);

  /**************************************************************
   *    Precompute Active Space Hamiltonian given input data    *
   *                                                            *
   *     This will be used to compute initial energies and      *
   *      gradients to decide whether to proceed with the       *
   *                   MCSCF optimization.                      *
   **************************************************************/

  // Compute Active Space Hamiltonian and Inactive Fock Matrix
  active_hamiltonian(norb, nact, ninact, T, LDT, V, LDV, F_inactive.data(), no,
                     T_active.data(), na, V_active.data(), na);

  // Compute Inactive Energy
  E_inactive = inactive_energy(ninact, T, LDT, F_inactive.data(), no);
  E_inactive += E_core;

  /**************************************************************
   *     Either compute or read initial RDMs from input         *
   *                                                            *
   * If the trace of the input 1RDM is != to the total number   *
   * of active electrons, RDMs will be computed, otherwise the  *
   *      input RDMs will be taken as an initial guess.         *
   **************************************************************/

  // Compute the trace of the input A1RDM
  double iAtr = 0.0;
  for (size_t i = 0; i < na; ++i) iAtr += A1RDM[i * (LDD1 + 1)];
  bool comp_rdms = std::abs(iAtr - nalpha.get() - nbeta.get()) > 1e-6;

  if (comp_rdms) {
    // Compute active RDMs
    logger->info("Computing Initial RDMs");
    std::fill_n(A1RDM, na2, 0.0);
    std::fill_n(A2RDM, na4, 0.0);
    rdm_op.rdms(settings, NumOrbital(na), nalpha.get(), nbeta.get(),
                T_active.data(), V_active.data(), A1RDM, A2RDM,
                X_CI MACIS_MPI_CODE(, comm)) +
        E_inactive;
  } else {
    logger->info("Using Passed RDMs");
  }

  /***************************************************************
   * Compute initial energy and gradient from computed (or read) *
   * RDMs                                                        *
   ***************************************************************/

  // Compute Energy from RDMs
  double E_1RDM = blas::dot(na2, A1RDM, 1, T_active.data(), 1);
  double E_2RDM = blas::dot(na4, A2RDM, 1, V_active.data(), 1);

  E0 = E_1RDM + E_2RDM + E_inactive;
  logger->info("{:8} = {:20.12f}", "E(1RDM)", E_1RDM);
  logger->info("{:8} = {:20.12f}", "E(2RDM)", E_2RDM);
  logger->info("{:8} = {:20.12f}", "E(CI)", E0);

  // Compute initial Fock and gradient
  active_fock_matrix(norb, ninact, nact, V, LDV, A1RDM, LDD1, F_active.data(),
                     no);
  aux_q_matrix(nact, norb, ninact, V, LDV, A2RDM, LDD2, Q.data(), na);
  generalized_fock_matrix(norb, ninact, nact, F_inactive.data(), no,
                          F_active.data(), no, A1RDM, LDD1, Q.data(), na,
                          F.data(), no);
  fock_to_linear_orb_grad(ninact, nact, num_virtual_orbitals, F.data(), no,
                          OG.data());

  /**************************************************************
   *      Compute initial Gradient norm and decide whether      *
   *           input data is sufficiently converged             *
   **************************************************************/

  grad_nrm = blas::nrm2(OG.size(), OG.data(), 1);
  converged = grad_nrm < settings.orb_grad_tol_mcscf;
  logger->info(fmt_string, 0, E0, 0.0, grad_nrm / rms_factor);

  /**************************************************************
   *                     MCSCF Iterations                       *
   **************************************************************/

  for (size_t iter = 0; iter < settings.max_macro_iter; ++iter) {
    // Check for convergence signal
    if (converged) break;

    // Save old data
    const double E0_old = E0;
    std::vector<double> K_total_sav(K_total);

    /************************************************************
     *                  Compute Orbital Step                    *
     ************************************************************/

    std::vector<double> K_step(no2);

    // Compute the step in linear storage
    std::vector<double> K_step_linear(orb_rot_sz);
    precond_cg_orbital_step(norb, ninact, nact, num_virtual_orbitals,
                            F_inactive.data(), no, F_active.data(), no,
                            F.data(), no, A1RDM, LDD1, OG.data(),
                            K_step_linear.data());

    // Compute norms / max
    auto step_nrm = blas::nrm2(orb_rot_sz, K_step_linear.data(), 1);
    auto step_amax = std::abs(
        K_step_linear[blas::iamax(orb_rot_sz, K_step_linear.data(), 1)]);
    logger->debug("{:12}step_nrm = {:.4e}, step_amax = {:.4e}", "", step_nrm,
                  step_amax);

    // Scale step if necessacary
    const auto max_step = settings.max_orbital_step;
    if (step_amax > max_step) {
      logger->info("  * decresing step from {:.2f} to {:.2f}", step_amax,
                   max_step);
      blas::scal(orb_rot_sz, max_step / step_amax, K_step_linear.data(), 1);
    }

    // Expand info full matrix
    linear_orb_rot_to_matrix(ninact, nact, num_virtual_orbitals,
                             K_step_linear.data(), K_step.data(), no);

    // Increment total step
    blas::axpy(no2, 1.0, K_step.data(), 1, K_total.data(), 1);

    // DIIS Extrapolation
    if (settings.enable_diis and iter >= settings.diis_start_iter) {
      diis.add_vector(K_total, OG);
      if (iter >= (settings.diis_start_iter + 2)) {
        K_total = diis.extrapolate();
      }
    }

    /************************************************************
     *   Compute orbital rotation matrix corresponding to the   *
     *                 total (accumulated) step                 *
     ************************************************************/
    if (!iter) {
      // If its the first iteration U_total = EXP[-K_total]
      compute_orbital_rotation(norb, 1.0, K_total.data(), no, U_total.data(),
                               no);

    } else {
      // Compute the rotation matrix for the *actual* step taken,
      // accounting for possible extrapolation
      //
      // U_step = EXP[-(K_total - K_total_old)]
      std::vector<double> U_step(no2);
      blas::axpy(no2, -1.0, K_total.data(), 1, K_total_sav.data(), 1);
      blas::scal(no2, -1.0, K_total_sav.data(), 1);
      compute_orbital_rotation(norb, 1.0, K_total_sav.data(), no, U_step.data(),
                               no);

      // U_total = U_total * U_step
      std::vector<double> tmp(no2);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 no, no, no, 1.0, U_total.data(), no, U_step.data(), no, 0.0,
                 tmp.data(), no);

      U_total = std::move(tmp);
    }

    /************************************************************
     *          Transform Hamiltonian into new MO basis         *
     ************************************************************/
    two_index_transform(no, no, T, LDT, U_total.data(), no, transT.data(), no);
    four_index_transform(no, no, V, LDV, U_total.data(), no, transV.data(), no);

    /************************************************************
     *      Compute Active Space Hamiltonian and associated     *
     *                    scalar quantities                     *
     ************************************************************/

    // Compute Active Space Hamiltonian + inactive Fock
    active_hamiltonian(norb, nact, ninact, transT.data(), no, transV.data(), no,
                       F_inactive.data(), no, T_active.data(), na,
                       V_active.data(), na);

    // Compute Inactive Energy
    E_inactive =
        inactive_energy(ninact, transT.data(), no, F_inactive.data(), no) +
        E_core;

    /************************************************************
     *       Compute new Active Space RDMs and GS energy        *
     ************************************************************/

    std::fill_n(A1RDM, na2, 0.0);
    std::fill_n(A2RDM, na4, 0.0);
    E0 = rdm_op.rdms(settings, NumOrbital(na), nalpha.get(), nbeta.get(),
                     T_active.data(), V_active.data(), A1RDM, A2RDM,
                     X_CI MACIS_MPI_CODE(, comm)) +
         E_inactive;

    /************************************************************
     *               Compute new Orbital Gradient               *
     ************************************************************/

    std::fill(F.begin(), F.end(), 0.0);

    // Update active fock + Q
    active_fock_matrix(norb, ninact, nact, transV.data(), no, A1RDM, LDD1,
                       F_active.data(), no);
    aux_q_matrix(nact, norb, ninact, transV.data(), no, A2RDM, LDD2, Q.data(),
                 na);

    // Compute Fock
    generalized_fock_matrix(norb, ninact, nact, F_inactive.data(), no,
                            F_active.data(), no, A1RDM, LDD1, Q.data(), na,
                            F.data(), no);
    fock_to_linear_orb_grad(ninact, nact, num_virtual_orbitals, F.data(), no,
                            OG.data());

    // Gradient Norm
    grad_nrm = blas::nrm2(OG.size(), OG.data(), 1);
    logger->info(fmt_string, iter + 1, E0, E0 - E0_old, grad_nrm / rms_factor);

    converged = grad_nrm / rms_factor < settings.orb_grad_tol_mcscf;
  }

  if (converged) logger->info("MCSCF Converged");
  return E0;
}

}  // namespace macis
