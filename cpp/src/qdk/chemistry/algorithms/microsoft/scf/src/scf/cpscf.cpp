// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/exc/gauxc_impl.h>
#include <spdlog/spdlog.h>

#include "scf/ks_impl.h"
#include "scf/scf_impl.h"
#include "util/blas.h"
#include "util/lapack.h"
#include "util/opt/gmresxx/arnoldi/arnoldi_gmres.h"
#include "util/timer.h"

namespace qdk::chemistry::scf {
void SCFImpl::polarizability_() {
  std::array<double, 9> polarizability;

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  MPI_Bcast(P_.data(),
            num_density_matrices_ * num_atomic_orbitals_ * num_atomic_orbitals_,
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  // Compute Dipole Integrals
  RowMajorMatrix dipole(3 * num_atomic_orbitals_, num_atomic_orbitals_);
  int1e_->dipole_integral(dipole.data());

  const size_t num_alpha = nelec_[0];
  const size_t num_alpha_virtual_orbitals = num_molecular_orbitals_ - num_alpha;
  const size_t nova = num_alpha * num_alpha_virtual_orbitals;
  const size_t num_beta = nelec_[1];
  const size_t num_beta_virtual_orbitals = num_molecular_orbitals_ - num_beta;
  const size_t novb = num_beta * num_beta_virtual_orbitals;
  size_t nov = nova;
  if (ctx_.cfg->unrestricted) nov = nova + novb;

  auto dot = [](size_t n, const auto* a, const auto* b) {
    auto res = a[0] * b[0];
    for (auto i = 1; i < n; ++i) res += a[i] * b[i];
    return res;
  };

  const double* Ca_occ_ptr = C_.data();
  const double* Ca_vir_ptr = Ca_occ_ptr + num_alpha;
  const double* Cb_occ_ptr =
      Ca_occ_ptr + num_atomic_orbitals_ * num_molecular_orbitals_;
  const double* Cb_vir_ptr = Cb_occ_ptr + num_beta;
  Eigen::MatrixXd temp(std::max(num_alpha, num_beta), num_atomic_orbitals_);

  // for each xyz-direction
  for (auto xyz_1d = 0; xyz_1d < 3; ++xyz_1d) {
    const double* dipole_x_ptr =
        dipole.data() + xyz_1d * num_atomic_orbitals_ * num_atomic_orbitals_;
    Eigen::VectorXd R_vec(nov);
    Eigen::VectorXd X_vec = Eigen::VectorXd::Zero(nov);

    if (ctx_.cfg->mpi.world_rank == 0) {
      spdlog::info("CPHF iteration for direction {}",
                   xyz_1d);  // forming R^x = u_ai

      // R_{ia} = \Sum_{uv} C_{ui} Dipole_{uv} C_{av}
      // (C is row-major, temp matrix is column-major, R_vec has
      // num_alpha_virtual_orbitals as fast-index)
      blas::gemm("N", "N", num_alpha, num_atomic_orbitals_,
                 num_atomic_orbitals_, 1.0, Ca_occ_ptr, num_molecular_orbitals_,
                 dipole_x_ptr, num_atomic_orbitals_, 0.0, temp.data(),
                 num_alpha);
      blas::gemm("N", "T", num_alpha_virtual_orbitals, num_alpha,
                 num_atomic_orbitals_, 1.0, Ca_vir_ptr, num_molecular_orbitals_,
                 temp.data(), num_alpha, 0.0, R_vec.data(),
                 num_alpha_virtual_orbitals);

      if (ctx_.cfg->unrestricted) {
        blas::gemm("N", "N", num_beta, num_atomic_orbitals_,
                   num_atomic_orbitals_, 1.0, Cb_occ_ptr,
                   num_molecular_orbitals_, dipole_x_ptr, num_atomic_orbitals_,
                   0.0, temp.data(), num_beta);
        blas::gemm("N", "T", num_beta_virtual_orbitals, num_beta,
                   num_atomic_orbitals_, 1.0, Cb_vir_ptr,
                   num_molecular_orbitals_, temp.data(), num_beta, 0.0,
                   R_vec.data() + nova, num_beta_virtual_orbitals);
      }
    }

    cpscf_(R_vec.data(),
           X_vec.data());  // Solve the linear system A X = R, A is constructed
                           // from the CPHF equations inside the function

    // Form D_\mu\nu^x = \sum_{ia} (C_{\mu i} C_{\nu a} + C_{\nu i} C_{\mu a})
    // R^x_{ia}
    if (ctx_.cfg->mpi.world_rank == 0) {
      RowMajorMatrix Dx(num_atomic_orbitals_, num_atomic_orbitals_);
      blas::gemm("T", "N", num_alpha, num_atomic_orbitals_,
                 num_alpha_virtual_orbitals, 1.0, X_vec.data(),
                 num_alpha_virtual_orbitals, Ca_vir_ptr,
                 num_molecular_orbitals_, 0.0, temp.data(), num_alpha);
      blas::gemm("T", "N", num_atomic_orbitals_, num_atomic_orbitals_,
                 num_alpha, 1.0, temp.data(), num_alpha, Ca_occ_ptr,
                 num_molecular_orbitals_, 0.0, Dx.data(), num_atomic_orbitals_);
      if (ctx_.cfg->unrestricted) {
        blas::gemm("T", "N", num_beta, num_atomic_orbitals_,
                   num_beta_virtual_orbitals, 1.0, X_vec.data() + nova,
                   num_beta_virtual_orbitals, Cb_vir_ptr,
                   num_molecular_orbitals_, 0.0, temp.data(), num_beta);
        blas::gemm("T", "N", num_atomic_orbitals_, num_atomic_orbitals_,
                   num_beta, 1.0, temp.data(), num_beta, Cb_occ_ptr,
                   num_molecular_orbitals_, 1.0, Dx.data(),
                   num_atomic_orbitals_);
      }
      // Symmetrize the matrix
      for (size_t i = 0; i < num_atomic_orbitals_; ++i)
        for (size_t j = i; j < num_atomic_orbitals_; ++j) {
          const auto symm_ij = Dx(i, j) + Dx(j, i);
          Dx(i, j) = symm_ij;
          Dx(j, i) = symm_ij;
        }

      for (auto xyz_2d = 0; xyz_2d < 3; ++xyz_2d) {
        // polarizability^{xy} = \sum_{\mu\nu} d_\mu\nu^x D_\mu\nu^y
        ctx_.result.scf_polarizability[xyz_1d * 3 + xyz_2d] =
            dot(num_atomic_orbitals_ * num_atomic_orbitals_,
                dipole.data() +
                    xyz_2d * num_atomic_orbitals_ * num_atomic_orbitals_,
                Dx.data());
        if (!ctx_.cfg->unrestricted)
          ctx_.result.scf_polarizability[xyz_1d * 3 + xyz_2d] *= 2;
      }
    }
  }

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  // MPI bcast polarizability
  MPI_Bcast(ctx_.result.scf_polarizability.data(), 9, MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
#endif
  ctx_.result.scf_isotropic_polarizability =
      (ctx_.result.scf_polarizability[0] + ctx_.result.scf_polarizability[4] +
       ctx_.result.scf_polarizability[8]) /
      3.0;

  if (ctx_.cfg->mpi.world_rank == 0) {
    spdlog::info("Polarizability (a.u.): ");
    for (auto i = 0; i < 3; ++i) {
      spdlog::info("{:.12f}, {:.12f}, {:.12f}",
                   ctx_.result.scf_polarizability[i * 3],
                   ctx_.result.scf_polarizability[i * 3 + 1],
                   ctx_.result.scf_polarizability[i * 3 + 2]);
    }
    spdlog::info("Isotropic polarizability (a.u.): {:.12f}",
                 ctx_.result.scf_isotropic_polarizability);
  }
}

void SCFImpl::cpscf_(const double* R_input, double* X_sol) {
  AutoTimer __timer("polarizability:: cpscf");
  const size_t num_alpha = nelec_[0];
  const size_t num_alpha_virtual_orbitals = num_molecular_orbitals_ - num_alpha;
  const size_t nova = num_alpha * num_alpha_virtual_orbitals;
  const size_t num_beta = nelec_[1];
  const size_t num_beta_virtual_orbitals = num_molecular_orbitals_ - num_beta;
  const size_t novb = num_beta * num_beta_virtual_orbitals;
  size_t nov = nova;
  if (ctx_.cfg->unrestricted) nov = nova + novb;

  const double* Ca_occ_ptr = C_.data();
  const double* Ca_vir_ptr = Ca_occ_ptr + num_alpha;
  const double* Cb_occ_ptr = Ca_occ_ptr + (num_density_matrices_ - 1) *
                                              num_atomic_orbitals_ *
                                              num_molecular_orbitals_;
  const double* Cb_vir_ptr = Cb_occ_ptr + num_beta;

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  int signal = 0;

  // MPI worker which will receive the signal and tP from root (rank 0)
  // and compute the trial Fock matrix
  auto worker_callback = [&]() {
    int halt = 0;
    while (!halt) {
      // Wait for a signal from root (rank 0)
      MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if (signal == 1) {
        // Receive the data, tP, from root
        MPI_Bcast(
            tP_.data(),
            num_density_matrices_ * num_atomic_orbitals_ * num_atomic_orbitals_,
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // This function computes the Fock matrix using the current density
        // matrix tP, and will bcast inside
        update_trial_fock_();
      } else if (signal == -1) {
        // HALT SIGNAL
        halt = 1;
      }
    }
  };
#endif

  auto root_func = [&]() {
    // Configure GMRES settings
    gmres_settings settings;
    settings.max_krylov_dim = ctx_.cfg->cpscf_input.max_iteration;
    settings.max_restart = ctx_.cfg->cpscf_input.max_restart;
    settings.tol = ctx_.cfg->cpscf_input.tolerance;
    settings.verbosity = ctx_.cfg->verbose > 3 ? ctx_.cfg->verbose - 3 : 0;
    // Shifts for GMRES (we use zero as we're not shifting)
    double shift = 0.0;
    Eigen::MatrixXd temp(std::max(num_alpha, num_beta), num_atomic_orbitals_);

    // Create a matrix operator for GMRES
    // This function computes Y = beta*Y + alpha*(A+B)*X
    // A+B represents the CPHF operator matrix
    auto A_op = [&](int32_t N, int32_t NRHS, const double alpha,
                    const double* X, int32_t LDX, double beta, double* Y,
                    int32_t LDY) {
      AutoTimer __timer("polarizability:: A_op");

      if (N != nov) {
        throw std::runtime_error("Matrix size mismatch in CPSCF GMRES solver.");
      }

      // Set Y to zero initially if we're not adding to it
      if (beta == 0.0)
        std::fill(Y, Y + N * NRHS, 0.0);
      else
        for (int32_t i = 0; i < N * NRHS; ++i) Y[i] *= beta;

      // For each right-hand side, apply the operator
      for (int32_t rhs = 0; rhs < NRHS; ++rhs) {
        const double* X_rhs = X + rhs * LDX;
        double* Y_rhs = Y + rhs * LDY;

        // calculate orbital energy difference term
        for (size_t i = 0; i < num_alpha; ++i)
          for (size_t a = 0; a < num_alpha_virtual_orbitals; ++a) {
            Y_rhs[i * num_alpha_virtual_orbitals + a] +=
                alpha * (eigenvalues_(0, a + num_alpha) - eigenvalues_(0, i)) *
                X_rhs[i * num_alpha_virtual_orbitals +
                      a];  // δij δab δστ (ϵaσ − ϵiτ )
          }
        if (ctx_.cfg->unrestricted) {
          for (size_t i = 0; i < num_beta; ++i)
            for (size_t a = 0; a < num_beta_virtual_orbitals; ++a)
              Y_rhs[nova + i * num_beta_virtual_orbitals + a] +=
                  alpha * (eigenvalues_(1, a + num_beta) - eigenvalues_(1, i)) *
                  X_rhs[nova + i * num_beta_virtual_orbitals +
                        a];  // δij δab δστ (ϵaσ − ϵiτ )
        }

        // tP_{uv} = \sum_{ia}  R_{ia} (C_{ui} C_{av} + C_{vi} C_{au})
        // R has num_alpha_virtual_orbitals as fast-index, tP is symmetric
        blas::gemm("T", "N", num_alpha, num_atomic_orbitals_,
                   num_alpha_virtual_orbitals, 1.0, X_rhs,
                   num_alpha_virtual_orbitals, Ca_vir_ptr,
                   num_molecular_orbitals_, 0.0, temp.data(), num_alpha);
        blas::gemm("T", "N", num_atomic_orbitals_, num_atomic_orbitals_,
                   num_alpha, 1.0, temp.data(), num_alpha, Ca_occ_ptr,
                   num_molecular_orbitals_, 0.0, tP_.data(),
                   num_atomic_orbitals_);
        for (size_t i = 0; i < num_atomic_orbitals_; ++i)
          for (size_t j = i; j < num_atomic_orbitals_; ++j) {
            const auto symm_ij = tP_(i, j) + tP_(j, i);
            tP_(i, j) = symm_ij;
            tP_(j, i) = symm_ij;
          }
        if (ctx_.cfg->unrestricted) {
          blas::gemm("T", "N", num_beta, num_atomic_orbitals_,
                     num_beta_virtual_orbitals, 1.0, X_rhs + nova,
                     num_beta_virtual_orbitals, Cb_vir_ptr,
                     num_molecular_orbitals_, 0.0, temp.data(), num_beta);
          blas::gemm("T", "N", num_atomic_orbitals_, num_atomic_orbitals_,
                     num_beta, 1.0, temp.data(), num_beta, Cb_occ_ptr,
                     num_molecular_orbitals_, 0.0,
                     tP_.data() + num_atomic_orbitals_ * num_atomic_orbitals_,
                     num_atomic_orbitals_);
          for (size_t i = 0; i < num_atomic_orbitals_; ++i)
            for (size_t j = i; j < num_atomic_orbitals_; ++j) {
              const auto symm_ij = tP_(i + num_atomic_orbitals_, j) +
                                   tP_(j + num_atomic_orbitals_, i);
              tP_(i + num_atomic_orbitals_, j) = symm_ij;
              tP_(j + num_atomic_orbitals_, i) = symm_ij;
            }
        }

        // Check if we're running in MPI mode with multiple ranks
        if (ctx_.cfg->mpi.world_size > 1) {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
          // Signal workers to start computation
          signal = 1;
          MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);

          // Broadcast data to workers
          MPI_Bcast(tP_.data(),
                    num_density_matrices_ * num_atomic_orbitals_ *
                        num_atomic_orbitals_,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

          // Root also performs computation
          update_trial_fock_();  // This function computes the Fock matrix using
                                 // the current density matrix tP_, and will
                                 // bcast inside
#endif
        } else {
          // Single process mode - just compute directly
          update_trial_fock_();
        }

        // ABX_{ia} = \sum_{uv} C_{ui} F_{uv} C_{av}
        blas::gemm("N", "N", num_alpha, num_atomic_orbitals_,
                   num_atomic_orbitals_, 1.0, Ca_occ_ptr,
                   num_molecular_orbitals_, tFock_.data(), num_atomic_orbitals_,
                   0.0, temp.data(), num_alpha);
        blas::gemm("N", "T", num_alpha_virtual_orbitals, num_alpha,
                   num_atomic_orbitals_, alpha, Ca_vir_ptr,
                   num_molecular_orbitals_, temp.data(), num_alpha, 1.0, Y_rhs,
                   num_alpha_virtual_orbitals);
        if (ctx_.cfg->unrestricted) {
          blas::gemm(
              "N", "N", num_beta, num_atomic_orbitals_, num_atomic_orbitals_,
              1.0, Cb_occ_ptr, num_molecular_orbitals_,
              tFock_.data() + num_atomic_orbitals_ * num_atomic_orbitals_,
              num_atomic_orbitals_, 0.0, temp.data(), num_beta);
          blas::gemm("N", "T", num_beta_virtual_orbitals, num_beta,
                     num_atomic_orbitals_, alpha, Cb_vir_ptr,
                     num_molecular_orbitals_, temp.data(), num_beta, 1.0,
                     Y_rhs + nova, num_beta_virtual_orbitals);
        }
      }
    };

    // Create a preconditioner based on orbital energy differences
    // This diagonal preconditioner helps accelerate convergence significantly
    // Solve M(shift)Y = X, storing Y in X
    // M is approximated by diagonal elements (ϵaσ − ϵiτ)
    // This is effectively dividing each element by its corresponding orbital
    // energy difference
    auto precond_op = [&](int32_t N, int32_t NRHS, double shift, double* X,
                          int32_t LDX) {
      for (int32_t rhs = 0; rhs < NRHS; ++rhs) {
        double* X_rhs = X + rhs * LDX;

        // Apply the preconditioner for each element of X
        for (size_t i = 0; i < num_alpha; ++i)
          for (size_t a = 0; a < num_alpha_virtual_orbitals; ++a) {
            // Use orbital energy difference as preconditioner
            double energy_diff =
                eigenvalues_(0, a + num_alpha) - eigenvalues_(0, i);
            // Avoid division by very small numbers
            if (std::abs(energy_diff) > 1e-12)
              X_rhs[i * num_alpha_virtual_orbitals + a] /= energy_diff;
          }
        if (ctx_.cfg->unrestricted) {
          for (size_t i = 0; i < num_beta; ++i)
            for (size_t a = 0; a < num_beta_virtual_orbitals; ++a) {
              double energy_diff =
                  eigenvalues_(1, a + num_beta) - eigenvalues_(1, i);
              if (std::abs(energy_diff) > 1e-12)
                X_rhs[nova + i * num_beta_virtual_orbitals + a] /= energy_diff;
            }
        }
      }
    };

    // Call GMRES with both matrix operator and preconditioner
    matrix_op_t<double> A_op_func = A_op;
    shifted_precond_op_t<double> precond_op_func = precond_op;
    arnoldi_gmres(nov, 1, 1, A_op_func, precond_op_func, &shift, R_input, nov,
                  X_sol, nov, settings);
#ifdef QDK_CHEMISTRY_ENABLE_MPI
    signal = -1;  // Set signal to halt for workers
    MPI_Bcast(&signal, 1, MPI_INT, 0,
              MPI_COMM_WORLD);  // Broadcast halt signal to workers
#endif
  };

  if (ctx_.cfg->mpi.world_rank != 0) {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
    worker_callback();
#endif
  } else {
    // Execute root function in root process
    root_func();
  }
}

void SCFImpl::update_trial_fock_() {
  AutoTimer __timer("polarizability::  SCFImpl::update_trial_fock");
  auto [alpha, beta, omega] = get_hyb_coeff_();
  eri_->build_JK(tP_.data(), tJ_.data(), tK_.data(), alpha, beta, omega);

  if (ctx_.cfg->unrestricted) {
    tFock_.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_) =
        tJ_.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_) +
        tJ_.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                  num_atomic_orbitals_) -
        tK_.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_);
    tFock_.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                 num_atomic_orbitals_) =
        tJ_.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                  num_atomic_orbitals_) +
        tJ_.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_) -
        tK_.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                  num_atomic_orbitals_);
  } else
    tFock_ = 2 * tJ_ - tK_;
}

void KSImpl::update_trial_fock_() {
  AutoTimer __timer("polarizability::  KSImpl::update_trial_fock_");
  SCFImpl::update_trial_fock_();
  // tP is row-major but does not affect since tPa and tPb is symmetric here
  exc_->eval_fxc_contraction(P_.data(), tP_.data(), tXC_.data());
  tFock_ += tXC_;
}

}  // namespace qdk::chemistry::scf
