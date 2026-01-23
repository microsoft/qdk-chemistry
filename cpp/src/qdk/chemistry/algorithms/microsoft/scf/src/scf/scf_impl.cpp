// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf/scf_impl.h"

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/scf/scf_solver.h>
#include <qdk/chemistry/scf/util/gauxc_util.h>
#include <qdk/chemistry/scf/util/int1e.h>

#include <qdk/chemistry/utils/logger.hpp>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#include <qdk/chemistry/scf/util/env_helper.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <lapack.hh>
#include <nlohmann/json.hpp>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <sstream>
#include <thread>

#include "util/macros.h"
#include "util/timer.h"

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>

#include "util/gpu/matrix_op.h"
#endif

#include <qdk/chemistry/scf/eri/eri_multiplexer.h>

#include "scf/guess.h"

#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

namespace qdk::chemistry::scf {

SCFImpl::SCFImpl(std::shared_ptr<Molecule> mol_ptr, const SCFConfig& cfg,
                 std::shared_ptr<BasisSet> basis_set,
                 std::shared_ptr<BasisSet> raw_basis_set, bool delay_eri,
                 bool skip_verify) {
  QDK_LOG_TRACE_ENTERING();
  auto& mol = *mol_ptr;
  ctx_.mol = mol_ptr.get();
  ctx_.cfg = &cfg;
  if (basis_set == nullptr) {
    ctx_.basis_set =
        BasisSet::from_database_json(mol_ptr, cfg.basis, cfg.basis_mode,
                                     !cfg.cartesian /*pure*/, true /*sort*/);
  } else {
    ctx_.basis_set = basis_set;
  }
  if (cfg.do_dfj) {
    if (basis_set == nullptr) {
      ctx_.aux_basis_set =
          BasisSet::from_database_json(mol_ptr, cfg.aux_basis, cfg.basis_mode,
                                       !cfg.cartesian /*pure*/, true /*sort*/);
    } else {
      ctx_.aux_basis_set = basis_set;
    }
  }
  if (cfg.output_basis_mode == BasisMode::RAW) {
    // create an unnormalized raw basis for output only
    if (raw_basis_set == nullptr) {
      ctx_.basis_set_raw =
          BasisSet::from_database_json(mol_ptr, cfg.basis, BasisMode::RAW,
                                       !cfg.cartesian /*pure*/, true /*sort*/);
    } else {
      ctx_.basis_set_raw = raw_basis_set;
    }
  }
  ctx_.result = {};

  num_atomic_orbitals_ = ctx_.basis_set->num_atomic_orbitals;
  num_molecular_orbitals_ = ctx_.basis_set->num_atomic_orbitals;
  ctx_.num_molecular_orbitals = num_molecular_orbitals_;
  num_density_matrices_ = cfg.unrestricted ? 2 : 1;
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  add_mm_charge_ = cfg.pointcharges != nullptr;
#endif

  auto n_ecp_electrons = ctx_.basis_set->n_ecp_electrons;
  auto spin = mol.multiplicity - 1;
  auto alpha = (mol.n_electrons - n_ecp_electrons + spin) / 2;
  auto beta = mol.n_electrons - n_ecp_electrons - alpha;

  if (cfg.mpi.world_rank == 0) {
    std::string fock_string = "";
    if (cfg.do_dfj) {
      fock_string = "DFJ/";
      if (cfg.k_eri.method == ERIMethod::SnK)
        fock_string += "SnK";
      else
        fock_string += "TradK";
    } else {
      fock_string = "TradJ";
      if (cfg.k_eri.method == ERIMethod::SnK)
        fock_string += "/SnK";
      else
        fock_string += "K";
    }

    QDK_LOGGER().info(
        "mol: atoms={}, electrons={}, n_ecp_electrons={}, charge={}, "
        "multiplicity={}, spin(2S)={}, alpha={}, beta={}",
        mol.n_atoms, mol.n_electrons, n_ecp_electrons, mol.charge,
        mol.multiplicity, spin, alpha, beta);
    QDK_LOGGER().info(
        "restricted={}, basis={}, pure={}, num_atomic_orbitals={}, "
        "density_threshold={:.2e}, "
        "og_threshold={:.2e}",
        !cfg.unrestricted, ctx_.basis_set->name, ctx_.basis_set->pure,
        num_atomic_orbitals_, cfg.scf_algorithm.density_threshold,
        cfg.scf_algorithm.og_threshold);
    QDK_LOGGER().info("fock_alg={}", fock_string);
    if (cfg.do_dfj) {
      QDK_LOGGER().info("aux_basis={}, naux={}", ctx_.aux_basis_set->name,
                        ctx_.aux_basis_set->num_atomic_orbitals);
    }
    QDK_LOGGER().info("eri_tolerance={:.2e}", cfg.eri.eri_threshold);

#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
    QDK_LOGGER().info("disp={}", to_string(cfg.disp));
#endif

#ifdef QDK_CHEMISTRY_ENABLE_PCM
    QDK_LOGGER().info("enable_pcm={}, use_ddx={}", cfg.enable_pcm, cfg.use_ddx);
#endif

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
    QDK_LOGGER().info("qmmm={}", add_mm_charge_);
#endif

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    QDK_LOGGER().info("world_size={}, omp_get_max_threads={}",
                      cfg.mpi.world_size, nthreads);
  }
  if (cfg.verbose > 5) {
    QDK_LOGGER().info("eri_method={}, exc_method={}", to_string(cfg.eri.method),
                      to_string(cfg.exc.method));
  }

  if (!skip_verify) {
    VERIFY_INPUT(alpha >= 0 && beta >= 0 && beta == alpha - spin,
                 "Invalid spin number or charge");
    VERIFY_INPUT(num_density_matrices_ == 2 || alpha == beta,
                 "Restricted requires n_alpha == n_beta");
  }

  // MAX_N = 46340. Stop the calculation early if the basis set is too large
  // A single MAX_NxMAX_N matrix will have ~2^31 double floating point numbers
  // and will take about ~16GB memory
  const int MAX_N =
      static_cast<int>(std::floor(std::sqrt(std::numeric_limits<int>::max())));
  if (num_atomic_orbitals_ > MAX_N) {
    throw std::runtime_error(
        fmt::format("Basis set too large: {}", num_atomic_orbitals_));
  }

  nelec_[0] = alpha;
  nelec_[1] = beta;

  int1e_ = std::make_unique<OneBodyIntegral>(ctx_.basis_set.get(), ctx_.mol,
                                             cfg.mpi);
  if (not delay_eri) {
    if (cfg.do_dfj) {
      TIMEIT(eri_ = ERIMultiplexer::create(*ctx_.basis_set, *ctx_.aux_basis_set,
                                           cfg, 0.0),
             "SCFImpl::SCFImpl->ERI::create");
    } else {
      TIMEIT(eri_ = ERIMultiplexer::create(*ctx_.basis_set, cfg, 0.0),
             "SCFImpl::SCFImpl->ERI::create");
    }
  }

  // Host allocations for purely AO quantities
  P_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                            num_atomic_orbitals_);
  J_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                            num_atomic_orbitals_);
  K_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                            num_atomic_orbitals_);
  if (cfg.mpi.world_rank == 0) {
    F_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                              num_atomic_orbitals_);
    scf_algorithm_ = SCFAlgorithm::create(ctx_);
  }

  // MO and mixed AO/MO quantities
  // These may be resized after the orthonormality check
  C_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                            num_molecular_orbitals_);
  eigenvalues_ =
      RowMajorMatrix::Zero(num_density_matrices_, num_molecular_orbitals_);

#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  ctx_.result.scf_dispersion_correction_energy = 0.0;
  disp_grad_ = RowMajorMatrix::Zero(mol.n_atoms, 3);
#endif

#ifdef QDK_CHEMISTRY_ENABLE_PCM
  Vpcm_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                               num_atomic_orbitals_);
  ctx_.result.scf_pcm_energy = 0.0;
  if (ctx_.cfg->enable_pcm) {
    TIMEIT(pcm_ = pcm::PCM::create(*ctx_.basis_set, num_density_matrices_, cfg),
           "PCM::initialize");
  }
#endif

  if (cfg.require_polarizability) {
    // Host allocations for CPSCF/TDDFT quantities
    tP_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                               num_atomic_orbitals_);
    tJ_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                               num_atomic_orbitals_);
    tK_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                               num_atomic_orbitals_);
    tFock_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                                  num_atomic_orbitals_);
  }
}

SCFImpl::SCFImpl(std::shared_ptr<Molecule> mol_ptr, const SCFConfig& cfg,
                 bool delay_eri)
    : SCFImpl(mol_ptr, cfg, nullptr, nullptr, delay_eri) {}

SCFImpl::SCFImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
                 const RowMajorMatrix& dm, std::shared_ptr<BasisSet> basis_set,
                 std::shared_ptr<BasisSet> raw_basis_set, bool delay_eri)
    : SCFImpl(mol, cfg, basis_set, raw_basis_set, delay_eri) {
  QDK_LOG_TRACE_ENTERING();
  VERIFY(dm.rows() == num_density_matrices_ * num_atomic_orbitals_ &&
         dm.cols() == num_atomic_orbitals_);
  P_ = dm;
  density_matrix_initialized_ = true;
}

const SCFContext& SCFImpl::run() {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  AutoTimer timer("SCF::run");
  const auto& cfg = *ctx_.cfg;

  iterate_();
  properties_();

  // Print Summary
  if (cfg.verbose > 3 && cfg.mpi.world_rank == 0) {
    const auto& res = ctx_.result;
    std::ostringstream oss;
    oss << fmt::format("{:-^65}\n", "");
    oss << fmt::format("Nuclear Repulsion Energy =         {:20.12f}\n",
                       res.nuclear_repulsion_energy);
    oss << fmt::format("One-Electron Energy =              {:20.12f}\n",
                       res.scf_one_electron_energy);
    oss << fmt::format("Two-Electron Energy =              {:20.12f}\n",
                       res.scf_two_electron_energy);
    oss << fmt::format("DFT Exchange-Correlation Energy =  {:20.12f}\n",
                       res.scf_xc_energy);
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
    oss << fmt::format("Dispersion Correction Energy =     {:20.12f}\n",
                       res.scf_dispersion_correction_energy);
#endif
#ifdef QDK_CHEMISTRY_ENABLE_PCM
    oss << fmt::format("PCM Polarization Energy =          {:20.12f}\n",
                       res.scf_pcm_energy);
#endif
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
    oss << fmt::format("QMMM Coulomb Energy =              {:20.12f}\n",
                       res.qmmm_coulomb_energy);
    oss << fmt::format("MM Self Energy =                   {:20.12f}\n",
                       res.mm_self_energy);
#endif
    oss << fmt::format("Total Energy =                     {:20.12f}\n",
                       res.scf_total_energy);
    oss << std::endl;
    oss << fmt::format("Total Dipole (a.u.)\n");
    oss << fmt::format("         X_ =                       {:20.12f}\n",
                       res.scf_dipole[0]);
    oss << fmt::format("         Y =                       {:20.12f}\n",
                       res.scf_dipole[1]);
    oss << fmt::format("         Z =                       {:20.12f}\n",
                       res.scf_dipole[2]);
    oss << std::endl;
    oss << fmt::format("Mulliken Charges (a.u.)\n");
    for (auto A = 0; A < ctx_.mol->n_atoms; ++A) {
      oss << fmt::format(" Atom {:>5} Z={:>3}                    {:20.8e}\n", A,
                         ctx_.mol->atomic_nums[A], res.mulliken_population[A]);
    }
    oss << fmt::format("{:-^65}", "");
    QDK_LOGGER().info("SCF converged: steps={}, E={:.12f}\n{}",
                      res.scf_iterations, res.scf_total_energy, oss.str());
  }

  // Compute Gradient
  if (cfg.require_gradient) {
    if (cfg.mpi.world_rank == 0) {
      QDK_LOGGER().info("Calculating gradient");
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
      if (add_mm_charge_) {
        QDK_LOGGER().info(
            "Calculating molecules' and point charges' gradients with respect "
            "to each other");
      }
#endif
    }
    const auto& grad = get_gradients_();
    const auto& mol = *ctx_.mol;
    if (cfg.verbose > 3 && cfg.mpi.world_rank == 0) {
      write_gradients_(grad, &mol);
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
      if (add_mm_charge_) {
        write_gradients_(ctx_.result.scf_point_charge_gradient);
      }
#endif
    }
  }

  // Compute polarizability
  if (cfg.require_polarizability) {
    if (cfg.mpi.world_rank == 0) {
      QDK_LOGGER().info("Calculating Static polarizability");
    }
    polarizability_();
  }

  return ctx_;
}

void SCFImpl::update_fock_() {
  QDK_LOG_TRACE_ENTERING();
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  if (pcm_ != nullptr) {
    if (ctx_.cfg->mpi.world_rank == 0) {
      F_ -= Vpcm_;
    }
    TIMEIT(pcm_->compute_PCM_terms(P_.data(), Vpcm_.data(),
                                   &ctx_.result.scf_pcm_energy),
           "PCM::compute_PCM_terms");
    if (ctx_.cfg->mpi.world_rank == 0) {
      F_ += Vpcm_;
    }
  }
#endif

  if (ctx_.cfg->mpi.world_rank == 0) {
    if (ctx_.cfg->unrestricted) {
      F_ += (J_.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_) +
             J_.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                      num_atomic_orbitals_))
                .replicate(2, 1) -
            K_;
    } else {
      F_ += J_ - 0.5 * K_;
    }
  }
}

void SCFImpl::reset_fock_() {
  QDK_LOG_TRACE_ENTERING();
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  F_ = H_ + Vpcm_;
#else
  F_ = H_;
#endif
}

double SCFImpl::total_energy_() {
  QDK_LOG_TRACE_ENTERING();
  auto& res = ctx_.result;
  res.scf_one_electron_energy = P_.cwiseProduct(H_).sum();
  res.scf_two_electron_energy = 0.5 * P_.cwiseProduct(F_ - H_).sum();
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  res.scf_two_electron_energy -= 0.5 * P_.cwiseProduct(Vpcm_).sum();
#endif

  double _total = res.nuclear_repulsion_energy + res.scf_one_electron_energy +
                  res.scf_two_electron_energy;
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  _total += res.scf_pcm_energy;
#endif
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  _total += res.scf_dispersion_correction_energy;
#endif
  return _total;
}

void SCFImpl::compute_orthogonalization_matrix_(const RowMajorMatrix& S_,
                                                RowMajorMatrix* ret) {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif

  RowMajorMatrix U_t(num_atomic_orbitals_, num_atomic_orbitals_);
  RowMajorMatrix s(num_atomic_orbitals_, 1);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  auto U_d = cuda::alloc<double>(num_atomic_orbitals_ * num_atomic_orbitals_);
  CUDA_CHECK(
      cudaMemcpy(U_d->data(), S_.data(),
                 sizeof(double) * num_atomic_orbitals_ * num_atomic_orbitals_,
                 cudaMemcpyHostToDevice));
  auto s_d = cuda::alloc<double>(num_atomic_orbitals_ * num_atomic_orbitals_);
  cusolver::ManagedcuSolverHandle handle;
  cusolver::syevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                  num_atomic_orbitals_, U_d->data(), num_atomic_orbitals_,
                  s_d->data());

  CUDA_CHECK(
      cudaMemcpy(U_t.data(), U_d->data(),
                 sizeof(double) * num_atomic_orbitals_ * num_atomic_orbitals_,
                 cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(s.data(), s_d->data(),
                        sizeof(double) * num_atomic_orbitals_,
                        cudaMemcpyDeviceToHost));
#else
  std::memcpy(U_t.data(), S_.data(),
              num_atomic_orbitals_ * num_atomic_orbitals_ * sizeof(double));
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, num_atomic_orbitals_,
               U_t.data(), num_atomic_orbitals_, s.data());
#endif

  RowMajorMatrix U = U_t.transpose();

  auto threshold = ctx_.cfg->lindep_threshold;
  if (threshold < 0.0) threshold = s.maxCoeff() / 1e9;

  num_molecular_orbitals_ = 0;
  for (int i = num_atomic_orbitals_ - 1; i >= 0; --i) {
    if (s(i) >= threshold) num_molecular_orbitals_++;
  }

  if (num_atomic_orbitals_ != num_molecular_orbitals_) {
    QDK_LOGGER().warn(
        "Orthogonalize: found linear dependency TOL={:.2e} "
        "num_atomic_orbitals_={} "
        "num_molecular_orbitals_={}",
        threshold, num_atomic_orbitals_, num_molecular_orbitals_);
  }

  auto sigma = s.bottomRows(num_molecular_orbitals_);
  auto sigma_invsqrt = sigma.array().sqrt().inverse().matrix().asDiagonal();

  auto U_cond = U.block(0, num_atomic_orbitals_ - num_molecular_orbitals_,
                        num_atomic_orbitals_, num_molecular_orbitals_);
  RowMajorMatrix X_ = U_cond * sigma_invsqrt;

  // Optional validation of orthonormality (disabled by default)
  const bool verify_orthonormality = false;
  if (verify_orthonormality) {
    RowMajorMatrix should_be_I = X_.transpose() * S_ * X_;
    auto norm = (should_be_I - RowMajorMatrix::Identity(should_be_I.rows(),
                                                        should_be_I.cols()))
                    .norm();
    VERIFY("Orthogonalize: ||X_^t * S_ * X_ - I||_2 should be zero" &&
           norm < 1e-10);
  }
  *ret = X_;
}

void SCFImpl::iterate_() {
  QDK_LOG_TRACE_ENTERING();
  auto cfg = ctx_.cfg;
  VERIFY_INPUT(cfg->incremental_fock_start_step > 0,
               "incremental_fock_start_step must be positive");
  VERIFY_INPUT(cfg->fock_reset_steps > 0, "fock_reset_steps must be positive");

  auto& res = ctx_.result;

  // Handle zero-electron systems (e.g., H+)
  // For these systems, the SCF energy is just the nuclear repulsion energy
  // with zero electronic contribution
  if (nelec_[0] == 0 && nelec_[1] == 0) {
    build_one_electron_integrals_();
    res.nuclear_repulsion_energy = calc_nuclear_repulsion_energy_();
    res.scf_one_electron_energy = 0.0;
    res.scf_two_electron_energy = 0.0;
    res.scf_total_energy = res.nuclear_repulsion_energy;
    res.scf_iterations = 0;
    res.converged = true;
    // Initialize coefficient matrix with identity (virtual orbitals only)
    C_ = RowMajorMatrix::Identity(num_molecular_orbitals_,
                                  num_molecular_orbitals_);
    // Initialize eigenvalues to zero for zero-electron system
    eigenvalues_ =
        RowMajorMatrix::Zero(num_density_matrices_, num_molecular_orbitals_);
    QDK_LOGGER().info(
        "Zero-electron system detected. SCF energy = nuclear repulsion = "
        "{:.12f}",
        res.scf_total_energy);
    return;
  }

  build_one_electron_integrals_();

  if (ctx_.cfg->mpi.world_rank == 0) {
    init_density_matrix_();
    res.nuclear_repulsion_energy = calc_nuclear_repulsion_energy_();

#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
    if (ctx_.cfg->disp != DispersionType::None) {
      dftd3_wrapper(*ctx_.mol, ctx_.cfg->exc.xc_name, ctx_.cfg->disp,
                    false /*atm*/, &res.scf_dispersion_correction_energy,
                    disp_grad_.data());
      QDK_LOGGER().debug("Dispersion energy: {}",
                         res.scf_dispersion_correction_energy);
    }
#endif
  }

#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{0, 0, 255}, "SCF::iterate"};
#endif
  Timer::start_timing("SCF::iterate");

  double energy_last = 0;
  RowMajorMatrix P_diff;
  RowMajorMatrix P_last = RowMajorMatrix::Zero(
      num_density_matrices_ * num_atomic_orbitals_, num_atomic_orbitals_);

  for (auto step = 0; step < cfg->scf_algorithm.max_iteration; ++step) {
#ifdef ENABLE_NVTX3
    nvtx3::scoped_range r{nvtx3::rgb{0, 0, 255}, "SCF::iterate_step"};
#endif
    AutoTimer timer("SCF::iterate_step");
#ifdef QDK_CHEMISTRY_ENABLE_MPI
    TIMEIT(MPI_Bcast(P_.data(),
                     num_density_matrices_ * num_atomic_orbitals_ *
                         num_atomic_orbitals_,
                     MPI_DOUBLE, 0, MPI_COMM_WORLD),
           "MPI_Bcast(P_)");
#endif

    if (step < cfg->incremental_fock_start_step ||
        step % cfg->fock_reset_steps == 0) {
      P_diff = P_;
      if (cfg->mpi.world_rank == 0) {
        QDK_LOGGER().info("Reset incremental Fock matrix");
        reset_fock_();
      }
    } else {
      P_diff = P_ - P_last;
    }
    P_last = P_;

    auto [alpha, beta, omega] = get_hyb_coeff_();
    eri_->build_JK(P_diff.data(), J_.data(), K_.data(), alpha, beta, omega);

    update_fock_();

    res.scf_total_energy = total_energy_();

    if (cfg->mpi.world_rank == 0) {
      if (std::isnan(res.scf_total_energy) or std::isinf(res.scf_total_energy))
        throw std::runtime_error("NaN or INF Encountered in SCF Energy");

      // Check convergence
      res.converged = scf_algorithm_->check_convergence(*this);
      // Perform SCF Algorithm iteration only if not converged
      scf_algorithm_->iterate(*this);

      res.scf_iterations = step + 1;
    }
#ifdef QDK_CHEMISTRY_ENABLE_MPI
    TIMEIT(MPI_Bcast(&res.converged, 1, MPI_INT8_T, 0, MPI_COMM_WORLD),
           "MPI_Bcast(converged)");
#endif

    if (res.converged) {
      break;
    }
  }
  if (!res.converged) {
    throw std::runtime_error(
        fmt::format("SCF failed to converge after {} steps",
                    cfg->scf_algorithm.max_iteration));
  }
  // update eigenvalues for GDM as GDM never updates eigenvalues_
  if (ctx_.cfg->scf_algorithm.method == SCFAlgorithmName::GDM ||
      ctx_.cfg->scf_algorithm.method == SCFAlgorithmName::DIIS_GDM) {
    if (cfg->mpi.world_rank == 0) {
      QDK_LOGGER().info("Reset incremental Fock matrix");
      reset_fock_();
    }
    auto [alpha, beta, omega] = get_hyb_coeff_();
    eri_->build_JK(P_.data(), J_.data(), K_.data(), alpha, beta, omega);
    update_fock_();
    for (int i = 0; i < num_density_matrices_; ++i) {
      scf_algorithm_->solve_fock_eigenproblem(
          F_, S_, X_, C_, eigenvalues_, P_, nelec_, num_atomic_orbitals_,
          num_molecular_orbitals_, i, ctx_.cfg->unrestricted);
    }
  }
}

void SCFImpl::properties_() {
  QDK_LOG_TRACE_ENTERING();
  auto& res = ctx_.result;

  /****** Multipoles *******/
  {
    // Compute Dipole Integrals
    RowMajorMatrix dipole(3 * num_atomic_orbitals_, num_atomic_orbitals_);
    int1e_->dipole_integral(dipole.data());

    // Compute elecric dipole
    std::array<double, 3> elec_dipole;
    auto dot = [](size_t n, const auto* a, const auto* b) {
      auto res = a[0] * b[0];
      for (auto i = 1; i < n; ++i) res += a[i] * b[i];
      return res;
    };
    for (auto i = 0; i < 3; ++i) {
      elec_dipole[i] =
          -dot(num_atomic_orbitals_ * num_atomic_orbitals_,
               dipole.data() + i * num_atomic_orbitals_ * num_atomic_orbitals_,
               P_.data());
      if (num_density_matrices_ == 2) {
        elec_dipole[i] += -dot(
            num_atomic_orbitals_ * num_atomic_orbitals_,
            dipole.data() + i * num_atomic_orbitals_ * num_atomic_orbitals_,
            P_.data() + num_atomic_orbitals_ * num_atomic_orbitals_);
      }
    }

    // Compute nuclear dipole
    std::array<double, 6> nuclear_dipole = {0, 0, 0};
    for (auto A = 0; A < ctx_.mol->n_atoms; ++A) {
      auto Z = ctx_.mol->atomic_charges[A];
      auto coords = ctx_.mol->coords[A];
      nuclear_dipole[0] += Z * coords[0];
      nuclear_dipole[1] += Z * coords[1];
      nuclear_dipole[2] += Z * coords[2];
    }

    // Combine for total dipole
    for (int i = 0; i < 3; ++i) {
      res.scf_dipole[i] = elec_dipole[i] + nuclear_dipole[i];
    }
  }

  /****** Population Analysis ******/
  {
    RowMajorMatrix PS(num_atomic_orbitals_, num_atomic_orbitals_);
    if (num_density_matrices_ == 1) {
      PS.noalias() = P_ * S_;
    } else {
      Eigen::Map<RowMajorMatrix> P_alpha(P_.data(), num_atomic_orbitals_,
                                         num_atomic_orbitals_);
      Eigen::Map<RowMajorMatrix> P_beta(
          P_.data() + num_atomic_orbitals_ * num_atomic_orbitals_,
          num_atomic_orbitals_, num_atomic_orbitals_);
      PS.noalias() = (P_alpha + P_beta) * S_;
    }
    res.mulliken_population.resize(ctx_.mol->n_atoms);
    for (auto A = 0; A < ctx_.mol->n_atoms; ++A) {
      auto Z = ctx_.mol->atomic_charges[A];
      res.mulliken_population[A] = Z;
    }

    for (auto i = 0; i < num_atomic_orbitals_; ++i) {
      int A = 0;
      for (A = 0; A < ctx_.mol->n_atoms; ++A) {
        if (ctx_.basis_set->get_atom2ao()[A * num_atomic_orbitals_ + i]) break;
      }
      res.mulliken_population[A] -= PS(i, i);
    }
  }
}

static void add_nuclear_repulsion_grad(double* dE, uint64_t n,
                                       const Molecule* mol);

std::tuple<double, double, double> SCFImpl::get_hyb_coeff_() const {
  QDK_LOG_TRACE_ENTERING();
  return std::make_tuple(1.0, 0.0, 0.0);
}

const std::vector<double>& SCFImpl::get_gradients_() {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{255, 128, 0}, "SCFImpl::get_gradients"};
#endif
  Timer::start_timing("SCF::gradient");
  auto& mol = ctx_.mol;

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  TIMEIT(MPI_Bcast(P_.data(),
                   num_density_matrices_ * num_atomic_orbitals_ *
                       num_atomic_orbitals_,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD),
         "MPI_Bcast(P_)");
#endif
  auto n_atoms = mol->n_atoms;
  RowMajorMatrix dE = RowMajorMatrix::Zero(3, n_atoms);
  RowMajorMatrix d_pc;
  if (ctx_.cfg->mpi.world_rank == 0) {
    for (uint64_t n = 0; n < n_atoms; ++n) {
      add_nuclear_repulsion_grad(dE.data(), n, mol);
    }
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
    if (add_mm_charge_) {
      d_pc = RowMajorMatrix::Zero(3, ctx_.cfg->pointcharges.get()->n_points);
      add_qmmm_coulomb_grad(dE.data(), d_pc.data(), mol,
                            ctx_.cfg->pointcharges.get());
    }
#endif
  }
  {
#ifdef ENABLE_NVTX3
    nvtx3::scoped_range r{nvtx3::rgb{255, 128, 0}, "int1e_grad"};
#endif
    AutoTimer timer("SCFImpl::get_gradients->int1e");
    RowMajorMatrix d_kinetic = RowMajorMatrix::Zero(3, n_atoms);
    RowMajorMatrix d_nuclear = RowMajorMatrix::Zero(3, n_atoms);
    RowMajorMatrix d_overlap = RowMajorMatrix::Zero(3, n_atoms);

    RowMajorMatrix P2 =
        num_density_matrices_ == 1
            ? P_
            : P_.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_) +
                  P_.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                           num_atomic_orbitals_);
    int1e_->kinetic_integral_deriv(P2.data(), d_kinetic.data());
    int1e_->nuclear_integral_deriv(P2.data(), d_nuclear.data());
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
    if (add_mm_charge_) {
      RowMajorMatrix d_pointcharge_mm = RowMajorMatrix::Zero(3, n_atoms);
      RowMajorMatrix d_pointcharge_pc =
          RowMajorMatrix::Zero(3, ctx_.cfg->pointcharges.get()->n_points);
      int1e_->pointcharge_integral_deriv(P2.data(), d_pointcharge_mm.data(),
                                         d_pointcharge_pc.data(),
                                         ctx_.cfg->pointcharges.get());
      d_nuclear += d_pointcharge_mm;  // part of electron gradient
      d_pc += d_pointcharge_pc;       // part of MM gradient
    }
#endif
    RowMajorMatrix W =
        RowMajorMatrix::Zero(num_atomic_orbitals_, num_atomic_orbitals_);
    if (ctx_.cfg->mpi.world_rank == 0) {
      for (auto i = 0; i < num_density_matrices_; ++i) {
        W -= P_.block(i * num_atomic_orbitals_, 0, num_atomic_orbitals_,
                      num_atomic_orbitals_) *
             F_.block(i * num_atomic_orbitals_, 0, num_atomic_orbitals_,
                      num_atomic_orbitals_) *
             P_.block(i * num_atomic_orbitals_, 0, num_atomic_orbitals_,
                      num_atomic_orbitals_);
      }
      if (num_density_matrices_ == 1) {
        W *= 0.5;
      }
    }
#ifdef QDK_CHEMISTRY_ENABLE_MPI
    MPI_Bcast(W.data(), num_atomic_orbitals_ * num_atomic_orbitals_, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
#endif
    int1e_->overlap_integral_deriv(W.data(), d_overlap.data());
    dE += d_kinetic + d_nuclear + d_overlap;

    if (ctx_.basis_set->ecp_shells.size() > 0) {
      RowMajorMatrix d_ecp = RowMajorMatrix::Zero(3, n_atoms);
      int1e_->ecp_integral_deriv(P2.data(), d_ecp.data());
      dE += d_ecp;
    }
  }

  auto [alpha, beta, omega] = get_hyb_coeff_();
  RowMajorMatrix dJ = RowMajorMatrix::Zero(3, n_atoms);
  RowMajorMatrix dK = RowMajorMatrix::Zero(3, n_atoms);
  QDK_LOGGER().trace("Calculating ERI gradients");
  eri_->get_gradients(P_.data(), dJ.data(), dK.data(), alpha, beta, omega);
  dE += dJ + dK;

  dE += get_vxc_grad_();

#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  if (ctx_.cfg->mpi.world_rank == 0 && ctx_.cfg->disp != DispersionType::None) {
    dE += disp_grad_.transpose();
  }
#endif

  if (ctx_.cfg->mpi.world_rank == 0) {
    dE.transposeInPlace();
    ctx_.result.scf_total_gradient =
        std::vector<double>(dE.data(), dE.data() + 3 * n_atoms);
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
    if (add_mm_charge_) {
      d_pc.transposeInPlace();
      ctx_.result.scf_point_charge_gradient = std::vector<double>(
          d_pc.data(),
          d_pc.data() + 3 * ctx_.cfg->pointcharges.get()->n_points);
    }
#endif
  }
  return ctx_.result.scf_total_gradient;
}

void SCFImpl::init_density_matrix_() {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  AutoTimer timer("SCFImpl::init_density_matrix");
  auto& mol = *ctx_.mol;
  auto method = ctx_.cfg->density_init_method;
  if (method == DensityInitializationMethod::SOAD) {
    soad_initialize_density_matrix(P_.data(), num_atomic_orbitals_,
                                   mol.atomic_nums.data(), mol.n_atoms);
    P_ *= 2.0;
  } else if (method == DensityInitializationMethod::Core) {
    // Use the SCF algorithm to update density matrix for each spin
    for (int i = 0; i < num_density_matrices_; ++i) {
      scf_algorithm_->solve_fock_eigenproblem(
          H_, S_, X_, C_, eigenvalues_, P_, nelec_, num_atomic_orbitals_,
          num_molecular_orbitals_, i, ctx_.cfg->unrestricted);
    }
  } else if (method == DensityInitializationMethod::Atom) {
    atom_guess(*ctx_.basis_set, mol, P_.data());
  } else if (method == DensityInitializationMethod::File) {
    std::ifstream ifsDM(ctx_.cfg->density_init_file, std::ios::binary);
    if (ifsDM.is_open()) {
      ifsDM.read((char*)P_.data(), P_.size() * sizeof(double));
      if (ctx_.cfg->mpi.world_rank == 0) {
        QDK_LOGGER().info("Guess read from file: {}",
                          ctx_.cfg->density_init_file);
      }
    } else {
      if (ctx_.cfg->mpi.world_rank == 0) {
        QDK_LOGGER().error("Failed to open dm file: {}",
                           ctx_.cfg->density_init_file);
        exit(EXIT_FAILURE);
      }
    }
  } else if (method == DensityInitializationMethod::UserProvided) {
    // Check if density matrix has already been initialized directly
    if (!density_matrix_initialized_) {
      throw std::runtime_error(
          "DensitiyInitializationMethod::UserProvided requires an input "
          "density matrix, but no density matrix has been set.");
    }
  }

  if (num_density_matrices_ == 2 &&
      (method != DensityInitializationMethod::UserProvided &&
       method != DensityInitializationMethod::File)) {
    memcpy(P_.data() + num_atomic_orbitals_ * num_atomic_orbitals_, P_.data(),
           sizeof(double) * num_atomic_orbitals_ * num_atomic_orbitals_);
    P_ *= 0.5;
    if (nelec_[0] == nelec_[1]) {  // spin(2S) = alpha - beta = 0
      QDK_LOGGER().warn(
          "Breaking symmetry not implemented for spin 0 molecule");
    }
  }
}

void SCFImpl::build_one_electron_integrals_() {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  Timer::start_timing("SCF::int1e");

  S_ = RowMajorMatrix::Zero(num_atomic_orbitals_, num_atomic_orbitals_);
  TIMEIT(int1e_->overlap_integral(S_.data()),
         "SCFImpl::build_one_electron_integrals->overlap_integral");
  if (ctx_.cfg->mpi.world_rank == 0) {
    TIMEIT(compute_orthogonalization_matrix_(S_, &X_),
           "SCFImpl::build_one_electron_integrals->compute_orthogonalization_"
           "matrix");
    num_molecular_orbitals_ = X_.cols();  // Reset num_molecular_orbitals_ based
                                          // on the rank of the overlap matrix
    ctx_.num_molecular_orbitals =
        num_molecular_orbitals_;  // Update the context to ensure proper
                                  // serialization
    // Resize MO quantities
    C_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                              num_molecular_orbitals_);
    eigenvalues_ =
        RowMajorMatrix::Zero(num_density_matrices_, num_molecular_orbitals_);
  }
  RowMajorMatrix T =
      RowMajorMatrix::Zero(num_atomic_orbitals_, num_atomic_orbitals_);
  RowMajorMatrix V =
      RowMajorMatrix::Zero(num_atomic_orbitals_, num_atomic_orbitals_);
  TIMEIT(int1e_->kinetic_integral(T.data()),
         "SCFImpl::build_one_electron_integrals->kinetic_integral");
  TIMEIT(int1e_->nuclear_integral(V.data()),
         "SCFImpl::build_one_electron_integrals->nuclear_integral");

  RowMajorMatrix ECP =
      RowMajorMatrix::Zero(num_atomic_orbitals_, num_atomic_orbitals_);
  if (ctx_.basis_set->ecp_shells.size() > 0) {
    TIMEIT(int1e_->ecp_integral(ECP.data()),
           "SCFImpl::build_one_electron_integrals->ecp_integral");
  }

  H_ = (T + V + ECP).replicate(num_density_matrices_, 1);
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  if (add_mm_charge_) {
    T_mm_ = RowMajorMatrix::Zero(num_atomic_orbitals_, num_atomic_orbitals_);
    TIMEIT(int1e_->point_charge_integral(ctx_.cfg->pointcharges.get(),
                                         T_mm_.data()),
           "SCFImpl::build_one_electron_integrals->point_charge_integral");
    H_ += T_mm_.replicate(num_density_matrices_, 1);
  }
#endif
}

double SCFImpl::calc_nuclear_repulsion_energy_() {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  const auto& mol = *ctx_.mol;
  double nre = 0.0;
  for (uint64_t i = 0; i < mol.n_atoms; ++i) {
    for (uint64_t j = i + 1; j < mol.n_atoms; ++j) {
      nre += mol.atomic_charges[i] * mol.atomic_charges[j] /
             sqrt((mol.coords[i][0] - mol.coords[j][0]) *
                      (mol.coords[i][0] - mol.coords[j][0]) +
                  (mol.coords[i][1] - mol.coords[j][1]) *
                      (mol.coords[i][1] - mol.coords[j][1]) +
                  (mol.coords[i][2] - mol.coords[j][2]) *
                      (mol.coords[i][2] - mol.coords[j][2]));
    }
  }
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  if (add_mm_charge_) {
    ctx_.result.qmmm_coulomb_energy =
        calc_qmmm_coulomb_energy(mol, *ctx_.cfg->pointcharges);
    ctx_.result.mm_self_energy = calc_mm_self_energy(*ctx_.cfg->pointcharges);
    nre = nre + ctx_.result.qmmm_coulomb_energy + ctx_.result.mm_self_energy;
  }
#endif
  return nre;
}

const RowMajorMatrix SCFImpl::get_vxc_grad_() const {
  QDK_LOG_TRACE_ENTERING();
  return RowMajorMatrix::Zero(3, ctx_.mol->n_atoms);
}

static void add_nuclear_repulsion_grad(double* dE, uint64_t n,
                                       const Molecule* mol) {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  auto n_atoms = mol->n_atoms;
  auto coord_n = mol->coords[n];

  for (uint64_t m = 0; m < n_atoms; ++m) {
    auto& coord_m = mol->coords[m];
    double d[] = {coord_n[0] - coord_m[0], coord_n[1] - coord_m[1],
                  coord_n[2] - coord_m[2]};
    double rab = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

    if (rab > 1e-10) {
      auto Za = mol->atomic_charges[n];
      auto Zb = mol->atomic_charges[m];
      auto Zab = Za * Zb / (rab * rab * rab);

      for (uint64_t dir = 0; dir < 3; ++dir) {
        dE[n_atoms * dir + n] -= d[dir] * Zab;
      }
    }
  }
}

void SCFImpl::write_gradients_(const std::vector<double>& gradients,
                               const Molecule* mol) {
  QDK_LOG_TRACE_ENTERING();
  std::ostringstream oss;
  oss << fmt::format("{:-^47}\n", "");
  oss << fmt::format("{:8} {:>15} {:>15} {:>15}\n", "", "x", "y", "z");
  if (mol != nullptr) {
    for (size_t i = 0; i < mol->n_atoms; ++i) {
      oss << fmt::format("Z={:<5} {:<2} {:15.10f} {:15.10f} {:15.10f}\n", i,
                         mol->atomic_nums[i], gradients[3 * i],
                         gradients[3 * i + 1], gradients[3 * i + 2]);
    }
  } else {
    for (size_t i = 0; i < gradients.size() / 3; ++i) {
      oss << fmt::format("{:<5} {:<2} {:12.10f} {:12.10f} {:12.10f}\n", i, "",
                         gradients[3 * i], gradients[3 * i + 1],
                         gradients[3 * i + 2]);
    }
  }
  oss << fmt::format("{:-^47}", "");
  if (mol != nullptr) {
    QDK_LOGGER().info("Molecule gradients:\n{}", oss.str());
  } else {
    QDK_LOGGER().info("Point Charge gradients:\n{}", oss.str());
  }
}

std::pair<double, RowMajorMatrix>
SCFImpl::evaluate_trial_density_energy_and_fock(
    const RowMajorMatrix& P_matrix, const std::source_location& loc) const {
  QDK_LOG_TRACE_ENTERING();

  QDK_LOGGER().debug(
      "Computing energy and Fock matrix by trial density matrix (called from "
      "function '{}' at line {})",
      loc.function_name(), loc.line());
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  AutoTimer timer("SCFImpl::evaluate_trial_density_energy_and_fock");

  RowMajorMatrix H_matrix = H_.eval();
  RowMajorMatrix F_matrix = H_matrix;
  auto [alpha, beta, omega] = get_hyb_coeff_();
  RowMajorMatrix J_matrix = RowMajorMatrix::Zero(
      num_density_matrices_ * num_atomic_orbitals_, num_atomic_orbitals_);
  RowMajorMatrix K_matrix = RowMajorMatrix::Zero(
      num_density_matrices_ * num_atomic_orbitals_, num_atomic_orbitals_);
  eri_->build_JK(P_matrix.data(), J_matrix.data(), K_matrix.data(), alpha, beta,
                 omega);
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  throw std::runtime_error("PCM is not supported in trial density evaluation.");
#endif

  if (ctx_.cfg->mpi.world_rank == 0) {
    if (ctx_.cfg->unrestricted) {
      F_matrix +=
          (J_matrix.block(0, 0, num_atomic_orbitals_, num_atomic_orbitals_) +
           J_matrix.block(num_atomic_orbitals_, 0, num_atomic_orbitals_,
                          num_atomic_orbitals_))
              .replicate(2, 1) -
          K_matrix;
    } else {
      F_matrix += J_matrix - 0.5 * K_matrix;
    }
  }

  double scf_one_electron_energy = P_matrix.cwiseProduct(H_matrix).sum();
  double scf_two_electron_energy =
      0.5 * P_matrix.cwiseProduct(F_matrix - H_matrix).sum();
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  scf_two_electron_energy -= 0.5 * P_matrix.cwiseProduct(Vpcm_matrix).sum();
#endif

  double total_energy = ctx_.result.nuclear_repulsion_energy +
                        scf_one_electron_energy + scf_two_electron_energy;
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  total_energy += scf_pcm_energy;
#endif
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  throw std::runtime_error(
      "DFT-D3 is not supported in trial density evaluation.");
#endif
  QDK_LOGGER().debug(
      "MPI world rank: {}, nuclear_repulsion_energy: {:.10e}, "
      "one_electron_energy: {:.10e}, two_electron_energy: {:.10e}",
      ctx_.cfg->mpi.world_rank, ctx_.result.nuclear_repulsion_energy,
      scf_one_electron_energy, scf_two_electron_energy);
  return {total_energy, F_matrix};
}

}  // namespace qdk::chemistry::scf
