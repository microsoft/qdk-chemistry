// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "stability.hpp"

#include <qdk/chemistry/scf/core/exc.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <spdlog/spdlog.h>

#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

// Local implementation details
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

namespace detail {
/**
 * @brief Compute trial Fock matrix for stability analysis
 *
 * Similar to KSImpl::update_trial_fock_(), this function builds the
 * trial Fock matrix using ERI and optionally XC contributions.
 * Only calls exc->eval_fxc_contraction when exc is not null (i.e., method
 * is not HF).
 *
 * @param eri The ERI multiplexer for computing J and K matrices
 * @param exc The exchange-correlation object (nullptr for HF)
 * @param trial_density The trial/perturbed density matrix
 * @param ground_density The ground state density matrix
 * @param trial_fock Output trial Fock matrix
 * @param J_scratch Scratch matrix for J contributions (also used for XC)
 * @param K_scratch Scratch matrix for K contributions
 */
void compute_trial_fock(const qcs::ERIMultiplexer& eri,
                        const std::shared_ptr<qcs::EXC>& exc,
                        const RowMajorMatrix& trial_density,
                        const RowMajorMatrix& ground_density,
                        RowMajorMatrix& trial_fock, RowMajorMatrix& J_scratch,
                        RowMajorMatrix& K_scratch) {
  const size_t num_atomic_orbitals = ground_density.cols();
  const bool unrestricted = (ground_density.rows() == 2 * num_atomic_orbitals);

  // Get hybrid coefficients (0,0,0 for HF)
  double alpha = 1.0, beta = 0.0, omega = 0.0;
  if (exc) {
    std::tie(alpha, beta, omega) = exc->get_hyb();
  }

  // Build J and K matrices
  J_scratch.setZero();
  K_scratch.setZero();
  eri.build_JK(trial_density.data(), J_scratch.data(), K_scratch.data(), alpha,
               beta, omega);

  // Compute Fock matrix: F = J - K for RHF, or appropriate combination for UHF
  if (unrestricted) {
    // For UHF: Fa = Ja + Jb - Ka, Fb = Ja + Jb - Kb
    trial_fock.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) =
        J_scratch.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) +
        J_scratch.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                        num_atomic_orbitals) -
        K_scratch.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
    trial_fock.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                     num_atomic_orbitals) =
        J_scratch.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                        num_atomic_orbitals) +
        J_scratch.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) -
        K_scratch.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                        num_atomic_orbitals);
  } else {
    // For RHF: F = 2*J - K
    trial_fock = 2.0 * J_scratch - K_scratch;
  }

  // Add XC contribution if DFT (similar to KSImpl::update_trial_fock_)
  // Only call exc->eval_fxc_contraction when exc is not null
  if (exc) {
    J_scratch.setZero();
    exc->eval_fxc_contraction(ground_density.data(), trial_density.data(),
                              J_scratch.data());
    trial_fock += J_scratch;
  }
}

/**
 * @brief Apply the stability analysis matrix-vector operation
 *
 * This function computes Y = (A+B)*X,
 * See J. Chem. Phys. 66, 3045 (1977) for the definition of A and B.
 *
 * @param X Input vectors (eigensize x num_vectors)
 * @param Y Output vectors (eigensize x num_vectors), updated to (A+B)*X
 * @param num_alpha Number of occupied alpha orbitals
 * @param num_beta Number of occupied beta orbitals
 * @param eigen_diff Diagonal preconditioner (orbital energy differences)
 * @param Ca Alpha MO coefficients (num_atomic_orbitals x
 * num_molecular_orbitals)
 * @param Cb Beta MO coefficients (num_atomic_orbitals x num_molecular_orbitals,
 * only used if unrestricted)
 * @param eri ERI multiplexer for computing J and K matrices
 * @param exc Exchange-correlation object (nullptr for HF)
 * @param ground_density Ground state density matrix
 */
void apply_stability_operator(const Eigen::MatrixXd& X, Eigen::MatrixXd& Y,
                              size_t num_alpha, size_t num_beta,
                              const Eigen::VectorXd& eigen_diff,
                              const Eigen::MatrixXd& Ca,
                              const Eigen::MatrixXd& Cb,
                              const qcs::ERIMultiplexer& eri,
                              const std::shared_ptr<qcs::EXC>& exc,
                              const RowMajorMatrix& ground_density) {
  AutoTimer __timer("stability:: apply_A_operator");

  // Calculate sizes
  const size_t num_atomic_orbitals = Ca.rows();
  const size_t num_molecular_orbitals = Ca.cols();
  const size_t num_alpha_virtual_orbitals = num_molecular_orbitals - num_alpha;
  const size_t num_beta_virtual_orbitals = num_molecular_orbitals - num_beta;
  const bool unrestricted = (ground_density.rows() == 2 * num_atomic_orbitals);

  const size_t nova = num_alpha * num_alpha_virtual_orbitals;
  const size_t eigensize =
      unrestricted ? nova + num_beta * num_beta_virtual_orbitals : nova;
  const size_t num_vectors = X.cols();

  if (X.rows() != static_cast<int>(eigensize)) {
    throw std::runtime_error(
        "Matrix size mismatch in stability operator: X.rows() = " +
        std::to_string(X.rows()) + ", expected " + std::to_string(eigensize));
  }
  if (Y.rows() != static_cast<int>(eigensize) ||
      Y.cols() != static_cast<int>(num_vectors)) {
    throw std::runtime_error(
        "Matrix size mismatch in stability operator: Y dimensions incorrect");
  }
  if (eigen_diff.size() != static_cast<int>(eigensize)) {
    throw std::runtime_error(
        "Preconditioner size mismatch in stability operator");
  }

  // Get pointers to orbital blocks
  const double* Ca_occ_ptr = Ca.data();
  const double* Ca_vir_ptr = Ca_occ_ptr + num_alpha * num_atomic_orbitals;
  const double* Cb_occ_ptr = unrestricted ? Cb.data() : nullptr;
  const double* Cb_vir_ptr =
      unrestricted ? Cb_occ_ptr + num_beta * num_atomic_orbitals : nullptr;

  // Allocate internal scratch matrices
  const size_t num_density_matrices = unrestricted ? 2 : 1;
  RowMajorMatrix trial_density = RowMajorMatrix::Zero(
      num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
  RowMajorMatrix trial_fock = RowMajorMatrix::Zero(
      num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
  RowMajorMatrix scratch1 = RowMajorMatrix::Zero(
      num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
  RowMajorMatrix scratch2 = RowMajorMatrix::Zero(
      num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
  double* temp = scratch1.data();

  // For each right-hand side, apply the operator
  for (size_t vec = 0; vec < num_vectors; ++vec) {
    const double* X_vec = X.col(vec).data();
    double* Y_vec = Y.col(vec).data();

    // calculate orbital energy difference term using preconditioner diagonal
    for (size_t idx = 0; idx < eigensize; ++idx) {
      Y_vec[idx] += eigen_diff(idx) * X_vec[idx];  // δij δab δστ (ϵaσ − ϵiτ )
    }

    // tP_{uv} = \sum_{ia}  X_{ai} (C_{ui} C_{va} + C_{vi} C_{ua})
    // R has num_alpha_virtual_orbitals as fast-index, tP is symmetric
    // Step 1: temp = Ca_vir * X (temp is num_atomic_orbitals x num_alpha in
    // ColMajor)
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals, num_alpha, num_alpha_virtual_orbitals, 1.0,
               Ca_vir_ptr, num_atomic_orbitals, X_vec,
               num_alpha_virtual_orbitals, 0.0, temp, num_atomic_orbitals);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               num_atomic_orbitals, num_atomic_orbitals, num_alpha, 1.0, temp,
               num_atomic_orbitals, Ca_occ_ptr, num_atomic_orbitals, 0.0,
               trial_density.data(), num_atomic_orbitals);
    for (size_t i = 0; i < num_atomic_orbitals; ++i)
      for (size_t j = i; j < num_atomic_orbitals; ++j) {
        const auto symm_ij = trial_density(i, j) + trial_density(j, i);
        trial_density(i, j) = symm_ij;
        trial_density(j, i) = symm_ij;
      }
    if (unrestricted) {
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 num_atomic_orbitals, num_beta, num_beta_virtual_orbitals, 1.0,
                 Cb_vir_ptr, num_atomic_orbitals, X_vec + nova,
                 num_beta_virtual_orbitals, 0.0, temp, num_atomic_orbitals);
      blas::gemm(
          blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
          num_atomic_orbitals, num_atomic_orbitals, num_beta, 1.0, temp,
          num_atomic_orbitals, Cb_occ_ptr, num_atomic_orbitals, 0.0,
          trial_density.data() + num_atomic_orbitals * num_atomic_orbitals,
          num_atomic_orbitals);
      for (size_t i = 0; i < num_atomic_orbitals; ++i)
        for (size_t j = i; j < num_atomic_orbitals; ++j) {
          const auto symm_ij = trial_density(i + num_atomic_orbitals, j) +
                               trial_density(j + num_atomic_orbitals, i);
          trial_density(i + num_atomic_orbitals, j) = symm_ij;
          trial_density(j + num_atomic_orbitals, i) = symm_ij;
        }
    }

    // Compute trial Fock matrix
    compute_trial_fock(eri, exc, trial_density, ground_density, trial_fock,
                       scratch1, scratch2);

    // ABX_{ia} = \sum_{uv} C_{ui} F_{uv} C_{av}
    // Step 1: temp = trial_fock^T * Ca_occ, trial_fock is symmetric
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_atomic_orbitals, num_alpha, num_atomic_orbitals, 1.0,
               trial_fock.data(), num_atomic_orbitals, Ca_occ_ptr,
               num_atomic_orbitals, 0.0, temp, num_atomic_orbitals);
    // Step 2: Y_vec += Ca_vir^T * temp
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_alpha_virtual_orbitals, num_alpha, num_atomic_orbitals, 1.0,
               Ca_vir_ptr, num_atomic_orbitals, temp, num_atomic_orbitals, 1.0,
               Y_vec, num_alpha_virtual_orbitals);
    if (unrestricted) {
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_atomic_orbitals, num_beta, num_atomic_orbitals, 1.0,
                 trial_fock.data() + num_atomic_orbitals * num_atomic_orbitals,
                 num_atomic_orbitals, Cb_occ_ptr, num_atomic_orbitals, 0.0,
                 temp, num_atomic_orbitals);
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_beta_virtual_orbitals, num_beta, num_atomic_orbitals, 1.0,
                 Cb_vir_ptr, num_atomic_orbitals, temp, num_atomic_orbitals,
                 1.0, Y_vec + nova, num_beta_virtual_orbitals);
    }
  }
}

}  // namespace detail

std::pair<bool, std::shared_ptr<data::StabilityResult>>
StabilityChecker::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  // Extract settings
  int nroots = _settings->get<int>("nroots");
  bool check_internal = _settings->get<bool>("internal");
  bool check_external = _settings->get<bool>("external");

  // Validate settings
  if (nroots <= 0) {
    throw std::runtime_error("nroots must be positive, got " +
                             std::to_string(nroots));
  }

  // Extract needed components, orbitals, basis set, coefficients, eigenvalues
  const auto orbitals = wavefunction->get_orbitals();
  const auto basis_set_qdk = orbitals->get_basis_set();
  const auto [Ca, Cb] = orbitals->get_coefficients();
  const auto [energies_alpha, energies_beta] = orbitals->get_energies();
  const auto num_atomic_orbitals = basis_set_qdk->get_num_atomic_orbitals();
  const auto num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  auto [n_alpha_electrons, n_beta_electrons] =
      wavefunction->get_total_num_electrons();
  bool unrestricted = orbitals->is_unrestricted();

  // Set sizes
  double num_density_matrices = unrestricted ? 2.0 : 1.0;
  const auto num_virtual_alpha_orbitals =
      num_molecular_orbitals - n_alpha_electrons;
  const auto num_virtual_beta_orbitals =
      num_molecular_orbitals - n_beta_electrons;
  auto eigensize = num_virtual_alpha_orbitals * n_alpha_electrons;
  if (unrestricted) eigensize += num_virtual_beta_orbitals * n_beta_electrons;

  // Get method from wavefunction metadata or settings
  std::string method = _settings->get_or_default<std::string>("method", "hf");
  std::transform(method.begin(), method.end(), method.begin(), ::tolower);

  // Convert QDK basis set to internal format
  auto basis_set_internal =
      utils::microsoft::convert_basis_set_from_qdk(*basis_set_qdk);

  // Create SCF configuration
  auto scf_config = std::make_unique<qcs::SCFConfig>();
  scf_config->mpi = qcs::mpi_default_input();
  scf_config->require_gradient = false;
  scf_config->basis = basis_set_internal->name;
  scf_config->cartesian = !basis_set_internal->pure;
  scf_config->unrestricted = unrestricted;
  scf_config->eri.method = qcs::ERIMethod::Libint2Direct;

  // Create exchange-correlation instance (only for DFT)
  std::shared_ptr<qcs::EXC> exc;
  if (method != "hf") {
    scf_config->exc.xc_name = method;
    exc = qcs::EXC::create(basis_set_internal, *scf_config);
  }

  // Create ERI instance
  std::shared_ptr<qcs::ERIMultiplexer> eri;
  eri = qcs::ERIMultiplexer::create(*basis_set_internal, *scf_config, 0.0);

  // Build density matrix
  RowMajorMatrix ground_density = RowMajorMatrix::Zero(
      num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
  if (restricted) {
    // Restricted case: build density matrix from occupied orbitals
    // P = 2 * C_occ * C_occ^T
    ground_density.noalias() =
        2.0 * Ca.block(0, 0, num_atomic_orbitals, n_alpha_electrons) *
        Ca.block(0, 0, num_atomic_orbitals, n_alpha_electrons).transpose();
  } else {
    // Unrestricted case: build separate alpha and beta density matrices

    // Alpha density matrix
    Eigen::Map<RowMajorMatrix> P_alpha(
        ground_density.data(), num_atomic_orbitals, num_atomic_orbitals);
    P_alpha.noalias() =
        Ca.block(0, 0, num_atomic_orbitals, n_alpha_electrons) *
        Ca.block(0, 0, num_atomic_orbitals, n_alpha_electrons).transpose();

    // Beta density matrix
    Eigen::Map<RowMajorMatrix> P_beta(
        ground_density.data() + num_atomic_orbitals * num_atomic_orbitals,
        num_atomic_orbitals, num_atomic_orbitals);
    P_beta.noalias() =
        Cb.block(0, 0, num_atomic_orbitals, n_beta_electrons) *
        Cb.block(0, 0, num_atomic_orbitals, n_beta_electrons).transpose();
  }

  // Prepare Diagonal elements for preconditioning
  Eigen::VectorXd eigen_diff = Eigen::VectorXd::Zero(eigensize);
  {
    size_t index = 0;
    // Alpha block
    for (size_t i = 0; i < n_alpha_electrons; ++i) {
      for (size_t a = n_alpha_electrons; a < num_molecular_orbitals; ++a) {
        eigen_diff(index) = energies_alpha(a) - energies_alpha(i);
        ++index;
      }
    }
    // Beta block (if unrestricted)
    if (unrestricted) {
      for (size_t i = 0; i < n_beta_electrons; ++i) {
        for (size_t a = n_beta_electrons; a < num_molecular_orbitals; ++a) {
          eigen_diff(index) = energies_beta(a) - energies_beta(i);
          ++index;
        }
      }
    }
  }

  // Contruct Initial Eigenvector (set HOMO-LUMO elements to 1)
  Eigen::VectorXd eigenvector = Eigen::VectorXd::Zero(eigensize);
  eigenvector((n_alpha_electrons - 1) * num_virtual_alpha_orbitals) = 1.0;

  // Placeholder for stability result
  bool is_stable = true;
  auto stability_result = std::make_shared<data::StabilityResult>();

  spdlog::warn(
      "StabilityChecker::_run_impl is a stub and not yet fully implemented");

  return std::make_pair(is_stable, stability_result);
}

}  // namespace qdk::chemistry::algorithms::microsoft
