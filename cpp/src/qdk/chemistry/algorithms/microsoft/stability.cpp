// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "stability.hpp"

#include <qdk/chemistry/scf/core/exc.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <spdlog/spdlog.h>

#include <macis/solvers/davidson.hpp>
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
void compute_trial_fock(const std::shared_ptr<qcs::ERIMultiplexer>& eri,
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
  eri->build_JK(trial_density.data(), J_scratch.data(), K_scratch.data(), alpha,
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
 * @brief Operator wrapper for stability analysis Davidson solver
 *
 * This class wraps the stability operator for use with the Davidson
 * eigensolver. It provides the operator_action interface required by
 * the Davidson algorithm.
 */
class StabilityOperator {
 private:
  size_t num_alpha_;
  size_t num_beta_;
  const Eigen::VectorXd& eigen_diff_;
  const Eigen::MatrixXd& Ca_;
  const Eigen::MatrixXd& Cb_;
  std::shared_ptr<qcs::ERIMultiplexer> eri_;
  std::shared_ptr<qcs::EXC> exc_;
  const RowMajorMatrix& ground_density_;

  // Mutable scratch matrices to avoid reallocation
  RowMajorMatrix scratch1_;
  RowMajorMatrix scratch2_;

  /**
   * @brief Apply the stability analysis matrix-vector operation
   *
   * This function computes Y += alpha * (A+B)*X,
   * See J. Chem. Phys. 66, 3045 (1977) for the definition of A and B.
   *
   * @param X Input vectors (eigensize x num_vectors)
   * @param Y Output vectors (eigensize x num_vectors), updated to Y +=
   * alpha*(A+B)*X
   * @param alpha Scaling factor for the result
   */
  void apply_stability_operator(const Eigen::MatrixXd& X, Eigen::MatrixXd& Y,
                                double alpha) const {
    AutoTimer __timer("stability:: apply_A_operator");

    // Calculate sizes
    const size_t num_atomic_orbitals = Ca_.rows();
    const size_t num_molecular_orbitals = Ca_.cols();
    const size_t num_alpha_virtual_orbitals =
        num_molecular_orbitals - num_alpha_;
    const size_t num_beta_virtual_orbitals = num_molecular_orbitals - num_beta_;
    const bool unrestricted =
        (ground_density_.rows() == 2 * num_atomic_orbitals);

    const size_t nova = num_alpha_ * num_alpha_virtual_orbitals;
    const size_t eigensize =
        unrestricted ? nova + num_beta_ * num_beta_virtual_orbitals : nova;
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
    if (eigen_diff_.size() != static_cast<int>(eigensize)) {
      throw std::runtime_error(
          "Preconditioner size mismatch in stability operator");
    }

    // Get pointers to orbital blocks
    const double* Ca_occ_ptr = Ca_.data();
    const double* Ca_vir_ptr = Ca_occ_ptr + num_alpha_ * num_atomic_orbitals;
    const double* Cb_occ_ptr = unrestricted ? Cb_.data() : nullptr;
    const double* Cb_vir_ptr =
        unrestricted ? Cb_occ_ptr + num_beta_ * num_atomic_orbitals : nullptr;

    // Allocate internal scratch matrices
    const size_t num_density_matrices = unrestricted ? 2 : 1;
    RowMajorMatrix trial_density = RowMajorMatrix::Zero(
        num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
    RowMajorMatrix trial_fock = RowMajorMatrix::Zero(
        num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
    double* temp = scratch1_.data();

    // For each right-hand side, apply the operator
    for (size_t vec = 0; vec < num_vectors; ++vec) {
      const double* X_vec = X.col(vec).data();
      double* Y_vec = Y.col(vec).data();

      // calculate orbital energy difference term using preconditioner diagonal
      for (size_t idx = 0; idx < eigensize; ++idx) {
        Y_vec[idx] +=
            alpha * eigen_diff_(idx) * X_vec[idx];  // δij δab δστ (ϵaσ − ϵiτ )
      }

      // tP_{uv} = \sum_{ia}  X_{ai} (C_{ui} C_{va} + C_{vi} C_{ua})
      // R has num_alpha_virtual_orbitals as fast-index, tP is symmetric
      // Step 1: temp = Ca_vir * X (temp is num_atomic_orbitals x num_alpha in
      // ColMajor)
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 num_atomic_orbitals, num_alpha, num_alpha_virtual_orbitals,
                 1.0, Ca_vir_ptr, num_atomic_orbitals, X_vec,
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
                   num_atomic_orbitals, num_beta, num_beta_virtual_orbitals,
                   1.0, Cb_vir_ptr, num_atomic_orbitals, X_vec + nova,
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
      compute_trial_fock(eri_, exc_, trial_density, ground_density_, trial_fock,
                         scratch1_, scratch2_);

      // ABX_{ia} = \sum_{uv} C_{ui} F_{uv} C_{av}
      // Step 1: temp = trial_fock^T * Ca_occ, trial_fock is symmetric
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_atomic_orbitals, num_alpha, num_atomic_orbitals, 1.0,
                 trial_fock.data(), num_atomic_orbitals, Ca_occ_ptr,
                 num_atomic_orbitals, 0.0, temp, num_atomic_orbitals);
      // Step 2: Y_vec += alpha * Ca_vir^T * temp
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_alpha_virtual_orbitals, num_alpha, num_atomic_orbitals,
                 alpha, Ca_vir_ptr, num_atomic_orbitals, temp,
                 num_atomic_orbitals, 1.0, Y_vec, num_alpha_virtual_orbitals);
      if (unrestricted) {
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
            num_atomic_orbitals, num_beta, num_atomic_orbitals, 1.0,
            trial_fock.data() + num_atomic_orbitals * num_atomic_orbitals,
            num_atomic_orbitals, Cb_occ_ptr, num_atomic_orbitals, 0.0, temp,
            num_atomic_orbitals);
        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   num_beta_virtual_orbitals, num_beta, num_atomic_orbitals,
                   alpha, Cb_vir_ptr, num_atomic_orbitals, temp,
                   num_atomic_orbitals, 1.0, Y_vec + nova,
                   num_beta_virtual_orbitals);
      }
    }
  }

 public:
  StabilityOperator(size_t num_alpha, size_t num_beta,
                    const Eigen::VectorXd& eigen_diff,
                    const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Cb,
                    std::shared_ptr<qcs::ERIMultiplexer> eri,
                    std::shared_ptr<qcs::EXC> exc,
                    const RowMajorMatrix& ground_density)
      : num_alpha_(num_alpha),
        num_beta_(num_beta),
        eigen_diff_(eigen_diff),
        Ca_(Ca),
        Cb_(Cb),
        eri_(eri),
        exc_(exc),
        ground_density_(ground_density) {
    // Pre-allocate scratch matrices
    const size_t num_atomic_orbitals = Ca_.rows();
    const bool unrestricted =
        (ground_density_.rows() == 2 * num_atomic_orbitals);
    const size_t num_density_matrices = unrestricted ? 2 : 1;
    scratch1_ = RowMajorMatrix::Zero(num_atomic_orbitals * num_density_matrices,
                                     num_atomic_orbitals);
    scratch2_ = RowMajorMatrix::Zero(num_atomic_orbitals * num_density_matrices,
                                     num_atomic_orbitals);
  }

  void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                       double beta, double* AV, size_t LDAV) const {
    const size_t N = eigen_diff_.size();

    // Wrap input and output pointers as Eigen matrices
    Eigen::Map<const Eigen::MatrixXd> X(V, N, m);
    Eigen::Map<Eigen::MatrixXd> Y(AV, N, m);

    // Scale Y by beta (Y = beta * Y)
    if (beta == 0.0) {
      Y.setZero();
    } else if (beta != 1.0) {
      Y *= beta;
    }

    // Apply the stability operator: Y += alpha * (A+B) * X
    apply_stability_operator(X, Y, alpha);
  }
};

}  // namespace detail

std::pair<bool, std::shared_ptr<data::StabilityResult>>
StabilityChecker::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  // Extract settings
  int nroots = _settings->get<int>("nroots");
  // Set Davidson parameters
  const int64_t davidson_max_subspace =
      _settings->get_or_default<int64_t>("max_subspace", 30);
  const double stability_tol =
      _settings->get_or_default<double>("stability_tolerance", -1.0e-4);
  const double davidson_tol =
      _settings->get_or_default<double>("davidson_tolerance", 1.0e-6);
  bool check_internal = _settings->get<bool>("internal");
  bool check_external = _settings->get<bool>("external");

  if (check_external) {
    throw std::runtime_error(
        "External stability analysis is not implemented yet.");
  }

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
  if (!unrestricted) {
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

  // Construct Initial Eigenvector using diagonal guess
  Eigen::VectorXd eigenvector = Eigen::VectorXd::Zero(eigensize);
  // Set the element corresponding to the minimum energy difference to 1
  Eigen::VectorXd::Index min_idx;
  eigen_diff.minCoeff(&min_idx);
  eigenvector(min_idx) = 1.0;

  // Create the stability operator wrapper
  detail::StabilityOperator stability_op(n_alpha_electrons, n_beta_electrons,
                                         eigen_diff, Ca, Cb, eri, exc,
                                         ground_density);

  const int64_t max_subspace =
      std::min(davidson_max_subspace, static_cast<int64_t>(eigensize));

  spdlog::info(
      "Starting Davidson eigensolver (size: {}, subspace: {}, tol: {:.2e})",
      eigensize, max_subspace, davidson_tol);

  // Call Davidson eigensolver
  auto [num_iterations, lowest_eigenvalue] =
      macis::davidson(eigensize, max_subspace, stability_op, eigen_diff.data(),
                      davidson_tol, eigenvector.data());

  spdlog::info("Davidson converged in {} iterations, lowest eigenvalue: {:.8f}",
               num_iterations, lowest_eigenvalue);

  // Determine stability: stable if the lowest eigenvalue is greater than the
  // stability tolerance
  bool internal_stable = (lowest_eigenvalue > stability_tol);
  bool external_stable = true;  // Not implementing external stability analysis

  // Prepare eigenvalue and eigenvector for constructor
  Eigen::VectorXd internal_eigenvalues(1);
  internal_eigenvalues(0) = lowest_eigenvalue;
  Eigen::MatrixXd internal_eigenvectors =
      Eigen::Map<Eigen::MatrixXd>(eigenvector.data(), eigenvector.size(), 1);
  Eigen::VectorXd external_eigenvalues = Eigen::VectorXd::Zero(0);
  Eigen::MatrixXd external_eigenvectors = Eigen::MatrixXd::Zero(0, 0);

  // Create the stability result object
  auto stability_result = std::make_shared<data::StabilityResult>(
      internal_stable, external_stable, internal_eigenvalues,
      internal_eigenvectors, external_eigenvalues, external_eigenvectors);

  bool stable = internal_stable && external_stable;

  return std::make_pair(stable, stability_result);
}

}  // namespace qdk::chemistry::algorithms::microsoft
