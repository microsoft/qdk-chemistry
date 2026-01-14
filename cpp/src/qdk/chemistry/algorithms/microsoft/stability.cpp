// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "stability.hpp"

#include <qdk/chemistry/scf/core/exc.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/util/gauxc_registry.h>

#include <macis/solvers/davidson.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

// Local implementation details
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;
// Bring logger types into scope
using qdk::chemistry::utils::Logger;
using qdk::chemistry::utils::LogLevel;

// Type aliases
using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
 * @param rhf_external Flag indicating if RHF external stability is requested
 */
void compute_trial_fock(const std::shared_ptr<qcs::ERI> eri,
                        const std::shared_ptr<qcs::EXC> exc,
                        const RowMajorMatrix& trial_density,
                        const RowMajorMatrix& ground_density,
                        RowMajorMatrix& trial_fock, bool rhf_external) {
  QDK_LOG_TRACE_ENTERING();
  const size_t num_atomic_orbitals = ground_density.cols();
  const bool unrestricted = (ground_density.rows() == 2 * num_atomic_orbitals);

  // Get hybrid coefficients (0,0,0 for HF)
  double alpha = 1.0, beta = 0.0, omega = 0.0;
  if (exc) {
    std::tie(alpha, beta, omega) = exc->get_hyb();
  }

  // JK part
  {
    // Build J and K matrices
    RowMajorMatrix J_scratch =
        RowMajorMatrix::Zero(ground_density.rows(), num_atomic_orbitals);
    RowMajorMatrix K_scratch =
        RowMajorMatrix::Zero(ground_density.rows(), num_atomic_orbitals);
    eri->build_JK(trial_density.data(), J_scratch.data(), K_scratch.data(),
                  alpha, beta, omega);

    // Compute Fock matrix: F = J - K for RHF, or appropriate combination for
    // UHF
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
      if (rhf_external)
        // For RHF external: F = - K
        trial_fock = -K_scratch;
      else
        // For RHF internal: F = 2*J - K
        trial_fock = 2.0 * J_scratch - K_scratch;
    }
  }

  // Add XC contribution if DFT (similar to KSImpl::update_trial_fock_)
  // Only call exc->eval_fxc_contraction when exc is not null
  if (exc) {
    if (!rhf_external) {
      // Internal
      RowMajorMatrix FXC =
          RowMajorMatrix::Zero(ground_density.rows(), num_atomic_orbitals);
      exc->eval_fxc_contraction(ground_density.data(), trial_density.data(),
                                FXC.data());
      trial_fock += FXC;
    } else {
      // External
      RowMajorMatrix ground_density_ext =
          RowMajorMatrix::Zero(2 * ground_density.rows(), num_atomic_orbitals);
      RowMajorMatrix trial_density_ext =
          RowMajorMatrix::Zero(2 * ground_density.rows(), num_atomic_orbitals);
      RowMajorMatrix FXC =
          RowMajorMatrix::Zero(2 * ground_density.rows(), num_atomic_orbitals);

      // Use Pb = Pa, tPb = -tPa to calculate triplet fxc
      ground_density_ext.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) =
          ground_density * 0.5;
      ground_density_ext.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                               num_atomic_orbitals) =
          ground_density_ext.block(0, 0, num_atomic_orbitals,
                                   num_atomic_orbitals);
      trial_density_ext.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) =
          trial_density;
      trial_density_ext.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                              num_atomic_orbitals) =
          -trial_density_ext.block(0, 0, num_atomic_orbitals,
                                   num_atomic_orbitals);

      exc->eval_fxc_contraction(ground_density_ext.data(),
                                trial_density_ext.data(), FXC.data());
      trial_fock += FXC.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
    }
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
  std::shared_ptr<qcs::ERI> eri_;
  std::shared_ptr<qcs::EXC> exc_;
  const RowMajorMatrix& ground_density_;
  bool rhf_external_ = false;

  /**
   * @brief Apply the stability analysis matrix-vector operation
   *
   * This function computes Y += alpha * (A+B)*X,
   * See J. Chem. Phys. 66, 3045 (1977) for the definition of A and B.
   *
   * @param X_ptr Pointer to input vectors (eigensize x num_vectors in
   * column-major)
   * @param Y_ptr Pointer to output vectors (eigensize x num_vectors in
   * column-major)
   * @param num_vectors Number of vectors to process
   * @param alpha Scaling factor for the result
   */
  void apply_stability_operator(const double* X_ptr, double* Y_ptr,
                                size_t num_vectors, double alpha) const {
    QDK_LOG_TRACE_ENTERING();
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

    // Allocate scratch matrices
    const size_t num_density_matrices = unrestricted ? 2 : 1;
    RowMajorMatrix trial_density = RowMajorMatrix::Zero(
        num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
    RowMajorMatrix trial_fock = RowMajorMatrix::Zero(
        num_atomic_orbitals * num_density_matrices, num_atomic_orbitals);
    RowMajorMatrix scratch =
        RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
    double* temp = scratch.data();

    // For each right-hand side, apply the operator
    for (size_t vec = 0; vec < num_vectors; ++vec) {
      const double* X_vec = X_ptr + vec * eigensize;
      double* Y_vec = Y_ptr + vec * eigensize;

      // calculate orbital energy difference term using preconditioner
      // diagonal
      for (size_t idx = 0; idx < eigensize; ++idx) {
        Y_vec[idx] +=
            alpha * eigen_diff_(idx) * X_vec[idx];  // δij δab δστ (ϵaσ − ϵiτ )
      }

      // tP_{uv} = \sum_{ia}  X_{ai} (C_{ui} C_{va} + C_{vi} C_{ua})
      // X has num_alpha_virtual_orbitals as fast-index: X[a + i*nvir]
      // Step 1: temp = Ca_vir * X (temp is num_atomic_orbitals x num_alpha)
      // X is (nvir x nocc) in column-major, we want: temp = Ca_vir * X
      trial_density.setZero();
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 num_atomic_orbitals, num_alpha_, num_alpha_virtual_orbitals,
                 1.0, Ca_vir_ptr, num_atomic_orbitals, X_vec,
                 num_alpha_virtual_orbitals, 0.0, temp, num_atomic_orbitals);
      // Step 2: trial_density = temp * Ca_occ^T (symmetrize later)
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                 num_atomic_orbitals, num_atomic_orbitals, num_alpha_, 1.0,
                 temp, num_atomic_orbitals, Ca_occ_ptr, num_atomic_orbitals,
                 0.0, trial_density.data(), num_atomic_orbitals);
      // Symmetrize: P = P + P^T
      for (size_t i = 0; i < num_atomic_orbitals; ++i)
        for (size_t j = i; j < num_atomic_orbitals; ++j) {
          const auto symm_ij = trial_density(i, j) + trial_density(j, i);
          trial_density(i, j) = symm_ij;
          trial_density(j, i) = symm_ij;
        }
      if (unrestricted) {
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   num_atomic_orbitals, num_beta_, num_beta_virtual_orbitals,
                   1.0, Cb_vir_ptr, num_atomic_orbitals, X_vec + nova,
                   num_beta_virtual_orbitals, 0.0, temp, num_atomic_orbitals);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
            num_atomic_orbitals, num_atomic_orbitals, num_beta_, 1.0, temp,
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
                         rhf_external_);

      // ABX_{ia} = \sum_{uv} C_{ui} F_{uv} C_{av}
      // Step 1: temp = trial_fock^T * Ca_occ, trial_fock is symmetric
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_atomic_orbitals, num_alpha_, num_atomic_orbitals, 1.0,
                 trial_fock.data(), num_atomic_orbitals, Ca_occ_ptr,
                 num_atomic_orbitals, 0.0, temp, num_atomic_orbitals);
      // Step 2: Y_vec += alpha * Ca_vir^T * temp
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_alpha_virtual_orbitals, num_alpha_, num_atomic_orbitals,
                 alpha, Ca_vir_ptr, num_atomic_orbitals, temp,
                 num_atomic_orbitals, 1.0, Y_vec, num_alpha_virtual_orbitals);
      if (unrestricted) {
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
            num_atomic_orbitals, num_beta_, num_atomic_orbitals, 1.0,
            trial_fock.data() + num_atomic_orbitals * num_atomic_orbitals,
            num_atomic_orbitals, Cb_occ_ptr, num_atomic_orbitals, 0.0, temp,
            num_atomic_orbitals);
        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   num_beta_virtual_orbitals, num_beta_, num_atomic_orbitals,
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
                    std::shared_ptr<qcs::ERI> eri,
                    std::shared_ptr<qcs::EXC> exc,
                    const RowMajorMatrix& ground_density, bool rhf_external)
      : num_alpha_(num_alpha),
        num_beta_(num_beta),
        eigen_diff_(eigen_diff),
        Ca_(Ca),
        Cb_(Cb),
        eri_(eri),
        exc_(exc),
        ground_density_(ground_density),
        rhf_external_(rhf_external) {}

  void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                       double beta, double* AV, size_t LDAV) const {
    QDK_LOG_TRACE_ENTERING();
    const size_t N = eigen_diff_.size();

    // Scale Y by beta (Y = beta * Y)
    if (beta == 0.0) {
      std::fill_n(AV, N * m, 0.0);
    } else if (beta != 1.0) {
      for (size_t i = 0; i < N * m; ++i) {
        AV[i] *= beta;
      }
    }

    apply_stability_operator(V, AV, m, alpha);
  }
};

/**
 * @brief Construct an initial eigenvector for the Davidson heuristic. Uses the
 * inverse of the orbital energy differences as a preconditioned guess and
 * enforces single dominant components in each occupied block. Safeguards of
 * 1e4/1e-4 avoid division by zero when the energy differences are extremely
 * small.
 *
 * The Davidson algorithm is highly sensitive to the choice of the initial
 * vector. Mixing the two initialization strategies reduces the risk of
 * Davidson failing to converge.
 *
 * @param eigen_diff Flattened array of orbital energy differences.
 * @param n_alpha_electrons Number of occupied alpha orbitals.
 * @param n_beta_electrons Number of occupied beta orbitals.
 * @param num_virtual_alpha_orbitals Number of virtual alpha orbitals.
 * @param num_virtual_beta_orbitals Number of virtual beta orbitals.
 * @param unrestricted True when the reference wavefunction is UHF.
 * @param eigenvector [in, out] Output vector that stores the initialized guess.
 */
void initialize_eigenvector(const Eigen::VectorXd& eigen_diff,
                            size_t n_alpha_electrons, size_t n_beta_electrons,
                            size_t num_virtual_alpha_orbitals,
                            size_t num_virtual_beta_orbitals, bool unrestricted,
                            Eigen::VectorXd& eigenvector) {
  QDK_LOG_TRACE_ENTERING();
  int eigensize = eigenvector.size();
  auto nova = num_virtual_alpha_orbitals * n_alpha_electrons;
  double min_abs_eigen_diff = 1e-4;
  eigenvector = Eigen::VectorXd::Constant(eigensize, 1.0 / min_abs_eigen_diff);
  for (int i = 0; i < eigen_diff.size(); ++i) {
    if (std::abs(eigen_diff(i)) > min_abs_eigen_diff) {
      eigenvector(i) = 1.0 / eigen_diff(i);
    }
  }
  eigenvector.normalize();
  if (n_alpha_electrons > 0)
    eigenvector((n_alpha_electrons - 1) * num_virtual_alpha_orbitals) = 1.0;
  if (unrestricted && n_beta_electrons > 0)
    eigenvector((n_beta_electrons - 1) * num_virtual_beta_orbitals + nova) =
        1.0;
  eigenvector.normalize();
}

/**
 * @brief Transpose eigenvector from column-major to row-major format in-place
 *
 * Converts eigenvector storage from (nvir x nocc) column-major to (nocc x nvir)
 * row-major.
 *
 * @param eigenvector_ptr Pointer to the eigenvector data to transpose in-place
 * @param num_virtual Number of virtual orbitals
 * @param num_occupied Number of occupied orbitals
 */
void transpose_eigenvector_to_rowmajor(double* eigenvector_ptr,
                                       size_t num_virtual,
                                       size_t num_occupied) {
  Eigen::Map<const Eigen::MatrixXd> col_major(eigenvector_ptr, num_virtual,
                                              num_occupied);
  Eigen::MatrixXd row_major = col_major.transpose();
  Eigen::Map<Eigen::MatrixXd>(eigenvector_ptr, num_occupied, num_virtual) =
      row_major;
}

}  // namespace detail

std::pair<bool, std::shared_ptr<data::StabilityResult>>
StabilityChecker::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  QDK_LOG_TRACE_ENTERING();
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  // Extract settings
  const int64_t davidson_max_subspace = _settings->get<int64_t>("max_subspace");
  const double stability_tol = _settings->get<double>("stability_tolerance");
  const double davidson_tol = _settings->get<double>("davidson_tolerance");
  bool check_internal = _settings->get<bool>("internal");
  bool check_external = _settings->get<bool>("external");

  if (!check_internal && !check_external) {
    throw std::invalid_argument(
        "At least one of 'internal' or 'external' stability checks must be "
        "enabled.");
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

  // Throw error if the scf is ROHF/ROKS
  if (!unrestricted && n_alpha_electrons != n_beta_electrons) {
    throw std::runtime_error(
        "ROHF/ROKS is currently not supported in internal backend stability "
        "checker. Please use pySCF instead.");
  }

  if (check_external && unrestricted) {
    throw std::invalid_argument(
        "External stability analysis (RHF -> UHF) is not supported for UHF "
        "wavefunctions.");
  }

  // Set sizes
  size_t num_density_matrices = unrestricted ? 2 : 1;
  const auto num_virtual_alpha_orbitals =
      num_molecular_orbitals - n_alpha_electrons;
  const auto num_virtual_beta_orbitals =
      num_molecular_orbitals - n_beta_electrons;
  auto nova = num_virtual_alpha_orbitals * n_alpha_electrons;
  auto eigensize = nova;
  if (unrestricted) eigensize += num_virtual_beta_orbitals * n_beta_electrons;

  // Get method from wavefunction metadata or settings
  std::string method = _settings->get_or_default<std::string>("method", "hf");
  std::transform(method.begin(), method.end(), method.begin(), ::toupper);

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

  // Create ERI instance
  std::shared_ptr<qcs::ERI> eri;
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

  Eigen::VectorXd eigenvector = Eigen::VectorXd::Zero(eigensize);
  detail::initialize_eigenvector(eigen_diff, n_alpha_electrons,
                                 n_beta_electrons, num_virtual_alpha_orbitals,
                                 num_virtual_beta_orbitals, unrestricted,
                                 eigenvector);

  const int64_t max_subspace =
      std::min(davidson_max_subspace, static_cast<int64_t>(eigensize));

  QDK_LOGGER().info(
      "Starting Davidson eigensolver (size: {}, subspace: {}, tol: {:.2e})",
      eigensize, max_subspace, davidson_tol);

  bool internal_stable = true;
  bool external_stable = true;
  Eigen::VectorXd internal_eigenvalues = Eigen::VectorXd::Zero(0);
  Eigen::MatrixXd internal_eigenvectors = Eigen::MatrixXd::Zero(0, 0);
  Eigen::VectorXd external_eigenvalues = Eigen::VectorXd::Zero(0);
  Eigen::MatrixXd external_eigenvectors = Eigen::MatrixXd::Zero(0, 0);

  if (check_internal) {
    // Create exchange-correlation instance
    std::shared_ptr<qcs::EXC> exc_internal;
    if (method != "HF") {
      scf_config->exc.xc_name = method;
      exc_internal = qcs::EXC::create(basis_set_internal, *scf_config);
    }

    internal_eigenvectors.resize(eigenvector.size(), 1);
    internal_eigenvectors = eigenvector;

    // Create the stability operator wrapper and run Davidson eigensolver
    detail::StabilityOperator stability_op(n_alpha_electrons, n_beta_electrons,
                                           eigen_diff, Ca, Cb, eri,
                                           exc_internal, ground_density, false);
    auto [num_iterations, lowest_eigenvalue] = macis::davidson(
        eigensize, max_subspace, stability_op, eigen_diff.data(), davidson_tol,
        internal_eigenvectors.data());

    // Determine stability: stable if the lowest eigenvalue is greater than the
    // stability tolerance
    internal_stable = (lowest_eigenvalue > stability_tol);

    QDK_LOGGER().info(
        "Davidson converged in {} iterations for internal stability, lowest "
        "eigenvalue: {:.8f}",
        num_iterations, lowest_eigenvalue);
    internal_eigenvalues.resize(1);
    internal_eigenvalues(0) = lowest_eigenvalue;

    // Transpose eigenvectors from column-major to row-major
    // to be compatible with the PySCF format
    detail::transpose_eigenvector_to_rowmajor(internal_eigenvectors.data(),
                                              num_virtual_alpha_orbitals,
                                              n_alpha_electrons);
    if (unrestricted) {
      detail::transpose_eigenvector_to_rowmajor(
          internal_eigenvectors.data() + nova, num_virtual_beta_orbitals,
          n_beta_electrons);
    }

    if (exc_internal) qcs::util::GAUXCRegistry::clear();  // clear GAUXC cache
  }

  if (check_external) {
    // Create exchange-correlation instance (only for DFT)
    std::shared_ptr<qcs::EXC> exc_external;
    if (method != "HF") {
      // deep copy the scf_config to avoid modifying the original
      auto scf_config_external = std::make_unique<qcs::SCFConfig>(*scf_config);
      scf_config_external->unrestricted = true;
      scf_config_external->exc.xc_name = method;
      exc_external = qcs::EXC::create(basis_set_internal, *scf_config_external);
    }

    external_eigenvectors.resize(eigenvector.size(), 1);
    external_eigenvectors = eigenvector;

    // Create the stability operator wrapper and run Davidson eigensolver
    detail::StabilityOperator stability_op(n_alpha_electrons, n_beta_electrons,
                                           eigen_diff, Ca, Cb, eri,
                                           exc_external, ground_density, true);
    auto [num_iterations, lowest_eigenvalue] = macis::davidson(
        eigensize, max_subspace, stability_op, eigen_diff.data(), davidson_tol,
        external_eigenvectors.data());

    // Determine stability: stable if the lowest eigenvalue is greater than the
    // stability tolerance
    external_stable = (lowest_eigenvalue > stability_tol);
    QDK_LOGGER().info(
        "Davidson converged in {} iterations for external stability, lowest "
        "eigenvalue: {:.8f}",
        num_iterations, lowest_eigenvalue);
    external_eigenvalues.resize(1);
    external_eigenvalues(0) = lowest_eigenvalue;

    // Transpose eigenvectors from column-major to row-major
    // to be compatible with the PySCF format
    detail::transpose_eigenvector_to_rowmajor(external_eigenvectors.data(),
                                              num_virtual_alpha_orbitals,
                                              n_alpha_electrons);

    if (exc_external) qcs::util::GAUXCRegistry::clear();  // clear GAUXC cache
  }

  // Create the stability result object
  auto stability_result = std::make_shared<data::StabilityResult>(
      internal_stable, external_stable, internal_eigenvalues,
      internal_eigenvectors, external_eigenvalues, external_eigenvectors);

  bool stable = internal_stable && external_stable;

  return std::make_pair(stable, stability_result);
}

}  // namespace qdk::chemistry::algorithms::microsoft
