// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "diis.h"

#include <qdk/chemistry/scf/config.h>

#include <cmath>
#include <deque>
#include <limits>
#include <qdk/chemistry/utils/logger.hpp>

#include "../scf/scf_impl.h"
#include "scf_matrix_handler.h"

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>

#include "util/gpu/matrix_op.h"
#endif

#include <lapack.hh>

#include "util/macros.h"
#include "util/timer.h"

namespace qdk::chemistry::scf {

namespace impl {

/**
 * @brief DIIS implementation class
 */
class DIIS {
 public:
  /**
   * @brief Construct DIIS implementation
   *
   * @param[in] ctx SCFContext reference
   * @param[in] rohf_enabled Flag indicating if ROHF is enabled
   * @param[in] subspace_size Maximum number of vectors to retain in DIIS
   * subspace
   */
  explicit DIIS(const SCFContext& ctx, bool rohf_enabled,
                const size_t subspace_size);
  /**
   * @brief Perform one DIIS iteration
   *
   * Computes the DIIS error, performs extrapolation or level shifting,
   * and stores the extrapolated Fock matrix as member variable.
   *
   * @param[in] P Density matrix
   * @param[in] F Current Fock matrix
   * @param[in] S Overlap matrix
   */
  void iterate(const RowMajorMatrix& P, const RowMajorMatrix& F,
               const RowMajorMatrix& S);

  /**
   * @brief Get the stored extrapolated Fock matrix from the last iterate()
   *
   * @return Reference to the extrapolated Fock matrix
   */
  const RowMajorMatrix& get_extrapolated_fock() const {
    QDK_LOG_TRACE_ENTERING();
    return F_extrapolated_;
  }
  /**
   * @brief Get the current DIIS error metric
   *
   * @return Current DIIS error
   */
  double get_diis_error() const {
    QDK_LOG_TRACE_ENTERING();
    return diis_error_;
  }

 private:
  /**
   * @brief Perform DIIS extrapolation to generate possibly improved Fock matrix
   *
   * Takes the current Fock matrix and error vector, adds them to the history,
   * and computes an extrapolated Fock matrix that should have reduced error.
   * The extrapolation is computed by solving:
   *
   *   min ||sum_i c_i * e_i||^2  subject to sum_i c_i = 1
   *
   * where e_i are the error vectors and c_i are the coefficients applied
   * to the Fock matrices F_i to produce F_diis = sum_i c_i * F_i.
   *
   * @param[in] x Current Fock matrix to add to history
   * @param[in] error Current error vector
   * @param[out] x_diis Output extrapolated Fock matrix with reduced error
   */
  void extrapolate_(const RowMajorMatrix& x, const RowMajorMatrix& error,
                    RowMajorMatrix& x_diis);

  /**
   * @brief Remove oldest Fock matrix and error vector when subspace is full
   *
   * Deletes the oldest Fock matrix and error vector from the history
   * to maintain the subspace size constraint.
   */
  void delete_oldest_();

  /**
   * @brief Apply level shift to Fock matrix for improved convergence
   *
   * Applies the level shift formula: F_ls = F + (S - SPS) * mu
   * which shifts the virtual orbital energies to stabilize convergence.
   *
   * @param[in] F Input Fock matrix
   * @param[in] P Density matrix
   * @param[in] S Overlap matrix
   * @param[out] F_ls Output level-shifted Fock matrix
   * @param[in] mu Level shift parameter (shift magnitude)
   */
  void apply_level_shift_(const RowMajorMatrix& F, const RowMajorMatrix& P,
                          const RowMajorMatrix& S, RowMajorMatrix& F_ls,
                          const double mu) const;

  bool rohf_enabled_;                ///< Flag indicating if ROHF is enabled
  const SCFContext& ctx_;            ///< Reference to SCFContext
  size_t subspace_size_;             ///< Maximum number of vectors in subspace
  std::deque<RowMajorMatrix> hist_;  ///< History of Fock matrices
  std::deque<RowMajorMatrix> errors_;  ///< History of error vectors
  RowMajorMatrix B_;                   ///< Overlap matrix of error vectors
  RowMajorMatrix F_extrapolated_;      ///< Extrapolated Fock matrix
  double diis_error_ =
      std::numeric_limits<double>::infinity();  ///< Current DIIS error
};

DIIS::DIIS(const SCFContext& ctx, bool rohf_enabled, const size_t subspace_size)
    : ctx_(ctx), rohf_enabled_(rohf_enabled), subspace_size_(subspace_size) {
  QDK_LOG_TRACE_ENTERING();
  if (subspace_size <= 0) {
    throw std::invalid_argument("subspace_size must be greater than 0");
  }
}

void DIIS::iterate(const RowMajorMatrix& P, const RowMajorMatrix& F,
                   const RowMajorMatrix& S) {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;
  auto& res = ctx_.result;

  int num_atomic_orbitals = ctx_.basis_set->num_atomic_orbitals;
  int num_density_matrices = (ctx_.cfg->unrestricted || rohf_enabled_) ? 2 : 1;
  int num_orbital_sets = ctx_.cfg->unrestricted ? 2 : 1;

  // Create error matrix for DIIS (use the base class error calculation)
  RowMajorMatrix error = RowMajorMatrix::Zero(
      num_density_matrices * num_atomic_orbitals, num_atomic_orbitals);
  diis_error_ =
      SCFAlgorithm::calculate_og_error_(F, P, S, error, num_orbital_sets);

  // Create extrapolated Fock matrix for density matrix update, instead of
  // modifying F in place
  RowMajorMatrix F_extrapolated = RowMajorMatrix::Zero(
      num_density_matrices * num_atomic_orbitals, num_atomic_orbitals);
  if (cfg->scf_algorithm.level_shift > 0.0) {  // Level Shifting
    double mu = cfg->scf_algorithm.level_shift;
    apply_level_shift_(F, P, S, F_extrapolated, mu);
  } else {
    // Use DIIS extrapolation
    extrapolate_(F, error, F_extrapolated);
  }

  // Store the extrapolated Fock matrix for use in main DIIS::iterate
  F_extrapolated_ = F_extrapolated;
}

void DIIS::extrapolate_(const RowMajorMatrix& x, const RowMajorMatrix& error,
                        RowMajorMatrix& x_diis) {
  QDK_LOG_TRACE_ENTERING();
  x_diis = x;
  if (hist_.size() == subspace_size_) delete_oldest_();
  hist_.push_back(x);
  errors_.push_back(error);

  size_t n = hist_.size();
  RowMajorMatrix B_old = B_;
  B_ = RowMajorMatrix::Zero(n, n);
  B_.block(0, 0, n - 1, n - 1) = B_old;

  // Build overlap matrix of error vectors
  for (size_t i = 0; i < n; i++) {
    B_(i, n - 1) = B_(n - 1, i) = errors_[i].cwiseProduct(errors_[n - 1]).sum();
  }

  for (;;) {
    size_t rank = hist_.size() + 1;
    // Set up the DIIS linear system: A c = rhs
    RowMajorMatrix A(rank, rank);
    A.col(0).setConstant(-1.0);
    A.row(0).setConstant(-1.0);
    A(0, 0) = 0.0;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rank);
    rhs[0] = -1.0;

    double b_max = B_.maxCoeff();
    if (b_max == 0.0) {
      // Fallback: just return the input x without extrapolation
      x_diis = x;
      return;
    }

    A.block(1, 1, rank - 1, rank - 1) =
        B_.block(0, 0, rank - 1, rank - 1) / b_max;

    Eigen::ColPivHouseholderQR<RowMajorMatrix> qr = A.colPivHouseholderQr();
    Eigen::VectorXd c = qr.solve(rhs);
    double absdet = qr.absDeterminant();

    const double diis_linear_dependence_threshold = 1e-12;
    if (absdet < diis_linear_dependence_threshold) {
      delete_oldest_();
    } else {
      x_diis.setZero();
      for (size_t i = 0; i < hist_.size(); i++) {
        x_diis += c[i + 1] * hist_[i];
      }
      break;
    }
  }
}

void DIIS::delete_oldest_() {
  QDK_LOG_TRACE_ENTERING();
  hist_.pop_front();
  errors_.pop_front();
  size_t sz = B_.rows();
  Eigen::MatrixXd tmp = B_.block(1, 1, sz - 1, sz - 1);
  B_ = tmp;
}

void DIIS::apply_level_shift_(const RowMajorMatrix& F, const RowMajorMatrix& P,
                              const RowMajorMatrix& S, RowMajorMatrix& F_ls,
                              const double mu) const {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;
  int num_atomic_orbitals = static_cast<int>(S.cols());
  int num_density_matrices = cfg->unrestricted ? 2 : 1;

  RowMajorMatrix SPS = RowMajorMatrix::Zero(
      num_density_matrices * num_atomic_orbitals, num_atomic_orbitals);
  RowMajorMatrix SP =
      RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);

  for (int spin = 0; spin < num_density_matrices; spin++) {
    SP.noalias() =
        S * Eigen::Map<const RowMajorMatrix>(
                P.data() + spin * num_atomic_orbitals * num_atomic_orbitals,
                num_atomic_orbitals, num_atomic_orbitals);
    SPS.block(spin * num_atomic_orbitals, 0, num_atomic_orbitals,
              num_atomic_orbitals)
        .noalias() = SP * S;
  }

  // Apply level shift: F_ls = F + (S - SPS) * mu
  const double* Fdata = F.data();
  double* F_lsdata = F_ls.data();
  const double* SPSdata = SPS.data();
  const double* Sdata = S.data();

  for (int i = 0;
       i < num_density_matrices * num_atomic_orbitals * num_atomic_orbitals;
       i++) {
    F_lsdata[i] =
        Fdata[i] +
        (Sdata[i % (num_atomic_orbitals * num_atomic_orbitals)] - SPSdata[i]) *
            mu;
  }
}

}  // namespace impl

// Constructor for SCFAlgorithm interface
DIIS::DIIS(const SCFContext& ctx, bool rohf_enabled, const size_t subspace_size)
    : SCFAlgorithm(ctx, rohf_enabled),
      diis_impl_(
          std::make_unique<impl::DIIS>(ctx, rohf_enabled, subspace_size)),
          rohf_enabled_(rohf_enabled) {
  QDK_LOG_TRACE_ENTERING();
  // Only create matrix handler for ROHF case
  if (rohf_enabled) {
    rohf_matrix_handler_ = std::make_unique<ROHFMatrixHandler>();
  } else {
    rohf_matrix_handler_ = nullptr;
  }
}

DIIS::~DIIS() noexcept = default;

void DIIS::iterate(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;

  // Get matrix references - use handler for ROHF, direct access for RHF/UHF
  const RowMajorMatrix* F_ptr;
  RowMajorMatrix* P_ptr;

  if (rohf_enabled_) {
    rohf_matrix_handler_->receive_F_P_matrices(
        scf_impl.get_fock_matrix(), scf_impl.density_matrix());
    F_ptr = &rohf_matrix_handler_->get_fock_matrix();
    P_ptr = &rohf_matrix_handler_->get_density_matrix();
  } else {
    F_ptr = &scf_impl.get_fock_matrix();
    P_ptr = &scf_impl.density_matrix();
  }

  const auto& F = *F_ptr;
  auto& P = *P_ptr;
  auto& C = scf_impl.orbitals_matrix();
  const auto& S = scf_impl.overlap();
  const auto& X = scf_impl.get_orthogonalization_matrix();
  auto& eigenvalues = scf_impl.eigenvalues();

  // Call DIIS implementation with individual parameters
  diis_impl_->iterate(P, F, S);
  // Update density matrices using the extrapolated Fock matrix
  const RowMajorMatrix& F_extrapolated = diis_impl_->get_extrapolated_fock();

  int num_atomic_orbitals = scf_impl.get_num_atomic_orbitals();
  int num_molecular_orbitals = scf_impl.get_num_molecular_orbitals();
  const int num_orbital_sets = scf_impl.get_num_orbital_sets();
  std::vector<int> nelec_vec = scf_impl.get_num_electrons();
  const int nelec[2] = {nelec_vec[0], nelec_vec[1]};

  for (auto i = 0; i < num_orbital_sets; ++i) {
    // Use extrapolated Fock matrix for density matrix update
    solve_fock_eigenproblem(F_extrapolated, S, X, C, eigenvalues, P, nelec,
                            num_atomic_orbitals, num_molecular_orbitals, i,
                            cfg->unrestricted);
  }

  double diis_error = diis_impl_->get_diis_error();
  bool should_apply_damping = cfg->scf_algorithm.enable_damping &&
                              diis_error > cfg->scf_algorithm.damping_threshold;
  if (should_apply_damping) {
    double factor = cfg->scf_algorithm.damping_factor;
    P = P_last_ * factor + P * (1.0 - factor);
  }
}

}  // namespace qdk::chemistry::scf
