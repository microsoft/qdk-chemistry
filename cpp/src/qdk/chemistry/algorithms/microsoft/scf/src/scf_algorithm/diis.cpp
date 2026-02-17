// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "diis.h"

#include <qdk/chemistry/scf/config.h>

#include <cstdint>
#include <deque>
#include <lapack.hh>
#include <limits>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <vector>

#include "../scf/scf_impl.h"
#include "util/macros.h"

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
   * @param[in] rohf_enabled Indicates if ROHF support is requested (affects
   * error construction)
   * @param[in] subspace_size Maximum number of vectors to retain in DIIS
   * subspace
   */
  explicit DIIS(const SCFContext& ctx, const size_t subspace_size);

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

  const SCFContext& ctx_;            ///< Reference to SCFContext
  size_t subspace_size_;             ///< Maximum number of vectors in subspace
  std::deque<RowMajorMatrix> hist_;  ///< History of Fock matrices
  std::deque<RowMajorMatrix> errors_;  ///< History of error vectors
  RowMajorMatrix B_;                   ///< Overlap matrix of error vectors
  RowMajorMatrix F_extrapolated_;      ///< Extrapolated Fock matrix
  double diis_error_ =
      std::numeric_limits<double>::infinity();  ///< Current DIIS error
};

class ROHFHelper {
 public:
  /**
   * @brief Construct helper with zeroed caches sized to the AO basis
   *
   * @param num_atomic_orbitals Dimension of each spin block
   */
  explicit ROHFHelper(int num_atomic_orbitals)
      : effective_F_(
            RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals)),
        total_P_(
            RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals)) {}

  /**
   * @brief Rebuild the effective ROHF Fock and density matrices
   *
   * Implements the averaged-block construction (Guest & Saunders 1974,
   * Plakhutin & Davidson 2014) to obtain the single-density ROHF view and
   * caches the total density $P_\alpha + P_\beta$ alongside it.
   *
   * @param F Spin-blocked Fock matrix with alpha and beta blocks stacked
   * @param C Molecular-orbital coefficients used for block transformations
   * @param P Spin-blocked density matrix
   * @param nelec_alpha Number of alpha electrons
   * @param nelec_beta Number of beta electrons
   */
  void build_rohf_f_p_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             const RowMajorMatrix& P, int nelec_alpha,
                             int nelec_beta) {
    QDK_LOG_TRACE_ENTERING();
    const int num_atomic_orbitals = static_cast<int>(F.cols());

    total_P_ = P.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) +
               P.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                       num_atomic_orbitals);

    if (effective_F_.rows() != num_atomic_orbitals ||
        effective_F_.cols() != num_atomic_orbitals) {
      effective_F_ =
          RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
    }

    if (C.isZero()) {
      effective_F_.noalias() =
          F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
      return;
    }

    const int num_molecular_orbitals = static_cast<int>(C.cols());
    RowMajorMatrix F_up_mo =
        RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
    RowMajorMatrix F_dn_mo = F_up_mo;
    RowMajorMatrix effective_F_mo = F_up_mo;

    F_up_mo.noalias() =
        C.transpose() *
        F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) * C;
    F_dn_mo.noalias() = C.transpose() *
                        F.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                                num_atomic_orbitals) *
                        C;

    auto average_block = [&](int row, int col, int rows, int cols) {
      if (rows <= 0 || cols <= 0) return;
      effective_F_mo.block(row, col, rows, cols).noalias() =
          0.5 * (F_up_mo.block(row, col, rows, cols) +
                 F_dn_mo.block(row, col, rows, cols));
    };
    auto copy_block = [&](const RowMajorMatrix& src, int row, int col, int rows,
                          int cols) {
      if (rows <= 0 || cols <= 0) return;
      effective_F_mo.block(row, col, rows, cols) =
          src.block(row, col, rows, cols);
    };

    const int nd = nelec_beta;
    const int ns = nelec_alpha - nelec_beta;
    const int nv = num_molecular_orbitals - nelec_alpha;

    average_block(0, 0, nd, nd);
    average_block(0, nd + ns, nd, nv);
    average_block(nd + ns, 0, nv, nd);
    average_block(nd + ns, nd + ns, nv, nv);
    average_block(nd, nd, ns, ns);
    copy_block(F_dn_mo, 0, nd, nd, ns);
    copy_block(F_dn_mo, nd, 0, ns, nd);
    copy_block(F_up_mo, nd, nd + ns, ns, nv);
    copy_block(F_up_mo, nd + ns, nd, nv, ns);

    const int matrix_dim = num_molecular_orbitals;
    using ColMajorMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    ColMajorMatrix Ct =
        Eigen::Map<const ColMajorMatrix>(C.data(), matrix_dim, C.rows());
    ColMajorMatrix temp_rhs = effective_F_mo;
    std::vector<int64_t> ipiv(matrix_dim);

    auto info = lapack::getrf(matrix_dim, matrix_dim, Ct.data(), matrix_dim,
                              ipiv.data());
    if (info != 0) {
      throw std::runtime_error("getrf failed while factorizing C^T");
    }

    info = lapack::getrs(lapack::Op::NoTrans, matrix_dim, matrix_dim, Ct.data(),
                         matrix_dim, ipiv.data(), temp_rhs.data(), matrix_dim);
    if (info != 0) {
      throw std::runtime_error("getrs failed while solving C^T X = F_mo");
    }

    temp_rhs.transposeInPlace();
    info = lapack::getrs(lapack::Op::NoTrans, matrix_dim, matrix_dim, Ct.data(),
                         matrix_dim, ipiv.data(), temp_rhs.data(), matrix_dim);
    if (info != 0) {
      throw std::runtime_error("getrs failed while solving C^T X = M^T");
    }

    effective_F_ = temp_rhs.transpose();
    if (!effective_F_.isApprox(effective_F_.transpose())) {
      effective_F_ = 0.5 * (effective_F_ + effective_F_.transpose().eval());
    }
  }

  /**
   * @brief Reconstruct spin-blocked densities from the ROHF MO matrix
   *
   * Generates $P_\alpha$ and $P_\beta$ blocks so we can hand the updated
   * density back to SCFImpl after diagonalization.
   *
   * @param P Spin-blocked density matrix to overwrite
   * @param C Molecular-orbital coefficients from latest diagonalization
   * @param nelec_alpha Number of alpha electrons
   * @param nelec_beta Number of beta electrons
   */
  void update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                             int nelec_alpha, int nelec_beta) {
    QDK_LOG_TRACE_ENTERING();
    const int num_atomic_orbitals = static_cast<int>(C.rows());

    auto build_density = [&](auto&& target, int n_occ) {
      if (n_occ <= 0) {
        target.setZero();
        return;
      }
      target.noalias() = C.block(0, 0, num_atomic_orbitals, n_occ) *
                         C.block(0, 0, num_atomic_orbitals, n_occ).transpose();
    };

    auto P_alpha = P.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
    auto P_beta = P.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                          num_atomic_orbitals);
    build_density(P_alpha, nelec_alpha);
    build_density(P_beta, nelec_beta);
  }

  /** @brief Access ROHF effective Fock matrix */
  const RowMajorMatrix& effective_fock() const { return effective_F_; }
  /** @brief Read-only access to cached total density */
  const RowMajorMatrix& total_density() const { return total_P_; }
  /** @brief Mutable access to cached total density */
  RowMajorMatrix& total_density() { return total_P_; }

 private:
  /** @brief Cached ROHF effective Fock matrix in AO basis */
  RowMajorMatrix effective_F_;
  /** @brief Cached total density (P_alpha + P_beta) */
  RowMajorMatrix total_P_;
};

DIIS::DIIS(const SCFContext& ctx, const size_t subspace_size)
    : ctx_(ctx), subspace_size_(subspace_size) {
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
  int num_orbital_sets = ctx_.cfg->is_unrestricted() ? 2 : 1;

  // Build Pulay error (F P S - S P F) in the AO basis for each spin block.
  RowMajorMatrix error = RowMajorMatrix::Zero(
      num_orbital_sets * num_atomic_orbitals, num_atomic_orbitals);
  diis_error_ =
      SCFAlgorithm::calculate_og_error_(F, P, S, error, num_orbital_sets);

  // Construct a candidate Fock matrix via either level-shifting or Pulay
  // extrapolation. This keeps the "active" F matrix immutable so higher-level
  // code can still access the unmodified SCFImpl views if needed.
  RowMajorMatrix F_extrapolated = RowMajorMatrix::Zero(
      num_orbital_sets * num_atomic_orbitals, num_atomic_orbitals);
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
  int num_density_matrices = cfg->is_unrestricted() ? 2 : 1;

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

DIIS::DIIS(const SCFContext& ctx, const size_t subspace_size)
    : SCFAlgorithm(ctx),
      diis_impl_(std::make_unique<impl::DIIS>(ctx, subspace_size)) {
  QDK_LOG_TRACE_ENTERING();
  if (ctx.cfg->is_rohf_enabled()) {
    rohf_helper_ = std::make_unique<impl::ROHFHelper>(
        static_cast<int>(ctx.basis_set->num_atomic_orbitals));
  }
}

DIIS::~DIIS() noexcept = default;

void DIIS::iterate(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();

  // Decide which Fock/density view to feed into Pulay: RHF/UHF use the direct
  // SCFImpl matrices, while ROHF works on the cached total-density view.
  RowMajorMatrix& working_density = select_working_density(scf_impl);
  const RowMajorMatrix& working_fock = select_working_fock(scf_impl);

  auto& C = scf_impl.orbitals_matrix();
  const auto& S = scf_impl.overlap();
  const auto& X = scf_impl.get_orthogonalization_matrix();
  auto& eigenvalues = scf_impl.eigenvalues();

  diis_impl_->iterate(working_density, working_fock, S);
  const RowMajorMatrix& F_extrapolated = diis_impl_->get_extrapolated_fock();

  const auto* cfg = ctx_.cfg;
  const auto nelec_vec = scf_impl.get_num_electrons();
  const int nelec[2] = {nelec_vec[0], nelec_vec[1]};
  const int num_atomic_orbitals = scf_impl.get_num_atomic_orbitals();
  const int num_molecular_orbitals = scf_impl.get_num_molecular_orbitals();
  const int num_orbital_sets = scf_impl.get_num_orbital_sets();

  // Solve the Fock eigenproblem for each spin block (or once, for restricted)
  // using the extrapolated Fock matrix, then repopulate the working density.
  for (int i = 0; i < num_orbital_sets; ++i) {
    solve_fock_eigenproblem(F_extrapolated, S, X, C, eigenvalues,
                            working_density, nelec, num_atomic_orbitals,
                            num_molecular_orbitals, i, cfg->is_unrestricted());
  }

  auto& density_matrix = scf_impl.density_matrix();
  update_density_matrix(density_matrix, C, ctx_.cfg->is_unrestricted(),
                        nelec[0], nelec[1]);

  // Optional damping blends the new density with the previous iteration when
  // DIIS error spikes
  double diis_error = current_diis_error();
  bool should_apply_damping = cfg->scf_algorithm.enable_damping &&
                              diis_error > cfg->scf_algorithm.damping_threshold;
  if (should_apply_damping) {
    double factor = cfg->scf_algorithm.damping_factor;
    density_matrix = P_last_ * factor + density_matrix * (1.0 - factor);
  }
}

void DIIS::update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                                 bool unrestricted, int nelec_alpha,
                                 int nelec_beta) {
  QDK_LOG_TRACE_ENTERING();
  if (ctx_.cfg->is_rohf_enabled()) {
    if (!rohf_helper_) {
      throw std::logic_error("ROHF helper not initialized");
    }
    rohf_helper_->update_density_matrix(P, C, nelec_alpha, nelec_beta);
    return;
  }
  SCFAlgorithm::update_density_matrix(P, C, unrestricted, nelec_alpha,
                                      nelec_beta);
}

void DIIS::build_rohf_f_p_matrix(const RowMajorMatrix& F,
                                 const RowMajorMatrix& C,
                                 const RowMajorMatrix& P, int nelec_alpha,
                                 int nelec_beta) {
  QDK_LOG_TRACE_ENTERING();
  if (!ctx_.cfg->is_rohf_enabled()) {
    throw std::logic_error("ROHF matrix build requested for non-ROHF run");
  }
  if (!rohf_helper_) {
    rohf_helper_ = std::make_unique<impl::ROHFHelper>(
        static_cast<int>(ctx_.basis_set->num_atomic_orbitals));
  }
  rohf_helper_->build_rohf_f_p_matrix(F, C, P, nelec_alpha, nelec_beta);
}

const RowMajorMatrix& DIIS::get_rohf_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  if (!rohf_helper_) {
    throw std::logic_error("ROHF helper not initialized");
  }
  return rohf_helper_->effective_fock();
}

const RowMajorMatrix& DIIS::get_rohf_density_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  if (!rohf_helper_) {
    throw std::logic_error("ROHF helper not initialized");
  }
  return rohf_helper_->total_density();
}

RowMajorMatrix& DIIS::rohf_density_matrix() {
  QDK_LOG_TRACE_ENTERING();
  if (!rohf_helper_) {
    throw std::logic_error("ROHF helper not initialized");
  }
  return rohf_helper_->total_density();
}

double DIIS::current_diis_error() const {
  QDK_LOG_TRACE_ENTERING();
  return diis_impl_->get_diis_error();
}

RowMajorMatrix& DIIS::select_working_density(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  if (ctx_.cfg->is_rohf_enabled()) {
    // The ROHF helper is refreshed inside SCFAlgorithm::check_convergence(),
    // so by the time iterate() is invoked the total density view is already
    // up to date.
    (void)scf_impl;  // kept for symmetry with the unrestricted branch
    return rohf_density_matrix();
  }
  return scf_impl.density_matrix();
}

const RowMajorMatrix& DIIS::select_working_fock(const SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  if (ctx_.cfg->is_rohf_enabled()) {
    return get_rohf_fock_matrix();
  }
  return scf_impl.get_fock_matrix();
}

}  // namespace qdk::chemistry::scf
