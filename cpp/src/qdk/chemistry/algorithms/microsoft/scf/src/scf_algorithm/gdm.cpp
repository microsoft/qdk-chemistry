// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "gdm.h"

#include <math.h>

#include <algorithm>
#include <blas.hh>
#include <iostream>
#include <lapack.hh>
#include <limits>
#include <qdk/chemistry/utils/logger.hpp>
#include <vector>

#include "../scf/scf_impl.h"
#include "line_search.h"
#include "qdk/chemistry/scf/core/scf.h"
#include "qdk/chemistry/scf/core/types.h"
#include "util/matrix_exp.h"
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include "util/gpu/cuda_helper.h"
#include "util/gpu/matrix_operations.h"
#endif

namespace qdk::chemistry::scf {

namespace impl {

/**
 * @brief Construct the antisymmetric kappa matrix and apply C * exp(kappa)
 * @param[in,out] C Molecular orbital coefficient matrix
 * @param[in] spin_index Spin index (0 for alpha, 1 for beta)
 * @param[in] kappa_vector The kappa vector to apply for rotation
 * @param[in] num_occupied_orbitals Number of occupied orbitals for this spin
 * @param[in] num_molecular_orbitals Number of molecular orbitals
 */
static void apply_orbital_rotation(RowMajorMatrix& C, const int spin_index,
                                   const Eigen::VectorXd& kappa_vector,
                                   const int num_occupied_orbitals,
                                   const int num_molecular_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  const int num_virtual_orbitals =
      num_molecular_orbitals - num_occupied_orbitals;

  // Build the rotation matrix exp(kappa)
  const RowMajorMatrix kappa_matrix = Eigen::Map<const RowMajorMatrix>(
      kappa_vector.data(), num_occupied_orbitals, num_virtual_orbitals);
  RowMajorMatrix kappa_complete =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);

  kappa_complete.block(0, num_occupied_orbitals, num_occupied_orbitals,
                       num_virtual_orbitals) = kappa_matrix / 2.0;
  kappa_complete.block(num_occupied_orbitals, 0, num_virtual_orbitals,
                       num_occupied_orbitals) = -kappa_matrix.transpose() / 2.0;

  RowMajorMatrix exp_kappa =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
  matrix_exp(kappa_complete.data(), exp_kappa.data(), num_molecular_orbitals);

  // Rotate C: C' = C * exp(kappa)
  RowMajorMatrix C_before_rotate =
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals);
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_molecular_orbitals, num_molecular_orbitals,
             num_molecular_orbitals, 1.0, C_before_rotate.data(),
             num_molecular_orbitals, exp_kappa.data(), num_molecular_orbitals,
             0.0,
             C.block(num_molecular_orbitals * spin_index, 0,
                     num_molecular_orbitals, num_molecular_orbitals)
                 .data(),
             num_molecular_orbitals);
}

/**
 * @brief Functor for evaluating GDM line search objective
 */
class GDMLineFunctor {
 public:
  using argument_type = Eigen::VectorXd;
  using return_type = double;

  /**
   * @brief Bind functor to a specific SCF state for line search evaluations.
   * @param scf_impl Reference to `SCFImpl` used to evaluate trial densities
   * @param C_pseudo_canonical Molecular orbitals in pseudo-canonical basis
   * @param num_electrons Occupied orbital counts per spin component
   * @param rotation_offset Starting index for each spin's rotation slice
   * @param rotation_size Number of rotation parameters per spin (n_occ*n_virt)
   * @param num_molecular_orbitals Total molecular orbitals in the system
   * @param unrestricted Whether alpha/beta densities are treated separately
   */
  GDMLineFunctor(const SCFImpl& scf_impl,
                 const RowMajorMatrix& C_pseudo_canonical,
                 const std::vector<int>& num_electrons,
                 const std::vector<int>& rotation_offset,
                 const std::vector<int>& rotation_size,
                 int num_molecular_orbitals, bool unrestricted)
      : scf_impl_(scf_impl),
        C_pseudo_canonical_(C_pseudo_canonical),
        num_electrons_(num_electrons),
        rotation_offset_(rotation_offset),
        rotation_size_(rotation_size),
        num_density_matrices_(unrestricted ? 2 : 1),
        num_molecular_orbitals_(num_molecular_orbitals),
        unrestricted_(unrestricted),
        cached_kappa_(Eigen::VectorXd()),
        cached_energy_(std::numeric_limits<double>::infinity()) {}

  /**
   * @brief Evaluate energy at given kappa vector x
   */
  double eval(const Eigen::VectorXd& x);

  /**
   * @brief Evaluate gradient at given kappa vector x. If the vector x has been
   * cached during eval(), the cached Fock matrix will be reused. Otherwise, it
   * will call eval() to compute both energy and Fock matrix.
   */
  Eigen::VectorXd grad(const Eigen::VectorXd& x);

  /**
   * @brief Compute dot product of two vectors to accommodate
   * line search method interface
   */
  static double dot(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
    return v1.dot(v2);
  }

  /**
   * @brief Perform axpy operation y = y + alpha * x to
   * accommodate line search method interface
   */
  static void axpy(double alpha, const Eigen::VectorXd& x, Eigen::VectorXd& y) {
    y.noalias() += alpha * x;
  }

  /**
   * @brief Get cached orbital coefficient matrix from last eval() call
   */
  const RowMajorMatrix& get_cached_C() const { return cached_C_; }

  /**
   * @brief Get cached density matrix from last eval() call
   */
  const RowMajorMatrix& get_cached_P() const { return cached_P_; }

 private:
  const double compare_kappa_tol_ = std::numeric_limits<double>::epsilon();
  // Const references to external data
  const SCFImpl& scf_impl_;
  const RowMajorMatrix& C_pseudo_canonical_;
  const std::vector<int>& num_electrons_;
  const std::vector<int>& rotation_offset_;
  const std::vector<int>& rotation_size_;

  // Value parameters
  const int num_density_matrices_;
  const int num_molecular_orbitals_;
  const bool unrestricted_;

  // Cache for avoiding redundant Fock matrix computation
  Eigen::VectorXd cached_kappa_;  // Cached kappa vector
  double cached_energy_;
  RowMajorMatrix cached_F_;  // Needed for gradient computation
  RowMajorMatrix cached_C_;  // For writing back to scf_impl
  RowMajorMatrix cached_P_;  // For writing back to scf_impl
};

double GDMLineFunctor::eval(const Eigen::VectorXd& x) {
  // Check if we've computed this kappa vector: if so, reuse cached result
  if (cached_kappa_.size() == x.size() &&
      (cached_kappa_ - x).norm() < compare_kappa_tol_) {
    return cached_energy_;
  }

  const Eigen::VectorXd& kappa_trial = x;

  cached_C_ = C_pseudo_canonical_;

  // Apply rotation for all spins with kappa_trial
  for (int i = 0; i < num_density_matrices_; i++) {
    auto kappa_spin =
        kappa_trial.segment(rotation_offset_[i], rotation_size_[i]);
    apply_orbital_rotation(cached_C_, i, kappa_spin, num_electrons_[i],
                           num_molecular_orbitals_);
  }

  // Compute P_trial from rotated C (for all spins)
  cached_P_ = RowMajorMatrix::Zero(
      num_density_matrices_ * num_molecular_orbitals_, num_molecular_orbitals_);

  for (int i = 0; i < num_density_matrices_; i++) {
    const int num_occupied_orbitals = num_electrons_[i];
    const double occupation_factor = unrestricted_ ? 1.0 : 2.0;

    cached_P_.block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                    num_molecular_orbitals_) =
        occupation_factor *
        cached_C_.block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                        num_occupied_orbitals) *
        cached_C_
            .block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                   num_occupied_orbitals)
            .transpose();
  }

  // Evaluate energy and Fock matrix using trial density matrix
  auto [energy, F_trial] =
      scf_impl_.evaluate_trial_density_energy_and_fock(cached_P_);

  // Cache all results for potential grad() call at same kappa
  cached_energy_ = energy;
  cached_F_ = F_trial;
  cached_kappa_ = x;

  return cached_energy_;
}

Eigen::VectorXd GDMLineFunctor::grad(const Eigen::VectorXd& x) {
  // Check if we've computed this kappa vector: if so, reuse cached result
  if (cached_kappa_.size() != x.size() ||
      (cached_kappa_ - x).norm() >= compare_kappa_tol_) {
    eval(x);
  }

  // Initialize the full gradient vector (concatenated for all spins)
  int total_rotation_size = 0;
  for (int i = 0; i < num_density_matrices_; i++) {
    total_rotation_size += rotation_size_[i];
  }
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(total_rotation_size);

  // Compute gradient for each spin component
  for (int i = 0; i < num_density_matrices_; i++) {
    const int num_occupied_orbitals = num_electrons_[i];
    const int num_virtual_orbitals =
        num_molecular_orbitals_ - num_occupied_orbitals;

    // Transform Fock matrix to MO basis: F_MO = C^T * F * C
    RowMajorMatrix F_MO =
        cached_C_
            .block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                   num_molecular_orbitals_)
            .transpose() *
        cached_F_.block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                        num_molecular_orbitals_) *
        cached_C_.block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                        num_molecular_orbitals_);

    // Extract occupied-virtual block and compute gradient
    // The -4.0 before F_{ia} comes from derivative of energy w.r.t. kappa
    // Reference: Helgaker, T., Jørgensen, P., & Olsen, J. (2000). Molecular
    // electronic-structure theory, Eq. 10.8.34 (2013 reprint edition)
    // -4.0 is for restricted closed-shell system. For unrestricted systems, the
    // gradient is computed separately for each spin component, in that case the
    // coefficient before F_{ia, spin} is -2.0
    RowMajorMatrix gradient_matrix =
        -(unrestricted_ ? 2.0 : 4.0) * F_MO.block(0, num_occupied_orbitals,
                                                  num_occupied_orbitals,
                                                  num_virtual_orbitals);

    // Flatten matrix to vector and store in appropriate segment
    gradient.segment(rotation_offset_[i], rotation_size_[i]) =
        Eigen::Map<const Eigen::VectorXd>(gradient_matrix.data(),
                                          rotation_size_[i]);
  }

  return gradient;
}

/**
 * @brief Implementation class for Geometric Direct Minimization (GDM)
 */
class GDM {
 public:
  /**
   * @brief Constructor for the GDM (Geometric Direct Minimization) class
   * @param[in] ctx Reference to SCFContext
   * @param[in] history_size_limit Maximum history size limit for BFGS in GDM
   *
   */
  explicit GDM(const SCFContext& ctx, const int history_size_limit);

  /**
   * @brief Perform one GDM SCF iteration for all spin components
   *
   * @param[in,out] scf_impl Reference to SCFImpl containing matrices and energy
   */
  void iterate(SCFImpl& scf_impl);

  /**
   * @brief Initialize GDM state when switching from DIIS
   *
   * @param[in] delta_energy_diis Energy change from DIIS algorithm
   * @param[in] total_energy Current SCF total energy
   */
  void initialize_from_diis(const double delta_energy_diis,
                            const double total_energy) {
    QDK_LOG_TRACE_ENTERING();
    delta_energy_ = delta_energy_diis;
    last_accepted_energy_ = total_energy;
    QDK_LOGGER().debug(
        "GDM initialized from DIIS: delta_energy={:.6e}, "
        "last_accepted_energy={:.12e}",
        delta_energy_, last_accepted_energy_);
  }

 private:
  /**
   * @brief Transform history matrices (either history_dgrad or history_kappa)
   * using current rotation matrices Uoo and Uvv to transform into the
   * pseudo-canonical orbital basis, K_new = Uoo^T * K_old * Uvv
   * @param[in,out] history History matrix block to be transformed (either
   * history_dgrad or history_kappa)
   * @param[in] history_size Number of history entries
   * @param[in] num_occupied_orbitals Number of electrons for current spin
   * @param[in] num_molecular_orbitals Number of molecular orbitals
   *
   */
  void transform_history_(Eigen::Block<RowMajorMatrix>& history,
                          const int history_size,
                          const int num_occupied_orbitals,
                          const int num_molecular_orbitals);

  /**
   * @brief Generate pseudo-canonical orbitals and apply transformations
   * @param[in] F Fock matrix in AO basis
   * @param[in,out] C Molecular orbital coefficient matrix
   * @param[in] spin_index Spin index (0 for alpha, 1 for beta)
   * @param[in,out] history_kappa_spin Block reference to history kappa for this
   * spin
   * @param[in,out] history_dgrad_spin Block reference to history dgrad for this
   * spin
   * @param[in,out] current_gradient_spin Segment reference to current gradient
   * for this spin
   *
   */
  void generate_pseudo_canonical_orbital_(
      const RowMajorMatrix& F, RowMajorMatrix& C, const int spin_index,
      Eigen::Block<RowMajorMatrix> history_kappa_spin,
      Eigen::Block<RowMajorMatrix> history_dgrad_spin,
      Eigen::VectorBlock<Eigen::VectorXd> current_gradient_spin);

  /// Reference to SCFContext
  const SCFContext& ctx_;  ///< Reference to SCFContext
  /// Energy change from the last step
  double delta_energy_ = std::numeric_limits<double>::infinity();

  /// Energy increase threshold for GDM step size rescaling
  const double nonpositive_threshold_ = std::numeric_limits<double>::epsilon();

  /// Number of electrons for alpha (0) and beta (1) spins
  std::vector<int> num_electrons_;

  /// History of kappa rotation vectors for each spin component
  RowMajorMatrix history_kappa_;
  /// History of gradient difference vectors for each spin component
  RowMajorMatrix history_dgrad_;
  /// Number of vectors saved in history
  int history_size_;
  /// Maximum number of vectors saved in history_kappa_ and history_dgrad_
  int history_size_limit_;
  /// Rotation size for each spin (n_occ * n_virt for alpha and beta)
  std::vector<int> rotation_size_;
  /// Offset for each spin in concatenated vectors
  std::vector<int> rotation_offset_;
  /// Total rotation size (sum of rotation_size_)
  int total_rotation_size_;

  /// Gradient vectors from the last iteration step for spin alpha and beta
  Eigen::VectorXd previous_gradient_;
  /// Gradient vectors from the current iteration step for spin alpha and beta
  Eigen::VectorXd current_gradient_;

  /// Eigenvalues of pseudo-canonical orbitals, used for building Hessian
  Eigen::VectorXd pseudo_canonical_eigenvalues_;

  /// Horizontal rotation matrix of occupied orbitals
  RowMajorMatrix Uoo_;
  /// Horizontal rotation matrix of virtual orbitals
  RowMajorMatrix Uvv_;

  Eigen::VectorXd kappa_;  // vertical rotation matrix of this step

  /// Energy of the last accepted step, used to decide if we rescale the kappa
  /// vector in this step
  double last_accepted_energy_;
  int gdm_step_count_;        // GDM iteration counter
  int num_density_matrices_;  // Number of density matrices (1 for restricted, 2
                              // for unrestricted)
};

GDM::GDM(const SCFContext& ctx, int history_size_limit)
    : ctx_(ctx),
      history_size_limit_(history_size_limit),
      last_accepted_energy_(std::numeric_limits<double>::infinity()),
      gdm_step_count_(0) {
  QDK_LOG_TRACE_ENTERING();
  const auto& cfg = *ctx.cfg;
  const auto& mol = *ctx.mol;

  const int num_molecular_orbitals =
      static_cast<int>(ctx.num_molecular_orbitals);
  const bool unrestricted = cfg.unrestricted;

  auto n_ecp_electrons = ctx.basis_set->n_ecp_electrons;
  auto spin = mol.multiplicity - 1;
  auto num_alpha_electrons =
      static_cast<int>((mol.n_electrons - n_ecp_electrons + spin) / 2);
  auto num_beta_electrons =
      static_cast<int>(mol.n_electrons - n_ecp_electrons - num_alpha_electrons);

  // Initialize member variables
  num_electrons_ = {num_alpha_electrons, num_beta_electrons};
  history_size_ = 0;
  pseudo_canonical_eigenvalues_ = Eigen::VectorXd::Zero(num_molecular_orbitals);
  if (history_size_limit < 1) {
    throw std::invalid_argument(
        "GDM history size limit must be at least 1, got: " +
        std::to_string(history_size_limit));
  }

  QDK_LOGGER().debug("GDM initialized with history_size_limit = {}",
                     history_size_limit_);
  num_density_matrices_ = unrestricted ? 2 : 1;

  // Calculate rotation sizes for each spin
  rotation_size_.resize(num_density_matrices_);
  rotation_offset_.resize(num_density_matrices_);

  total_rotation_size_ = 0;
  for (int spin_index = 0; spin_index < num_density_matrices_; spin_index++) {
    const int num_occupied_orbitals = num_electrons_[spin_index];
    const int num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;
    // Validate dimensions (negative values indicate invalid input)
    // Zero occupied or virtual orbitals is valid for unrestricted calculations
    // (e.g., H atom has 0 beta electrons)
    if (num_occupied_orbitals < 0) {
      throw std::invalid_argument(
          std::string("GDM: num_occupied_orbitals must be non-negative, got ") +
          std::to_string(num_occupied_orbitals) + " for spin " +
          std::to_string(spin_index));
    }
    if (num_virtual_orbitals < 0) {
      throw std::invalid_argument(
          std::string("GDM: num_virtual_orbitals must be non-negative, got ") +
          std::to_string(num_virtual_orbitals) + " for spin " +
          std::to_string(spin_index));
    }
    rotation_size_[spin_index] = num_occupied_orbitals * num_virtual_orbitals;
    rotation_offset_[spin_index] = total_rotation_size_;
    total_rotation_size_ += rotation_size_[spin_index];
  }

  // Initialize concatenated matrices and vectors
  history_kappa_ =
      RowMajorMatrix::Zero(history_size_limit_, total_rotation_size_);
  history_dgrad_ =
      RowMajorMatrix::Zero(history_size_limit_, total_rotation_size_);
  previous_gradient_ = Eigen::VectorXd::Zero(total_rotation_size_);
  current_gradient_ = Eigen::VectorXd::Zero(total_rotation_size_);
  kappa_ = Eigen::VectorXd::Zero(total_rotation_size_);
}

void GDM::transform_history_(Eigen::Block<RowMajorMatrix>& history,
                             const int history_size,
                             const int num_occupied_orbitals,
                             const int num_molecular_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  const int num_virtual_orbitals =
      num_molecular_orbitals - num_occupied_orbitals;
  // Validate dimensions (negative values indicate invalid input)
  if (num_occupied_orbitals < 0 || num_virtual_orbitals < 0) {
    throw std::invalid_argument(
        std::string(
            "transform_history_: invalid dimensions (num_occupied_orbitals=") +
        std::to_string(num_occupied_orbitals) +
        ", num_virtual_orbitals=" + std::to_string(num_virtual_orbitals) + ")");
  }
  // Skip transformation if either dimension is zero (no rotations for this
  // spin)
  if (num_occupied_orbitals == 0 || num_virtual_orbitals == 0) {
    return;
  }
  RowMajorMatrix temp =
      RowMajorMatrix::Zero(num_occupied_orbitals, num_virtual_orbitals);
  for (int line = 0; line < history_size; line++) {
    double* history_line_ptr = history.row(line).data();
    // K_ov (new) = Uoo^T * K_ov * Uvv
    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_occupied_orbitals, num_virtual_orbitals,
               num_virtual_orbitals, 1.0, history_line_ptr,
               num_virtual_orbitals, Uvv_.data(), num_virtual_orbitals, 0.0,
               temp.data(), num_virtual_orbitals);
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_occupied_orbitals, num_virtual_orbitals,
               num_occupied_orbitals, 1.0, Uoo_.data(), num_occupied_orbitals,
               temp.data(), num_virtual_orbitals, 0.0, history_line_ptr,
               num_virtual_orbitals);
  }
}

void GDM::generate_pseudo_canonical_orbital_(
    const RowMajorMatrix& F, RowMajorMatrix& C, const int spin_index,
    Eigen::Block<RowMajorMatrix> history_kappa_spin,
    Eigen::Block<RowMajorMatrix> history_dgrad_spin,
    Eigen::VectorBlock<Eigen::VectorXd> current_gradient_spin) {
  const int num_molecular_orbitals = C.cols();
  const int num_occupied_orbitals = num_electrons_[spin_index];
  const int num_virtual_orbitals =
      num_molecular_orbitals - num_occupied_orbitals;
  // Validate dimensions (negative values indicate invalid input)
  if (num_occupied_orbitals < 0 || num_virtual_orbitals < 0) {
    throw std::invalid_argument(
        std::string("generate_pseudo_canonical_orbital_: invalid dimensions "
                    "(num_occupied_orbitals=") +
        std::to_string(num_occupied_orbitals) +
        ", num_virtual_orbitals=" + std::to_string(num_virtual_orbitals) + ")");
  }
  // Skip if either dimension is zero (no rotations for this spin)
  if (num_occupied_orbitals == 0 || num_virtual_orbitals == 0) {
    return;
  }
  const int rotation_size = num_occupied_orbitals * num_virtual_orbitals;

  RowMajorMatrix F_MO =
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals)
          .transpose() *
      F.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals) *
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals);

  // Perform pseudo-canonical transformation and BFGS
  // Obtain pseudo-canonical orbitals. Foo and Fvv are symmetric matrices, but
  // the output eigenvectors are column-major
  Uoo_ = F_MO.block(0, 0, num_occupied_orbitals, num_occupied_orbitals);
  Uvv_ = F_MO.block(num_occupied_orbitals, num_occupied_orbitals,
                    num_virtual_orbitals, num_virtual_orbitals);

  // Compute eigenvalues/eigenvectors of occupied-occupied and virtual-virtual
  // blocks for pseudo-canonical orbital transformation
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, num_occupied_orbitals,
               Uoo_.data(), num_occupied_orbitals,
               pseudo_canonical_eigenvalues_.data());
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, num_virtual_orbitals,
               Uvv_.data(), num_virtual_orbitals,
               pseudo_canonical_eigenvalues_.data() + num_occupied_orbitals);

  // Transpose to convert column-major eigenvectors to row-major format
  Uoo_.transposeInPlace();
  Uvv_.transposeInPlace();

  // Transform occupied orbitals
  auto C_occ_view = C.block(num_molecular_orbitals * spin_index, 0,
                            num_molecular_orbitals, num_occupied_orbitals);
  RowMajorMatrix C_occ = C_occ_view.eval();
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_molecular_orbitals, num_occupied_orbitals,
             num_occupied_orbitals, 1.0, C_occ.data(), num_occupied_orbitals,
             Uoo_.data(), num_occupied_orbitals, 0.0, C_occ_view.data(),
             num_molecular_orbitals);

  // Transform virtual orbitals
  auto C_virt_view =
      C.block(num_molecular_orbitals * spin_index, num_occupied_orbitals,
              num_molecular_orbitals, num_virtual_orbitals);
  RowMajorMatrix C_virt = C_virt_view.eval();
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_molecular_orbitals, num_virtual_orbitals, num_virtual_orbitals,
             1.0, C_virt.data(), num_virtual_orbitals, Uvv_.data(),
             num_virtual_orbitals, 0.0, C_virt_view.data(),
             num_molecular_orbitals);

  // Transform the vectors in history_kappa and history_dgrad to
  // accommodate current pseudo-canonical orbitals
  transform_history_(history_kappa_spin, history_size_, num_occupied_orbitals,
                     num_molecular_orbitals);
  transform_history_(history_dgrad_spin, history_size_, num_occupied_orbitals,
                     num_molecular_orbitals);

  // Transform the gradient to accommodate current pseudo-canonical orbitals
  RowMajorMatrix current_gradient_matrix =
      Eigen::Map<RowMajorMatrix>(current_gradient_spin.data(),
                                 num_occupied_orbitals, num_virtual_orbitals);
  RowMajorMatrix current_gradient_transformed_matrix =
      Uoo_.transpose() * current_gradient_matrix * Uvv_;
  Eigen::VectorXd current_gradient_transformed = Eigen::Map<Eigen::VectorXd>(
      current_gradient_transformed_matrix.data(), rotation_size);
  current_gradient_spin = current_gradient_transformed;
}

void GDM::iterate(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  auto& C = scf_impl.orbitals_matrix();
  const auto& F = scf_impl.get_fock_matrix();

  const auto* cfg = ctx_.cfg;
  const int num_molecular_orbitals =
      static_cast<int>(ctx_.num_molecular_orbitals);
  const int num_density_matrices = cfg->unrestricted ? 2 : 1;

  // Check if there are any virtual orbitals for any spin component
  // If not, orbital rotation is not possible and we should skip GDM iteration
  if (total_rotation_size_ == 0) {
    QDK_LOGGER().warn(
        "GDM: No virtual orbitals available for orbital rotation. "
        "Skipping GDM iteration.");
    return;
  }

  // Compute current gradient and dgrad for each spin
  for (int i = 0; i < num_density_matrices; ++i) {
    const int num_occupied_orbitals = num_electrons_[i];
    const int num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;
    const int rotation_size = num_occupied_orbitals * num_virtual_orbitals;
    RowMajorMatrix F_MO =
        C.block(num_molecular_orbitals * i, 0, num_molecular_orbitals,
                num_molecular_orbitals)
            .transpose() *
        F.block(num_molecular_orbitals * i, 0, num_molecular_orbitals,
                num_molecular_orbitals) *
        C.block(num_molecular_orbitals * i, 0, num_molecular_orbitals,
                num_molecular_orbitals);

    // Extract occupied-virtual block and compute gradient
    // The -4.0 before F_{ia} comes from derivative of energy w.r.t. kappa
    // Reference: Helgaker, T., Jørgensen, P., & Olsen, J. (2000). Molecular
    // electronic-structure theory, Eq. 10.8.34 (2013 reprint edition)
    // -4.0 is for restricted closed-shell system. For unrestricted systems, the
    // gradient is computed separately for each spin component, in that case the
    // coefficient before F_{ia, spin} is -2.0
    RowMajorMatrix current_gradient_matrix =
        -(cfg->unrestricted ? 2.0 : 4.0) * F_MO.block(0, num_occupied_orbitals,
                                                      num_occupied_orbitals,
                                                      num_virtual_orbitals);
    current_gradient_.segment(rotation_offset_[i], rotation_size_[i]) =
        Eigen::Map<const Eigen::VectorXd>(current_gradient_matrix.data(),
                                          rotation_size);

    if (gdm_step_count_ != 0) {
      // Add new gradient difference to history for this spin
      history_dgrad_
          .block(0, rotation_offset_[i], history_size_limit_, rotation_size_[i])
          .row(history_size_) =
          current_gradient_.segment(rotation_offset_[i], rotation_size_[i]) -
          previous_gradient_.segment(rotation_offset_[i], rotation_size_[i]);
    }
  }

  // Update history size and manage history overflow. History for both spins are
  // concatenated together, so we only need to check once.
  if (gdm_step_count_ != 0) {
    history_size_++;

    if (history_size_ == history_size_limit_) {
      QDK_LOGGER().info(
          "GDM history size reached limit {}, removing oldest history "
          "vectors",
          history_size_limit_);
      const int num_rows_to_shift = history_size_limit_ - 1;
      history_kappa_.topRows(num_rows_to_shift) =
          history_kappa_.middleRows(1, num_rows_to_shift);
      history_dgrad_.topRows(num_rows_to_shift) =
          history_dgrad_.middleRows(1, num_rows_to_shift);
      history_size_--;
    }
  }

  // Build concatenated initial Hessian for all spins
  Eigen::VectorXd initial_hessian = Eigen::VectorXd::Zero(total_rotation_size_);

  for (int i = 0; i < num_density_matrices; ++i) {
    const int num_occupied_orbitals = num_electrons_[i];
    const int num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;

    auto history_kappa_spin = history_kappa_.block(
        0, rotation_offset_[i], history_size_limit_, rotation_size_[i]);
    auto history_dgrad_spin = history_dgrad_.block(
        0, rotation_offset_[i], history_size_limit_, rotation_size_[i]);
    auto current_gradient_spin =
        current_gradient_.segment(rotation_offset_[i], rotation_size_[i]);

    // Generate pseudo-canonical orbitals and transform gradient and history
    generate_pseudo_canonical_orbital_(
        F, C, i, history_kappa_spin, history_dgrad_spin, current_gradient_spin);

    // Build this spin's segment of initial Hessian
    // Reference: Helgaker, T., Jørgensen, P., & Olsen, J. (2000). Molecular
    // electronic-structure theory, Eq. 10.8.56 (2013 reprint edition)
    // 4.0 is for restricted closed-shell system. For unrestricted systems, the
    // gradient is computed separately for each spin component, in that case the
    // coefficient should be 2.0
    double initial_hessian_coeff = cfg->unrestricted ? 2.0 : 4.0;
    for (int j = 0; j < num_occupied_orbitals; j++) {
      for (int v = 0; v < num_virtual_orbitals; v++) {
        int index = rotation_offset_[i] + j * num_virtual_orbitals + v;
        double pseudo_canonical_energy_diff =
            std::abs(pseudo_canonical_eigenvalues_(num_occupied_orbitals + v) -
                     pseudo_canonical_eigenvalues_(j));
        initial_hessian(index) =
            std::max(initial_hessian_coeff * (std::abs(delta_energy_) +
                                              pseudo_canonical_energy_diff),
                     nonpositive_threshold_);
      }
    }
  }

  double latest_inverse_rho = 1.0;
  // BFGS two-loop recursion on concatenated vectors (runs once for all spins)
  if (history_size_ > 0) {
    QDK_LOGGER().debug(
        "Applying BFGS two-loop recursion with {} historical records",
        history_size_);

    std::vector<double> inverse_rho_values;
    for (int hist_idx = 0; hist_idx < history_size_; hist_idx++) {
      double sy_dot =
          history_kappa_.row(hist_idx).dot(history_dgrad_.row(hist_idx));
      inverse_rho_values.push_back(sy_dot);
    }
    latest_inverse_rho = inverse_rho_values[history_size_ - 1];

    if (latest_inverse_rho < nonpositive_threshold_) {
      // The kappa_ from the last step almost orthogonal to dgrad, or violates
      // curvature condition. Clear BFGS history.
      QDK_LOGGER().warn(
          "Invalid BFGS history curvature condition detected: latest inverse "
          "rho = {:.6e} < 0.",
          latest_inverse_rho);
      history_size_ = 0;
    } else {
      // BFGS two-loop recursion algorithm
      Eigen::VectorXd q = current_gradient_;
      std::vector<double> alpha_values;

      for (int hist_idx = history_size_ - 1; hist_idx >= 0; hist_idx--) {
        // inverse_rho_values[hist_idx] is independent of pseudo-canonical
        // transformation. The previous inverse_rho_values are larger than
        // nonpositive_threshold_. The latest_inverse_rho has been checked.
        double alpha =
            history_kappa_.row(hist_idx).dot(q) / inverse_rho_values[hist_idx];
        q = q - alpha * history_dgrad_.row(hist_idx).transpose();
        alpha_values.push_back(alpha);
      }

      Eigen::VectorXd r = Eigen::VectorXd::Zero(total_rotation_size_);
      for (int index = 0; index < total_rotation_size_; index++) {
        r(index) = q(index) / initial_hessian(index);
      }

      for (int j = 0; j < history_size_; j++) {
        double beta_value =
            history_dgrad_.row(j).dot(r) / inverse_rho_values[j];
        r = r + history_kappa_.row(j).transpose() *
                    (alpha_values[history_size_ - j - 1] - beta_value);
      }

      // Log BFGS debug information (last 5 values only)
#ifndef NDEBUG
      const int rho_size = static_cast<int>(inverse_rho_values.size());
      const int rho_start = std::max(0, rho_size - 5);
      const int rho_num_entries = rho_size - rho_start;

      std::string rho_str;
      rho_str.reserve(20 + 15 * rho_num_entries + 10);
      rho_str = "inverse Rho values: ";
      if (rho_start > 0) {
        rho_str += "... ";
      }
      for (int j = rho_start; j < rho_size; j++) {
        rho_str += fmt::format("{:.6e}; ", inverse_rho_values[j]);
      }
      QDK_LOGGER().debug(rho_str);

      const int alpha_size = static_cast<int>(alpha_values.size());
      const int alpha_start = std::max(0, alpha_size - 5);
      const int alpha_num_entries = alpha_size - alpha_start;

      std::string alpha_str;
      alpha_str.reserve(20 + 15 * alpha_num_entries + 10);
      alpha_str = "alpha values: ";
      if (alpha_start > 0) {
        alpha_str += "... ";
      }
      for (int j = alpha_start; j < alpha_size; j++) {
        alpha_str += fmt::format("{:.6e}; ", alpha_values[j]);
      }
      QDK_LOGGER().debug(alpha_str);
#endif

      kappa_ = -r;
      double kappa_dot_grad = kappa_.dot(current_gradient_);
      if (kappa_dot_grad > 0.0) {
        // Non-descent direction detected. Clear BFGS history
        QDK_LOGGER().warn(
            "Invalid BFGS search direction detected: kappa·grad = {:.6e} > 0. "
            "This indicates a non-descent direction.",
            kappa_dot_grad);
        history_size_ = 0;
      }
    }
  }

  if (history_size_ == 0) {
    // No history available, either first step or cleared history
    // kappa_ =  -H_0^{-1} * gradient
    QDK_LOGGER().info("No history available, using initial Hessian inverse");
    for (int index = 0; index < total_rotation_size_; index++) {
      kappa_(index) = -current_gradient_(index) / initial_hessian(index);
    }
  }

  // Save pseudo-canonical C for trials in the line search
  // NOTE: The call to C.eval() creates a full copy of the coefficient matrix.
  // This is necessary because the line search functor may modify the matrix
  // during energy evaluations, and we need to restore the original
  // pseudo-canonical state for each iteration.
  RowMajorMatrix C_pseudo_canonical = C.eval();

  // Create line search functor for energy evaluation
  GDMLineFunctor line_functor(scf_impl, C_pseudo_canonical, num_electrons_,
                              rotation_offset_, rotation_size_,
                              num_molecular_orbitals, cfg->unrestricted);

  Eigen::VectorXd start_kappa = Eigen::VectorXd::Zero(kappa_.size());
  Eigen::VectorXd kappa_dir = kappa_;  // Search direction
  double step_size = 1.0;              // Initial step size

  // assign variables for line search
  double energy_at_start_point = last_accepted_energy_;
  Eigen::VectorXd grad_at_start_point = current_gradient_;
  Eigen::VectorXd searched_kappa = Eigen::VectorXd::Zero(kappa_.size());
  // Function value at new point, initialized to energy_at_start_point
  double energy_at_searched_kappa = energy_at_start_point;
  // Gradient vector value at new point, initialized to grad_at_start_point
  Eigen::VectorXd grad_at_searched_kappa = grad_at_start_point;

  try {
    // Call Nocedal-Wright line search with strong Wolfe conditions
    nocedal_wright_line_search(line_functor, start_kappa, kappa_dir, step_size,
                               searched_kappa, energy_at_searched_kappa,
                               grad_at_searched_kappa);
  } catch (const std::exception& e) {
    // BFGS line search failed - likely bad search direction
    QDK_LOGGER().warn(
        "BFGS line search failed: {}. Falling back to steepest descent.",
        e.what());

    // Try line search with steepest descent direction
    try {
      kappa_dir = -current_gradient_;
      step_size = 1.0;
      energy_at_searched_kappa = energy_at_start_point;
      grad_at_searched_kappa = grad_at_start_point;
      searched_kappa.setZero();
      nocedal_wright_line_search(
          line_functor, start_kappa, kappa_dir, step_size, searched_kappa,
          energy_at_searched_kappa, grad_at_searched_kappa);
    } catch (const std::exception& e2) {
      // Even steepest descent line search failed; fall back to gradient norm
      QDK_LOGGER().warn(
          "Steepest descent line search also failed: {}. Checking gradient "
          "norm for convergence.",
          e2.what());

      const double grad_norm =
          current_gradient_.norm() / num_molecular_orbitals;
      const double og_threshold = ctx_.cfg->scf_algorithm.og_threshold;

      // grad_norm condition is to make the step acceptable when it meets the
      // convergence criterion. grad_norm_coeff here is to make it
      // consistent with |FPS - SPF| criterion in SCFImpl::check_convergence(),
      // |FPS-SPF|^2 / 2 = |grad / 4|^2 for restricted case and
      // |FPS-SPF|^2 / 2 = |grad / 2|^2 for unrestricted case.
      const double grad_norm_coeff =
          cfg->unrestricted ? std::sqrt(2.0) : std::sqrt(8.0);
      if (grad_norm < og_threshold * grad_norm_coeff) {
        QDK_LOGGER().warn(
            "Gradient norm {:.6e} below threshold; accepting zero orbital "
            "rotation.",
            grad_norm);
        searched_kappa.setZero();
        energy_at_searched_kappa = last_accepted_energy_;
        grad_at_searched_kappa = current_gradient_;
      } else {
        QDK_LOGGER().error(
            "Line search failed and gradient norm {:.6e} exceeds threshold. "
            "SCF iteration aborted.",
            grad_norm);
        throw std::runtime_error(
            "GDM SCF optimization failed: unable to find acceptable step; "
            "aborting SCF procedure.");
      }
    }
  }

  // Add optimal kappa to history and update previous gradient
  if (searched_kappa.norm() > nonpositive_threshold_) {
    scf_impl.orbitals_matrix() = line_functor.get_cached_C();
    scf_impl.density_matrix() = line_functor.get_cached_P();
  }

  delta_energy_ = energy_at_searched_kappa - last_accepted_energy_;
  last_accepted_energy_ = energy_at_searched_kappa;
  history_kappa_.row(history_size_) = searched_kappa;
  previous_gradient_ = current_gradient_;

  gdm_step_count_++;
}

}  // namespace impl

// Constructor for SCFAlgorithm interface
GDM::GDM(const SCFContext& ctx, const GDMConfig& gdm_config)
    : SCFAlgorithm(ctx),
      gdm_impl_(std::make_unique<impl::GDM>(
          ctx, gdm_config.gdm_bfgs_history_size_limit)) {
  QDK_LOG_TRACE_ENTERING();
}

GDM::~GDM() noexcept = default;

void GDM::iterate(SCFImpl& scf_impl) { gdm_impl_->iterate(scf_impl); }

void GDM::initialize_from_diis(const double delta_energy_diis,
                               const double total_energy) {
  QDK_LOG_TRACE_ENTERING();
  gdm_impl_->initialize_from_diis(delta_energy_diis, total_energy);
}

}  // namespace qdk::chemistry::scf
