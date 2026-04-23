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
static void apply_restricted_unrestricted_orbital_rotation(
    RowMajorMatrix& C, const int spin_index,
    const Eigen::VectorXd& kappa_vector, const int num_occupied_orbitals,
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
 * @brief Construct the ROHF kappa matrix and apply rotation C * exp(kappa)
 *
 * @param[in,out] C Molecular orbital coefficient matrix
 * @param[in] num_electrons Occupied orbital counts (alpha, beta)
 * @param[in] kappa_vector Concatenated ROHF rotation parameters
 * @param[in] num_molecular_orbitals Total molecular orbitals in the system
 */
static void apply_restricted_open_shell_orbital_rotation(
    RowMajorMatrix& C, const std::vector<int>& num_electrons,
    const Eigen::VectorXd& kappa_vector, const int num_molecular_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  const int num_closed_orbitals = num_electrons[1];
  const int num_open_orbitals = num_electrons[0] - num_closed_orbitals;
  const int num_virtual_orbitals = num_molecular_orbitals - num_electrons[0];

  int offset = 0;
  const int iw_size = num_closed_orbitals * num_open_orbitals;
  const int wa_size = num_open_orbitals * num_virtual_orbitals;
  const int ia_size = num_closed_orbitals * num_virtual_orbitals;

  const auto kappa_iw = Eigen::Map<const RowMajorMatrix>(
      kappa_vector.data() + offset, num_closed_orbitals, num_open_orbitals);
  offset += iw_size;

  const auto kappa_wa = Eigen::Map<const RowMajorMatrix>(
      kappa_vector.data() + offset, num_open_orbitals, num_virtual_orbitals);
  offset += wa_size;

  const auto kappa_ia = Eigen::Map<const RowMajorMatrix>(
      kappa_vector.data() + offset, num_closed_orbitals, num_virtual_orbitals);
  offset += ia_size;

  RowMajorMatrix kappa_complete =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);

  kappa_complete.block(0, num_closed_orbitals, num_closed_orbitals,
                       num_open_orbitals) = kappa_iw;
  kappa_complete.block(num_closed_orbitals, 0, num_open_orbitals,
                       num_closed_orbitals) = -kappa_iw.transpose();

  const int open_start = num_closed_orbitals;
  const int virtual_start = num_closed_orbitals + num_open_orbitals;

  kappa_complete.block(open_start, virtual_start, num_open_orbitals,
                       num_virtual_orbitals) = kappa_wa;
  kappa_complete.block(virtual_start, open_start, num_virtual_orbitals,
                       num_open_orbitals) = -kappa_wa.transpose();

  kappa_complete.block(0, virtual_start, num_closed_orbitals,
                       num_virtual_orbitals) = kappa_ia;
  kappa_complete.block(virtual_start, 0, num_virtual_orbitals,
                       num_closed_orbitals) = -kappa_ia.transpose();

  // 0.5 is for consistency with the gradient definition
  kappa_complete *= 0.5;

  RowMajorMatrix exp_kappa =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
  matrix_exp(kappa_complete.data(), exp_kappa.data(), num_molecular_orbitals);

  RowMajorMatrix C_before_rotate = C;
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_molecular_orbitals, num_molecular_orbitals,
             num_molecular_orbitals, 1.0, C_before_rotate.data(),
             num_molecular_orbitals, exp_kappa.data(), num_molecular_orbitals,
             0.0, C.data(), num_molecular_orbitals);
}

/**
 * @brief Compute restricted/unrestricted orbital gradients for all spins
 * @param[in] F Fock matrix in AO basis
 * @param[in] C Molecular orbital coefficient matrix
 * @param[in] num_electrons Occupied orbital counts per spin component
 * @param[in] rotation_offset Starting index for each spin's rotation slice
 * @param[in] rotation_size Number of rotation parameters per spin
 * (n_occ*n_virt)
 * @param[in] num_orbital_spin_blocks Number of spin blocks to iterate
 * @param[in] num_molecular_orbitals Total molecular orbitals in the system
 * @param[out] gradient Output gradient vector (concatenated across spins)
 */
static void compute_restricted_unrestricted_gradient(
    const RowMajorMatrix& F, const RowMajorMatrix& C,
    const std::vector<int>& num_electrons,
    const std::vector<int>& rotation_offset,
    const std::vector<int>& rotation_size, int num_orbital_spin_blocks,
    int num_molecular_orbitals, Eigen::VectorXd& gradient) {
  int total_rotation_size = 0;
  for (int i = 0; i < num_orbital_spin_blocks; ++i) {
    total_rotation_size += rotation_size[i];
  }
  gradient.setZero(total_rotation_size);

  for (int spin_index = 0; spin_index < num_orbital_spin_blocks; ++spin_index) {
    const int num_occupied_orbitals = num_electrons[spin_index];
    const int num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;
    const int spin_rotation_size = rotation_size[spin_index];

    if (spin_rotation_size == 0) {
      continue;
    }

    RowMajorMatrix F_MO =
        C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
                num_molecular_orbitals)
            .transpose() *
        F.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
                num_molecular_orbitals) *
        C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
                num_molecular_orbitals);

    // Extract occupied-virtual block and compute gradient
    // The -4.0 before F_{ia} comes from derivative of energy w.r.t. kappa
    // Reference: Helgaker, T., Jørgensen, P., & Olsen, J. (2000). Molecular
    // electronic-structure theory, Eq. 10.8.34 (2013 reprint edition)
    // -4.0 is for restricted closed-shell system. For unrestricted systems, the
    // gradient is computed separately for each spin component, in that case the
    // coefficient before F_{ia, spin} is -2.0
    RowMajorMatrix gradient_matrix =
        -((num_orbital_spin_blocks == 2) ? 2.0 : 4.0) *
        F_MO.block(0, num_occupied_orbitals, num_occupied_orbitals,
                   num_virtual_orbitals);

    gradient.segment(rotation_offset[spin_index], spin_rotation_size) =
        Eigen::Map<const Eigen::VectorXd>(gradient_matrix.data(),
                                          spin_rotation_size)
            .eval();
  }
}

/**
 * @brief Compute ROHF orbital gradients using the generalized Fock matrix
 *
 * The gradient is packed as (iw, wa, ia) blocks following the same
 * segmentation used by the ROHF kappa vector.
 *
 * @param[in] scf_impl SCF implementation for J/K construction
 * @param[in] C Molecular orbital coefficient matrix
 * @param[in] density_matrix Density matrix in AO basis
 * @param[in] num_electrons Occupied orbital counts (alpha, beta)
 * @param[in] rotation_size Rotation size for the ROHF kappa vector
 * @param[in] num_molecular_orbitals Total molecular orbitals in the system
 * @param[out] generalized_fock_mo Preallocated generalized Fock matrix in MO
 * @param[out] gradient Output gradient vector
 */
static void compute_restricted_open_shell_gradient(
    const SCFImpl& scf_impl, const RowMajorMatrix& C,
    const RowMajorMatrix& density_matrix, const std::vector<int>& num_electrons,
    const std::vector<int>& rotation_size, int num_molecular_orbitals,
    RowMajorMatrix& generalized_fock_mo, Eigen::VectorXd& gradient) {
  const int total_rotation_size = rotation_size[0];
  gradient.setZero(total_rotation_size);

  const int num_closed_orbitals = num_electrons[1];
  const int num_open_orbitals = num_electrons[0] - num_closed_orbitals;
  const int num_virtual_orbitals = num_molecular_orbitals - num_electrons[0];

  if (generalized_fock_mo.rows() != num_molecular_orbitals ||
      generalized_fock_mo.cols() != num_molecular_orbitals) {
    throw std::invalid_argument(
        "generalized_fock_mo must be preallocated to MO dimensions.");
  }

  const int num_atomic_orbitals = scf_impl.get_num_atomic_orbitals();
  const auto& H_ao_full = scf_impl.get_core_hamiltonian();
  const RowMajorMatrix H_ao =
      H_ao_full.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);

  RowMajorMatrix J_ao =
      RowMajorMatrix::Zero(2 * num_atomic_orbitals, num_atomic_orbitals);
  RowMajorMatrix K_ao =
      RowMajorMatrix::Zero(2 * num_atomic_orbitals, num_atomic_orbitals);
  scf_impl.build_jk_matrices(density_matrix, J_ao, K_ao);

  const auto J_alpha_ao = Eigen::Map<const RowMajorMatrix>(
      J_ao.data(), num_atomic_orbitals, num_atomic_orbitals);
  const auto J_beta_ao = Eigen::Map<const RowMajorMatrix>(
      J_ao.data() + num_atomic_orbitals * num_atomic_orbitals,
      num_atomic_orbitals, num_atomic_orbitals);
  const auto K_alpha_ao = Eigen::Map<const RowMajorMatrix>(
      K_ao.data(), num_atomic_orbitals, num_atomic_orbitals);
  const auto K_beta_ao = Eigen::Map<const RowMajorMatrix>(
      K_ao.data() + num_atomic_orbitals * num_atomic_orbitals,
      num_atomic_orbitals, num_atomic_orbitals);

  const auto C_ao_mo =
      C.block(0, 0, num_atomic_orbitals, num_molecular_orbitals);

  // Calculate Generalized Fock matrix in MO basis
  RowMajorMatrix H_mo = C_ao_mo.transpose() * H_ao * C_ao_mo;
  RowMajorMatrix J_alpha_mo = C_ao_mo.transpose() * J_alpha_ao * C_ao_mo;
  RowMajorMatrix J_beta_mo = C_ao_mo.transpose() * J_beta_ao * C_ao_mo;
  RowMajorMatrix K_alpha_mo = C_ao_mo.transpose() * K_alpha_ao * C_ao_mo;
  RowMajorMatrix K_beta_mo = C_ao_mo.transpose() * K_beta_ao * C_ao_mo;

  RowMajorMatrix F_I = H_mo + 2.0 * J_beta_mo - K_beta_mo;
  RowMajorMatrix F_A = J_alpha_mo - J_beta_mo - 0.5 * (K_alpha_mo - K_beta_mo);
  RowMajorMatrix Q = J_alpha_mo - J_beta_mo - (K_alpha_mo - K_beta_mo);

  RowMajorMatrix F_sum = F_I + F_A;
  generalized_fock_mo.block(0, 0, num_closed_orbitals, num_molecular_orbitals) =
      2.0 * F_sum.block(0, 0, num_closed_orbitals, num_molecular_orbitals);
  generalized_fock_mo.block(num_closed_orbitals, 0, num_open_orbitals,
                            num_molecular_orbitals) =
      (F_I + Q).block(num_closed_orbitals, 0, num_open_orbitals,
                      num_molecular_orbitals);

  int offset = 0;
  RowMajorMatrix grad_iw =
      -2.0 *
      (generalized_fock_mo.block(0, num_closed_orbitals, num_closed_orbitals,
                                 num_open_orbitals) -
       generalized_fock_mo
           .block(num_closed_orbitals, 0, num_open_orbitals,
                  num_closed_orbitals)
           .transpose());
  gradient.segment(offset, num_closed_orbitals * num_open_orbitals) =
      Eigen::Map<const Eigen::VectorXd>(grad_iw.data(), grad_iw.size()).eval();
  offset += num_closed_orbitals * num_open_orbitals;

  RowMajorMatrix grad_wa =
      -2.0 * generalized_fock_mo.block(num_closed_orbitals,
                                       num_closed_orbitals + num_open_orbitals,
                                       num_open_orbitals, num_virtual_orbitals);
  gradient.segment(offset, num_open_orbitals * num_virtual_orbitals) =
      Eigen::Map<const Eigen::VectorXd>(grad_wa.data(), grad_wa.size()).eval();
  offset += num_open_orbitals * num_virtual_orbitals;

  RowMajorMatrix grad_ia =
      -2.0 *
      generalized_fock_mo.block(0, num_closed_orbitals + num_open_orbitals,
                                num_closed_orbitals, num_virtual_orbitals);
  gradient.segment(offset, num_closed_orbitals * num_virtual_orbitals) =
      Eigen::Map<const Eigen::VectorXd>(grad_ia.data(), grad_ia.size()).eval();
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
   * @param scf_orbital_type Spin symmetry used across SCF algorithms
   */
  GDMLineFunctor(const SCFImpl& scf_impl,
                 const RowMajorMatrix& C_pseudo_canonical,
                 const std::vector<int>& num_electrons,
                 const std::vector<int>& rotation_offset,
                 const std::vector<int>& rotation_size,
                 int num_molecular_orbitals, SCFOrbitalType scf_orbital_type)
      : scf_impl_(scf_impl),
        C_pseudo_canonical_(C_pseudo_canonical),
        num_electrons_(num_electrons),
        rotation_offset_(rotation_offset),
        rotation_size_(rotation_size),
        num_orbital_spin_blocks_(
            scf_orbital_type == SCFOrbitalType::Unrestricted ? 2 : 1),
        num_density_matrices_(
            scf_orbital_type == SCFOrbitalType::Restricted ? 1 : 2),
        num_molecular_orbitals_(num_molecular_orbitals),
        scf_orbital_type_(scf_orbital_type),
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
  const int num_orbital_spin_blocks_;
  const int num_density_matrices_;
  const int num_molecular_orbitals_;
  const SCFOrbitalType scf_orbital_type_;

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
  if (scf_orbital_type_ == SCFOrbitalType::RestrictedOpenShell) {
    apply_restricted_open_shell_orbital_rotation(
        cached_C_, num_electrons_, kappa_trial, num_molecular_orbitals_);
  } else {
    for (int i = 0; i < num_orbital_spin_blocks_; i++) {
      auto kappa_spin =
          kappa_trial.segment(rotation_offset_[i], rotation_size_[i]);
      apply_restricted_unrestricted_orbital_rotation(
          cached_C_, i, kappa_spin, num_electrons_[i], num_molecular_orbitals_);
    }
  }

  // Compute P_trial from rotated C (for all spins)
  cached_P_ = RowMajorMatrix::Zero(
      num_density_matrices_ * num_molecular_orbitals_, num_molecular_orbitals_);

  for (int i = 0; i < num_density_matrices_; i++) {
    const int num_occupied_orbitals = num_electrons_[i];
    const double occupation_factor =
        (scf_orbital_type_ == SCFOrbitalType::Restricted) ? 2.0 : 1.0;
    auto P_block =
        cached_P_.block(num_molecular_orbitals_ * i, 0, num_molecular_orbitals_,
                        num_molecular_orbitals_);

    double* C_block_data = nullptr;
    if (scf_orbital_type_ == SCFOrbitalType::RestrictedOpenShell) {
      auto C_block =
          cached_C_.block(0, 0, num_molecular_orbitals_, num_occupied_orbitals);
      C_block_data = C_block.data();
    } else {
      auto C_block =
          cached_C_.block(num_molecular_orbitals_ * i, 0,
                          num_molecular_orbitals_, num_occupied_orbitals);
      C_block_data = C_block.data();
    }
    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::Trans,
               num_molecular_orbitals_, num_molecular_orbitals_,
               num_occupied_orbitals, occupation_factor, C_block_data,
               num_molecular_orbitals_, C_block_data, num_molecular_orbitals_,
               0.0, P_block.data(), num_molecular_orbitals_);
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

  Eigen::VectorXd gradient;
  if (scf_orbital_type_ == SCFOrbitalType::RestrictedOpenShell) {
    RowMajorMatrix generalized_fock_mo =
        RowMajorMatrix::Zero(num_molecular_orbitals_, num_molecular_orbitals_);
    compute_restricted_open_shell_gradient(
        scf_impl_, cached_C_, cached_P_, num_electrons_, rotation_size_,
        num_molecular_orbitals_, generalized_fock_mo, gradient);
  } else {
    compute_restricted_unrestricted_gradient(
        cached_F_, cached_C_, num_electrons_, rotation_offset_, rotation_size_,
        num_orbital_spin_blocks_, num_molecular_orbitals_, gradient);
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
   * using current rotation matrices to transform into the
   * pseudo-canonical orbital basis, K_new = U_left^T * K_old * U_right
   * @param[in,out] history History matrix block to be transformed (either
   * history_dgrad or history_kappa)
   * @param[in] u_left Left rotation matrix (e.g., Uii or Uaa)
   * @param[in] u_right Right rotation matrix (e.g., Uaa or Uww)
   * @param[in] history_size Number of history entries
   * @param[in] num_rows Number of rows in each unpacked history block
   * @param[in] num_cols Number of cols in each unpacked history block
   *
   */
  void transform_history_(Eigen::Block<RowMajorMatrix>& history,
                          const RowMajorMatrix& u_left,
                          const RowMajorMatrix& u_right, const int history_size,
                          const int num_rows, const int num_cols);

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
  void generate_restricted_unrestricted_pseudo_canonical_orbital_(
      const RowMajorMatrix& F, RowMajorMatrix& C, const int spin_index,
      Eigen::Block<RowMajorMatrix> history_kappa_spin,
      Eigen::Block<RowMajorMatrix> history_dgrad_spin,
      Eigen::VectorBlock<Eigen::VectorXd> current_gradient_spin);

  /**
   * @brief Build ROHF pseudo-canonical orbitals and transform ROHF history
   *
   * Diagonalizes closed/open/virtual MO blocks, rotates orbital coefficients
   * to the pseudo-canonical basis, and transforms packed ROHF BFGS history and
   * current gradient blocks (iw, wa, ia) into the updated basis.
   *
   * @param[in] F Spin-blocked Fock matrix in AO basis
   * @param[in,out] C ROHF orbital coefficient matrix to rotate
   * @param[in,out] history_kappa Packed ROHF kappa-history matrix
   * @param[in,out] history_dgrad Packed ROHF gradient-difference history
   * @param[in,out] current_gradient Packed ROHF current gradient vector
   */
  void generate_restricted_open_shell_pseudo_canonical_orbital_(
      const RowMajorMatrix& F, RowMajorMatrix& C, RowMajorMatrix& history_kappa,
      RowMajorMatrix& history_dgrad, Eigen::VectorXd& current_gradient);

  /**
   * @brief Build pseudo-canonical orbitals and initial Hessian across spins
   * @param[in] F Fock matrix in AO basis
   * @param[in,out] C Molecular orbital coefficient matrix
   * @param[in] num_molecular_orbitals Total number of molecular orbitals
   * @param[out] initial_hessian Output concatenated initial Hessian
   */
  void build_restricted_unrestricted_pseudo_canonical_orbitals_hessian_(
      const RowMajorMatrix& F, RowMajorMatrix& C, int num_molecular_orbitals,
      Eigen::VectorXd& initial_hessian);

  /**
   * @brief Build ROHF initial Hessian in pseudo-canonical basis
   *
   * Calls ROHF pseudo-canonical transformation, then fills the diagonal
   * preconditioner for packed ROHF rotations (iw, wa, ia) using orbital-energy
   * differences and the current absolute energy change.
   *
   * @param[in] F Spin-blocked Fock matrix in AO basis
   * @param[in,out] C ROHF orbital coefficient matrix (updated in place)
   * @param[in] num_molecular_orbitals Total number of molecular orbitals
   * @param[out] initial_hessian Output ROHF initial Hessian vector
   */
  void build_restricted_open_shell_pseudo_canonical_orbitals_hessian_(
      const RowMajorMatrix& F, RowMajorMatrix& C, int num_molecular_orbitals,
      Eigen::VectorXd& initial_hessian);

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

  /// Generalized Fock matrix in MO basis for ROHF
  RowMajorMatrix generalized_fock_mo_;

  Eigen::VectorXd kappa_;  // vertical rotation matrix of this step

  /// Energy of the last accepted step, used to decide if we rescale the kappa
  /// vector in this step
  double last_accepted_energy_;
  int gdm_step_count_;  // GDM iteration counter

  // Number of spin blocks (1 for restricted, 2 for unrestricted)
  const int num_orbital_spin_blocks_;
  // Number of density matrices (1 for closed shell, 2 for open shell)
  const int num_density_matrices_;
};

GDM::GDM(const SCFContext& ctx, int history_size_limit)
    : ctx_(ctx),
      history_size_limit_(history_size_limit),
      last_accepted_energy_(std::numeric_limits<double>::infinity()),
      gdm_step_count_(0),
      num_orbital_spin_blocks_(
          ctx.cfg->scf_orbital_type == SCFOrbitalType::Unrestricted ? 2 : 1),
      num_density_matrices_(
          ctx.cfg->scf_orbital_type == SCFOrbitalType::Restricted ? 1 : 2) {
  QDK_LOG_TRACE_ENTERING();
  const auto& cfg = *ctx.cfg;
  const auto& mol = *ctx.mol;

  const int num_molecular_orbitals =
      static_cast<int>(ctx.num_molecular_orbitals);

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
  if (cfg.scf_orbital_type == SCFOrbitalType::RestrictedOpenShell) {
    generalized_fock_mo_ =
        RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
  } else {
    generalized_fock_mo_ = RowMajorMatrix(0, 0);
  }
  if (history_size_limit < 1) {
    throw std::invalid_argument(
        "GDM history size limit must be at least 1, got: " +
        std::to_string(history_size_limit));
  }

  QDK_LOGGER().debug("GDM initialized with history_size_limit = {}",
                     history_size_limit_);

  // Calculate rotation sizes for each spin
  rotation_size_.resize(num_orbital_spin_blocks_);
  rotation_offset_.resize(num_orbital_spin_blocks_);

  if (cfg.scf_orbital_type == SCFOrbitalType::RestrictedOpenShell) {
    int num_closed_orbitals = num_electrons_[1];
    int num_open_orbitals = num_electrons_[0] - num_closed_orbitals;
    int num_virtual_orbitals = num_molecular_orbitals - num_electrons_[0];
    if (num_closed_orbitals < 0 || num_open_orbitals < 0 ||
        num_virtual_orbitals < 0) {
      throw std::invalid_argument(
          "Invalid ROHF system: "
          "num_closed_orbitals=" +
          std::to_string(num_closed_orbitals) +
          ", num_open_orbitals=" + std::to_string(num_open_orbitals) +
          ", num_virtual_orbitals=" + std::to_string(num_virtual_orbitals));
    }
    rotation_size_[0] = num_closed_orbitals * num_open_orbitals +
                        num_open_orbitals * num_virtual_orbitals +
                        num_closed_orbitals * num_virtual_orbitals;
    rotation_offset_[0] = 0;
    total_rotation_size_ = rotation_size_[0];
  } else {
    total_rotation_size_ = 0;
    for (int spin_index = 0; spin_index < num_orbital_spin_blocks_;
         spin_index++) {
      const int num_occupied_orbitals = num_electrons_[spin_index];
      const int num_virtual_orbitals =
          num_molecular_orbitals - num_occupied_orbitals;
      // Validate dimensions (negative values indicate invalid input)
      // Zero occupied or virtual orbitals is valid for unrestricted
      // calculations (e.g., H atom has 0 beta electrons)
      if (num_occupied_orbitals < 0) {
        throw std::invalid_argument(
            std::string(
                "GDM: num_occupied_orbitals must be non-negative, got ") +
            std::to_string(num_occupied_orbitals) + " for spin " +
            std::to_string(spin_index));
      }
      if (num_virtual_orbitals < 0) {
        throw std::invalid_argument(
            std::string(
                "GDM: num_virtual_orbitals must be non-negative, got ") +
            std::to_string(num_virtual_orbitals) + " for spin " +
            std::to_string(spin_index));
      }
      rotation_size_[spin_index] = num_occupied_orbitals * num_virtual_orbitals;
      rotation_offset_[spin_index] = total_rotation_size_;
      total_rotation_size_ += rotation_size_[spin_index];
    }
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
                             const RowMajorMatrix& u_left,
                             const RowMajorMatrix& u_right,
                             const int history_size, const int num_rows,
                             const int num_cols) {
  QDK_LOG_TRACE_ENTERING();
  // Validate dimensions (negative values indicate invalid input)
  if (num_rows < 0 || num_cols < 0) {
    throw std::invalid_argument(
        std::string("transform_history_: invalid dimensions (num_rows=") +
        std::to_string(num_rows) + ", num_cols=" + std::to_string(num_cols) +
        ")");
  }
  // Skip transformation if either dimension is zero
  if (num_rows == 0 || num_cols == 0) {
    return;
  }
  RowMajorMatrix temp = RowMajorMatrix::Zero(num_rows, num_cols);
  for (int line = 0; line < history_size; line++) {
    double* history_line_ptr = history.row(line).data();
    // K_new = U_left^T * K_old * U_right
    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_rows, num_cols, num_cols, 1.0, history_line_ptr, num_cols,
               u_right.data(), num_cols, 0.0, temp.data(), num_cols);
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_rows, num_cols, num_rows, 1.0, u_left.data(), num_rows,
               temp.data(), num_cols, 0.0, history_line_ptr, num_cols);
  }
}

/**
 * @brief Diagonalize an MO sub-block and rotate the corresponding orbital block
 *
 * This helper computes eigenpairs of a symmetric MO-space sub-block using
 * LAPACK `syev`, stores the resulting eigenvalues into a target slice, and
 * applies the eigenvector rotation to the corresponding orbital columns:
 * C_block <- C_block * U.
 *
 * @param[in,out] input_block_output_eigenvectors On input: symmetric block to
 * diagonalize. On output: eigenvectors used for rotation.
 * @param[in] block_size Dimension of the square sub-block to diagonalize.
 * @param[in,out] eigenvalues Vector receiving eigenvalues for this block.
 * @param[in] eigenvalue_start_index Start index in `eigenvalues` where this
 * block's eigenvalues are written.
 * @param[in,out] transformed_orbitals Orbital coefficient sub-block rotated in
 * place.
 * @param[in] num_atomic_orbitals Row count used for GEMM in the orbital
 * rotation.
 * @param[in] num_molecular_orbitals Leading dimension of the parent orbital
 * coefficient matrix.
 */
static void calculate_pseudo_canonical_orbital_block(
    RowMajorMatrix& input_block_output_eigenvectors, const int block_size,
    Eigen::VectorXd& eigenvalues, const int eigenvalue_start_index,
    Eigen::Block<RowMajorMatrix> transformed_orbitals,
    const int num_atomic_orbitals, const int num_molecular_orbitals) {
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, block_size,
               input_block_output_eigenvectors.data(), block_size,
               eigenvalues.data() + eigenvalue_start_index);
  input_block_output_eigenvectors.transposeInPlace();
  RowMajorMatrix copied_orbitals = transformed_orbitals;
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_atomic_orbitals, block_size, block_size, 1.0,
             copied_orbitals.data(), block_size,
             input_block_output_eigenvectors.data(), block_size, 0.0,
             transformed_orbitals.data(), num_molecular_orbitals);
}

/**
 * @brief Rotate a packed gradient sub-block to the pseudo-canonical basis
 *
 * Interprets a contiguous slice of current_vector (starting at start_index)
 * as a num_rows x num_cols matrix G, applies G' = u_left^T * G * u_right,
 * and writes the flattened result back into the corresponding slice of
 * transformed_vector.
 *
 * This keeps the gradient representation aligned with the updated
 * pseudo-canonical orbitals after block diagonalization and orbital rotation.
 *
 * @param[in] current_vector Source packed gradient vector.
 * @param[in] start_index Start index of the sub-block inside the packed vector.
 * @param[in] num_rows Row count of the unpacked gradient block.
 * @param[in] num_cols Column count of the unpacked gradient block.
 * @param[in] u_left Left rotation matrix.
 * @param[in] u_right Right rotation matrix.
 * @param[out] transformed_vector Destination packed vector receiving the
 * transformed sub-block.
 */
static void rotate_gradient_to_pseudo_canonical_basis(
    const Eigen::Ref<const Eigen::VectorXd>& current_vector,
    const int start_index, const int num_rows, const int num_cols,
    const RowMajorMatrix& u_left, const RowMajorMatrix& u_right,
    Eigen::Ref<Eigen::VectorXd> transformed_vector) {
  RowMajorMatrix current_matrix = Eigen::Map<const RowMajorMatrix>(
      current_vector.data() + start_index, num_rows, num_cols);
  RowMajorMatrix transformed_matrix =
      u_left.transpose() * current_matrix * u_right;
  transformed_vector.segment(start_index, num_rows * num_cols) =
      Eigen::Map<const Eigen::VectorXd>(transformed_matrix.data(),
                                        num_rows * num_cols);
}

void GDM::generate_restricted_unrestricted_pseudo_canonical_orbital_(
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

  RowMajorMatrix F_MO =
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals)
          .transpose() *
      F.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals) *
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals);

  // Perform pseudo-canonical transformation
  // Diagonalize occupied/virtual blocks and rotate orbitals to the
  // pseudo-canonical basis.
  RowMajorMatrix Uii =
      F_MO.block(0, 0, num_occupied_orbitals, num_occupied_orbitals);
  auto C_occ_view = C.block(num_molecular_orbitals * spin_index, 0,
                            num_molecular_orbitals, num_occupied_orbitals);
  calculate_pseudo_canonical_orbital_block(
      Uii, num_occupied_orbitals, pseudo_canonical_eigenvalues_, 0, C_occ_view,
      num_molecular_orbitals, num_molecular_orbitals);

  RowMajorMatrix Uaa = F_MO.block(num_occupied_orbitals, num_occupied_orbitals,
                                  num_virtual_orbitals, num_virtual_orbitals);
  auto C_virt_view =
      C.block(num_molecular_orbitals * spin_index, num_occupied_orbitals,
              num_molecular_orbitals, num_virtual_orbitals);
  calculate_pseudo_canonical_orbital_block(
      Uaa, num_virtual_orbitals, pseudo_canonical_eigenvalues_,
      num_occupied_orbitals, C_virt_view, num_molecular_orbitals,
      num_molecular_orbitals);

  // Transform the vectors in history_kappa, history_dgrad, and
  // current_gradient_spin to accommodate current pseudo-canonical orbitals
  transform_history_(history_kappa_spin, Uii, Uaa, history_size_,
                     num_occupied_orbitals, num_virtual_orbitals);
  transform_history_(history_dgrad_spin, Uii, Uaa, history_size_,
                     num_occupied_orbitals, num_virtual_orbitals);

  rotate_gradient_to_pseudo_canonical_basis(
      current_gradient_spin, 0, num_occupied_orbitals, num_virtual_orbitals,
      Uii, Uaa, current_gradient_spin);
}

void GDM::build_restricted_unrestricted_pseudo_canonical_orbitals_hessian_(
    const RowMajorMatrix& F, RowMajorMatrix& C, int num_molecular_orbitals,
    Eigen::VectorXd& initial_hessian) {
  initial_hessian.setZero(total_rotation_size_);

  for (int i = 0; i < num_orbital_spin_blocks_; ++i) {
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
    generate_restricted_unrestricted_pseudo_canonical_orbital_(
        F, C, i, history_kappa_spin, history_dgrad_spin, current_gradient_spin);

    // Build this spin's segment of initial Hessian
    // Reference: Helgaker, T., Jorgensen, P., & Olsen, J. (2000). Molecular
    // electronic-structure theory, Eq. 10.8.56 (2013 reprint edition)
    // 4.0 is for restricted closed-shell system. For unrestricted systems, the
    // gradient is computed separately for each spin component, in that case the
    // coefficient should be 2.0
    double initial_hessian_coeff = (num_orbital_spin_blocks_ == 2) ? 2.0 : 4.0;
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
}

void GDM::generate_restricted_open_shell_pseudo_canonical_orbital_(
    const RowMajorMatrix& F, RowMajorMatrix& C, RowMajorMatrix& history_kappa,
    RowMajorMatrix& history_dgrad, Eigen::VectorXd& current_gradient) {
  const int num_molecular_orbitals = C.cols();
  const int num_atomic_orbitals = C.rows();
  const int num_closed_orbitals = num_electrons_[1];
  const int num_open_orbitals = num_electrons_[0] - num_closed_orbitals;
  const int num_occupied_orbitals = num_electrons_[0];
  const int num_virtual_orbitals = num_molecular_orbitals - num_electrons_[0];
  const int size_closed_open = num_closed_orbitals * num_open_orbitals;
  const int size_open_virtual = num_open_orbitals * num_virtual_orbitals;
  const int size_closed_virtual = num_closed_orbitals * num_virtual_orbitals;

  RowMajorMatrix F_up_mo =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
  RowMajorMatrix F_dn_mo = F_up_mo;

  F_up_mo.noalias() =
      C.transpose() *
      F.block(0, 0, num_atomic_orbitals, num_molecular_orbitals) * C;
  F_dn_mo.noalias() = C.transpose() *
                      F.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                              num_molecular_orbitals) *
                      C;

  // Perform pseudo-canonical transformation
  // Diagonalize occupied/virtual blocks and rotate orbitals to the
  // pseudo-canonical basis.
  RowMajorMatrix Uii =
      0.5 * (F_up_mo.block(0, 0, num_closed_orbitals, num_closed_orbitals) +
             F_dn_mo.block(0, 0, num_closed_orbitals, num_closed_orbitals));
  auto C_closed_view = C.block(0, 0, num_atomic_orbitals, num_closed_orbitals);
  calculate_pseudo_canonical_orbital_block(
      Uii, num_closed_orbitals, pseudo_canonical_eigenvalues_, 0, C_closed_view,
      num_atomic_orbitals, num_molecular_orbitals);

  RowMajorMatrix Uww =
      0.5 * (F_up_mo.block(num_closed_orbitals, num_closed_orbitals,
                           num_open_orbitals, num_open_orbitals) +
             F_dn_mo.block(num_closed_orbitals, num_closed_orbitals,
                           num_open_orbitals, num_open_orbitals));
  auto C_open_view =
      C.block(0, num_closed_orbitals, num_atomic_orbitals, num_open_orbitals);
  calculate_pseudo_canonical_orbital_block(
      Uww, num_open_orbitals, pseudo_canonical_eigenvalues_,
      num_closed_orbitals, C_open_view, num_atomic_orbitals,
      num_molecular_orbitals);

  RowMajorMatrix Uaa =
      0.5 * (F_up_mo.block(num_occupied_orbitals, num_occupied_orbitals,
                           num_virtual_orbitals, num_virtual_orbitals) +
             F_dn_mo.block(num_occupied_orbitals, num_occupied_orbitals,
                           num_virtual_orbitals, num_virtual_orbitals));
  auto C_virt_view = C.block(0, num_occupied_orbitals, num_atomic_orbitals,
                             num_virtual_orbitals);
  calculate_pseudo_canonical_orbital_block(
      Uaa, num_virtual_orbitals, pseudo_canonical_eigenvalues_,
      num_occupied_orbitals, C_virt_view, num_atomic_orbitals,
      num_molecular_orbitals);

  // Transform the vectors in history_kappa, history_dgrad, and current_gradient
  // to accommodate current pseudo-canonical orbitals
  int offset = 0;
  auto history_kappa_iw =
      history_kappa.block(0, 0, history_size_, size_closed_open);
  transform_history_(history_kappa_iw, Uii, Uww, history_size_,
                     num_closed_orbitals, num_open_orbitals);
  auto history_dgrad_iw =
      history_dgrad.block(0, 0, history_size_, size_closed_open);
  transform_history_(history_dgrad_iw, Uii, Uww, history_size_,
                     num_closed_orbitals, num_open_orbitals);
  offset += size_closed_open;

  auto history_kappa_wa =
      history_kappa.block(0, offset, history_size_, size_open_virtual);
  transform_history_(history_kappa_wa, Uww, Uaa, history_size_,
                     num_open_orbitals, num_virtual_orbitals);
  auto history_dgrad_wa =
      history_dgrad.block(0, offset, history_size_, size_open_virtual);
  transform_history_(history_dgrad_wa, Uww, Uaa, history_size_,
                     num_open_orbitals, num_virtual_orbitals);
  offset += size_open_virtual;

  auto history_kappa_ia =
      history_kappa.block(0, offset, history_size_, size_closed_virtual);
  transform_history_(history_kappa_ia, Uii, Uaa, history_size_,
                     num_closed_orbitals, num_virtual_orbitals);
  auto history_dgrad_ia =
      history_dgrad.block(0, offset, history_size_, size_closed_virtual);
  transform_history_(history_dgrad_ia, Uii, Uaa, history_size_,
                     num_closed_orbitals, num_virtual_orbitals);

  Eigen::VectorXd current_gradient_transformed =
      Eigen::VectorXd(current_gradient.size());

  offset = 0;
  rotate_gradient_to_pseudo_canonical_basis(
      current_gradient, offset, num_closed_orbitals, num_open_orbitals, Uii,
      Uww, current_gradient_transformed);
  offset += size_closed_open;

  rotate_gradient_to_pseudo_canonical_basis(
      current_gradient, offset, num_open_orbitals, num_virtual_orbitals, Uww,
      Uaa, current_gradient_transformed);
  offset += size_open_virtual;

  rotate_gradient_to_pseudo_canonical_basis(
      current_gradient, offset, num_closed_orbitals, num_virtual_orbitals, Uii,
      Uaa, current_gradient_transformed);

  current_gradient = current_gradient_transformed;
}

void GDM::build_restricted_open_shell_pseudo_canonical_orbitals_hessian_(
    const RowMajorMatrix& F, RowMajorMatrix& C, int num_molecular_orbitals,
    Eigen::VectorXd& initial_hessian) {
  initial_hessian.setZero(total_rotation_size_);

  generate_restricted_open_shell_pseudo_canonical_orbital_(
      F, C, history_kappa_, history_dgrad_, current_gradient_);

  const int num_closed_orbitals = num_electrons_[1];
  const int num_open_orbitals = num_electrons_[0] - num_closed_orbitals;
  const int num_virtual_orbitals = num_molecular_orbitals - num_electrons_[0];

  // between closed and open shells, or between open and virtual orbitals, the
  // coefficient should be 2.0
  double initial_hessian_coeff = 2.0;
  int offset = 0;
  for (int j = 0; j < num_closed_orbitals; j++) {
    for (int v = 0; v < num_open_orbitals; v++) {
      int index = j * num_open_orbitals + v;
      double pseudo_canonical_energy_diff =
          std::abs(pseudo_canonical_eigenvalues_(num_closed_orbitals + v) -
                   pseudo_canonical_eigenvalues_(j));
      initial_hessian(index) =
          std::max(initial_hessian_coeff *
                       (std::abs(delta_energy_) + pseudo_canonical_energy_diff),
                   nonpositive_threshold_);
    }
  }
  offset += num_closed_orbitals * num_open_orbitals;

  for (int v = 0; v < num_open_orbitals; v++) {
    for (int a = 0; a < num_virtual_orbitals; a++) {
      int index = offset + v * num_virtual_orbitals + a;
      double pseudo_canonical_energy_diff =
          std::abs(pseudo_canonical_eigenvalues_(num_electrons_[0] + a) -
                   pseudo_canonical_eigenvalues_(num_closed_orbitals + v));
      initial_hessian(index) =
          std::max(initial_hessian_coeff *
                       (std::abs(delta_energy_) + pseudo_canonical_energy_diff),
                   nonpositive_threshold_);
    }
  }
  offset += num_open_orbitals * num_virtual_orbitals;

  // between closed and virtual orbitals, the coefficient should be 4.0
  initial_hessian_coeff = 4.0;
  for (int j = 0; j < num_closed_orbitals; j++) {
    for (int a = 0; a < num_virtual_orbitals; a++) {
      int index = offset + j * num_virtual_orbitals + a;
      double pseudo_canonical_energy_diff =
          std::abs(pseudo_canonical_eigenvalues_(num_electrons_[0] + a) -
                   pseudo_canonical_eigenvalues_(j));
      initial_hessian(index) =
          std::max(initial_hessian_coeff *
                       (std::abs(delta_energy_) + pseudo_canonical_energy_diff),
                   nonpositive_threshold_);
    }
  }
}

void GDM::iterate(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  auto& C = scf_impl.orbitals_matrix();
  const auto& F = scf_impl.get_fock_matrix();

  const auto* cfg = ctx_.cfg;
  const int num_molecular_orbitals =
      static_cast<int>(ctx_.num_molecular_orbitals);

  // Check if there are any virtual orbitals for any spin component
  // If not, orbital rotation is not possible and we should skip GDM iteration
  if (total_rotation_size_ == 0) {
    QDK_LOGGER().warn(
        "GDM: No virtual orbitals available for orbital rotation. "
        "Skipping GDM iteration.");
    return;
  }

  if (cfg->scf_orbital_type == SCFOrbitalType::RestrictedOpenShell) {
    generalized_fock_mo_.setZero();
    compute_restricted_open_shell_gradient(
        scf_impl, C, scf_impl.get_density_matrix(), num_electrons_,
        rotation_size_, num_molecular_orbitals, generalized_fock_mo_,
        current_gradient_);
  } else {
    compute_restricted_unrestricted_gradient(
        F, C, num_electrons_, rotation_offset_, rotation_size_,
        num_orbital_spin_blocks_, num_molecular_orbitals, current_gradient_);
  }

  if (gdm_step_count_ != 0) {
    // Add new gradient difference to history for all spins
    history_dgrad_.row(history_size_) = current_gradient_ - previous_gradient_;
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

  Eigen::VectorXd initial_hessian;
  if (cfg->scf_orbital_type == SCFOrbitalType::RestrictedOpenShell) {
    build_restricted_open_shell_pseudo_canonical_orbitals_hessian_(
        F, C, num_molecular_orbitals, initial_hessian);
  } else {
    build_restricted_unrestricted_pseudo_canonical_orbitals_hessian_(
        F, C, num_molecular_orbitals, initial_hessian);
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
                              num_molecular_orbitals, cfg->scf_orbital_type);

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
          (cfg->scf_orbital_type == SCFOrbitalType::Unrestricted)
              ? std::sqrt(2.0)
              : std::sqrt(8.0);
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
