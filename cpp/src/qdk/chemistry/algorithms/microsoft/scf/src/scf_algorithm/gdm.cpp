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
#include <vector>

#include "../scf/scf_impl.h"
#include "qdk/chemistry/scf/core/scf.h"
#include "qdk/chemistry/scf/core/types.h"
#include "spdlog/spdlog.h"
#include "util/matrix_exp.h"

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include "util/gpu/cuda_helper.h"
#include "util/gpu/matrix_operations.h"
#endif

namespace qdk::chemistry::scf {

namespace impl {
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
   * This method drives the GDM algorithm by performing orbital optimization
   * for each spin component (alpha and beta for unrestricted calculations).
   *
   * @param[in] F Current Fock matrix
   * @param[in,out] P Density matrix
   * @param[in,out] C Molecular orbital coefficients
   * @param[in] energy Current SCF energy
   */
  void iterate(const RowMajorMatrix& F, RowMajorMatrix& P, RowMajorMatrix& C,
               const double energy);

  /**
   * @brief Set the energy change from last two DIIS cycles for GDM algorithm
   *
   * @param[in] delta_energy_diis Energy change from DIIS algorithm
   */
  void set_delta_energy_diis(const double delta_energy_diis) {
    delta_energy_ = delta_energy_diis;
  }

 private:
  /**
   * @brief Apply orbital rotation using kappa vector
   * @param[in,out] C Molecular orbital coefficient matrix
   * @param[in] spin_index Spin index (0 for alpha, 1 for beta)
   * @param[in] kappa_vector The kappa vector to apply for rotation (can be
   * positive or negative)
   *
   * @details
   * This function constructs the antisymmetric kappa matrix and applies the
   * orbital rotation C = C * exp(kappa). For restoration, pass
   * -kappa_[spin_index]; for normal rotation, pass kappa_[spin_index].
   */
  void apply_orbital_rotation_(RowMajorMatrix& C, const int spin_index,
                               const Eigen::VectorXd& kappa_vector);

  /**
   * @brief Transform history matrices (either history_dgrad or history_kappa)
   * using current rotation matrices Uoo and Uvv to transform into the
   * pseudo-canonical orbital basis
   * @param[in,out] history History matrix to be transformed (either
   * history_dgrad or history_kappa)
   * @param[in] history_size Number of history entries
   * @param[in] num_occupied_orbitals Number of electrons for current spin
   * @param[in] num_molecular_orbitals Number of molecular orbitals
   *
   * @details
   * For each history entry, applies the transformation:
   * K_new = Uoo^T * K_old * Uvv
   * where K can be either kappa rotation vectors or gradient difference
   * vectors.
   */
  void transform_history_(RowMajorMatrix& history, const int history_size,
                          const int num_occupied_orbitals,
                          const int num_molecular_orbitals);

  /**
   * @brief Execute a complete GDM iteration step combining all GDM operations
   * @param[in] F Fock matrix in AO basis
   * @param[in,out] C Molecular orbital coefficient matrix
   * @param[in,out] P Density matrix
   * @param[in] spin_index Spin index (0 for alpha, 1 for beta)
   * @param[in] scf_total_energy Current SCF total energy
   * @param[in] occupation_factor Occupation factor for density matrix
   * construction
   *
   *
   * This is the main driver function that combines all GDM algorithmic
   * components into a single iteration step.
   */
  void gdm_iteration_step_(const RowMajorMatrix& F, RowMajorMatrix& C,
                           RowMajorMatrix& P, const int spin_index,
                           const double scf_total_energy,
                           const double occupation_factor);
  /// Reference to SCFContext
  const SCFContext& ctx_;  ///< Reference to SCFContext
  /// Energy change from the last step
  double delta_energy_ = std::numeric_limits<double>::infinity();

  /// Energy increase threshold for GDM step size rescaling
  const double rescale_kappa_denergy_threshold_ = 5e-4;

  /// Number of electrons for alpha (0) and beta (1) spins
  std::vector<int> num_electrons_;

  /// History of kappa rotation vectors for each spin component
  std::vector<RowMajorMatrix> history_kappa_;
  /// History of gradient difference vectors for each spin component
  std::vector<RowMajorMatrix> history_dgrad_;
  /// Number of vectors saved in history
  std::vector<int> history_size_;
  /// Maximum number of vectors saved in history_kappa_ and history_dgrad_
  int history_size_limit_;

  /// Gradient vectors from the last iteration step for spin alpha and beta
  std::vector<Eigen::VectorXd> previous_gradient_;
  /// Gradient vectors from the current iteration step for spin alpha and beta
  std::vector<Eigen::VectorXd> current_gradient_;

  /// Eigenvalues of pseudo-canonical orbitals, used for building Hessian
  Eigen::VectorXd pseudo_canonical_eigenvalues_;

  /// Horizontal rotation matrix of occupied orbitals
  RowMajorMatrix Uoo_;
  /// Horizontal rotation matrix of virtual orbitals
  RowMajorMatrix Uvv_;

  std::vector<Eigen::VectorXd> kappa_;  // vertical rotation matrix of this step
  /// Energy of the last accepted step, used to decide if we rescale the kappa
  /// vector in this step
  double last_accepted_energy_;
  /// The scale factor applied to kappa in the current step if energy increased
  /// too much compared to last accepted energy
  double kappa_scale_factor_;
  int gdm_step_count_;        // GDM iteration counter
  int num_density_matrices_;  // Number of density matrices (1 for restricted, 2
                              // for unrestricted)
};

GDM::GDM(const SCFContext& ctx, int history_size_limit)
    : ctx_(ctx),
      history_size_limit_(history_size_limit),
      last_accepted_energy_(std::numeric_limits<double>::infinity()),
      kappa_scale_factor_(0.0),
      gdm_step_count_(0) {
  // Calculate values from SCFContext
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
  history_size_ = std::vector<int>(unrestricted ? 2 : 1, 0);
  pseudo_canonical_eigenvalues_ = Eigen::VectorXd::Zero(num_molecular_orbitals);
  if (history_size_limit < 1) {
    throw std::invalid_argument(
        "GDM history size limit must be at least 1, got: " +
        std::to_string(history_size_limit));
  }

  spdlog::debug("GDM initialized with history_size_limit = {}",
                history_size_limit_);
  num_density_matrices_ = unrestricted ? 2 : 1;

  // Initialize vectors with proper size
  history_kappa_.resize(num_density_matrices_);
  history_dgrad_.resize(num_density_matrices_);
  previous_gradient_.resize(num_density_matrices_);
  current_gradient_.resize(num_density_matrices_);
  kappa_.resize(num_density_matrices_);

  for (int spin_index = 0; spin_index < num_density_matrices_; spin_index++) {
    history_size_[spin_index] = 0;
    const int num_occupied_orbitals = num_electrons_[spin_index];
    const int num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;
    const int rotation_size = num_occupied_orbitals * num_virtual_orbitals;
    history_kappa_[spin_index] =
        RowMajorMatrix::Zero(history_size_limit_, rotation_size);
    history_dgrad_[spin_index] =
        RowMajorMatrix::Zero(history_size_limit_, rotation_size);
    previous_gradient_[spin_index] = Eigen::VectorXd::Zero(rotation_size);
    current_gradient_[spin_index] = Eigen::VectorXd::Zero(rotation_size);
    kappa_[spin_index] = Eigen::VectorXd::Zero(rotation_size);
  }
}

void GDM::transform_history_(RowMajorMatrix& history, const int history_size,
                             const int num_occupied_orbitals,
                             const int num_molecular_orbitals) {
  const int num_virtual_orbitals =
      num_molecular_orbitals - num_occupied_orbitals;
  RowMajorMatrix temp =
      RowMajorMatrix::Zero(num_occupied_orbitals, num_virtual_orbitals);
  for (int line = 0; line < history_size; line++) {
    // K_ov (new) = Uoo^T * K_ov * Uvv
    // Note: BLAS expects column-major matrices, but our matrices are row-major
    double* history_line_ptr = history.row(line).data();

    // First step: temp = K_ov * Uvv (in row-major view)
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_virtual_orbitals, num_occupied_orbitals,
               num_virtual_orbitals, 1.0, Uvv_.data(), num_virtual_orbitals,
               history_line_ptr, num_virtual_orbitals, 0.0, temp.data(),
               num_virtual_orbitals);

    // Second step: result = Uoo^T * temp (in row-major view)
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               num_virtual_orbitals, num_occupied_orbitals,
               num_occupied_orbitals, 1.0, temp.data(), num_virtual_orbitals,
               Uoo_.data(), num_occupied_orbitals, 0.0, history_line_ptr,
               num_virtual_orbitals);
  }
}

void GDM::apply_orbital_rotation_(RowMajorMatrix& C, const int spin_index,
                                  const Eigen::VectorXd& kappa_vector) {
  const int num_molecular_orbitals = C.cols();
  int num_occupied_orbitals = num_electrons_[spin_index];
  const int num_virtual_orbitals =
      num_molecular_orbitals - num_occupied_orbitals;

  // Build the new MO by rotation matrix exp(kappa)
  const RowMajorMatrix kappa_matrix = Eigen::Map<const RowMajorMatrix>(
      kappa_vector.data(), num_occupied_orbitals, num_virtual_orbitals);
  RowMajorMatrix kappa_complete =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);

  // Construct antisymmetric kappa matrix
  kappa_complete.block(0, num_occupied_orbitals, num_occupied_orbitals,
                       num_virtual_orbitals) = kappa_matrix / 2.0;
  kappa_complete.block(num_occupied_orbitals, 0, num_virtual_orbitals,
                       num_occupied_orbitals) = -kappa_matrix.transpose() / 2.0;

  RowMajorMatrix exp_kappa =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
  matrix_exp(kappa_complete.data(), exp_kappa.data(), num_molecular_orbitals);

  RowMajorMatrix C_before_rotate =
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_molecular_orbitals, num_molecular_orbitals,
             num_molecular_orbitals, 1.0, exp_kappa.data(),
             num_molecular_orbitals, C_before_rotate.data(),
             num_molecular_orbitals, 0.0,
             C.block(num_molecular_orbitals * spin_index, 0,
                     num_molecular_orbitals, num_molecular_orbitals)
                 .data(),
             num_molecular_orbitals);
}

void GDM::gdm_iteration_step_(const RowMajorMatrix& F, RowMajorMatrix& C,
                              RowMajorMatrix& P, const int spin_index,
                              const double scf_total_energy,
                              const double occupation_factor) {
  // Numerical tolerance for avoiding division by zero
  static constexpr double denominator_min_limit = 1.0e-14;
  // Compute gradient vector
  const int num_molecular_orbitals = C.cols();
  const int num_occupied_orbitals = num_electrons_[spin_index];
  const int num_virtual_orbitals =
      num_molecular_orbitals - num_occupied_orbitals;
  const int rotation_size = num_occupied_orbitals * num_virtual_orbitals;
  RowMajorMatrix F_MO =
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals)
          .transpose() *
      F.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals) *
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_molecular_orbitals);
  RowMajorMatrix current_gradient_matrix =
      -4.0 * F_MO.block(0, num_occupied_orbitals, num_occupied_orbitals,
                        num_virtual_orbitals);
  current_gradient_[spin_index] = Eigen::Map<const Eigen::VectorXd>(
      current_gradient_matrix.data(), rotation_size);

  if (gdm_step_count_ != 0) {
    delta_energy_ = scf_total_energy - last_accepted_energy_;

    // Check if history is full and remove oldest vector if needed
    if (history_size_[spin_index] == history_size_limit_ - 1) {
      spdlog::info(
          "GDM history size reached limit {}, removing oldest history vectors",
          history_size_limit_);
      // Remove oldest history vectors by shifting all vectors forward
      const int num_rows_to_shift = history_size_limit_ - 1;
      history_kappa_[spin_index].topRows(num_rows_to_shift) =
          history_kappa_[spin_index].middleRows(1, num_rows_to_shift);
      history_dgrad_[spin_index].topRows(num_rows_to_shift - 1) =
          history_dgrad_[spin_index].middleRows(1, num_rows_to_shift - 1);
      history_size_[spin_index]--;
    }

    // Add new gradient difference to history
    history_dgrad_[spin_index].row(history_size_[spin_index]) =
        current_gradient_[spin_index] - previous_gradient_[spin_index];
    history_size_[spin_index]++;
  }

  // Judge whether to use shortened kappa step based on energy change
  if ((scf_total_energy - last_accepted_energy_ >
       rescale_kappa_denergy_threshold_) &&
      gdm_step_count_ != 0) {
    // energy increases too much, shorten kappa in the next step and
    // retain last_accepted_energy_
    spdlog::info(
        "Energy increased too much (increase: {:.6e}, record_good_energy: "
        "{:.6e}, "
        "tolerance: {:.6e}), will restore orbitals and shorten kappa in "
        "next step",
        scf_total_energy - last_accepted_energy_, last_accepted_energy_,
        rescale_kappa_denergy_threshold_);

    // if energy increase is too large, shorten kappa more; scale factor is
    // inverse of energy increase
    kappa_scale_factor_ =
        1.0 / std::max(scf_total_energy - last_accepted_energy_, 2.0);

    // Restore orbitals
    history_size_[spin_index]--;
    // At present history_kappa_ has history_size_[spin_index] + 1 rows, one
    // more than history_dgrad_
    kappa_[spin_index] =
        history_kappa_[spin_index].row(history_size_[spin_index]);
    const Eigen::VectorXd negative_kappa = -kappa_[spin_index];
    apply_orbital_rotation_(C, spin_index, negative_kappa);

    // Scale kappa for this step
    kappa_[spin_index] *= kappa_scale_factor_;
    history_kappa_[spin_index].row(history_size_[spin_index]) =
        kappa_[spin_index];  // update the history_kappa_
    spdlog::info(
        "Restored orbitals to previous step and scaled kappa by factor of {}",
        kappa_scale_factor_);
  } else {
    // else do real work this step
    last_accepted_energy_ = scf_total_energy;  // update the record
    kappa_scale_factor_ = 0.0;

    {  // Obtain pseudo-canonical orbitals
      // Occupied-occupied block. Foo is symmetric
      Uoo_ = F_MO.block(0, 0, num_occupied_orbitals, num_occupied_orbitals);
      // Virtual-virtual block. Fvv is also symmetric
      Uvv_ = F_MO.block(num_occupied_orbitals, num_occupied_orbitals,
                        num_virtual_orbitals, num_virtual_orbitals);

      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, num_occupied_orbitals,
                   Uoo_.data(), num_occupied_orbitals,
                   pseudo_canonical_eigenvalues_.data());

      lapack::syev(
          lapack::Job::Vec, lapack::Uplo::Lower, num_virtual_orbitals,
          Uvv_.data(), num_virtual_orbitals,
          pseudo_canonical_eigenvalues_.data() + num_occupied_orbitals);

      // Transpose to convert column-major eigenvectors to row-major format
      Uoo_.transposeInPlace();
      Uvv_.transposeInPlace();

      RowMajorMatrix C_occ =
          C.block(num_molecular_orbitals * spin_index, 0,
                  num_molecular_orbitals, num_occupied_orbitals);
      RowMajorMatrix C_occ_pseudo_canonical =
          RowMajorMatrix::Zero(num_molecular_orbitals, num_occupied_orbitals);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 num_occupied_orbitals, num_molecular_orbitals,
                 num_occupied_orbitals, 1.0, Uoo_.data(), num_occupied_orbitals,
                 C_occ.data(), num_occupied_orbitals, 0.0,
                 C_occ_pseudo_canonical.data(), num_occupied_orbitals);

      RowMajorMatrix C_virt =
          C.block(num_molecular_orbitals * spin_index, num_occupied_orbitals,
                  num_molecular_orbitals, num_virtual_orbitals);
      RowMajorMatrix C_virt_pseudo_canonical =
          RowMajorMatrix::Zero(num_molecular_orbitals, num_virtual_orbitals);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 num_virtual_orbitals, num_molecular_orbitals,
                 num_virtual_orbitals, 1.0, Uvv_.data(), num_virtual_orbitals,
                 C_virt.data(), num_virtual_orbitals, 0.0,
                 C_virt_pseudo_canonical.data(), num_virtual_orbitals);

      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_occupied_orbitals) = C_occ_pseudo_canonical;
      C.block(num_molecular_orbitals * spin_index, num_occupied_orbitals,
              num_molecular_orbitals, num_virtual_orbitals) =
          C_virt_pseudo_canonical;

      // Transform the vectors in history_kappa and history_dgrad to
      // accommodate current pseudo-canonical orbitals
      transform_history_(history_kappa_[spin_index], history_size_[spin_index],
                         num_occupied_orbitals, num_molecular_orbitals);
      transform_history_(history_dgrad_[spin_index], history_size_[spin_index],
                         num_occupied_orbitals, num_molecular_orbitals);

      // Transform the gradient to accommodate current pseudo-canonical orbitals
      RowMajorMatrix current_gradient_matrix = Eigen::Map<RowMajorMatrix>(
          current_gradient_[spin_index].data(), num_occupied_orbitals,
          num_virtual_orbitals);
      RowMajorMatrix current_gradient_transformed_matrix =
          Uoo_.transpose() * current_gradient_matrix * Uvv_;
      Eigen::VectorXd current_gradient_transformed =
          Eigen::Map<Eigen::VectorXd>(
              current_gradient_transformed_matrix.data(), rotation_size);
      current_gradient_[spin_index] = current_gradient_transformed;  // rotate
    }

    {  // Solve kappa vectors by BFGS two-loop recursion
      // Build diagonal initial Hessian matrix Hinit by energy difference of
      // occupied and virtual pseudo-canonical orbitals
      Eigen::VectorXd sqrt_initial_hessian =
          Eigen::VectorXd::Zero(rotation_size);
      for (int i = 0; i < num_occupied_orbitals; i++) {
        // Hessian index is iv = i * (num_molecular_orbitals -
        // num_occupied_orbitals) + v
        for (int v = 0; v < num_virtual_orbitals; v++) {
          double pseudo_canonical_energy_diff = std::max(
              pseudo_canonical_eigenvalues_(num_occupied_orbitals + v) -
                  pseudo_canonical_eigenvalues_(i),
              0.0);
          double sqrt_initial_hessian_diag_entry = sqrt(
              2.0 * (std::abs(delta_energy_) + pseudo_canonical_energy_diff));
          sqrt_initial_hessian(i * num_virtual_orbitals + v) =
              std::max(sqrt_initial_hessian_diag_entry, denominator_min_limit);
        }
      }

      // Scale history and gradient vectors
      RowMajorMatrix scaled_history_kappa =
          RowMajorMatrix::Zero(history_size_[spin_index], rotation_size);
      RowMajorMatrix scaled_history_dgrad =
          RowMajorMatrix::Zero(history_size_[spin_index], rotation_size);
      Eigen::VectorXd scaled_gradient_vector =
          Eigen::VectorXd::Zero(rotation_size);

      for (int hist = 0; hist < history_size_[spin_index]; hist++) {
        for (int index = 0; index < rotation_size; index++) {
          scaled_history_kappa(hist, index) =
              sqrt_initial_hessian(index) *
              history_kappa_[spin_index](hist, index);
          scaled_history_dgrad(hist, index) =
              history_dgrad_[spin_index](hist, index) /
              sqrt_initial_hessian(index);
        }
      }
      for (int index = 0; index < rotation_size; index++) {
        scaled_gradient_vector(index) =
            current_gradient_[spin_index](index) / sqrt_initial_hessian(index);
      }

      // BFGS two-loop recursion to solve scaled kappa vector
      Eigen::VectorXd scaled_kappa_vector;
      if (history_size_[spin_index] == 0) {
        // No history available, return gradient (identity matrix)
        spdlog::info("No history available, using identity matrix");
        scaled_kappa_vector = -scaled_gradient_vector;
      } else {
        spdlog::debug(
            "Applying BFGS two-loop recursion with {} historical records",
            history_size_[spin_index]);

        // Pre-compute valid BFGS correction pairs using indices
        std::vector<double> rho_values;  // rho_k = 1 / (s_k^T * y_k)

        // BFGS two-loop recursion algorithm
        // First loop: compute alpha values and update q
        Eigen::VectorXd q = scaled_gradient_vector;
        std::vector<double> alpha_values;

        for (int i = 0; i < history_size_[spin_index]; i++) {
          const auto& s_k = scaled_history_kappa.row(i);
          const auto& y_k = scaled_history_dgrad.row(i);

          double sy_dot = s_k.dot(y_k) > denominator_min_limit
                              ? s_k.dot(y_k)
                              : denominator_min_limit;
          rho_values.push_back(1.0 / sy_dot);
        }

        for (int i = history_size_[spin_index] - 1; i >= 0; i--) {
          double rho_kappa_dot_q =
              rho_values[i] * scaled_history_kappa.row(i).dot(q);
          q = q - rho_kappa_dot_q * scaled_history_dgrad.row(i).transpose();
          alpha_values.push_back(rho_kappa_dot_q);
        }

        // Apply initial Hessian approximation H_0 = I (identity)
        Eigen::VectorXd r = q;

        // Second loop: compute beta values and update r
        for (int i = 0; i < history_size_[spin_index]; i++) {
          double rho_dgrad_dot_r =
              rho_values[i] * scaled_history_dgrad.row(i).dot(r);
          r = r + scaled_history_kappa.row(i).transpose() *
                      (alpha_values[history_size_[spin_index] - i - 1] -
                       rho_dgrad_dot_r);
        }

        scaled_kappa_vector = -r;
      }

      // scale kappa vector back
      for (int index = 0; index < rotation_size; index++) {
        kappa_[spin_index](index) =
            scaled_kappa_vector(index) / sqrt_initial_hessian(index);
      }
      double kappa_norm = kappa_[spin_index].norm();
      double kappa_norm_limit = 10.0;
      if (kappa_norm > kappa_norm_limit) {
        spdlog::warn("Kappa norm is too large ({}), scaling down to {}",
                     kappa_norm, kappa_norm_limit);
        kappa_[spin_index] = kappa_[spin_index] / kappa_norm * kappa_norm_limit;
      }

      history_kappa_[spin_index].row(history_size_[spin_index]) =
          kappa_[spin_index];
      // Update previous gradient
      previous_gradient_[spin_index] = current_gradient_[spin_index];
    }
  }

  // Vertical orbital rotation
  apply_orbital_rotation_(C, spin_index, kappa_[spin_index]);

  // Update occupation matrix
  P.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
          num_molecular_orbitals) =
      occupation_factor *
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_occupied_orbitals) *
      C.block(num_molecular_orbitals * spin_index, 0, num_molecular_orbitals,
              num_occupied_orbitals)
          .transpose();
}

void GDM::iterate(const RowMajorMatrix& F, RowMajorMatrix& P, RowMajorMatrix& C,
                  const double energy) {
  const auto* cfg = ctx_.cfg;
  const int num_molecular_orbitals =
      static_cast<int>(ctx_.num_molecular_orbitals);
  const int num_density_matrices = cfg->unrestricted ? 2 : 1;

  // Perform GDM iteration for each spin
  for (int i = 0; i < num_density_matrices; ++i) {
    gdm_iteration_step_(F, C, P, i, energy, cfg->unrestricted ? 1.0 : 2.0);
  }

  // Increment GDM step counter
  gdm_step_count_++;
}

}  // namespace impl

// Constructor for SCFAlgorithm interface
GDM::GDM(const SCFContext& ctx, const GDMConfig& gdm_config)
    : SCFAlgorithm(ctx),
      gdm_impl_(std::make_unique<impl::GDM>(
          ctx, gdm_config.gdm_bfgs_history_size_limit)) {}

GDM::~GDM() noexcept = default;

void GDM::iterate(SCFImpl& scf_impl) {
  // Extract needed parameters from SCFImpl
  auto& P = scf_impl.density_matrix();
  const auto& F = scf_impl.get_fock_matrix();
  auto& C = scf_impl.orbitals_matrix();
  auto& res = ctx_.result;

  // Call impl with minimal parameters
  gdm_impl_->iterate(F, P, C, res.scf_total_energy);
}

void GDM::set_delta_energy_diis(const double delta_energy_diis) {
  gdm_impl_->set_delta_energy_diis(delta_energy_diis);
}

}  // namespace qdk::chemistry::scf
