// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "vvhv.hpp"

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/int1e.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <iostream>
#include <lapack.hh>
#include <memory>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "../utils.hpp"
#include "iterative_localizer_base.hpp"
#include "pipek_mezey.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

/**
 * @brief VV-HV localization scheme implementation.
 *
 * This class implements the VV-HV localization algorithm for molecular
 * orbitals. It partitions the virtual space into valence virtuals and hard
 * virtuals based on projection onto a minimal basis, then uses a pluggable
 * localization scheme (e.g., Pipek-Mezey) to localize the occupied space and
 * valence virtuals space and project Atomic orbitals into the hard virtual
 * space.
 *
 * This class holds a pointer to an IterativeOrbitalLocalizationScheme for the
 * actual localization work. This allows flexibility to use different
 * localization methods (Pipek-Mezey, Foster-Boys, etc.) for occupied orbitals
 * and valence virtuals in the future.
 */
class VVHVLocalization : public IterativeOrbitalLocalizationScheme {
 public:
  /**
   * @brief Constructor for VVHVLocalization.
   *
   * @param settings Localization settings
   * @param basis_set The basis set used for the orbitals
   * @param ao_overlap Atomic orbital overlap matrix
   * @param minimal_basis_name Name of the minimal basis set (e.g., "STO-3G")
   * @param inner_localizer Reusable inner localization scheme (e.g.,
   * Pipek-Mezey)
   */
  VVHVLocalization(
      const IterativeOrbitalLocalizationSettings& settings,
      std::shared_ptr<data::BasisSet> basis_set,
      const Eigen::MatrixXd& ao_overlap, const std::string& minimal_basis_name,
      std::shared_ptr<IterativeOrbitalLocalizationScheme> inner_localizer)
      : IterativeOrbitalLocalizationScheme(settings),
        basis_set_(basis_set),
        overlap_ori_(ao_overlap),
        minimal_basis_name_(minimal_basis_name),
        basis_ori_fp_(utils::microsoft::convert_basis_set_from_qdk(*basis_set)),
        inner_localizer_(inner_localizer) {
    QDK_LOG_TRACE_ENTERING();

    // Initialize all data structures and pre-compute integrals
    initialize();
  }

  ~VVHVLocalization() = default;

  /**
   * @brief Localize the virtual orbitals using the VV-HV algorithm.
   *
   * This method performs the localization of virtual molecular orbitals
   * according to the VV-HV scheme. It constructs the valence virtual space by
   * projecting the occupied space out of the minimal basis, then localizes
   * valence virtuals and constructs hard virtual orbitals.
   *
   * @param occupied_orbitals Matrix of occupied orbital coefficients
   * (num_atomic_orbitals x num_occupied_orbitals)
   * @return Localized virtual orbital coefficient matrix (num_atomic_orbitals x
   * num_virtual_orbitals)
   */
  Eigen::MatrixXd localize(const Eigen::MatrixXd& occupied_orbitals);

 private:
  // Input parameters
  std::shared_ptr<data::BasisSet> basis_set_;
  std::string minimal_basis_name_;

  // Inner Localization scheme
  // Currently uses Pipek-Mezey, but can be replaced with other schemes in the
  // future.
  std::shared_ptr<IterativeOrbitalLocalizationScheme> inner_localizer_;

  // Pre-computed integral data (computed during initialization)
  const Eigen::MatrixXd& overlap_ori_;  // Overlap in original basis
  Eigen::MatrixXd
      overlap_mix_;  // Cross overlap between original and minimal basis

  // Pre-computed dipole and quadrupole integrals (if weighted_orthogonalization
  // is true)
  std::unique_ptr<qcs::RowMajorMatrix>
      dipole_integrals_;  // 3*num_atomic_orbitals x num_atomic_orbitals matrix
  std::unique_ptr<qcs::RowMajorMatrix>
      quadrupole_integrals_;  // 6*num_atomic_orbitals x num_atomic_orbitals
                              // matrix

  // Basis set data
  std::shared_ptr<qcs::BasisSet>
      basis_ori_fp_;  // Original basis set in LightAIMD format
  std::shared_ptr<qcs::BasisSet>
      minimal_basis_fp_;  // Minimal basis set in LightAIMD format

  /**
   * @brief Perform symmetric orthonormalization of orbital coefficients.
   *
   * Computes the overlap matrix S = C^T * overlap_inp * C, diagonalizes it,
   * validates the eigenvalue structure, and transforms the orbitals to be
   * orthonormal with respect to overlap_inp.
   *
   * @param num_atomic_orbitals Number of atomic orbitals (rows in C and
   * overlap_inp)
   * @param num_orbitals Number of orbitals (columns in C)
   * @param overlap_inp Overlap matrix (num_atomic_orbitals x
   * num_atomic_orbitals) in the representation which input orbitals C are given
   * @param C Input orbital coefficient matrix (num_atomic_orbitals x
   * num_orbitals)
   * @param C_out Output orthonormalized orbital coefficient matrix
   * (num_atomic_orbitals x num_orbitals_out, num_orbitals_out = num_orbitals -
   * expected_near_zero)
   * @param ortho_threshold Threshold for orthonormalization (eigenvalue cutoff)
   * @param expected_near_zero Expected number of near-zero eigenvalues to skip
   * (skip check if 0)
   * @param error_label Label for error messages
   * @param separation_ratio Required ratio of eigenvalue[M+1]/eigenvalue[M] for
   * sufficient separation
   */
  void orthonormalization(int num_atomic_orbitals, int num_orbitals,
                          const double* overlap_inp, double* C, double* C_out,
                          double ortho_threshold = 1e-6,
                          unsigned int expected_near_zero = 0,
                          const std::string& error_label = "",
                          double separation_ratio = 5.0);

  /**
   * @brief Check the eigenvalue structure when projecting out some space.
   *
   * Validates that the number of near-zero eigenvalues matches expectations and
   * that there is sufficient separation between near-zero and non-near-zero
   * eigenvalues.
   *
   * @param eigenvalues Array of eigenvalues to check (length total_eigenvalues)
   * @param expected_near_zero Expected number of near-zero eigenvalues
   * @param total_eigenvalues Total number of eigenvalues in the array
   * @param error_label Label for error messages
   * @param separation_ratio Required ratio of eigenvalue[M+1]/eigenvalue[M] for
   * sufficient separation
   */
  void check_eigenvalue_structure(const double* eigenvalues,
                                  int expected_near_zero, int total_eigenvalues,
                                  const std::string& error_label,
                                  double separation_ratio = 5.0) const;

  /**
   * @brief Calculate orbital spreads for given orbitals using dipole and
   * quadrupole integrals.
   *
   * Uses the dipole and quadrupole class members to compute orbital spreads.
   *
   * @param orbitals Matrix of orbital coefficients (num_atomic_orbitals x
   * num_orbitals)
   * @param spreads Output vector of orbital spreads
   * :math:`\left( \langle r^2 \rangle -\lvert\langle r \rangle\rvert^2 \right)`
   */
  void calculate_orbital_spreads(const Eigen::MatrixXd& orbitals,
                                 Eigen::VectorXd& spreads) const;

  /**
   * @brief Build (optionally sub-localize) hard virtual orbitals for a single
   * atom+angular momentum block.
   *
   * @param overlap_ori_al Overlap matrix block (original basis) for the atom+l
   * block (size num_atomic_orbitals_al_ori x num_atomic_orbitals_al_ori)
   * @param overlap_mix_al Mixed overlap block between original and minimal
   * basis (size num_atomic_orbitals_al_ori x num_atomic_orbitals_al_min)
   * @param bf_al_ori Index list (global AO indices) for this atom+l in the
   * original basis
   * @param bf_al_min Index list for this atom+l in the minimal basis
   * @param C_hv_al (Output) Matrix (num_atomic_orbitals_ori x nhv_al) to
   * receive hard virtual coefficients (global AO representation)
   * @param num_atomic_orbitals_ori Total number of original atomic orbitals
   * (global row dimension for C_hv_al)
   * @param atom_index Atom index (for logging / diagnostics)
   * @param l Angular momentum quantum number (for logging / diagnostics)
   */
  void proto_hv(const Eigen::MatrixXd& overlap_ori_al,
                const Eigen::MatrixXd& overlap_mix_al,
                const std::vector<int>& bf_al_ori,
                const std::vector<int>& bf_al_min, Eigen::MatrixXd& C_hv_al,
                int num_atomic_orbitals_ori, int atom_index, int l);

  /**
   * @brief Initialize data structures and compute overlap matrices and
   * integrals.
   *
   * This method computes overlap matrices, basis set transformations, and
   * optionally dipole/quadrupole integrals for orbital spread calculations.
   * Should be called once during construction.
   */
  void initialize();

  /**
   * @brief Calculate the valence virtual space from occupied orbitals.
   *
   * This method constructs the unlocalized valence virtual space by projecting
   * occupied space out of the minimal basis and orthonormalizing.
   *
   * @param occupied_orbitals Input occupied orbital coefficients matrix
   * @return Unlocalized valence virtual orbitals
   */
  Eigen::MatrixXd calculate_valence_virtual(
      const Eigen::MatrixXd& occupied_orbitals);

  /**
   * @brief Localize valence virtual orbitals using the inner localizer.
   *
   * This method localizes valence virtual orbitals using the inner_localizer_
   * (currently Pipek-Mezey by default).
   *
   * @param C_valence_unloc Unlocalized valence virtual orbitals
   * @return Localized valence virtual orbitals
   */
  Eigen::MatrixXd localize_valence_virtual(
      const Eigen::MatrixXd& C_valence_unloc);

  /**
   * @brief Localize hard virtual orbitals for given valence virtual orbitals.
   *
   * Hard virtuals are constructed atom-by-atom and
   * angular-momentum-by-angular-momentum, then optionally localized within each
   * block.
   *
   * @param C_minimal_unloc Combined minimal space orbitals (valence virtual +
   * occupied)
   * @return Hard virtual orbitals only
   */
  Eigen::MatrixXd localize_hard_virtuals(
      const Eigen::MatrixXd& C_minimal_unloc);
};

// VVHVLocalization implementation
Eigen::MatrixXd VVHVLocalization::localize(
    const Eigen::MatrixXd& occupied_orbitals) {
  QDK_LOG_TRACE_ENTERING();

  const auto* ori_bs = this->basis_ori_fp_.get();  // Original/full basis set
  const auto* min_bs = this->minimal_basis_fp_.get();  // Minimal basis set
  const auto num_atomic_orbitals_ori = ori_bs->num_atomic_orbitals;
  const auto num_atomic_orbitals_min = min_bs->num_atomic_orbitals;
  const auto num_occupied_orbitals = occupied_orbitals.cols();
  const auto n_val_virt = num_atomic_orbitals_min - num_occupied_orbitals;
  const auto nhv = num_atomic_orbitals_ori - num_atomic_orbitals_min;
  const auto num_virtual_orbitals = n_val_virt + nhv;

  // Calculate valence virtuals
  Eigen::MatrixXd C_valence_virtual =
      calculate_valence_virtual(occupied_orbitals);

  // Localize valence virtual orbitals
  Eigen::MatrixXd C_valence_loc = localize_valence_virtual(C_valence_virtual);

  // Combine C_valence_virtual and occupied_orbitals to get C_minimal
  Eigen::MatrixXd C_minimal(num_atomic_orbitals_ori, num_atomic_orbitals_min);
  C_minimal.block(0, 0, num_atomic_orbitals_ori, num_occupied_orbitals) =
      occupied_orbitals;
  C_minimal.block(0, num_occupied_orbitals, num_atomic_orbitals_ori,
                  n_val_virt) = C_valence_virtual;

  // Localize hard virtuals and combine with valence virtuals
  Eigen::MatrixXd hard_orbitals_loc = localize_hard_virtuals(C_minimal);

  // Concatenate C_valence_loc and hard_orbitals_loc to form localized_orbitals
  Eigen::MatrixXd localized_orbitals =
      Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, num_virtual_orbitals);
  localized_orbitals.block(0, 0, num_atomic_orbitals_ori, n_val_virt) =
      C_valence_loc;
  localized_orbitals.block(0, n_val_virt, num_atomic_orbitals_ori, nhv) =
      hard_orbitals_loc;

  converged_ = true;
  QDK_LOGGER().info("VV-HV localization completed successfully");

  return localized_orbitals;
}

void VVHVLocalization::initialize() {
  QDK_LOG_TRACE_ENTERING();

  // Check that minimal_basis_name_ is either sto-3g or sto-3g*
  if (minimal_basis_name_ != "sto-3g" && minimal_basis_name_ != "sto-3g*") {
    throw std::runtime_error(
        "VVHVLocalization requires minimal_minimal_basis_name_ to be either "
        "'sto-3g' or 'sto-3g*', got: " +
        minimal_basis_name_);
  }

  // Initialize overlap matrices and transformation matrix similar to
  // BasisMapper
  const auto* ori_bs = this->basis_ori_fp_.get();  // Original/full basis set

  // Create minimal basis set
  auto mol_structure = ori_bs->mol;
  this->minimal_basis_fp_ = qcs::BasisSet::from_database_json(
      mol_structure, minimal_basis_name_, ori_bs->mode, ori_bs->pure, true);
  const auto* minimal_bs = this->minimal_basis_fp_.get();  // Minimal basis set

  const auto num_atoms = mol_structure->n_atoms;
  const auto num_atomic_orbitals_ori = ori_bs->num_atomic_orbitals;
  const auto num_atomic_orbitals_min = minimal_bs->num_atomic_orbitals;
  this->overlap_mix_.resize(num_atomic_orbitals_ori, num_atomic_orbitals_min);

  // Compute the original basis overlap matrix overlap_ori
  qcs::OneBodyIntegral ori_bs_1ee(ori_bs, ori_bs->mol.get(),
                                  qcs::mpi_default_input());

  // Pre-compute dipole and quadrupole integrals if weighted orthogonalization
  // is enabled
  bool weighted_orthogonalization =
      settings_.get_or_default<bool>("weighted_orthogonalization", true);
  if (weighted_orthogonalization) {
    // Allocate and compute dipole integrals
    dipole_integrals_ = std::make_unique<qcs::RowMajorMatrix>(
        3 * num_atomic_orbitals_ori, num_atomic_orbitals_ori);
    ori_bs_1ee.dipole_integral(dipole_integrals_->data());

    // Allocate and compute quadrupole integrals
    quadrupole_integrals_ = std::make_unique<qcs::RowMajorMatrix>(
        6 * num_atomic_orbitals_ori, num_atomic_orbitals_ori);
    ori_bs_1ee.quadrupole_integral(quadrupole_integrals_->data());

    QDK_LOGGER().debug(
        "VVHVLocalization: Pre-computed dipole and quadrupole integrals for "
        "orbital spread calculations");
  }

  // Compute the mixed overlap matrix overlap_mix
  {
    auto ori_bs_libs = qcs::libint2_util::convert_to_libint_basisset(*ori_bs);
    auto minimal_bs_libs =
        qcs::libint2_util::convert_to_libint_basisset(*minimal_bs);
    auto basis_mode_bra = ori_bs->mode;

    libint2::Engine engine(
        libint2::Operator::overlap,
        std::max(ori_bs_libs.max_nprim(), minimal_bs_libs.max_nprim()),
        std::max(ori_bs_libs.max_l(), minimal_bs_libs.max_l()), 0);

    auto shell2bf_ori = ori_bs_libs.shell2bf();
    auto shell2bf_min = minimal_bs_libs.shell2bf();

    for (auto i = 0; i < ori_bs_libs.size(); ++i)
      for (auto j = 0; j < minimal_bs_libs.size(); ++j) {
        auto& bra = ori_bs_libs[i];
        auto& ket = minimal_bs_libs[j];

        const auto nbra = bra.size();
        const auto nket = ket.size();

        const auto bra_st = shell2bf_ori[i];
        const auto ket_st = shell2bf_min[j];

        engine.compute(bra, ket);
        auto* buf = engine.results()[0];
        if (buf) {
          Eigen::Map<const qcs::RowMajorMatrix> buf_map(buf, nbra, nket);
          this->overlap_mix_.block(bra_st, ket_st, nbra, nket) = buf_map;
        }
      }
  }
}

Eigen::MatrixXd VVHVLocalization::calculate_valence_virtual(
    const Eigen::MatrixXd& occupied_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  QDK_LOGGER().debug("VVHV::calculate_valence_virtual()");
  const auto* ori_bs = this->basis_ori_fp_.get();  // Original/full basis set
  const auto num_atomic_orbitals_ori = ori_bs->num_atomic_orbitals;
  const auto num_atomic_orbitals_min =
      this->minimal_basis_fp_.get()->num_atomic_orbitals;
  const auto num_occupied_orbitals = occupied_orbitals.cols();

  const auto n_val_virt = num_atomic_orbitals_min - num_occupied_orbitals;

  if (n_val_virt < 0) {
    throw std::runtime_error("VVHVLocalization: minimal basis size (" +
                             std::to_string(num_atomic_orbitals_min) +
                             ") smaller than occupied count (" +
                             std::to_string(num_occupied_orbitals) + ").");
  }

  Eigen::MatrixXd temp =
      Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, num_atomic_orbitals_ori);
  Eigen::MatrixXd temp2 =
      Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, num_atomic_orbitals_min);

  // print number of occupied and valence virtual orbitals
  QDK_LOGGER().info(
      "VVHV::now using minimal basis '{}' with {} atomic orbitals",
      minimal_basis_name_, num_atomic_orbitals_min);
  QDK_LOGGER().debug(
      "VVHV::number of occupied orbitals: {}, valence virtual orbitals: {}",
      num_occupied_orbitals, n_val_virt);

  if (num_atomic_orbitals_min > num_atomic_orbitals_ori) {
    throw std::runtime_error(
        "VVHVLocalization: minimal basis size exceeds original basis size.");
  }

  // Form T = overlap_ori**-1 * overlap_mix, which is the minimal basis in the
  // representation of the original basis namely xi coefficients in the old
  // tutorial (Subotnik et al. JCP 123, 114108 (2005)) or ~p in the new tutorial
  // (Wang et al. JCTC 21, 1163 (2025)). We need to orthonormalize T first
  // according to new tutorial
  Eigen::MatrixXd T(num_atomic_orbitals_ori, num_atomic_orbitals_min);
  {
    temp = this->overlap_ori_;   // lapack::gelss overwrites input
    temp2 = this->overlap_mix_;  // the unnormalized T
    std::vector<double> W11(num_atomic_orbitals_ori);
    double _rcond = -1;
    int64_t RANK11;
    lapack::gelss(num_atomic_orbitals_ori, num_atomic_orbitals_ori,
                  num_atomic_orbitals_min, temp.data(), num_atomic_orbitals_ori,
                  temp2.data(), num_atomic_orbitals_ori, W11.data(), _rcond,
                  &RANK11);
    this->orthonormalization(num_atomic_orbitals_ori, num_atomic_orbitals_min,
                             this->overlap_ori_.data(), temp2.data(), T.data(),
                             1e-6, 0,
                             "initial orthonormalization of minimal basis");
  }

  const double* C_occ_ptr = occupied_orbitals.data();

  // Now we want to project out all components of the occupied space from the
  // virtual space C_v' = (I - C_o * C_o**H * S_11) * T
  Eigen::MatrixXd C_mp_wo_occ = T;  //
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_atomic_orbitals_ori, num_atomic_orbitals_min,
             num_atomic_orbitals_ori, 1.0, this->overlap_ori_.data(),
             num_atomic_orbitals_ori, T.data(), num_atomic_orbitals_ori, 0.0,
             temp.data(), num_atomic_orbitals_ori);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             num_occupied_orbitals, num_atomic_orbitals_min,
             num_atomic_orbitals_ori, 1.0, C_occ_ptr, num_atomic_orbitals_ori,
             temp.data(), num_atomic_orbitals_ori, 0.0, temp2.data(),
             num_occupied_orbitals);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_atomic_orbitals_ori, num_atomic_orbitals_min,
             num_occupied_orbitals, -1.0, C_occ_ptr, num_atomic_orbitals_ori,
             temp2.data(), num_occupied_orbitals, 1.0, C_mp_wo_occ.data(),
             num_atomic_orbitals_ori);

  // Then form the new overlap matrix, S_ij = \sim_uv C_{ui} S_11_{uv} C_{vj}, C
  // =C_mp_wo_occ here
  this->orthonormalization(
      num_atomic_orbitals_ori, num_atomic_orbitals_min,
      this->overlap_ori_.data(), C_mp_wo_occ.data(), temp.data(), 1e-6,
      num_occupied_orbitals,
      "calculating minimal space (after projecting out occupied space)", 5.0);

  // Assign minimal basis unlocalized valence+virtual orbitals
  Eigen::MatrixXd C_valence_unloc =
      temp.block(0, 0, num_atomic_orbitals_ori,
                 num_atomic_orbitals_min - num_occupied_orbitals);

  return C_valence_unloc;
}

Eigen::MatrixXd VVHVLocalization::localize_valence_virtual(
    const Eigen::MatrixXd& C_valence_unloc) {
  QDK_LOG_TRACE_ENTERING();

  const auto* ori_bs = this->basis_ori_fp_.get();  // Original/full basis set
  const auto num_atomic_orbitals_ori = ori_bs->num_atomic_orbitals;
  const auto n_val_virt = C_valence_unloc.cols();

  QDK_LOGGER().debug("VVHV::localize_valence_virtual()");

  // Localize valence virtual orbitals using the inner localizer
  Eigen::MatrixXd result = C_valence_unloc;
  if (n_val_virt > 0) {
    QDK_LOGGER().info(
        "*** Localizing Valence Virtual Orbitals (VVHV Sub-scheme) ***");
    result = this->inner_localizer_->localize(C_valence_unloc);
  }

  return result;
}

void VVHVLocalization::proto_hv(const Eigen::MatrixXd& overlap_ori_al,
                                const Eigen::MatrixXd& overlap_mix_al,
                                const std::vector<int>& bf_al_ori,
                                const std::vector<int>& bf_al_min,
                                Eigen::MatrixXd& C_hv_al,
                                int num_atomic_orbitals_ori, int atom_index,
                                int l) {
  QDK_LOG_TRACE_ENTERING();
  const int num_atomic_orbitals_al_ori = static_cast<int>(bf_al_ori.size());
  const int num_atomic_orbitals_al_min = static_cast<int>(bf_al_min.size());
  const int nhv_al = num_atomic_orbitals_al_ori - num_atomic_orbitals_al_min;
  if (nhv_al <= 0) return;
  Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(num_atomic_orbitals_al_ori,
                                               num_atomic_orbitals_al_ori);

  // Allocate local hard-virtual coefficient matrix with correct dimensions
  Eigen::MatrixXd C_psi = Eigen::MatrixXd::Identity(num_atomic_orbitals_al_ori,
                                                    num_atomic_orbitals_al_ori);
  // orthogonalize the C_psi
  temp = C_psi;
  this->orthonormalization(
      num_atomic_orbitals_al_ori, num_atomic_orbitals_al_ori,
      overlap_ori_al.data(), temp.data(), C_psi.data(), 1e-6, 0,
      "initial orthonormalization of original basis on atom " +
          std::to_string(atom_index) + " angular momentum " +
          std::to_string(l));

  if (num_atomic_orbitals_al_min != 0) {
    // Get T_al = overlap_ori_al**-1 * overlap_mix_al, corresponding to xi
    // coefficients in the literature, in the representation of the original
    // basis
    Eigen::MatrixXd T_al = overlap_mix_al;
    {
      Eigen::MatrixXd overlap_ori_copy =
          overlap_ori_al;  // lapack::gelss overwrites input
      std::vector<double> W11(num_atomic_orbitals_al_ori);
      double _rcond = -1.;
      int64_t _tmp_rank;
      lapack::gelss(num_atomic_orbitals_al_ori, num_atomic_orbitals_al_ori,
                    num_atomic_orbitals_al_min, overlap_ori_copy.data(),
                    num_atomic_orbitals_al_ori, T_al.data(),
                    num_atomic_orbitals_al_ori, W11.data(), _rcond, &_tmp_rank);
    }

    // Get overlap of xi, S = T_al^T * overlap_ori_al * T_al = overlap_mix_al^T
    // * overlap_ori_al**-1 * overlap_mix_al = overlap_mix_al^T * T_al
    Eigen::MatrixXd S_xi = Eigen::MatrixXd::Zero(num_atomic_orbitals_al_min,
                                                 num_atomic_orbitals_al_min);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_atomic_orbitals_al_min, num_atomic_orbitals_al_min,
               num_atomic_orbitals_al_ori, 1.0, overlap_mix_al.data(),
               num_atomic_orbitals_al_ori, T_al.data(),
               num_atomic_orbitals_al_ori, 0.0, S_xi.data(),
               num_atomic_orbitals_al_min);

    // Compute S_xi^-1 * overlap_mix^T using lapack::gelss
    // Transformation matrix for proto hard virtual construction
    // C_psi = (I - T S_xi^-1 overlap_mix^T) C_psi,
    {
      // Solve S_xi * X = overlap_mix^T * C_psi for X, storing result in RHS
      Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(num_atomic_orbitals_al_min,
                                                  num_atomic_orbitals_al_ori);
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 num_atomic_orbitals_al_min, num_atomic_orbitals_al_ori,
                 num_atomic_orbitals_al_ori, 1.0, overlap_mix_al.data(),
                 num_atomic_orbitals_al_ori, C_psi.data(),
                 num_atomic_orbitals_al_ori, 0.0, RHS.data(),
                 num_atomic_orbitals_al_min);
      std::vector<double> W_xi(num_atomic_orbitals_al_min);
      double _rcond = -1.0;
      int64_t _tmp_rank;
      lapack::gelss(num_atomic_orbitals_al_min, num_atomic_orbitals_al_min,
                    num_atomic_orbitals_al_ori, S_xi.data(),
                    num_atomic_orbitals_al_min, RHS.data(),
                    num_atomic_orbitals_al_min, W_xi.data(), _rcond,
                    &_tmp_rank);
      // Compute C_psi - = T * RHS (where RHS now contains S_xi^-1 *
      // overlap_mix^T)
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 num_atomic_orbitals_al_ori, num_atomic_orbitals_al_ori,
                 num_atomic_orbitals_al_min, -1.0, T_al.data(),
                 num_atomic_orbitals_al_ori, RHS.data(),
                 num_atomic_orbitals_al_min, 1.0, C_psi.data(),
                 num_atomic_orbitals_al_ori);
    }
  }

  // Use orthonormalization for the orthogonalization step
  this->orthonormalization(
      num_atomic_orbitals_al_ori, num_atomic_orbitals_al_ori,
      overlap_ori_al.data(), C_psi.data(), temp.data(), 1e-6,
      num_atomic_orbitals_al_min,
      "generating prototype hard virtuals on atom " +
          std::to_string(atom_index) + " angular momentum " + std::to_string(l),
      5.0);

  // Copy from temp (which contains the orthonormalized result) to the right
  // place in C_hv_al
  for (int i = 0; i < num_atomic_orbitals_al_ori; ++i) {
    for (int j = 0; j < nhv_al; ++j) {
      C_hv_al(bf_al_ori[i], j) = temp(i, j);
    }
  }

  if (nhv_al > 1) {
    C_hv_al = this->inner_localizer_->localize(C_hv_al);
  }
}

Eigen::MatrixXd VVHVLocalization::localize_hard_virtuals(
    const Eigen::MatrixXd& C_minimal_unloc) {
  QDK_LOG_TRACE_ENTERING();

  QDK_LOGGER().debug("VVHV::localize_hard_virtuals()");

  const auto* ori_bs = this->basis_ori_fp_.get();  // Original/full basis set
  const auto* min_bs = this->minimal_basis_fp_.get();  // Minimal basis set
  const auto num_atomic_orbitals_ori = ori_bs->num_atomic_orbitals;
  const auto num_atomic_orbitals_min = min_bs->num_atomic_orbitals;
  const auto nhv = num_atomic_orbitals_ori - num_atomic_orbitals_min;
  auto mol_structure = ori_bs->mol;
  const auto num_atoms = mol_structure->n_atoms;
  const auto max_l_ori = ori_bs->max_angular_momentum();
  const auto max_l_min = min_bs->max_angular_momentum();
  if (max_l_min > max_l_ori) {
    throw std::runtime_error(
        "VVHVLocalization currently only supports minimal basis sets with max "
        "angular momentum less than or equal to the original basis set");
  }

  // If no hard virtuals, return empty matrix
  if (nhv == 0) {
    return Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, 0);
  }

  // prepare the mapping of atom -> angular momentum -> basis
  std::vector<std::vector<std::vector<int>>> al_to_bf_ori(num_atoms);
  std::vector<std::vector<std::vector<int>>> al_to_bf_min(num_atoms);
  for (auto& v : al_to_bf_ori) v.resize(max_l_ori + 1);
  // Fix: resize al_to_bf_min to max_l_ori + 1 to avoid out-of-bounds access
  for (auto& v : al_to_bf_min) v.resize(max_l_ori + 1);

  const auto& ori_shells = ori_bs->shells;
  int bf_idx = 0;
  bool pure = ori_bs->pure;
  for (auto& sh : ori_shells) {
    // Add bounds checking for atom_index
    if (sh.atom_index >= num_atoms) {
      throw std::runtime_error("VVHVLocalization: Shell atom_index " +
                               std::to_string(sh.atom_index) +
                               " exceeds number of atoms " +
                               std::to_string(num_atoms));
    }
    int sz = pure ? 2 * sh.angular_momentum + 1
                  : (sh.angular_momentum + 1) * (sh.angular_momentum + 2) / 2;
    for (int i = 0; i < sz; i++, bf_idx++)
      al_to_bf_ori[sh.atom_index][sh.angular_momentum].push_back(bf_idx);
  }

  const auto& min_shells = min_bs->shells;
  bf_idx = 0;
  for (auto& sh : min_shells) {
    // Add bounds checking for atom_index
    if (sh.atom_index >= num_atoms) {
      throw std::runtime_error("VVHVLocalization: Minimal shell atom_index " +
                               std::to_string(sh.atom_index) +
                               " exceeds number of atoms " +
                               std::to_string(num_atoms));
    }
    int sz = pure ? 2 * sh.angular_momentum + 1
                  : (sh.angular_momentum + 1) * (sh.angular_momentum + 2) / 2;
    for (int i = 0; i < sz; i++, bf_idx++)
      al_to_bf_min[sh.atom_index][sh.angular_momentum].push_back(bf_idx);
  }

  Eigen::MatrixXd C_hard_virtuals =
      Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, nhv);

  int idx_hv = 0;  // index for placing hard virtuals

  Eigen::MatrixXd temp =
      Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, num_atomic_orbitals_ori);

  // Loop over atoms to construct hard virtuals
  for (auto atom_a = 0; atom_a < num_atoms; ++atom_a) {
    // First collect all atomic orbitals on atom A for both basis sets
    std::vector<int> bf_list_ori;
    std::vector<int> bf_list_min;
    for (auto l = 0; l <= max_l_ori; ++l) {
      auto& bf_l_ori = al_to_bf_ori[atom_a][l];
      auto& bf_l_min = al_to_bf_min[atom_a][l];
      bf_list_ori.insert(bf_list_ori.end(), bf_l_ori.begin(), bf_l_ori.end());
      bf_list_min.insert(bf_list_min.end(), bf_l_min.begin(), bf_l_min.end());
    }

    const int num_atomic_orbitals_a_ori =
        bf_list_ori.size();  // number of atomic orbitals on
                             // atom A in original basis set
    const int num_atomic_orbitals_a_min =
        bf_list_min.size();  // number of atomic orbitals on
                             // atom A in minimal basis set
    const int nhv_a =
        num_atomic_orbitals_a_ori -
        num_atomic_orbitals_a_min;  // number of hard virtuals on atom A
    if (num_atomic_orbitals_a_min <= 0)
      throw std::runtime_error(
          "VVHVLocalization: Atom " + std::to_string(atom_a) +
          " in the minimal basis set has no atomic orbitals.");
    if (nhv_a == 0) continue;  // no hard virtuals on this atom

    // Add bounds checking for bf_list_ori indices
    for (const auto& bf_idx : bf_list_ori) {
      if (bf_idx >= num_atomic_orbitals_ori) {
        throw std::runtime_error("VVHVLocalization: atomic orbital index " +
                                 std::to_string(bf_idx) +
                                 " exceeds total number of atomic orbitals " +
                                 std::to_string(num_atomic_orbitals_ori) +
                                 " on atom " + std::to_string(atom_a));
      }
    }

    // Loop over angular momenta to construct the proto_hard_virtuals
    Eigen::MatrixXd proto_hv =
        Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, nhv_a);
    int proto_hv_idx = 0;

    for (auto l = 0; l <= max_l_ori; ++l) {
      auto& bf_al_ori = al_to_bf_ori[atom_a][l];
      if (bf_al_ori.size() == 0)
        continue;  // no atomic orbitals with this angular momentum on this atom
                   // in the original basis set
      auto& bf_al_min = al_to_bf_min[atom_a][l];
      const int num_atomic_orbitals_al_ori =
          bf_al_ori.size();  // number of atomic orbitals on
                             // atom A with angular momentum
                             // l in the original basis set
      const int num_atomic_orbitals_al_min =
          bf_al_min.size();  // number of atomic orbitals on
                             // atom A with angular momentum
                             // l in the minimal basis set
      const int nhv_al =
          num_atomic_orbitals_al_ori -
          num_atomic_orbitals_al_min;  // number of hard virtuals on atom A
                                       // with angular momentum l
      if (nhv_al < 0)
        throw std::runtime_error(
            "VVHVLocalization: Number of atomic orbitals on atom " +
            std::to_string(atom_a) + " with angular momentum " +
            std::to_string(l) +
            " in the original basis set is less than that in the minimal "
            "basis "
            "set.");
      if (nhv_al == 0) continue;  // no hard virtuals for this angular momentum

      // Extract overlap_ori and overlap_mix blocks for this atom and angular
      // momentum
      Eigen::MatrixXd overlap_ori_al = this->overlap_ori_(bf_al_ori, bf_al_ori);
      Eigen::MatrixXd overlap_mix_al =
          (num_atomic_orbitals_al_min > 0)
              ? Eigen::MatrixXd(this->overlap_mix_(bf_al_ori, bf_al_min))
              : Eigen::MatrixXd::Zero(num_atomic_orbitals_al_ori, 0);
      Eigen::MatrixXd C_hv_al =
          Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, nhv_al);
      this->proto_hv(overlap_ori_al, overlap_mix_al, bf_al_ori, bf_al_min,
                     C_hv_al, num_atomic_orbitals_ori, atom_a, l);
      // place C_hv_al into proto_hv
      proto_hv.block(0, proto_hv_idx, num_atomic_orbitals_ori, nhv_al) =
          C_hv_al;
      proto_hv_idx += nhv_al;

    }  // Loop over angular momenta to construct the proto_hard_virtuals

    if (proto_hv_idx != nhv_a) {
      throw std::runtime_error(
          "VVHVLocalization: Mismatch in number of proto hard virtuals "
          "constructed for atom " +
          std::to_string(atom_a) + " (expected " + std::to_string(nhv_a) +
          ", got " + std::to_string(proto_hv_idx) + ")");
    }

    // First we need to orthogonalize the original atomic orbitals on atom A
    Eigen::MatrixXd C_normal_a = Eigen::MatrixXd::Zero(
        num_atomic_orbitals_ori, num_atomic_orbitals_a_ori);
    Eigen::MatrixXd C_eta_a = Eigen::MatrixXd::Zero(num_atomic_orbitals_ori,
                                                    num_atomic_orbitals_a_ori);
    for (int i = 0; i < num_atomic_orbitals_a_ori; ++i)
      C_eta_a(bf_list_ori[i], i) = 1.0;  // the identity belong to atom A
    // Orthonormalize C_eta_A
    this->orthonormalization(
        num_atomic_orbitals_ori, num_atomic_orbitals_a_ori,
        this->overlap_ori_.data(), C_eta_a.data(), C_normal_a.data(), 1e-6, 0,
        "initial orthonormalization of original basis on atom " +
            std::to_string(atom_a));

    // Now we want to project out all components of the minimal space from the
    // orbitals on A C_eta_A = (I - C_minimal_unloc * C_minimal_unloc^T *
    // overlap_ori) * C_normal_A
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals_ori, num_atomic_orbitals_a_ori,
               num_atomic_orbitals_ori, 1.0, this->overlap_ori_.data(),
               num_atomic_orbitals_ori, C_normal_a.data(),
               num_atomic_orbitals_ori, 0.0, C_eta_a.data(),
               num_atomic_orbitals_ori);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_atomic_orbitals_min, num_atomic_orbitals_a_ori,
               num_atomic_orbitals_ori, 1.0, C_minimal_unloc.data(),
               num_atomic_orbitals_ori, C_eta_a.data(), num_atomic_orbitals_ori,
               0.0, temp.data(), num_atomic_orbitals_min);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals_ori, num_atomic_orbitals_a_ori,
               num_atomic_orbitals_min, -1.0, C_minimal_unloc.data(),
               num_atomic_orbitals_ori, temp.data(), num_atomic_orbitals_min,
               0.0, C_eta_a.data(), num_atomic_orbitals_ori);
    C_eta_a += C_normal_a;

    // Form normalized hard unmatched hard virtuals on atom A (xi in the paper
    // is unnormalized)
    Eigen::MatrixXd C_hv_a =
        Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, nhv_a);

    this->orthonormalization(
        num_atomic_orbitals_ori, num_atomic_orbitals_a_ori,
        this->overlap_ori_.data(), C_eta_a.data(), C_hv_a.data(), 1e-6,
        num_atomic_orbitals_a_min,
        "projecting out the minimal space from the orbitals on atom " +
            std::to_string(atom_a),
        2.0);

    // Form T = C_hv_A^T * overlap_ori * proto_hv
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(nhv_a, nhv_a);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               nhv_a, num_atomic_orbitals_ori, num_atomic_orbitals_ori, 1.0,
               C_hv_a.data(), num_atomic_orbitals_ori,
               this->overlap_ori_.data(), num_atomic_orbitals_ori, 0.0,
               temp.data(), nhv_a);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               nhv_a, nhv_a, num_atomic_orbitals_ori, 1.0, temp.data(), nhv_a,
               proto_hv.data(), num_atomic_orbitals_ori, 0.0, T.data(), nhv_a);
    // Now to form Z, Z is just orthonormalized T in our case
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(nhv_a, nhv_a);
    Eigen::MatrixXd Iden = Eigen::MatrixXd::Identity(nhv_a, nhv_a);
    this->orthonormalization(nhv_a, nhv_a, Iden.data(), T.data(), Z.data(),
                             1e-6);

    // Finally form the hard virtuals on atom A (gamma in the paper) in the
    // representation of the original basis
    Eigen::MatrixXd C_hv_final =
        Eigen::MatrixXd::Zero(num_atomic_orbitals_ori, nhv_a);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals_ori, nhv_a, nhv_a, 1.0, C_hv_a.data(),
               num_atomic_orbitals_ori, Z.data(), nhv_a, 0.0, C_hv_final.data(),
               num_atomic_orbitals_ori);

    // Place C_hv_final into the right place in this->C_hard_virtuals
    if (idx_hv + nhv_a > nhv) {
      throw std::runtime_error(
          "VVHVLocalization: Hard virtual placement index overflow (idx_hv=" +
          std::to_string(idx_hv) + ", nhv_A=" + std::to_string(nhv_a) +
          ", total_nhv=" + std::to_string(nhv) + ") on atom " +
          std::to_string(atom_a));
    }
    C_hard_virtuals.block(0, idx_hv, num_atomic_orbitals_ori, nhv_a) =
        C_hv_final;

    idx_hv += nhv_a;

  }  // Loop over atoms

  // Calculate the orbital spread of each hard virtual orbital, then do weighted
  // orthogonalization if requested

  bool weighted_orthogonalization =
      settings_.get_or_default<bool>("weighted_orthogonalization", true);
  if (weighted_orthogonalization) {
    Eigen::VectorXd spreads_hv(nhv);
    this->calculate_orbital_spreads(C_hard_virtuals, spreads_hv);
    // Weight each orbital by spread
    for (int orb = 0; orb < nhv; ++orb)
      C_hard_virtuals.col(orb) *= spreads_hv(orb);
  }

  // Now hard virtuals are only orthonormal on each atom, we need to
  // orthogonalize them globally
  this->orthonormalization(num_atomic_orbitals_ori, nhv,
                           this->overlap_ori_.data(), C_hard_virtuals.data(),
                           temp.data(), 1e-6, 0,
                           "VVHVLocalization: Final orthonormalization of hard "
                           "virtuals");  // No expected near-zero eigenvalues

  Eigen::MatrixXd result_hard_virtuals =
      temp.block(0, 0, num_atomic_orbitals_ori, nhv);

  return result_hard_virtuals;
}

void VVHVLocalization::calculate_orbital_spreads(
    const Eigen::MatrixXd& orbitals, Eigen::VectorXd& spreads) const {
  QDK_LOG_TRACE_ENTERING();

  const auto* ori_bs = this->basis_ori_fp_.get();  // Original/full basis set
  const auto num_atomic_orbitals_ori = ori_bs->num_atomic_orbitals;
  const int num_orbitals = orbitals.cols();

  // Check if spreads vector has the correct size
  if (spreads.size() != num_orbitals) {
    throw std::runtime_error(
        "VVHVLocalization::calculate_orbital_spreads: spreads vector size (" +
        std::to_string(spreads.size()) +
        ") does not match number of orbitals (" + std::to_string(num_orbitals) +
        ")");
  }

  // Use pre-computed integrals if available, otherwise compute them locally
  const qcs::RowMajorMatrix* dipole;
  const qcs::RowMajorMatrix* quadrupole;
  std::unique_ptr<qcs::RowMajorMatrix> local_dipole;
  std::unique_ptr<qcs::RowMajorMatrix> local_quadrupole;

  if (dipole_integrals_ && quadrupole_integrals_) {
    // Use pre-computed integrals
    dipole = dipole_integrals_.get();
    quadrupole = quadrupole_integrals_.get();
    QDK_LOGGER().debug(
        "VVHVLocalization: Using pre-computed dipole and quadrupole integrals "
        "for orbital spreads");
  } else {
    // Compute integrals locally
    local_dipole = std::make_unique<qcs::RowMajorMatrix>(
        3 * num_atomic_orbitals_ori, num_atomic_orbitals_ori);
    local_quadrupole = std::make_unique<qcs::RowMajorMatrix>(
        6 * num_atomic_orbitals_ori, num_atomic_orbitals_ori);

    // Create one-body integral object to compute dipole and quadrupole
    // integrals
    qcs::OneBodyIntegral ori_bs_1ee(ori_bs, ori_bs->mol.get(),
                                    qcs::mpi_default_input());
    ori_bs_1ee.dipole_integral(local_dipole->data());
    ori_bs_1ee.quadrupole_integral(local_quadrupole->data());

    dipole = local_dipole.get();
    quadrupole = local_quadrupole.get();
    QDK_LOGGER().debug(
        "VVHVLocalization: Computed dipole and quadrupole integrals locally "
        "for orbital spreads");
  }

  // Calculate spread for each orbital
  for (int orb = 0; orb < num_orbitals; ++orb) {
    // Extract orbital coefficients for this MO
    Eigen::VectorXd c = orbitals.col(orb);

    // Calculate <r> = <psi|r|psi> for each component using dipole integrals
    std::array<double, 3> r_mean = {0.0, 0.0, 0.0};
    for (int comp = 0; comp < 3; ++comp) {
      Eigen::Map<const qcs::RowMajorMatrix> dipole_comp(
          dipole->data() +
              comp * num_atomic_orbitals_ori * num_atomic_orbitals_ori,
          num_atomic_orbitals_ori, num_atomic_orbitals_ori);
      r_mean[comp] = c.transpose() * dipole_comp * c;
    }

    // Calculate <r²> = <psi|r²|psi>
    // r² = x² + y² + z² = quadrupole(0,0) + quadrupole(3,3) + quadrupole(5,5)
    // where quadrupole components are: xx(0), xy(1), xz(2), yy(3), yz(4), zz(5)
    double r2_mean = 0.0;

    // x² component (xx)
    Eigen::Map<const qcs::RowMajorMatrix> quad_xx(
        quadrupole->data() +
            0 * num_atomic_orbitals_ori * num_atomic_orbitals_ori,
        num_atomic_orbitals_ori, num_atomic_orbitals_ori);
    r2_mean += c.transpose() * quad_xx * c;

    // y² component (yy)
    Eigen::Map<const qcs::RowMajorMatrix> quad_yy(
        quadrupole->data() +
            3 * num_atomic_orbitals_ori * num_atomic_orbitals_ori,
        num_atomic_orbitals_ori, num_atomic_orbitals_ori);
    r2_mean += c.transpose() * quad_yy * c;

    // z² component (zz)
    Eigen::Map<const qcs::RowMajorMatrix> quad_zz(
        quadrupole->data() +
            5 * num_atomic_orbitals_ori * num_atomic_orbitals_ori,
        num_atomic_orbitals_ori, num_atomic_orbitals_ori);
    r2_mean += c.transpose() * quad_zz * c;

    // Calculate |<r>|² = <x>² + <y>² + <z>²
    double r_mean_squared =
        r_mean[0] * r_mean[0] + r_mean[1] * r_mean[1] + r_mean[2] * r_mean[2];

    // Calculate spread = <r²> - |<r>|²
    spreads(orb) = std::max(0.0, r2_mean - r_mean_squared);
  }
}

void VVHVLocalization::orthonormalization(int num_atomic_orbitals,
                                          int num_orbitals,
                                          const double* overlap_inp, double* C,
                                          double* C_out, double ortho_threshold,
                                          unsigned int expected_near_zero,
                                          const std::string& error_label,
                                          double separation_ratio) {
  QDK_LOG_TRACE_ENTERING();

  // Compute overlap matrix S = C^T * overlap_inp * C
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(num_orbitals, num_orbitals);
  {
    Eigen::MatrixXd temp =
        Eigen::MatrixXd::Zero(num_orbitals, num_atomic_orbitals);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               num_orbitals, num_atomic_orbitals, num_atomic_orbitals, 1.0, C,
               num_atomic_orbitals, overlap_inp, num_atomic_orbitals, 0.0,
               temp.data(), num_orbitals);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_orbitals, num_orbitals, num_atomic_orbitals, 1.0,
               temp.data(), num_orbitals, C, num_atomic_orbitals, 0.0, S.data(),
               num_orbitals);
  }

  // Diagonalize S = U * Lambda * U^T
  Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(num_orbitals);
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, num_orbitals, S.data(),
               num_orbitals,
               eigenvalues.data());  // S now contains eigenvectors U

  if (expected_near_zero > 0) {
    // Check eigenvalue structure if selection needed
    VVHVLocalization::check_eigenvalue_structure(
        eigenvalues.data(), expected_near_zero, num_orbitals, error_label,
        separation_ratio);
    // Compute W = U * Lambda^(-1/2) (store in S)
    for (int i = expected_near_zero; i < num_orbitals; ++i) {
      double lambda_inv_sqrt = (eigenvalues[i] > ortho_threshold)
                                   ? 1.0 / std::sqrt(eigenvalues[i])
                                   : 0.0;
      double* temp_col_i = S.data() + i * num_orbitals;
      for (int j = 0; j < num_orbitals; ++j) temp_col_i[j] *= lambda_inv_sqrt;
    }
    // Compute C_out = C * W
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals, num_orbitals - expected_near_zero,
               num_orbitals, 1.0, C, num_atomic_orbitals,
               S.data() + expected_near_zero * num_orbitals, num_orbitals, 0.0,
               C_out, num_atomic_orbitals);
  } else {
    // If no selection needed,
    // compute C_out = C *  U * Lambda^(-1/2) * U^T for symmetric
    // orthonormalization

    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(num_orbitals, num_orbitals);
    // Scale eigenvectors by Lambda^(-1/2) in-place: U_scaled = U *
    // Lambda^(-1/2)
    for (int i = expected_near_zero; i < num_orbitals; ++i) {
      double lambda_inv_sqrt = (eigenvalues[i] > ortho_threshold)
                                   ? 1.0 / std::sqrt(eigenvalues[i])
                                   : 0.0;
      temp.col(i) = S.col(i) * lambda_inv_sqrt;
    }

    // Compute orthonorm_transform = U_scaled * U^T using temporary storage
    Eigen::MatrixXd orthonorm_transform =
        Eigen::MatrixXd::Zero(num_orbitals, num_orbitals);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               num_orbitals, num_orbitals, num_orbitals, 1.0, temp.data(),
               num_orbitals, S.data(), num_orbitals, 0.0,
               orthonorm_transform.data(), num_orbitals);
    // Compute C_out = C * orthonorm_transform
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals, num_orbitals, num_orbitals, 1.0, C,
               num_atomic_orbitals, orthonorm_transform.data(), num_orbitals,
               0.0, C_out, num_atomic_orbitals);
  }
}

void VVHVLocalization::check_eigenvalue_structure(
    const double* eigenvalues, int expected_near_zero, int total_eigenvalues,
    const std::string& error_label, double separation_ratio) const {
  QDK_LOG_TRACE_ENTERING();

  if (expected_near_zero < 0 || expected_near_zero > total_eigenvalues) {
    throw std::runtime_error("VVHVLocalization (" + error_label +
                             "): Invalid expected_near_zero value: " +
                             std::to_string(expected_near_zero));
  }

  // Check the ratio of M+1 and M eigenvalues instead of scanning all
  // eigenvalues M is expected_near_zero-1 (0-indexed), M+1 is
  // expected_near_zero
  double eigenvalue_M =
      eigenvalues[expected_near_zero - 1];  // M-th eigenvalue (0-indexed)
  double eigenvalue_M_plus_1 =
      eigenvalues[expected_near_zero];  // (M+1)-th eigenvalue (0-indexed)

  // Check if the ratio is sufficient for separation
  std::ostringstream oss;
  oss << "VVHVLocalization: Unexpected number of near-zero eigenvalues in "
      << error_label << ".\n";
  oss << "Expected " << expected_near_zero
      << " eigenvalues close to zero, but got eigenvalues around the expected "
         "zero index:\n";
  oss << "Separation ratio: " << eigenvalue_M_plus_1 / eigenvalue_M
      << " (required: " << separation_ratio
      << "). The following are the eigenvalues around separation. (Index "
         "starts at 0)\n";
  int start = std::max(0, expected_near_zero - 5);
  int end = std::min(total_eigenvalues, expected_near_zero + 5);
  for (int i = start; i < end; ++i)
    oss << std::setw(4) << i << ": " << std::scientific << std::setprecision(6)
        << eigenvalues[i] << ";   ";
  oss << "\n";
  if (abs(eigenvalue_M_plus_1 / eigenvalue_M) < separation_ratio)
    QDK_LOGGER().warn(oss.str());
}

std::shared_ptr<data::Wavefunction> VVHVLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();
  // Get electron counts from settings
  auto [n_alpha_electrons, n_beta_electrons] =
      wavefunction->get_total_num_electrons();

  // Check if electron counts have been set
  if (n_alpha_electrons < 0 || n_beta_electrons < 0) {
    throw std::invalid_argument(
        "n_alpha_electrons and n_beta_electrons must be set in localizer "
        "settings before calling _run_impl()");
  }

  // Get number of molecular orbitals first
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Validate that indices are sorted
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (!std::is_sorted(loc_indices_b.begin(), loc_indices_b.end())) {
    throw std::invalid_argument("loc_indices_b must be sorted");
  }

  // Check that indices only cover virtual orbitals
  auto check_virtual_indices = [num_molecular_orbitals](
                                   const std::vector<size_t>& indices,
                                   size_t n_electrons) {
    const size_t num_virtual_orbitals = num_molecular_orbitals - n_electrons;
    if (indices.size() != num_virtual_orbitals) {
      return false;
    }
    std::vector<bool> covered(num_virtual_orbitals, false);
    for (size_t idx : indices) {
      if (idx < n_electrons || idx >= num_molecular_orbitals) {
        return false;  // Index is not in virtual range
      }
      covered[idx - n_electrons] = true;
    }
    for (bool c : covered) {
      if (!c) return false;
    }
    return true;
  };

  if (!check_virtual_indices(loc_indices_a, n_alpha_electrons)) {
    throw std::invalid_argument(
        "VVHVLocalizer requires all alpha virtual orbital indices to be "
        "covered. loc_indices_a must contain all alpha virtual orbital indices "
        "from n_alpha_electrons to num_molecular_orbitals-1.");
  }

  if (!orbitals->is_restricted() &&
      !check_virtual_indices(loc_indices_b, n_beta_electrons)) {
    throw std::invalid_argument(
        "VVHVLocalizer requires all beta virtual orbital indices to be "
        "covered. loc_indices_b must contain all beta virtual orbital indices "
        "from n_beta_electrons to num_molecular_orbitals-1.");
  }

  if (orbitals->is_restricted() && !(loc_indices_a == loc_indices_b)) {
    throw std::invalid_argument(
        "For restricted orbitals, loc_indices_a and loc_indices_b must be "
        "identical");
  }
  const auto& basis_set = orbitals->get_basis_set();
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  if (num_atomic_orbitals != num_molecular_orbitals) {
    throw std::runtime_error(
        "Current VVHVLocalizer can not handle basis set linear dependence");
  }

  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  // Get minimal basis name from settings or use default
  std::string minimal_basis_name =
      _settings->get_or_default<std::string>("minimal_basis", "sto-3g");

  auto [coeffs_alpha, coeffs_beta] = orbitals->get_coefficients();
  auto ao_overlap = orbitals->get_overlap_matrix();

  // TODO (DBWY): Adding configurable inner localizer
  // Work Item: 41816
  // Create reusable Pipek-Mezey localizer for inner localization
  const size_t num_atoms = basis_set->get_structure()->get_num_atoms();
  std::vector<int> bf_to_atom(num_atomic_orbitals);
  for (size_t i = 0; i < num_atomic_orbitals; ++i) {
    bf_to_atom[i] = basis_set->get_atom_index_for_atomic_orbital(i);
  }
  auto inner_localizer = std::make_shared<PipekMezeyLocalization>(
      *static_cast<const IterativeOrbitalLocalizationSettings*>(
          _settings.get()),
      ao_overlap, num_atoms, bf_to_atom);

  if (orbitals->is_restricted()) {
    // Restricted case: RHF or ROHF - only handle virtual orbitals
    const size_t num_occupied_orbitals =
        std::max(n_alpha_electrons, n_beta_electrons);
    const size_t num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;
    VVHVLocalization localizer(
        *static_cast<const IterativeOrbitalLocalizationSettings*>(
            _settings.get()),
        basis_set, ao_overlap, minimal_basis_name, inner_localizer);

    // Extract occupied orbitals and pass to localize, get back localized
    // virtual orbitals only
    Eigen::MatrixXd occupied_orbitals =
        coeffs_alpha.block(0, 0, num_atomic_orbitals, num_occupied_orbitals);
    Eigen::MatrixXd C_virt_loc = localizer.localize(occupied_orbitals);

    // Reconstruct full coefficient matrix with original occupied orbitals
    Eigen::MatrixXd C_lmo = coeffs_alpha;
    C_lmo.block(0, num_occupied_orbitals, num_atomic_orbitals,
                num_virtual_orbitals) = C_virt_loc;

    auto new_orbitals = std::make_shared<data::Orbitals>(
        C_lmo,
        std::nullopt,   // no energies for localized orbitals
        ao_overlap,     // Atomic Orbital overlap
        basis_set,      // basis set
        std::nullopt);  // no active space indices
    return detail::new_wavefunction(wavefunction, new_orbitals);
  } else {
    // Unrestricted case: UHF - only handle virtual orbitals
    const size_t num_alpha_virtual_orbitals =
        num_molecular_orbitals - n_alpha_electrons;
    const size_t num_beta_virtual_orbitals =
        num_molecular_orbitals - n_beta_electrons;

    // Create a single VVHVLocalization instance and reuse for both channels
    VVHVLocalization localizer(
        *static_cast<const IterativeOrbitalLocalizationSettings*>(
            _settings.get()),
        basis_set, ao_overlap, minimal_basis_name, inner_localizer);

    // Alpha channel
    Eigen::MatrixXd occupied_alpha =
        coeffs_alpha.block(0, 0, num_atomic_orbitals, n_alpha_electrons);
    Eigen::MatrixXd C_virt_alpha_loc = localizer.localize(occupied_alpha);

    // Beta channel - reuse the same localizer instance
    Eigen::MatrixXd occupied_beta =
        coeffs_beta.block(0, 0, num_atomic_orbitals, n_beta_electrons);
    Eigen::MatrixXd C_virt_beta_loc = localizer.localize(occupied_beta);

    // Reconstruct full coefficient matrices with original occupied orbitals
    Eigen::MatrixXd C_alpha = coeffs_alpha;
    Eigen::MatrixXd C_beta = coeffs_beta;
    C_alpha.block(0, n_alpha_electrons, num_atomic_orbitals,
                  num_alpha_virtual_orbitals) = C_virt_alpha_loc;
    C_beta.block(0, n_beta_electrons, num_atomic_orbitals,
                 num_beta_virtual_orbitals) = C_virt_beta_loc;

    auto new_orbitals = std::make_shared<data::Orbitals>(
        C_alpha, C_beta, std::nullopt,
        std::nullopt,   // no energies for localized orbitals
        ao_overlap,     // Atomic Orbital overlap
        basis_set,      // basis set
        std::nullopt);  // no active space indices
    return detail::new_wavefunction(wavefunction, new_orbitals);
  }
}

}  // namespace qdk::chemistry::algorithms::microsoft
