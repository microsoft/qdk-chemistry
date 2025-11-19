// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/types.h>

#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

#include "iterative_localizer_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

/**
 * @class VVHVLocalizerSettings
 * @brief Settings class for VVHV localization
 *
 * Contains configuration parameters specific to Valence Virtual - Hard Virtual
 * (VV-HV) localization, including all parameters from the base iterative
 * localization settings plus any VVHV-specific options.
 */
class VVHVLocalizerSettings : public IterativeOrbitalLocalizationSettings {
 public:
  VVHVLocalizerSettings() {
    set_default("minimal_basis", std::string("sto-3g"));
    set_default("weighted_orthogonalization", true);
  }
};

/**
 * @brief Valence Virtual - Hard Virtual (VV-HV) implementation of the Localizer
 * interface.
 *
 * This class provides a concrete implementation of the Localizer interface
 * using the VV-HV localization algorithm.
 *
 * The VV-HV algorithm partitions virtual orbitals into valence virtuals (VVs)
 * and hard virtuals (HVs) based on projection onto a minimal basis, then
 * localizes each space separately using Pipek-Mezey localization.
 *
 * Implementation based on:
 *   Subotnik et al. JCP 123, 114108 (2005)
 *   Wang et al. JCTC 21, 1163 (2025)
 *
 * @note This localizer requires all orbital indices to be covered in the
 * localize call.
 */
class VVHVLocalizer : public qdk::chemistry::algorithms::Localizer {
 public:
  VVHVLocalizer() { _settings = std::make_unique<VVHVLocalizerSettings>(); };

  /**
   * @brief Localize orbitals using the VV-HV algorithm.
   *
   * This method performs VV-HV localization on the input orbitals.
   * It requires that all orbital indices are covered (loc_indices_a and
   * loc_indices_b must span all orbitals).
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param orbitals The orbitals to localize
   * @param loc_indices_a Indices of alpha orbitals (must cover all orbitals,
   * must be sorted)
   * @param loc_indices_b Indices of beta orbitals (must cover all orbitals,
   * must be sorted)
   * \endcond
   *
   * @return The localized molecular orbitals
   * @throws std::invalid_argument If loc_indices_a or loc_indices_b are not
   * sorted
   * @throws std::invalid_argument If not all orbital indices are covered
   * @throws std::invalid_argument If restricted orbitals have different
   * alpha/beta indices
   * @throws std::runtime_error If basis set has linear dependence issues
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;

  std::string name() const override { return "qdk_vvhv"; }
};

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
      std::shared_ptr<IterativeOrbitalLocalizationScheme> inner_localizer);

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
   * (num_basis_funcs x num_occupied_orbitals)
   * @return Localized virtual orbital coefficient matrix (num_basis_funcs x
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
      dipole_integrals_;  // 3*num_basis_funcs x num_basis_funcs matrix
  std::unique_ptr<qcs::RowMajorMatrix>
      quadrupole_integrals_;  // 6*num_basis_funcs x num_basis_funcs matrix

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
   * @param num_basis_funcs Number of basis functions (rows in C and
   * overlap_inp)
   * @param num_orbitals Number of orbitals (columns in C)
   * @param overlap_inp Overlap matrix (num_basis_funcs x num_basis_funcs) in
   * the representation which input orbitals C are given
   * @param C Input orbital coefficient matrix (num_basis_funcs x num_orbitals)
   * @param C_out Output orthonormalized orbital coefficient matrix
   * (num_basis_funcs x num_orbitals_out, num_orbitals_out = num_orbitals -
   * expected_near_zero)
   * @param ortho_threshold Threshold for orthonormalization (eigenvalue cutoff)
   * @param expected_near_zero Expected number of near-zero eigenvalues to skip
   * (skip check if 0)
   * @param error_label Label for error messages
   * @param separation_ratio Required ratio of eigenvalue[M+1]/eigenvalue[M] for
   * sufficient separation
   */
  void orthonormalization(int num_basis_funcs, int num_orbitals,
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
   * @param orbitals Matrix of orbital coefficients (num_basis_funcs x
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
   * block (size num_basis_funcs_al_ori x num_basis_funcs_al_ori)
   * @param overlap_mix_al Mixed overlap block between original and minimal
   * basis (size num_basis_funcs_al_ori x num_basis_funcs_al_min)
   * @param bf_al_ori Index list (global AO indices) for this atom+l in the
   * original basis
   * @param bf_al_min Index list for this atom+l in the minimal basis
   * @param C_hv_al (Output) Matrix (num_basis_funcs_ori x nhv_al) to receive
   * hard virtual coefficients (global AO representation)
   * @param num_basis_funcs_ori Total number of original basis functions (global
   * row dimension for C_hv_al)
   * @param atom_index Atom index (for logging / diagnostics)
   * @param l Angular momentum quantum number (for logging / diagnostics)
   */
  void proto_hv(const Eigen::MatrixXd& overlap_ori_al,
                const Eigen::MatrixXd& overlap_mix_al,
                const std::vector<int>& bf_al_ori,
                const std::vector<int>& bf_al_min, Eigen::MatrixXd& C_hv_al,
                int num_basis_funcs_ori, int atom_index, int l);

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

}  // namespace qdk::chemistry::algorithms::microsoft
