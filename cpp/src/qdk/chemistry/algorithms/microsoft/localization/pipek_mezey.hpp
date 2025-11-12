// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

#include "iterative_localizer_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Default Pipek-Mezey implementation of the Localizer interface.
 *
 * This class provides a concrete implementation of the Localizer interface
 * using the Pipek-Mezey localization algorithm as the default method.
 * It inherits from the base `qdk::chemistry::algorithms::Localizer` class and
 * implements the `localize` method to perform orbital localization.
 *
 * The Pipek-Mezey algorithm maximizes the sum of squares of atomic orbital
 * populations on atoms, resulting in orbitals that are more localized to
 * individual atoms or bonds. This implementation separately localizes occupied
 * and virtual orbitals to maintain the occupied-virtual separation.
 *
 * @see PipekMezeyLocalization for the underlying algorithm implementation
 */
class PipekMezeyLocalizer : public qdk::chemistry::algorithms::Localizer {
 public:
  PipekMezeyLocalizer() {
    _settings = std::make_unique<IterativeOrbitalLocalizationSettings>();
  };

  virtual ~PipekMezeyLocalizer() override = default;
  virtual std::string name() const final { return "qdk_pipek_mezey"; };

  /**
   * @brief Implementation of orbital localization using Pipek-Mezey algorithm.
   *
   * @param wavefunction Molecular wavefunction to be localized
   * @param loc_indices_a Indices of alpha orbitals to localize (must be sorted)
   * @param loc_indices_b Indices of beta orbitals to localize (must be sorted)
   * @return Localized wavefunction
   *
   * @throws std::invalid_argument if loc_indices_a or loc_indices_b are not
   * sorted
   * @throws std::invalid_argument if any orbital index is >=
   * num_molecular_orbitals
   * @throws std::invalid_argument if restricted orbitals have different
   * alpha/beta indices
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

/**
 * @brief Pipek-Mezey localization scheme implementation.
 *
 * Implementation according to:
 *   J. W. Boughton, P. Pulay; J. Comp. Chem. 14 (1993), 736 - 740.
 *   https://doi.org/10.1002/jcc.540140615
 *
 * This class implements the Pipek-Mezey localization algorithm for molecular
 * orbitals. It inherits from the IterativeOrbitalLocalizationScheme base class
 * and provides a concrete implementation of the localize method.
 */
class PipekMezeyLocalization : public IterativeOrbitalLocalizationScheme {
 public:
  /**
   * @brief Construct a new Pipek-Mezey localization object.
   *
   * Initializes the Pipek-Mezey localization scheme with the necessary
   * parameters for orbital localization.
   *
   * @param settings Configuration settings for the iterative localization
   *                 algorithm, including convergence tolerances, maximum
   *                 iterations, and rotation thresholds
   * @param overlap_matrix Atomic orbital overlap matrix (S_μν) used in the
   *                       localization procedure to ensure orthonormality
   *                       of the localized orbitals
   * @param num_atoms Total number of atoms in the molecular system
   * @param bf_to_atom_map Mapping from basis function index to atom index,
   *                       required for computing atomic populations. Must have
   *                       size equal to the number of basis functions, with
   *                       each element indicating which atom the corresponding
   *                       basis function belongs to
   *
   * @note The overlap_matrix must be symmetric and positive definite.
   * @note The bf_to_atom_map indices must be in the range [0, num_atoms).
   *
   * @see IterativeOrbitalLocalizationSettings for available configuration
   *      options
   */
  PipekMezeyLocalization(IterativeOrbitalLocalizationSettings settings,
                         const Eigen::MatrixXd& overlap_matrix,
                         size_t num_atoms, std::vector<int> bf_to_atom_map);

  ~PipekMezeyLocalization() override = default;

  /**
   * @brief Localize the orbitals using the Pipek-Mezey algorithm.
   *
   * This method performs the localization of molecular orbitals according to
   * the Pipek-Mezey scheme.
   *
   * @param initial_orbitals Matrix of initial molecular orbital coefficients to
   * be localized
   * @return Matrix of localized molecular orbital coefficients
   */
  Eigen::MatrixXd localize(const Eigen::MatrixXd& initial_orbitals) override;

 private:
  Eigen::MatrixXd overlap_matrix_;   // Overlap matrix for the orbitals
  std::vector<int> bf_to_atom_map_;  // Basis function to atom mapping
  size_t num_atoms_;                 // Number of atoms in the system
};
}  // namespace qdk::chemistry::algorithms::microsoft
