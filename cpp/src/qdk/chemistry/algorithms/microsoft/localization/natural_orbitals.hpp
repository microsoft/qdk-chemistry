// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/localization.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class NaturalOrbitalLocalizer
 * @brief Localizer implementation that transforms orbitals to natural orbitals
 * by diagonalizing the spin-traced one-particle reduced density matrix (1-RDM)
 *
 * This class provides a concrete implementation of the Localizer interface
 * that transforms molecular orbitals into natural orbitals. Natural orbitals
 * are eigenfunctions of the one-particle reduced density matrix (1-RDM), and
 * their eigenvalues are the occupation numbers.
 *
 * Restrictions:
 * - Requires loc_indices_a == loc_indices_b (natural orbitals are a single set)
 * - Requires a spin-traced 1-RDM to be available in the wavefunction
 * - The wavefunction must have an active space defined
 *
 * For an unrestricted Slater-determinant input, the spin-traced 1-RDM is
 * expressed in the alpha MO basis, so the rotation is applied to the alpha
 * coefficients. The output is always a restricted set of natural orbitals.
 *
 * Algorithm steps:
 * 1. Extract the spin-traced 1-RDM from the wavefunction
 * 2. Diagonalize the 1-RDM to obtain natural orbital coefficients and
 *    occupation numbers
 * 3. Transform the active orbital coefficients to the natural orbital basis
 * 4. Construct a new wavefunction with the transformed orbitals
 */
class NaturalOrbitalLocalizer : public Localizer {
 public:
  /**
   * @brief Default constructor
   */
  NaturalOrbitalLocalizer() = default;

  /**
   * @brief Virtual destructor
   */
  ~NaturalOrbitalLocalizer() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_natural_orbitals"; };

 protected:
  /**
   * @brief Transform orbitals into natural orbitals via 1-RDM diagonalization
   *
   * This method diagonalizes the spin-traced one-particle reduced density
   * matrix to obtain natural orbitals. The eigenvectors define the
   * transformation from the current orbital basis to the natural orbital basis,
   * and the eigenvalues are the occupation numbers.
   *
   * @param wavefunction The input wavefunction with a 1-RDM
   * @param loc_indices_a Indices of alpha orbitals to transform (must be
   * sorted)
   * @param loc_indices_b Indices of beta orbitals to transform (must be sorted)
   * @return A new wavefunction with orbitals transformed to the natural orbital
   * basis
   *
   * @throws std::invalid_argument if loc_indices_a != loc_indices_b
   * @throws std::invalid_argument if loc_indices_a or loc_indices_b are not
   * sorted
   * @throws std::invalid_argument if any orbital index is >=
   * num_molecular_orbitals
   * @throws std::invalid_argument if the wavefunction has no spin-traced 1-RDM
   * @throws std::invalid_argument if the wavefunction has no active space
   * @throws std::invalid_argument if the orbitals are unrestricted and the
   * wavefunction container is not a SlaterDeterminantContainer
   * @throws std::invalid_argument if 1-RDM dimensions do not match the number
   * of selected orbitals
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
