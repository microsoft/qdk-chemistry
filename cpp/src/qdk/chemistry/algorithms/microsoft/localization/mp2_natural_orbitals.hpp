// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/localization.hpp>

namespace qdk::chemistry::algorithms::microsoft {

#define QDK_MP2_NATURAL_ORBITAL_LOCALIZER_DEPRECATION_MESSAGE              \
  "MP2NaturalOrbitalLocalizer is deprecated; use NaturalOrbitalLocalizer " \
  "(qdk_natural_orbitals) with a wavefunction that already contains the "  \
  "MP2 active-space 1-RDM."

/**
 * @class MP2NaturalOrbitalLocalizer
 * @brief Localizer implementation that transforms orbitals to MP2 natural
 * orbitals
 *
 * @deprecated Use NaturalOrbitalLocalizer (qdk_natural_orbitals) with a
 * wavefunction that already contains the active-space MP2 one-particle reduced
 * density matrix (1-RDM).
 *
 * This class provides a concrete implementation of the Localizer interface
 * that transforms canonical molecular orbitals into natural orbitals derived
 * from second-order Møller-Plesset perturbation theory (MP2). Natural orbitals
 * are eigenfunctions of the first-order reduced density matrix.
 *
 * MP2 natural orbitals often provide a more compact representation of the
 * electronic wavefunction, which can improve computational efficiency in
 * correlation methods. The transformation diagonalizes the MP2 density matrix
 * to produce orbitals with well-defined occupation numbers.
 *
 * Restrictions:
 * - Only supports restricted orbitals
 * - Only supports closed-shell systems (equal alpha and beta electrons,
 *    loc_indices_a and loc_indices_b must be identical)
 *
 * Algorithm steps:
 * 1. Construct the molecular Hamiltonian from the input orbitals
 * 2. Compute the MP2 natural orbitals using the Hamiltonian integrals
 * 3. Transform the original orbital coefficients to the natural orbital basis
 * 4. Construct the transformed orbitals with updated coefficients
 * 5. Return a new Aufbau determinant wavefunction containing the MP2 natural
 *    orbitals.
 */
class [[deprecated(QDK_MP2_NATURAL_ORBITAL_LOCALIZER_DEPRECATION_MESSAGE)]]
MP2NaturalOrbitalLocalizer : public Localizer {
 public:
  /**
   * @brief Default constructor
   */
  MP2NaturalOrbitalLocalizer() = default;

  /**
   * @brief Virtual destructor
   */
  ~MP2NaturalOrbitalLocalizer() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_mp2_natural_orbitals"; };

  std::optional<std::string> deprecation_message() const final {
    return QDK_MP2_NATURAL_ORBITAL_LOCALIZER_DEPRECATION_MESSAGE;
  }

 protected:
  /**
   * @brief Transform canonical orbitals into MP2 natural orbitals
   *
   * This method performs the transformation from canonical molecular orbitals
   * to MP2 natural orbitals. It constructs the molecular Hamiltonian, computes
   * the MP2 density matrix, diagonalizes it to obtain natural orbitals, and
   * transforms the original orbitals to the natural orbital basis.
   *
   * @param wavefunction The input canonical molecular wavefunction to transform
   * @param loc_indices_a Indices of alpha orbitals to transform (must be
   * sorted)
   * @param loc_indices_b Indices of beta orbitals to transform (must be sorted)
   * @return The MP2 natural orbitals with updated coefficients and occupations
   *
   * @throws std::invalid_argument if input orbitals are not canonical (missing
   * energies)
   * @throws std::invalid_argument if no occupied orbitals are present
   * @throws std::invalid_argument if loc_indices_a or loc_indices_b are not
   * sorted
   * @throws std::invalid_argument if any orbital index is >=
   * num_molecular_orbitals
   * @throws std::invalid_argument If orbitals are unrestricted
   * @throws std::invalid_argument If the system is not closed-shell (alpha !=
   * beta or loc_indices_a != loc_indices_b)
   * @throws std::invalid_argument if both occupied and virtual orbitals are not
   * present in selection
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

#undef QDK_MP2_NATURAL_ORBITAL_LOCALIZER_DEPRECATION_MESSAGE

}  // namespace qdk::chemistry::algorithms::microsoft
