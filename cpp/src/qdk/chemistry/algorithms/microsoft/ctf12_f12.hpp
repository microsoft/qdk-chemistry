// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <libint2.hpp>
#include <memory>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::scf {
class BasisSet;
}

namespace qdk::chemistry::algorithms::microsoft::ctf12 {

/**
 * @brief Build a spin-restricted orbital index set over @p num_modes orbitals.
 *
 * @param num_modes Total number of molecular orbitals (the index universe).
 * @param indices Selected orbital indices, strictly increasing.
 * @return Index set carrying @p indices in both spin channels.
 */
std::shared_ptr<const data::SymmetryBlockedIndexSet> restricted_index_set(
    std::size_t num_modes, const std::vector<std::size_t>& indices);

/**
 * @brief Inputs describing the F12-HF reference for a closed-shell system.
 *
 * All molecular-orbital data are expressed in the orbital basis set (OBS).
 * Occupied orbitals are columns @c [0, n_occupied) of @ref mo_coefficients,
 * ordered by ascending orbital energy; the geminal-generating (valence)
 * occupied orbitals are columns @c [n_core, n_occupied) under the frozen-core
 * (a) convention.
 */
struct F12HartreeFockInput {
  std::shared_ptr<scf::BasisSet>
      scf_basis_set;                 ///< Internal OBS metadata for SCF.
  ::libint2::BasisSet obs;           ///< Orbital basis set.
  Eigen::MatrixXd mo_coefficients;   ///< MO coefficients, @c [n_ao, n_mo].
  Eigen::VectorXd orbital_energies;  ///< Canonical orbital energies, @c [n_mo].
  std::size_t n_occupied = 0;        ///< Number of doubly occupied orbitals.
  std::size_t n_core = 0;            ///< Frozen core orbitals (formulation a).
  ::libint2::BasisSet cabs_ri_basis;  ///< OBS-union-aux RI basis for CABS.
  Eigen::MatrixXd cabs_coefficients;  ///< CABS coefficients, @c [n_ri, n_cabs].
  std::vector<std::pair<double, std::array<double, 3>>>
      nuclei;          ///< Nuclear charges and positions (atomic units).
  double gamma = 1.5;  ///< Slater geminal exponent.
};

/**
 * @brief Build a closed-shell F12-HF input from a reference wavefunction.
 *
 * Extracts the orbital basis, molecular-orbital coefficients, orbital energies
 * and nuclear data from a restricted single-determinant reference and builds
 * the complementary auxiliary basis set (CABS). The CABS is built from
 * @p cabs_auxiliary_basis when one is supplied; otherwise from the named
 * @p cabs_basis, or from @c "<orbital-basis>-optri" when @p cabs_basis is
 * empty.
 *
 * @param reference The restricted reference wavefunction.
 * @param gamma The Slater geminal exponent (atomic units).
 * @param cabs_basis The named OptRI/CABS auxiliary basis (empty to derive one).
 * @param frozen_core The number of frozen core orbitals (formulation a).
 * @param cabs_auxiliary_basis Explicit CABS-generating auxiliary basis, or null
 *        to resolve one by name.
 * @return The F12-HF reference description.
 * @throws std::invalid_argument if the reference is not a closed-shell
 *         single-determinant wavefunction with an associated basis set, or if
 *         @p cabs_auxiliary_basis describes a different molecular structure.
 */
F12HartreeFockInput f12_input_from_wavefunction(
    const data::Wavefunction& reference, double gamma,
    const std::string& cabs_basis, std::size_t frozen_core,
    const std::shared_ptr<const data::AuxiliaryBasis>& cabs_auxiliary_basis =
        nullptr);

/**
 * @brief Result of the self-consistent F12-HF step over the dressed
 *        Hamiltonian.
 *
 * The dressed integrals are expressed in the @em original (canonical) MO basis
 * in which the geminal amplitudes were fixed; @ref relaxation rotates that
 * basis onto the F12-HF-relaxed one. The two-body integrals use the
 * physicists' @f$ \langle pq|rs \rangle @f$ convention with the flat layout
 * @c ((p*n+q)*n+r)*n+s.
 */
struct F12HartreeFockResult {
  std::size_t n_mo = 0;        ///< Number of molecular orbitals.
  std::size_t n_occupied = 0;  ///< Number of doubly occupied orbitals.
  std::size_t n_core = 0;      ///< Frozen core orbitals (formulation a).
  double e_hf = 0.0;           ///< Bare Hartree-Fock electronic energy.
  double e_f12hf = 0.0;        ///< Self-consistent F12-HF electronic energy.
  Eigen::MatrixXd one_body;    ///< Dressed one-body integrals, @c [n_mo, n_mo].
  std::vector<double> two_body;      ///< Dressed <pq|rs>, flat @c n_mo^4.
  Eigen::MatrixXd relaxation;        ///< Original-MO to relaxed-MO rotation.
  Eigen::VectorXd relaxed_energies;  ///< Dressed-Fock eigenvalues, @c [n_mo].
};

/**
 * @brief Build the dressed CT-F12 Hamiltonian and relax the orbitals in it.
 *
 * Constructs @f$ \bar{H}_{F12} @f$ (paper Eqs. 14-28) from the original
 * Hartree-Fock reference, with the geminal amplitudes fixed from the original
 * occupied orbitals, then relaxes the closed-shell orbitals in its mean field.
 * The geminal generator is not re-derived as the orbitals move.
 *
 * @param input The F12-HF reference description.
 * @return The dressed integrals in the original MO basis together with the
 *         relaxation and the bare and F12-HF electronic energies.
 */
F12HartreeFockResult run_f12_hf(const F12HartreeFockInput& input);

// The remaining entry points are independent reference implementations of the
// canonical transcorrelated F12 energies, used to validate the emitted
// Hamiltonian against the published values of J. Chem. Phys. 136, 084107
// (2012). They are not part of the production path.

/**
 * @brief Diagonal F12 pair intermediates over the geminal-generating space.
 *
 * Each tensor is indexed @c [n_val, n_val, n_val, n_val] over the valence
 * occupied orbitals, with @c V, @c X, and @c B corresponding to paper Eqs.
 * 23-25 of J. Chem. Phys. 136, 084107 (2012).
 */
struct F12Intermediates {
  std::size_t n_val = 0;             ///< Number of valence occupied orbitals.
  std::vector<double> v;             ///< V^{ij}_{kl} = g^{ij}_{ab} G^{ab}_{kl}.
  std::vector<double> x;             ///< X^{ij}_{kl} = G^{ab}_{ij} G^{ab}_{kl}.
  std::vector<double> b;             ///< B^{ij}_{kl} from G f G (Eq. 25).
  Eigen::VectorXd valence_energies;  ///< Orbital energies of valence orbitals.
};

/**
 * @brief Build the diagonal V/X/B F12 intermediates for an HF reference.
 *
 * @param input The F12-HF reference description.
 * @return The V/X/B tensors over the valence occupied space.
 */
F12Intermediates build_intermediates(const F12HartreeFockInput& input);

/**
 * @brief Evaluate the first-order F12-HF energy correction @f$ \langle
 *        \mathrm{HF} | \bar{H}_{F12} | \mathrm{HF} \rangle - E_{\mathrm{HF}}
 * @f$.
 *
 * This is the diagonal V/X/B geminal contribution evaluated with the original
 * (unrelaxed) Hartree-Fock orbitals. It is the inexpensive "standard" estimate;
 * the self-consistent @ref f12_hf_scf_energy additionally relaxes the orbitals
 * in the dressed field and is the quantity tabulated as "F12-HF" in the
 * reference literature.
 *
 * @param intermediates The diagonal V/X/B intermediates.
 * @return The first-order F12-HF energy correction in Hartree.
 */
double f12_hf_energy(const F12Intermediates& intermediates);

/**
 * @brief Evaluate the self-consistent F12-HF energy correction.
 *
 * Builds the full dressed transcorrelated Hamiltonian @f$ \bar{H}_{F12} @f$
 * (paper Eqs. 14-28) over the orbital basis and relaxes the closed-shell
 * orbitals in its mean field. The returned value is the converged
 * @f$ E(\text{F12-HF}) - E_{\mathrm{HF}} @f$, matching the "F12-HF" reference
 * energies of the canonical transcorrelated theory.
 *
 * @param input The F12-HF reference description.
 * @return The self-consistent F12-HF energy correction in Hartree.
 */
double f12_hf_scf_energy(const F12HartreeFockInput& input);

/**
 * @brief Conventional frozen-core closed-shell MP2 correlation energy.
 *
 * Evaluated in the orbital basis from the canonical Hartree-Fock orbitals, with
 * the lowest @c n_core orbitals left uncorrelated.
 *
 * @param input The F12-HF reference description.
 * @return The MP2 correlation energy in Hartree.
 */
double mp2_energy(const F12HartreeFockInput& input);

/**
 * @brief Total F12-MP2 correlation energy using the dressed Hamiltonian.
 *
 * Relaxes the orbitals in the dressed mean field (as @ref f12_hf_scf_energy)
 * and adds frozen-core closed-shell MP2 over the dressed two-electron integrals
 * with the dressed-Fock orbital energies. The returned value is the total
 * correlation relative to the bare Hartree-Fock reference, i.e. the F12-HF
 * relaxation plus the residual MP2.
 *
 * @param input The F12-HF reference description.
 * @return The total F12-MP2 correlation energy in Hartree.
 */
double f12_mp2_energy(const F12HartreeFockInput& input);

/**
 * @brief Conventional explicitly-correlated MP2-F12 correction (fixed-amplitude
 *        SP ansatz, approximation C).
 *
 * Adds the geminal-conventional-doubles coupling to the diagonal V/X/B
 * intermediates. The returned value is @f$ E(\mathrm{MP2\text{-}F12}) -
 * E(\mathrm{MP2}) @f$; the total MP2-F12 correlation energy is this plus
 * @ref mp2_energy.
 *
 * @param input The F12-HF reference description.
 * @return The F12 correction to the MP2 correlation energy in Hartree.
 */
double mp2_f12_correction(const F12HartreeFockInput& input);

}  // namespace qdk::chemistry::algorithms::microsoft::ctf12
