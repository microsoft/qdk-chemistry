// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

#include "ctf12_f12.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace ctf12 {

/**
 * @brief The dressed CT-F12 Hamiltonian expressed in a molecular-orbital basis.
 *
 * Holds the one- and two-body integrals of the transcorrelated Hamiltonian
 * @f$ \bar{H}_{F12} @f$ over the full orbital basis, together with the orbital
 * data of the basis in which they are expressed. The two-body integrals use the
 * chemists' @f$ (pq|rs) @f$ convention with the flat layout
 * @c ((p*n+q)*n+r)*n+s, matching @ref
 * data::CanonicalFourCenterHamiltonianContainer.
 */
struct DressedHamiltonian {
  std::size_t n_mo = 0;              ///< Number of molecular orbitals.
  std::size_t n_occupied = 0;        ///< Number of doubly occupied orbitals.
  std::size_t n_core = 0;            ///< Frozen core orbitals (formulation a).
  Eigen::MatrixXd mo_coefficients;   ///< AO->MO coefficients, @c [n_ao, n_mo].
  Eigen::VectorXd orbital_energies;  ///< Orbital energies, @c [n_mo].
  Eigen::MatrixXd one_body;  ///< Dressed one-body integrals, @c [n_mo, n_mo].
  std::vector<double> two_body;  ///< Dressed (pq|rs), flat @c n_mo^4.
  double e_hf = 0.0;             ///< Bare Hartree-Fock electronic energy.
  double e_f12hf = 0.0;          ///< Self-consistent F12-HF electronic energy.
};

/**
 * @brief Express the dressed CT-F12 Hamiltonian in a molecular-orbital basis.
 *
 * Runs @ref run_f12_hf and repackages its integrals in the chemists'
 * convention. When @p relax_orbitals is true the Hamiltonian is rotated into
 * the relaxed F12-HF canonical basis and carries the dressed-Fock orbital
 * energies; conventional post-Hartree-Fock methods over it then reproduce the
 * canonical F12-MP2/F12-CCSD energies. When false it is returned in the
 * original reference basis (a drop-in replacement for the bare Hamiltonian)
 * with the input orbital energies.
 *
 * @param input The F12-HF reference description.
 * @param relax_orbitals Express the Hamiltonian in the relaxed F12-HF basis.
 * @return The dressed Hamiltonian in the requested orbital basis.
 */
DressedHamiltonian build_dressed_hamiltonian(const F12HartreeFockInput& input,
                                             bool relax_orbitals);

}  // namespace ctf12

/**
 * @brief Settings for the canonical transcorrelated F12 (CT-F12) Hamiltonian
 *        constructor.
 *
 * All configuration lives here and is locked at run(). See the CT-F12 design
 * for the meaning of each key.
 */
class CtF12HamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  CtF12HamiltonianSettings() {
    set_default<double>("gamma", 1.0,
                        "Slater geminal exponent gamma (atomic units)",
                        data::BoundConstraint<double>{0.0, 100.0});
    set_default<int64_t>("frozen_core", 0,
                         "Number of frozen core orbitals (formulation (a))",
                         data::BoundConstraint<int64_t>{0});
    set_default("eri_method", std::string("direct"),
                "ERI evaluation method: 'direct' computes integrals "
                "on-the-fly, 'incore' stores all integrals in memory",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"direct", "incore"}}});
    set_default("slater_factor", std::string("stg"),
                "Slater factor representation: 'stg' genuine Slater-type "
                "geminal, 'cgtg' Gaussian-fitted geminal",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"stg", "cgtg"}}});
    set_default("orbital_basis", std::string("relaxed"),
                "Orbital basis of the emitted Hamiltonian: 'relaxed' relaxes "
                "the closed-shell orbitals in the dressed mean field and emits "
                "the F12-HF canonical basis (canonical post-HF over it "
                "reproduces F12-MP2/F12-CCSD); 'reference' keeps the reference "
                "orbital basis (a drop-in replacement for the bare "
                "Hamiltonian)",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"relaxed", "reference"}}});
    set_default("symmetrize_two_body", false,
                "Experimental 8-fold averaging for external solvers that "
                "require it; changes the CT-F12 Hamiltonian and is not "
                "required by native QDK MP2 or MACIS");
  }
  ~CtF12HamiltonianSettings() override = default;
};

/**
 * @brief Canonical transcorrelated F12 (CT-F12) effective-Hamiltonian
 *        constructor.
 *
 * Produces an a priori, Hermitian, two-body effective Hamiltonian by an
 * approximate canonical (unitary) similarity transformation of the molecular
 * Hamiltonian with a fixed-amplitude Slater-type geminal generator. The
 * reduced density matrices that close the cumulant reduction are read from the
 * reference wavefunction, so a single-determinant reference yields the
 * single-reference flavor while a multi-determinant reference yields the
 * multireference flavor through the same code path.
 *
 * The dressed integrals are rebuilt from the reference orbitals, so the input
 * Hamiltonian only fixes the outer orbital window @f$W@f$.
 *
 * The external space folded in has two parts: the complementary auxiliary
 * basis (CABS), which lies outside @f$W@f$ altogether, and the occupied
 * orbitals of @f$W \setminus P@f$, which become inactive. The CABS is
 * mandatory and is read from the @c AuxiliaryBasisRole::CABS entry of the
 * auxiliary bases passed to @c run().
 *
 * @c frozen_core and @c p_indices control different things and may disagree:
 * @c frozen_core selects the geminal-generating occupied set (formulation
 * (a)), while @c p_indices selects the emitted active space. An occupied
 * orbital may therefore generate geminals and still be frozen in the output.
 *
 * @see algorithms::EffectiveHamiltonianConstructor
 */
class CtF12HamiltonianConstructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  CtF12HamiltonianConstructor() {
    _settings = std::make_unique<CtF12HamiltonianSettings>();
  };
  ~CtF12HamiltonianConstructor() override = default;

  std::string name() const final { return "qdk_ct_f12"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices,
      std::shared_ptr<const data::AuxiliaryBasisCollection> auxiliary_bases)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
