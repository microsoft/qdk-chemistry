// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <limits>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

#include "ctf12_f12.hpp"

namespace qdk::chemistry::algorithms::microsoft {

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
                        data::BoundConstraint<double>{
                            std::numeric_limits<double>::min(), 100.0});
    set_default<int64_t>("frozen_core", 0,
                         "Number of frozen core orbitals (formulation (a))",
                         data::BoundConstraint<int64_t>{0});
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
 * Hamiltonian with a fixed-amplitude Slater-type geminal generator.
 *
 * Only closed-shell single-determinant references are supported: the geminal
 * amplitudes are fixed from the reference occupied orbitals and no reduced
 * density matrices are read from @p reference, so a multi-determinant
 * wavefunction is treated as though it were its own orbital basis.
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
 * The geminal amplitudes are carried by pairs of @em occupied OBS orbitals,
 * the contiguous block @f$[n_{core}, n_{occ})@f$ fixed by @c frozen_core, while
 * the external index of the generator is always the whole CABS plus OBS virtual
 * space. @c frozen_core is therefore independent of @c p_indices: it selects
 * which electrons are correlated at the F12 level, whereas @c p_indices selects
 * which orbitals the emitted Hamiltonian acts on. An occupied orbital left out
 * of P still carries geminal amplitudes unless it is also in the frozen core;
 * either way it is folded into the inactive Fock matrix and the constant energy
 * term, and @c data::Orbitals has no frozen class to tell the two apart in the
 * emitted @c inactive_indices().
 *
 * @warning Under @c orbital_basis="relaxed" the emitted orbitals are the
 * F12-HF-relaxed ones, so @c p_indices select positions in the @em relaxed
 * ordering rather than the reference orbitals they were validated against.
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
