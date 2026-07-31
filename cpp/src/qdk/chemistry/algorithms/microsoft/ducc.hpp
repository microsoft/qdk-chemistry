// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Spin-orbital DUCC input data extracted from the full-MO Hamiltonian
 * and the CC-amplitude Wavefunction (DUCC step "2a").
 *
 * Occupied-first spin-BLOCKED spin-orbital layout
 * ``[alpha-occ, beta-occ, alpha-vir, beta-vir]`` (handles restricted and
 * unrestricted references; the interleaved layout would require
 * nocc_a==nocc_b). All arrays are row-major (C order). See @ref
 * extract_spinorbital_data.
 */
struct SpinOrbitalData {
  std::size_t nso = 0;      ///< total spin-orbitals (2*nmo)
  std::size_t nocc_so = 0;  ///< occupied spin-orbitals (n_alpha + n_beta)
  std::size_t nvir_so = 0;  ///< virtual spin-orbitals
  std::size_t nocc_a =
      0;  ///< alpha occupied (spatial), for active-space mapping
  std::size_t nocc_b =
      0;                  ///< beta occupied (spatial), for active-space mapping
  double scalar = 0.0;    ///< reference (HF) energy E0
  std::vector<double> F;  ///< Fock [nso, nso]
  std::vector<double> V;  ///< antisymmetrized <PQ||RS> [nso, nso, nso, nso]
  std::vector<double> T1;  ///< CC singles [nvir_so, nocc_so]  (a, i)
  std::vector<double>
      T2;  ///< CC doubles [nvir_so, nvir_so, nocc_so, nocc_so]  (a, b, i, j)
};

/**
 * @brief Extract spin-orbital DUCC data from a full-MO Hamiltonian + amplitude
 *        Wavefunction (DUCC step "2a").
 *
 * The Hamiltonian supplies the full-space spin-blocked MO integrals (chemist
 * ``(pq|rs)``) via the container-agnostic accessors (works for
 * CanonicalFourCenter and Cholesky, restricted and unrestricted). The
 * Wavefunction's amplitude container supplies the full-space CC T1/T2. The
 * restricted/unrestricted character is taken from the Hamiltonian; a mismatch
 * with the amplitude storage is rejected.
 *
 * @param hamiltonian Full-space Hamiltonian (all MOs).
 * @param wavefunction Wavefunction whose container is an AmplitudeContainer.
 * @return Spin-orbital F/V/T1/T2 + scalar, in the occ-first blocked layout.
 * @throws std::bad_cast if the wavefunction container is not an
 * AmplitudeContainer.
 * @throws std::runtime_error if the Hamiltonian and wavefunction disagree on
 *         restrictedness.
 */
SpinOrbitalData extract_spinorbital_data(
    const data::Hamiltonian& hamiltonian,
    const data::Wavefunction& wavefunction);

/**
 * @brief Assemble the active-space effective Hamiltonian from the (dressed)
 *        spin-orbital data (DUCC step "2c").
 *
 * Restricts the spin-orbital Fock @c dressed.F (@f$\bar f@f$) and
 * antisymmetrized two-body @c dressed.V (@f$\bar v = \langle PQ||RS\rangle@f$)
 * to the active spin-orbitals, re-normal-orders from the Fermi vacuum to the
 * physical vacuum
 * (@f$\gamma \to \chi@f$, absorbing the active-occupied contractions into the
 * one-body term and scalar per Bauman et al., JCP 151, 014107), and packs the
 * result into a @ref data::CanonicalFourCenterHamiltonianContainer. The
 * two-body integrals are emitted in qdk's chemist convention: the same-spin
 * blocks store the chemist representative @f$g =
 * \tfrac12\,\chi_2^{(0,2,1,3)}@f$
 * (@ref qdk::chemistry::utils::antisymmetrized_to_chemist) and the
 * opposite-spin block @f$2g@f$. This matches the qdk storage (only 4-fold
 * symmetry is assumed; consumers re-antisymmetrize the same-spin block), so it
 * is valid at every BCH level -- it is computed from the antisymmetrized
 * two-body alone.
 *
 * Output container form. A restricted @c ducc_level-0 result retains full
 * 8-fold two-body symmetry and is emitted as a single-block restricted
 * container, matching the @c hamiltonian_constructor input format. The BCH
 * dressing
 * (@c ducc_level > 0) lowers the two-body to 4-fold symmetry, which a single
 * restricted block cannot convey to consumers that assume 8-fold symmetry (e.g.
 * the qubit mapper's restricted fast path). A dressed result is therefore
 * always emitted in spin-blocked (@f$aaaa/bbbb/aabb@f$) form -- with identical
 * alpha/beta blocks for a restricted system -- so every consumer reads the
 * reduced-symmetry two-body through its unrestricted path. Consequently a
 * restricted @c ducc_level > 0 output is an unrestricted-type container
 * consumed by the qubit mapper (or PySCF @c direct_uhf), not the
 * restricted-only MACIS solver.
 *
 * For @c ducc_level 0 (no BCH) @c dressed is the bare extraction, so the output
 * reproduces the input Hamiltonian restricted to the active space to CI-energy
 * accuracy: CASCI on the input equals FCI on the output. The active space is
 * given by the per-spin active spatial-MO index sets (empty means "all MOs").
 *
 * @param dressed Spin-orbital data (F/V/scalar + dimensions); F/V may be the
 *        bare extraction (level 0) or the BCH-transformed values.
 * @param active_a_spatial Active alpha spatial-MO indices.
 * @param active_b_spatial Active beta spatial-MO indices.
 * @param restricted Whether the underlying Hamiltonian is spin-restricted; a
 *        single-block container is emitted only together with @c ducc_level 0
 *        (see "Output container form").
 * @param ducc_level BCH truncation level of @p dressed; 0 emits a restricted
 *        single-block container (when @p restricted), > 0 always emits the
 *        spin-blocked container.
 * @param active_orbitals The active-space @ref data::Orbitals carried through
 * to the output container, so the effective Hamiltonian retains the real
 *        orbital type/coefficients/basis of the active-space input (matching
 * the
 *        @c hamiltonian_constructor CAS convention) rather than abstract model
 *        orbitals. Its active-space designation must be consistent with
 *        @p active_a_spatial / @p active_b_spatial and @p restricted.
 * @return The effective active-space Hamiltonian.
 * @throws std::runtime_error if the active space has unequal alpha/beta counts.
 */
std::shared_ptr<data::Hamiltonian> assemble_active_hamiltonian(
    const SpinOrbitalData& dressed,
    const std::vector<std::size_t>& active_a_spatial,
    const std::vector<std::size_t>& active_b_spatial, bool restricted,
    int ducc_level, std::shared_ptr<data::Orbitals> active_orbitals);

/**
 * @class DuccSolver
 * @brief DUCC effective-Hamiltonian builder evaluated with SeQuant + BTAS.
 *
 * Builds the active-space effective Hamiltonian by a truncated
 * Baker-Campbell-Hausdorff (BCH) expansion of the unitarily
 * similarity-transformed Hamiltonian @f$ \bar H = e^{-\sigma} H e^{\sigma} @f$
 * with anti-Hermitian generator @f$ \sigma = T - T^\dagger @f$ (DUCC). The
 * symbolic transformation and partial Wick contraction are performed with the
 * SeQuant many-body algebra library (full-H Lie transform, `unitary=true`);
 * the resulting tensor network is evaluated numerically with the BTAS backend.
 *
 * Design notes carried over from the validated standalone prototype:
 * - **No SeQuant source patch.** The @f$ \le 2 @f$-body truncation is imposed
 * by a public-API post-filter on the residual-operator rank (applied before
 *   `simplify`), reproducing a rank-capped partial Wick exactly.
 * - **Active-only output.** Only the active block of @f$ \bar H @f$ is
 *   evaluated: the residual (free) legs are restricted to the active space so
 *   the BTAS contractions run at active extents (real work reduction, not
 *   compute-then-slice).
 *
 * The active space and reference coupled-cluster amplitudes are taken from the
 * input @ref data::Wavefunction; the BCH truncation level is the sole setting
 * (`ducc_level`).
 *
 * @note Level 0 (the bare active-space Hamiltonian, no BCH dressing) is served
 *       directly in C++. Levels 1-2 build the transformed @f$\bar H@f$ with the
 *       SeQuant/BTAS numeric backend (always linked). Both are validated to
 *       machine precision against the wicked-based DUCC reference.
 */
class DuccSolver : public qdk::chemistry::algorithms::EffectiveHamiltonian {
 public:
  /**
   * @brief Default constructor. Inherits the shared effective-Hamiltonian
   *        settings (the BCH truncation level `ducc_level`).
   */
  DuccSolver() = default;

  /**
   * @brief Virtual destructor.
   */
  ~DuccSolver() = default;

  /**
   * @brief Access the algorithm's variant name.
   * @return "ducc".
   */
  std::string name() const final { return "ducc"; }

 protected:
  /**
   * @brief Build the DUCC effective active-space Hamiltonian via SeQuant +
   * BTAS.
   *
   * @param hamiltonian The full-space Hamiltonian to transform.
   * @param wavefunction Full-space wavefunction supplying the reference
   *        coupled-cluster amplitudes.
   * @param active_orbitals Active-space orbitals whose active-space indices
   *        designate the active subset of @p wavefunction's orbitals.
   * @return The effective active-space Hamiltonian.
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<data::Wavefunction> wavefunction,
      std::shared_ptr<data::Orbitals> active_orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
