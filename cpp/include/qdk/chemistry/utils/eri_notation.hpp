// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @file eri_notation.hpp
 * @brief Convert rank-4 two-electron integrals between the "chemist" and
 *        "physicist antisymmetrized" notations.
 *
 * Two spin-orbital conventions appear throughout the code:
 * - **chemist** @f$(PQ|RS)@f$ — Mulliken ordering, 8-fold permutation
 *   symmetric, used by the Hamiltonian containers / CI solvers.
 * - **physicist antisymmetrized** @f$\langle PQ\|RS\rangle = \langle
 * PQ|RS\rangle
 *   - \langle PQ|SR\rangle@f$ — Dirac ordering, antisymmetric under @f$P
 *   \leftrightarrow Q@f$ and @f$R \leftrightarrow S@f$, produced by the DUCC /
 *   coupled-cluster many-body algebra (e.g. the SeQuant backend).
 *
 * All tensors are dense, row-major (C order), with a common extent @p n along
 * every mode (index @f$((P n + Q) n + R) n + S@f$).
 */

/**
 * @brief Convert chemist @f$(PQ|RS)@f$ to physicist antisymmetrized
 *        @f$\langle PQ\|RS\rangle@f$.
 *
 * @f$\langle PQ\|RS\rangle = (PR|QS) - (PS|QR)@f$. Exact and involutive-free;
 * every antisymmetrized element is fully determined by the chemist tensor.
 *
 * @param chemist Chemist integrals @f$(PQ|RS)@f$, size @f$n^4@f$.
 * @param n Extent of each mode.
 * @return Antisymmetrized integrals @f$\langle PQ\|RS\rangle@f$, size
 * @f$n^4@f$.
 * @throws std::invalid_argument if @p chemist is not of size @f$n^4@f$.
 */
std::vector<double> chemist_to_antisymmetrized(
    const std::vector<double>& chemist, std::size_t n);

/**
 * @brief Convert physicist antisymmetrized @f$\langle PQ\|RS\rangle@f$ to a
 *        chemist representative @f$g@f$.
 *
 * @f$g[P,Q,R,S] = \tfrac12\,\langle PR\|QS\rangle@f$ (i.e.
 * @f$\tfrac12@f$ of the antisymmetrized tensor reindexed by the
 * @f$(0,2,1,3)@f$ permutation). This is the natural spin-orbital chemist
 * packing: the same-spin block is @f$g@f$ and the opposite-spin block is
 * @f$2g@f$ (the opposite-spin exchange vanishes, so @f$2g@f$ recovers the
 * chemist integral exactly, while the same-spin @f$g@f$ carries the
 * antisymmetric part the CI solver re-antisymmetrizes internally).
 *
 * This is a partial inverse of @ref chemist_to_antisymmetrized: the symmetric
 * part of a same-spin chemist tensor is not recoverable (it does not contribute
 * to any physical energy), whereas opposite-spin blocks round-trip exactly.
 *
 * @param antisymmetrized Antisymmetrized integrals @f$\langle PQ\|RS\rangle@f$,
 *        size @f$n^4@f$.
 * @param n Extent of each mode.
 * @return Chemist representative @f$g@f$, size @f$n^4@f$.
 * @throws std::invalid_argument if @p antisymmetrized is not of size @f$n^4@f$.
 */
std::vector<double> antisymmetrized_to_chemist(
    const std::vector<double>& antisymmetrized, std::size_t n);

}  // namespace qdk::chemistry::utils
