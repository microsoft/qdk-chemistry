// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <libint2.hpp>

namespace qdk::chemistry::scf::cabs {

/**
 * @brief Result of a complementary auxiliary basis set (CABS) construction.
 *
 * The CABS functions are expressed as linear combinations of the atomic
 * orbitals of the @ref ri_basis union (orbital basis followed by the auxiliary
 * basis). Integrals involving a CABS index are therefore evaluated over
 * @ref ri_basis and transformed with @ref cabs_coeff.
 */
struct CabsResult {
  ::libint2::BasisSet ri_basis;  ///< Union basis (OBS shells then aux shells).
  Eigen::MatrixXd cabs_coeff;    ///< CABS coefficients, shape [ri_nbf, n_cabs].
};

/**
 * @brief Compute the two-center AO overlap @f$ S_{pq} = (p | q) @f$.
 *
 * The two index spaces may come from different basis sets, as required when
 * forming cross-overlaps between the orbital and auxiliary bases.
 *
 * @param bs1 Basis set for the first (row) index.
 * @param bs2 Basis set for the second (column) index.
 * @return Row-major overlap matrix of shape @c [bs1.nbf(), bs2.nbf()].
 */
Eigen::MatrixXd ao_overlap(const ::libint2::BasisSet& bs1,
                           const ::libint2::BasisSet& bs2);

/**
 * @brief Construct the CABS+ complementary auxiliary basis for @p obs.
 *
 * Implements Valeev's CABS+ construction: the orbital basis and the auxiliary
 * (OptRI/RI) basis are combined into a single resolution-of-the-identity space,
 * and the component orthogonal to the orbital basis is extracted by canonical
 * orthogonalization followed by a singular value decomposition. The resulting
 * CABS functions are orthonormal and strongly orthogonal to the orbital basis.
 *
 * @param obs Orbital basis set.
 * @param aux Auxiliary (OptRI/RI) basis set.
 * @param lindep_tol Eigenvalue/singular-value threshold for linear dependence.
 * @return The union basis and the CABS coefficients within it.
 */
CabsResult build_cabs(const ::libint2::BasisSet& obs,
                      const ::libint2::BasisSet& aux, double lindep_tol = 1e-8);

}  // namespace qdk::chemistry::scf::cabs
