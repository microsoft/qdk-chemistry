/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/fci/string_space.hpp>
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Spin-resolved one- and two-particle RDMs of an FCI vector.
 *
 * Same layout and normalization as HamiltonianGenerator::form_rdms_spin_dep
 * (column-major spans; E^s_pq = a+_ps a_qs):
 *   ordm_ss(p,q)       = <E^s_pq>,
 *   trdm_ssss(p,q,r,s) = 1/2 <a+_ps a+_rs a_ss a_qs>,
 *   trdm_aabb(p,q,r,s) = 1/2 <E^beta_pq E^alpha_rs>.
 * Spans with a null data handle are skipped; the others are overwritten.
 *
 * @param[in] alpha     Alpha string space of the CI vector
 * @param[in] beta      Beta string space of the CI vector
 * @param[in] C         Real CI vector, alpha-major (C[Ia * Nb + Ib])
 * @param[in] spin_flip Non-zero when C^T = spin_flip * C (equal alpha and
 *                      beta strings; see FCIHamiltonian::sigma): the beta
 *                      RDMs then equal the alpha ones and are copied
 * @throws std::invalid_argument if spin_flip is non-zero for different
 *                               alpha and beta string spaces
 */
void fci_rdms_spin_dep(const FCIStringSpace& alpha, const FCIStringSpace& beta,
                       const double* C, matrix_span<double> ordm_aa,
                       matrix_span<double> ordm_bb,
                       rank4_span<double> trdm_aaaa,
                       rank4_span<double> trdm_bbbb,
                       rank4_span<double> trdm_aabb, int spin_flip = 0);

}  // namespace macis
