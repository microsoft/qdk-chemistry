// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <libint2.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace qdk::chemistry::scf::geminal {

/**
 * @brief Compute genuine Slater-type-geminal AO integrals over four basis sets.
 *
 * Evaluates the two-particle AO integrals @f$ (p q | \hat{O} | r s) @f$ in
 * chemist notation for a genuine (non-Gaussian-fitted) Slater-type geminal
 * kernel, where @f$ \hat{O} @f$ is selected by @p op. The four shell positions
 * may come from four distinct basis sets, which is required for the mixed
 * OBS / CABS index spaces of explicitly correlated (F12) methods.
 *
 * Supported operators (the Slater exponent @p gamma is passed to the libint2
 * engine constructor, not via @c set_params):
 * - @c libint2::Operator::stg : @f$ \exp(-\gamma r_{12}) @f$ (geminal factor)
 * - @c libint2::Operator::stg_x_coulomb : @f$ \exp(-\gamma r_{12}) / r_{12} @f$
 *
 * The caller is responsible for applying any scalar prefactor of the
 * correlation factor (e.g. the @f$ -1/\gamma @f$ in @f$ F_{12} @f$) and the
 * strong-orthogonality projector.
 *
 * @param op Geminal operator (must be @c stg or @c stg_x_coulomb).
 * @param gamma Slater exponent of the geminal.
 * @param bs1 Basis set for the first (bra) index @c p.
 * @param bs2 Basis set for the second (bra) index @c q.
 * @param bs3 Basis set for the third (ket) index @c r.
 * @param bs4 Basis set for the fourth (ket) index @c s.
 * @return Row-major buffer of size @c bs1.nbf()*bs2.nbf()*bs3.nbf()*bs4.nbf()
 *         holding @f$ (p q | \hat{O} | r s) @f$.
 * @throws std::invalid_argument if @p op is not a supported Slater kernel.
 */
std::unique_ptr<double[]> stg_geminal_eri(::libint2::Operator op, double gamma,
                                          const ::libint2::BasisSet& bs1,
                                          const ::libint2::BasisSet& bs2,
                                          const ::libint2::BasisSet& bs3,
                                          const ::libint2::BasisSet& bs4);

/**
 * @brief Compute Coulomb AO integrals over four (possibly distinct) basis sets.
 *
 * Evaluates the chemist-notation electron-repulsion integrals
 * @f$ (p q | r s) = \int p(1) q(1) r_{12}^{-1} r(2) s(2) @f$ with each of the
 * four shell positions drawn from an independent basis set, as required for the
 * mixed orbital/CABS index spaces of explicitly correlated methods. Unlike the
 * single-basis @c opt_eri, no permutational symmetry is assumed.
 *
 * @param bs1,bs2,bs3,bs4 Basis sets for the four indices.
 * @return Row-major buffer of size @c bs1.nbf()*bs2.nbf()*bs3.nbf()*bs4.nbf().
 */
std::unique_ptr<double[]> four_center_coulomb(const ::libint2::BasisSet& bs1,
                                              const ::libint2::BasisSet& bs2,
                                              const ::libint2::BasisSet& bs3,
                                              const ::libint2::BasisSet& bs4);

/**
 * @brief Compute the kinetic-energy AO matrix over a single basis set.
 * @param bs Basis set.
 * @return Symmetric matrix @c [bs.nbf(), bs.nbf()].
 */
Eigen::MatrixXd kinetic_matrix(const ::libint2::BasisSet& bs);

/**
 * @brief Compute the nuclear-attraction AO matrix over a single basis set.
 * @param bs Basis set.
 * @param charges Nuclear charges and Cartesian positions (atomic units).
 * @return Symmetric matrix @c [bs.nbf(), bs.nbf()].
 */
Eigen::MatrixXd nuclear_matrix(
    const ::libint2::BasisSet& bs,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges);

/**
 * @brief Transform a four-index AO tensor to the MO basis.
 *
 * Computes @f$ T_{pqrs} = \sum_{\mu\nu\lambda\sigma} C^1_{\mu p} C^2_{\nu q}
 * C^3_{\lambda r} C^4_{\sigma s} A_{\mu\nu\lambda\sigma} @f$, where @p ao is a
 * row-major @c [n1,n2,n3,n4] AO tensor and each @c Ck has shape @c [nk, mk].
 * The four coefficient blocks may differ, which is required to project the AO
 * indices onto distinct MO spaces (occupied, OBS-virtual, CABS).
 *
 * @param ao Row-major AO integral tensor of shape @c [n1,n2,n3,n4].
 * @param n1,n2,n3,n4 AO dimensions of @p ao.
 * @param c1,c2,c3,c4 Transformation matrices, each @c [nk, mk] (column-major).
 * @return Row-major MO tensor of shape @c [m1,m2,m3,m4].
 */
std::unique_ptr<double[]> mo_transform_4index(
    const double* ao, std::size_t n1, std::size_t n2, std::size_t n3,
    std::size_t n4, const Eigen::MatrixXd& c1, const Eigen::MatrixXd& c2,
    const Eigen::MatrixXd& c3, const Eigen::MatrixXd& c4);

}  // namespace qdk::chemistry::scf::geminal
