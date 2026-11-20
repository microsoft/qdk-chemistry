// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/scf.h>

#include <libint2.hpp>
#include <memory>
#include <qdk/chemistry/data/two_body_kernel.hpp>

namespace qdk::chemistry::scf::libint2_util {

using qdk::chemistry::scf::BasisMode;
using qdk::chemistry::scf::BasisSet;
using qdk::chemistry::scf::Shell;

/**
 * @brief Convert QDK/Chemistry-SCF basis set to libint2 format
 * @param obs QDK/Chemistry-SCF basis set
 * @return Libint2 basis set
 */
::libint2::BasisSet convert_to_libint_basisset(const BasisSet& obs);

/**
 * @brief Convert QDK/Chemistry-SCF shell to libint2 format
 * @param shell QDK/Chemistry-SCF shell
 * @param pure Whether to use pure spherical harmonics
 * @return Libint2 shell
 */
::libint2::Shell convert_to_libint_shell(const Shell& shell, bool pure);

/**
 * @brief Construct a Libint2 engine for the integrals of a two-body kernel
 * @param kernel Two-body kernel
 * @param obs Orbital basis set
 * @param deriv Derivative order
 * @return Engine for the integrals of kernel
 * @throws std::invalid_argument If Libint2 does not provide the kernel
 */
::libint2::Engine make_eri_engine(const data::TwoBodyKernel& kernel,
                                  const ::libint2::BasisSet& obs, int deriv);

/**
 * @brief Compute a shell quartet of two-electron integrals of the engine's
 * kernel, as constructed by make_eri_engine
 * @tparam Deriv Derivative order of the engine
 * @param engine Engine, whose results receive the integrals
 * @param s1 First bra shell
 * @param s2 Second bra shell
 * @param s3 First ket shell
 * @param s4 Second ket shell
 * @param sp12 Precomputed data of the bra shell pair, if any
 * @param sp34 Precomputed data of the ket shell pair, if any
 */
template <size_t Deriv>
void compute_eri(::libint2::Engine& engine, const ::libint2::Shell& s1,
                 const ::libint2::Shell& s2, const ::libint2::Shell& s3,
                 const ::libint2::Shell& s4,
                 const ::libint2::ShellPair* sp12 = nullptr,
                 const ::libint2::ShellPair* sp34 = nullptr) {
  using ::libint2::BraKet;
  using ::libint2::Operator;
  switch (engine.oper()) {
    case Operator::erf_coulomb:
      engine.compute2<Operator::erf_coulomb, BraKet::xx_xx, Deriv>(
          s1, s2, s3, s4, sp12, sp34);
      break;
    case Operator::erfc_coulomb:
      engine.compute2<Operator::erfc_coulomb, BraKet::xx_xx, Deriv>(
          s1, s2, s3, s4, sp12, sp34);
      break;
    default:
      engine.compute2<Operator::coulomb, BraKet::xx_xx, Deriv>(s1, s2, s3, s4,
                                                               sp12, sp34);
  }
}

/**
 * @brief Compute ERI integrals with debug-level checks
 * @param basis_mode atomic orbital type (cartesian/spherical)
 * @param obs Orbital basis set
 * @param kernel Two-body kernel of the integrals
 * @param i_lo Starting atomic orbital index
 * @param i_hi Ending atomic orbital index
 * @return ERI integral buffer
 */
std::unique_ptr<double[]> debug_eri(BasisMode basis_mode,
                                    const ::libint2::BasisSet& obs,
                                    const data::TwoBodyKernel& kernel,
                                    size_t i_lo, size_t i_hi);

/**
 * @brief Compute ERI integrals with optimized screening
 * @param basis_mode atomic orbital type (cartesian/spherical)
 * @param obs Orbital basis set
 * @param kernel Two-body kernel of the integrals
 * @param i_lo Starting atomic orbital index
 * @param i_hi Ending atomic orbital index
 * @return ERI integral buffer
 */
std::unique_ptr<double[]> opt_eri(BasisMode basis_mode,
                                  const ::libint2::BasisSet& obs,
                                  const data::TwoBodyKernel& kernel,
                                  size_t i_lo, size_t i_hi);

/**
 * @brief Compute density-fitted ERI integrals (Q|μν)
 * @param basis_mode atomic orbital type (cartesian/spherical)
 * @param obs Orbital basis set
 * @param abs Auxiliary basis set for density fitting
 * @param i_lo Starting atomic orbital index
 * @param i_hi Ending atomic orbital index
 * @return Three-center integral buffer
 */
std::unique_ptr<double[]> eri_df(BasisMode basis_mode,
                                 const ::libint2::BasisSet& obs,
                                 const ::libint2::BasisSet& abs, size_t i_lo,
                                 size_t i_hi);

/**
 * @brief Compute density-fitted metric integrals (Q|P)
 * @param basis_mode atomic orbital type (cartesian/spherical)
 * @param abs Auxiliary basis set for density fitting
 * @return Two-center integral buffer
 */
std::unique_ptr<double[]> metric_df(BasisMode basis_mode,
                                    const ::libint2::BasisSet& abs);

/**
 * @brief Compute gradient contributions from density-fitted ERI integrals
 * @param dJ Output gradient buffer for Coulomb contribution
 * @param P Density matrix
 * @param X DF fitting coefficients
 * @param basis_mode atomic orbital type (cartesian/spherical)
 * @param obs Orbital basis set
 * @param abs Auxiliary basis set
 * @param obs_sh2atom Shell to atom mapping for orbital basis
 * @param abs_sh2atom Shell to atom mapping for auxiliary basis
 * @param n_atoms Number of atoms
 * @param mpi MPI configuration
 */
void eri_df_grad(double* dJ, const double* P, const double* X,
                 BasisMode basis_mode, const ::libint2::BasisSet& obs,
                 const ::libint2::BasisSet& abs,
                 const std::vector<int>& obs_sh2atom,
                 const std::vector<int>& abs_sh2atom, size_t n_atoms,
                 ParallelConfig mpi);

/**
 * @brief Compute gradient contributions from density-fitted metric integrals
 * @param dJ Output gradient buffer for metric contribution
 * @param X DF fitting coefficients
 * @param basis_mode atomic orbital type (cartesian/spherical)
 * @param abs Auxiliary basis set
 * @param abs_sh2atom Shell to atom mapping for auxiliary basis
 * @param n_atoms Number of atoms
 * @param mpi MPI configuration
 */
void metric_df_grad(double* dJ, const double* X, BasisMode basis_mode,
                    const ::libint2::BasisSet& abs,
                    const std::vector<int>& abs_sh2atom, size_t n_atoms,
                    ParallelConfig mpi);

}  // namespace qdk::chemistry::scf::libint2_util
