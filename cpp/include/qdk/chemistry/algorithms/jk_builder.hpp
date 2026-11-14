// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstddef>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class JKEngine
 * @brief Coulomb (J) and exchange (K) builds in one atomic-orbital basis set.
 *
 * Created by JKBuilder::run for a basis set and reused for every build of a
 * calculation (for example, each SCF iteration).
 */
class JKEngine {
 public:
  virtual ~JKEngine() = default;

  /**
   * @brief Build J and K for each density matrix.
   *
   * J[P]_{mn} = sum_{ls} (mn|ls) P_{ls} and
   * K[P]_{mn} = sum_{ls} (ml|v|ns) P_{ls} with the exchange operator
   * v = alpha / r + beta erfc(omega r) / r (v = (alpha + beta) / r for
   * omega = 0). The matrices are in the atomic orbitals of the engine's basis
   * set.
   *
   * @param P num_densities symmetric num_atomic_orbitals x num_atomic_orbitals
   * matrices, stored consecutively
   * @param num_atomic_orbitals Number of atomic orbitals of the basis set
   * @param num_densities Number of density matrices
   * @param J Output Coulomb matrices in the layout of P, or nullptr to skip J
   * @param K Output exchange matrices in the layout of P, or nullptr to skip K
   * @param alpha Coefficient of the full-range exchange operator
   * @param beta Coefficient of the short-range exchange operator
   * @param omega Range-separation parameter
   */
  virtual void build(const double* P, size_t num_atomic_orbitals,
                     size_t num_densities, double* J, double* K, double alpha,
                     double beta, double omega) = 0;

  /**
   * @brief Build the nuclear gradients of the Coulomb and exchange energies.
   *
   * Derivatives with respect to the nuclear coordinates, at fixed densities,
   * of E_J = 1/2 Tr(P J[P]) for the total density P and of
   * E_K = -1/4 Tr(P K[P]) for one density matrix or
   * E_K = -1/2 sum_s Tr(P_s K[P_s]) for two (alpha and beta), with J and K as
   * in build.
   *
   * @param P Density matrices, as in build
   * @param num_atomic_orbitals Number of atomic orbitals of the basis set
   * @param num_densities Number of density matrices
   * @param num_atoms Number of atoms of the basis set's structure
   * @param dJ Output 3 x num_atoms row-major gradient of E_J, or nullptr to
   * skip it
   * @param dK Output gradient of E_K in the layout of dJ, or nullptr to skip
   * it
   * @param alpha Coefficient of the full-range exchange operator
   * @param beta Coefficient of the short-range exchange operator
   * @param omega Range-separation parameter
   * @throws std::runtime_error if the engine does not build gradients (the
   * default)
   */
  virtual void build_gradients(const double* P, size_t num_atomic_orbitals,
                               size_t num_densities, size_t num_atoms,
                               double* dJ, double* dK, double alpha,
                               double beta, double omega);
};

/**
 * @class JKBuilder
 * @brief Creates the J/K engine of a basis set.
 *
 * The "qdk" ScfSolver builds its Coulomb and exchange matrices, and their
 * nuclear gradients, with the builder named by its "jk_builder" setting.
 */
class JKBuilder : public Algorithm<JKBuilder, std::shared_ptr<JKEngine>,
                                   std::shared_ptr<data::BasisSet>, double> {
 public:
  JKBuilder() = default;
  virtual ~JKBuilder() = default;

  /**
   * @brief Create the J/K engine of a basis set
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param basis_set Basis set; the engine's matrices are in its atomic
   * orbitals, in their order and with each contracted shell normalized
   * @param eri_threshold Integral screening threshold
   * \endcond
   * @return The engine
   */
  using Algorithm::run;

  virtual std::string name() const = 0;

  std::string type_name() const final { return "jk_builder"; };

 protected:
  virtual std::shared_ptr<JKEngine> _run_impl(
      std::shared_ptr<data::BasisSet> basis_set,
      double eri_threshold) const = 0;
};

/**
 * @brief Factory for J/K builders.
 *
 * The library provides "qdk" (the default); plugins can register others.
 */
struct JKBuilderFactory : public AlgorithmFactory<JKBuilder, JKBuilderFactory> {
  static std::string algorithm_type_name() { return "jk_builder"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
