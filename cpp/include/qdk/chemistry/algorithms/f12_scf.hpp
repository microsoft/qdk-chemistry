// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class F12HartreeFockSolver
 * @brief Abstract base class for relaxing a reference in an F12-dressed mean
 *        field.
 *
 * Maps a reference @ref data::Wavefunction to a new @ref data::Wavefunction
 * whose orbitals are relaxed in the mean field of a similarity-transformed
 * (transcorrelated) Hamiltonian. The reference supplies the orbital basis from
 * which the geminal is generated; the returned wavefunction carries the relaxed
 * orbital coefficients and the corresponding dressed-Fock orbital energies, so
 * it is a drop-in canonical reference for downstream correlated methods.
 *
 * @see data::Wavefunction
 * @see data::Settings
 * @see EffectiveHamiltonianConstructor
 */
class F12HartreeFockSolver
    : public Algorithm<F12HartreeFockSolver,
                       std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  F12HartreeFockSolver() = default;
  virtual ~F12HartreeFockSolver() = default;

  /**
   * @brief Relax a reference wavefunction in the F12-dressed mean field.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param reference The reference wavefunction supplying the orbital basis
   * \endcond
   * @return The relaxed F12-HF wavefunction
   *
   * @throw std::runtime_error if the solve fails
   * @throw std::invalid_argument if the reference is incomplete or invalid
   * @throws qdk::chemistry::data::SettingsAreLocked if attempting to modify
   * settings after run() is called
   *
   * @note Settings are automatically locked when this method is called.
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name
   * @return The algorithm's type name
   */
  std::string type_name() const final { return "f12_hartree_fock_solver"; };

 protected:
  /**
   * @brief Implementation of the F12-HF relaxation.
   *
   * Automatically called by run() after settings have been locked.
   *
   * @param reference The reference wavefunction supplying the orbital basis
   * @return The relaxed F12-HF wavefunction
   */
  virtual std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> reference) const = 0;
};

/**
 * @brief Factory class for creating F12-HF solvers.
 *
 * Mirrors @ref EffectiveHamiltonianConstructorFactory: maintains a registry of
 * implementations identified by string keys, created via create().
 *
 * @see F12HartreeFockSolver
 */
struct F12HartreeFockSolverFactory
    : public AlgorithmFactory<F12HartreeFockSolver,
                              F12HartreeFockSolverFactory> {
  static std::string algorithm_type_name() { return "f12_hartree_fock_solver"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_ct_f12"; }
};

}  // namespace qdk::chemistry::algorithms
