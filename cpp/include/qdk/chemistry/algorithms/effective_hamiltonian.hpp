// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class EffectiveHamiltonianConstructor
 * @brief Abstract base for downfolding an active-space effective Hamiltonian.
 *
 * Given a reference `Wavefunction` (whose occupations/RDMs define the reference
 * density) an input `Hamiltonian` built over the whole downfolding window
 * W = P u Q, and an explicit kept space `P` (a `SymmetryBlockedIndexSet` of
 * window orbital indices), a concrete constructor folds the external space Q
 * into an effective Hamiltonian acting on P. This is a distinct algorithm type
 * from `HamiltonianConstructor` (which builds the bare integral Hamiltonian
 * from `Orbitals`).
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 */
class EffectiveHamiltonianConstructor
    : public Algorithm<EffectiveHamiltonianConstructor,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<const data::SymmetryBlockedIndexSet>> {
 public:
  EffectiveHamiltonianConstructor() = default;
  virtual ~EffectiveHamiltonianConstructor() = default;

  using Algorithm::run;

  /**
   * @brief Access the algorithm's name.
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name.
   */
  std::string type_name() const final {
    return "effective_hamiltonian_constructor";
  }

 protected:
  /**
   * @brief Downfold the window Hamiltonian onto the kept space P.
   *
   * @param reference Reference wavefunction; its occupations/RDMs define the
   *        reference density over the window.
   * @param hamiltonian Input Hamiltonian built over the whole window W = P u Q.
   * @param p_indices The kept space P as a `SymmetryBlockedIndexSet` of
   *        global (spatial) orbital indices into the window Hamiltonian's
   *        active space W.
   * @return The effective Hamiltonian acting on P.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const = 0;
};

/**
 * @brief Factory for creating effective-Hamiltonian constructor instances.
 *
 * @see EffectiveHamiltonianConstructor
 */
struct EffectiveHamiltonianConstructorFactory
    : public AlgorithmFactory<EffectiveHamiltonianConstructor,
                              EffectiveHamiltonianConstructorFactory> {
  static std::string algorithm_type_name() {
    return "effective_hamiltonian_constructor";
  }
  // Concrete variants (swpt2, ...) register here (effective_hamiltonian.cpp).
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_swpt2"; }
};

}  // namespace qdk::chemistry::algorithms
