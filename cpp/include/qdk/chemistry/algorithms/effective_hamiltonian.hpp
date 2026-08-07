// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class EffectiveHamiltonianConstructor
 * @brief Abstract base for constructing an effective Hamiltonian.
 *
 * Given a reference wavefunction and an input Hamiltonian, a concrete
 * implementation constructs an effective Hamiltonian in an explicitly
 * specified target P-space.
 *
 * Typical usage:
 * @code
 * auto constructor =
 *   EffectiveHamiltonianConstructorFactory::create("algorithm_name");
 * auto effective_hamiltonian =
 *   constructor->run(reference, hamiltonian, p_indices);
 * @endcode
 *
 * @see EffectiveHamiltonianConstructorFactory for creating instances
 * @see data::Wavefunction for the reference wavefunction input
 * @see data::Hamiltonian for the input and output Hamiltonians
 * @see data::SymmetryBlockedIndexSet for the target P-space indices
 */
class EffectiveHamiltonianConstructor
    : public Algorithm<EffectiveHamiltonianConstructor,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<const data::SymmetryBlockedIndexSet>> {
 public:
  /**
   * @brief Default constructor.
   */
  EffectiveHamiltonianConstructor() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~EffectiveHamiltonianConstructor() = default;

  /**
   * @brief Construct the effective Hamiltonian acting on the target space P.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param reference Reference wavefunction providing the reference state.
   * @param hamiltonian Input Hamiltonian built over the whole window W = P u Q.
   * @param p_indices The target space P within the reference wavefunction's
   *        active orbital space.
   * \endcond
   * @return The effective Hamiltonian acting on the target space P.
   * @throws qdk::chemistry::data::SettingsAreLocked if attempting to modify
   *         settings after run() is called.
   * @note Settings are automatically locked when this method is called and
   *       cannot be modified during or after execution.
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name.
   */
  virtual std::string name() const override = 0;

  /**
   * @brief Access the algorithm's type name.
   */
  std::string type_name() const override {
    return "effective_hamiltonian_constructor";
  }

 protected:
  /**
   * @brief Validate the common nested-space input contract.
   *
   * Concrete implementations may call this helper before performing
   * method-specific validation or computation. Validation is opt-in; the base
   * @ref run method does not call it automatically.
   *
   * @param reference Reference wavefunction whose active orbital space must be
   *        a subset of the Hamiltonian's active orbital window.
   * @param hamiltonian Input Hamiltonian defining the outer orbital window.
   * @param p_indices Target P-space, which must be a subset of the reference
   *        wavefunction's active orbital space.
   * @throws std::invalid_argument if an input is null, the Hamiltonian and
   *         wavefunction use incompatible orbital bases or spin restrictions,
   *         or the spaces do not satisfy P subset W_ref subset W_H.
   */
  void _validate_inputs(
      const std::shared_ptr<data::Wavefunction>& reference,
      const std::shared_ptr<data::Hamiltonian>& hamiltonian,
      const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices)
      const;

  /**
   * @brief Implementation of the effective-Hamiltonian construction.
   *
   * Contains the actual construction logic. It is automatically called by
   * run() after settings have been locked, and must be implemented by derived
   * classes.
   *
   * @param reference Reference wavefunction providing the reference state.
   * @param hamiltonian Input Hamiltonian built over the whole window W = P u Q.
   * @param p_indices The target space P within the reference wavefunction's
   *        active orbital space.
   * @return The effective Hamiltonian acting on the target space P.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices)
      const override = 0;
};

/**
 * @brief Factory for effective-Hamiltonian constructor instances.
 */
struct EffectiveHamiltonianConstructorFactory
    : public AlgorithmFactory<EffectiveHamiltonianConstructor,
                              EffectiveHamiltonianConstructorFactory> {
  static std::string algorithm_type_name() {
    return "effective_hamiltonian_constructor";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return ""; }
};

}  // namespace qdk::chemistry::algorithms
