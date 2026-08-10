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
 * @c p_indices holds absolute molecular-orbital indices, drawn from the same
 * index universe as @c data::Orbitals::active_indices().
 *
 * The returned Hamiltonian is expressed over P and must satisfy:
 * - its orbitals have @c active_indices() equal to @c p_indices;
 * - its orbitals carry the input Hamiltonian's @c inactive_indices()
 *   unchanged, so a wavefunction later solved in P stays consistent with it;
 * - the input Hamiltonian's inactive Fock matrix, when present, is carried
 *   over unchanged: it spans the full MO space and is fixed by the inactive
 *   density, neither of which downfolding changes;
 * - the orbitals of @f$Q = W \setminus P@f$ are left unclassified rather than
 *   marked inactive, because @ref data::Hamiltonian assumes inactive orbitals
 *   are fully occupied while Q generally also spans virtuals;
 * - the scalar shift from folding in Q is added to the constant core energy
 *   term, and the remaining Q contribution is folded into the integrals.
 *
 * Input validation is opt-in. The base @ref run method does not validate its
 * arguments; concrete implementations decide whether to call
 * @ref _validate_inputs.
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
   * @param hamiltonian Input Hamiltonian built over the whole window @f$W = P
   * \cup Q@f$.
   * @param p_indices Absolute molecular-orbital indices of the target space P,
   *        which must lie within the reference wavefunction's active space.
   * \endcond
   * @return The effective Hamiltonian acting on the target space P, following
   *         the output contract documented on this class.
   * @throws qdk::chemistry::data::SettingsAreLocked if attempting to modify
   *         settings after run() is called.
   * @note Settings are automatically locked when this method is called and
   *       cannot be modified during or after execution.
   * @note This method performs no input validation of its own. Any argument
   *       checking is the responsibility of the concrete implementation.
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
   * @param p_indices Target P-space as absolute molecular-orbital indices,
   *        which must be a subset of the reference wavefunction's active
   *        orbital space.
   * @throws std::invalid_argument if an input is null, the Hamiltonian and
   *         wavefunction use incompatible orbital bases or spin restrictions,
   *         or the spaces do not satisfy @f$P \subseteq W_{\mathrm{ref}}
   * \subseteq W_H@f$.
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
   * @param hamiltonian Input Hamiltonian built over the whole window @f$W = P
   * \cup Q@f$.
   * @param p_indices Absolute molecular-orbital indices of the target space P,
   *        which must lie within the reference wavefunction's active space.
   * @return The effective Hamiltonian acting on the target space P. It must
   *         satisfy the output contract documented on this class.
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
