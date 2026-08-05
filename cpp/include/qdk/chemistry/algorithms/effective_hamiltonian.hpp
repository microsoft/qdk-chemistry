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
  virtual std::string name() const override = 0;

  /**
   * @brief Access the algorithm's type name.
   */
  std::string type_name() const override {
    return "effective_hamiltonian_constructor";
  }

 protected:
  /**
   * @brief Construct an effective Hamiltonian from a reference wavefunction.
   *
   * @param reference Reference wavefunction providing the reference state.
   * @param hamiltonian Input Hamiltonian to transform.
   * @param p_space_indices Target P-space orbital indices.
   * @return The effective Hamiltonian.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_space_indices)
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
