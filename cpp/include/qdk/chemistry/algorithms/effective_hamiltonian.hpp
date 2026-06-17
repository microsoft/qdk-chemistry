// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class EffectiveHamiltonianConstructor
 * @brief Abstract base class for constructing effective Hamiltonian operators
 *        from a reference wavefunction.
 *
 * Unlike @ref HamiltonianConstructor, which maps @ref data::Orbitals to a bare
 * @ref data::Hamiltonian, an effective-Hamiltonian constructor maps a reference
 * @ref data::Wavefunction to a dressed @ref data::Hamiltonian. The reference
 * supplies both the orbital basis (via @ref data::Wavefunction::get_orbitals)
 * and the reduced density matrices used by density-driven similarity
 * transformations such as canonical transcorrelated F12.
 *
 * The output has the same shape and quartic complexity as a bare Hamiltonian,
 * so it plugs into every existing downstream solver and qubit mapper unchanged.
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 * @see data::Settings
 */
class EffectiveHamiltonianConstructor
    : public Algorithm<EffectiveHamiltonianConstructor,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  EffectiveHamiltonianConstructor() = default;
  virtual ~EffectiveHamiltonianConstructor() = default;

  /**
   * @brief Construct an effective Hamiltonian from a reference wavefunction.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param reference The reference wavefunction supplying orbitals and RDMs
   * \endcond
   * @return The constructed effective Hamiltonian
   *
   * @throw std::runtime_error if construction fails
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
  std::string type_name() const final {
    return "effective_hamiltonian_constructor";
  };

 protected:
  /**
   * @brief Implementation of effective-Hamiltonian construction.
   *
   * Automatically called by run() after settings have been locked.
   *
   * @param reference The reference wavefunction supplying orbitals and RDMs
   * @return The constructed effective Hamiltonian
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference) const = 0;
};

/**
 * @brief Factory class for creating effective-Hamiltonian constructors.
 *
 * Mirrors @ref HamiltonianConstructorFactory: maintains a registry of
 * implementations identified by string keys, created via create().
 *
 * @see EffectiveHamiltonianConstructor
 */
struct EffectiveHamiltonianConstructorFactory
    : public AlgorithmFactory<EffectiveHamiltonianConstructor,
                              EffectiveHamiltonianConstructorFactory> {
  static std::string algorithm_type_name() {
    return "effective_hamiltonian_constructor";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_ct_f12"; }
};

}  // namespace qdk::chemistry::algorithms
