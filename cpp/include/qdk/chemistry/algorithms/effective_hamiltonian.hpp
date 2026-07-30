// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class EffectiveHamiltonianSettings
 * @brief Common settings for effective-Hamiltonian algorithms.
 *
 * Effective-Hamiltonian builders are configured almost entirely by their data
 * inputs: the reference amplitudes are carried by the full-space input
 * @ref data::Wavefunction and the active space by a separate active-space
 * @ref data::Orbitals argument. The only genuine algorithm configuration is the
 * Baker-Campbell-Hausdorff (BCH) truncation level.
 */
class EffectiveHamiltonianSettings : public data::Settings {
 public:
  EffectiveHamiltonianSettings() {
    set_default("ducc_level", static_cast<int64_t>(2),
                "BCH truncation level: 0 (bare active-space Hamiltonian), "
                "1 (MBPT(2)-consistent), or 2 (MBPT(3)-consistent).",
                data::BoundConstraint<int64_t>{0, 2});
  }
};

/**
 * @class EffectiveHamiltonian
 * @brief Abstract base class for effective-Hamiltonian algorithms.
 *
 * An effective-Hamiltonian algorithm transforms a full-space Hamiltonian into
 * an effective active-space Hamiltonian that folds in dynamical correlation
 * from the external (non-active) orbitals. The DUCC family realizes this via a
 * unitary coupled-cluster similarity transformation evaluated through a
 * truncated BCH expansion.
 *
 * The run signature takes the full-space Hamiltonian, a full-space
 * @ref data::Wavefunction supplying the reference coupled-cluster amplitudes
 * (through an amplitude container), and an active-space @ref data::Orbitals
 * that designates the active orbitals as a subset of the wavefunction's
 * orbitals (through its active-space indices). The alpha/beta occupancy is
 * derived from the wavefunction. Only the BCH truncation level is a setting.
 *
 * Example usage:
 * @code
 * auto builder =
 *     qdk::chemistry::algorithms::create<EffectiveHamiltonian>("ducc");
 * builder->settings().set("ducc_level", static_cast<int64_t>(2));
 * auto effective =
 *     builder->run(full_hamiltonian, ccsd_wavefunction, active_orbitals);
 * @endcode
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 */
class EffectiveHamiltonian
    : public Algorithm<EffectiveHamiltonian, std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Orbitals>> {
 public:
  /**
   * @brief Default constructor installing the shared effective-Hamiltonian
   *        settings.
   */
  EffectiveHamiltonian() {
    _settings = std::make_unique<EffectiveHamiltonianSettings>();
  }

  /**
   * @brief Virtual destructor for proper inheritance.
   */
  virtual ~EffectiveHamiltonian() = default;

  using Algorithm::run;

  /**
   * @brief Access the algorithm's variant name.
   * @return The algorithm's name (e.g. "ducc").
   */
  virtual std::string name() const override = 0;

  /**
   * @brief Access the algorithm's type name.
   * @return The fixed type name "effective_hamiltonian".
   */
  std::string type_name() const final { return "effective_hamiltonian"; }

 protected:
  /**
   * @brief Build the effective active-space Hamiltonian.
   *
   * @param hamiltonian The full-space Hamiltonian to transform.
   * @param wavefunction A full-space wavefunction whose amplitude container
   *        supplies the reference coupled-cluster amplitudes.
   * @param active_orbitals Active-space orbitals whose active-space indices
   *        designate the active subset of @p wavefunction's orbitals.
   * @return The effective active-space Hamiltonian.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<data::Wavefunction> wavefunction,
      std::shared_ptr<data::Orbitals> active_orbitals) const override = 0;
};

/**
 * @brief Factory class for creating effective-Hamiltonian algorithm instances.
 */
struct EffectiveHamiltonianFactory
    : public AlgorithmFactory<EffectiveHamiltonian,
                              EffectiveHamiltonianFactory> {
  static std::string algorithm_type_name() { return "effective_hamiltonian"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "ducc"; }
};

}  // namespace qdk::chemistry::algorithms
