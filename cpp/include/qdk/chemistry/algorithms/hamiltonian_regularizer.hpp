// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class HamiltonianRegularizer
 * @brief Abstract base class for Hamiltonian regularization/shift algorithms
 *
 * A HamiltonianRegularizer maps a Hamiltonian, together with the target
 * number of alpha/beta electrons, to a new Hamiltonian that is
 * energetically equivalent within the target electron-number sector but
 * whose LCU/qubitization coefficients (e.g. the fermionic 1-norm lambda)
 * may be reduced. Implementations typically apply a symmetry shift built
 * from operators that annihilate every state with the target electron
 * count (e.g. block-invariant symmetry shift / BLISS techniques), so the
 * physical energy of the target-electron-count sector is preserved while
 * the operator's norm outside that sector -- and hence resource estimates
 * for algorithms like qubitized phase estimation -- can shrink.
 *
 * @see data::Hamiltonian
 * @see qdk::chemistry::utils::hamiltonian_one_norm for a standalone,
 *      Algorithm-independent way to inspect a Hamiltonian's fermionic
 *      1-norm without running a regularizer.
 */
class HamiltonianRegularizer
    : public Algorithm<HamiltonianRegularizer,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>, unsigned int,
                       unsigned int> {
 public:
  /**
   * @brief Default constructor
   *
   * Creates a Hamiltonian regularizer with default settings.
   */
  HamiltonianRegularizer() = default;

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of derived classes.
   */
  virtual ~HamiltonianRegularizer() = default;

  /**
   * @brief Regularize/shift a Hamiltonian for a target electron count
   *
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The Hamiltonian to regularize
   * @param n_alpha_electrons The target number of alpha electrons
   * @param n_beta_electrons The target number of beta electrons
   * \endcond
   * @return A new, shifted Hamiltonian that agrees with the input
   *         Hamiltonian's energy in the (n_alpha_electrons,
   *         n_beta_electrons)-electron sector.
   *
   * @throws std::runtime_error if the regularization fails
   * @throws std::invalid_argument if the Hamiltonian is invalid or
   *         unsupported (e.g. unrestricted, when the implementation only
   *         supports restricted Hamiltonians)
   * @throws qdk::chemistry::data::SettingsAreLocked if attempting to modify
   * settings after run() is called
   *
   * @note Settings are automatically locked when this method is called and
   * cannot be modified during or after execution.
   *
   * @see data::Hamiltonian
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name
   *
   * @return The algorithm's type name
   */
  std::string type_name() const final { return "hamiltonian_regularizer"; };

 protected:
  /**
   * @brief Implementation of Hamiltonian regularization
   *
   * This method contains the actual regularization logic. It is
   * automatically called by run() after settings have been locked.
   *
   * @param hamiltonian The Hamiltonian to regularize
   * @param n_alpha_electrons The target number of alpha electrons
   * @param n_beta_electrons The target number of beta electrons
   * @return The regularized/shifted Hamiltonian
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const = 0;
};

/**
 * @brief Factory class for creating Hamiltonian regularizer instances.
 *
 * The HamiltonianRegularizerFactory implements the Factory design pattern to
 * dynamically create and manage different implementations of Hamiltonian
 * regularizers (e.g. different symmetry-shift/BLISS-style strategies).
 *
 * Typical usage:
 * ```
 * auto regularizer =
 *     qdk::chemistry::algorithms::HamiltonianRegularizerFactory::create("flr_bliss");
 * regularizer->settings().set("df_truncation_threshold", 1e-8);
 * auto shifted = regularizer->run(hamiltonian, n_alpha, n_beta);
 * ```
 *
 * @see HamiltonianRegularizer for the interface implemented by concrete
 * regularizers
 */
struct HamiltonianRegularizerFactory
    : public AlgorithmFactory<HamiltonianRegularizer,
                              HamiltonianRegularizerFactory> {
  static std::string algorithm_type_name() {
    return "hamiltonian_regularizer";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "flr_bliss"; }
};

}  // namespace qdk::chemistry::algorithms
