// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class DoubleFactorizationSettings
 * @brief Settings container for DoubleFactorization.
 *
 * @see DoubleFactorization
 */
class DoubleFactorizationSettings : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor that initializes the default settings.
   */
  DoubleFactorizationSettings() = default;
  ~DoubleFactorizationSettings() override = default;
};

/**
 * @class DoubleFactorization
 * @brief Produces a double-factorized Hamiltonian :cite:`vonBurg2021`.
 *
 * @details The input must be a restricted qdk::chemistry::data::Hamiltonian
 * backed by a qdk::chemistry::data::CholeskyHamiltonianContainer.
 *
 * @see qdk::chemistry::data::DFTHCHamiltonianContainer
 */
class DoubleFactorization : public HamiltonianFactorization {
 public:
  /**
   * @brief Default constructor. Uses default DoubleFactorizationSettings.
   */
  DoubleFactorization() {
    _settings = std::make_unique<DoubleFactorizationSettings>();
  }

  /**
   * @brief Virtual destructor.
   */
  ~DoubleFactorization() override = default;

  /**
   * @brief Double-factorize a Hamiltonian.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian Restricted Cholesky Hamiltonian.
   * \endcond
   * @return The double-factorized Hamiltonian.
   *
   * @note Settings are automatically locked when this method is called.
   */
  using HamiltonianFactorization::run;

  /**
   * @brief Access the algorithm's name.
   *
   * @return "double_factorization".
   */
  std::string name() const override { return "double_factorization"; }

 protected:
  /**
   * @brief Factorize the two-electron tensor.
   *
   * @throws std::invalid_argument if the input Hamiltonian or its Cholesky
   *         factors are invalid.
   * @throws std::runtime_error if an eigendecomposition fails.
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian) const override;
};

}  // namespace qdk::chemistry::algorithms
