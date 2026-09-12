// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class HamiltonianFactorization
 * @brief Abstract base class for Hamiltonian factorization algorithms.
 *
 * Implementations transform a Hamiltonian into a factorized representation
 * while preserving the physical operator.
 */
class HamiltonianFactorization
    : public Algorithm<HamiltonianFactorization,
                       std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>> {
 public:
  HamiltonianFactorization() = default;
  virtual ~HamiltonianFactorization() = default;

  using Algorithm::run;

  /**
   * @brief Access the implementation name.
   */
  virtual std::string name() const override = 0;

  /**
   * @brief Access the algorithm type name.
   */
  std::string type_name() const final { return "hamiltonian_factorization"; }

 protected:
  /**
   * @brief Factorize a Hamiltonian.
   *
   * @param hamiltonian Hamiltonian to factorize.
   * @return The factorized Hamiltonian.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian) const override = 0;
};

/**
 * @brief Factory for Hamiltonian factorization implementations.
 */
struct HamiltonianFactorizationFactory
    : public AlgorithmFactory<HamiltonianFactorization,
                              HamiltonianFactorizationFactory> {
  static std::string algorithm_type_name() {
    return "hamiltonian_factorization";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "double_factorization"; }
};

}  // namespace qdk::chemistry::algorithms
