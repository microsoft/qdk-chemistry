// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <variant>
#include <vector>

namespace qdk::chemistry::algorithms {

/**
 * @brief Structure or wavefunction input for population analysis.
 */
using PopulationAnalysisInput =
    std::variant<std::shared_ptr<data::Structure>,
                 std::shared_ptr<data::Wavefunction>>;

/**
 * @class PopulationAnalyzer
 * @brief Base class for assigning particle populations to centers.
 */
class PopulationAnalyzer
    : public Algorithm<PopulationAnalyzer, std::vector<double>,
                       PopulationAnalysisInput, int, int, unsigned int> {
 public:
  virtual ~PopulationAnalyzer() = default;

  /**
   * @brief Compute per-center particle populations.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param input Structure or wavefunction to analyze
   * @param charge Total molecular charge
   * @param spin_multiplicity Spin multiplicity of the molecular system
   * @param n_inactive_orbitals Number of doubly occupied orbitals excluded
   * from active-space treatments; full-population analyses ignore this value
   * \endcond
   *
   * @return Per-center populations in center order.
   */
  using Algorithm::run;

  virtual std::string name() const = 0;

  /**
   * @brief Return the factory type name for population analyzers.
   */
  std::string type_name() const final { return "population_analyzer"; }

 protected:
  /**
   * @brief Implementation hook for derived population analyzers.
   *
   * @param input Structure or wavefunction to analyze
   * @param charge Total molecular charge
   * @param spin_multiplicity Spin multiplicity of the molecular system
   * @param n_inactive_orbitals Number of doubly occupied orbitals excluded
   * from active-space treatments; full-population analyses ignore this value
   * @return Per-center populations in center order
   */
  virtual std::vector<double> _run_impl(
      PopulationAnalysisInput input, int charge, int spin_multiplicity,
      unsigned int n_inactive_orbitals) const = 0;
};

/**
 * @brief Factory for population analyzer implementations.
 */
struct PopulationAnalyzerFactory
    : public AlgorithmFactory<PopulationAnalyzer, PopulationAnalyzerFactory> {
  /**
   * @brief Return the algorithm type name managed by this factory.
   */
  static std::string algorithm_type_name() { return "population_analyzer"; }

  /**
   * @brief Register built-in population analyzer implementations.
   */
  static void register_default_instances();

  /**
   * @brief Return the default population analyzer implementation name.
   */
  static std::string default_algorithm_name() { return "qdk"; }
};

/**
 * @class QdkPopulationAnalyzer
 * @brief Internal QDK population analyzer.
 */
class QdkPopulationAnalyzer : public PopulationAnalyzer {
 public:
  /**
   * @brief Return the implementation name.
   */
  std::string name() const final { return "qdk"; }

  /**
   * @brief Return accepted aliases for this implementation.
   */
  std::vector<std::string> aliases() const final {
    return {"qdk", "internal", "mulliken"};
  }

 protected:
  /**
   * @brief Compute per-center populations.
   */
  std::vector<double> _run_impl(
      PopulationAnalysisInput input, int charge, int spin_multiplicity,
      unsigned int n_inactive_orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms
