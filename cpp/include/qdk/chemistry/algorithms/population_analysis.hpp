// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms {

/**
 * @class PopulationAnalysisSettings
 * @brief Common settings for particle-population analysis.
 */
class PopulationAnalysisSettings : public data::Settings {
 public:
  PopulationAnalysisSettings() {
    set_default("method", std::string("mulliken"),
                "Particle-population analysis method",
                data::ListConstraint<std::string>{{"mulliken"}});
  }
};

/**
 * @class PopulationAnalyzer
 * @brief Base class for assigning particle populations to centers.
 */
class PopulationAnalyzer
    : public Algorithm<PopulationAnalyzer, std::vector<double>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  PopulationAnalyzer() {
    _settings = std::make_unique<PopulationAnalysisSettings>();
  }

  virtual ~PopulationAnalyzer() = default;

  /**
   * @brief Compute per-center particle populations.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param wavefunction Wavefunction to analyze
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
   * @param wavefunction Wavefunction to analyze
   * @return Per-center populations in center order
   */
  virtual std::vector<double> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const = 0;
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

 protected:
  /**
   * @brief Compute per-center populations.
   */
  std::vector<double> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const override;
};

}  // namespace qdk::chemistry::algorithms
