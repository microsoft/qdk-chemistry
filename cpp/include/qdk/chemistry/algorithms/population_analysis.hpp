// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/settings.hpp>
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
 * @class PopulationAnalysisSettings
 * @brief Settings for assigning particle populations to centers.
 */
class PopulationAnalysisSettings : public data::Settings {
 public:
  /**
   * @brief Construct population-analysis settings.
   */
  PopulationAnalysisSettings() {
    set_default("method", std::string("mulliken"),
                "Population-analysis method used to assign per-center "
                "populations.");
    set_default("charge", static_cast<int64_t>(0),
                "Total molecular charge used by structure-only population "
                "analysis implementations.");
    set_default("spin_multiplicity", static_cast<int64_t>(1),
                "Spin multiplicity (2S+1) used by structure-only population "
                "analysis implementations.");
  }
};

/**
 * @class PopulationAnalyzer
 * @brief Base class for assigning particle populations to centers.
 */
class PopulationAnalyzer
    : public Algorithm<PopulationAnalyzer, std::vector<double>,
                       PopulationAnalysisInput> {
 public:
  /**
   * @brief Construct a population analyzer with shared settings.
   */
  PopulationAnalyzer() {
    _settings = std::make_unique<PopulationAnalysisSettings>();
  }
  virtual ~PopulationAnalyzer() = default;

  using Algorithm::run;

  virtual std::string name() const = 0;

  /**
   * @brief Return the factory type name for population analyzers.
   */
  std::string type_name() const final { return "population_analyzer"; }

 protected:
  /**
   * @brief Implementation hook for derived population analyzers.
   */
  virtual std::vector<double> _run_impl(
      PopulationAnalysisInput input) const = 0;
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
  std::vector<double> _run_impl(PopulationAnalysisInput input) const override;
};

}  // namespace qdk::chemistry::algorithms
