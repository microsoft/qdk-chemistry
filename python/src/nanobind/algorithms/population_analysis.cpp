// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;

class PopulationAnalyzerBase : public PopulationAnalyzer {
 public:
  NB_TRAMPOLINE(PopulationAnalyzer);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::vector<double> _run_impl(
      std::shared_ptr<qdk::chemistry::data::Wavefunction> wavefunction)
      const override {
    NB_OVERRIDE_PURE(_run_impl, wavefunction);
  }
};

void bind_population_analysis(nb::module_& m) {
  nb::class_<PopulationAnalyzer, PopulationAnalyzerBase> analyzer(
      m, "PopulationAnalyzer",
      R"(
    Base class for particle population analysis algorithms.

    Population analyzers take a Wavefunction and return a list of per-center
    populations in center order.
    )");

  analyzer.def(nb::init<>(), R"(Create a population analyzer.)");
  analyzer.def(
      "run",
      [](const PopulationAnalyzer& self,
         std::shared_ptr<qdk::chemistry::data::Wavefunction> wavefunction) {
        return self.run(wavefunction);
      },
      nb::arg("input"),
      R"(
Compute per-center populations.

Args:
  input: Wavefunction to analyze.

Returns:
    list[float]: Per-center populations in center order.
)");
  analyzer.def(
      "hash", &PopulationAnalyzer::hash, nb::arg("input"),
      R"(Compute a deterministic content hash for a population analysis run.)");
  analyzer.def("settings", &PopulationAnalyzer::settings,
               nb::rv_policy::reference_internal,
               R"(Return the analyzer settings.)");
  analyzer.def_prop_rw(
      "_settings",
      [](PopulationAnalyzer& self) -> qdk::chemistry::data::Settings& {
        return self.settings();
      },
      [](PopulationAnalyzer& self,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<PopulationAnalyzerBase&>(self).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
      R"(Internal settings replacement hook for Python subclasses.)");
  analyzer.def("name", &PopulationAnalyzer::name,
               R"(Return the implementation name.)");
  analyzer.def("aliases", &PopulationAnalyzer::aliases,
               R"(Return all registered names for the implementation.)");
  analyzer.def("type_name", &PopulationAnalyzer::type_name,
               R"(Return the algorithm type name.)");
  analyzer.def("__repr__", [](const PopulationAnalyzer& self) {
    return "<qdk_chemistry.algorithms.PopulationAnalyzer name='" + self.name() +
           "'>";
  });

  qdk::chemistry::python::bind_algorithm_factory<
      PopulationAnalyzerFactory, PopulationAnalyzer, PopulationAnalyzerBase>(
      m, "PopulationAnalyzerFactory");
  qdk::chemistry::python::bind_create_nested(analyzer);

  nb::class_<QdkPopulationAnalyzer, PopulationAnalyzer>(
      m, "QdkPopulationAnalyzer", R"(Internal QDK population analyzer.)")
      .def(nb::init<>(), R"(Create a QDK population analyzer.)");
}
