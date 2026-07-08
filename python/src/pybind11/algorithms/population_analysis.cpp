// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;

class PopulationAnalyzerBase : public PopulationAnalyzer,
                               public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, PopulationAnalyzer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, PopulationAnalyzer, aliases);
  }

  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::vector<double> _run_impl(PopulationAnalysisInput input) const override {
    PYBIND11_OVERRIDE_PURE(std::vector<double>, PopulationAnalyzer, _run_impl,
                           input);
  }
};

void bind_population_analysis(py::module& m) {
  py::class_<PopulationAnalyzer, PopulationAnalyzerBase, py::smart_holder>
      analyzer(m, "PopulationAnalyzer",
               R"(
    Base class for particle population analysis algorithms.

    Population analyzers take either a Structure or Wavefunction and return a
    list of per-center populations in center order.
    )");

  analyzer.def(py::init<>(), R"(Create a population analyzer.)");
  analyzer.def(
      "run",
      [](const PopulationAnalyzer& self, PopulationAnalysisInput input) {
        return self.run(input);
      },
      py::arg("input"),
      R"(
Compute per-center populations.

Args:
    input: Structure or Wavefunction to analyze.

Returns:
    list[float]: Per-center populations in center order.
)");
  analyzer.def("settings", &PopulationAnalyzer::settings,
               py::return_value_policy::reference_internal,
               R"(Return the analyzer settings.)");
  analyzer.def_property(
      "_settings",
      [](PopulationAnalyzerBase& self) -> qdk::chemistry::data::Settings& {
        return self.settings();
      },
      [](PopulationAnalyzerBase& self,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        self.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(Internal settings replacement hook for Python subclasses.)");
  analyzer.def("name", &PopulationAnalyzer::name,
               R"(Return the implementation name.)");
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

  py::class_<QdkPopulationAnalyzer, PopulationAnalyzer, py::smart_holder>(
      m, "QdkPopulationAnalyzer", R"(Internal QDK population analyzer.)")
      .def(py::init<>(), R"(Create a QDK population analyzer.)");
}
