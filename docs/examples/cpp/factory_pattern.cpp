// Factory Pattern usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// start-cell-1
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;

// Create default implementation
auto scf_solver = ScfSolverFactory::create();

// Create specific implementation by name
auto localizer = LocalizerFactory::create("pipek-mezey");

// Configure and use the instance
scf_solver->settings().set("basis_set", "def2-tzvp");
auto [E_scf, orbitals] = scf_solver->solve(structure);
// end-cell-1

// start-cell-2
#include <external_program/api.h>  // Your external program's API

#include <qdk/chemistry/algorithms/scf_solver.hpp>

namespace my_namespace {

class ExternalProgramScfSolver : public qdk::chemistry::algorithms::ScfSolver {
 public:
  ExternalProgramScfSolver() = default;
  ~ExternalProgramScfSolver() override = default;

  // Implement the interface method that connects to your external program
  std::tuple<double, qdk::chemistry::data::Orbitals> solve(
      const qdk::chemistry::data::Structure& structure) override {
    // Convert QDK/Chemistry structure to external program format
    auto ext_molecule = convert_to_external_format(structure);

    // Run calculation using external program's API
    auto ext_results =
        external_program::run_scf(ext_molecule, settings().get_all());

    // Convert results back to QDK/Chemistry format
    double energy = ext_results.energy;
    qdk::chemistry::data::Orbitals orbitals =
        convert_from_external_format(ext_results.orbitals);

    return {energy, orbitals};
  }

 private:
  // Helper functions for format conversion
  external_program::Molecule convert_to_external_format(
      const qdk::chemistry::data::Structure& structure);
  qdk::chemistry::data::Orbitals convert_from_external_format(
      const external_program::Orbitals& ext_orbitals);
};

}  // namespace my_namespace
// end-cell-2
