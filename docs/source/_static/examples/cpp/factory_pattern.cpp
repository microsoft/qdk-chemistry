// Factory Pattern usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-scf-localizer
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create a simple molecule
auto structure = Structure::from_xyz("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4");

// Create an SCF solver using the default implementation
auto scf_solver = ScfSolverFactory::create();

// Create an orbital localizer using a specific implementation
auto localizer = LocalizerFactory::create("qdk_pipek_mezey");

// Configure the SCF solver and run
scf_solver->settings().set("basis_set", "cc-pvdz");
auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);
// end-cell-scf-localizer
// -----------------------------------------------------------------------------
