// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Factory Pattern usage examples.
// -----------------------------------------------------------------------------
// start-cell-scf-localizer
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;

// Create default implementation
auto scf_solver = ScfSolverFactory::create();

// Create specific implementation by name
auto localizer = LocalizerFactory::create("pipek-mezey");

// Configure and use the instance
scf_solver->settings().set("basis_set", "def2-tzvp");
auto [E_scf, orbitals] = scf_solver->solve(structure);
// end-cell-scf-localizer
// -----------------------------------------------------------------------------
