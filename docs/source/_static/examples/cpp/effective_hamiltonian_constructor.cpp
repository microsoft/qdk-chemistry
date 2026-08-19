// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Effective Hamiltonian constructor usage examples.
#include <iostream>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// --------------------------------------------------------------------------------------------
// start-cell-list-implementations
auto names = EffectiveHamiltonianConstructorFactory::available();
for (const auto& name : names) {
  std::cout << name << std::endl;
}
// end-cell-list-implementations
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
// Create a second-order Schrieffer-Wolff downfolder
auto downfolder = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Weaken the denominator regularization, and discard rather than fold the
// three-body terms the transformation generates
downfolder->settings().set("regularizer_sigma2", 4.0);
downfolder->settings().set("fold_above_two_body", false);
// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// docs:xyz ../data/water.structure.xyz
// start-cell-downfold
auto structure = Structure::from_xyz(R"(3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
)");

auto [E_scf, wfn] = ScfSolverFactory::create()->run(structure, 0, 1, "sto-3g");

// Assign an active space to the mean-field reference
auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
selector->settings().set("num_active_electrons", 6);
selector->settings().set("num_active_orbitals", 5);
auto reference = selector->run(wfn);

// Build the window Hamiltonian from the pre-selection orbitals, so that every
// orbital of W is active
auto window_hamiltonian =
    HamiltonianConstructorFactory::create()->run(wfn->get_orbitals());

// Keep the reference active space as P and fold the rest of the window into it
auto p_indices = reference->get_orbitals()->active_indices();

auto effective_hamiltonian =
    downfolder->run(reference, window_hamiltonian, p_indices);

std::cout << effective_hamiltonian->get_summary() << std::endl;
// end-cell-downfold
// --------------------------------------------------------------------------------------------
