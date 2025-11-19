// Localizer usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// start-cell-1
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;

// Create an MP2 natural orbital localizer
auto mp2_localizer = LocalizerFactory::create("mp2_natural_orbitals");
// end-cell-1

// start-cell-2
// Set the convergence threshold
localizer->settings().set("tolerance", 1.0e-6);
// end-cell-2

// start-cell-3
// Obtain a valid Orbitals instance
Orbitals orbitals;
/* orbitals = ... */

// Configure electron counts in settings for methods that require them
localizer->settings().set("n_alpha_electrons", n_alpha);
localizer->settings().set("n_beta_electrons", n_beta);

// Create indices for orbitals to localize
std::vector<size_t> loc_indices_a = {0, 1, 2, 3};  // Alpha orbital indices
std::vector<size_t> loc_indices_b = {0, 1, 2, 3};  // Beta orbital indices

// Localize the specified orbitals
auto localized_orbitals =
    localizer->run(orbitals, loc_indices_a, loc_indices_b);
// end-cell-3
