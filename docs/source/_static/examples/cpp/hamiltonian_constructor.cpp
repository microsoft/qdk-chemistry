// Hamiltonian Constructor usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;

// --------------------------------------------------------------------------------------------
// start-cell-create
// Create the default HamiltonianConstructor instance
auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Specify active orbitals for active space Hamiltonian
std::vector<int> active_orbitals = {4, 5, 6, 7};  // Example indices (0-based)
hamiltonian_constructor->settings().set("active_orbitals", active_orbitals);
// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-construct
// Obtain a valid Orbitals instance
Orbitals orbitals;
/* orbitals = ... */

// Construct the Hamiltonian
auto hamiltonian = hamiltonian_constructor->run(orbitals);

// Access the resulting integrals
auto h1 = hamiltonian.get_one_body_integrals();
auto h2 = hamiltonian.get_two_body_integrals();
// end-cell-construct
// --------------------------------------------------------------------------------------------
