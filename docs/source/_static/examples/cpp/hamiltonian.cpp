// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Hamiltonian usage examples.
// --------------------------------------------------------------------------------------------
// start-cell-hamiltonian-creation
// Load structure from XYZ file
auto structure = Structure::from_xyz_file("../data/water.structure.xyz");

// Run initial SCF to get orbitals
auto scf_solver = ScfSolverFactory::create();
auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);
auto orbitals = wfn->get_orbitals();

// Create a Hamiltonian constructor
// Returns std::shared_ptr<HamiltonianConstructor>
auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

// Construct the Hamiltonian from orbitals
auto hamiltonian = hamiltonian_constructor->run(orbitals);
// end-cell-hamiltonian-creation
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-properties
// Example indices for one- and two-body integral access
size_t i = 0;
size_t j = 1;
size_t k = 2;
size_t l = 3;

// Access one-electron integrals, returns tuple of const Eigen::MatrixXd&
auto [h1_a, h1_b] = hamiltonian.get_one_body_integrals();

// Access two-electron integrals, returns triple of const Eigen::VectorXd&
auto [h2_aaaa, h2_aabb, h2_bbbb] = hamiltonian.get_two_body_integrals();

// Access a specific one-electron integral <ij> (for aa spin channel)
double one_body_element =
    hamiltonian.get_one_body_element(i, j, SpinChannel::aa);

// Access a specific two-electron integral <ij|kl> (for aaaa spin channel)
double two_body_element =
    hamiltonian.get_two_body_element(i, j, k, l, SpinChannel::aaaa);

// Get core energy (nuclear repulsion + inactive orbital energy), returns double
auto core_energy = hamiltonian.get_core_energy();

// Get orbital data, returns const Orbitals&
const auto& orbitals_access = hamiltonian.get_orbitals();
// end-cell-properties
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-validation
// Check if the Hamiltonian data is complete and consistent
// Returns bool
bool valid = hamiltonian.is_valid();

// Check if specific components are available
// All return bool
bool has_one_body = hamiltonian.has_one_body_integrals();
bool has_two_body = hamiltonian.has_two_body_integrals();
bool has_orbitals = hamiltonian.has_orbitals();
// end-cell-validation
// --------------------------------------------------------------------------------------------
