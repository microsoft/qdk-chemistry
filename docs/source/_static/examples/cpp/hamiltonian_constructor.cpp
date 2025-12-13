// Hamiltonian Constructor usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------
#include <iomanip>
#include <iostream>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
// --------------------------------------------------------------------------------------------
// start-cell-create
// Create the default HamiltonianConstructor instance
auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Configure settings (check available options)
// Note: Available settings can be inspected at runtime

// Set ERI method if needed
hamiltonian_constructor->settings().set("eri_method", "direct");
// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-construct
// Create a structure
std::vector<Eigen::Vector3d> coords = {Eigen::Vector3d{0.0, 0.0, 0.0},
                                       Eigen::Vector3d{0.0, 0.0, 1.4}};
std::vector<std::string> symbols = {"H", "H"};
auto structure = std::make_shared<Structure>(coords, symbols);

// Run a SCF to get orbitals
auto scf_solver = ScfSolverFactory::create();
scf_solver->settings().set("basis_set", "sto-3g");
auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);
auto orbitals = wfn->get_orbitals();

// Construct the Hamiltonian from orbitals
auto hamiltonian = hamiltonian_constructor->run(orbitals);

// Access the resulting integrals
auto [h1_a, h1_b] = hamiltonian->get_one_body_integrals();
auto [h2_aaaa, h2_aabb, h2_bbbb] = hamiltonian->get_two_body_integrals();
auto core_energy = hamiltonian->get_core_energy();

std::cout << "One-body integrals shape: " << h1_a.rows() << "x" << h1_a.cols()
          << std::endl;
std::cout << "Two-body integrals shape: " << h2_aaaa.dimension(0) << "x"
          << h2_aaaa.dimension(1) << "x" << h2_aaaa.dimension(2) << "x"
          << h2_aaaa.dimension(3) << std::endl;
std::cout << "Core energy: " << std::fixed << std::setprecision(10)
          << core_energy << " Hartree" << std::endl;
std::cout << hamiltonian->get_summary() << std::endl;
// end-cell-construct
// --------------------------------------------------------------------------------------------
