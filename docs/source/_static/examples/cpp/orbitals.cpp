// Orbitals usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
// Obtain orbitals from an SCF calculation
auto scf_solver = ScfSolverFactory::create();
auto [E_scf, orbitals] = scf_solver->solve(structure);

// set coefficients manually example (restricted)
Orbitals orbs_manual;
Eigen::MatrixXd coeffs = /* coefficient matrix */;
orbs_manual.set_coefficients(coeffs);  // Same for alpha and beta

// set coefficients manually example (unrestricted)
Orbitals orbs_unrestricted;
Eigen::MatrixXd coeffs_alpha = /* alpha coefficients */;
Eigen::MatrixXd coeffs_beta = /* beta coefficients */;
orbs_unrestricted.set_coefficients(coeffs_alpha, coeffs_beta);
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-access
// Access orbital coefficients (returns std::pair<const Eigen::MatrixXd&, const
// Eigen::MatrixXd&>)
auto [coeffs_alpha, coeffs_beta] = orbitals.get_coefficients();

// Access orbital energies (returns std::pair<const Eigen::VectorXd&, const
// Eigen::VectorXd&>)
auto [energies_alpha, energies_beta] = orbitals.get_energies();

// Access orbital occupations (returns std::pair<const Eigen::VectorXd&, const
// Eigen::VectorXd&>)
auto [occs_alpha, occs_beta] = orbitals.get_occupations();

// Access atomic orbital overlap matrix (returns const Eigen::MatrixXd&)
const auto& ao_overlap = orbitals.get_overlap_matrix();

// Access basis set information (returns const BasisSet&)
const auto& basis_set = orbitals.get_basis_set();

// Check calculation type
bool is_restricted = orbitals.is_restricted();
bool is_open_shell = orbitals.is_open_shell();

// Get size information
size_t num_molecular_orbitals = orbitals.get_num_molecular_orbitals();
size_t num_atomic_orbitals = orbitals.get_num_atomic_orbitals();
auto [n_electrons_alpha, n_electrons_beta] =
    orbitals.get_num_electrons();  // returns std::pair<double, double>

std::string summary = orbitals.get_summary();
std::cout << summary << std::endl;
// end-cell-access
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-serialization
// Generic serialization with format specification
orbitals.to_file("molecule.orbitals.json", "json");
auto orbitals_from_file = Orbitals::from_file("molecule.orbitals.json", "json");

// JSON serialization
orbitals.to_json_file("molecule.orbitals.json");
auto orbitals_from_json_file =
    Orbitals::from_json_file("molecule.orbitals.json");

// Direct JSON conversion
nlohmann::json j = orbitals.to_json();
auto orbitals_from_json = Orbitals::from_json(j);

// HDF5 serialization
orbitals.to_hdf5_file("molecule.orbitals.h5");
auto orbitals_from_hdf5_file = Orbitals::from_hdf5_file("molecule.orbitals.h5");
// end-cell-serialization
// --------------------------------------------------------------------------------------------
