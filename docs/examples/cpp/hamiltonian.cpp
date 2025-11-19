// Hamiltonian usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// start-cell-1
// Create a Hamiltonian constructor
// Returns std::shared_ptr<HamiltonianConstructor>
auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

// Set active orbitals if needed
std::vector<size_t> active_orbitals = {4, 5, 6, 7};  // Example indices
hamiltonian_constructor->settings().set("active_orbitals", active_orbitals);

// Construct the Hamiltonian from orbitals
// Returns Hamiltonian
auto hamiltonian = hamiltonian_constructor->run(orbitals);

// Alternatively, create a Hamiltonian directly
Hamiltonian direct_hamiltonian(one_body_integrals, two_body_integrals, orbitals,
                               selected_orbital_indices, num_electrons,
                               core_energy);
// end-cell-1

// start-cell-2
// Access one-electron integrals, returns const Eigen::MatrixXd&
auto h1 = hamiltonian.get_one_body_integrals();

// Access two-electron integrals, returns const Eigen::VectorXd&
auto h2 = hamiltonian.get_two_body_integrals();

// Access a specific two-electron integral <ij|kl>
double element = hamiltonian.get_two_body_element(i, j, k, l);

// Get core energy (nuclear repulsion + inactive orbital energy), returns double
auto core_energy = hamiltonian.get_core_energy();

// Get inactive Fock matrix (if available), returns const Eigen::MatrixXd&
if (hamiltonian.has_inactive_fock_matrix()) {
  auto inactive_fock = hamiltonian.get_inactive_fock_matrix();
}

// Get orbital data, returns const Orbitals&
const auto& orbitals = hamiltonian.get_orbitals();

// Get active space information, returns const std::vector<size_t>&
auto active_indices = hamiltonian.get_selected_orbital_indices();
// Returns size_t
auto num_electrons = hamiltonian.get_num_electrons();
// Returns size_t
auto num_orbitals = hamiltonian.get_num_orbitals();
// end-cell-2

// start-cell-3
// Serialize to JSON file
hamiltonian.to_json_file("molecule.hamiltonian.json");

// Deserialize from JSON file
auto hamiltonian_from_json_file =
    Hamiltonian::from_json_file("molecule.hamiltonian.json");

// Serialize to HDF5 file
hamiltonian.to_hdf5_file("molecule.hamiltonian.h5");

// Deserialize from HDF5 file
auto hamiltonian_from_hdf5_file =
    Hamiltonian::from_hdf5_file("molecule.hamiltonian.h5");

// Generic file I/O based on type parameter
hamiltonian.to_file("molecule.hamiltonian.json", "json");
auto hamiltonian_from_file =
    Hamiltonian::from_file("molecule.hamiltonian.h5", "hdf5");

// Convert to JSON object
// Returns nlohmann::json
nlohmann::json j = hamiltonian.to_json();

// Load from JSON object
auto hamiltonian_from_json = Hamiltonian::from_json(j);
// end-cell-3

// start-cell-4
// Check if the Hamiltonian data is complete and consistent
// Returns bool
bool valid = hamiltonian.is_valid();

// Check if specific components are available
// All return bool
bool has_one_body = hamiltonian.has_one_body_integrals();
bool has_two_body = hamiltonian.has_two_body_integrals();
bool has_orbitals = hamiltonian.has_orbitals();
bool has_inactive_fock = hamiltonian.has_inactive_fock_matrix();
// end-cell-4
