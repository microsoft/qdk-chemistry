// Basis Set usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-basis-set-create
// Create an empty basis set with a name
BasisSet basis_set("6-31G", AOType::Spherical);

// Add a shell with multiple primitives
size_t atom_index = 0;                      // First atom
OrbitalType orbital_type = OrbitalType::P;  // p orbital
Eigen::VectorXd exponents(2);
exponents << 0.16871439, 0.62391373;
Eigen::VectorXd coefficients(2);
coefficients << 0.43394573, 0.56604777;
basis_set.add_shell(atom_index, orbital_type, exponents, coefficients);

// Add a shell with a single primitive
basis_set.add_shell(1, OrbitalType::S, 0.5, 1.0);

// Set molecular structure
basis_set.set_structure(structure);
// end-cell-basis-set-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-basis-set-get
// Get basis set type and name (returns AOType)
auto atomic_orbital_type = basis_set.get_atomic_orbital_type();
// Get basis set name (returns std::string)
auto name = basis_set.get_name();

// Get all shells (returns const std::vector<Shell>&)
auto all_shells = basis_set.get_shells();
// Get shells for specific atom (returns const std::vector<const Shell>&)
auto shells_for_atom = basis_set.get_shells_for_atom(0);
// Get specific shell by index (returns const Shell&)
const Shell& specific_shell = basis_set.get_shell(3);

// Get counts
size_t num_shells = basis_set.get_num_shells();
size_t num_atomic_orbitals = basis_set.get_num_atomic_orbitals();
size_t num_atoms = basis_set.get_num_atoms();

// Get atomic orbital information (returns std::pair<size_t, int>)
auto [shell_index, m_quantum_number] = basis_set.get_atomic_orbital_info(5);
size_t atom_index = basis_set.get_atom_index_for_atomic_orbital(5);

// Get indices for specific atoms or orbital types
// Returns std::vector<size_t>
auto atomic_orbital_indices = basis_set.get_atomic_orbital_indices_for_atom(1);
// Returns std::vector<size_t>
auto shell_indices =
    basis_set.get_shell_indices_for_orbital_type(OrbitalType::P);
// Returns std::vector<size_t>
auto shell_indices_specific =
    basis_set.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::D);

// Validation
bool is_valid = basis_set.is_valid();
bool is_consistent = basis_set.is_consistent_with_structure();
// end-cell-basis-set-get
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-shells
// Get shell by index (returns const Shell&)
const Shell& shell = basis_set.get_shell(0);
size_t atom_idx = shell.atom_index;
OrbitalType orb_type = shell.orbital_type;
// Get exponents (returns const Eigen::VectorXd&)
const Eigen::VectorXd& exps = shell.exponents;
// Get coefficients (returns const Eigen::VectorXd&)
const Eigen::VectorXd& coeffs = shell.coefficients;

// Get information from shell
size_t num_primitives = shell.get_num_primitives();
size_t num_atomic_orbitals = shell.get_num_atomic_orbitals(AOType::Spherical);
int angular_momentum = shell.get_angular_momentum();
// end-cell-shells
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-serialization
// Generic serialization with format specification
basis_set.to_file("molecule.basis.json", "json");
basis_set.from_file("molecule.basis.json", "json");

// JSON serialization
basis_set.to_json_file("molecule.basis.json");
basis_set.from_json_file("molecule.basis.json");

// Direct JSON conversion
nlohmann::json j = basis_set.to_json();
basis_set.from_json(j);

// HDF5 serialization
basis_set.to_hdf5_file("molecule.basis.h5");
basis_set.from_hdf5_file("molecule.basis.h5");
// end-cell-serialization
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-utility-functions
// Convert orbital type to string (returns std::string)
std::string orbital_str =
    BasisSet::orbital_type_to_string(OrbitalType::D);  // "d"
// Convert string to orbital type (returns OrbitalType)
OrbitalType orbital_type =
    BasisSet::string_to_orbital_type("f");  // OrbitalType::F

// Get angular momentum (returns int)
int l_value = BasisSet::get_angular_momentum(OrbitalType::P);  // 1
// Get number of orbitals for angular momentum (returns int)
int num_orbitals = BasisSet::get_num_orbitals_for_l(2, AOType::Spherical);  // 5

// Convert basis type to string (returns std::string)
std::string basis_str =
    BasisSet::atomic_orbital_type_to_string(AOType::Cartesian);  // "cartesian"
// Convert string to basis type (returns AOType)
AOType atomic_orbital_type =
    BasisSet::string_to_atomic_orbital_type("spherical");  // AOType::Spherical
// end-cell-utility-functions
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-library
// Create a basis set from a predefined library (returns
// std::unique_ptr<BasisSet>)
auto basis_set = BasisSet::create("6-31G");

// List all available basis sets (returns std::vector<std::string>)
auto available_basis_sets = BasisSet::get_available_basis_sets();

// Check if a basis set exists in the library (returns bool)
bool has_basis = BasisSet::has_basis_set("cc-pvdz");
// end-cell-library
// --------------------------------------------------------------------------------------------
