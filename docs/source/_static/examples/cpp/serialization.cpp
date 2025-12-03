// Serialization usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-json
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

// Structure data class example
std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
std::vector<std::string> symbols = {"H", "H"};
std::vector<double> custom_masses{1.001, 0.999};
std::vector<double> custom_charges = {0.9, 1.1};
Structure structure(coords, symbols, custom_masses, custom_charges);

// Serialize to JSON object
auto structure_data = structure.to_json();

// Deserialize from JSON object
// "Structure" is the data type to de-serialize into (will throw, if it doesn't
// match)
auto structure_from_json = Structure::from_json(structure_data);

// Write to json file
structure.to_json_file(
    "filename.structure.json");  // Extension depends on object type

// Read from json file
auto structure_from_json_file =
    Structure::from_json_file("filename.structure.json");

// end-cell-json
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-hdf5

// Hamiltonian data class example
// Create dummy data for Hamiltonian class
Eigen::MatrixXd one_body = Eigen::MatrixXd::Identity(2, 2);
Eigen::VectorXd two_body = 2 * Eigen::VectorXd::Ones(16);
auto orbitals =
    std::make_shared<ModelOrbitals>(2, true);  // 2 orbitals, restricted
double core_energy = 1.5;
Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Zero(0, 0);

Hamiltonian h_example(one_body, two_body, orbitals, core_energy, inactive_fock);

h_example.to_hdf5_file(
    "h_example.hamiltonian.h5");  // Extension depends on object type

// Deserialize from HDF5 file
auto h_example_from_hdf5_file =
    Hamiltonian::from_hdf5_file("h_example.hamiltonian.h5");

// end-cell-hdf5
// --------------------------------------------------------------------------------------------
