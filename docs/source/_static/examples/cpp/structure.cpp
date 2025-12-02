// Structure usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

// Create an empty structure
Structure structure;

// Add atoms with their 3D coordinates and element symbols (coordinates in
// Bohr/atomic units)
structure.add_atom(Eigen::Vector3d(0.0, 0.0, 0.0), "H");
structure.add_atom(Eigen::Vector3d(0.0, 0.0, 1.4), "H");
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-load
// Load from XYZ file
auto structure = Structure::from_xyz_file(
    "../data/water.structure.xyz");  // Required .structure.xyz suffix

// Load from JSON file
auto structure = Structure::from_json_file(
    "../data/water.structure.json");  // Required .structure.json suffix
// end-cell-load
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-data
// Get coordinates of a specific atom in angstrom
Eigen::Vector3d coords = structure.get_atom_coordinates(0);  // First atom

// Get element of a specific atom
std::string element = structure.get_atom_element(0);  // First atom

// Get all coordinates (in angstrom) as a matrix
Eigen::MatrixXd all_coords = structure.get_coordinates();

// Get all elements as a vector
std::vector<std::string> all_elements = structure.get_elements();
// end-cell-data
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-serialize
// Serialize to JSON object
auto json_data = structure.to_json();

// Deserialize from JSON object
auto structure_from_json = Structure::from_json(json_data);

// Serialize to JSON file
structure.to_json_file(
    "molecule.structure.json");  // Required .structure.json suffix

// Get XYZ format as string
std::string xyz_string = structure.to_xyz();

// Load from XYZ string
auto structure_from_xyz = Structure::from_xyz(xyz_string);

// Serialize to XYZ file
structure.to_xyz_file(
    "molecule.structure.xyz");  // Required .structure.xyz suffix
// end-cell-serialize
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-manip
// Add an atom with coordinates and element
structure.add_atom(Eigen::Vector3d(1.0, 0.0, 0.0), "O");  // Add an oxygen atom

// Remove an atom
structure.remove_atom(2);  // Remove the third atom
// end-cell-manip
// --------------------------------------------------------------------------------------------
