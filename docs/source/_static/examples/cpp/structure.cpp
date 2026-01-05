// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Structure usage examples.
// --------------------------------------------------------------------------------------------
// start-cell-create
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

int main() {
  // Specify a structure using coordinates (in Bohr), and either symbols or
  // elements
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0},
                                         {0.0, 0.0, 1.4}};  // Bohr
  std::vector<std::string> symbols = {"H", "H"};

  Structure structure(coords, symbols);

  // Equivalent to 'structure', but uses 'elements' instead of 'symbols'
  std::vector<Element> elements = {Element::H, Element::H};
  Structure structure_alternative(coords, elements);

  // Another variation on construction: can specify custom masses and/or charges
  std::vector<double> custom_masses{1.001, 0.999};
  std::vector<double> custom_charges = {0.9, 1.1};
  Structure structure_custom(coords, elements, custom_masses, custom_charges);

  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-from-file
  // Load a structure from an XYZ file (coordinates in Angstrom are converted to
  // Bohr)
  auto structure_from_file =
      Structure::from_xyz_file("../data/h2.structure.xyz");

  // Load a structure from a JSON file (coordinates are stored in Bohr)
  auto structure_from_json =
      Structure::from_json_file("../data/water.structure.json");
  // end-cell-from-file
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-data
  // Get coordinates of a specific atom in Bohr
  Eigen::Vector3d atom_coords =
      structure.get_atom_coordinates(0);  // First atom

  // Get element of a specific atom
  Element element = structure.get_atom_element(0);  // First atom

  // Get all coordinates (in Bohr) as a matrix
  Eigen::MatrixXd all_coords = structure.get_coordinates();

  // Get all elements as a vector
  std::vector<Element> all_elements = structure.get_elements();
  // end-cell-data
  // --------------------------------------------------------------------------------------------
  return 0;
}
