// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Basis Set usage examples.
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-loading
  // Create a water molecule structure
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {0.757, 0.586, 0.0}, {-0.757, 0.586, 0.0}};
  std::vector<std::string> symbols = {"O", "H", "H"};
  Structure structure(coords, symbols);

  // Create basis sets from the library using basis set name
  auto basis_from_name = BasisSet::from_basis_name("sto-3g", structure);

  // Create basis sets from the library using element-based mapping
  std::map<std::string, std::string> basis_map = {{"H", "sto-3g"},
                                                  {"O", "def2-svp"}};
  auto basis_from_element = BasisSet::from_element_map(basis_map, structure);

  // Create basis sets from the library using index-based mapping
  std::map<size_t, std::string> index_basis_map = {
      {0, "def2-svp"}, {1, "sto-3g"}, {2, "sto-3g"}};  // O at 0, H at 1 and 2
  auto basis_from_index = BasisSet::from_index_map(index_basis_map, structure);
  // end-cell-loading
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-create
  // Create a shell with multiple primitives
  size_t atom_index = 0;                      // First atom
  OrbitalType orbital_type = OrbitalType::P;  // p orbital
  Eigen::VectorXd exponents(2);
  exponents << 0.16871439, 0.62391373;
  Eigen::VectorXd coefficients(2);
  coefficients << 0.43394573, 0.56604777;
  Shell shell1(atom_index, orbital_type, exponents, coefficients);

  // Add a shell with a single primitive
  Shell shell2(1, OrbitalType::S, Eigen::VectorXd::Constant(1, 0.5),
               Eigen::VectorXd::Constant(1, 1.0));

  // Create a basis set from the shells
  std::vector<Shell> shells = {shell1, shell2};
  std::string name = "6-31G";
  BasisSet basis_set(name, shells, structure, AOType::Spherical);
  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-access
  // Get basis set type and name (returns AOType)
  auto basis_atomic_orbital_type = basis_set.get_atomic_orbital_type();
  // Get basis set name (returns std::string)
  auto basis_name = basis_set.get_name();

  // Get all shells (returns const std::vector<Shell>&)
  auto all_shells = basis_set.get_shells();
  // Get shells for specific atom (returns const std::vector<const Shell>&)
  auto shells_for_atom = basis_set.get_shells_for_atom(0);
  // Get specific shell by index (returns const Shell&)
  const Shell& specific_shell = basis_set.get_shell(1);

  // Get counts
  size_t num_shells = basis_set.get_num_shells();
  size_t num_atomic_orbitals = basis_set.get_num_atomic_orbitals();
  size_t num_atoms = basis_set.get_num_atoms();

  // Get atomic orbital information (returns std::pair<size_t, int>)
  auto [shell_index, m_quantum_number] = basis_set.get_atomic_orbital_info(2);
  size_t atom_index = basis_set.get_atom_index_for_atomic_orbital(2);

  // Get indices for specific atoms or orbital types
  // Returns std::vector<size_t>
  auto atomic_orbital_indices =
      basis_set.get_atomic_orbital_indices_for_atom(1);
  // Returns std::vector<size_t>
  auto shell_indices =
      basis_set.get_shell_indices_for_orbital_type(OrbitalType::P);
  // Returns std::vector<size_t>
  auto shell_indices_specific =
      basis_set.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::D);
  // end-cell-access
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
  size_t num_aos = shell.get_num_atomic_orbitals(AOType::Spherical);
  int angular_momentum = shell.get_angular_momentum();
  // end-cell-shells
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-serialization
  // Generic serialization with format specification
  basis_set.to_file("molecule.basis_set.json", "json");
  auto basis_set_from_file =
      BasisSet::from_file("molecule.basis_set.json", "json");

  // JSON serialization
  basis_set.to_json_file("molecule.basis_set.json");
  auto basis_set_from_json_file =
      BasisSet::from_json_file("molecule.basis_set.json");

  // Direct JSON conversion
  nlohmann::json j = basis_set.to_json();
  auto basis_set_from_json = BasisSet::from_json(j);

  // HDF5 serialization
  basis_set.to_hdf5_file("molecule.basis_set.h5");
  auto basis_set_from_hdf5 = BasisSet::from_hdf5_file("molecule.basis_set.h5");
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
  int num_orbitals =
      BasisSet::get_num_orbitals_for_l(2, AOType::Spherical);  // 5

  // Convert basis type to string (returns std::string)
  std::string basis_str = BasisSet::atomic_orbital_type_to_string(
      AOType::Cartesian);  // "cartesian"
  // Convert string to basis type (returns AOType)
  AOType atomic_orbital_type = BasisSet::string_to_atomic_orbital_type(
      "spherical");  // AOType::Spherical
  // end-cell-utility-functions
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-library
  // Check supported basis sets
  auto supported_basis_sets = BasisSet::get_supported_basis_set_names();

  // Check supported elements for basis set
  auto supported_elements =
      BasisSet::get_supported_elements_for_basis_set("sto-3g");
  // end-cell-library
  // --------------------------------------------------------------------------------------------

  return 0;
}
