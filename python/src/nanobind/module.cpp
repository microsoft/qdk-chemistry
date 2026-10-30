// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/exceptions.hpp>

namespace nb = nanobind;

void bind_base_class(nb::module_& m);
void bind_symmetry(nb::module_& m);
void bind_symmetry_blocked_scalar(nb::module_& m);
void bind_symmetry_blocked_tensor(nb::module_& m);
void bind_symmetry_blocked_index_set(nb::module_& m);
void bind_symmetry_blocked_sparse_map(nb::module_& m);
void bind_element_data(nb::module_& m);
void bind_orbitals(nb::module_& m);
void bind_hamiltonian(nb::module_& m);
void bind_wavefunction(nb::module_& m);
void bind_ansatz(nb::module_& m);
void bind_configuration_set(nb::module_& m);
void bind_localizer(nb::module_& m);
void bind_stability(nb::module_& m);
void bind_stability_result(nb::module_& m);
void bind_nuclear_gradients(nb::module_& m);
void bind_nuclear_hessian(nb::module_& m);
void bind_settings(nb::module_& m);
void bind_structure(nb::module_& m);
void bind_basis_set(nb::module_& m);
void bind_serialization(nb::module_& m);
void bind_mc(nb::module_& m);
void bind_mcscf(nb::module_& m);
void bind_hamiltonian_constructor(nb::module_& m);
void bind_effective_hamiltonian_constructor(nb::module_& m);
void bind_hamiltonian_factorization(nb::module_& m);
void bind_double_factorization(nb::module_& m);
void bind_scf(nb::module_& m);
void bind_nuclear_derivative(nb::module_& m);
void bind_population_analysis(nb::module_& m);
void bind_geometry_optimization(nb::module_& m);
void bind_active_space(nb::module_& m);
void bind_constants(nb::module_& m);
void bind_pmc(nb::module_& m);
void bind_configuration(nb::module_& m);
void bind_qdk_chemistry_config(nb::module_& m);
void bind_pauli_operator(nb::module_& m);
void bind_valence_space(nb::module_& m);
void bind_majorana_mapping(nb::module_& m);
void bind_orbital_rotation(nb::module_& m);
void bind_dynamical_correlation_calculator(nb::module_& m);
void bind_logger(nb::module_& m);
void bind_davidson_solver(nb::module_& m);
void bind_syev_solver(nb::module_& m);
void bind_lattice_geometry(nb::module_& m);
void bind_lattice_graph(nb::module_& m);
void bind_model_hamiltonians(nb::module_& m);
void bind_cube_generator(nb::module_& m);

NB_MODULE(_core, m) {
  m.doc() = "QDK/Chemistry C++ core bindings";

  nb::exception<qdk::chemistry::DuplicateRegistrationError>(
      m, "DuplicateRegistrationError", PyExc_ValueError);

  auto data = m.def_submodule("data");
  data.doc() = R"(Data submodule)";

  auto algorithms = m.def_submodule("_algorithms");
  algorithms.doc() = R"(Algorithms submodule)";

  auto utils = m.def_submodule("utils");
  utils.doc() = R"(Utilities submodule)";

  auto symmetry = data.def_submodule("symmetry");
  symmetry.doc() =
      R"(Single-particle SymmetryProduct and symmetry-blocked storage)";

  // Ordering is important!

  bind_base_class(data);
  bind_symmetry(symmetry);  // axis types + SBT/SBIS before orbital containers
  bind_symmetry_blocked_scalar(symmetry);
  bind_symmetry_blocked_tensor(symmetry);
  bind_symmetry_blocked_index_set(symmetry);
  bind_symmetry_blocked_sparse_map(symmetry);
  bind_element_data(data);  // Element enums must be bound before Structure
  bind_structure(data);
  bind_settings(data);
  bind_basis_set(data);
  bind_orbitals(data);
  bind_lattice_geometry(data);
  bind_lattice_graph(data);
  bind_hamiltonian(data);
  bind_configuration(data);
  bind_configuration_set(data);
  bind_wavefunction(data);
  bind_ansatz(data);
  bind_stability_result(data);
  bind_nuclear_gradients(data);
  bind_nuclear_hessian(data);
  bind_pauli_operator(data);
  bind_majorana_mapping(
      data);  // Depends on SparsePauliWord from pauli_operator
  bind_serialization(data);

  bind_localizer(algorithms);
  bind_mc(algorithms);
  bind_mcscf(algorithms);
  bind_hamiltonian_constructor(algorithms);
  bind_effective_hamiltonian_constructor(algorithms);
  bind_hamiltonian_factorization(algorithms);
  bind_double_factorization(algorithms);
  bind_scf(algorithms);
  bind_nuclear_derivative(algorithms);
  bind_population_analysis(algorithms);
  bind_geometry_optimization(algorithms);
  bind_active_space(algorithms);
  bind_dynamical_correlation_calculator(algorithms);
  bind_pmc(algorithms);
  bind_stability(algorithms);
  bind_davidson_solver(algorithms);
  bind_syev_solver(algorithms);

  // Bind utilities
  bind_valence_space(utils);
  bind_orbital_rotation(utils);
  bind_model_hamiltonians(utils);
  bind_logger(utils);
  bind_cube_generator(utils);

  // Bind constants and config at the top level
  bind_constants(m);
  bind_qdk_chemistry_config(m);

  // Install the global create function so AlgorithmRef can auto-resolve
  // default settings via the Python algorithm registry.
  qdk::chemistry::data::AlgorithmRef::create_default_settings =
      [](const std::string& type, const std::string& name)
      -> std::shared_ptr<qdk::chemistry::data::Settings> {
    nb::gil_scoped_acquire gil;
    try {
      nb::module_ reg =
          nb::module_::import_("qdk_chemistry.algorithms.registry");
      nb::object instance = reg.attr("create")(type, name);
      nb::object py_settings = instance.attr("settings")();
      auto json_obj =
          nb::cast<qdk::chemistry::data::Settings&>(py_settings).to_json();
      return qdk::chemistry::data::Settings::from_json(json_obj);
    } catch (...) {
      return nullptr;
    }
  };
}
