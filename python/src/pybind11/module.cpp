// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_base_class(py::module& m);
void bind_orbitals(py::module& m);
void bind_hamiltonian(py::module& m);
void bind_wavefunction(py::module& m);
void bind_ansatz(py::module& m);
void bind_localizer(py::module& m);
void bind_stability(py::module& m);
void bind_stability_result(py::module& m);
void bind_settings(py::module& m);
void bind_structure(py::module& m);
void bind_basis_set(py::module& m);
void bind_coupled_cluster(py::module& m);
void bind_mc(py::module& m);
void bind_mcscf(py::module& m);
void bind_hamiltonian_constructor(py::module& m);
void bind_scf(py::module& m);
void bind_active_space(py::module& m);
void bind_constants(py::module& m);
void bind_cc(py::module& m);
void bind_pmc(py::module& m);
void bind_configuration(py::module& m);
void bind_qdk_chemistry_config(py::module& m);
void bind_utils(py::module& m);

PYBIND11_MODULE(_core, m) {
  m.doc() = "QDK/Chemistry C++ core bindings";

  auto data = m.def_submodule("data");
  data.doc() = R"(Data submodule)";

  auto algorithms = m.def_submodule("_algorithms");
  algorithms.doc() = R"(Algorithms submodule)";

  auto utils = m.def_submodule("utils");
  utils.doc() = R"(Utilities submodule)";

  // Ordering is important!
  bind_base_class(data);
  bind_structure(data);
  bind_settings(data);
  bind_basis_set(data);
  bind_orbitals(data);
  bind_hamiltonian(data);
  bind_configuration(data);
  bind_wavefunction(data);
  bind_ansatz(data);
  bind_coupled_cluster(data);
  bind_stability_result(data);

  bind_localizer(algorithms);
  bind_mc(algorithms);
  bind_mcscf(algorithms);
  bind_hamiltonian_constructor(algorithms);
  bind_scf(algorithms);
  bind_active_space(algorithms);
  bind_cc(algorithms);
  bind_pmc(algorithms);
  bind_stability(algorithms);

  // Bind utilities
  bind_utils(utils);

  // Bind constants and config at the top level
  bind_constants(m);
  bind_qdk_chemistry_config(m);
}
