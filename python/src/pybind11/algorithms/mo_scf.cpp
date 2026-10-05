// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/mo_scf.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/algorithms/mo_scf.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

using MoScfReturn = std::pair<double, std::shared_ptr<Ansatz>>;

class MoScfSolverBase : public MoScfSolver,
                        public py::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, MoScfSolver, name);
  }
  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, MoScfSolver, aliases);
  }
  void replace_settings(std::unique_ptr<Settings> settings) {
    _settings = std::move(settings);
  }

 protected:
  MoScfReturn _run_impl(std::shared_ptr<Hamiltonian> hamiltonian,
                        unsigned int nalpha,
                        unsigned int nbeta) const override {
    PYBIND11_OVERRIDE_PURE(MoScfReturn, MoScfSolver, _run_impl, hamiltonian,
                           nalpha, nbeta);
  }
};

void bind_mo_scf(py::module& m) {
  py::class_<MoScfSolver, MoScfSolverBase, py::smart_holder> solver(
      m, "MoScfSolver", R"(
Hartree-Fock optimization from integrals in an orthonormal MO basis.

The input Hamiltonian supplies the active-space integrals and constant energy.
No molecular geometry or atomic basis is required. Implementations return
the energy and an Ansatz containing a consistently rotated Hamiltonian and
single-determinant wavefunction.
)");
  solver.def(py::init<>());
  solver.def("run", &MoScfSolver::run, py::arg("hamiltonian"),
             py::arg("n_active_alpha_electrons"),
             py::arg("n_active_beta_electrons"), R"(
Optimize the active orbitals and transform the Hamiltonian to the new basis.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): Input MO-basis Hamiltonian.
    n_active_alpha_electrons (int): Number of active alpha electrons.
    n_active_beta_electrons (int): Number of active beta electrons.

Returns:
    tuple[float, qdk_chemistry.data.Ansatz]: Total energy in Hartree and optimized Ansatz.

Raises:
    ValueError: For invalid counts or unsupported inputs.
    RuntimeError: If the SCF iterations fail to converge.
)");
  solver.def("hash", &MoScfSolver::hash, py::arg("hamiltonian"),
             py::arg("n_active_alpha_electrons"),
             py::arg("n_active_beta_electrons"));
  solver.def("settings", &MoScfSolver::settings,
             py::return_value_policy::reference_internal);
  solver.def_property(
      "_settings",
      [](MoScfSolverBase& self) -> Settings& { return self.settings(); },
      [](MoScfSolverBase& self, std::unique_ptr<Settings> settings) {
        self.replace_settings(std::move(settings));
      },
      py::return_value_policy::reference_internal);
  solver.def("name", &MoScfSolver::name);
  solver.def("aliases", &MoScfSolver::aliases);
  solver.def("type_name", &MoScfSolver::type_name);
  bind_create_nested(solver);
  bind_algorithm_factory<MoScfSolverFactory, MoScfSolver, MoScfSolverBase>(
      m, "MoScfSolverFactory");
  py::class_<microsoft::MoScfSolver, MoScfSolver, py::smart_holder>(
      m, "QdkMoScfSolver", R"(
Native QDK MO-basis HF solver, sharing the molecular SCF engine's DIIS and GDM.

Supports RHF, ROHF, and UHF optimization from real, restricted input integrals.
Inactive and external orbitals are held fixed. For ModelOrbitals inputs, the
returned coefficients are rotations relative to the original model basis.
Spin-dependent input integrals, DFT, and multi-rank MPI are not supported.
)")
      .def(py::init<>());
}
