// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/f12_scf.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class F12HartreeFockSolverBase : public F12HartreeFockSolver,
                                 public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, F12HartreeFockSolver, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, F12HartreeFockSolver, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> reference) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Wavefunction>, F12HartreeFockSolver,
                           _run_impl, reference);
  }
};

void bind_f12_hartree_fock_solver(py::module &m) {
  py::class_<F12HartreeFockSolver, F12HartreeFockSolverBase, py::smart_holder>
      f12_hartree_fock_solver(m, "F12HartreeFockSolver",
                              R"(
Abstract base class for F12-Hartree-Fock solvers.

This class defines the interface for relaxing a reference wavefunction in the
mean field of a similarity-transformed (transcorrelated) Hamiltonian. The
returned wavefunction carries the relaxed orbital coefficients and the
corresponding dressed-Fock orbital energies. Concrete implementations should
inherit from this class and implement the ``_run_impl`` method.

Examples:
    To create a custom F12-Hartree-Fock solver, inherit from this class::

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyF12HartreeFockSolver(alg.F12HartreeFockSolver):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     def _run_impl(self, reference: data.Wavefunction) -> data.Wavefunction:
        ...         # Custom F12-HF relaxation implementation
        ...         return wavefunction

)");

  f12_hartree_fock_solver.def(py::init<>(),
                              R"(
Create an ``F12HartreeFockSolver`` instance.

Default constructor for the abstract base class. This should typically be called
from derived class constructors.

)");

  f12_hartree_fock_solver.def("run", &F12HartreeFockSolver::run,
                              R"(
Relax the given reference wavefunction in the F12-dressed mean field.

This method automatically locks settings before execution to prevent
modifications during the solve.

Args:
    reference (qdk_chemistry.data.Wavefunction): The reference wavefunction supplying the orbital basis

Returns:
    qdk_chemistry.data.Wavefunction: The relaxed F12-HF wavefunction

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called

)",
                              py::arg("reference"));

  f12_hartree_fock_solver.def("settings", &F12HartreeFockSolver::settings,
                              R"(
Access the solver's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the solver

)",
                              py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  f12_hartree_fock_solver.def_property(
      "_settings",
      [](F12HartreeFockSolverBase &solver) -> Settings & {
        return solver.settings();
      },
      [](F12HartreeFockSolverBase &solver,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        solver.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a
specialized Settings subclass in their constructors.

)");

  f12_hartree_fock_solver.def("name", &F12HartreeFockSolver::name,
                              R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  f12_hartree_fock_solver.def("type_name", &F12HartreeFockSolver::type_name,
                              R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  f12_hartree_fock_solver.def("hash", &F12HartreeFockSolver::hash,
                              py::arg("reference"));

  // Factory class binding
  qdk::chemistry::python::bind_algorithm_factory<F12HartreeFockSolverFactory,
                                                 F12HartreeFockSolver,
                                                 F12HartreeFockSolverBase>(
      m, "F12HartreeFockSolverFactory");

  f12_hartree_fock_solver.def("__repr__", [](const F12HartreeFockSolver &) {
    return "<qdk_chemistry.algorithms.F12HartreeFockSolver>";
  });

  qdk::chemistry::python::bind_create_nested(f12_hartree_fock_solver);

  // Bind concrete microsoft::CtF12HartreeFockSolver implementation
  py::class_<microsoft::CtF12HartreeFockSolver, F12HartreeFockSolver,
             py::smart_holder>(m, "QdkCtF12HartreeFockSolver", R"(
QDK implementation of the canonical transcorrelated F12 (CT-F12) Hartree-Fock
solver.

This class builds the dressed transcorrelated Hamiltonian from the reference
orbitals and relaxes the closed-shell orbitals in its mean field. The returned
wavefunction carries the relaxed orbital coefficients and the dressed-Fock
orbital energies, with the frozen core marked inactive, so its conventional MP2
yields the F12-MP2 energy.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    solver = alg.QdkCtF12HartreeFockSolver()
    solver.settings().set("gamma", 1.5)
    relaxed = solver.run(reference_wavefunction)

See Also:
    :class:`F12HartreeFockSolver`
    :class:`QdkCtF12HamiltonianConstructor`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a CT-F12 Hartree-Fock solver with default settings.

)");
}
