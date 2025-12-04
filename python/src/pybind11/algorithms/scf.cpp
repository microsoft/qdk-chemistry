// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/scf.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

using ReturnType = std::pair<double, std::shared_ptr<Wavefunction>>;
// Trampoline class for enabling Python inheritance
class ScfSolverBase : public ScfSolver,
                      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, ScfSolver, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, ScfSolver, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  ReturnType _run_impl(
      std::shared_ptr<Structure> structure, int charge, int spin_multiplicity,
      std::optional<std::shared_ptr<Orbitals>> initial_guess) const override {
    PYBIND11_OVERRIDE_PURE(ReturnType, ScfSolver, _run_impl, structure, charge,
                           spin_multiplicity, initial_guess);
  }
};

void bind_scf(py::module &m) {
  // ScfSolver abstract base class
  py::class_<ScfSolver, ScfSolverBase, py::smart_holder> scf_solver(
      m, "ScfSolver", R"(
Abstract base class for Self-Consistent Field (SCF) solvers.

This class defines the interface for SCF calculations that compute molecular orbitals from a molecular structure.
Concrete implementations should inherit from this class and implement the solve method.

Examples:
    >>> # To create a custom SCF solver, inherit from this class:
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyScfSolver(alg.ScfSolver):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     # Implement the _run_impl method
    ...     def _run_impl(self, structure: data.Structure, charge: int, spin_multiplicity: int, initial_guess=None) -> tuple[float, data.Wavefunction]:
    ...         # Custom SCF implementation
    ...         return energy, wavefunction

)");

  scf_solver.def(py::init<>(),
                 R"(
Create an ScfSolver instance.

Initializes a new Self-Consistent Field (SCF) solver with default settings.
Configuration options can be modified through the ``settings()`` method.

Examples:
    >>> scf = alg.ScfSolver()
    >>> scf.settings().set("max_iterations", 100)
    >>> scf.settings().set("convergence_threshold", 1e-8)

)");

  scf_solver.def(
      "run",
      [](const ScfSolver &solver,
         std::shared_ptr<qdk::chemistry::data::Structure> structure, int charge,
         int spin_multiplicity,
         std::optional<std::shared_ptr<qdk::chemistry::data::Orbitals>>
             initial_guess) {
        return solver.run(structure, charge, spin_multiplicity, initial_guess);
      },
      R"(
Perform SCF calculation on the given molecular structure.

This method automatically locks settings before execution.

Args:
    structure (qdk_chemistry.data.Structure): The molecular structure to solve
    charge (int): The molecular charge
    spin_multiplicity (int): The spin multiplicity of the molecular system
    initial_guess (qdk_chemistry.data.Orbitals | None): Initial orbital guess for the SCF calculation. Defaults to ``None``.

Returns:
    tuple[float, qdk_chemistry.data.Wavefunction]: Converged total energy (nuclear + electronic) and the resulting wavefunction.

)",
      py::arg("structure"), py::arg("charge"), py::arg("spin_multiplicity"),
      py::arg("initial_guess") = std::nullopt);

  scf_solver.def(
      "run",
      [](const ScfSolver &solver,
         std::shared_ptr<qdk::chemistry::data::Structure> structure, int charge,
         int spin_multiplicity) {
        return solver.run(structure, charge, spin_multiplicity);
      },
      R"(
Perform SCF calculation on the given molecular structure (without initial guess).

This method automatically locks settings before execution.

Args:
    structure (qdk_chemistry.data.Structure): The molecular structure to solve
    charge (int): The molecular charge
    spin_multiplicity (int): The spin multiplicity of the molecular system

Returns:
    tuple[float, qdk_chemistry.data.Wavefunction]: Converged total energy (nuclear + electronic) and the resulting wavefunction

)",
      py::arg("structure"), py::arg("charge"), py::arg("spin_multiplicity"));

  scf_solver.def("settings", &ScfSolver::settings,
                 R"(
Access the solver's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the solver

)",
                 py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  scf_solver.def_property(
      "_settings",
      [](ScfSolverBase &solver) -> Settings & { return solver.settings(); },
      [](ScfSolverBase &solver,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        solver.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyScfSolver(alg.ScfSolver):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()

)");

  scf_solver.def("name", &ScfSolver::name,
                 R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  scf_solver.def("type_name", &ScfSolver::type_name,
                 R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  // Factory class binding - creates ScfSolverFactory class with static methods
  bind_algorithm_factory<ScfSolverFactory, ScfSolver, ScfSolverBase>(
      m, "ScfSolverFactory");

  scf_solver.def("__repr__", [](const ScfSolver &) {
    return "<qdk_chemistry.algorithms.ScfSolver>";
  });

  // Bind concrete microsoft::ScfSolver implementation
  py::class_<microsoft::ScfSolver, ScfSolver, py::smart_holder>(
      m, "QdkScfSolver", R"(
QDK implementation of the SCF solver.

This class provides a concrete implementation of the SCF (Self-Consistent
Field) solver using the internal backend.
It inherits from the base :class:`ScfSolver` class and implements the
``solve`` method to perform self-consistent field calculations on molecular
structures.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg
    import qdk_chemistry.data as data

    # Create a molecular structure
    water = data.Structure(
        positions=[[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]],
        elements=[data.Element.O, data.Element.H, data.Element.H]
    )

    # Create an SCF solver instance
    scf_solver = alg.QdkScfSolver()

    # Configure settings if needed
    scf_solver.settings().set("basis_set", "sto-3g")

    # Perform SCF calculation
    energy, wavefunction = scf_solver.run(water, 0, 1)

See Also:
    :class:`ScfSolver`
    :class:`qdk_chemistry.data.Structure`
    :class:`qdk_chemistry.data.Orbitals`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes an SCF solver with default settings.

)");
}
