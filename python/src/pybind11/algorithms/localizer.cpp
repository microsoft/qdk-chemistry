// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/mp2_natural_orbitals.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/pipek_mezey.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/vvhv.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class LocalizerBase : public Localizer,
                      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, Localizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, Localizer, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction,
      const std::vector<size_t> &loc_indices_a,
      const std::vector<size_t> &loc_indices_b) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Wavefunction>, Localizer, _run_impl,
                           wavefunction, loc_indices_a, loc_indices_b);
  }
};

void bind_localizer(py::module &m) {
  // Localizer abstract base class
  py::class_<Localizer, LocalizerBase, py::smart_holder> localizer(
      m, "OrbitalLocalizer", R"(
Abstract base class for orbital localizers.

This class defines the interface for localizing molecular orbitals.
Localization transforms canonical molecular orbitals into localized orbitals that are spatially confined to specific regions or bonds.
Concrete implementations should inherit from this class and implement the localize method.

Examples:
    >>> # To create a custom orbital localizer, inherit from this class.
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyLocalizer(alg.Localizer):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     # Implement the _run_impl method
    ...     def _run_impl(self, wavefunction: data.Wavefunction, loc_indices_a: list, loc_indices_b: list) -> data.Wavefunction:
    ...         # Custom localization implementation
    ...         return localized_wavefunction
)");

  localizer.def(py::init<>(),
                R"(
Create a Localizer instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MyLocalizer(alg.Localizer):
    ...     def __init__(self):
    ...         super().__init__()  # Calls this constructor

)");

  localizer.def("run", &Localizer::run,
                R"(
Localize molecular orbitals in the given wavefunction.

Args:
    wavefunction (qdk_chemistry.data.Wavefunction): The canonical molecular wavefunction to localize
    loc_indices_a (list[int]): Indices of alpha orbitals to localize (empty for no localization)
    loc_indices_b (list[int]): Indices of beta orbitals to localize (empty for no localization)

Notes:
    For restricted orbitals, ``loc_indices_b`` must match ``loc_indices_a``.

Returns:
    qdk_chemistry.data.Wavefunction: The localized molecular wavefunction

Raises:
    ValueError: If orbital indices are invalid or inconsistent
    RuntimeError: If localization fails due to numerical issues

)",
                py::arg("wavefunction"), py::arg("loc_indices_a"),
                py::arg("loc_indices_b"));

  localizer.def("settings", &Localizer::settings,
                R"(
Access the localizer's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the localizer

)",
                py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  localizer.def_property(
      "_settings",
      [](LocalizerBase &loc) -> Settings & { return loc.settings(); },
      [](LocalizerBase &loc,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        loc.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyLocalizer(alg.Localizer):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()

)");

  localizer.def("name", &Localizer::name,
                R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  localizer.def("type_name", &Localizer::type_name,
                R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  // Factory class binding - creates LocalizerFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<LocalizerFactory, Localizer,
                                                 LocalizerBase>(
      m, "LocalizerFactory");

  localizer.def("__repr__", [](const Localizer &) {
    return "<qdk_chemistry.algorithms.OrbitalLocalizer>";
  });

  // Bind concrete microsoft::PipekMezeyLocalizer implementation
  py::class_<microsoft::PipekMezeyLocalizer, Localizer, py::smart_holder>(
      m, "QdkPipekMezeyLocalizer", R"(
QDK Pipek-Mezey orbital localizer.

This class provides a concrete implementation of the orbital localizer using
the Pipek-Mezey localization algorithm. The Pipek-Mezey algorithm maximizes
the sum of squares of atomic orbital populations on atoms, resulting in
orbitals that are more localized to individual atoms or bonds.

This implementation separately localizes occupied and virtual orbitals to
maintain the occupied-virtual separation.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create a Pipek-Mezey localizer
    localizer = alg.QdkPipekMezeyLocalizer()

    # Configure settings if needed
    localizer.settings().set("max_iterations", 100)
    localizer.settings().set("convergence_tolerance", 1e-8)

    # Localize orbitals
    localized_wfn = localizer.run(wavefunction, loc_indices_a, loc_indices_b)

See Also:
    :class:`OrbitalLocalizer`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a Pipek-Mezey localizer with default settings.

)");

  // Bind concrete microsoft::MP2NaturalOrbitalLocalizer implementation
  py::class_<microsoft::MP2NaturalOrbitalLocalizer, Localizer,
             py::smart_holder>(m, "QdkMP2NaturalOrbitalLocalizer", R"(
QDK MP2 natural orbital transformer.

This class provides a concrete implementation that transforms canonical
molecular orbitals into natural orbitals derived from second-order
Møller-Plesset perturbation theory (MP2). Natural orbitals are eigenfunctions
of the first-order reduced density matrix.

MP2 natural orbitals often provide a more compact representation of the
electronic wavefunction, which can improve computational efficiency in
correlation methods.

.. note::
    Only supports restricted orbitals and closed-shell systems.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create an MP2 natural orbital localizer
    localizer = alg.QdkMP2NaturalOrbitalLocalizer()

    # Transform to MP2 natural orbitals
    no_wfn = localizer.run(wavefunction, loc_indices_a, loc_indices_b)

See Also:
    :class:`OrbitalLocalizer`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes an MP2 natural orbital transformer with default settings.

)");

  // Bind concrete microsoft::VVHVLocalizer implementation
  py::class_<microsoft::VVHVLocalizer, Localizer, py::smart_holder>(
      m, "QdkVVHVLocalizer", R"(
QDK Valence Virtual - Hard Virtual (VV-HV) orbital localizer.

This class provides a concrete implementation of the orbital localizer using
the VV-HV localization algorithm. The VV-HV algorithm partitions virtual
orbitals into valence virtuals (VVs) and hard virtuals (HVs) based on
projection onto a minimal basis, then localizes each space separately.

The algorithm is particularly useful for post-Hartree-Fock methods where
separate treatment of valence and Rydberg-like virtual orbitals improves
computational efficiency and accuracy.

Implementation based on:
  Subotnik et al. JCP 123, 114108 (2005)
  Wang et al. JCTC 21, 1163 (2025)

.. note::
    This localizer requires all orbital indices to be covered in the
    localization call.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create a VV-HV localizer
    localizer = alg.QdkVVHVLocalizer()

    # Configure settings if needed
    localizer.settings().set("minimal_basis", "sto-3g")
    localizer.settings().set("weighted_orthogonalization", True)
    localizer.settings().set("max_iterations", 100)
    localizer.settings().set("convergence_tolerance", 1e-8)

    # Localize orbitals (must include all orbital indices)
    localized_wfn = localizer.run(wavefunction, loc_indices_a, loc_indices_b)

See Also:
    :class:`OrbitalLocalizer`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a VV-HV localizer with default settings.

)");
}
