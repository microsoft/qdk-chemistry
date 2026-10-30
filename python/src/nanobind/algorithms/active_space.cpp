// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/active_space/autocas_active_space.hpp"
#include "qdk/chemistry/algorithms/microsoft/active_space/entropy_active_space.hpp"
#include "qdk/chemistry/algorithms/microsoft/active_space/occupation_active_space.hpp"
#include "qdk/chemistry/algorithms/microsoft/active_space/valence_active_space.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class ActiveSpaceSelectorBase : public ActiveSpaceSelector {
 public:
  NB_TRAMPOLINE(ActiveSpaceSelector);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction) const override {
    NB_OVERRIDE_PURE(_run_impl, wavefunction);
  }
};

void bind_active_space(nb::module_& m) {
  // ActiveSpaceSelector abstract base class
  nb::class_<ActiveSpaceSelector, ActiveSpaceSelectorBase> selector(
      m, "ActiveSpaceSelector", R"(
Abstract base class for active space selector algorithms.

This class defines the interface for selecting active spaces from a set of orbitals.
Active space selection is a critical step in many quantum chemistry methods, particularly for multireference calculations.
Concrete implementations should inherit from this class and implement the ``select_active_space`` method.

.. rubric:: Return value semantics

Implementations return a new ``Wavefunction`` object with active-space data populated.
Some selectors (e.g., occupation/valence) return a copy with only metadata updated.
Others (e.g., AVAS) may rotate/canonicalize orbitals and recompute occupations,
so the returned coefficients/occupations can differ from the input.
The input ``Wavefunction`` object is never modified.

Examples:
    To create a custom active space selector, inherit from this class.::

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyActiveSpaceSelector(alg.ActiveSpaceSelector):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     # Implement the _run_impl method (called by run())
        ...     def _run_impl(self, wavefunction: data.Wavefunction) -> data.Wavefunction:
        ...         # Custom active space selection implementation that returns orbitals with active space populated.
        ...         # For selectors that only annotate, return a copied object with metadata set; for others,
        ...         # you may also rotate/transform the coefficients as needed.
        ...         act_orbitals = data.Orbitals(wavefunction.get_orbitals()) # Copy base data by default
        ...         act_orbitals.set_active_space_indices([0, 1, 2, 3])
        ...         act_orbitals.set_num_active_electrons(2)
        ...         ... # Additional logic to modify orbitals if needed
        ...         act_wavefunction = data.Wavefunction(...)
        ...         return act_wavefunction # Return modified wavefunction object

)");

  selector.def(nb::init<>(),
               R"(
Create an ActiveSpaceSelector instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MySelector(alg.ActiveSpaceSelector):
    ...     def __init__(self):
    ...         super().__init__()  # Calls parent constructor

)");

  selector.def("run", &ActiveSpaceSelector::run, nb::arg("wavefunction"),
               R"(
Select active space orbitals from the given wavefunction.

This method automatically locks settings before execution to prevent modifications during selection.

Args:
    wavefunction (qdk_chemistry.data.Wavefunction): The wavefunction from which to select the active space

Returns:
    qdk_chemistry.data.Wavefunction: Wavefunction with active space data populated

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called

)");

  selector.def("settings", &ActiveSpaceSelector::settings,
               nb::rv_policy::reference,
               R"(
Access the selector's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the active space selector
)");

  // Expose _settings as a writable property for derived classes
  selector.def_prop_rw(
      "_settings",
      [](ActiveSpaceSelector& sel) -> Settings& { return sel.settings(); },
      [](ActiveSpaceSelector& sel,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<ActiveSpaceSelectorBase&>(sel).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MySelector(alg.ActiveSpaceSelector):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()

)");

  selector.def("name", &ActiveSpaceSelector::name,
               R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  selector.def("aliases", &ActiveSpaceSelector::aliases,
               R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm

)");

  selector.def("type_name", &ActiveSpaceSelector::type_name,
               R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  selector.def("hash", &ActiveSpaceSelector::hash, nb::arg("wavefunction"));

  // Factory class binding - creates ActiveSpaceSelectorFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<
      ActiveSpaceSelectorFactory, ActiveSpaceSelector, ActiveSpaceSelectorBase>(
      m, "ActiveSpaceSelectorFactory");

  selector.def("__repr__", [](const ActiveSpaceSelector&) {
    return "<qdk_chemistry.algorithms.ActiveSpaceSelector>";
  });

  qdk::chemistry::python::bind_create_nested(selector);

  // Bind concrete microsoft::OccupationActiveSpaceSelector implementation
  nb::class_<microsoft::OccupationActiveSpaceSelector, ActiveSpaceSelector>(
      m, "QdkOccupationActiveSpaceSelector", R"(
QDK occupation-based active space selector.

This class selects active space orbitals based on their occupation numbers.
It identifies orbitals that have partial occupations (not close to 0 or 2
electrons), which typically indicates significant multi-reference character
and strong electron correlation.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create an occupation-based selector
    selector = alg.QdkOccupationActiveSpaceSelector()

    # Configure occupation threshold
    selector.settings().set("occupation_threshold", 0.1)

    # Select active space
    active_wfn = selector.run(wavefunction)

See Also:
    :class:`ActiveSpaceSelector`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes an occupation-based active space selector with default settings.

)");

  // Bind concrete microsoft::ValenceActiveSpaceSelector implementation
  nb::class_<microsoft::ValenceActiveSpaceSelector, ActiveSpaceSelector>(
      m, "QdkValenceActiveSpaceSelector", R"(
QDK valence-based active space selector.

This class selects active space orbitals based on valence orbital criteria.
It identifies valence orbitals that are chemically significant for the
molecular system under study.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create a valence-based selector
    selector = alg.QdkValenceActiveSpaceSelector()

    # Select active space
    active_wfn = selector.run(wavefunction)

See Also:
    :class:`ActiveSpaceSelector`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a valence-based active space selector with default settings.

)");

  // Bind concrete microsoft::AutocasEosActiveSpaceSelector implementation
  nb::class_<microsoft::AutocasEosActiveSpaceSelector, ActiveSpaceSelector>(
      m, "QdkAutocasEosActiveSpaceSelector", R"(
QDK entropy-based active space selector.

This class selects active space orbitals based on orbital entropy measures.
It identifies orbitals with high entropy, which indicates strong electron
correlation and multi-reference character.
This method is a variant of the autoCAS method that uses entropy as the primary
criterion for selecting active orbitals :cite:`Stein2019`.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create an entropy-based selector
    selector = alg.QdkAutocasEosActiveSpaceSelector()

    # Configure entropy threshold
    selector.settings().set("entropy_threshold", 0.1)

    # Select active space
    active_wfn = selector.run(wavefunction)

See Also:
    :class:`ActiveSpaceSelector`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes an entropy-based active space selector with default settings.

)");

  // Bind concrete microsoft::AutocasActiveSpaceSelector implementation
  nb::class_<microsoft::AutocasActiveSpaceSelector, ActiveSpaceSelector>(
      m, "QdkAutocasActiveSpaceSelector", R"(
QDK Automated Complete Active Space (autoCAS) selector.

This class provides an automated approach to selecting active space orbitals
based on various criteria including occupation numbers, orbital energies,
and other chemical information :cite:`Stein2019`.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create an autoCAS selector
    selector = alg.QdkAutocasActiveSpaceSelector()

    # Select active space automatically
    active_wfn = selector.run(wavefunction)

See Also:
    :class:`ActiveSpaceSelector`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes an autoCAS selector with default settings.

)");
}
