// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <memory>
#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/active_space_qio.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/gauge_fixing.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/mp2_natural_orbitals.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/natural_orbitals.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/pipek_mezey.hpp"
#include "qdk/chemistry/algorithms/microsoft/localization/vvhv.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class LocalizerBase : public Localizer {
 public:
  NB_TRAMPOLINE(Localizer);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction,
      const std::vector<size_t> &loc_indices_a,
      const std::vector<size_t> &loc_indices_b) const override {
    NB_OVERRIDE_PURE(_run_impl, wavefunction, loc_indices_a, loc_indices_b);
  }
};

void bind_localizer(nb::module_ &m) {
  m.def(
      "new_aufbau_determinant_wavefunction",
      [](std::shared_ptr<Wavefunction> wavefunction,
         std::shared_ptr<Orbitals> orbitals) {
        return qdk::chemistry::algorithms::detail::
            new_aufbau_determinant_wavefunction(std::move(wavefunction),
                                                std::move(orbitals));
      },
      nb::arg("wavefunction"), nb::arg("orbitals"), R"(
Create an Aufbau determinant wavefunction for an orbital basis.

The determinant is the canonical Aufbau determinant implied by the input
wavefunction's electron counts and the supplied orbitals. If the orbitals define
an active space, the determinant is projected into that active space.

Args:
    wavefunction (qdk_chemistry.data.Wavefunction): Wavefunction providing electron counts
    orbitals (qdk_chemistry.data.Orbitals): Orbital basis for the returned wavefunction

Returns:
    qdk_chemistry.data.Wavefunction: Aufbau determinant wavefunction with the supplied orbitals
)");

  // Localizer abstract base class
  nb::class_<Localizer, LocalizerBase> localizer(m, "OrbitalLocalizer", R"(
Abstract base class for orbital localization and transformation algorithms.

This class defines the interface for selecting and transforming molecular orbitals.
Implementations may produce spatially localized orbitals or other representations,
such as natural orbitals. Concrete implementations should inherit from this class
and implement the _run_impl method.

Examples:
    >>> # To create a custom orbital localizer, inherit from this class.
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyLocalizer(alg.OrbitalLocalizer):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     # Implement the _run_impl method
    ...     def _run_impl(self, wavefunction: data.Wavefunction, loc_indices_a: list, loc_indices_b: list) -> data.Wavefunction:
    ...         # Custom orbital transformation implementation
    ...         return transformed_wavefunction
)");

  localizer.def(nb::init<>(),
                R"(
Create an OrbitalLocalizer instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MyLocalizer(alg.OrbitalLocalizer):
    ...     def __init__(self):
    ...         super().__init__()  # Calls this constructor

)");

  localizer.def("run", &Localizer::run,
                R"(
Transform selected molecular orbitals in the given wavefunction.

Args:
    wavefunction (qdk_chemistry.data.Wavefunction): The molecular wavefunction to transform
    loc_indices_a (list[int]): Indices of alpha orbitals to transform; empty selects none
    loc_indices_b (list[int]): Indices of beta orbitals to transform; empty selects none

Notes:
    For restricted orbitals, ``loc_indices_b`` must match ``loc_indices_a``.
    If both index lists are empty, the orbital transformation is a no-op.

Returns:
    qdk_chemistry.data.Wavefunction: The output wavefunction with transformed orbitals

Raises:
    ValueError: If orbital indices are invalid or inconsistent
    RuntimeError: If the transformation fails due to numerical issues

)",
                nb::arg("wavefunction"), nb::arg("loc_indices_a"),
                nb::arg("loc_indices_b"));

  localizer.def("settings", &Localizer::settings,
                R"(
Access the localizer's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the localizer

)",
                nb::rv_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  localizer.def_prop_rw(
      "_settings", [](Localizer &loc) -> Settings & { return loc.settings(); },
      [](Localizer &loc,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<LocalizerBase &>(loc).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyLocalizer(alg.OrbitalLocalizer):
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

  localizer.def("aliases", &Localizer::aliases,
                R"(
 The algorithm's aliases.

 Returns:
    list[str]: All registered names for the algorithm
)");

  localizer.def("type_name", &Localizer::type_name,
                R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  localizer.def("hash", &Localizer::hash, nb::arg("wavefunction"),
                nb::arg("loc_indices_a"), nb::arg("loc_indices_b"));

  // Factory class binding - creates LocalizerFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<LocalizerFactory, Localizer,
                                                 LocalizerBase>(
      m, "LocalizerFactory");

  localizer.def("__repr__", [](const Localizer &) {
    return "<qdk_chemistry.algorithms.OrbitalLocalizer>";
  });

  qdk::chemistry::python::bind_create_nested(localizer);

  // Bind concrete microsoft::PipekMezeyLocalizer implementation
  nb::class_<microsoft::PipekMezeyLocalizer, Localizer>(
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
      .def(nb::init<>(), R"(
Default constructor.

Initializes a Pipek-Mezey localizer with default settings.

)");

  // Bind concrete microsoft::MP2NaturalOrbitalLocalizer implementation.
  // This intentionally exposes the deprecated MP2 localizer as a Python facade
  // (it warns at runtime via warn_if_deprecated_algorithm), so suppress the
  // compile-time deprecation warning for naming the deprecated class here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  nb::class_<microsoft::MP2NaturalOrbitalLocalizer, Localizer>(
      m, "QdkMP2NaturalOrbitalLocalizer", R"(
QDK MP2 natural orbital transformer.

.. deprecated:: 2.0.0
    Use :class:`QdkNaturalOrbitalLocalizer` (``qdk_natural_orbitals``) with a
    wavefunction that already contains the active-space one-particle reduced
    density matrix (1-RDM).

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
      .def(nb::new_([]() {
             auto instance =
                 std::make_unique<microsoft::MP2NaturalOrbitalLocalizer>();
             qdk::chemistry::python::warn_if_deprecated_algorithm(*instance);
             return instance;
           }),
           R"(
Default constructor.

Initializes an MP2 natural orbital transformer with default settings.

)");
#pragma GCC diagnostic pop

  // Bind concrete microsoft::NaturalOrbitalLocalizer implementation
  nb::class_<microsoft::NaturalOrbitalLocalizer, Localizer>(
      m, "QdkNaturalOrbitalLocalizer", R"(
QDK natural orbital transformer.

This class provides a concrete implementation that transforms molecular
orbitals into natural orbitals by diagonalizing the spin-traced one-particle
reduced density matrix (1-RDM). Natural orbitals are eigenfunctions of the
1-RDM, and their eigenvalues are the occupation numbers.

.. note::
    Requires ``loc_indices_a == loc_indices_b`` (natural orbitals are a single set).
    Requires ``loc_indices_a``/``loc_indices_b`` to match the orbitals' active-space indices exactly.
    Requires a spin-traced 1-RDM and an active space in the wavefunction.
    For unrestricted Slaterdeterminant, the 1-RDM is expressed in the
    alpha MO basis and the output is always a restricted set of natural orbitals.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create a natural orbital localizer
    localizer = alg.QdkNaturalOrbitalLocalizer()

    # Transform to natural orbitals
    no_wfn = localizer.run(wavefunction, active_indices, active_indices)

See Also:
    :class:`OrbitalLocalizer`
    :class:`QdkMP2NaturalOrbitalLocalizer`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a natural orbital transformer with default settings.

)");

  // Bind concrete microsoft::GaugeFixingLocalizer implementation
  nb::class_<microsoft::GaugeFixingLocalizer, Localizer>(
      m, "QdkGaugeFixingLocalizer", R"(
QDK gauge-fixing orbital localizer.

Orbitals with equal occupation numbers span a well-defined subspace without
being uniquely determined within it. This class resolves that freedom
deterministically, using bounded coordinate-descent sweeps to reduce the mapped
Hamiltonian coefficient 1-norm ``lambda = sum_l |h_l|``, which sets the evolution
time in phase estimation and the normalization of a linear-combination-of-unitaries
block encoding.

.. note::
    Requires restricted orbitals, an active space, an overlap matrix, and a
    spin-traced active 1-RDM that is diagonal in the input orbital basis, which
    is what :class:`QdkNaturalOrbitalLocalizer` produces. Requires
    ``loc_indices_a == loc_indices_b``, and those indices must be a subset of
    the active-space indices that does not split a degenerate block.

See Also:
    :class:`OrbitalLocalizer`
    :class:`QdkNaturalOrbitalLocalizer`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a gauge-fixing localizer with default settings.

)");

  // Bind concrete microsoft::ActiveSpaceQIOLocalizer implementation
  nb::class_<microsoft::ActiveSpaceQIOLocalizer, Localizer>(
      m, "QdkActiveSpaceQIOLocalizer", R"(
QDK quantum-information orbital (QIO) active-space localizer.

Rotates restricted active orbitals to minimize the total single-orbital entropy
using gradient-free Jacobi sweeps.

This class minimizes the QIO objective restricted to rotations within a fixed
active space. It does not implement full-space QIO or QICAS, both of which mix
orbitals across the active-space boundary.

.. note::
    Requires a restricted active orbital space, matching alpha and beta
    localization indices, and spin-dependent active-space 1- and 2-RDMs.

See Also:
    :class:`OrbitalLocalizer`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a quantum-information orbital localizer with default settings.

)");

  // Bind concrete microsoft::VVHVLocalizer implementation
  nb::class_<microsoft::VVHVLocalizer, Localizer>(m, "QdkVVHVLocalizer", R"(
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
      .def(nb::init<>(), R"(
Default constructor.

Initializes a VV-HV localizer with default settings.

)");
}
