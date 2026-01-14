Stability analysis
==================

The :class:`~qdk_chemistry.algorithms.StabilityChecker` algorithm in QDK/Chemistry performs wavefunction stability analysis to verify that a Self-Consistent Field (:term:`SCF`) solution corresponds to a true energy minimum rather than a saddle point.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Wavefunction <../data/wavefunction>` instance as input and produces a :class:`~qdk_chemistry.data.StabilityResult` object containing stability information as output.
Its primary purpose is to identify directions in orbital space where the energy can be further lowered, which is crucial for ensuring the reliability of quantum chemistry calculations.

Unstable :term:`SCF` solutions can occur in challenging cases such as stretched molecular geometries, open-shell systems, and molecules with near-degenerate orbitals.
This iterative stability analysis of checking stability, rotating orbitals, and re-running :term:`SCF` continues until a stable minimum is reached.

Overview
--------

Stability analysis :cite:`Schlegel1991` examines the second-order response of the wavefunction energy to orbital rotations.
Mathematically, this involves computing the eigenvalues of the electronic Hessian matrix.
A stable wavefunction should have all positive (or non-negative) eigenvalues, indicating that the energy cannot be lowered by any orbital rotation.
Negative eigenvalues signal instabilities and identify directions in which the energy can decrease.

There are two types of stability that can be checked:

Internal stability
  Examines whether the wavefunction is stable with respect to rotations within the same wavefunction type.
  For example, restricted Hartree-Fock (:term:`RHF`) internal stability checks if the energy can be lowered while maintaining the restricted formalism.

External stability
  Examines whether the wavefunction is stable with respect to transitions to a different wavefunction type.
  The most common case is :term:`RHF` external stability, which checks for :term:`RHF` to unrestricted Hartree-Fock (:term:`UHF`) instabilities.
  This is particularly important for systems that may benefit from spin symmetry breaking, such as stretched bonds or transition metal complexes.

The typical workflow for handling instabilities involves an iterative procedure:

1. Run an :term:`SCF` calculation to obtain initial orbitals
2. Check stability of the resulting wavefunction
3. If unstable, extract the eigenvector corresponding to the most negative eigenvalue
4. Rotate orbitals along this eigenvector direction
5. Use rotated orbitals as initial guess for a new :term:`SCF` calculation
6. Repeat until a stable solution is found

When an external instability is detected in a restricted calculation, the procedure typically switches to an unrestricted calculation type, as the instability indicates that spin symmetry breaking would lower the energy.

Running a stability check
--------------------------

This section demonstrates how to setup, configure, and run a stability check.
The ``run`` method returns two values: a boolean indicating overall stability status and a :class:`~qdk_chemistry.data.StabilityResult` object containing detailed analysis results including eigenvalues and eigenvectors.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.StabilityChecker` requires a converged wavefunction from an :term:`SCF` calculation:

Wavefunction
  A :doc:`Wavefunction <../data/wavefunction>` instance containing orbital information.
  This is typically obtained as output from a :class:`~qdk_chemistry.algorithms.ScfSolver` calculation.

.. rubric:: Creating a stability checker

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/stability_checker_workflow.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/stability_checker_workflow.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available settings`_ below for a complete list of options.
Key settings include whether to perform internal and external stability checks, and the tolerance for determining stability.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/stability_checker_workflow.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/stability_checker_workflow.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the stability check and iterative workflow

The example below shows a complete iterative workflow for achieving a stable wavefunction.
This includes running an initial :term:`SCF` calculation, checking stability, and iteratively rotating orbitals and re-running :term:`SCF` until convergence to a stable solution.

The workflow handles both internal instabilities (requiring orbital rotation within the same calculation type) and external instabilities (requiring a switch from restricted to unrestricted calculation).
Note that the Davidson eigenvalue solver used in stability analysis may occasionally fail to converge; when this occurs, increasing ``max_subspace`` or adjusting ``davidson_tolerance`` can help.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/stability_checker_workflow.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/stability_checker_workflow.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run


Available settings
------------------

The :class:`~qdk_chemistry.algorithms.StabilityChecker` accepts settings to control stability analysis behavior.
All implementations share a common base set of settings:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``internal``
     - bool
     - ``True``
     - Check internal stability (within same wavefunction type)
   * - ``external``
     - bool
     - Implementation-dependent
     - Check external stability (:term:`RHF` → :term:`UHF` instabilities). Only supported for :term:`RHF` wavefunctions
   * - ``method``
     - string
     - ``"hf"``
     - The method to match the wavefunction: ``"hf"`` for Hartree-Fock, or a :term:`DFT` functional name
   * - ``stability_tolerance``
     - float
     - ``-1e-4``
     - Eigenvalue threshold for determining stability. Eigenvalues below this value indicate instability
   * - ``davidson_tolerance``
     - float
     - ``1e-8``
     - Convergence tolerance for the Davidson eigenvalue solver
   * - ``max_subspace``
     - int
     - ``80``
     - Maximum subspace dimension for the Davidson solver

See :doc:`Settings <settings>` for a more general treatment of settings in QDK/Chemistry.

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.StabilityChecker` provides a unified interface to stability analysis across various quantum chemistry packages.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/stability_checker_workflow.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/stability_checker_workflow.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. _qdk-stability-native:

QDK (Native)
~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk"`` (default)

The native QDK/Chemistry implementation provides high-performance stability analysis using the built-in quantum chemistry engine.

.. rubric:: Capabilities

- Internal stability analysis for :term:`RHF` and :term:`UHF` wavefunctions
- External stability analysis for :term:`RHF` wavefunctions (:term:`RHF` → :term:`UHF` instabilities)
- Efficient Davidson eigenvalue solver for computing lowest eigenvalues
- Support for both Hartree-Fock and :term:`DFT` wavefunctions

.. rubric:: Settings

The QDK implementation uses the common base settings listed in the `Available settings`_ section above.
The default value for ``external`` is ``False`` for this implementation.

PySCF
~~~~~

.. rubric:: Factory name: ``"pyscf"``

The PySCF plugin provides access to stability analysis through the `PySCF <https://pyscf.org/>`_ quantum chemistry package.

.. rubric:: Capabilities

- Internal stability analysis for :term:`RHF`, restricted open-shell Hartree-Fock (:term:`ROHF`), and :term:`UHF` wavefunctions
- External stability analysis for :term:`RHF` wavefunctions (:term:`RHF` → :term:`UHF` instabilities)
- Support for both Hartree-Fock and :term:`DFT` wavefunctions
- Point group symmetry considerations in stability analysis
- Configurable Davidson solver parameters and PySCF verbosity
- Note: ROHF wavefunctions are not supported by the QDK backend for now

.. rubric:: Settings

The PySCF implementation supports all common base settings. The following table shows settings that differ from or extend the base implementation:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``external``
     - bool
     - ``True``
     - Check external stability (differs from QDK default of ``False``)
   * - ``with_symmetry``
     - bool
     - ``False``
     - Whether to respect point group symmetry during analysis
   * - ``nroots``
     - int
     - ``3``
     - Number of eigenvalue roots to compute in the Davidson solver
   * - ``xc_grid``
     - int
     - ``3``
     - :term:`DFT` integration grid density level (0=coarse, 9=very fine)
   * - ``pyscf_verbose``
     - int
     - ``4``
     - PySCF verbosity level for logging (0=silent, 4=info, 5=debug)

.. note::

   The PySCF implementation automatically detects the wavefunction type (:term:`RHF`, :term:`ROHF`, or :term:`UHF`) and applies the appropriate stability analysis method.
   External stability analysis is only supported for :term:`RHF` wavefunctions and will raise an error if requested for :term:`ROHF` or :term:`UHF`.

For more details on how to extend QDK/Chemistry with additional implementations, see the :doc:`plugin system <../plugins>` documentation.

Understanding stability results
-------------------------------

The :class:`~qdk_chemistry.data.StabilityResult` object returned by the stability checker contains detailed information about the stability analysis:

Overall stability status
  The boolean return value indicates whether the wavefunction is stable overall (both internally and externally if checked).

Internal stability
  The :meth:`~qdk_chemistry.data.StabilityResult.is_internal_stable` method indicates whether internal stability is satisfied.
  Internal instabilities suggest the energy can be lowered while maintaining the same wavefunction type.

External stability
  The :meth:`~qdk_chemistry.data.StabilityResult.is_external_stable` method indicates whether external stability is satisfied (if external stability was checked).
  External instabilities in :term:`RHF` calculations indicate that breaking spin symmetry (switching to :term:`UHF`) would lower the energy.

Eigenvalues and eigenvectors
  The stability result provides access to the computed eigenvalues and their corresponding eigenvectors:

  - :meth:`~qdk_chemistry.data.StabilityResult.get_internal_eigenvalues` and :meth:`~qdk_chemistry.data.StabilityResult.get_internal_eigenvectors`
  - :meth:`~qdk_chemistry.data.StabilityResult.get_external_eigenvalues` and :meth:`~qdk_chemistry.data.StabilityResult.get_external_eigenvectors`
  - :meth:`~qdk_chemistry.data.StabilityResult.get_smallest_eigenvalue` returns the overall smallest eigenvalue
  - :meth:`~qdk_chemistry.data.StabilityResult.get_smallest_internal_eigenvalue_and_vector` and :meth:`~qdk_chemistry.data.StabilityResult.get_smallest_external_eigenvalue_and_vector` provide convenient access to the most unstable direction

The eigenvector corresponding to the smallest (most negative) eigenvalue indicates the direction in orbital space that would most effectively lower the energy.
This eigenvector can be used with the :func:`~qdk_chemistry.utils.rotate_orbitals` utility function to generate rotated orbitals for use as an initial guess in a subsequent :term:`SCF` calculation.

Related classes
---------------

- :doc:`ScfSolver <scf_solver>`: Performs :term:`SCF` calculations that produce wavefunctions for stability analysis
- :doc:`Wavefunction <../data/wavefunction>`: Input wavefunction to analyze
- :doc:`Orbitals <../data/orbitals>`: Can be rotated based on stability analysis results

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/stability_checker_workflow.py>`_ script or `C++ <../../../_static/examples/cpp/stability_checker_workflow.cpp>`_ source file.
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
- :cite:`Schlegel1991`: Original reference for stability analysis in quantum chemistry
