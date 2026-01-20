Self-consistent field (SCF) solver
==================================

The :class:`~qdk_chemistry.algorithms.ScfSolver` algorithm in QDK/Chemistry performs Self-Consistent Field (:term:`SCF`) calculations to optimize molecular orbitals for a given molecular structure.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Structure <../data/structure>` instance, total molecular charge and multiplicity, and a desired basis representation as input ,and produces an :doc:`Orbitals <../data/orbitals>` instance and the associated energy as output.
Its primary purpose is to find the best single-particle orbitals within a mean-field approximation.
For Hartree-Fock (:term:`HF`) theory, it yields the mean field energy, which misses electron correlation and typically requires post-:term:`HF` methods for accurate energetics.
For Density Functional Theory (:term:`DFT`), some correlation effects are included through the exchange-correlation functional.

Overview
--------

:term:`SCF` theory encompasses both :term:`HF` and :term:`DFT` methods in quantum chemistry.
Both methods rely on a single Slater determinant representation of the many-electron wavefunction, using molecular orbitals that are optimized to minimize the electronic energy.
This single-determinant approach is a key simplification that makes these methods computationally efficient but limits their ability to capture certain correlation effects.
The :term:`SCF` procedure iteratively refines these orbitals until self-consistency is achieved.


The orbitals from :term:`SCF` calculations typically serve as input for these post-:term:`SCF` methods which capture correlation effects.
:term:`SCF` methods thus serve as the foundation for more advanced quantum and classical electronic structure calculations and provide essential insights into molecular properties, reactivity, and spectroscopic characteristics.

Running an :term:`SCF` calculation
----------------------------------

This section demonstrates how to setup, configure, and run a :term:`SCF` calculation.
The ``run`` method returns two values: a scalar representing the converged :term:`SCF` energy and a :class:`~qdk_chemistry.data.Wavefunction` object containing the optimized molecular orbitals.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.ScfSolver` requires several inputs to perform a calculation:

Structure
  A :doc:`Structure <../data/structure>` instance defining the molecular geometry (atomic positions and element types).

Charge
  The total molecular charge (integer). A neutral molecule has charge 0.

Spin multiplicity
  The spin multiplicity of the system, defined as :math:`2S + 1` where :math:`S` is the total spin. Common values are 1 (singlet), 2 (doublet), 3 (triplet), etc.

Basis set or initial guess
  This required input specifies the atomic orbital basis for the calculation and can be provided in several forms:

  String
    A standard basis set name (e.g., ``"sto-3g"``, ``"def2-svp"``, ``"cc-pvdz"``). See the :doc:`basis set documentation <../basis_functionals>` for available options.
  BasisSet object
    A :class:`~qdk_chemistry.data.BasisSet` instance for custom basis sets. See the :doc:`BasisSet <../data/basis_set>` documentation for details.
  Orbitals object
    A :class:`~qdk_chemistry.data.Orbitals` instance which provides both the basis set and an initial orbital guess for the :term:`SCF` optimization.


.. rubric:: Creating a :term:`SCF` solver

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/scf_solver.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/scf_solver.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available settings`_ below for a complete list of options.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/scf_solver.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/scf_solver.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/scf_solver.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/scf_solver.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

.. rubric:: Alternative run options

The ``run`` method also accepts either :doc:`Orbitals <../data/orbitals>` as an initial guess or a :doc:`BasisSet <../data/basis_set>` object.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/scf_solver.cpp
      :language: cpp
      :start-after: // start-cell-alternative-run
      :end-before: // end-cell-alternative-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/scf_solver.py
      :language: python
      :start-after: # start-cell-alternative-run
      :end-before: # end-cell-alternative-run


Available settings
------------------

The :class:`~qdk_chemistry.algorithms.ScfSolver` accepts a range of settings to control its behavior.
All implementations share a common base set of settings from ``ElectronicStructureSettings``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``method``
     - string
     - ``"hf"``
     - The method to use: ``"hf"`` for Hartree-Fock, or a :term:`DFT` functional name (e.g., ``"b3lyp"``, ``"pbe"``)
   * - ``convergence_threshold``
     - float
     - ``1e-7``
     - Convergence tolerance for orbital gradient norm
   * - ``max_iterations``
     - int
     - ``50``
     - Maximum number of :term:`SCF` iterations (must be ≥ 1)

See :doc:`Settings <settings>` for a more general treatment of settings in QDK/Chemistry.

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.ScfSolver` provides a unified interface to :term:`SCF` calculations across various quantum chemistry packages.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/scf_solver.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/scf_solver.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. _qdk-scf-native:

QDK (Native)
~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk"`` (default)

The native QDK/Chemistry implementation provides high-performance :term:`SCF` calculations using the built-in quantum chemistry engine.

.. rubric:: Capabilities

- Restricted Hartree-Fock (:term:`RHF`) and Unrestricted Hartree-Fock (:term:`UHF`)
- Restricted Kohn-Sham (:term:`RKS`) and Unrestricted Kohn-Sham (:term:`UKS`) :term:`DFT`
- Extensive library of :doc:`basis sets <../basis_functionals>` including Pople, Dunning, and Karlsruhe families
- Full range of :doc:`exchange-correlation functionals <../basis_functionals>` for :term:`DFT`
  - Optimization algorithms including the direct inversion in the iterative subspace (:term:`DIIS`) method :cite:`Pulay1982`, and the geometric direct minimization (:term:`GDM`) method :cite:`VanVoorhis2002`

.. _scf-convergence-algorithms:

SCF Convergence Algorithms in QDK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Achieving stable :term:`SCF` convergence is a non-trivial problem in computational chemistry.
QDK/Chemistry implements two complementary algorithms that can be used independently or in combination.

.. rubric:: Direct Inversion in the Iterative Subspace (DIIS)

:term:`DIIS` is an extrapolation technique that accelerates :term:`SCF` convergence by constructing an optimal linear combination of previous Fock matrices :cite:`Pulay1982`.
:term:`DIIS` is highly effective for well-behaved systems, often achieving convergence in low number of  iterations.
However, it can fail for challenging cases such as open-shell systems or molecules with near-degenerate orbitals, where the error surface is highly nonlinear.

.. rubric:: Geometric Direct Minimization (GDM)

When :term:`DIIS` encounters difficulties, the :term:`GDM` algorithm provides a robust alternative :cite:`VanVoorhis2002`.
Rather than extrapolating Fock matrices, :term:`GDM` directly minimizes the energy with respect to orbital rotation parameters using a quasi-Newton optimization approach.

The key insight of :term:`GDM` is to parameterize orbital changes through unitary rotations, which converts the constrained optimization problem of determining the energy-minimizing set of orthonormal orbitals into an unconstrained optimization over exponentials :cite:`Higham2005` of anti-Hermitian matrices.
This allows the use of standard nonlinear optimization techniques while preserving orbital orthonormality.


The :term:`GDM` algorithm then proceeds via a slightly modified :cite:`VanVoorhis2002` :term:`BFGS` optimization :cite:`Liu1989` which smoothly converges to a nearby energy minimum. If provided a guess close to the true minimum, :term:`GDM` can converge in a similar number of iterations as :term:`DIIS`, but it is more robust for difficult cases. However, if initialized further from the minimum, :term:`GDM` may converge to local minima, which may require additional strategies (e.g. :doc:`Stability analysis<stability_checker>`) to ensure the global minimum is found. This may be overcome in many cases by combining :term:`GDM` with :term:`DIIS` in a hybrid approach.

.. rubric:: Hybrid DIIS-GDM Strategy

By default, the native QDK implementation uses :term:`DIIS` alone (``enable_gdm=False``).
When enabled, the hybrid strategy (``enable_gdm=True``) provides enhanced robustness:

1. Start with :term:`DIIS` for rapid initial convergence
2. Monitor energy changes; if the energy change exceeds ``energy_thresh_diis_switch`` (default: :math:`10^{-3}` Ha), switch to :term:`GDM`
3. Once switched, continue with :term:`GDM` until convergence

This hybrid approach combines the speed of :term:`DIIS` for typical systems with the robustness of :term:`GDM` for challenging cases.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``method``
     - string
     - ``"hf"``
     - Method: ``"hf"`` for Hartree-Fock, or a :term:`DFT` functional name
   * - ``convergence_threshold``
     - float
     - ``1e-7``
     - Convergence tolerance for orbital gradient norm
   * - ``max_iterations``
     - int
     - ``50``
     - Maximum number of :term:`SCF` iterations
   * - ``max_scf_steps``
     - int
     - ``100``
     - Maximum number of overall :term:`SCF` steps
   * - ``enable_gdm``
     - bool
     - ``False``
     - Enable geometric direct minimization (:term:`GDM`) algorithm
   * - ``gdm_max_diis_iteration``
     - int
     - ``50``
     - Maximum :term:`DIIS` iterations in :term:`GDM`
   * - ``gdm_bfgs_history_size_limit``
     - int
     - ``50``
     - :term:`BFGS` history size limit for :term:`GDM`
   * - ``energy_thresh_diis_switch``
     - float
     - ``0.001``
     - Energy threshold for :term:`DIIS` switch
   * - ``level_shift``
     - float
     - ``-1.0``
     - Level shift parameter (negative = auto)
   * - ``eri_threshold``
     - float
     - ``-1.0``
     - Electron repulsion integral threshold (negative = auto)
   * - ``eri_use_atomics``
     - bool
     - ``False``
     - Use atomic operations for :term:`ERI` computation
   * - ``fock_reset_steps``
     - int
     - ``1073741824``
     - Number of steps between Fock matrix resets

PySCF
~~~~~

.. rubric:: Factory name: ``"pyscf"``

The PySCF plugin provides access to the comprehensive `PySCF <https://pyscf.org/>`_ quantum chemistry package.

.. rubric:: Capabilities

- Full :term:`HF` support: :term:`RHF`, :term:`UHF`, :term:`ROHF`
- Full :term:`DFT` support: :term:`RKS`, :term:`UKS`, :term:`ROKS` with extensive functional library
- Automatic spin-restricted/unrestricted selection based on multiplicity

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``method``
     - string
     - ``"hf"``
     - Method: ``"hf"`` for Hartree-Fock, or a :term:`DFT` functional name
   * - ``convergence_threshold``
     - float
     - ``1e-7``
     - Convergence tolerance for orbital gradient norm
   * - ``max_iterations``
     - int
     - ``50``
     - Maximum number of :term:`SCF` iterations
   * - ``scf_type``
     - string
     - ``"auto"``
     - Type of :term:`SCF` calculation:

       * ``"auto"``: Automatically detect based on spin
       * ``"restricted"``: Force restricted calculation
       * ``"unrestricted"``: Force unrestricted calculation

.. rubric:: Example

.. literalinclude:: ../../../_static/examples/python/scf_solver.py
   :language: python
   :start-after: # start-cell-pyscf-example
   :end-before: # end-cell-pyscf-example

For more details on how to extend QDK/Chemistry with additional implementations, see the :doc:`plugin system <../plugins>` documentation.

Related classes
---------------

- :doc:`Structure <../data/structure>`: Input molecular structure
- :doc:`Orbitals <../data/orbitals>`: Output optimized molecular orbitals

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/scf_solver.py>`_ script or `C++ <../../../_static/examples/cpp/scf_solver.cpp>`_ source file.
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
- :doc:`../basis_functionals`: Exchange-correlation functionals for :term:`DFT` calculations
