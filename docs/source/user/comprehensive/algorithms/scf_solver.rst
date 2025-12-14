Self-consistent field (SCF) solver
==================================

The :class:`~qdk_chemistry.algorithms.ScfSolver` algorithm in QDK/Chemistry performs Self-Consistent Field (SCF) calculations to optimize molecular orbitals for a given molecular structure.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Structure <../data/structure>` instance as input and produces an :doc:`Orbitals <../data/orbitals>` instance as output.
Its primary purpose is to find the best single-particle orbitals within a mean-field approximation.
For Hartree-Fock (HF) theory, it yields the mean field energy, which misses electron correlation and typically requires post-HF methods for accurate energetics.
For Density Functional Theory (DFT), some correlation effects are included through the exchange-correlation functional.

Overview
--------

:term:`SCF` theory encompasses both :term:`HF` and :term:`DFT` methods in quantum chemistry.
Both methods rely on a single Slater determinant representation of the many-electron wavefunction, using molecular orbitals that are optimized to minimize the electronic energy.
This single-determinant approach is a key simplification that makes these methods computationally efficient but limits their ability to capture certain correlation effects.
The :term:`SCF` procedure iteratively refines these orbitals until self-consistency is achieved.

:term:`SCF` methods provide an excellent starting point, but they miss important electronic correlation effects:

- **Static correlation**: Essential for systems with near-degenerate states or bond-breaking processes.
  See the :doc:`MCCalculator <mc_calculator>` documentation for electronic structure methods targeted at capturing static correlation.
- **Dynamical correlation**: Required for all molecular systems to account for instantaneous electron-electron interactions.
  See the :doc:`DynamicalCorrelationCalculator <dynamical_correlation>` documentation for electronic structure methods targeting dynamical correlation.

The orbitals from :term:`SCF` calculations typically serve as input for these post-:term:`SCF` methods which capture correlation effects.
:term:`SCF` methods thus serve as the foundation for more advanced quantum and classical electronic structure calculations and provide essential insights into molecular properties, reactivity, and spectroscopic characteristics.

Running an :term:`SCF` calculation
----------------------------------

This section demonstrates how to setup, configure, and run a SCF calculation.
The ``run`` method returns two values: a scalar representing the converged SCF energy and an :doc:`Orbitals <../data/orbitals>` object containing the optimized molecular orbitals.

**Creating a SCF solver:**

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

**Configuring settings:**

Settings can be modified using the ``settings()`` object.
See `Available settings`_ below for a complete list of options, or :doc:`basis sets and functionals <../basis_functionals>` for supported basis sets.

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

**Running the calculation:**

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
     - The method to use: ``"hf"`` for Hartree-Fock, or a DFT functional name (e.g., ``"b3lyp"``, ``"pbe"``)
   * - ``basis_set``
     - string
     - ``"def2-svp"``
     - The basis set to use for the calculation; see :doc:`basis sets documentation <../basis_functionals>`
   * - ``convergence_threshold``
     - float
     - ``1e-7``
     - Convergence tolerance for orbital gradient norm
   * - ``max_iterations``
     - int
     - ``50``
     - Maximum number of SCF iterations (must be ≥ 1)

See :doc:`Settings <settings>` for a more general treatment of settings in QDK/Chemistry.
Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.ScfSolver` provides a unified interface to SCF calculations across various quantum chemistry packages.
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

**Factory name:** ``"qdk"`` (default)

The native QDK/Chemistry implementation provides high-performance SCF calculations using the built-in quantum chemistry engine.

**Capabilities:**

- Restricted Hartree-Fock (RHF) and Unrestricted Hartree-Fock (UHF)
- Restricted Kohn-Sham (RKS) and Unrestricted Kohn-Sham (UKS) DFT
- Extensive library of :doc:`basis sets <../basis_functionals>` including Pople, Dunning, and Karlsruhe families
- Full range of :doc:`exchange-correlation functionals <../basis_functionals>` for DFT
- Advanced convergence algorithms including the direct inversion in the iterative subspace (DIIS) method :cite:`Pulay1982`, and the geometric direct minimization (GDM) method :cite:`VanVoorhis2002`

.. _scf-convergence-algorithms:

SCF Convergence Algorithms in QDK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Achieving stable SCF convergence is a non-trivial problem in computational chemistry.
QDK/Chemistry implements two complementary algorithms that can be used independently or in combination.

**Direct Inversion in the Iterative Subspace (DIIS)**

DIIS is an extrapolation technique that accelerates SCF convergence by constructing an optimal linear combination of previous Fock matrices :cite:`Pulay1982`.
At each iteration, the algorithm. DIIS is highly effective for well-behaved systems, often achieving convergence in 10–20 iterations.
However, it can fail for challenging cases such as transition metal complexes, open-shell systems, or molecules with near-degenerate orbitals, where the error surface is highly non-linear.

**Geometric Direct Minimization (GDM)**

When DIIS encounters difficulties, the GDM algorithm provides a robust alternative :cite:`VanVoorhis2002`.
Rather than extrapolating Fock matrices, GDM directly minimizes the energy with respect to orbital rotation parameters using a quasi-Newton optimization approach.

The key insight of GDM is to parameterize orbital changes through unitary rotations, which converts the constrained optimizxation problem of determining the energy-minimizing set of orthonormal orbitals into an unconstrained optimization over anti-Hermitian matrices.
This allows the use of standard nonlinear optimization techniques while preserving orbital orthonormality.
Given a set of orbital coefficients, :math:`\mathbf{C}`, new orbitals, :math:`\mathbf{C}'`, are generated via:

.. math::

   \mathbf{C}' = \mathbf{C} \exp(\mathbf{X})

where :math:`\mathbf{X}` is an anti-Hermitian matrix containing non-zeros only in the occupied-virtual blocks. The matrix exponential :math:`\exp(\mathbf{X})`, computed by scaling and squaring within the QDK :cite:`Higham2005`, generates a unitary transformation that preserves orbital orthonormality.

The GDM algorithm then proceeds via a slightly modified :cite:`VanVoorhis2002` BFGS optimization :cite:`Liu1989` which smoothly converges to a nearby energy minimum. If provided a guess close to the true minimum, GDM can converge in a similar number of iterations as DIIS, but it is more robust for difficult cases. However, if initialized further from the minimum, GDM may converge to local minima, which may require additional strategies (e.g. :doc:`Stability analysis<stability_checker>`) to ensure the global minimum is found. This may be overcome in many cases by combining GDM with DIIS in a hybrid approach.



**Hybrid DIIS-GDM Strategy**

By default, the native QDK implementation uses DIIS alone (``enable_gdm=False``).
When enabled, the hybrid strategy (``enable_gdm=True``) provides enhanced robustness:

1. Start with DIIS for rapid initial convergence
2. Monitor energy changes; if the energy change exceeds ``energy_thresh_diis_switch`` (default: :math:`10^{-3}` Ha), switch to GDM
3. Once switched, continue with GDM until convergence

This hybrid approach combines the speed of DIIS for typical systems with the robustness of GDM for challenging cases.

**Settings:**

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
     - Method: ``"hf"`` for Hartree-Fock, or a DFT functional name
   * - ``basis_set``
     - string
     - ``"def2-svp"``
     - Basis set for the calculation
   * - ``convergence_threshold``
     - float
     - ``1e-7``
     - Convergence tolerance for orbital gradient norm
   * - ``max_iterations``
     - int
     - ``50``
     - Maximum number of SCF iterations
   * - ``max_scf_steps``
     - int
     - ``100``
     - Maximum number of overall SCF steps
   * - ``enable_gdm``
     - bool
     - ``False``
     - Enable geometric direct minimization (GDM) algorithm
   * - ``gdm_max_diis_iteration``
     - int
     - ``50``
     - Maximum DIIS iterations in GDM
   * - ``gdm_bfgs_history_size_limit``
     - int
     - ``50``
     - BFGS history size limit for GDM
   * - ``energy_thresh_diis_switch``
     - float
     - ``0.001``
     - Energy threshold for DIIS switch
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
     - Use atomic operations for ERI computation
   * - ``fock_reset_steps``
     - int
     - ``1073741824``
     - Number of steps between Fock matrix resets

PySCF
~~~~~

**Factory name:** ``"pyscf"``

The PySCF plugin provides access to the comprehensive `PySCF <https://pyscf.org/>`_ quantum chemistry package.

**Capabilities:**

- Full HF support: RHF, UHF, ROHF
- Full DFT support: RKS, UKS, ROKS with extensive functional library
- Automatic spin-restricted/unrestricted selection based on multiplicity
- Support for custom basis sets and effective core potentials (ECPs)

**Settings:**

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
     - Method: ``"hf"`` for Hartree-Fock, or a DFT functional name
   * - ``basis_set``
     - string
     - ``"def2-svp"``
     - Basis set for the calculation
   * - ``convergence_threshold``
     - float
     - ``1e-7``
     - Convergence tolerance for orbital gradient norm
   * - ``max_iterations``
     - int
     - ``50``
     - Maximum number of SCF iterations
   * - ``scf_type``
     - string
     - ``"auto"``
     - Type of SCF calculation:

       * ``"auto"``: Automatically detect based on spin
       * ``"restricted"``: Force restricted calculation
       * ``"unrestricted"``: Force unrestricted calculation

**Example:**

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
- :doc:`../basis_functionals`: Exchange-correlation functionals for DFT calculations
