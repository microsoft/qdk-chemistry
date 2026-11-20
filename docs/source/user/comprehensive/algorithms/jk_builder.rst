J/K builders
============

The :class:`~qdk_chemistry.algorithms.JKBuilder` algorithm in QDK/Chemistry builds the mean-field Coulomb (:math:`\mathbf{J}`) and exchange (:math:`\mathbf{K}`) matrices of density matrices, and the :class:`~qdk_chemistry.algorithms.JKGradientBuilder` algorithm builds the nuclear gradients of the mean-field Coulomb and exchange energies.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, both take a :doc:`BasisSet <../data/basis_set>` instance, two-body kernels, and density matrices as input; the J/K builder produces :math:`\mathbf{J}` and :math:`\mathbf{K}` as :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2` instances, and the J/K gradient builder produces :class:`~qdk_chemistry.data.NuclearGradients` instances as output.
They separate the two-electron integral evaluation from the methods that use it, so that integral backends are interchangeable; for example, the ``qdk`` :doc:`SCF solver <scf_solver>` builds its :math:`\mathbf{J}` and :math:`\mathbf{K}` matrices, and their contributions to its analytic nuclear gradients, with them.

Overview
--------

For the density matrix :math:`\mathbf{P}^s` of each spin block :math:`s`, in the atomic orbitals :math:`\{\chi_\mu\}` of the basis set, a J/K builder computes

.. math::

   J_{\mu\nu}[\mathbf{P}^s] = \sum_{\lambda\sigma} (\mu\nu|g_J|\lambda\sigma) P^s_{\lambda\sigma},
   \qquad
   K_{\mu\nu}[\mathbf{P}^s] = \sum_{\lambda\sigma} (\mu\lambda|g_K|\nu\sigma) P^s_{\lambda\sigma},

with the two-body kernels :math:`g_J` and :math:`g_K` of the run.
A kernel is a :class:`~qdk_chemistry.data.TwoBodyKernel` instance:

- :class:`~qdk_chemistry.data.CoulombKernel`: :math:`g(r) = 1 / r`
- :class:`~qdk_chemistry.data.ErfCoulombKernel`: :math:`g(r) = \operatorname{erf}(\omega r) / r`
- :class:`~qdk_chemistry.data.ErfcCoulombKernel`: :math:`g(r) = \operatorname{erfc}(\omega r) / r`

with the range-separation parameter :math:`\omega` in inverse bohr.
A ``None`` (C++: null) kernel skips its matrix, which is then returned as ``None``.
:math:`\mathbf{J}` and :math:`\mathbf{K}` are returned in the spin structure of the density matrices, so the mean-field Coulomb potential of the electrons is :math:`\mathbf{J}[\mathbf{P}^\alpha] + \mathbf{J}[\mathbf{P}^\beta]`.
The density matrices need not be those of a state: response and other derived density matrices are contracted the same way.

A J/K gradient builder computes the derivatives with respect to the nuclear coordinates, at fixed density matrices, of the mean-field Coulomb and exchange energies

.. math::

   E_J = \tfrac{1}{2} \operatorname{Tr}(\mathbf{P} \mathbf{J}[\mathbf{P}]),
   \qquad
   E_K = -\tfrac{1}{2} \sum_{s = \alpha, \beta} \operatorname{Tr}(\mathbf{P}^s \mathbf{K}[\mathbf{P}^s]),

with :math:`\mathbf{P} = \mathbf{P}^\alpha + \mathbf{P}^\beta`.

Using the JKBuilder
-------------------

This section demonstrates how to use J/K builders in an :term:`SCF` calculation, on their own, and as custom implementations.

Input requirements
~~~~~~~~~~~~~~~~~~

The J/K builders require the following input:

BasisSet
   A :doc:`BasisSet <../data/basis_set>` instance whose atomic orbitals index the matrices.
   The atomic orbitals are those of the basis set, in their order, with each contracted shell normalized and its functions ordered as in PySCF: :math:`x, y, z` for p shells and real solid harmonics :math:`m = -l, \ldots, l` for higher shells.

Two-body kernels
   The kernels :math:`g_J` of :math:`\mathbf{J}` and :math:`g_K` of :math:`\mathbf{K}`, either of which may be ``None``.
   A builder raises an error for a kernel it does not support.

Density matrices
   Symmetric matrices in these atomic orbitals, the alpha-alpha and beta-beta blocks of a spin-diagonal :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2`, which :func:`~qdk_chemistry.data.symmetry.make_spin_diagonal_rank2_sbt` builds.
   On a restricted spin axis the two blocks share storage, and the builders compute :math:`\mathbf{J}` and :math:`\mathbf{K}` once.

.. rubric:: Using J/K builders in an SCF calculation

The ``jk_builder`` and ``jk_gradient_builder`` settings of the ``qdk`` :doc:`SCF solver <scf_solver>` hold :class:`~qdk_chemistry.data.AlgorithmRef` references to the builders and their settings, ``qdk`` by default.
The solver sets the ``eri_threshold`` of its builders from its own ``eri_threshold`` setting.
It builds :math:`\mathbf{J}` with the Coulomb kernel and :math:`\mathbf{K}` of the exchange operator :math:`\alpha / r_{12} + \beta \operatorname{erfc}(\omega r_{12}) / r_{12}` of its density functional from the fewest kernels that sum to it.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/jk_builder.py
      :language: python
      :start-after: # start-cell-scf
      :end-before: # end-cell-scf

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/jk_builder.cpp
      :language: cpp
      :start-after: // start-cell-scf
      :end-before: // end-cell-scf

.. rubric:: Building J and K

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/jk_builder.py
      :language: python
      :start-after: # start-cell-build
      :end-before: # end-cell-build

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/jk_builder.cpp
      :language: cpp
      :start-after: // start-cell-build
      :end-before: // end-cell-build

.. rubric:: Implementing a J/K builder

A J/K builder derives from :class:`~qdk_chemistry.algorithms.JKBuilder` and implements ``_run_impl``, which returns :math:`\mathbf{J}` and :math:`\mathbf{K}` for a basis set, two-body kernels, and density matrices; a J/K gradient builder derives from :class:`~qdk_chemistry.algorithms.JKGradientBuilder` and returns the gradients of :math:`E_J` and :math:`E_K`.
The :term:`SCF` solver uses one builder for all iterations of a calculation, so a builder can keep data derived from the basis set, such as integral screening data, between runs.
The example below times the builds of the ``qdk`` builder:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/jk_builder.py
      :language: python
      :start-after: # start-cell-custom
      :end-before: # end-cell-custom

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/jk_builder.cpp
      :language: cpp
      :start-after: // start-cell-custom
      :end-before: // end-cell-custom

   .. literalinclude:: ../../../_static/examples/cpp/jk_builder.cpp
      :language: cpp
      :start-after: // start-cell-register
      :end-before: // end-cell-register

Builders are registered like any other algorithm implementation; see :ref:`adding-plugins`, and :ref:`cpp-plugins` for C++ add-ons compiled into QDK/Chemistry.

Available settings
------------------

All J/K builders and J/K gradient builders share a common base set of settings:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``eri_threshold``
     - float
     - ``1e-12``
     - Integral screening threshold

See :doc:`Settings <settings>` for a more general treatment of settings in QDK/Chemistry.

Available implementations
-------------------------

QDK (Native)
~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk"`` (default)

The ``qdk`` J/K builder evaluates the integrals of the Coulomb, erf, and erfc kernels with Libint2, directly or in memory, and the ``qdk`` J/K gradient builder evaluates the integral derivatives directly.
Both keep the integral data of each kernel of the last basis set between runs.

.. rubric:: Settings

In addition to the common base settings, both builders have

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``shell_pair_threshold``
     - float
     - ``1e-12``
     - Overlap-based shell pair pre-screening threshold. Shell pairs with overlap norm below this value are excluded.

and the J/K builder also has

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``eri_method``
     - string
     - ``"direct"``
     - :term:`ERI` evaluation method: ``"direct"`` computes integrals on the fly, ``"incore"`` stores all integrals in memory
   * - ``eri_use_atomics``
     - bool
     - ``False``
     - Accumulate direct J/K builds with atomic operations instead of per-thread buffers

Related classes
---------------

- :doc:`BasisSet <../data/basis_set>`: Input basis set
- :class:`~qdk_chemistry.data.TwoBodyKernel`: Input kernels of :math:`\mathbf{J}` and :math:`\mathbf{K}`
- :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2`: Input density matrices and output :math:`\mathbf{J}` and :math:`\mathbf{K}`
- :class:`~qdk_chemistry.data.NuclearGradients`: Output gradients of the J/K gradient builder

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/jk_builder.py>`_ or `C++ <../../../_static/examples/cpp/jk_builder.cpp>`_ scripts.
- :doc:`ScfSolver <scf_solver>`: Uses J/K builders for its Fock matrices and analytic gradients
- :doc:`Plugins <../plugins>`: Registering custom implementations
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
