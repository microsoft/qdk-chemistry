J/K builder
===========

The :class:`~qdk_chemistry.algorithms.JKBuilder` algorithm in QDK/Chemistry provides the Coulomb (:math:`\mathbf{J}`) and exchange (:math:`\mathbf{K}`) matrices of the ``qdk`` :doc:`SCF solver <scf_solver>`, and their contributions to its analytic nuclear gradients.
It takes a :doc:`BasisSet <../data/basis_set>` and an integral screening threshold as input and produces a :class:`~qdk_chemistry.algorithms.JKEngine`, which builds :math:`\mathbf{J}` and :math:`\mathbf{K}` of density matrices in that basis set.
The SCF solver creates one engine per calculation and uses it in every iteration, so implementations can keep data such as integral screening information between builds.

Overview
--------

For a density matrix :math:`\mathbf{P}` in the atomic orbitals :math:`\{\chi_\mu\}` of the basis set, the engine computes

.. math::

   J_{\mu\nu}[\mathbf{P}] = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) P_{\lambda\sigma},
   \qquad
   K_{\mu\nu}[\mathbf{P}] = \sum_{\lambda\sigma} (\mu\lambda|\hat v|\nu\sigma) P_{\lambda\sigma},

with the exchange operator :math:`\hat v = \alpha / r_{12} + \beta \operatorname{erfc}(\omega r_{12}) / r_{12}` of the density functional (:math:`\alpha = 1`, :math:`\beta = 0` for Hartree-Fock).
The atomic orbitals are those of the basis set, in their order, with each contracted shell normalized (Cartesian shells: the :math:`x^l` function).

A build receives one density matrix for restricted calculations and two (:math:`\alpha`, :math:`\beta`) for unrestricted and restricted open-shell calculations.
With incremental Fock builds, the density matrices are differences between successive iterations, so their elements have no fixed bound.

For analytic nuclear gradients, the engine's ``build_gradients`` computes the derivatives, at fixed density matrices, of the Coulomb and exchange energies

.. math::

   E_J = \tfrac{1}{2} \operatorname{Tr}(\mathbf{P} \mathbf{J}[\mathbf{P}]),
   \qquad
   E_K = -\tfrac{1}{4} \operatorname{Tr}(\mathbf{P} \mathbf{K}[\mathbf{P}])
   \quad \text{or} \quad
   E_K = -\tfrac{1}{2} \sum_{s = \alpha, \beta} \operatorname{Tr}(\mathbf{P}^s \mathbf{K}[\mathbf{P}^s]),

for one or two density matrices, with :math:`\mathbf{P}` the total density in :math:`E_J`.
Engines that do not implement ``build_gradients`` raise an error when gradients are requested.

Using a J/K builder
-------------------

The ``jk_builder`` setting of the ``qdk`` :doc:`SCF solver <scf_solver>` holds an :class:`~qdk_chemistry.data.AlgorithmRef` to the builder and its settings, ``qdk`` by default:

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

An engine can also be used on its own:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/jk_builder.py
      :language: python
      :start-after: # start-cell-engine
      :end-before: # end-cell-engine

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/jk_builder.cpp
      :language: cpp
      :start-after: // start-cell-engine
      :end-before: // end-cell-engine

Available implementations
-------------------------

QDK
~~~

.. rubric:: Factory name: ``"qdk"`` (default)

Evaluates the integrals with Libint2, directly or in memory.
Gradients are always evaluated directly.

.. rubric:: Settings

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
   * - ``shell_pair_threshold``
     - float
     - ``1e-12``
     - Overlap-based shell pair pre-screening threshold. Shell pairs with overlap norm below this value are excluded.

Implementing a J/K builder
--------------------------

A J/K builder derives from :class:`~qdk_chemistry.algorithms.JKBuilder` and returns a :class:`~qdk_chemistry.algorithms.JKEngine` from ``_run_impl``; the engine implements ``build`` and, to support analytic nuclear gradients, ``build_gradients``.
In C++, ``build`` receives the matrices as consecutive :math:`n \times n` arrays, and ``build_gradients`` writes row-major :math:`3 \times N_\text{atoms}` gradients.
Python implementations receive NumPy views, valid during the call: ``densities`` (read-only) of shape ``(num_densities, n, n)``, and ``coulomb`` and ``exchange`` (``None`` when not requested), filled in place, of shape ``(num_densities, n, n)`` in ``build`` and ``(3, num_atoms)`` in ``build_gradients``.
The example below times the builds of the ``qdk`` engine:

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
