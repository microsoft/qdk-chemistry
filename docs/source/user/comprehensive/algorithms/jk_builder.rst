J/K builders
============

The :class:`~qdk_chemistry.algorithms.JKBuilder` algorithm in QDK/Chemistry builds the Coulomb (:math:`\mathbf{J}`) and exchange (:math:`\mathbf{K}`) matrices of spin densities, and the :class:`~qdk_chemistry.algorithms.JKGradientBuilder` algorithm builds the nuclear gradients of the Coulomb and exchange energies.
Both take a :doc:`BasisSet <../data/basis_set>` and spin densities as input.
The ``qdk`` :doc:`SCF solver <scf_solver>` builds its :math:`\mathbf{J}` and :math:`\mathbf{K}` matrices, and their contributions to its analytic nuclear gradients, with them.

Overview
--------

The spin densities :math:`\mathbf{P}^\alpha` and :math:`\mathbf{P}^\beta`, in the atomic orbitals :math:`\{\chi_\mu\}` of the basis set, are the alpha-alpha and beta-beta blocks of a spin-diagonal :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2`; on a restricted spin axis the two blocks share storage.
The atomic orbitals are those of the basis set, in their order, with each contracted shell normalized (Cartesian shells: the :math:`x^l` function).
For each spin density :math:`\mathbf{P}^s`, a J/K builder computes

.. math::

   J_{\mu\nu}[\mathbf{P}^s] = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) P^s_{\lambda\sigma},
   \qquad
   K_{\mu\nu}[\mathbf{P}^s] = \sum_{\lambda\sigma} (\mu\lambda|\hat v|\nu\sigma) P^s_{\lambda\sigma},

with the exchange operator :math:`\hat v = \alpha / r_{12} + \beta \operatorname{erfc}(\omega r_{12}) / r_{12}` of its ``alpha``, ``beta``, and ``omega`` settings (:math:`\alpha = 1`, :math:`\beta = 0` for Hartree-Fock).
:math:`\mathbf{J}` and :math:`\mathbf{K}` are returned in the spin structure of the densities, so the Coulomb potential of the electrons is :math:`\mathbf{J}[\mathbf{P}^\alpha] + \mathbf{J}[\mathbf{P}^\beta]`.

A J/K gradient builder computes the derivatives with respect to the nuclear coordinates, at fixed spin densities, of the Coulomb and exchange energies

.. math::

   E_J = \tfrac{1}{2} \operatorname{Tr}(\mathbf{P} \mathbf{J}[\mathbf{P}]),
   \qquad
   E_K = -\tfrac{1}{2} \sum_{s = \alpha, \beta} \operatorname{Tr}(\mathbf{P}^s \mathbf{K}[\mathbf{P}^s]),

with :math:`\mathbf{P} = \mathbf{P}^\alpha + \mathbf{P}^\beta`, as :class:`~qdk_chemistry.data.NuclearGradients` of the basis set's structure.

Using a J/K builder
-------------------

The ``jk_builder`` and ``jk_gradient_builder`` settings of the ``qdk`` :doc:`SCF solver <scf_solver>` hold :class:`~qdk_chemistry.data.AlgorithmRef` references to the builders and their settings, ``qdk`` by default.
The solver sets the ``eri_threshold`` of its builders from its own ``eri_threshold`` setting, and their exchange operator from its density functional:

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

A builder can also be used on its own:

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

Available implementations
-------------------------

QDK
~~~

.. rubric:: Factory name: ``"qdk"`` (default)

The ``qdk`` J/K builder evaluates the integrals with Libint2, directly or in memory, and the ``qdk`` J/K gradient builder evaluates the integral derivatives directly.
Both keep the integral data of the last basis set between runs.

.. rubric:: Settings

Both builders have the settings

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
   * - ``alpha``
     - float
     - ``1.0``
     - Coefficient of the full-range exchange operator
   * - ``beta``
     - float
     - ``0.0``
     - Coefficient of the short-range exchange operator
   * - ``omega``
     - float
     - ``0.0``
     - Range-separation parameter of the short-range exchange operator
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

Implementing a J/K builder
--------------------------

A J/K builder derives from :class:`~qdk_chemistry.algorithms.JKBuilder` and implements ``_run_impl``, which returns :math:`\mathbf{J}` and :math:`\mathbf{K}` for a basis set and spin densities; a J/K gradient builder derives from :class:`~qdk_chemistry.algorithms.JKGradientBuilder` and returns the gradients of :math:`E_J` and :math:`E_K`.
In Python, :func:`~qdk_chemistry.data.symmetry.make_spin_diagonal_rank2_sbt` builds the spin-diagonal tensors.
The SCF solver uses one builder for all iterations of a calculation, so a builder can keep data derived from the basis set, such as integral screening data, between runs.
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
