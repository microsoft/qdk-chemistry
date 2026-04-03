Model Hamiltonians
==================

QDK/Chemistry provides functionality to construct and manipulate model Hamiltonians used in quantum chemistry and condensed matter physics.
These model Hamiltonians serve as simplified representations of complex quantum systems to study their properties and behaviors using quantum computing techniques.

Unlike molecular Hamiltonians, model Hamiltonians do not require a molecular structure or precomputed integrals.
They are defined directly in terms of their parameters and a :doc:`LatticeGraph <data/lattice_graph>` that specifies the site connectivity.

Overview
--------

QDK/Chemistry supports two families of model Hamiltonians:

Fermionic models
   Operate on fermionic degrees of freedom (creation and annihilation operators) and produce :doc:`Hamiltonian <data/hamiltonian>` objects that are compatible with all QDK/Chemistry algorithms.

   * **Hückel** — tight-binding model with one-body hopping only
   * **Hubbard** — extends Hückel with on-site electron-electron repulsion
   * **Pariser-Parr-Pople (PPP)** — extends Hubbard with long-range intersite Coulomb interactions

Spin models
   Operate on spin-½ degrees of freedom and produce :class:`~qdk_chemistry.data.QubitHamiltonian` objects expressed as sums of Pauli operators.

   * **Heisenberg** — anisotropic spin-spin coupling with external magnetic fields
   * **Ising** — special case of Heisenberg with ZZ coupling and transverse X field

All model Hamiltonian builders take a :doc:`LatticeGraph <data/lattice_graph>` as their first argument, which defines the site connectivity and hopping structure.
For a brief description of the available model Hamiltonian builders, see the table below.
For a more detailed description of each model Hamiltonian and their parameters, see the following sections.

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Builder function
     - Type
     - Output
     - Description
   * - ``create_huckel_hamiltonian``
     - Fermionic
     - Hamiltonian
     - Tight-binding with hopping only
   * - ``create_hubbard_hamiltonian``
     - Fermionic
     - Hamiltonian
     - Hopping + on-site Coulomb repulsion
   * - ``create_ppp_hamiltonian``
     - Fermionic
     - Hamiltonian
     - Hubbard + long-range Coulomb interactions
   * - ``create_heisenberg_hamiltonian``
     - Spin
     - QubitHamiltonian
     - Anisotropic spin coupling + external fields
   * - ``create_ising_hamiltonian``
     - Spin
     - QubitHamiltonian
     - ZZ coupling + transverse X field

Fermionic models
----------------

.. _model-huckel:

Hückel model
~~~~~~~~~~~~

The Hückel (tight-binding) model describes non-interacting electrons hopping on a lattice:

.. math::

   \hat{H}_\text{Hückel} = \sum_i \varepsilon_i\, \hat{n}_i - \sum_{\langle i,j \rangle} t_{ij}\, w_{ij}\, (\hat{a}_i^\dagger \hat{a}_j + \hat{a}_j^\dagger \hat{a}_i)

where :math:`\hat{a}_i^\dagger` and :math:`\hat{a}_i` are the fermionic creation and annihilation operators for site *i*, :math:`\hat{n}_i = \sum_\sigma \hat{a}_{i,\sigma}^\dagger \hat{a}_{i,\sigma}` is the number operator, :math:`\varepsilon_i` are on-site energies, :math:`t_{ij}` are hopping integrals, :math:`w_{ij}` is the edge weight from the lattice adjacency matrix, and the sum runs over connected site pairs.
This model produces a :doc:`Hamiltonian <data/hamiltonian>` with one-body integrals only.

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-create-huckel
      :end-before: // end-cell-create-huckel

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-create-huckel
      :end-before: # end-cell-create-huckel

.. _model-hubbard:

Hubbard model
~~~~~~~~~~~~~

The Hubbard model extends the Hückel model with on-site Coulomb repulsion:

.. math::

   \hat{H}_\text{Hubbard} = \hat{H}_\text{Hückel} + \sum_i U_i\, \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}

where :math:`U_i` is the on-site repulsion strength.
This model produces a :doc:`Hamiltonian <data/hamiltonian>` with both one-body and two-body integrals.

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-create-hubbard
      :end-before: // end-cell-create-hubbard

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-create-hubbard
      :end-before: # end-cell-create-hubbard

The Hubbard model naturally extends to 2D lattices for studying strongly correlated materials:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-create-hubbard-2d
      :end-before: // end-cell-create-hubbard-2d

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-create-hubbard-2d
      :end-before: # end-cell-create-hubbard-2d

.. _model-ppp:

Pariser-Parr-Pople (PPP) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PPP model extends the Hubbard model with long-range intersite Coulomb interactions:

.. math::

   \hat{H}_\text{PPP} = \hat{H}_\text{Hubbard} + \frac{1}{2} \sum_{i \ne j} V_{ij}\, (\hat{n}_i - z_i)(\hat{n}_j - z_j)

where :math:`V_{ij}` is the intersite Coulomb repulsion and :math:`z_i` are effective core charges.
The intersite potential :math:`V_{ij}` is typically computed using the Ohno or Mataga-Nishimoto parametrizations (see `Intersite potentials`_ below).

.. note::

   The stored two-body integrals do **not** include the :math:`\frac{1}{2}` prefactor.
   This follows the standard quantum chemistry convention where the factor is applied at contraction time rather than stored in the integrals.

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-create-ppp
      :end-before: // end-cell-create-ppp

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-create-ppp
      :end-before: # end-cell-create-ppp

.. _intersite-potentials:

Intersite potentials
^^^^^^^^^^^^^^^^^^^^

For the :ref:`PPP model <model-ppp>`, the intersite Coulomb interaction :math:`V_{ij}` is typically computed from a distance-dependent parametrization.
QDK/Chemistry provides two standard potentials and a custom potential interface.

By default, all potential functions compute :math:`V_{ij}` for **every** pair of sites, not just lattice-connected neighbours.
This is consistent with the PPP Hamiltonian, which sums the Coulomb term over all pairs :math:`i \ne j`.
All three potential functions accept an optional ``nearest_neighbor_only`` flag (default ``false``) that restricts the evaluation to lattice-connected pairs only, setting :math:`V_{ij} = 0` for non-adjacent sites.

Ohno potential
""""""""""""""

.. math::

   V_{ij} = \frac{U_{ij}}{\sqrt{1 + \left(U_{ij}\,\varepsilon_r\,R_{ij}\right)^2}}

where :math:`U_{ij} = \sqrt{U_i U_j}` is the geometric mean of on-site parameters, :math:`R_{ij}` is the intersite distance, and :math:`\varepsilon_r` is the relative permittivity.

Mataga-Nishimoto potential
""""""""""""""""""""""""""

.. math::

   V_{ij} = \frac{U_{ij}}{1 + U_{ij}\,\varepsilon_r\,R_{ij}}

Custom pairwise potential
"""""""""""""""""""""""""

The ``pairwise_potential`` function accepts a user-defined callable ``func(i, j, U_ij, R_ij) -> V_ij`` for arbitrary distance-dependent potentials.

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-potentials
      :end-before: // end-cell-potentials

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-potentials
      :end-before: # end-cell-potentials

Spin models
-----------

.. _model-heisenberg:

Heisenberg model
~~~~~~~~~~~~~~~~

The anisotropic Heisenberg model describes spin-½ particles interacting on a lattice with external magnetic fields:

.. math::

   \hat{H}_\text{Heisenberg} = \sum_{\langle i,j \rangle} w_{ij}\,\bigl(
           J_x^{ij}\,\hat{\sigma}_i^x \hat{\sigma}_j^x
         + J_y^{ij}\,\hat{\sigma}_i^y \hat{\sigma}_j^y
         + J_z^{ij}\,\hat{\sigma}_i^z \hat{\sigma}_j^z
       \bigr)
     + \sum_i \bigl(
           h_x^{i}\,\hat{\sigma}_i^x
         + h_y^{i}\,\hat{\sigma}_i^y
         + h_z^{i}\,\hat{\sigma}_i^z
       \bigr)

where :math:`J_x, J_y, J_z` are the spin-spin coupling constants, :math:`h_x, h_y, h_z` are external magnetic field components, and :math:`w_{ij}` is the edge weight from the lattice adjacency matrix.
Each qubit corresponds to a lattice site.

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-create-heisenberg
      :end-before: # end-cell-create-heisenberg

Special cases of the Heisenberg model include:

- **Isotropic (XXX)**: :math:`J_x = J_y = J_z`
- **XXZ**: :math:`J_x = J_y \ne J_z`
- **XY**: :math:`J_z = 0`

.. _model-ising:

Ising model
~~~~~~~~~~~

The transverse-field Ising model is a special case of the Heisenberg model with ZZ coupling and a transverse X field:

.. math::

   \hat{H}_\text{Ising} = \sum_{\langle i,j \rangle} w_{ij}\,J^{ij}\,\hat{\sigma}_i^z \hat{\sigma}_j^z
     + \sum_i h^{i}\,\hat{\sigma}_i^x

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-create-ising
      :end-before: # end-cell-create-ising

Parameter flexibility
---------------------

All model Hamiltonian builders accept parameters as either scalars (applied uniformly to all sites or pairs) or arrays (specifying per-site or per-pair values).
This allows modelling inhomogeneous systems such as impurities or spatially varying fields.

Per-site parameters
   Scalar ``float`` (broadcast to all sites) or ``numpy.ndarray`` of length *n* (one value per site).
   Used for: on-site energy (:math:`\varepsilon`), on-site repulsion (:math:`U`), core charges (:math:`z`), magnetic fields (:math:`h_x, h_y, h_z`).

Per-pair parameters
   Scalar ``float`` (broadcast to all pairs) or ``(n, n)`` ``numpy.ndarray`` (one value per pair).
   Used for: hopping (:math:`t`), intersite potential (:math:`V`), spin couplings (:math:`J_x, J_y, J_z`).

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-site-dependent
      :end-before: // end-cell-site-dependent

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-site-dependent
      :end-before: # end-cell-site-dependent

Using model Hamiltonians with algorithms
-----------------------------------------

Fermionic model Hamiltonians produce :doc:`Hamiltonian <data/hamiltonian>` objects that are fully compatible with all QDK/Chemistry algorithms, including:

* :doc:`Multi-configuration calculators <algorithms/mc_calculator>` (:term:`FCI`, :term:`ASCI`, etc.)
* :doc:`Qubit mapping <algorithms/qubit_mapper>` (Jordan-Wigner, Bravyi-Kitaev, etc.)
* :doc:`Phase estimation <algorithms/phase_estimation>` (:term:`IQPE`, standard :term:`QPE`)

Spin model Hamiltonians produce :class:`~qdk_chemistry.data.QubitHamiltonian` objects directly, which can be used with quantum algorithms without an intermediate qubit mapping step.

.. rubric:: Example: exact diagonalization of the Hubbard model

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/model_hamiltonians.cpp
      :language: cpp
      :start-after: // start-cell-solve-hubbard
      :end-before: // end-cell-solve-hubbard

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-solve-hubbard
      :end-before: # end-cell-solve-hubbard

.. rubric:: Example: exact diagonalization of the Ising model

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/model_hamiltonians.py
      :language: python
      :start-after: # start-cell-solve-ising
      :end-before: # end-cell-solve-ising

Related classes
---------------

- :doc:`data/lattice_graph` — Lattice topology defining site connectivity
- :doc:`data/hamiltonian` — Hamiltonian data class produced by fermionic models
- :class:`~qdk_chemistry.data.QubitHamiltonian` — Qubit Hamiltonian produced by spin models
- :doc:`data/orbitals` — Full Orbitals class documentation

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../_static/examples/cpp/model_hamiltonians.cpp>`_ and `Python <../../_static/examples/python/model_hamiltonians.py>`_ scripts.
- :doc:`algorithms/mc_calculator` — Solving fermionic model Hamiltonians with exact diagonalization
- :class:`~qdk_chemistry.algorithms.QubitHamiltonianSolver` — Solving spin model Hamiltonians with exact diagonalization
