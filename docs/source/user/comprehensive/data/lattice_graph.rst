LatticeGraph
============

The :class:`~qdk_chemistry.data.LatticeGraph` class in QDK/Chemistry represents a weighted graph defining the connectivity and geometry of a lattice of sites.
It provides static methods to generate common lattice geometries.
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

A :class:`~qdk_chemistry.data.LatticeGraph` stores a (possibly weighted) adjacency matrix for a lattice of sites.
It is the primary input to the :doc:`model Hamiltonian <../model_hamiltonians>` builders, where it defines which sites are connected and with what hopping strength.
Each qubit or orbital in the resulting Hamiltonian corresponds to a site in the lattice.

Properties
~~~~~~~~~~

Number of sites
   Total number of vertices in the lattice.

Number of edges
   Number of unique undirected edges (counted once per pair).

Adjacency matrix
   Sparse or dense matrix of edge weights.

Symmetry
   Whether the adjacency matrix is symmetric (required for physical Hamiltonians).

Usage
-----

The :class:`~qdk_chemistry.data.LatticeGraph` is typically the starting point for any model Hamiltonian workflow in QDK/Chemistry.
It defines the lattice topology before model parameters (hopping, on-site energies, interactions) are applied.

.. note::
   All built-in lattice factory methods produce symmetric (bidirectional) graphs by default.
   For custom lattices constructed from edge dictionaries, use ``make_bidirectional()`` if needed.

Creating lattice graphs
-----------------------

QDK/Chemistry provides static methods to create lattice graphs for common geometries.
For a brief overview of the available geometries, see the following table.
For detailed information about each geometry and how to create them, see the following sections.

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Lattice type
     - Dimensions
     - Total sites
     - Description
   * - ``chain(n)``
     - 1D
     - n
     - Linear chain with nearest-neighbour edges
   * - ``square(nx, ny)``
     - 2D
     - nx × ny
     - Square lattice with 4 neighbours per bulk site
   * - ``triangular(nx, ny)``
     - 2D
     - nx × ny
     - Triangular lattice with 6 neighbours per bulk site
   * - ``honeycomb(nx, ny)``
     - 2D
     - 2 × nx × ny
     - Honeycomb with 3 neighbours per site (2 sites/unit cell)
   * - ``kagome(nx, ny)``
     - 2D
     - 3 × nx × ny
     - Kagome with corner-sharing triangles (3 sites/unit cell)

One-dimensional lattices
~~~~~~~~~~~~~~~~~~~~~~~~

Chain lattice
^^^^^^^^^^^^^

The simplest lattice geometry is a 1D chain of sites connected by nearest-neighbour edges.
Setting ``periodic=True`` adds an edge between the first and last site to form a ring.

.. code-block:: text

   Chain (n=6):  0 --- 1 --- 2 --- 3 --- 4 --- 5

   Ring (n=6):   0 --- 1 --- 2 --- 3 --- 4 --- 5
                 |_____________________________|

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-create-chain
      :end-before: // end-cell-create-chain

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-create-chain
      :end-before: # end-cell-create-chain

Two-dimensional lattices
~~~~~~~~~~~~~~~~~~~~~~~~

QDK/Chemistry provides static methods for the most commonly studied 2D lattice geometries.
Sites are indexed in row-major order.

Each 2D geometry supports independent **periodic boundary conditions** along the x and y axes
(``periodic_x`` and ``periodic_y``).
When enabled, opposite edges of the lattice are connected, giving the lattice the topology of a
cylinder (one axis periodic) or a torus (both axes periodic).
See :ref:`lattice-periodic-boundary-conditions` for more information.

Square lattice
^^^^^^^^^^^^^^

The square lattice is the simplest 2D geometry, with four nearest neighbours per bulk site.
With periodic boundary conditions, the horizontal and vertical edges wrap, so every site has exactly four neighbours.

.. code-block:: text

   4x3 square lattice:

     8 --- 9 ---10 ---11
     |     |     |     |
     4 --- 5 --- 6 --- 7
     |     |     |     |
     0 --- 1 --- 2 --- 3

Triangular lattice
^^^^^^^^^^^^^^^^^^

The triangular lattice adds a diagonal bond to each square plaquette, giving six nearest neighbours per bulk site.
With periodic boundary conditions, all three bond directions (horizontal, vertical, and diagonal) wrap, so every site has exactly six neighbours.

.. code-block:: text

   3x3 triangular lattice:

     6 --- 7 --- 8
     |  /  |  /  |
     3 --- 4 --- 5
     |  /  |  /  |
     0 --- 1 --- 2

Honeycomb lattice
^^^^^^^^^^^^^^^^^

The honeycomb lattice has two sites per unit cell (A and B sublattices), giving three nearest neighbours per site.
Total sites: ``2 * nx * ny``.
With periodic boundary conditions, the inter-cell bonds between the B and A sublattices wrap around the edges, so every site retains exactly three neighbours.

.. code-block:: text

   3x4 honeycomb lattice:

              18-19-20-21-22-23
               |     |     |
           12-13-14-15-16-17
            |     |     |
         6--7--8--9-10-11
         |     |     |
      0--1--2--3--4--5

Kagome lattice
^^^^^^^^^^^^^^

The kagome lattice has three sites per unit cell, arranged as corner-sharing triangles.
Total sites: ``3 * nx * ny``.
With periodic boundary conditions, the inter-cell bonds that form the down-triangles wrap around the edges, maintaining the corner-sharing pattern across the boundary.

.. code-block:: text

   3x2 kagome:

        11       14       17
       /  \     /  \     /  \
      9---10--12---13--15---16
     /     \  /     \  /
    2       5        8
   / \     / \      / \
  0---1---3---4----6---7

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-create-2d
      :end-before: // end-cell-create-2d

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-create-2d
      :end-before: # end-cell-create-2d

Creating from adjacency data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For geometries not covered by the built-in methods, you can construct a :class:`~qdk_chemistry.data.LatticeGraph` from a dense adjacency matrix, a sparse adjacency matrix, or an edge-weight dictionary.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-from-matrix
      :end-before: // end-cell-from-matrix

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-from-matrix
      :end-before: # end-cell-from-matrix

.. _lattice-periodic-boundary-conditions:

Periodic boundary conditions
-----------------------------

All built-in lattice factory methods support periodic boundary conditions.
For 1D chains, ``periodic=True`` adds an edge between the first and last site to form a ring.
For 2D lattices, boundary conditions along each axis are controlled independently:

- ``periodic_x=True`` connects the rightmost column back to the leftmost column, adding an edge for every row between the site at ``x = nx-1`` and the site at ``x = 0``.
- ``periodic_y=True`` connects the top row back to the bottom row, adding an edge for every column between the site at ``y = ny-1`` and the site at ``y = 0``.
- When both are enabled, the lattice has the topology of a **torus** — there are no boundary sites, so every site has the same coordination number as a bulk site.

Periodic boundary conditions are commonly used to reduce finite-size effects in condensed matter simulations.
Without them, sites on the edges and corners of the lattice have fewer neighbours than interior sites, which introduces artifacts.
By wrapping the lattice, all sites become equivalent, better approximating the thermodynamic (infinite-lattice) limit.

The following diagram illustrates this for a 4×3 square lattice with both ``periodic_x`` and ``periodic_y`` enabled.
The ``~~~`` edges show the wrap-around connections that turn the open lattice into a torus:

.. code-block:: text

   4x3 square with periodic_x and periodic_y:

     8 --- 9 ---10 ---11 ~~~ 8
     |     |     |     |     |
     4 --- 5 --- 6 --- 7 ~~~ 4
     |     |     |     |     |
     0 --- 1 --- 2 --- 3 ~~~ 0
     ~     ~     ~     ~
     8     9    10    11

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-periodic
      :end-before: // end-cell-periodic

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-periodic
      :end-before: # end-cell-periodic

Accessing lattice data
----------------------

The :class:`~qdk_chemistry.data.LatticeGraph` class provides methods to query connectivity, edge weights, and structural properties.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-properties
      :end-before: // end-cell-properties

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-properties
      :end-before: # end-cell-properties

Serialization
-------------

The :class:`~qdk_chemistry.data.LatticeGraph` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <serialization>` documentation.

.. note::
   Lattice graph files use the ``.lattice_graph`` suffix before the file type extension, for example ``chain.lattice_graph.json`` and ``square.lattice_graph.hdf5``.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-serialization
      :end-before: // end-cell-serialization

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization

Edge coloring
-------------

Calling ``lattice.edge_coloring()`` returns a ``dict[tuple[int, int], int]`` that assigns a color index to each undirected edge such that edges sharing a vertex receive distinct colors.
The number of distinct colors (the chromatic index) is available via ``lattice.chromatic_index``.

This coloring is the topological ingredient that powers geometry-aware Trotter scheduling: edges of the same color have disjoint qubit supports, so their Pauli exponentials can be applied in parallel inside one Trotter step.
The :doc:`spin model Hamiltonian builders <../model_hamiltonians>` consume the coloring automatically when ``include_term_groups=True`` and store the result on :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition`.

For recognised lattice geometries (chain, square, honeycomb) the coloring is deterministic and optimal.
Custom lattices use a randomised greedy heuristic; pass ``trials > 1`` for tighter results.

Related classes
---------------

- :doc:`Model Hamiltonians <../model_hamiltonians>`: Using lattice graphs to build model Hamiltonians
- :doc:`Hamiltonian <hamiltonian>`: The Hamiltonian class produced by fermionic model Hamiltonian builders

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/lattice_graph.cpp>`_ and `Python <../../../_static/examples/python/lattice_graph.py>`_ scripts.
- :doc:`Serialization <serialization>`: Data serialization and deserialization in QDK/Chemistry
