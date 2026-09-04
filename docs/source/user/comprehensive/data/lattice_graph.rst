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

Geometry
   Optional Cartesian site positions and periodic supercell vectors used to identify geometric neighbor shells.

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

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-create-chain
      :end-before: # end-cell-create-chain

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-create-chain
      :end-before: // end-cell-create-chain

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

Use ``honeycomb_plaquettes(nx, ny)`` to size a patch by complete hexagonal
plaquettes instead of unit cells. Open directions include the boundary sites
needed to complete those plaquettes. A fully open patch contains
``2 * (nx + 1) * (ny + 1) - 2`` sites; ``honeycomb()`` and
``honeycomb_plaquettes()`` produce the same lattice when both axes are periodic.

.. code-block:: text

   1x1 open complete-plaquette lattice:

       1---2
      /     \
     0       5
      \     /
       3---4

The ``1 x 1`` patch has six perimeter bonds. Its geometric first-, second-, and
third-neighbor shells contain 6, 6, and 3 pairs, respectively, with the standard
honeycomb ``X``, ``Y``, and ``Z`` flavor definitions. A ``4 x 4`` open patch
contains 16 complete plaquettes and 48 sites.

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

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-create-2d
      :end-before: # end-cell-create-2d

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-create-2d
      :end-before: // end-cell-create-2d

Creating from adjacency data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For geometries not covered by the built-in methods, you can construct a :class:`~qdk_chemistry.data.LatticeGraph` from a dense adjacency matrix, a sparse adjacency matrix, or an edge-weight dictionary.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-from-matrix
      :end-before: # end-cell-from-matrix

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-from-matrix
      :end-before: // end-cell-from-matrix

Geometric neighbor shells
-------------------------

Built-in lattice factories store Cartesian site positions alongside the adjacency matrix, plus up to two supercell vectors for periodic axes.
The read-only ``positions`` property returns a copy of the factory-generated ``(num_sites, 2)`` Cartesian position matrix.
The ``mth_nearest_neighbors(m)`` method returns the canonical pairs ``(i, j)`` in the *m*-th distinct geometric distance shell, where ``m=1`` denotes nearest neighbors.
Distances are computed lazily when shells are requested, using the minimum image under the factory's periodic boundary conditions.
The ``nearest_neighbor_shells(shells)`` method classifies multiple requested shells in one pass.
For example, in the bulk or on a sufficiently large open lattice, the first three square-lattice shells have distances :math:`1`, :math:`\sqrt{2}`, and :math:`2`, while the honeycomb-lattice shells have distances :math:`1`, :math:`\sqrt{3}`, and :math:`2` in units of the nearest-neighbor spacing.

Graphs constructed only from adjacency data do not define a geometric embedding, so geometric neighbor shells cannot be queried on them.

Geometric shells contain site pairs only; physical coupling strengths belong to the :doc:`model Hamiltonian <../model_hamiltonians>`.

Geometric bond classes
~~~~~~~~~~~~~~~~~~~~~~

The ``neighbor_connections(shells)`` method retains more geometric information than the pair-only shell methods.
Each :class:`~qdk_chemistry.data.NeighborConnection` identifies a physical connection by its two finite-lattice sites, periodic image shift, Cartesian displacement, and :class:`~qdk_chemistry.data.BondClass`.
A bond class combines the radial shell with an unoriented displacement axis, so vectors :math:`\boldsymbol{d}` and :math:`-\boldsymbol{d}` share one orientation class.

This classification is available for every built-in lattice embedding.
The number of orientations depends on the geometry: a chain has one orientation per shell, while square, triangular, honeycomb, and kagome lattices generally have several.
Distinct periodic-image connections remain distinct even when they project onto the same canonical finite-lattice site pair.
By contrast, ``mth_nearest_neighbors()`` and ``nearest_neighbor_shells()`` intentionally deduplicate those connections and continue to return canonical pairs.

Semantic bond flavors are optional labels on geometric shell-axis classes.
Use :class:`~qdk_chemistry.data.BondFlavorDefinition` and ``with_bond_flavors()`` to label classes on another embedded lattice.
Unlabeled lattices retain all shell and orientation functionality, and their connection ``flavor`` properties are ``None``.

The honeycomb factory predefines :class:`~qdk_chemistry.data.BondFlavor` labels ``X``, ``Y``, and ``Z`` for the standard axes at distances :math:`1`, :math:`\sqrt{3}`, and :math:`2`.
Thus each honeycomb shell decomposes as

.. math::

   N_m = X_m \cup Y_m \cup Z_m,

when those distances are present as shells :math:`m=1,2,3`.
An interior site has one connection of each flavor in shells 1 and 3, and two connections of each flavor in shell 2.
These labels are physical model metadata and remain distinct from ``edge_coloring``, whose colors describe conflict-free scheduling layers.
On narrow finite patches, one of the standard distances can be absent. Because shell indices rank the distances present in that finite graph, a later distance that occupies the same rank remains unlabeled rather than being assigned a different physical honeycomb flavor.

Built-in lattice embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in factories use primitive lattice vectors to assign Cartesian positions to sites.
Those positions are used only to derive geometric neighbor shells: the adjacency matrix continues to define the graph topology.
For a site in cell :math:`(x,y)` with basis offset :math:`\boldsymbol{b}_s`, its position is

.. math::

   \boldsymbol{r}_{xys}=x\boldsymbol{a}_1+y\boldsymbol{a}_2+\boldsymbol{b}_s.

Lengths are measured in units of the nearest-neighbor spacing.
The chain uses :math:`\boldsymbol{a}_1=(1,0)`, with
:math:`\boldsymbol{a}_2=\boldsymbol{b}_0=(0,0)`.
The square lattice uses
:math:`\boldsymbol{a}_1=(1,0)`,
:math:`\boldsymbol{a}_2=(0,1)`, and
:math:`\boldsymbol{b}_0=(0,0)`.

The triangular factory uses
:math:`\boldsymbol{a}_1=(1,0)` and
:math:`\boldsymbol{a}_2=(-1/2,\sqrt{3}/2)`, with
:math:`\boldsymbol{b}_0=(0,0)`.
These are the directions :math:`\boldsymbol{u}_1` and
:math:`\boldsymbol{u}_2-\boldsymbol{u}_1` of Guo and Franz; this unimodular primitive-basis change makes
:math:`\boldsymbol{a}_1`, :math:`\boldsymbol{a}_2`, and
:math:`\boldsymbol{a}_1+\boldsymbol{a}_2` the three unit-length bond directions used by the factory. :footcite:p:`GuoFranz2009`

The honeycomb factory uses
:math:`\boldsymbol{a}_1=(3/2,\sqrt{3}/2)` and
:math:`\boldsymbol{a}_2=(3/2,-\sqrt{3}/2)`, exactly as in Eq. (1) of Castro Neto *et al.*,
with basis offsets :math:`\boldsymbol{b}_A=(0,0)` and :math:`\boldsymbol{b}_B=(1,0)`. :footcite:p:`CastroNeto2009`
The latter is equivalent to the paper's nearest-neighbor vectors up to bond orientation and interchange of the two sublattices.

The kagome factory uses
:math:`\boldsymbol{a}_1=(2,0)`,
:math:`\boldsymbol{a}_2=(1,\sqrt{3})`, and basis offsets
:math:`(0,0)`, :math:`(1,0)`, and :math:`(1/2,\sqrt{3}/2)`.
This is the triangular Bravais lattice with a three-point basis in Guo and Franz, whose nearest-neighbor directions are half of the two primitive vectors. :footcite:p:`GuoFranz2009`

For a periodic axis, the corresponding supercell period is
:math:`N_x\boldsymbol{a}_1` or :math:`N_y\boldsymbol{a}_2`.
The distance calculation minimizes over integer translations by those periods, so sites across a periodic boundary receive their minimum-image distance.
The vectors therefore cannot be removed without retaining equivalent geometric information, such as a Gram matrix and fractional basis coordinates.
Keeping the Cartesian vectors is the shorter representation for these two-dimensional factories.

.. footbibliography::

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

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-periodic
      :end-before: # end-cell-periodic

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-periodic
      :end-before: // end-cell-periodic

Accessing lattice data
----------------------

The :class:`~qdk_chemistry.data.LatticeGraph` class provides methods to query connectivity, edge weights, and structural properties.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-properties
      :end-before: # end-cell-properties

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-properties
      :end-before: // end-cell-properties

Serialization
-------------

The :class:`~qdk_chemistry.data.LatticeGraph` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <serialization>` documentation.

.. note::
   Lattice graph files use the ``.lattice_graph`` suffix before the file type extension, for example ``chain.lattice_graph.json`` and ``square.lattice_graph.hdf5``.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-serialization
      :end-before: // end-cell-serialization

Edge coloring
-------------

The ``edge_coloring`` property returns an optional ``dict[tuple[int, int], int]`` that assigns a color index to each undirected edge such that edges sharing a vertex receive distinct colors.
Factory methods for recognised topologies (chain, square, honeycomb) pre-populate this with a deterministic optimal coloring; triangular and kagome lattices use a greedy heuristic.
Custom lattices built from raw adjacency matrices have ``edge_coloring`` set to ``None`` — callers can compute and supply their own coloring.

This coloring is the topological ingredient that powers geometry-aware Trotter scheduling: edges of the same color have disjoint qubit supports, so their Pauli exponentials can be applied in parallel inside one Trotter step.
The :doc:`spin model Hamiltonian builders <../model_hamiltonians>` consume the coloring automatically when ``include_term_groups=True`` and store the result on :attr:`~qdk_chemistry.data.QubitOperator.term_partition`.

Related classes
---------------

- :doc:`Model Hamiltonians <../model_hamiltonians>`: Using lattice graphs to build model Hamiltonians
- :doc:`Hamiltonian <hamiltonian>`: The Hamiltonian class produced by fermionic model Hamiltonian builders

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/lattice_graph.cpp>`_ and `Python <../../../_static/examples/python/lattice_graph.py>`_ scripts.
- :doc:`Serialization <serialization>`: Data serialization and deserialization in QDK/Chemistry
