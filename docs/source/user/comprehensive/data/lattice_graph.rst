LatticeGraph
============

The :class:`~qdk_chemistry.data.LatticeGraph` class represents weighted connectivity between indexed lattice sites.
Site coordinates of built-in lattices belong to the separate :doc:`LatticeGeometry <lattice_geometry>` class.
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

A :class:`~qdk_chemistry.data.LatticeGraph` stores weighted edges between finite-lattice site pairs.
Graphs built from a geometry also label each edge with its neighbor shell and an optional semantic flavor.
Graphs built from adjacency data are unlabelled unless the caller supplies the same :ref:`edge labels <lattice-custom-edge-labels>`.
For example, :doc:`model Hamiltonian <../model_hamiltonians>` builders consume this connectivity together with interaction parameters.

Properties
~~~~~~~~~~

Number of sites
   Total number of vertices in the lattice.

Number of edges
   ``num_edges`` counts stored upper-triangular adjacency entries, once per distinct site pair.
   It is not a count of physical images or of nearest-neighbor bonds alone.

Adjacency matrix
   Sparse or dense matrix of edge weights.
   ``num_nonzeros`` counts stored sparse entries, which can include explicit zeros.

Symmetry
   Whether the adjacency matrix is symmetric (required for physical Hamiltonians).

Edge labels
   ``edge_labels`` maps each canonical pair ``(i, j)`` with ``i < j`` to an :class:`~qdk_chemistry.data.EdgeLabel` holding its ``shell`` and optional ``flavor``; it is empty for unlabelled graphs.

Usage
-----

Choose geometry and select the required connectivity before applying interaction parameters.
A graph may contain the union of several interactions' supports; each consumer uses the edges relevant to its operation.

.. note::
   All built-in lattice factory methods produce symmetric (bidirectional) graphs by default.
   For one-directional edge dictionaries, use :meth:`~qdk_chemistry.data.LatticeGraph.make_bidirectional` if needed.
   This computes :math:`A+A^{\mathsf T}`; it doubles already-symmetric weights rather than merely filling missing reverse edges.

Selecting explicit connections
------------------------------

Use :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` to materialize the requested shells from a :class:`~qdk_chemistry.data.LatticeGeometry` in one query:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-from-geometry
      :end-before: # end-cell-from-geometry

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-from-geometry
      :end-before: // end-cell-from-geometry

The Python call is ``LatticeGraph.from_geometry(geometry, shells, bond_flavors=[], weight=1.0, tolerance=1e-9, coloring_seed=0)``.
``shells`` is deduplicated; an empty selection or unavailable finite shells create no edges.
Each physical connection becomes one edge whose weight is the finite ``weight``.
``tolerance`` must be positive and less than 1, the lattice unit length, and controls distance and axis comparisons.
Optional :class:`~qdk_chemistry.data.BondFlavorDefinition` objects assign semantic labels while constructing the graph.
A bond whose axis lies within ``tolerance`` of several flavor axes of its shell is rejected.
The graph does not retain the geometry.
Each edge is a single bond, so small periodic cells where several periodic images join one pair, or a site neighbors its own image, are rejected.

Once constructed, the graph's edges are explicit.
Passing shell mappings to a :doc:`model builder <../model_hamiltonians>` does not discover or add edges.
To use additional shells or flavors, construct a new graph from the same geometry.

Nearest-neighbor graph factories
--------------------------------

The existing :class:`~qdk_chemistry.data.LatticeGraph` factories remain nearest-neighbor convenience constructors, preserving their adjacency weights, site ordering, boundary behavior, and stored topology colorings.
Their edges are unlabelled, and model builders treat them as shell-1 edges; build models that use other shells or flavors from :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` or :ref:`custom edge labels <lattice-custom-edge-labels>`.
The available constructors are:

* :meth:`~qdk_chemistry.data.LatticeGraph.chain` — a chain or ring of ``n`` sites.
* :meth:`~qdk_chemistry.data.LatticeGraph.square` — ``nx * ny`` sites on a square grid.
* :meth:`~qdk_chemistry.data.LatticeGraph.triangular` — ``nx * ny`` sites on a triangular lattice.
* :meth:`~qdk_chemistry.data.LatticeGraph.honeycomb` — two sites per unit cell.
* :meth:`~qdk_chemistry.data.LatticeGraph.kagome` — three sites per unit cell.

For a honeycomb patch sized by complete hexagons, pass :meth:`~qdk_chemistry.data.LatticeGeometry.honeycomb_plaquettes` to :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry`.
See :doc:`LatticeGeometry <lattice_geometry>` for the coordinate and indexing conventions.

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
Sites follow the :doc:`geometry's unit-cell and sublattice order <lattice_geometry>`.

Each 2D geometry supports independent **periodic boundary conditions** along its primitive directions
(``periodic_x`` and ``periodic_y``).
When enabled, opposite edges of the lattice are connected, giving the lattice the topology of a
cylinder (one axis periodic) or a torus (both axes periodic).
See :ref:`lattice-periodic-boundary-conditions` for more information.

Square lattice
^^^^^^^^^^^^^^

The square lattice is the simplest 2D geometry, with four nearest neighbours per bulk site.
With periodic boundary conditions, the horizontal and vertical edges wrap.
Sufficiently large fully periodic lattices have four distinct nearest neighbours per site.

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
With periodic boundary conditions, all three bond directions wrap.
Sufficiently large fully periodic lattices have six distinct nearest neighbours per site.

.. code-block:: text

   3x3 triangular lattice:

     6 --- 7 --- 8
     |  /  |  /  |
     3 --- 4 --- 5
     |  /  |  /  |
     0 --- 1 --- 2

Honeycomb lattice
^^^^^^^^^^^^^^^^^

The honeycomb lattice has two sites per unit cell (A and B sublattices), giving three nearest neighbours per bulk site.
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

The ``1 x 1`` graph has six perimeter bonds. Its :doc:`geometry <lattice_geometry>` has first-, second-, and
third-neighbor shells containing 6, 6, and 3 pairs, respectively; selecting all three creates a 15-edge graph.
A ``4 x 4`` open patch contains 16 complete plaquettes and 48 sites.

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

For arbitrary connectivity, construct a :class:`~qdk_chemistry.data.LatticeGraph` from a dense adjacency matrix, a sparse adjacency matrix, or an edge-weight dictionary.
These constructors preserve directed or asymmetric input and assign shells or flavors only when :ref:`edge labels <lattice-custom-edge-labels>` are supplied.
For the built-in lattices, :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` derives the labels from a :class:`~qdk_chemistry.data.LatticeGeometry` instead.

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

Edge labels and bond flavors
----------------------------

Each :class:`~qdk_chemistry.data.EdgeLabel` records the edge's neighbor shell and its ``flavor``, an optional non-negative integer ID, not an enum object or a color.
The edge's weight is its adjacency entry.

Use :class:`~qdk_chemistry.data.BondFlavorDefinition` to associate a shell and unoriented axis with a semantic ID, and pass definitions to :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry`.
Axes are normalized before comparison; opposite directions describe the same class.
Unmatched edges become unlabeled (``flavor is None``).
Definitions neither select shells nor create edges.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-bond-flavors
      :end-before: # end-cell-bond-flavors

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-bond-flavors
      :end-before: // end-cell-bond-flavors

Flavor meanings belong to the consumer, not the geometry.
For example, the :ref:`Kitaev model builder <model-kitaev>` interprets IDs 0, 1, and 2 as X, Y, and Z spin interactions.
These semantic labels are independent of scheduling colors.

.. _lattice-custom-edge-labels:

Custom edge labels
~~~~~~~~~~~~~~~~~~

Graphs built from adjacency data can carry the same labels, which defines neighbor shells and flavors for connectivity that no built-in geometry describes.
Pass ``edge_labels``, a mapping from every stored pair, keyed by ``(i, j)`` with ``i < j`` whichever direction stores it, to an :class:`~qdk_chemistry.data.EdgeLabel`, to the edge-weight constructor, :meth:`~qdk_chemistry.data.LatticeGraph.from_dense_matrix`, or :meth:`~qdk_chemistry.data.LatticeGraph.from_sparse_matrix`.
The labels must cover exactly the stored pairs, each with a shell from 1 to :math:`2^{53}`; an empty mapping leaves the graph unlabelled.
Model builders then select shells and flavors exactly as for graphs built from a geometry.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-custom-labels
      :end-before: # end-cell-custom-labels

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-custom-labels
      :end-before: // end-cell-custom-labels

.. _lattice-periodic-boundary-conditions:

Periodic boundary conditions
-----------------------------

All built-in lattice factory methods support periodic boundary conditions.
For 1D chains, ``periodic=True`` adds an edge between the first and last site to form a ring.
For 2D lattices, boundary conditions along each primitive direction are controlled independently:

- ``periodic_x=True`` wraps the first primitive direction; on a square lattice, this joins the rightmost and leftmost columns.
- ``periodic_y=True`` wraps the second primitive direction; on a square lattice, this joins the top and bottom rows.
- With one periodic direction the patch forms a cylinder; with both, it forms a torus with no open boundaries.

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

Small periodic cells can have several physical connections for one site pair; distinct-neighbor counts need not equal bulk coordination numbers.
The nearest-neighbor graph factories preserve their existing adjacency weights, while :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` rejects such cells.
See :ref:`geometry-periodic-images` for the image convention.

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
Files store the sparse adjacency and any stored edge colors; labelled graphs also store one ``[i, j, shell, flavor]`` row per edge.
Adjacency-only files retain their topology without inferring edge labels.
To use such a lattice in a shell- or flavor-dependent model, construct a labelled graph with :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry`.
Files record their serialization ``version``; convert files written before lattice graphs were versioned with the :doc:`migration tool <../../migrating-data-files>`.
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

.. _lattice-edge-coloring:

Edge coloring
-------------

The ``edge_coloring`` property contains an optional ``dict[tuple[int, int], int]`` describing the graph's stored coloring; reading it returns an independent copy.
:meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` computes it once over all labelled pairs with ``i < j``.
Zero weights do not remove pairs from this coloring.
A geometry graph with no edges has an empty coloring, not ``None``.
Nearest-neighbor convenience factories retain their existing topology colorings; raw-adjacency constructors do not assign one.

:meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` uses the native greedy search with its ``coloring_seed`` (default ``0``) and 32 trials, like the other factories that color greedily; it is deterministic for a given seed but is not guaranteed to be optimal.
Edges sharing a color have disjoint vertices, enabling parallel Pauli exponentials in a :doc:`Trotter step <../algorithms/unitary_builder>`.
:meth:`~qdk_chemistry.data.LatticeGraph.permute` relabels their endpoints without recoloring.
Serialization retains the stored assignment.

Consumers restrict this single graph coloring to their active pair supports rather than recoloring each interaction family.
For spin models, contributions are accumulated before zero coefficients and empty color layers are discarded.
Different families can use different subsets of the stored colors, even on the same graph; restricting a coloring need not minimize a family's layer count:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-coloring
      :end-before: # end-cell-coloring

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-coloring
      :end-before: // end-cell-coloring

The :ref:`spin model builders <model-term-partition>` perform this filtering automatically when ``include_term_groups=True`` and a stored coloring is available, then store the resulting layers on :attr:`~qdk_chemistry.data.QubitOperator.term_partition`.

Related classes
---------------

- :doc:`LatticeGeometry <lattice_geometry>`: Site coordinates and periodic vectors of built-in lattices
- :doc:`Model Hamiltonians <../model_hamiltonians>`: Using lattice graphs to build model Hamiltonians
- :doc:`Hamiltonian <hamiltonian>`: The Hamiltonian class produced by fermionic model Hamiltonian builders

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/lattice_graph.cpp>`_ and `Python <../../../_static/examples/python/lattice_graph.py>`_ scripts.
- :doc:`Serialization <serialization>`: Data serialization and deserialization in QDK/Chemistry
