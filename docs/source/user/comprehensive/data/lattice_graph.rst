LatticeGraph
============

The :class:`~qdk_chemistry.data.LatticeGraph` class represents weighted connectivity between indexed lattice sites.
Coordinates and geometric neighbor queries belong to the separate :doc:`LatticeGeometry <lattice_geometry>` class.
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

A :class:`~qdk_chemistry.data.LatticeGraph` can store resolved physical connections, including their shells, axes, periodic images, weights, and optional semantic flavors.
Its adjacency matrix is a projection of these records onto finite-lattice site pairs.
It also supports adjacency-only input without inventing geometric labels.
For example, :doc:`model Hamiltonian <../model_hamiltonians>` builders consume this connectivity together with interaction parameters.

Properties
~~~~~~~~~~

Number of sites
   Total number of vertices in the lattice.

Number of edges
   ``num_edges`` counts stored upper-triangular adjacency entries, once per distinct site pair, across all selected shells.
   It is not a count of physical images or of nearest-neighbor bonds alone; diagonal self-images are excluded.

Adjacency matrix
   Sparse or dense matrix of edge weights. For explicit connections, weights of distinct images joining the same pair are summed symmetrically; self-image weights contribute once to the diagonal.
   ``num_nonzeros`` counts stored sparse entries, which can include explicit zeros.

Symmetry
   Whether the adjacency matrix is symmetric (required for physical Hamiltonians).

Geometry
   ``geometry`` is an optional immutable :class:`~qdk_chemistry.data.LatticeGeometry`; it is ``None`` for adjacency-only input.

Selected shells
   ``selected_shells`` contains sorted, unique positive shell indices, including selected shells with no connections.

Connections
   ``connections`` contains :class:`~qdk_chemistry.data.NeighborConnection` records ordered by shell, orientation, endpoints, and image shift.

Usage
-----

Choose geometry and select the required connectivity before applying interaction parameters.
A graph may contain the union of several interactions' supports; each consumer uses the records relevant to its operation.

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

The Python call is ``LatticeGraph.from_geometry(geometry, shells, bond_flavors=[], weight=1.0, tolerance=1e-9)``.
``shells`` is sorted and deduplicated; an empty selection creates a graph with all sites but no connections.
Unavailable finite shells remain in ``selected_shells`` with no corresponding records.
``weight`` must be finite and is assigned to each physical connection, not to the sum over its periodic images.
``tolerance`` must be finite and positive and controls distance and axis comparisons.
Optional :class:`~qdk_chemistry.data.BondFlavorDefinition` objects assign semantic labels while constructing the graph.

Once constructed, the graph's selection is explicit.
Querying its geometry, assigning flavors, or passing shell mappings to a :doc:`model builder <../model_hamiltonians>` does not discover or add connections.
To use additional shells, construct a new graph from the same geometry with the required union.

For already-resolved data, :meth:`~qdk_chemistry.data.LatticeGraph.from_connections` accepts a site count, physical connection records, optional geometry, and optional selected shells.
The records are authoritative: supplied geometry is not queried to relabel them.
Endpoints and image directions are canonicalized, duplicate endpoint/image records are rejected, and shells present in the records are included in the selection.
Explicit records can therefore describe labeled connectivity without retaining geometry.

Nearest-neighbor graph factories
--------------------------------

The existing :class:`~qdk_chemistry.data.LatticeGraph` factories remain nearest-neighbor convenience constructors, preserving their adjacency weights, site ordering, boundary behavior, and stored topology colorings.
They retain a :class:`~qdk_chemistry.data.LatticeGeometry` and select shell 1, not every shell available in that geometry.
The available constructors are:

* :meth:`~qdk_chemistry.data.LatticeGraph.chain` — a chain or ring of ``n`` sites.
* :meth:`~qdk_chemistry.data.LatticeGraph.square` — ``nx * ny`` sites on a square grid.
* :meth:`~qdk_chemistry.data.LatticeGraph.triangular` — ``nx * ny`` sites on a triangular lattice.
* :meth:`~qdk_chemistry.data.LatticeGraph.honeycomb` — two sites per unit cell.
* :meth:`~qdk_chemistry.data.LatticeGraph.honeycomb_plaquettes` — a patch sized by complete hexagons.
* :meth:`~qdk_chemistry.data.LatticeGraph.kagome` — three sites per unit cell.

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
These constructors preserve directed or asymmetric input without assigning geometry, shells, or flavors.
If Cartesian coordinates are available and geometric selection is desired, construct a :class:`~qdk_chemistry.data.LatticeGeometry` instead and use :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry`.

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

Physical connections and bond flavors
-------------------------------------

Each stored :class:`~qdk_chemistry.data.NeighborConnection` retains its endpoints, :class:`~qdk_chemistry.data.BondClass`, displacement, and periodic image shift, as described under :ref:`geometric-bond-classes`.
Its ``weight`` is the physical connection's weight, and its ``flavor`` is an optional non-negative integer ID, not an enum object or a color.
Distinct images remain separate in ``connections`` even when adjacency combines them into one site pair.

Use :class:`~qdk_chemistry.data.BondFlavorDefinition` to associate a shell and unoriented axis with a semantic ID.
Pass definitions to :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry`, or use :meth:`~qdk_chemistry.data.LatticeGraph.with_bond_flavors` to replace labels on existing records.
Axes are normalized before comparison; opposite directions describe the same class.
Unmatched connections become unlabeled (``flavor is None``).
Definitions neither select shells nor create connections, and relabeling preserves the graph's weights and geometry.

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
These semantic labels are independent of shell-local orientation indices and scheduling colors.

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

For physical-image queries, use :meth:`~qdk_chemistry.data.LatticeGeometry.neighbor_connections` on the geometry, or inspect ``graph.connections`` for the selected weighted records.
Small periodic cells can have several physical connections for one site pair; distinct-neighbor counts need not equal bulk coordination numbers.
The nearest-neighbor graph factories preserve their existing adjacency weights by distributing each pair's weight over its physical images.
In contrast, :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` assigns ``weight`` to every image separately, so it need not reproduce a convenience factory's weights on a small periodic cell.
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
Explicit graphs persist resolved connection records, selected shells, optional geometry, and a checked adjacency cache.
Weights, flavors, and image multiplicity are retained; an explicitly empty selection is distinct from adjacency-only input.
Legacy adjacency files retain their topology and optional coordinates without inferring connection records.
To use their geometry in a shell- or flavor-dependent model, construct a selected graph with :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` after loading.
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

The ``edge_coloring`` property contains an optional ``dict[tuple[int, int], int]`` describing a stored topology coloring.
Nearest-neighbor convenience factories populate it; graphs built from geometry, explicit records, or raw adjacency have no stored coloring.

Use :meth:`~qdk_chemistry.data.LatticeGraph.color_edges` to color a supplied active pair support independently of adjacency weights or stored topology colors.
Pairs must be canonical (``i < j``) and in range; duplicate pairs are ignored.
The greedy search defaults to ``seed=0`` and ``trials=32`` and is not guaranteed to be optimal.
Edges sharing a color have disjoint vertices, enabling parallel Pauli exponentials in a :doc:`Trotter step <../algorithms/hamiltonian_unitary_builder>`.

One union graph does **not** imply one union coloring.
First accumulate contributions to each emitted Pauli interaction family and discard zero coefficients, then color that family's remaining distinct pairs.
Different families can have different supports and layer counts, even on the same graph:

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

The :ref:`shell-based spin model builders <model-term-partition>` perform this per-family coloring automatically when ``include_term_groups=True`` and store the result on :attr:`~qdk_chemistry.data.QubitOperator.term_partition`.

.. _lattice-geometry-migration:

Migrating geometry-aware code
-----------------------------

Existing nearest-neighbor graph factories and adjacency-based model calls retain their behavior.
Code that used a graph as a geometry container, or relied on a model to discover additional shells, needs the following changes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Previous access
     - Explicit geometry/graph access
   * - ``LatticeGraph.square(nx, ny)`` for coordinates or shell discovery
     - ``LatticeGeometry.square(nx, ny)``; select connectivity separately with ``LatticeGraph.from_geometry``.
   * - ``graph.positions``
     - ``graph.geometry.positions`` when ``graph.geometry is not None``.
   * - ``graph.mth_nearest_neighbors(m)``
     - ``graph.geometry.mth_nearest_neighbors(m)`` for an open geometry.
   * - ``graph.nearest_neighbor_shells(shells)``
     - ``graph.geometry.nearest_neighbor_shells(shells)`` for open-geometry queries, not graph selection.
   * - ``graph.neighbor_connections(shells)``
     - ``graph.geometry.neighbor_connections(shells)`` for geometric discovery; ``graph.connections`` for already-selected weighted/flavored records.
   * - ``graph.bond_flavor_definitions``
     - Supply definitions when constructing or relabeling a graph; inspect each stored connection's integer ``flavor`` instead of a retained definition recipe.

Select the union of nonzero requested interaction shells **before** constructing a :doc:`model Hamiltonian <../model_hamiltonians>`.
For example, a first- and second-neighbor Heisenberg model needs ``shells=[1, 2]``; a mapping ``{2: J2}`` on a nearest-neighbor-only graph no longer adds second-neighbor edges.
An active mapped shell absent from ``selected_shells`` raises an error, whereas a selected but geometrically unavailable shell contributes no terms.
Zero or empty mappings do not require selecting extra shells.

On an open graph, count distinct shell-1 pairs when reporting nearest-neighbor bonds; ``num_edges`` now describes the selected union:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-count-shell-edges
      :end-before: # end-cell-count-shell-edges

Do not use pair-only counts to identify physical edges under periodic boundaries; retain the image shift in each :class:`~qdk_chemistry.data.NeighborConnection`.

Related classes
---------------

- :doc:`LatticeGeometry <lattice_geometry>`: Coordinates, periodic vectors, and geometric neighbor queries
- :doc:`Model Hamiltonians <../model_hamiltonians>`: Using lattice graphs to build model Hamiltonians
- :doc:`Hamiltonian <hamiltonian>`: The Hamiltonian class produced by fermionic model Hamiltonian builders

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/lattice_graph.cpp>`_ and `Python <../../../_static/examples/python/lattice_graph.py>`_ scripts.
- :doc:`Serialization <serialization>`: Data serialization and deserialization in QDK/Chemistry
