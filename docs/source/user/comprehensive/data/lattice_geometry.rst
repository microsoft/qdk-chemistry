LatticeGeometry
===============

The :class:`~qdk_chemistry.data.LatticeGeometry` class describes indexed site positions and optional periodic supercell vectors independently of connectivity.
It supports geometric neighbor queries for built-in and custom Cartesian embeddings.
Like other :doc:`data classes <../design/index>`, it is immutable and supports :doc:`serialization <serialization>`.

Geometry does not store an adjacency matrix, interaction weights, semantic flavors, or edge colors.
Use :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` to turn selected geometric shells into an explicit :doc:`LatticeGraph <lattice_graph>`.
That constructor computes and stores one :ref:`edge coloring <lattice-edge-coloring>` for the selected distinct-site pairs, including zero-weight pairs; model builders filter it rather than recoloring individual interaction families.
The same geometry can be reused for different connectivity selections and consumers, such as :doc:`model Hamiltonians <../model_hamiltonians>` or visualization.

Properties
----------

``num_sites``
   Number of indexed sites, including isolated or coincident positions.

``dimension``
   Number of Cartesian components per position.

``positions``
   Finite Cartesian ``(num_sites, dimension)`` matrix in site-index order, for any positive dimension.
   Built-in factories are two-dimensional, including for a chain.
   In Python, reading this property returns an independent copy.

``periods``
   Optional matrix with at most ``dimension`` finite, nonzero, independent row vectors of length ``dimension``, in image-shift order.
   ``None`` denotes an open geometry. Python returns a copy when periods are present.

Creating geometry
-----------------

Built-in factories use unit nearest-neighbor spacing and preserve a consistent indexing convention:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Factory
     - Site indexing and size
   * - :meth:`~qdk_chemistry.data.LatticeGeometry.chain`
     - ``chain(n)`` has ``n`` sites at ``(i, 0)``.
   * - :meth:`~qdk_chemistry.data.LatticeGeometry.square`
     - ``square(nx, ny)`` has ``nx * ny`` sites, indexed by ``y * nx + x``.
   * - :meth:`~qdk_chemistry.data.LatticeGeometry.triangular`
     - ``triangular(nx, ny)`` has ``nx * ny`` sites in primitive-cell order.
   * - :meth:`~qdk_chemistry.data.LatticeGeometry.honeycomb`
     - ``honeycomb(nx, ny)`` has two sites per cell, indexed by ``2 * (y * nx + x) + sublattice`` with A before B.
   * - :meth:`~qdk_chemistry.data.LatticeGeometry.honeycomb_plaquettes`
     - ``honeycomb_plaquettes(nx, ny)`` sizes a patch by complete hexagons, including the required open-boundary sites.
   * - :meth:`~qdk_chemistry.data.LatticeGeometry.kagome`
     - ``kagome(nx, ny)`` has three sites per cell, indexed by ``3 * (y * nx + x) + sublattice``.

A fully open ``honeycomb_plaquettes(1, 1)`` is a six-site hexagon; a ``4 x 4`` patch contains 48 sites.
Each open direction gains a boundary cell. Fully open patches omit the first A and last B corner sites and retain the remaining unit-cell order.
With both axes periodic, plaquette and unit-cell sizing coincide.

For custom embeddings, construct :class:`~qdk_chemistry.data.LatticeGeometry` directly from positions and optional periods.
An empty position matrix with shape ``(0, dimension)`` is valid.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-create-geometry
      :end-before: # end-cell-create-geometry

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-create-geometry
      :end-before: // end-cell-create-geometry

Geometric neighbor shells
-------------------------

Shells are the distinct positive distances actually present in the geometry, ordered from shortest to longest; shell 1 is the shortest.
For a finite open patch, this is not a fixed bulk-lattice shell table.
For example, the first three shells on sufficiently large open square lattices have distances :math:`1`, :math:`\sqrt{2}`, and :math:`2`; honeycomb lattices have :math:`1`, :math:`\sqrt{3}`, and :math:`2`.

The following queries accept a finite, positive ``tolerance`` (default ``1e-9``) for relative distance and absolute axis comparisons:

* :meth:`~qdk_chemistry.data.LatticeGeometry.mth_nearest_neighbors` returns sorted, unique pairs ``(i, j)`` with ``i < j`` for one open-geometry shell.
* :meth:`~qdk_chemistry.data.LatticeGeometry.nearest_neighbor_shells` classifies all requested shells together and returns a mapping from shell indices to pairs.
* :meth:`~qdk_chemistry.data.LatticeGeometry.neighbor_connections` returns physical connections classified by shell and axis, retaining periodic images.

Shell indices are positive integers; duplicate requests are ignored.
Neighbor queries currently require a two-dimensional geometry and raise ``RuntimeError`` otherwise.
Unavailable finite shells have no connections and map to empty lists in the pair-query result.
The pair-only methods reject periodic geometries because projecting to a pair would discard physical image multiplicity.
Queries do not modify any graph's selected connectivity, and geometric records have unit weight and no flavor.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-query-geometry
      :end-before: # end-cell-query-geometry

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-query-geometry
      :end-before: // end-cell-query-geometry

Built-in factories retain compact integer unit-cell coordinates for stencil-based shell queries.
Cartesian-only geometries use the general fallback; no lattice-specific neighbor cap is imposed by the query interface.

.. _geometric-bond-classes:

Geometric bond classes
----------------------

A :class:`~qdk_chemistry.data.BondClass` contains a one-based ``shell``, a shell-local ``orientation`` index, and a canonical unit ``axis``.
The axis is unoriented: displacements :math:`\boldsymbol{d}` and :math:`-\boldsymbol{d}` belong to the same class.
Orientation indices classify geometry; they do not encode a spin component, semantic flavor, or scheduling color.

Each :class:`~qdk_chemistry.data.NeighborConnection` contains ``site_i``, ``site_j``, ``bond_class``, ``displacement``, and ``image_shift``, together with ``flavor`` and ``weight`` fields.
Geometry queries return ``flavor=None`` and ``weight=1.0``.
Their records are ordered by shell, orientation, endpoints, and image shift, with ``site_i <= site_j``.
For open geometries the endpoints are distinct and ``image_shift`` is zero.
For periodic self-images, the first nonzero image component is positive.
Interaction-specific weights and labels are assigned by a :doc:`LatticeGraph <lattice_graph>`, not by the geometry.

.. _geometry-periodic-images:

Periodic images
---------------

``chain`` accepts ``periodic=True``; the two-dimensional factories accept independent ``periodic_x`` and ``periodic_y`` flags along their primitive directions.
Two-dimensional periodic directions require a size greater than one.
Periodic one- and two-site chain geometries are supported, including their distinct physical images.

If the rows of ``periods`` are :math:`\boldsymbol{P}_p`, a connection's displacement is

.. math::

   \boldsymbol{d}_{ij}=\boldsymbol{r}_j-\boldsymbol{r}_i+
   \sum_p n_p\boldsymbol{P}_p,

where ``image_shift`` stores the integer coefficients :math:`n_p` in row order, with one entry per spatial dimension, padded with zeros when fewer periods are present.
Distinct images are not merged even when their finite-lattice endpoints coincide.
A record may also connect a site to its own nonzero periodic image; individual consumers may reject such interactions.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-periodic-geometry
      :end-before: # end-cell-periodic-geometry

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-periodic-geometry
      :end-before: // end-cell-periodic-geometry

Built-in Cartesian embeddings
-----------------------------

For a site in cell :math:`(x,y)` with basis offset :math:`\boldsymbol{b}_s`, its position is

.. math::

   \boldsymbol{r}_{xys}=x\boldsymbol{a}_1+y\boldsymbol{a}_2+\boldsymbol{b}_s.

Lengths are measured in units of the nearest-neighbor spacing.
The chain uses :math:`\boldsymbol{a}_1=(1,0)`, with :math:`\boldsymbol{a}_2=\boldsymbol{b}_0=(0,0)`.
The square lattice uses :math:`\boldsymbol{a}_1=(1,0)`, :math:`\boldsymbol{a}_2=(0,1)`, and :math:`\boldsymbol{b}_0=(0,0)`.

The triangular factory uses :math:`\boldsymbol{a}_1=(1,0)` and :math:`\boldsymbol{a}_2=(-1/2,\sqrt{3}/2)`, with :math:`\boldsymbol{b}_0=(0,0)`.
These are the directions :math:`\boldsymbol{u}_1` and :math:`\boldsymbol{u}_2-\boldsymbol{u}_1` of Guo and Franz; this primitive-basis change makes :math:`\boldsymbol{a}_1`, :math:`\boldsymbol{a}_2`, and :math:`\boldsymbol{a}_1+\boldsymbol{a}_2` the three unit-length bond directions. :footcite:p:`GuoFranz2009`

The honeycomb factories use :math:`\boldsymbol{a}_1=(3/2,\sqrt{3}/2)` and :math:`\boldsymbol{a}_2=(3/2,-\sqrt{3}/2)`, as in Eq. (1) of Castro Neto *et al.*, with basis offsets :math:`\boldsymbol{b}_A=(0,0)` and :math:`\boldsymbol{b}_B=(1,0)`. :footcite:p:`CastroNeto2009`
The basis offset is equivalent to the paper's nearest-neighbor vectors up to bond orientation and interchange of the two sublattices.

The kagome factory uses :math:`\boldsymbol{a}_1=(2,0)`, :math:`\boldsymbol{a}_2=(1,\sqrt{3})`, and basis offsets :math:`(0,0)`, :math:`(1,0)`, and :math:`(1/2,\sqrt{3}/2)`.
This is the triangular Bravais lattice with a three-point basis in Guo and Franz, whose nearest-neighbor directions are half of the two primitive vectors. :footcite:p:`GuoFranz2009`

For a periodic direction, the corresponding supercell vector is :math:`N_x\boldsymbol{a}_1` or :math:`N_y\boldsymbol{a}_2`.
These vectors are part of the geometry and determine physical images across periodic boundaries.

.. footbibliography::

Relabeling and serialization
----------------------------

:meth:`~qdk_chemistry.data.LatticeGeometry.permute` returns a new geometry with new site ``i`` taken from original site ``path[i]``.
``path`` must be a permutation of every site. Positions and retained integer coordinates are reordered together; periodic vectors are unchanged.
Use :meth:`~qdk_chemistry.data.LatticeGraph.permute` when connectivity and geometry must be relabeled together.

JSON and HDF5 :doc:`serialization <serialization>` preserve only positions and optional periods.
Deserialized geometries use Cartesian neighbor queries; the factory's integer-coordinate cache is not serialized.
The data type identifier is ``lattice_geometry``; a typical filename is ``patch.lattice_geometry.json``.

Related documentation
---------------------

* :doc:`LatticeGraph <lattice_graph>` — selecting connectivity, assigning flavors, and reusing stored edge colors
* :doc:`Model Hamiltonians <../model_hamiltonians>` — applying interactions to an explicit graph
