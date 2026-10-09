LatticeGeometry
===============

The :class:`~qdk_chemistry.data.LatticeGeometry` class describes the site positions and optional periodic supercell vectors of a built-in two-dimensional lattice, independently of connectivity.
Like other :doc:`data classes <../design/index>`, it is immutable and supports :doc:`serialization <serialization>`.

Geometry does not store an adjacency matrix, interaction weights, semantic flavors, or edge colors.
Use :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` to turn selected geometric shells into an explicit :doc:`LatticeGraph <lattice_graph>`.
That constructor computes and stores one :ref:`edge coloring <lattice-edge-coloring>` for the selected distinct-site pairs, including zero-weight pairs; model builders filter it rather than recoloring individual interaction families.
The same geometry can be reused for different connectivity selections and consumers, such as :doc:`model Hamiltonians <../model_hamiltonians>` or visualization.
For connectivity that no factory describes, build a graph from adjacency data with :ref:`custom edge labels <lattice-custom-edge-labels>`.

Properties
----------

``num_sites``
   Number of indexed sites.

``positions``
   Cartesian ``(num_sites, 2)`` matrix in site-index order; a chain lies along the x axis.
   In Python, reading this property returns an independent copy.

``periods``
   Optional matrix of periodic supercell vectors, one row per periodic direction.
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

Neighbor shells
---------------

Shells are the distinct positive distances present on the given lattice, including distances to periodic images, ordered from shortest to longest; shell 1 is the shortest.
For a finite open patch, this is not a fixed bulk-lattice shell table.
For example, the first three shells on sufficiently large open square lattices have distances :math:`1`, :math:`\sqrt{2}`, and :math:`2`; honeycomb lattices have :math:`1`, :math:`\sqrt{3}`, and :math:`2`.
A thin patch has its own shells: ``square(1, 5)`` has the same shells as ``chain(5)``, and ``square(2, 8)`` has no :math:`2\sqrt{2}` distance, so its fifth shell is distance :math:`3`.

:meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` materializes the requested shells as labelled edges.
Its ``tolerance`` (default ``1e-9``) applies to relative distance and absolute axis comparisons and must be less than 1, the lattice unit length; unavailable finite shells contribute no edges.
Factories retain their integer unit-cell layout, so shell searches do not compare all site pairs.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/lattice_graph.py
      :language: python
      :start-after: # start-cell-geometry-shells
      :end-before: # end-cell-geometry-shells

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/lattice_graph.cpp
      :language: cpp
      :start-after: // start-cell-geometry-shells
      :end-before: // end-cell-geometry-shells

.. _geometry-periodic-images:

Periodic images
---------------

``chain`` accepts ``periodic=True``; the two-dimensional factories accept independent ``periodic_x`` and ``periodic_y`` flags along their primitive directions.
Two-dimensional periodic directions require a size greater than one.

If the rows of ``periods`` are :math:`\boldsymbol{P}_p`, the displacements from site :math:`i` to the images of site :math:`j` are

.. math::

   \boldsymbol{d}_{ij}=\boldsymbol{r}_j-\boldsymbol{r}_i+
   \sum_p n_p\boldsymbol{P}_p

for integers :math:`n_p`, and shell ranking counts each image separately.
:meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` adds the weights of images that join one pair with the same shell and flavor; for example, a periodic two-site chain joins its sites through two images, so the edge gets twice the weight. It rejects a selection in which images of different shells or flavors join one pair, or a site neighbors its own image.

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

Serialization
-------------

JSON and HDF5 :doc:`serialization <serialization>` store the integer unit-cell layout, from which positions and periods are rebuilt exactly.
The data type identifier is ``lattice_geometry``; a typical filename is ``patch.lattice_geometry.json``.

Related documentation
---------------------

* :doc:`LatticeGraph <lattice_graph>` — selecting connectivity, assigning flavors, and reusing stored edge colors
* :doc:`Model Hamiltonians <../model_hamiltonians>` — applying interactions to an explicit graph
