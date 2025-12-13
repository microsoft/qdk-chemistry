Active space selection
======================

The :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector` algorithm in QDK/Chemistry performs active space selection to identify the most chemically relevant orbitals for multireference calculations.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Wavefunction <../data/wavefunction>` instance as input and produces a :doc:`Wavefunction <../data/wavefunction>` instance with active space information as output.
Its primary purpose is to reduce the cost of quantum chemistry calculations by focusing on a specific set of relevant (active) orbitals while treating others as either fully occupied (core) or empty (virtual).

Overview
--------

Active space methods classify molecular orbitals into three categories:

1. **Inactive (core) orbitals**: Always doubly occupied and not explicitly correlated
2. **Active orbitals**: Allow variable occupation and are explicitly correlated
3. **Virtual orbitals**: Always empty and not explicitly correlated

The key challenge is selecting which orbitals to include in the active space.
An ideal active space should:

- Include all orbitals with significant multireference character
- Be as small as possible to keep computational cost manageable
- Capture the essential chemistry of the system

At its core, active space selection:

1. **Analyzes molecular orbitals** from a mean-field calculation (typically :term:`SCF`)
2. **Applies selection criteria** based on orbital properties (occupations, energies, entropies, etc.)
3. **Identifies chemically relevant orbitals** that show strong correlation effects
4. **Returns updated orbitals** with active space indices and metadata

The selected active space then serves as input for post-:term:`SCF` methods like :doc:`multi-configuration calculations <mc_calculator>` that explicitly treat electron correlation within the active space.

The ``run`` method returns a :doc:`Wavefunction <../data/wavefunction>` object with:

- **Active orbital indices**: Which orbitals are in the active space
- **Core orbital indices**: Orbitals treated as doubly occupied
- **Updated orbital metadata**: Information about the selection process

Running an active space selection
----------------------------------

This section demonstrates how to create, configure, and run an active space selection.
The ``run`` method takes a :doc:`Wavefunction <../data/wavefunction>` from a prior :term:`SCF` calculation and returns a new :doc:`Wavefunction <../data/wavefunction>` with active space information.

**Creating an active space selector:**

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/active_space_selector.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/active_space_selector.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

**Configuring settings:**

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/active_space_selector.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/active_space_selector.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

**Running the selection:**

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/active_space_selector.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/active_space_selector.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector` provides implementations for various selection strategies.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/active_space_selector.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/active_space_selector.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

QDK Valence
~~~~~~~~~~~

**Factory name:** ``"qdk_valence"`` (default)

Manual valence-based selection where users specify the number of active electrons and orbitals.
Selects orbitals near the HOMO-LUMO gap.

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``num_active_electrons``
     - int
     - ``-1``
     - Number of electrons in the active space (required)
   * - ``num_active_orbitals``
     - int
     - ``-1``
     - Number of orbitals in the active space (required)

QDK Occupation
~~~~~~~~~~~~~~

**Factory name:** ``"qdk_occupation"``

Automatic selection based on orbital occupation numbers, identifying orbitals with fractional occupation.

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``occupation_threshold``
     - float
     - ``0.1``
     - Orbitals with occupations deviating from 0 or 2 by more than this threshold are selected

QDK AutoCAS
~~~~~~~~~~~

**Factory name:** ``"qdk_autocas"``

Entropy-based automatic selection using single orbital entropies to identify strongly correlated orbitals.
Uses histogram-based plateau detection.

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``entropy_threshold``
     - float
     - ``0.14``
     - Entropy threshold for selection
   * - ``min_plateau_size``
     - int
     - ``10``
     - Minimum size of entropy plateau for selection
   * - ``num_bins``
     - int
     - ``100``
     - Number of histogram bins for plateau detection
   * - ``normalize_entropies``
     - bool
     - ``True``
     - Whether to normalize entropy values

QDK AutoCAS EOS
~~~~~~~~~~~~~~~

**Factory name:** ``"qdk_autocas_eos"``

Enhanced entropy-based selection using entanglement of orbitals with entropy differences.

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``entropy_threshold``
     - float
     - ``0.14``
     - Entropy threshold for selection
   * - ``diff_threshold``
     - float
     - ``0.1``
     - Difference threshold for EOS-based selection
   * - ``normalize_entropies``
     - bool
     - ``True``
     - Whether to normalize entropy values

PySCF AVAS
~~~~~~~~~~

**Factory name:** ``"pyscf_avas"``

The PySCF plugin provides access to the Automated Valence Active Space (AVAS) method from `PySCF <https://pyscf.org/>`_.
AVAS selects active orbitals by projecting molecular orbitals onto a target atomic orbital basis.
See the `original AVAS publication <https://doi.org/10.1021/acs.jctc.7b00128>`_ for details.

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``ao_labels``
     - list[str]
     - ``[]``
     - Atomic orbital labels to include (e.g., ``["Fe 3d", "Fe 4d"]``); required
   * - ``canonicalize``
     - bool
     - ``False``
     - Whether to canonicalize active orbitals after selection
   * - ``openshell_option``
     - int
     - ``2``
     - Handling of singly-occupied orbitals: ``2`` = project as alpha, ``3`` = keep in active space

**Example:**

.. literalinclude:: ../../../_static/examples/python/active_space_selector.py
   :language: python
   :start-after: # start-cell-avas-example
   :end-before: # end-cell-avas-example

For more details on how to extend QDK/Chemistry with additional implementations, see the :doc:`plugin system <../plugins>` documentation.

Related classes
---------------

- :doc:`Wavefunction <../data/wavefunction>`: Input wavefunction from SCF calculation
- :doc:`Orbitals <../data/orbitals>`: Contains orbital information and active space indices

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/active_space_selector.py>`_ script or `C++ <../../../_static/examples/cpp/active_space_selector.cpp>`_ source file.
- :doc:`MCCalculator <mc_calculator>`: Uses active space for multireference calculations
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
