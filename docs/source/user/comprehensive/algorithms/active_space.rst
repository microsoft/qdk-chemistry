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

Capabilities
------------

The :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector` in QDK/Chemistry provides several selection strategies:

- **Valence-based selection**: Selects orbitals near the HOMO-LUMO gap based on user-specified numbers of electrons and orbitals

  - ``qdk_valence``: Manual specification of active electrons and orbitals

- **Occupation-based selection**: Identifies orbitals with fractional occupation numbers

  - ``qdk_occupation``: Selects orbitals with occupations deviating from 0 or 2

- **Entropy-based selection**: Uses orbital entropies to identify strongly correlated orbitals

  - ``qdk_autocas``: Automated selection based on single orbital entropies
  - ``qdk_autocas_eos``: Enhanced entropy-based selection with plateau detection

- **Orbital type support**:

  - Restricted orbitals (closed-shell and open-shell systems)
  - Support for both canonical and localized orbitals

Creating an active space selector
---------------------------------

Below is an example of how to create and run an active space selector using the default (Microsoft) QDK/Chemistry implementation:

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

Configuring the active space selection
--------------------------------------

The :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector` can be configured using the ``Settings`` object.
Different selectors have different configuration options depending on their selection strategy.

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

Running active space selection
------------------------------

Once configured, the active space selection can be executed on a wavefunction from a prior :term:`SCF` calculation.
The ``run`` method returns a :doc:`Wavefunction <../data/wavefunction>` object with active space information populated.

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

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector` accepts different settings depending on the selection method.

Valence selector settings (``qdk_valence``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``num_active_electrons``
     - int
     - -1
     - Number of electrons to include in the active space (required)
   * - ``num_active_orbitals``
     - int
     - -1
     - Number of orbitals to include in the active space (required)

Occupation selector settings (``qdk_occupation``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``occupation_threshold``
     - float
     - 0.1
     - Threshold for selecting orbitals with fractional occupation

Entropy selector settings (``qdk_autocas``, ``qdk_autocas_eos``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``min_plateau_size``
     - int
     - 2
     - Minimum size of entropy plateau for autocas selection
   * - ``entropy_threshold``
     - float
     - 0.1
     - Entropy threshold for eos-based selection

Implemented interfaces
----------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector` provides implementations for various selection strategies:

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **qdk_valence**: Manual valence-based selection
- **qdk_occupation**: Occupation-based automatic selection
- **qdk_autocas**: Entropy-based automatic selection
- **qdk_autocas_eos**: Enhanced entropy-based selection

The factory pattern allows seamless selection between these implementations based on the system requirements and desired level of automation.

Related classes
---------------

- :doc:`Wavefunction <../data/wavefunction>`: Input wavefunction from SCF calculation
- :doc:`Orbitals <../data/orbitals>`: Contains orbital information and active space indices
- :doc:`MCCalculator <mc_calculator>`: Uses active space for multireference calculations

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/active_space_selector.py>`_ script or `C++ <../../../_static/examples/cpp/active_space_selector.cpp>`_ source file.
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <../design/factory_pattern>`: Understanding algorithm creation
