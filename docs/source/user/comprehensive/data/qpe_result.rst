QpeResult
=========

The :class:`~qdk_chemistry.data.QpeResult` class in QDK/Chemistry represents the outcome of a quantum phase estimation calculation.
It encapsulates the measured phase, reconstructed energy, alias candidates, and measurement metadata, providing a complete record of a :term:`QPE` experiment.

Overview
--------

Quantum phase estimation measures a phase fraction :math:`\phi \in [0, 1)` that encodes an eigenvalue :math:`E` of the target Hamiltonian.
The relationship between phase and energy depends on the evolution time :math:`t`:

.. math::

   E = \frac{2\pi\phi}{t}

However, because the phase is periodic, the measured :math:`\phi` is only unique modulo :math:`2\pi / t`.
This means multiple energy values — called *aliases* — can produce the same measured phase.
The :class:`~qdk_chemistry.data.QpeResult` class handles this ambiguity automatically by computing all alias candidates and optionally resolving to the physically correct energy using a reference value.

:class:`~qdk_chemistry.data.QpeResult` is the output of the :doc:`PhaseEstimation <../algorithms/phase_estimation>` algorithm and supports full :doc:`serialization <serialization>` to JSON and HDF5 formats.
For details on how different :term:`QPE` implementations (:ref:`IQPE <iqpe-algorithm>`, :ref:`standard QFT-based <standard-qpe-algorithm>`) populate this result, see the :doc:`PhaseEstimation algorithm documentation <../algorithms/phase_estimation>`.


Properties
----------

The :class:`~qdk_chemistry.data.QpeResult` stores the following information:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Type
     - Description
   * - ``method``
     - str
     - Algorithm identifier (e.g., ``"iterative"`` for :ref:`IQPE <iqpe-algorithm>` or ``"qiskit_standard"`` for :ref:`standard QPE <standard-qpe-algorithm>`).
   * - ``evolution_time``
     - float
     - Time parameter :math:`t` used in the evolution :math:`U = e^{-iHt}`.
   * - ``phase_fraction``
     - float
     - Raw measured phase :math:`\phi \in [0, 1)`.
   * - ``phase_angle``
     - float
     - Raw phase angle in radians: :math:`2\pi\phi`.
   * - ``canonical_phase_fraction``
     - float
     - Alias-resolved phase fraction. Equals ``phase_fraction`` when no alias resolution is performed.
   * - ``canonical_phase_angle``
     - float
     - Alias-resolved phase angle in radians.
   * - ``raw_energy``
     - float
     - Energy computed directly from the raw phase: :math:`E = 2\pi\phi / t`.
   * - ``branching``
     - tuple[float, ...]
     - All alias energy candidates :math:`E + k \cdot 2\pi/t` for a range of integer shifts :math:`k`.
   * - ``resolved_energy``
     - float | None
     - The alias candidate closest to the reference energy, or ``None`` if no reference was provided.
   * - ``bits_msb_first``
     - tuple[int, ...] | None
     - Measured phase bits ordered from most significant to least significant. Available for :ref:`IQPE <iqpe-algorithm>`; may be ``None`` for other methods.
   * - ``bitstring_msb_first``
     - str | None
     - Binary string representation of the measured phase (e.g., ``"0110110010"``).
   * - ``metadata``
     - dict | None
     - Caller-defined metadata for provenance tracking (e.g., molecule name, basis set, reference energy).


.. _qpe-alias-resolution:

Alias resolution
----------------

Phase estimation measures a phase :math:`\phi \in [0, 1)`, but the underlying energy eigenvalue can be negative, positive, or arbitrarily large.
Different energy values that differ by integer multiples of :math:`2\pi / t` all map to the same phase.

The alias resolution algorithm works as follows:

1. Compute the raw energy: :math:`E_{\text{raw}} = 2\pi\phi / t` where the angle is mapped to :math:`(-\pi, \pi]`
2. Enumerate alias candidates: :math:`E_k = E_{\text{raw}} + k \cdot 2\pi/t` for each :math:`k` in the branch shift range (default: :math:`k \in \{-2, -1, 0, 1, 2\}`)
3. Include negative reflections: :math:`-E_k` for symmetry
4. If a reference energy is provided, select the candidate closest to the reference as ``resolved_energy``

The branch shift range can be customized via the ``branch_shifts`` parameter in :meth:`~qdk_chemistry.data.QpeResult.from_phase_fraction`.

.. tip::

   A good choice for ``reference_energy`` is the energy from a classical multi-configuration calculation (e.g., :term:`CASCI`), which is typically close to the true eigenvalue.


Construction
------------

:class:`~qdk_chemistry.data.QpeResult` objects are typically created by the :doc:`PhaseEstimation <../algorithms/phase_estimation>` algorithm.
They can also be constructed manually from a measured phase fraction using the :meth:`~qdk_chemistry.data.QpeResult.from_phase_fraction` class method:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-create-from-phase
      :end-before: # end-cell-create-from-phase


Inspecting results
~~~~~~~~~~~~~~~~~~

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-inspect
      :end-before: # end-cell-inspect


Working with aliases
~~~~~~~~~~~~~~~~~~~~

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-alias
      :end-before: # end-cell-alias


Serialization
-------------

:class:`~qdk_chemistry.data.QpeResult` supports the same :doc:`serialization <serialization>` formats as other QDK/Chemistry data classes:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization


Related classes
---------------

- :doc:`PhaseEstimation <../algorithms/phase_estimation>`: Algorithm that produces ``QpeResult`` objects
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Input qubit Hamiltonian
- :class:`~qdk_chemistry.data.Wavefunction`: Source wavefunction for state preparation

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/qpe_result.py>`_ script.
- :doc:`PhaseEstimation <../algorithms/phase_estimation>`: Phase estimation algorithms
- :doc:`UnitaryBuilder <../algorithms/unitary_builder>`: Hamiltonian simulation methods
- :doc:`Serialization <serialization>`: Data persistence formats
- See the ``examples/qpe_stretched_n2.ipynb`` notebook for an end-to-end :term:`QPE` workflow
