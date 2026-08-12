QpeResult
=========

The :class:`~qdk_chemistry.data.QpeResult` class in QDK/Chemistry represents the outcome of a quantum phase estimation calculation.
It encapsulates the measured phase, reconstructed energy, alias candidates, and measurement metadata, providing a complete record of a :term:`QPE` experiment.

Overview
--------

Quantum phase estimation measures a phase fraction :math:`\varphi \in [0, 1)` that encodes an eigenvalue :math:`E` of the target Hamiltonian.
The relationship between the measured phase and energy depends on the type of unitary being phase-estimated.

QDK/Chemistry uses a unified factory method :meth:`~qdk_chemistry.data.QpeResult.from_phase_fraction` that accepts a callable ``eigenvalue_from_phase``
mapping the measured phase to the Hamiltonian eigenvalue. Each unitary container provides this mapping as an instance method.

**Time evolution**

When QPE acts on :math:`U = e^{-iHt}`, the eigenvalues are :math:`e^{-iEt}` and the energy is recovered via
:meth:`~qdk_chemistry.data.unitary_representation.containers.pauli_product_formula.PauliProductFormulaContainer.eigenvalue_from_phase`:

.. math::

   E = -\frac{2\pi\varphi'}{t}

where :math:`\varphi' \in (-1/2, 1/2]` is the measured phase fraction wrapped from :math:`[0, 1)` into :math:`(-1/2, 1/2]`.

**Qubitization**

When QPE acts on the qubitization walk operator :math:`W`, the eigenvalues are :math:`e^{\pm i \arccos(E/\lambda)}` and the energy is recovered via
:meth:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.QuantumWalkContainer.eigenvalue_from_phase`:

.. math::

   E = \lambda \cos(2\pi\varphi)

where :math:`\lambda = \sum_j |\alpha_j|` is the 1-norm of the Hamiltonian coefficients.

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
   * - ``phase_fraction``
     - float
     - Raw measured phase :math:`\varphi \in [0, 1)`.
   * - ``phase_angle``
     - float
     - Raw phase angle in radians: :math:`2\pi\varphi`.
   * - ``canonical_phase_fraction``
     - float
     - Alias-resolved phase fraction. Equals ``phase_fraction`` when no alias resolution is performed (e.g., qubitization).
   * - ``canonical_phase_angle``
     - float
     - Alias-resolved phase angle in radians.
   * - ``raw_energy``
     - float
     - Energy computed from the measured phase via the container's ``eigenvalue_from_phase`` method.
   * - ``branching``
     - tuple[float, ...]
     - Energy candidates. For the unified ``from_phase_fraction`` factory, this is a single-element tuple containing ``raw_energy``.
   * - ``resolved_energy``
     - float | None
     - Reserved for alias resolution (currently ``None`` when using ``from_phase_fraction``).
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

Alias resolution is relevant for **time-evolution-based QPE** (Trotter) where the phase is periodic.
It is not needed for qubitization because the cosine mapping is injective over the measurable range.

Phase estimation measures a phase :math:`\varphi \in [0, 1)`, but the underlying energy eigenvalue can be negative, positive, or arbitrarily large.
Different energy values that differ by integer multiples of :math:`2\pi / t` all map to the same phase.

.. note::

   The current ``from_phase_fraction`` factory does not perform alias resolution automatically.
   Alias resolution is the responsibility of the calling algorithm when needed.


Construction
------------

:class:`~qdk_chemistry.data.QpeResult` objects are typically created by the :doc:`PhaseEstimation <../algorithms/phase_estimation>` algorithm.
They can also be constructed manually using :meth:`~qdk_chemistry.data.QpeResult.from_phase_fraction`, passing ``container.eigenvalue_from_phase`` as the phase-to-energy mapping.

Time evolution example
~~~~~~~~~~~~~~~~~~~~~~

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-create-from-time-evolution
      :end-before: # end-cell-create-from-time-evolution

Qubitization example
~~~~~~~~~~~~~~~~~~~~

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-create-from-qubitization
      :end-before: # end-cell-create-from-qubitization


Inspecting results
~~~~~~~~~~~~~~~~~~

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-inspect
      :end-before: # end-cell-inspect


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
- :class:`~qdk_chemistry.data.QubitOperator`: Input qubit Hamiltonian
- :class:`~qdk_chemistry.data.Wavefunction`: Source wavefunction for state preparation

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/qpe_result.py>`_ script.
- :doc:`PhaseEstimation <../algorithms/phase_estimation>`: Phase estimation algorithms
- :doc:`HamiltonianUnitaryBuilder <../algorithms/hamiltonian_unitary_builder>`: Hamiltonian simulation or block encoding methods
- :doc:`Serialization <serialization>`: Data persistence formats
- See the ``examples/qpe_stretched_n2.ipynb`` notebook for an end-to-end :term:`QPE` workflow
