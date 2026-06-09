QpeResult
=========

The :class:`~qdk_chemistry.data.QpeResult` class in QDK/Chemistry represents the outcome of a quantum phase estimation calculation.
It encapsulates the measured phase, reconstructed energy, alias candidates, and measurement metadata, providing a complete record of a :term:`QPE` experiment.

Overview
--------

Quantum phase estimation measures a phase fraction :math:`\varphi \in [0, 1)` that encodes an eigenvalue :math:`E` of the target Hamiltonian.
The relationship between the measured phase and energy depends on the type of unitary being phase-estimated.

QDK/Chemistry supports two phase-to-energy mappings:

**Time evolution**

When QPE acts on :math:`U = e^{-iHt}`, the eigenvalues are :math:`e^{-iEt}` and the energy is:

.. math::

   E = \frac{2\pi\varphi'}{t}

where :math:`\varphi' \in (-1/2, 1/2]` is the measured phase fraction mapped from :math:`[0, 1)` into :math:`(-1/2, 1/2]`.
Because the phase is periodic, multiple energy values — called *aliases* — differing by :math:`2\pi / t` all map to the same phase.
The alias resolution algorithm selects the correct branch using a classical reference energy.

**Qubitization**

When QPE acts on the qubitization walk operator :math:`W`, the eigenvalues are :math:`e^{\pm i \arccos(E/\lambda)}` and the energy is:

.. math::

   E = \lambda \cos(2\pi\varphi)

where :math:`\lambda = \sum_j |\alpha_j|` is the 1-norm of the Hamiltonian coefficients.
No alias resolution is needed because the cosine maps the full :math:`[0, 1)` range to a unique energy in :math:`[-\lambda, \lambda]`.

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
     - Energy computed directly from the raw phase. For time evolution: :math:`E = 2\pi\varphi' / t`. For qubitization: :math:`E = \lambda\cos(2\pi\varphi)`.
   * - ``branching``
     - tuple[float, ...]
     - All alias energy candidates. For time evolution: :math:`E + k \cdot 2\pi/t` for a range of integer shifts :math:`k`. For qubitization: a single-element tuple containing ``raw_energy``.
   * - ``resolved_energy``
     - float | None
     - The alias candidate closest to the reference energy, or ``None`` if no reference was provided or not applicable.
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

Alias resolution applies only to **time-evolution-based QPE** (Trotter).
It is not needed for qubitization because the cosine mapping is injective over the measurable range.

Phase estimation measures a phase :math:`\varphi \in [0, 1)`, but the underlying energy eigenvalue can be negative, positive, or arbitrarily large.
Different energy values that differ by integer multiples of :math:`2\pi / t` all map to the same phase.

The alias resolution algorithm works as follows:

1. Compute the raw energy: :math:`E_{\text{raw}} = 2\pi\varphi' / t` where :math:`\varphi'` is mapped to :math:`(-1/2, 1/2]`
2. Enumerate alias candidates: :math:`E_k = E_{\text{raw}} + k \cdot 2\pi/t` for each :math:`k` in the branch shift range (default: :math:`k \in \{-2, -1, 0, 1, 2\}`)
3. Include negative reflections: :math:`-E_k` for symmetry
4. If a reference energy is provided, select the candidate closest to the reference as ``resolved_energy``

The branch shift range can be customized via the ``branch_shifts`` parameter in :meth:`~qdk_chemistry.data.QpeResult.from_time_evolution_result`.

.. tip::

   A good choice for ``reference_energy`` is the energy from a classical multi-configuration calculation (e.g., :term:`CASCI`), which is typically close to the true eigenvalue.


Construction
------------

:class:`~qdk_chemistry.data.QpeResult` objects are typically created by the :doc:`PhaseEstimation <../algorithms/phase_estimation>` algorithm.
They can also be constructed manually using the appropriate class method:

- :meth:`~qdk_chemistry.data.QpeResult.from_time_evolution_result` — for Trotter-based QPE (requires ``evolution_time``)
- :meth:`~qdk_chemistry.data.QpeResult.from_qubitization_result` — for qubitization QPE (requires ``lambda_val``)

Time evolution example
~~~~~~~~~~~~~~~~~~~~~~

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qpe_result.py
      :language: python
      :start-after: # start-cell-create-from-phase
      :end-before: # end-cell-create-from-phase

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
- :doc:`HamiltonianUnitaryBuilder <../algorithms/hamiltonian_unitary_builder>`: Hamiltonian simulation or block encoding methods
- :doc:`Serialization <serialization>`: Data persistence formats
- See the ``examples/qpe_stretched_n2.ipynb`` notebook for an end-to-end :term:`QPE` workflow
