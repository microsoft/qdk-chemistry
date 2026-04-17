Unitary builder
======================

The :class:`~qdk_chemistry.algorithms.UnitaryBuilder` algorithm in QDK/Chemistry constructs quantum circuits that implement the Hamiltonian simulation unitary :math:`U(t) = e^{-iHt}` or block-encoded unitary :math:`U = \frac{H}{\|H\|}`.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.QubitHamiltonian` and a time parameter as input and produces a :class:`~qdk_chemistry.data.UnitaryRepresentation` as output.

Overview
--------

Hamiltonian simulation — constructing the unitary :math:`U(t) = e^{-iHt}` — is a central subroutine in many quantum algorithms.
The :class:`~qdk_chemistry.algorithms.UnitaryBuilder` provides a unified interface for methods that construct this operator from a :class:`~qdk_chemistry.data.QubitHamiltonian`.

QDK/Chemistry currently provides Trotter-Suzuki product formulas for this task.
These decompose :math:`e^{-iHt}` into a sequence of elementary Pauli rotations :math:`e^{-i\theta P}` that can be directly implemented as quantum gates, with controllable approximation error via the Trotter order and number of time divisions :cite:`Suzuki1992`.
The resulting :class:`~qdk_chemistry.data.UnitaryRepresentation` objects wrap a ``PauliProductFormulaContainer`` — a list of exponentiated Pauli terms with a repetition count.


Using the UnitaryBuilder
------------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a time evolution builder.
The ``run`` method returns a :class:`~qdk_chemistry.data.UnitaryRepresentation` object that can be used by any algorithm that requires a Hamiltonian simulation unitary (e.g., :doc:`PhaseEstimation <phase_estimation>`).

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.UnitaryBuilder` requires the following inputs:

QubitHamiltonian
   A :class:`~qdk_chemistry.data.QubitHamiltonian` containing the Pauli-string representation of the Hamiltonian.
   This can be obtained from the :doc:`QubitMapper <qubit_mapper>` algorithm, constructed from a :doc:`model Hamiltonian <../model_hamiltonians>`, or built directly.

Time
   A float specifying the evolution time :math:`t` in :math:`U(t) = e^{-iHt}`.

.. rubric:: Creating a builder

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/unitary_builder.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings vary by implementation.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/unitary_builder.py
      :language: python
      :start-after: # start-cell-configure-trotter
      :end-before: # end-cell-configure-trotter

.. rubric:: Running the builder

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/unitary_builder.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.UnitaryBuilder` provides a unified interface for Hamiltonian simulation methods.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/unitary_builder.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. _trotter-builder:

Trotter-Suzuki product formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. rubric:: Factory name: ``"trotter"``

The Trotter-Suzuki decomposition approximates the time-evolution operator as a product of individual Pauli exponentials.
For a Hamiltonian :math:`H = \sum_j \alpha_j P_j`:

**First-order Trotter** (:math:`p = 1`):

.. math::

   e^{-iHt} \approx \left[\prod_j e^{-i\alpha_j P_j t / N}\right]^N

**Second-order Trotter** (Strang splitting, :math:`p = 2`):

.. math::

   e^{-iHt} \approx \left[\prod_{j=1}^{L-1} e^{-i\alpha_j P_j t / 2N} \cdot e^{-i\alpha_L P_L t / N} \cdot \prod_{j=L-1}^{1} e^{-i\alpha_j P_j t / 2N}\right]^N

**Higher even orders** are constructed via the recursive Suzuki composition :cite:`Suzuki1992`:

.. math::

   S_{2k}(t) = S_{2k-2}(u_k t)^2 \, S_{2k-2}\!\bigl((1-4u_k) t\bigr) \, S_{2k-2}(u_k t)^2

where :math:`u_k = 1/(4 - 4^{1/(2k-1)})`.

The number of Trotter steps :math:`N` can be specified directly (``num_divisions``) or computed automatically from a ``target_accuracy`` using one of two error bounds:

Commutator bound (default)
   A tighter bound :cite:`Childs2021` based on nested commutators:
   :math:`N = \lceil \frac{t^{2}}{2\epsilon} \sum_{j<k}\lVert[\alpha_jP_j,\alpha_kP_k]\rVert \rceil`

Naive bound
   A looser triangle-inequality bound:
   :math:`N = \lceil (\sum_j|\alpha_j|)^{2}t^{2}/\epsilon \rceil`

When both ``num_divisions`` and ``target_accuracy`` are specified, the builder uses whichever requires more Trotter steps.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``order``
     - int
     - Trotter-Suzuki order (1 for first-order, 2+ for higher even orders). Default is 1.
   * - ``target_accuracy``
     - float
     - Target approximation error :math:`\epsilon`. When set to 0.0 (default), automatic step-count estimation is disabled.
   * - ``num_divisions``
     - int
     - Explicit number of Trotter steps :math:`N`. When set to 0 (default), determined from ``target_accuracy``.
   * - ``error_bound``
     - str
     - Error bound strategy: ``"commutator"`` (default, tighter) or ``"naive"`` (simpler).
   * - ``weight_threshold``
     - float
     - Coefficient threshold below which Pauli terms are discarded. Default is 1e-12.


Related classes
---------------

- :class:`~qdk_chemistry.data.UnitaryRepresentation`: Output data class wrapping the exponentiated Pauli terms or linear combinations of unitaries
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Input qubit Hamiltonian
- :doc:`PhaseEstimation <phase_estimation>`: One consumer of time-evolution unitaries

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/unitary_builder.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Quantum phase estimation algorithms
- :doc:`QubitMapper <qubit_mapper>`: Map fermionic Hamiltonians to qubit operators
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
