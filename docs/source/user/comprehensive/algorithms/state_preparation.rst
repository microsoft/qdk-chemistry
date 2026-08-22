State preparation
=================

The :class:`~qdk_chemistry.algorithms.StatePreparation` algorithm in QDK/Chemistry constructs quantum circuits that load classical representations of target wavefunctions onto qubits.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Wavefunction` instance as input and produces a :class:`~qdk_chemistry.data.Circuit` as output.
The output circuit, when executed, prepares the qubit register in a state that encodes the input wavefunction.

Overview
--------

The :class:`~qdk_chemistry.algorithms.StatePreparation` module provides tools for constructing quantum circuits that load classical representations of wavefunctions (e.g., a Slater determinant or a linear combination thereof, represented by the `Wavefunction` class)  onto qubits. It supports multiple approaches for state preparation, allowing users to choose the method best suited to their problem. Each approach is designed to efficiently encode quantum states for chemistry applications.

For details on individual methods and their technical implementations, see the `Available implementations`_ section below.

Using the StatePreparation
--------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a state preparation.
The ``run`` method returns a circuit object that, when executed, loads the input wavefunction onto a qubit register.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.StatePreparation` requires the following input:

Wavefunction
   A :class:`~qdk_chemistry.data.Wavefunction` instance containing the quantum state to be loaded onto qubits. This is typically obtained from a multi-configuration calculation using the :doc:`MultiConfigurationCalculator <mc_calculator>`. The method with which this encoding is achieved is implementation dependent.


.. rubric:: Creating a state preparation algorithm

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

Once configured, the :class:`~qdk_chemistry.algorithms.StatePreparation` can be used to generate a quantum circuit from a :class:`~qdk_chemistry.data.Wavefunction`.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.StatePreparation` provides a unified interface for state preparation methods.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. _sparse-isometry-gf2x:

Sparse Isometry
~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"sparse_isometry"``

This method is an optimized approach that leverages sparsity in the target wavefunction. It is a modification of the original sparse isometry work in :cite:`Malvetti2021`, and is native to QDK/Chemistry. By working only with the non-zero amplitudes, it substantially reduces circuit depth and gate count compared with dense methods, and is especially efficient for wavefunctions with sparse amplitude structure.

.. rubric:: How it works

A wavefunction built from :math:`d` determinants is written as

.. math::

   \left| \psi \right\rangle = \sum_{j=1}^{d} c_j \left| b_j \right\rangle ,
   \qquad b_j \in \{0,1\}^{n} ,

where each :math:`b_j` is the occupation bitstring of one determinant on :math:`n` qubits. For chemically relevant states :math:`d \ll 2^{n}`, so all but a vanishing fraction of the :math:`2^{n}` amplitudes are zero. Dense methods still pay for every one of them; the sparse isometry pays only for the :math:`d` that matter.

The support is collected into a binary matrix :math:`M \in \mathrm{GF}(2)^{n \times d}` whose columns are the determinant bitstrings. The algorithm then proceeds in four steps:

1. **Reduce.** Gaussian elimination over :math:`\mathrm{GF}(2)` brings :math:`M` to row echelon form of rank :math:`r \le n`. Elimination is preceded by two cheap simplifications: duplicate rows are cancelled against each other, and all-ones rows are cleared. When the reduced matrix is diagonal, an additional cascade removes one further row.

2. **Record.** Every row operation used in the reduction is tracked. A row addition over :math:`\mathrm{GF}(2)` is exactly a :term:`CNOT` on the qubit register and a row negation is exactly an ``X`` gate, so the elimination sequence is itself a Clifford circuit :math:`E` satisfying :math:`E \cdot M = M_{\mathrm{ref}}`.

3. **Prepare.** The reduced support occupies only :math:`r` qubits, so the amplitudes :math:`c_j` are loaded onto that smaller register by a nested state preparation algorithm, selected with the ``dense_state_prep`` setting.

4. **Recovery.** Replaying the recorded operations in reverse applies :math:`E^{-1}`, mapping the reduced basis states back onto the original determinant bitstrings :math:`b_j`.

The cost is therefore governed by :math:`d` and :math:`r` rather than by :math:`2^{n}`, and step 3 is the only part that is exponential in its (much smaller) register.

.. rubric:: Binary encoding

Setting ``binary_encoding`` to ``True`` replaces step 3 with a tighter encoding. Row echelon form bounds the reduced register at the *rank* :math:`r`, but distinguishing :math:`d` determinants only requires :math:`m = \lceil \log_2 d \rceil` qubits, and :math:`m` is frequently much smaller than :math:`r`. Binary encoding constructs an explicit bijection between the :math:`d` determinants and the integers :math:`0, \ldots, d-1`, written in binary on an :math:`m`-qubit dense register, so the nested preparation runs on :math:`2^{m}` amplitudes instead of :math:`2^{r}`.

The compression circuit is synthesized in two stages:

* **Diagonal encoding.** The pivot block of the reduced matrix is an identity, i.e. a *unary* encoding that spends one qubit per determinant. A staircase of :term:`CNOT` gates normalizes it, and a divide-and-conquer cascade of :term:`CNOT` and Toffoli gates then folds the unary pattern into a binary counter, collapsing the pivot columns onto the :math:`m` dense rows.

* **Non-pivot processing.** The remaining columns carry no pivot and are handled in power-of-two batches. Each batch emits an address-controlled lookup block that writes the correct binary label into the dense register conditioned on the sparse indicator rows, and then clears those rows. The synthesizer costs a single lookup against a split into smaller chunks and keeps whichever needs fewer Toffoli gates.

Once both stages complete, every sparse row is guaranteed to hold :math:`\left| 0 \right\rangle`, so the state lives entirely in the dense register. The amplitudes are prepared there, and the whole compression circuit is inverted to scatter back to the full register.

Lookup blocks need helper qubits. Rather than allocating fresh ancillas, the synthesizer first borrows idle system qubits — those absent from the reduced support — and allocates additional qubits only when that pool is exhausted.

Binary encoding requires spare rows to compress into. When the state is already dense, meaning :math:`m \ge n_{\mathrm{rows}}`, it does not apply and the algorithm transparently falls back to the standard path described above.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-configure-binary-encoding
      :end-before: # end-cell-configure-binary-encoding

.. note::
   Setting ``measurement_based_uncompute`` to ``True`` uncomputes the helper qubits of each lookup block by measurement and a classically controlled correction rather than by Toffoli gates. This trades Toffoli count for mid-circuit measurement and feedforward, so the resulting circuit requires a target profile that supports adaptive execution.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``binary_encoding``
     - bool
     - Compress the reduced subspace with binary encoding instead of preparing it directly. Best effort: ``True`` requests the encoding rather than guaranteeing it, since it is skipped when :math:`m \ge n_{\mathrm{rows}}` and the standard path is used instead. Default is False.
   * - ``dense_state_prep``
     - AlgorithmRef
     - State preparation algorithm used for the dense subspace. Default is ``dense_pure_state``.
   * - ``include_negative_controls``
     - bool
     - Allow anti-controls as well as controls in the lookup blocks. Default is True.
   * - ``measurement_based_uncompute``
     - bool
     - Uncompute lookup helper qubits by measurement instead of Toffoli gates. Default is False.

This algorithm declares no transpilation settings of its own. Transpilation applies only when
the nested ``dense_state_prep`` algorithm emits a Qiskit circuit, and is configured on that
algorithm:

.. literalinclude:: /_static/examples/python/state_preparation.py
   :language: python
   :start-after: start-cell-configure
   :end-before: end-cell-configure

Dense Pure State
~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"dense_pure_state"``

This method expands the wavefunction into its full amplitude vector and synthesizes that vector exactly. Synthesis is delegated to `PreparePureStateD <https://github.com/microsoft/qdk/blob/main/library/std/src/Std/StatePreparation.qs>`_ from the Q# standard library, which follows the construction of Shende, Bullock, and Markov :cite:`Shende2006`.
Each determinant's coefficient is placed at the index given by its occupation bitstring, giving a dense vector of :math:`2^{n}` real amplitudes that is handed to ``PreparePureStateD``.

.. rubric:: Requirements

The coefficients must be real; a wavefunction with a non-zero imaginary part is rejected. The register is limited to 32 qubits, which bounds the size of the dense amplitude vector.

.. rubric:: Settings

This implementation exposes no settings.

Regular Isometry
~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qiskit_regular_isometry"``

This method uses regular isometry synthesis via `Qiskit <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.StatePreparation>`_, implementing the isometry-based approach proposed by Matthias Christandl :cite:`Christandl2016`. It provides a general solution for state preparation, and is suitable for cases where a dense representation is required or preferred. Like `Dense Pure State`_ it synthesizes the full amplitude vector, but it is provided through the :ref:`plugin system <plugin-system>` and returns an OpenQASM circuit, which makes it the natural choice for Qiskit-based workflows.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``basis_gates``
     - list[str]
     - Basis gates for transpilation. Default is ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"].
   * - ``transpile``
     - bool
     - Whether to transpile the circuit. Default is True.
   * - ``transpile_optimization_level``
     - int
     - Optimization level for transpilation (0-3). Default is 0.

For more details on how QDK/Chemistry interfaces with external packages, see the :ref:`plugin system <plugin-system>` documentation.

Alias Sampling
~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"alias_sampling"``

This method implements the coherent alias sampling PREPARE oracle of Babbush et al. :cite:`Babbush2018` (section III.D). Given :math:`L` non-negative coefficients :math:`c_\ell`, it prepares

.. math::

   \sum_{\ell} \sqrt{\tilde{p}_\ell} \left| \ell \right\rangle \left| \mathrm{garbage}_\ell \right\rangle ,
   \qquad \tilde{p}_\ell \approx \frac{|c_\ell|}{\sum_k |c_k|} ,

where :math:`\tilde{p}` is the target distribution discretized to :math:`\mu` bits. Its Toffoli count is dominated by a single :math:`O(L)` QROM lookup rather than by the amplitude precision, which is what makes it attractive for large Hamiltonians.

.. warning::
   This is a **block-encoding subroutine, not a general-purpose state preparation**. It differs from `Sparse Isometry`_ and `Dense Pure State`_ in two ways. First, the index register is left entangled with ancilla, so the output is only meaningful inside an :term:`LCU` or qubitization circuit where :math:`\mathrm{PREPARE}^\dagger` later uncomputes the garbage. Second, it realizes :math:`\sqrt{|c_\ell| / \sum_k |c_k|}` rather than :math:`c_\ell / \lVert c \rVert_2`, and has no way to represent a coefficient's sign, so negative coefficients are rejected. Index :math:`\ell` is the position of a coefficient in the wavefunction's coefficient vector, not a determinant bitstring, so the resulting circuit carries no fermionic encoding.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``bits_precision``
     - int
     - Number of bits :math:`\mu` of precision for the alias table's keep probabilities. Each prepared probability lands within :math:`2^{-\mu}` of the target, at the cost of one extra uniform qubit and one extra QROM output qubit per bit. Default is 10.

QROM
~~~~

.. rubric:: Factory name: ``"qrom"``

This method prepares an :math:`n`-qubit state with :math:`n` layers of multiplexed :math:`R_y` rotations, where each layer's rotation angles are loaded from a QROM table and applied through a phase gradient register. It uses only :math:`n = \lceil \log_2 L \rceil` state qubits, plus scratch ancilla per lookup, in exchange for :math:`n` QROM lookups.

As with `Alias Sampling`_, the index is the position of a coefficient in the wavefunction's coefficient vector rather than a determinant bitstring, so the resulting circuit carries no fermionic encoding.

.. warning::
   **Negative coefficients are not supported yet.** :math:`R_y` rotations only generate non-negative amplitudes, so signs are applied by a separate QROM-loaded ``Z`` phase kickback. That lookup is not correctly uncomputed: the sign ancilla is released while still entangled with the state register, so it is implicitly measured and the signs collapse at random. Magnitudes remain correct, but the sign pattern varies between simulator seeds. Passing a negative coefficient emits a :class:`RuntimeWarning`; use ``dense_pure_state`` for signed amplitudes until this is fixed.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``rotation_bit_precision``
     - int
     - Number of bits of precision used for the QROM-loaded :math:`R_y` rotation angles. Higher values reduce the synthesis error of each multiplexed rotation at the cost of a wider QROM output register. Default is 10.

Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`: Input wavefunction for circuit construction
- :class:`~qdk_chemistry.data.Circuit`: Output circuit that prepares the wavefunction on qubits

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/state_preparation.py>`_ script.
- :doc:`ExpectationEstimator <expectation_estimator>`: Estimate the energy of prepared states
- :doc:`QubitMapper <qubit_mapper>`: Map Hamiltonians to qubit operators
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
