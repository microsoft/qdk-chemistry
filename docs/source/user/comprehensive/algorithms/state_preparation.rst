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

.. _sparse-isometry:
.. _sparse-isometry-gf2x:

Sparse Isometry
~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"sparse_isometry"``

This method is an optimized approach that leverages sparsity in the target wavefunction. It is a modification of the original sparse isometry work in :cite:`Malvetti2021`, and is native to QDK/Chemistry, described in :cite:`Chen2026`. By working only with the non-zero amplitudes, it substantially reduces circuit depth and gate count compared with dense methods, and is especially efficient for wavefunctions with sparse amplitude structure.

.. rubric:: How it works

A wavefunction built from :math:`d` determinants is written as

.. math::

   \left| \psi \right\rangle = \sum_{j=1}^{d} c_j \left| b_j \right\rangle ,
   \qquad b_j \in \{0,1\}^{n} ,

where each :math:`b_j` is the occupation bitstring of one determinant on :math:`n` qubits. For chemically relevant states :math:`d \ll 2^{n}`, so all but a vanishing fraction of the :math:`2^{n}` amplitudes are zero. Dense methods still pay for every one of them; the sparse isometry pays only for the :math:`d` that matter.

The support is collected into a binary matrix :math:`M \in \mathrm{GF}(2)^{n \times d}` whose columns are the determinant bitstrings. The algorithm then proceeds in four steps:

1. **Reduce.** Gaussian elimination over :math:`\mathrm{GF}(2)` brings :math:`M` to row echelon form of rank :math:`r \le n`. Elimination is preceded by two simplifications: duplicate rows are cancelled against each other, and all-ones rows are cleared. When the reduced matrix is diagonal, an additional cascade removes one further row, so the final reduced width may be smaller than :math:`r`.

2. **Record.** Every row operation used in the reduction is tracked. A row addition over :math:`\mathrm{GF}(2)` is exactly a :term:`CNOT` on the qubit register and a row negation is exactly an ``X`` gate, so the elimination sequence is itself a Clifford circuit :math:`E` satisfying :math:`E \cdot M = M_{\mathrm{ref}}`.

3. **Prepare.** The amplitudes :math:`c_j` are loaded onto the remaining reduced rows by a nested state-preparation algorithm selected with the ``dense_state_prep`` setting.

4. **Recovery.** Replaying the recorded operations in reverse applies :math:`E^{-1}`, mapping the reduced basis states back onto the original determinant bitstrings :math:`b_j`.

The cost is therefore governed by :math:`d` and the reduced width rather than by :math:`2^{n}`, and step 3 is the only part that is exponential in that smaller register.

.. rubric:: Binary encoding

Setting ``binary_encoding`` to ``True`` replaces step 3 when :math:`m = \lceil \log_2 d \rceil` is smaller than the number of rows left after reduction. It constructs a one-to-one map from the :math:`d` determinants into the :math:`2^m` basis states of an :math:`m`-qubit dense register, so the nested preparation runs on that smaller amplitude vector.

The compression circuit is synthesized in two stages:

* **Diagonal encoding.** The pivot block of the reduced matrix is an identity, i.e. a *unary* encoding that spends one qubit per determinant. A staircase of :term:`CNOT` gates normalizes it, and a divide-and-conquer cascade of :term:`CNOT` and Toffoli gates then folds the unary pattern into a binary counter, collapsing the pivot columns onto the :math:`m` dense rows.

* **Non-pivot processing.** The remaining columns carry no pivot and are handled in power-of-two batches. Each batch emits an address-controlled lookup block that writes the correct binary label into the dense register conditioned on the sparse indicator rows, and then clears those rows. The synthesizer costs a single lookup against a split into smaller chunks and keeps whichever needs fewer Toffoli gates.

Once both stages complete, every sparse row is guaranteed to hold :math:`\left| 0 \right\rangle`, so the state lives entirely in the dense register. The amplitudes are prepared there, and the whole compression circuit is inverted to scatter back to the full register.

Lookup blocks add CCZ operations and need helper qubits. Rather than allocating fresh ancillas, the synthesizer first borrows idle system qubits — those absent from the reduced support — and allocates additional qubits only when that pool is exhausted. The smaller amplitude-loading register therefore does not imply fewer qubits or non-Clifford gates for the complete circuit.

Binary encoding applies only when :math:`m < n_{\mathrm{rows}}`, where :math:`n_{\mathrm{rows}}` is the width left after reduction. Otherwise the algorithm transparently falls back to the standard path described above.

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

This method implements the coherent alias sampling oracle of Babbush et al. :cite:`Babbush2018` (section III.D). Given :math:`L` non-negative coefficients :math:`c_\ell`, it prepares

.. math::

   \sum_{\ell} \sqrt{\tilde{p}_\ell} \left| \ell \right\rangle \left| \mathrm{garbage}_\ell \right\rangle ,
   \qquad \tilde{p}_\ell \approx \frac{c_\ell^2}{\sum_k c_k^2} ,

where :math:`\tilde{p}` is the target distribution discretized to :math:`\mu` bits.

.. warning::
   This is a **block-encoding subroutine, not a general state preparation for algorithms like QPE**. The index register is left entangled with ancilla, so the output is only meaningful inside an :term:`LCU` or qubitization circuit where :math:`\mathrm{PREPARE}^\dagger` later uncomputes the garbage. Negative coefficients are not supported.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``bits_precision``
     - int
     - Number of bits :math:`\mu` of precision for the alias table's keep probabilities. Each prepared probability is within :math:`1/(L 2^{\mu})` of the target for :math:`L` coefficients. The upper bound of 30 is a sanity limit as :math:`2^{-30}` is far below chemical accuracy. Default is 10.

.. _matrix-product-state:

Matrix Product State
~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"matrix_product_state"``

This method prepares a matrix product state (MPS) stored in an :class:`~qdk_chemistry.data.MPSContainer` following :cite:`Berry2025` and :cite:`Rupprecht2026`.
The MPS must be real and right-canonical with orthogonality center at site 0, with open boundaries, and either the physical basis ``('0', 'u', 'd', '2')`` on every site or the physical basis ``('0', '1')`` on every site.
The physical basis is loaded in the blocked Jordan-Wigner layout.
An ancilla register of :math:`\lceil \log_2 \chi \rceil` qubits carries the virtual bond of maximal dimension :math:`\chi`.
Every site unitary acts on the qubits of its site and the ancilla register, and is synthesized by one of two methods selected with the ``unitary_synthesis`` setting, excluding site 0, which is prepared as the initial state.
Givens decomposition converts the orthogonal factors into adjacent rotations and sign corrections for the circuit.
A final layer of CZ gates applies the fermionic signs of reordering these modes into qubit order.
It relies on the container's validation of adjacent bond spaces.
The utility ``matrix_product_state_synthesis(container, ancilla_dimension, unitary_synthesis="dense")`` returns the synthesized sites in chain order, excluding site 0.
Single-site synthesis is available through ``dense_unitary_synthesis(site, ancilla_dimension, following_right_factor)`` and ``block_sparse_unitary_synthesis(site, ancilla_dimension)``, both taking an ``MPSSite``.
They return the native ``DenseSiteSynthesis`` and ``SparseSiteSynthesis`` objects of ``qdk_chemistry.utils.unitary_synthesis``, which Q# structs of the same names mirror field by field.
To obtain preparation data without building a circuit, call ``preparer.generate_matrix_product_state_preparation_data(mps_container)``.
This instance method accepts only an ``MPSContainer`` and uses the preparer's ``unitary_synthesis`` setting.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``allocate_phase_gradient``
     - bool
     - Whether the circuit allocates and prepares its own phase gradient register. If False, the operation takes ``rotation_bit_precision`` trailing qubits holding a phase gradient state owned by the caller. Default is True.
   * - ``rotation_bit_precision``
     - int
     - Size of the phase gradient register, which sets the precision of every rotation angle. The upper bound of 30 is a sanity limit as :math:`2^{-30}` is far below chemical accuracy. Default is 10.
   * - ``unitary_synthesis``
     - str
     - Site unitary synthesis method: ``"dense"`` or ``"block_sparse"``. Default is ``"dense"``.

Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`: Input wavefunction for circuit construction
- :class:`~qdk_chemistry.data.MPSContainer`: Matrix product state input for the MPS-based methods
- :class:`~qdk_chemistry.data.Circuit`: Output circuit that prepares the wavefunction on qubits

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/state_preparation.py>`_ script.
- :doc:`ExpectationEstimator <expectation_estimator>`: Estimate the energy of prepared states
- :doc:`QubitMapper <qubit_mapper>`: Map Hamiltonians to qubit operators
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
