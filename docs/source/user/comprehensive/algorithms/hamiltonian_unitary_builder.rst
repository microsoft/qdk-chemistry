Hamiltonian Unitary Builder
===========================

The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` algorithm in QDK/Chemistry constructs a unitary based on the Hamiltonian, such as time simulation unitary :math:`U(t) = e^{-iHt}` or block-encoded unitary :math:`U = \frac{H}{\|H\|}`.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.QubitHamiltonian` and produces a :class:`~qdk_chemistry.data.UnitaryRepresentation` as output.

Overview
--------

Building unitary from Hamiltonian — such as the Hamiltonian simulation unitary :math:`U(t) = e^{-iHt}` or block encoding unitary :math:`U = \frac{H}{\|H\|}` — is a central subroutine in many quantum algorithms.
The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` provides a unified interface for methods that construct this operator from a :class:`~qdk_chemistry.data.QubitHamiltonian`.

QDK/Chemistry currently provides two families of implementations for this task: Trotter-Suzuki product formulas and block encoding.

The resulting :class:`~qdk_chemistry.data.UnitaryRepresentation` objects wrap either a ``PauliProductFormulaContainer`` (Trotter) or an ``LCUContainer`` (block encoding).


Using the HamiltonianUnitaryBuilder
------------------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a Hamiltonian unitary builder.
The ``run`` method returns a :class:`~qdk_chemistry.data.UnitaryRepresentation` object that can be used by any algorithm that requires a Hamiltonian simulation unitary (e.g., :doc:`PhaseEstimation <phase_estimation>`).

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` requires the following inputs:

QubitHamiltonian
   A :class:`~qdk_chemistry.data.QubitHamiltonian` containing the Pauli-string representation of the Hamiltonian.
   This can be obtained from the :doc:`QubitMapper <qubit_mapper>` algorithm, constructed from a :doc:`model Hamiltonian <../model_hamiltonians>`, or built directly.

.. rubric:: Creating a builder

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_unitary_builder.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings vary by implementation.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_unitary_builder.py
      :language: python
      :start-after: # start-cell-configure-trotter
      :end-before: # end-cell-configure-trotter

.. rubric:: Running the builder

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_unitary_builder.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` provides a unified interface for Hamiltonian simulation methods.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_unitary_builder.py
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


.. _zassenhaus-builder:

Zassenhaus product formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. rubric:: Factory name: ``"zassenhaus"``

The Zassenhaus decomposition approximates the time-evolution operator by recursively generating commutator correction terms.
For a Hamiltonian :math:`H = \sum_j \alpha_j P_j`, the first-order approximation is the Trotter product, and higher-order approximations introduce explicit product-formula factors for lower-order commutator corrections:

**First-order Zassenhaus** (:math:`p = 1`):
Matches the first-order Trotter product formula.

**Second-order Zassenhaus** (:math:`p = 2`):

.. math::

   e^{-iHt} \approx \left[ \prod_{j > k} e^{-\frac{1}{2} [H_j, H_k] t^2} \cdot \prod_j e^{-i H_j t} \right]^N

**Higher orders** (up to :math:`p = 4`) recursively construct explicit nested-commutator terms. Unlike Trotter-Suzuki formulas where step count is used to reduce commutator errors, the Zassenhaus builder computes low-order corrections explicitly.

The number of divisions can be specified directly (``num_divisions``) or computed automatically from a ``target_accuracy`` using one of two bounds:

Commutator bound (default)
   A tighter bound based on nested commutators of the first omitted exponent:
   :math:`N = \lceil \frac{\|C_{p+1}(-iH_1, \dots, -iH_m)\|^{1/p} t^{1+1/p}}{\epsilon^{1/p}} \rceil`

Naive bound
   A looser bound using the absolute coefficient sum of the first omitted exponent and the 1-norm of the Hamiltonian:
   :math:`N = \lceil \frac{(\kappa_{p+1} 2^p \|H\|_1^{p+1})^{1/p} t^{1+1/p}}{\epsilon^{1/p}} \rceil`

When ``order`` is set to ``0`` (auto), the builder dynamically sweeps orders 2, 3, and 4 to select the order that minimizes the total gate count (product of step reps and step terms) for the target accuracy.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``order``
     - int
     - Zassenhaus order (0 for auto, or 1 to 4). Default is 0.
   * - ``target_accuracy``
     - float
     - Target approximation error :math:`\epsilon`. When set to 0.0 (default), automatic step-count estimation is disabled.
   * - ``num_divisions``
     - int
     - Explicit number of Zassenhaus steps :math:`N`. When set to 0 (default), determined from ``target_accuracy``.
   * - ``error_bound``
     - str
     - Error bound strategy: ``"commutator"`` (default, tighter) or ``"naive"`` (simpler).
   * - ``weight_threshold``
     - float
     - Coefficient threshold below which Pauli terms are discarded. Default is 1e-12.


Consuming term partitions
-------------------------

When the input :class:`~qdk_chemistry.data.QubitHamiltonian` carries a populated :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition`, the Trotter builder consumes it directly:

* :class:`~qdk_chemistry.data.LayeredPartition` (group → layer → index) is used as-is — the outer level controls the Strang/Suzuki splitting and each inner layer becomes one parallelisable sub-step.
* :class:`~qdk_chemistry.data.FlatPartition` (group → index) is interpreted as a layered partition with one layer per group.

In both cases groups are sorted by ascending layer count so that the smallest groups sit on the outside of the Strang/Suzuki splitting, which maximises merging at recursion boundaries.
This typically reduces the number of distinct exponentials per Trotter step and the saving compounds through the recursion at higher orders.

When ``term_partition is None`` each Pauli term is exponentiated as its own group.
Pre-populate the partition using the :ref:`term_grouper algorithm <algorithms-term-grouper>` or one of the :ref:`spin model Hamiltonian builders <model-term-partition>` to enable group-aware scheduling.

For lattice models, the edge coloring stored on :class:`~qdk_chemistry.data.LatticeGraph` feeds directly into the partition: edges of the same color have disjoint qubit supports and form a single parallelisable layer.
The :ref:`model Hamiltonian builders <model-term-partition>` consume this coloring automatically, so no manual geometry handling is required.

Example::

    from qdk_chemistry.data import LatticeGraph
    from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian
    from qdk_chemistry.algorithms import registry

    # 4-site open Ising chain: H = Σ Z_i Z_{i+1} + 0.5 Σ X_i
    graph = LatticeGraph.chain(4, periodic=False)
    hamiltonian = create_ising_hamiltonian(graph, j=1.0, h=0.5)

    # The edge coloring partitions ZZ terms into two layers by color:
    #   group 0 (field):  1 layer  → [X₀, X₁, X₂, X₃]
    #   group 1 (ZZ):     2 layers → [{Z₀Z₁, Z₂Z₃}, {Z₁Z₂}]
    print(hamiltonian.term_partition)
    # LayeredPartition(strategy='geometry_coloring', num_groups=2)

    # Second-order Trotter with 1 division produces the Strang splitting:
    #   S₂(t) = e^{fields·t/2}  e^{ZZ_layer0·t}  e^{ZZ_layer1·t}
    #           e^{fields·t/2}
    # where same-layer ZZ terms (e.g. Z₀Z₁ and Z₂Z₃) have disjoint
    # qubit support and are exponentiated independently within one step.
    trotter = registry.create("time_evolution_builder", "trotter")
    trotter.settings().update({"order": 2, "num_divisions": 1})
    evolution = trotter.run(hamiltonian, time=1.0)
    container = evolution.get_container()

    # The grouped schedule uses 11 exponentiated terms per step,
    # vs. 13 for the ungrouped fallback — a 15% reduction that
    # compounds at higher Suzuki orders.
    print(f"{len(container.step_terms)} terms per Trotter step")
    for term in container.step_terms:
        label = ['I'] * 4
        for q, p in term.pauli_term.items():
            label[q] = p
        print(f"  exp(-i * {term.angle:+.4f} * {''.join(reversed(label))})")
    # Output:
    #   exp(-i * +0.2500 * IIIX)
    #   exp(-i * +0.2500 * IIXI)
    #   exp(-i * +0.2500 * IXII)
    #   exp(-i * +0.2500 * XIII)
    #   exp(-i * +1.0000 * IIZZ)
    #   exp(-i * +1.0000 * ZZII)
    #   exp(-i * +1.0000 * IZZI)
    #   exp(-i * +0.2500 * IIIX)
    #   exp(-i * +0.2500 * IIXI)
    #   exp(-i * +0.2500 * IXII)
    #   exp(-i * +0.2500 * XIII)


.. _block-encoding-builder:

Block encoding
~~~~~~~~~~~~~~

Block encoding is a technique for embedding a non-unitary operator (such as :math:`H / \lambda`) into a larger unitary circuit.
Given an :math:`n`-qubit Hamiltonian :math:`H`, a block encoding uses :math:`a` ancilla qubits to construct a unitary :math:`U` such that:

.. math::

   (\langle 0|^{\otimes a} \otimes I)\, U\, (|0\rangle^{\otimes a} \otimes I) = \frac{H}{\lambda}

where :math:`\lambda \geq \|H\|` is a normalization factor (typically the L1 norm of the coefficients).
The Hamiltonian is encoded in the top-left block of the larger unitary — hence the name "block encoding."

Block encodings are the foundation for quantum algorithms such as qubitization, QSVT, and quantum linear system solvers.
QDK/Chemistry currently provides the **LCU** (Linear Combination of Unitaries) method for constructing block encodings.

.. _lcu-builder:

LCU (Linear Combination of Unitaries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. rubric:: Factory name: ``"lcu"``

The LCU builder constructs a block encoding of :math:`H / \lambda` using the PREPARE–SELECT pattern :cite:`Childs2012`.
For a Hamiltonian :math:`H = \sum_{j=1}^{L} \alpha_j P_j` with :math:`\lambda = \sum_j |\alpha_j|`:

**PREPARE oracle**

Prepares an ancilla register in the state :math:`|\psi\rangle = \sum_j \sqrt{|\alpha_j| / \lambda}\, |j\rangle` using :math:`\lceil \log_2 L \rceil` qubits.

**SELECT oracle**

Applies the corresponding Pauli string controlled on the ancilla index:

.. math::

   \text{SELECT} = \sum_{j=0}^{L-1} |j\rangle\langle j| \otimes P_j

with sign corrections absorbed into a phase vector.

**Block encoding circuit**

The composite circuit :math:`\text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP}` realizes the block encoding:

.. math::

   (\langle 0| \otimes I)\, \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP}\, (|0\rangle \otimes I) = \frac{H}{\lambda}

**Quantum walk operator**

When ``quantum_walk=True``, the builder wraps the block encoding with a reflection to form the qubitization walk operator:

.. math::

   W = (2|0\rangle\langle 0| - I) \cdot \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP}

The walk operator has eigenvalues :math:`e^{\pm i \arccos(E_k/\lambda)}` where :math:`E_k` are the eigenvalues of :math:`H`.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``power``
     - int
     - The power to which the walk operator is raised. Default is 1.
   * - ``quantum_walk``
     - bool
     - If ``True``, wrap the block encoding with a reflection to form the qubitization walk operator. If ``False``, produce the plain block encoding. Default is ``False``.
   * - ``tolerance``
     - float
     - Minimum L1 norm below which the decomposition is considered ill-defined. Default is 1e-12.

.. rubric:: Example

::

    from qdk_chemistry.algorithms import registry

    lcu = registry.create("hamiltonian_unitary_builder", "lcu")
    lcu.settings().update({"quantum_walk": True})
    unitary = lcu.run(qubit_hamiltonian)

The resulting :class:`~qdk_chemistry.data.UnitaryRepresentation` wraps an ``LCUContainer`` containing the Prepare and Select oracles.


Related classes
---------------

- :class:`~qdk_chemistry.data.UnitaryRepresentation`: Output data class wrapping the exponentiated Pauli terms or LCU container
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Input qubit Hamiltonian
- :doc:`PhaseEstimation <phase_estimation>`: Consumer of the hamiltonian unitary

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/hamiltonian_unitary_builder.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Quantum phase estimation algorithms
- :doc:`QubitMapper <qubit_mapper>`: Map fermionic Hamiltonians to qubit operators
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
