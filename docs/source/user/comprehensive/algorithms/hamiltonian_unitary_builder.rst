Hamiltonian Unitary Builder
===========================

The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` algorithm in QDK/Chemistry constructs a unitary based on the Hamiltonian, such as time simulation unitary :math:`U(t) = e^{-iHt}` or block-encoded unitary :math:`U = \frac{H}{\|H\|}`.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.QubitHamiltonian` and produces a :class:`~qdk_chemistry.data.UnitaryRepresentation` as output.

Overview
--------

Building unitary from Hamiltonian — such as the Hamiltonian simulation unitary :math:`U(t) = e^{-iHt}` or block encoding unitary :math:`U = \frac{H}{\|H\|}` — is a central subroutine in many quantum algorithms.
The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` provides a unified interface for methods that construct this operator from a :class:`~qdk_chemistry.data.QubitHamiltonian`.

QDK/Chemistry provides several families of builders for this task. Three are *product-formula* methods that decompose :math:`e^{-iHt}` into a sequence of elementary Pauli rotations :math:`e^{-i\theta P}` that can be directly implemented as quantum gates:

- **Trotter-Suzuki** — deterministic product formulas that apply every Hamiltonian term in a fixed, repeated sequence, with controllable error via the Trotter order and number of time divisions :cite:`Suzuki1992`.
- **qDRIFT** — randomized product formulas that sample terms with probability proportional to their coefficient magnitude, trading the fixed term ordering for a gate cost that depends on the Hamiltonian 1-norm rather than the number of terms :cite:`Campbell2019`.
- **Partially randomized** — a hybrid that treats the largest-weight terms deterministically with Trotter and the remaining long tail with qDRIFT-style sampling :cite:`Guenther2025`.

A fourth family, **block encoding**, instead embeds :math:`U = \frac{H}{\|H\|}` as a sub-block of a larger unitary via a linear combination of unitaries (LCU).

The resulting :class:`~qdk_chemistry.data.UnitaryRepresentation` objects wrap either a ``PauliProductFormulaContainer`` (product formulas) or an ``LCUContainer`` (block encoding).


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

.. _qdrift-builder:

qDRIFT randomized product formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. rubric:: Factory name: ``"qdrift"``

The qDRIFT method replaces the fixed term ordering of Trotter-Suzuki with *randomized sampling* :cite:`Campbell2019`.
Instead of applying every term once per step, it draws each elementary exponential independently from a probability distribution weighted by coefficient magnitude.
For a Hamiltonian :math:`H = \sum_j h_j P_j` with 1-norm :math:`\lambda = \sum_j |h_j|`:

#. Form the distribution :math:`p_j = |h_j| / \lambda`.
#. Draw :math:`N` terms independently according to :math:`p_j`.
#. Apply each sampled term at a *fixed* angle that depends only on :math:`\lambda`, :math:`t`, and :math:`N`:

.. math::

   U(t) \approx \prod_{k=1}^{N} e^{-i\,\mathrm{sign}(h_{j_k})\,\frac{\lambda t}{N}\,P_{j_k}}

The per-term angle :math:`\lambda t / N` is independent of the individual coefficient :math:`h_j`, so the gate cost depends on the 1-norm :math:`\lambda` and the target accuracy rather than on the number of terms :math:`L`.
This makes qDRIFT attractive for Hamiltonians with many small terms.

The approximation error (in expectation over the randomness) is bounded by :cite:`Campbell2019`

.. math::

   \epsilon \leq \frac{2 \lambda^2 t^2}{N}.

The number of samples :math:`N` can be set directly (``num_samples``) or computed automatically from a ``target_accuracy`` :math:`\epsilon` by inverting the bound, :math:`N = \lceil 2\lambda^2 t^2 / \epsilon \rceil`.
When both are provided, the larger of the two values is used, so ``num_samples`` acts as a floor.

Consecutive sampled terms that are identical and lie within a mutually commuting run can be fused via ``merge_duplicate_terms`` to reduce circuit depth; the merge is exact and preserves the error bound.
The ``commutation_type`` setting controls how commuting runs are detected: ``"qubit_wise"`` is stricter but always safe, while ``"general"`` (default) admits larger merge groups.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``num_samples``
     - int
     - Number of random samples :math:`N`. Acts as a floor when ``target_accuracy`` is set. Default is 100.
   * - ``target_accuracy``
     - float
     - Target approximation error :math:`\epsilon`. When set to 0.0 (default), automatic sample-count estimation is disabled.
   * - ``error_bound``
     - str
     - Strategy for the qDRIFT error bound. Currently only ``"campbell"`` (default) is supported.
   * - ``weight_threshold``
     - float
     - Coefficient threshold below which Pauli terms are discarded. Default is 1e-12.
   * - ``seed``
     - int
     - Random seed for reproducibility. Use -1 (default) for non-deterministic sampling.
   * - ``merge_duplicate_terms``
     - bool
     - Fuse identical Pauli terms within consecutive commuting runs to reduce circuit depth. Default is True.
   * - ``commutation_type``
     - str
     - Commutation check used when merging: ``"qubit_wise"`` (stricter) or ``"general"`` (default).

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_unitary_builder.py
      :language: python
      :start-after: # start-cell-configure-qdrift
      :end-before: # end-cell-configure-qdrift

.. _partially-randomized-builder:

Partially randomized product formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. rubric:: Factory name: ``"partially_randomized"``

The partially randomized method is a hybrid that combines deterministic Trotter with randomized qDRIFT sampling :cite:`Guenther2025`.
It splits the Hamiltonian into a deterministic part and a random part:

.. math::

   H = H_D + H_R,

where :math:`H_D` collects the largest-weight terms (treated with a first- or second-order Trotter formula) and :math:`H_R` collects the remaining long tail of small terms (treated with qDRIFT-style sampling).
This is effective for chemistry Hamiltonians where a few terms dominate the weight while many small terms form a long tail: the dominant terms get the better :math:`\epsilon`-scaling of Trotter, while the tail gets the depth savings of randomization.

Each step applies :math:`H_D` and sandwiches a freshly sampled :math:`H_R` block inside it.
For a second-order (symmetric) Trotter formula the deterministic part is applied with half-angles in a palindromic sweep around the random block; the first-order variant applies :math:`H_D` at full angle followed by :math:`H_R`.

The split point is controlled by ``weight_threshold``: terms with :math:`|h_j|` at or above the threshold go into :math:`H_D`.
Setting it to ``-1.0`` selects the split automatically (top 10% of terms by weight, or a cost-optimal split when ``target_accuracy`` is set).

When ``target_accuracy`` :math:`\epsilon` is set, the builder becomes accuracy-aware.
The squared error budget is split in quadrature between the two parts,

.. math::

   \epsilon_D^2 + \epsilon_R^2 = \epsilon^2, \qquad \epsilon_D^2 = s\,\epsilon^2,

with the fraction :math:`s` given by ``accuracy_split``.
The evolution is divided into :math:`r` outer Trotter steps sized from :math:`\epsilon_D` (using ``trotter_error_bound``), and the per-step qDRIFT sample count is sized from :math:`\epsilon_R` (Campbell bound).
Each of the :math:`r` steps draws a *fresh* independent qDRIFT block, which is required for the randomized error to add up correctly across steps.
With ``target_accuracy = 0.0`` (the default) a single step is used with ``num_random_samples`` samples, preserving the legacy behavior.

The randomized part contributes :math:`O(\lambda_R^2 / \epsilon^2)` Pauli rotations, where :math:`\lambda_R = \sum_m |h_m|` is the 1-norm of :math:`H_R` :cite:`Guenther2025`.
Because the deterministic part removes the dominant terms from :math:`\lambda_R`, this is typically much smaller than the cost of applying qDRIFT to the full Hamiltonian.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``weight_threshold``
     - float
     - Terms with :math:`|h_j|` at or above this value are treated deterministically. Use -1.0 (default) for automatic selection.
   * - ``trotter_order``
     - int
     - Order of the Trotter formula for the deterministic part (1 or 2). Default is 2.
   * - ``num_random_samples``
     - int
     - Number of qDRIFT samples for :math:`H_R`. Acts as a per-step floor when ``target_accuracy`` is set. Default is 100.
   * - ``target_accuracy``
     - float
     - Target approximation error :math:`\epsilon`. When set to 0.0 (default), automatic parameterization is disabled.
   * - ``accuracy_split``
     - float
     - Fraction :math:`s` of the squared error budget given to the deterministic part (:math:`\epsilon_D^2 = s\,\epsilon^2`). Clamped to (0, 1). Default is 0.5.
   * - ``trotter_error_bound``
     - str
     - Error bound for sizing the outer Trotter step count: ``"commutator"`` (default, tighter) or ``"naive"``.
   * - ``seed``
     - int
     - Random seed for reproducibility. Use -1 (default) for non-deterministic sampling.
   * - ``tolerance``
     - float
     - Coefficient threshold below which Pauli terms are discarded. Default is 1e-12.
   * - ``merge_duplicate_terms``
     - bool
     - Fuse identical Pauli terms within consecutive commuting runs of the random block. Default is True.
   * - ``commutation_type``
     - str
     - Commutation check used when merging: ``"qubit_wise"`` (stricter) or ``"general"`` (default).

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_unitary_builder.py
      :language: python
      :start-after: # start-cell-configure-pr
      :end-before: # end-cell-configure-pr


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
