Hamiltonian Unitary Builder
===========================

The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` algorithm in QDK/Chemistry constructs a unitary based on the Hamiltonian, such as time simulation unitary :math:`U(t) = e^{-iHt}` or block-encoded unitary :math:`U = \frac{H}{\|H\|}`.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.QubitHamiltonian` and produces a :class:`~qdk_chemistry.data.UnitaryRepresentation` as output.

Overview
--------

Building unitary from Hamiltonian — such as the Hamiltonian simulation unitary :math:`U(t) = e^{-iHt}` or block encoding unitary :math:`U = \frac{H}{\|H\|}` — is a central subroutine in many quantum algorithms.
The :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder` provides a unified interface for methods that construct this operator from a :class:`~qdk_chemistry.data.QubitHamiltonian`.

QDK/Chemistry currently provides two product-formula methods for this task: Trotter-Suzuki and the Zassenhaus expansion.
Both decompose :math:`e^{-iHt}` into a sequence of elementary Pauli rotations :math:`e^{-i\theta P}` that can be directly implemented as quantum gates, with controllable approximation error via the expansion order and number of time divisions :cite:`Suzuki1992,Casas2012`.
The resulting :class:`~qdk_chemistry.data.UnitaryRepresentation` objects wrap a ``PauliProductFormulaContainer`` — a list of exponentiated Pauli terms with a repetition count.


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

Zassenhaus expansion
~~~~~~~~~~~~~~~~~~~~~
.. rubric:: Factory name: ``"zassenhaus"``

The Zassenhaus expansion is the dual of the Baker-Campbell-Hausdorff formula :cite:`Wilcox1967`: it factorises a single exponential of a sum into an ordered product of exponentials in which low-order nested-commutator corrections appear as **explicit** factors, rather than being suppressed implicitly by the step count.
For two anti-Hermitian generators :math:`X = -i t A` and :math:`Y = -i t B`:

.. math::

   e^{X + Y} = e^{X}\, e^{Y}\, e^{C_2}\, e^{C_3}\, e^{C_4}\cdots

where the lowest corrections are the nested commutators :cite:`Casas2012`

.. math::

   C_2 &= -\tfrac{1}{2}[X, Y] \\
   C_3 &= \tfrac{1}{3}[Y, [X, Y]] + \tfrac{1}{6}[X, [X, Y]] \\
   C_4 &= -\tfrac{1}{24}[X, [X, [X, Y]]] - \tfrac{1}{8}[Y, [X, [X, Y]]] - \tfrac{1}{8}[Y, [Y, [X, Y]]].

Truncating after :math:`C_p` yields an operator-norm error of :math:`O(t^{p+1})` at a single time division.
Unlike Trotter-Suzuki, where the commutator structure is absorbed into the step count, here it is cancelled directly :cite:`Childs2021` — an advantage when the dominant commutators are sparse (lattice models with limited geometric coupling, or chemistry Hamiltonians under term partitions with small inter-group commutators).

**Multi-operator form.** A Hamiltonian generally has more than two non-commuting parts, so the builder partitions :math:`H` into internally-commuting groups :math:`G_0, \ldots, G_{K-1}` (each :math:`e^{-i t G_i}` is then exact) and uses the multi-operator Zassenhaus formula directly,

.. math::

   e^{-i t H} = e^{X_0} \cdots e^{X_{K-1}}\; e^{C_2}\, e^{C_3} \cdots e^{C_p}, \qquad X_i = -i t G_i,

with a single genuine :math:`K`-operator exponent :math:`C_n` per degree (the two-operator formulae above are the :math:`K = 2` instances).
The :math:`C_n` are **not hard-coded**: for the requested order they are generated once as exact rational coefficients of a free-Lie-algebra word series, then evaluated on the group operators through nested commutators (the Dynkin map) using the QDK-native ``commutator``.
The expansion therefore extends to any order; orders **2-6** are validated.
Each correction :math:`e^{C_n}` is in turn realised by a Trotter-Suzuki product of just-high-enough order, so its splitting error never dominates the :math:`O(t^{p+1})` truncation.

Groups are taken from the Hamiltonian's :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition` when it is present and internally commuting; otherwise the builder groups terms greedily with the :ref:`commuting term grouper <algorithms-term-grouper>`.
The output is a ``PauliProductFormulaContainer`` of the same shape as the :ref:`Trotter builder <trotter-builder>`, so it is consumable by downstream algorithms such as :doc:`PhaseEstimation <phase_estimation>` without changes.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``order``
     - int
     - Zassenhaus expansion order :math:`p` (:math:`\ge 2`; validated for 2-6); order 1 (no commutator corrections) falls back to the first-order Trotter builder. Default is 2.
   * - ``num_divisions``
     - int
     - Number of time divisions :math:`N` (:math:`\ge 1`). The Zassenhaus product approximates :math:`e^{-iHt/N}` and is repeated :math:`N` times. Default is 1.
   * - ``target_accuracy``
     - float
     - Target operator-norm accuracy. When positive, :math:`N` is raised automatically so the leading error :math:`\lVert C_{p+1}\rVert\, t^{p+1}/N^{p}` falls below it (the larger of this estimate and ``num_divisions`` is used). ``0.0`` (default) disables it.
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


Related classes
---------------

- :class:`~qdk_chemistry.data.UnitaryRepresentation`: Output data class wrapping the exponentiated Pauli terms or linear combinations of unitaries
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Input qubit Hamiltonian
- :doc:`PhaseEstimation <phase_estimation>`: One consumer of time-evolution unitaries

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/hamiltonian_unitary_builder.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Quantum phase estimation algorithms
- :doc:`QubitMapper <qubit_mapper>`: Map fermionic Hamiltonians to qubit operators
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
