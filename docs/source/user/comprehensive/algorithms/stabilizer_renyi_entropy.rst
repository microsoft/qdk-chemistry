Stabilizer Renyi entropy
========================

The ``stabilizer_renyi_entropy`` algorithm computes the second-order stabilizer
Renyi entropy (2-SRE) of a normalized, real, coefficient-based
:class:`~qdk_chemistry.data.Wavefunction`.
It measures non-stabilizerness, often called *magic*.
The default and currently only backend is ``replica``.

Overview
--------

Definition and conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

For an active occupation register of :math:`n` qubits,

.. math::

   |\psi\rangle = \sum_{x=0}^{2^n-1} c_x |x\rangle,\qquad
   \sum_{x=0}^{2^n-1} |c_x|^2 = 1,\qquad
   x = \sum_{q=0}^{n-1} x_q 2^q.

The entropy is reported in bits:

.. math::

   R_2 = 2^{-n}\sum_{P\in\{I,X,Y,Z\}^{\otimes n}}
       |\langle\psi|P|\psi\rangle|^4,\qquad
   M_2 = -\log_2 R_2.

The sum runs over all phase-free :math:`n`-qubit Pauli strings (including identity).
Stabilizer states have zero entropy.
See the original definition in
Stabilizer Renyi Entropy, Physical Review Letters 128, 050402 (2022)
:cite:`Leone2022`.

The register maps the **active space**, using Jordan-Wigner occupations (without
tapering, i.e. without symmetry-based qubit reduction).
For molecular configurations, qubits in increasing index order contain
the alpha occupations followed by the beta occupations (blocked ordering).
The rightmost ket bit is qubit zero.
For example, ``Configuration.from_spin_half_string("20")`` maps to
:math:`|0101\rangle`, or :math:`x=5`, on four qubits.
For molecular configurations, the register width :math:`n` is the number of
active spin-orbitals, twice the number of active spatial orbitals.
These occupation-ordering conventions follow those of QDK Chemistry.

Orbital basis and mapping
~~~~~~~~~~~~~~~~~~~~~~~~~

The reported magic characterizes the :math:`n`-qubit state obtained from the supplied
fermionic wavefunction in the chosen orbital basis and fermion-to-qubit mapping.
The reported value depends on the chosen orbital basis and on the mapping or
encoding of the electronic state. In general, fermionic orbital rotations and
changes of encoding can change its value.


Using the algorithm
-------------------

.. note::
   This algorithm is currently available only in the Python API.

Given a normalized real ``wavefunction`` from a coefficient-based calculation:

.. code-block:: python

   from qdk_chemistry import algorithms

   sre_calculator = algorithms.create("stabilizer_renyi_entropy", "replica")
   result = sre_calculator.run(wavefunction)
   print(result.entropy)  # M_2 in bits
   print(result.get_summary())

Direct construction with
:class:`~qdk_chemistry.algorithms.ReplicaStabilizerRenyiEntropy` is also supported.

Complete examples
-----------------

The following sections form one standalone example using public QDK APIs.
The synthetic qubit states are mathematical examples, not fixed-electron-number
molecular wavefunctions. Binary configurations and :class:`~qdk_chemistry.data.ModelOrbitals`
provide their occupation registers.

.. literalinclude:: ../../../_static/examples/python/stabilizer_renyi_entropy.py
   :language: python
   :start-after: # start-cell-setup
   :end-before: # end-cell-setup

Known states
~~~~~~~~~~~~

By definition, stabilizer states have zero magic.
The :math:`|+\rangle` state is an example one-qubit stabilizer state, and all four Bell states are
entangled two-qubit stabilizer states.
On the contrary, the real H-state

.. math::

   |H\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle,

has non-zero magic

.. math::

   M_2(|H\rangle) = \log_2(4/3) \approx 0.415.

The product of two H-states has twice this entropy. Applying CNOT creates
an entangled magic state with the same entropy, since CNOT is Clifford.
The example checks every result against its analytical expectation.

.. literalinclude:: ../../../_static/examples/python/stabilizer_renyi_entropy.py
   :language: python
   :start-after: # start-cell-known-states
   :end-before: # end-cell-known-states

Small molecular calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This H2 example uses a bond length of :math:`0.740848\,\mathrm{\AA}`, the
STO-3G basis, restricted Hartree-Fock orbitals and a CAS(2,2) ground state.
The occupation register contains four qubits. Its magic is specific to this
state and representation.

.. literalinclude:: ../../../_static/examples/python/stabilizer_renyi_entropy.py
   :language: python
   :start-after: # start-cell-h2
   :end-before: # end-cell-h2

Truncation and normalization
----------------------------

Truncation is **off by default**.
Every nonzero coefficient is used, regardless of magnitude.
To enable it:

.. code-block:: python

   sre_calculator = algorithms.create("stabilizer_renyi_entropy")
   sre_calculator.settings().set("truncate", True)
   # Optional: the default amplitude cutoff is 1e-12.
   sre_calculator.settings().set("coefficient_cutoff", 1e-10)
   result = sre_calculator.run(wavefunction)

With truncation enabled, only coefficients satisfying
:math:`|c_x| > \epsilon` are retained.
The retained state is always renormalized before the stabilizer Renyi
entropy is computed.
The cutoff must be finite and non-negative; an empty retained state raises an error.
When truncation is disabled the cutoff setting is ignored.

The **original input**, before any requested truncation, must have a finite
squared norm differing from one by at most :math:`10^{-10}`.
Larger deviations raise an error.
Nonfinite coefficients and genuinely complex amplitudes are rejected.
However, complex storage containing exactly real values is accepted.
Wavefunctions must provide an explicit coefficient expansion using a
:class:`~qdk_chemistry.data.StateVectorContainer`.
Amplitude containers (e.g., coupled-cluster or MP2) and MPS containers
are not accepted.

Available settings
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``truncate``
     - bool
     - ``False``
     - Enable coefficient truncation; the retained state is renormalized.
   * - ``coefficient_cutoff``
     - double
     - ``1e-12``
     - Strict amplitude threshold; used only when truncation is enabled.

Result and persistence
----------------------

:class:`~qdk_chemistry.data.StabilizerRenyiEntropyResult` is immutable and records:

* ``entropy``: :math:`M_2` in bits;
* ``alpha=2`` and ``log_base=2``;
* ``num_qubits``, ``encoding="jordan-wigner"``, and
  ``fermion_mode_order="blocked"``;
* ``num_configurations`` and ``num_retained_configurations``: numbers of nonzero
  configurations before and after truncation;
* ``truncation_enabled`` and ``coefficient_cutoff`` (``None`` when disabled);
* ``discarded_weight``: probability weight removed from the original normalized
  state, before renormalizing the retained coefficients.

The result supports standard content hashing, registry discovery, and versioned
JSON/HDF5 persistence:

.. code-block:: python

   from qdk_chemistry.data import StabilizerRenyiEntropyResult

   result.to_json_file("state.stabilizer_renyi_entropy_result.json")
   loaded = StabilizerRenyiEntropyResult.from_json_file(
       "state.stabilizer_renyi_entropy_result.json"
   )
   result.to_hdf5_file("state.stabilizer_renyi_entropy_result.h5")

Available implementations
-------------------------

Replica
~~~~~~~

.. rubric:: Factory name: ``"replica"`` (default)

Implementation and limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 2-SRE is implemented following the replica method introduced in Magic in generalized Rokhsar-Kivelson wavefunctions
:cite:`Tarabunga2024`. For real states, the replica identity is

.. math::

   R_2 = \sum_{x,y,z,w\in S}
       c_x c_y c_z c_w
       c_{x\oplus y\oplus z}
       c_{x\oplus y\oplus w}
       c_{x\oplus z\oplus w}
       c_{y\oplus z\oplus w},

where :math:`S` is the set of :math:`N_{\mathrm{conf}}=|S|` retained configurations with nonzero coefficients
after any requested cutoff, and :math:`c_x` are the normalized coefficients of
the retained state.

Worst-case runtime is :math:`O(N_{\mathrm{conf}}^4)`; auxiliary storage is
:math:`O(N_{\mathrm{conf}}^2)`, independent of the full Hilbert-space dimension.
The implementation is serial at this stage.

The method is thus best suited to sparse states with a
tractable number of retained configurations; illustrative timings are provided
below. For dense states on small qubit registers, Pauli-based calculations may
outperform the present replica-based implementation.

.. list-table:: Illustrative timings for the serial replica implementation
   :header-rows: 1
   :widths: 50 50

   * - Retained configurations
     - Replica-kernel wall time
   * - 122
     - About 0.075 seconds
   * - 345
     - About 0.90 seconds
   * - 1,051
     - About 25.1 seconds
   * - 2,092
     - About 4.6 minutes
   * - 3,421
     - About 30.8 minutes

These are illustrative replica-kernel timings (median of three calls per
cutoff), obtained with the same 24-qubit :math:`\mathrm{N}_2` CAS(10,12)
wavefunction evaluated at different amplitude cutoffs.
Wavefunction loading and the remaining work within the algorithm call
are excluded. The latter adds a few seconds per call in this benchmark.


The initial backend does not support complex coefficients, alternative
fermion-to-qubit mappings, tapered states, other Renyi orders, or non-local
magic diagnostics.

Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`
- :class:`~qdk_chemistry.data.StabilizerRenyiEntropyResult`

Further reading
---------------

- Stabilizer Renyi Entropy :cite:`Leone2022`
- Replica technique for the 2-SRE :cite:`Tarabunga2024`
- :doc:`Factory Pattern <factory_pattern>`
- :doc:`Settings <settings>`
- :doc:`Multi-configuration calculations <mc_calculator>`
- The complete :download:`Python example <../../../_static/examples/python/stabilizer_renyi_entropy.py>`
