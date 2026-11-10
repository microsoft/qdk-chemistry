Stabilizer Renyi entropy
=======================

The ``stabilizer_renyi_entropy`` algorithm computes the second-order stabilizer
Renyi entropy (2-SRE) of a normalized, real, coefficient-based
:class:`~qdk_chemistry.data.Wavefunction`.
It measures non-stabilizer resource content, often called *magic*.
The default and currently only backend is ``replica``.

Definition and conventions
--------------------------

For an active occupation register of :math:`n` qubits,

.. math::

   |\psi\rangle = \sum_x c_x |x\rangle,\qquad
   \sum_x |c_x|^2 = 1,\qquad
   x = \sum_{q=0}^{n-1} x_q 2^q.

The entropy is reported in bits:

.. math::

   R_2 = 2^{-n}\sum_{P\in\{I,X,Y,Z\}^{\otimes n}}
       |\langle\psi|P|\psi\rangle|^4,\qquad
   M_2 = -\log_2 R_2.

This definition includes identity and all phase-free Pauli strings, not just
terms in a molecular Hamiltonian.
Stabilizer states have zero entropy.

The register is the **active space**, using Jordan-Wigner occupations without
tapering.
For spin-half molecular configurations, qubits in increasing index order contain
the alpha occupations followed by the beta occupations (blocked ordering).
The rightmost ket bit is qubit zero.
For example, ``Configuration.from_spin_half_string("20")`` maps to
:math:`x=5`, or :math:`|0101\rangle`, on four qubits.
The register width is obtained from configuration-space metadata, including
unoccupied upper modes, rather than from the largest occupied integer label.
Binary spinless configurations are also supported.

Using the algorithm
-------------------

Given a normalized real ``wavefunction`` from a coefficient-based calculation:

.. code-block:: python

   from qdk_chemistry import algorithms

   diagnostic = algorithms.create("stabilizer_renyi_entropy", "replica")
   result = diagnostic.run(wavefunction)
   print(result.entropy)  # M_2 in bits
   print(result.get_summary())

Direct construction with
:class:`~qdk_chemistry.algorithms.ReplicaStabilizerRenyiEntropy` is also supported.
Neither path constructs a quantum state-preparation circuit.

Truncation and normalization
----------------------------

Truncation is **off by default**.
Every nonzero coefficient is used, regardless of magnitude.
To enable it:

.. code-block:: python

   diagnostic = algorithms.create("stabilizer_renyi_entropy")
   diagnostic.settings().set("truncate", True)
   # Optional: the default amplitude cutoff is 1e-12.
   diagnostic.settings().set("coefficient_cutoff", 1e-10)
   result = diagnostic.run(wavefunction)

With truncation enabled, only coefficients satisfying
:math:`|c_x| > \epsilon` are retained; equality is discarded.
The retained state is always renormalized before the entropy is computed.
The cutoff must be finite and non-negative; an empty retained state is an error.
When truncation is disabled the cutoff setting is ignored.
Neither operation modifies the input wavefunction.

The **original input**, before any requested truncation, must have a finite
squared norm differing from one by at most :math:`10^{-10}`.
Larger normalization errors are rejected, not silently repaired.
Only bounded numerical normalization drift within this tolerance is corrected.
Nonfinite coefficients and genuinely complex amplitudes are rejected.
Complex storage containing exactly real values is accepted; no imaginary
component, however small, is silently discarded.
Coupled-cluster/MP2 amplitude containers are not coefficient expansions and are
not accepted.

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
* ``discarded_weight``: probability mass removed from the original normalized
  state, before renormalizing the retained coefficients.

Discarded mass is summed directly, rather than subtracting the retained norm
from one, so very small discarded weights are not lost through cancellation.
A small discarded probability is **not** a certified error bar for the entropy.
With truncation enabled the reported entropy belongs to the normalized
truncated state, not the original state.

The result supports standard content hashing, registry discovery, and versioned
JSON/HDF5 persistence:

.. code-block:: python

   from qdk_chemistry.data import StabilizerRenyiEntropyResult

   result.to_json_file("state.stabilizer_renyi_entropy_result.json")
   loaded = StabilizerRenyiEntropyResult.from_json_file(
       "state.stabilizer_renyi_entropy_result.json"
   )
   result.to_hdf5_file("state.stabilizer_renyi_entropy_result.h5")

Implementation and limitations
-------------------------------

The real replica identity is

.. math::

   R_2 = \sum_{x,y,z,w\in S}
       c_x c_y c_z c_w
       c_{x\oplus y\oplus z}
       c_{x\oplus y\oplus w}
       c_{x\oplus z\oplus w}
       c_{y\oplus z\oplus w},

where :math:`S` is the set of configurations with nonzero coefficients,
:math:`N_{\mathrm{conf}}=|S|` is their number, and absent coefficients are zero.
There is no additional factor of :math:`2^{-n}` in this coefficient identity.
See Eq. (8) and Appendix A of
`Magic in generalized Rokhsar-Kivelson wavefunctions
<https://arxiv.org/abs/2311.08463>`_.
Signed real coefficients are supported; non-negativity is not required for this
exact sum.

The implementation groups equal replica indices and counts permutation
multiplicities, using factors 24, 12, 14, and 1 for the distinct, one-pair,
two-pair/triple, and all-equal groups.
These are combinatorial replica symmetries, **not molecular symmetries**.
For more than 16 configurations, pair-XOR equivalence classes are constructed
once, using Python integer labels without a 64-bit restriction. For each fixed
second index, these classes generate a reusable table of triple-XOR partner
indices. Missing partners index a zero sentinel. The two remaining distinct
indices are evaluated in NumPy batches, rejecting absent partners before the
first-index loop. The one-pair contribution is also batched. No molecular
symmetries are assumed, and no state or lookup array of size :math:`2^n` is
allocated. Expansions with at most 16 configurations use a scalar kernel to
avoid array-setup overhead.

Worst-case runtime remains :math:`O(N_{\mathrm{conf}}^4)`; auxiliary storage is
:math:`O(N_{\mathrm{conf}}^2)`, independent of the full Hilbert-space dimension.
The quadratic tables trade memory for removal of per-quartet Python dictionary
lookups. This remains an exact method for modest configuration counts, not a
scalable estimator for arbitrary large expansions. Neither batching nor
sparse storage guarantees a speedup over Pauli methods for every state.

Batches use NumPy pairwise floating-point reductions; batch totals and the
final grouped contributions are combined with :func:`math.fsum`. The
two-pair/triple contribution is summed directly, without subtracting nearly
equal quantities. Last-bit differences from scalar evaluation are possible.
A fourth moment above one by at most :math:`10^{-12}` is clipped to one as
roundoff; nonpositive, nonfinite, or materially out-of-range values raise an
error.
Entropy exceeding :math:`n` by at most the same absolute tolerance is clipped
to :math:`n`; larger violations raise an error.
The initial backend does not support complex coefficients, alternative
fermion-to-qubit mappings, tapered states, other Renyi orders, or non-local
magic diagnostics.
