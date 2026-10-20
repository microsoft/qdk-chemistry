Hamiltonian factorization
=========================

:class:`~qdk_chemistry.algorithms.HamiltonianFactorization` transforms a
:class:`~qdk_chemistry.data.Hamiltonian` into a factorized representation.
It consumes the supplied numerical integrals rather than reconstructing a
Hamiltonian from its orbitals.

Overview
--------

Double factorization :cite:`vonBurg2021` diagonalizes each molecular-orbital
Cholesky vector into weighted orbital projectors. The result is stored in a
:class:`~qdk_chemistry.data.DFTHCHamiltonianContainer`, which also represents
the more general double-factorized tensor-hypercontraction (DFTHC) form
:cite:`Low2025`.

Using the HamiltonianFactorization
----------------------------------

Construct compatible Cholesky input with the
:doc:`HamiltonianConstructor <hamiltonian_constructor>`, then factorize it:

.. literalinclude:: ../../../_static/examples/python/release_notes_v2_3.py
   :language: python
   :start-after: # start-cell-factorization
   :end-before: # end-cell-factorization

The input must contain restricted
:class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` data with
:math:`N^2` rows and one auxiliary vector per column. The pair index is
:math:`pN+q`, where :math:`N` is the number of active spatial orbitals.
Non-Cholesky inputs, unrestricted integrals, incompatible dimensions,
non-finite entries, and nonsymmetric Cholesky vectors are rejected.

The supplied vectors are used directly, including redundant vectors.
The one-electron integrals, orbitals, core energy, inactive Fock matrix,
and Hamiltonian type are preserved.
First-stage accuracy remains controlled by the Cholesky producer's settings,
including ``cholesky_tolerance`` and ``eri_threshold``; the factorizer does
not perform another four-index decomposition.

Available implementations
-------------------------

Double factorization
~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"double_factorization"``

:class:`~qdk_chemistry.algorithms.DoubleFactorization` produces ordinary DF
data with :math:`B=N`, :math:`C=1`, and zero identity weights ``WB``.
The general DFTHC container stores normalized real basis rows
``U[R, B, N]``, weights ``W[R, B, C]``, and identity weights ``WB[R, C]``.
Its ordinary-DF constructor infers the dimensions and supplies zero ``WB``
when that argument is omitted.

The container stores conventional one-electron integrals.
``get_h1_prime()`` derives the adjusted one-body matrix needed for the
:ref:`sum-of-squares mapping <sum-of-squares-qubit-mapper>`; it does not
change the meaning of ``get_one_body_integrals()``.
``get_lambda()`` computes the factorization normalization

.. math::

   \Lambda_{\mathrm{factor}}
   = \sum_j |\epsilon_j(h'_1)|
     + \frac14 \sum_{r,c}
       \left(|WB^{rc}| + \sum_b |W_{rbc}|\right)^2.

This is a factorization diagnostic, not an energy or an exact operator norm.
The :ref:`SOSSA builder <sossa-builder>` computes its normalization from the
mapped generators. Without screening the two expressions agree; screening
can change both the represented operator and its normalization.

Related classes
---------------

- :class:`~qdk_chemistry.data.Hamiltonian`: Input and output Hamiltonian
- :class:`~qdk_chemistry.data.DFTHCHamiltonianContainer`: Factor storage and conventional-integral access
- :doc:`QubitMapper <qubit_mapper>`: Produces the qubit-operator representation

Further reading
---------------

- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Produces Cholesky input
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Constructs SOSSA block encodings
- :doc:`Settings <settings>` and :doc:`Factory Pattern <factory_pattern>`
- The executable :download:`release example <../../../_static/examples/python/release_notes_v2_3.py>`
