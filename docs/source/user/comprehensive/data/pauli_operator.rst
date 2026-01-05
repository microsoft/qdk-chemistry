Pauli Operators
===============

The :class:`~qdk_chemistry.data.PauliOperator` class enables building quantum operator expressions using natural mathematical notation.
Arithmetic operators (``*``, ``+``, ``-``) combine Pauli operators into products and sums, making it easy to construct Hamiltonians and other quantum expressions.

Creating Operators
------------------

Use factory methods to create Pauli operators on specific qubits:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pauli_operator.cpp
      :language: cpp
      :start-after: // start-cell-creation
      :end-before: // end-cell-creation

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pauli_operator.py
      :language: python
      :start-after: # start-cell-creation
      :end-before: # end-cell-creation

Building Expressions
--------------------

Combine operators using arithmetic to build expressions:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pauli_operator.cpp
      :language: cpp
      :start-after: // start-cell-expressions
      :end-before: // end-cell-expressions

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pauli_operator.py
      :language: python
      :start-after: # start-cell-expressions
      :end-before: # end-cell-expressions

Simplifying Expressions
-----------------------

The ``simplify()`` method applies Pauli algebra rules and combines like terms.
The ``distribute()`` method expands products over sums.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pauli_operator.cpp
      :language: cpp
      :start-after: // start-cell-simplify
      :end-before: // end-cell-simplify

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pauli_operator.py
      :language: python
      :start-after: # start-cell-simplify
      :end-before: # end-cell-simplify

Canonical Representation
------------------------

Use ``to_canonical_string()`` to get a circuit-compatible string format, or ``to_canonical_terms()`` to get coefficient-string pairs:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pauli_operator.cpp
      :language: cpp
      :start-after: // start-cell-canonical
      :end-before: // end-cell-canonical

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pauli_operator.py
      :language: python
      :start-after: # start-cell-canonical
      :end-before: # end-cell-canonical

Further Reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/pauli_operator.cpp>`_ and `Python <../../../_static/examples/python/pauli_operator.py>`_ scripts.
- :doc:`QubitMapper <../algorithms/qubit_mapper>`: Maps molecular Hamiltonians to Pauli operator representations.
