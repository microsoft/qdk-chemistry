Symmetries
==========

The :class:`~qdk_chemistry.data.Symmetries` class in QDK/Chemistry represents the conserved quantum numbers of an electronic state, including particle number and spin.
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

Certain quantum algorithms can reduce resource requirements when the symmetries of the target quantum state are known in advance.
The :class:`~qdk_chemistry.data.Symmetries` class encapsulates these conserved quantities so that they can be passed to any algorithm that exploits them, such as the :ref:`symmetry-conserving Bravyi-Kitaev <encoding-scbk>` qubit mapping.

The class stores the number of alpha (spin-up) and beta (spin-down) electrons in the active space.
From these two values, several derived quantities are available as read-only properties.

Properties
~~~~~~~~~~

n_alpha
   Number of alpha (spin-up) electrons in the active space.

n_beta
   Number of beta (spin-down) electrons in the active space.

n_particles
   Total number of active electrons, :math:`n_\alpha + n_\beta`.

sz
   Spin projection quantum number, :math:`S_z = (n_\alpha - n_\beta) / 2`.

spin_multiplicity
   Spin multiplicity, :math:`2S + 1 = n_\alpha - n_\beta + 1`.

Creating a Symmetries object
----------------------------

A :class:`~qdk_chemistry.data.Symmetries` object can be created by specifying electron counts directly, or by using a factory method to extract them from an existing :class:`~qdk_chemistry.data.Wavefunction` or :class:`~qdk_chemistry.data.Ansatz`.

Direct construction
~~~~~~~~~~~~~~~~~~~

Provide the alpha and beta electron counts explicitly:

.. code-block:: python

   from qdk_chemistry.data import Symmetries

   # Closed-shell singlet with 4 active electrons
   sym = Symmetries(n_alpha=2, n_beta=2)
   print(sym.n_particles)       # 4
   print(sym.sz)                # 0.0
   print(sym.spin_multiplicity) # 1

   # Open-shell doublet
   sym = Symmetries(n_alpha=3, n_beta=2)
   print(sym.sz)                # 0.5
   print(sym.spin_multiplicity) # 2

From a wavefunction
~~~~~~~~~~~~~~~~~~~

If a :class:`~qdk_chemistry.data.Wavefunction` is available, for example from an :doc:`ScfSolver <../algorithms/scf_solver>` or :doc:`MCCalculator <../algorithms/mc_calculator>` calculation, the factory method :meth:`~qdk_chemistry.data.Symmetries.from_wavefunction` extracts the active-space electron counts automatically:

.. code-block:: python

   from qdk_chemistry.data import Symmetries

   symmetries = Symmetries.from_wavefunction(wavefunction)

From an ansatz
~~~~~~~~~~~~~~

When an :doc:`Ansatz <ansatz>` is available, for example after a multi-configuration calculation, the electron counts can be derived from the ansatz's wavefunction:

.. code-block:: python

   from qdk_chemistry.data import Symmetries

   symmetries = Symmetries.from_ansatz(ansatz)

.. note::

   ``Symmetries.from_ansatz(ansatz)`` is equivalent to
   ``Symmetries.from_wavefunction(ansatz.get_wavefunction())``.


Related classes
---------------

- :doc:`Wavefunction <wavefunction>`: Source of active-space electron counts via :meth:`~qdk_chemistry.data.Symmetries.from_wavefunction`
- :doc:`Ansatz <ansatz>`: Combined Hamiltonian + Wavefunction container, used with :meth:`~qdk_chemistry.data.Symmetries.from_ansatz`
- :doc:`Hamiltonian <hamiltonian>`: Input to qubit mapping algorithms
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Output of qubit mapping algorithms

Further reading
---------------

- :doc:`QubitMapper <../algorithms/qubit_mapper>`: Fermion-to-qubit mapping algorithms that consume ``Symmetries``
- :doc:`Design principles <../design/index>`: Data class design principles in QDK/Chemistry
