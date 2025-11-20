Reference-derived calculations
==============================

Reference-derived calculations in QDK/Chemistry are post-Hartree-Fock methods that build upon one or several reference wavefunctions
to account for electron correlation effects. These methods include Møller-Plesset perturbation theory (MP2) and Coupled Cluster (CC) theory.

Overview
--------

Reference-derived calculators inherit from the ``ReferenceDerivedCalculator`` base class and operate on an
:doc:`Ansatz <../data/ansatz>` object that contains a reference wavefunction and a :doc:`Hamiltonian <../data/hamiltonian>`.
These methods systematically improve upon the mean-field approximation by including electron-electron correlation effects.

Available calculators
---------------------

QDK/Chemistry currently provides the following reference-derived calculators:

- **MP2 Calculator** (``microsoft_mp2_calculator``): Second-order Møller-Plesset perturbation theory
- **PySCF Coupled Cluster** (``pyscf_coupled_cluster``): CCSD implementation via PySCF integration

MP2 Calculator
--------------

The MP2 (second-order Møller-Plesset perturbation theory) calculator computes the lowest-order correlation energy
correction beyond Hartree-Fock theory.

Features
~~~~~~~~

- **Restricted MP2** (RMP2): For closed-shell systems with equal numbers of alpha and beta electrons
- **Unrestricted MP2** (UMP2): For open-shell systems or when using unrestricted orbitals

The calculator automatically determines whether to use restricted or unrestricted MP2 based on:

1. Whether the input Hamiltonian is unrestricted (``hamiltonian.is_unrestricted()``)
2. Whether the number of alpha and beta electrons differs

Usage
~~~~~

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry/algorithms/microsoft/mp2.hpp>
      #include <qdk/chemistry/algorithms/factory.hpp>

      // Create MP2 calculator
      auto mp2_calculator = ReferenceDerivedCalculatorFactory::create("microsoft_mp2_calculator");

      // Run MP2 calculation on an ansatz
      auto [energy, wavefunction] = mp2_calculator->run(ansatz);

      // The energy is the total MP2 energy (reference + correlation)
      // To get correlation energy:
      double reference_energy = ansatz->calculate_energy();
      double correlation_energy = energy - reference_energy;

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import create
      from qdk_chemistry.data import Ansatz

      # Create MP2 calculator
      mp2_calculator = create("reference_derived_calculator", "microsoft_mp2_calculator")

      # Assuming you have a reference wavefunction and hamiltonian
      ansatz = Ansatz(hamiltonian, reference_wavefunction)

      # Run MP2 calculation
      total_energy, mp2_wavefunction = mp2_calculator.run(ansatz)

      # Calculate correlation energy
      reference_energy = ansatz.calculate_energy()
      correlation_energy = total_energy - reference_energy

Complete workflow example
~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example showing the an MP2 workflow starting with a HF calculation:

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import create
      from qdk_chemistry.data import Structure, Ansatz

      # Create a molecular structure
      structure = Structure.from_xyz("molecule.xyz")

      # Run SCF calculation
      scf_solver = create("scf_solver")
      scf_solver.settings().set("basis_set", "6-31g")
      scf_solver.settings().set("method", "hf")
      scf_energy, hf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)

      # Build Hamiltonian from HF orbitals
      orbitals = hf_wavefunction.get_orbitals()
      ham_constructor = create("hamiltonian_constructor")
      hamiltonian = ham_constructor.run(orbitals)

      # Create ansatz for MP2 calculation
      ansatz = Ansatz(hamiltonian, hf_wavefunction)

      # Run MP2 calculation
      mp2_calculator = create("reference_derived_calculator", "microsoft_mp2_calculator")
      mp2_energy, mp2_wavefunction = mp2_calculator.run(ansatz)

      # Print results
      print(f"SCF Energy: {scf_energy:.8f}")
      print(f"MP2 Total Energy: {mp2_energy:.8f}")
      print(f"MP2 Correlation Energy: {mp2_energy - scf_energy:.8f}")

Unrestricted MP2 example
~~~~~~~~~~~~~~~~~~~~~~~~

Here is a unrestricted example for an open-shell system.

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import create
      from qdk_chemistry.data import Structure, Ansatz

      # Create O2 molecule (open-shell triplet)
      o2 = Structure(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]]), ["O", "O"])

      # Run unrestricted HF calculation
      scf_solver = create("scf_solver")
      scf_solver.settings().set("basis_set", "6-31g")
      scf_solver.settings().set("method", "hf")
      uhf_energy, uhf_wavefunction = scf_solver.run(o2, charge=0, spin_multiplicity=3)

      # Build unrestricted Hamiltonian
      orbitals = uhf_wavefunction.get_orbitals()
      ham_constructor = create("hamiltonian_constructor")
      hamiltonian = ham_constructor.run(orbitals)

      # Hamiltonian is automatically unrestricted
      assert hamiltonian.is_unrestricted()

      # Create ansatz and run UMP2
      ansatz = Ansatz(hamiltonian, uhf_wavefunction)
      mp2_calculator = create("reference_derived_calculator", "microsoft_mp2_calculator")
      ump2_energy, ump2_wavefunction = mp2_calculator.run(ansatz)

      print(f"UHF Energy: {uhf_energy:.8f}")
      print(f"UMP2 Total Energy: {ump2_energy:.8f}")
      print(f"UMP2 Correlation Energy: {ump2_energy - uhf_energy:.8f}")

PySCF Coupled Cluster Calculator
---------------------------------

The PySCF Coupled Cluster calculator provides access to CCSD calculations via integration with the PySCF quantum chemistry package.

Settings
~~~~~~~~

The ``PyscfCoupledClusterCalculator`` can be configured using the following settings:

- ``conv_tol`` (float): Energy convergence threshold (default: 1e-7)
- ``conv_tol_normt`` (float): Convergence threshold for norm(t1,t2) (default: 1e-5)
- ``max_cycle`` (int): Maximum number of iterations (default: 50)
- ``diis_space`` (int): DIIS space size for acceleration (default: 6)
- ``diis_start_cycle`` (int): Iteration to start DIIS (default: 0)
- ``direct`` (bool): Use AO-direct CCSD (default: False)
- ``async_io`` (bool): Allow asynchronous I/O (default: True)
- ``incore_complete`` (bool): Avoid all I/O operations (default: True)
- ``store_amplitudes`` (bool): Store T amplitudes in wavefunction (default: False)

Usage
~~~~~

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import create
      from qdk_chemistry.data import Ansatz, Structure

      # Create molecule
      structure = Structure.from_xyz("molecule.xyz")

      # Run SCF calculation
      scf_solver = create("scf_solver")
      scf_solver.settings().set("basis_set", "cc-pvdz")
      scf_solver.settings().set("method", "hf")
      scf_energy, hf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)

      # Build Hamiltonian
      orbitals = hf_wavefunction.get_orbitals()
      ham_constructor = create("hamiltonian_constructor")
      hamiltonian = ham_constructor.run(orbitals)

      # Create ansatz
      ansatz = Ansatz(hamiltonian, hf_wavefunction)

      # Create and configure CCSD calculator
      ccsd_calculator = create("reference_derived_calculator", "pyscf_coupled_cluster")
      ccsd_calculator.settings().set("max_cycle", 100)
      ccsd_calculator.settings().set("conv_tol", 1e-8)

      # Run CCSD calculation
      ccsd_energy, ccsd_wavefunction = ccsd_calculator.run(ansatz)

      # Print results
      print(f"HF Energy: {scf_energy:.8f}")
      print(f"CCSD Total Energy: {ccsd_energy:.8f}")
      print(f"CCSD Correlation Energy: {ccsd_energy - scf_energy:.8f}")

Unrestricted Coupled Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PySCF Coupled Cluster calculator automatically handles unrestricted calculations when provided with an
unrestricted Hamiltonian:

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import create
      from qdk_chemistry.data import Ansatz

      # For an open-shell system with unrestricted orbitals
      scf_solver = create("scf_solver")
      scf_solver.settings().set("basis_set", "cc-pvdz")
      scf_solver.settings().set("method", "hf")
      # Multiplicity 3 for triplet state
      uhf_energy, uhf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=3)

      # Build unrestricted Hamiltonian
      orbitals = uhf_wavefunction.get_orbitals()
      ham_constructor = create("hamiltonian_constructor")
      hamiltonian = ham_constructor.run(orbitals)

      # Run unrestricted CCSD (UCCSD)
      ansatz = Ansatz(hamiltonian, uhf_wavefunction)
      ccsd_calculator = create("reference_derived_calculator", "pyscf_coupled_cluster")
      uccsd_energy, uccsd_wavefunction = ccsd_calculator.run(ansatz)

Related topics
--------------

- :doc:`SCF Solver <scf_solver>`: Generate reference wavefunctions
- :doc:`Hamiltonian Constructor <hamiltonian_constructor>`: Build Hamiltonians from orbitals
- :doc:`../data/hamiltonian`: Hamiltonian data structure, including unrestricted Hamiltonians
- :doc:`../data/orbitals`: Orbital data structure, including unrestricted orbitals
- :doc:`../data/ansatz`: Ansatz combining reference wavefunction and Hamiltonian
- :doc:`../data/wavefunction`: Wavefunction container for correlation methods

.. note::
   For additional examples and validation tests, refer to the test suite in ``python/tests/test_mp2.py``
   and ``python/tests/test_pyscf_plugin.py``.
