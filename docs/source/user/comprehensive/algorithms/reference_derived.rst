Reference-derived calculations
==============================

Reference-derived calculations in QDK/Chemistry are post-Hartree-Fock methods that build upon the wavefunction with one or several reference
determinants to account for electron correlation effects. These methods include Møller-Plesset perturbation theory (MP2) and Coupled Cluster (CC) theory.

Overview
--------

Reference-derived calculators inherit from the ``ReferenceDerivedCalculator`` base class and operate on an
:doc:`Ansatz <../data/ansatz>` object that contains the reference wavefunction and a :doc:`Hamiltonian <../data/hamiltonian>`.
These methods systematically improve upon the mean-field approximation by including electron-electron correlation effects.

Available calculators
---------------------

Currently, the following implementations are available:

- **MP2 Calculator** (``microsoft_mp2_calculator``): Second-order Møller-Plesset perturbation theory
- **PySCF Coupled Cluster** (``pyscf_coupled_cluster``): Coupled cluster implementations via PySCF integration

MP2 Calculator
--------------

The MP2 (second-order Møller-Plesset perturbation theory) calculator computes the lowest-order correlation energy
correction beyond Hartree-Fock theory.

Here's a complete example showing how to calculate MP2 corrections to a HF wavefunction:

.. tab:: C++ API

   .. code-block:: cpp
      // Create a simple structure
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}};
      std::vector<std::string> symbols = {"H", "H"};
      Structure structure(coords, symbols);

      // Run initial SCF
      auto scf_solver = ScfSolverFactory::create();
      auto [E_HF, wfn_HF] = scf_solver->run(structure, 0, 1);

      // Create a Hamiltonian constructor
      auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

      // Construct the Hamiltonian from orbitals
      auto hamiltonian = hamiltonian_constructor->run(wfn_HF->get_orbitals());

      // Create ansatz for MP2 calculation
      auto ansatz = std::make_shared<Ansatz>(*hamiltonian, *wfn_HF);

      // Run MP2
      auto mp2_calculator = ReferenceDerivedCalculatorFactory::create("microsoft_mp2_calculator");

      // Get energies
      auto [mp2_total_energy, final_wavefunction] = mp2_calculator->run(ansatz);

      // If desired, we can extract only the correlation energy
      double mp2_corr_energy = mp2_total_energy - E_HF;
      }
.. tab:: Python API

   .. code-block:: python

   .. literalinclude:: ../../../../examples/reference_derived.py
      :language: python
      :lines: 8-34


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
