Quickstart
==========

This document is intended to provide a brief introduction to the QDK/Chemistry library in a variety of common use cases.
The focus is primarily on high-level concepts and common coding patterns.
A more comprehensive documentation can be found at :doc:`comprehensive/index`.


The QDK/Chemistry Workflow
-----------------

.. graphviz:: /_static/diagrams/workflow.dot

.. todo::
   TODO (NAB):  finish documentation
    https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41390

Creating a ``Structure`` Object
-------------------------------

The ``Structure`` class represents the state of a molecular structure, i.e. the coordinates, charge, and spin multiplicity representing a system of interest.

.. todo::
   🔧 TODO: It seems that we actually *don't* store charge and multiplicity here...

.. todo::
   TODO (NAB):  finish documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41390

``Structure`` objects can be constructed manually, or via deserialization from a file.
QDK/Chemistry supports multiple serialization formats for ``Structure`` objects, including the standard `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_, as well as QDK/Chemistry-specific JSON and HDF5 serialization schemes (see documentation).
See below for language specific examples of creating and serializing ``Structure`` objects.

.. todo::
   **TODO** describe describe quirks for input units after Bohr PR

.. todo::
   TODO (NAB):  finish documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41390

.. todo::
   TODO (NAB):  check code after API changes
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41366

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;

      int main() {
          // Create the Structure manually
          Structure structure;
          structure.add_atom(Eigen::Vector3d(0.0, 0.0, 0.0), "H");
          structure.add_atom(Eigen::Vector3d(0.0, 0.0, 1.0), "H");

          // Read from an XYZ file
          auto structure_from_xyz = Structure::from_xyz_file("h2.structure.xyz"); // Required .structure.xyz suffix

          // Deserialize from JSON file
          auto structure_from_json_file = Structure::from_json_file("h2.structure.json"); // Required .structure.json suffix

          // Serialize to in-memory JSON object
          auto json_data = structure.to_json();

          // Deserialize from JSON object
          auto structure_from_json = Structure::from_json(json_data);

          // Serialize to JSON file
          structure.to_json_file("h2.new.structure.json"); // Required .structure.json suffix

          return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Structure

      if __name__ == '__main__':
          # Create the Structure manually
          structure = Structure()
          structure.add_atom([0.0, 0.0, 0.0], 'H')
          structure.add_atom([0.0, 0.0, 1.0], 'H')

          # Read from an XYZ file
          structure_from_xyz = Structure.from_xyz_file("h2.structure.xyz") # Required .structure.xyz suffix

          # Deserialize from JSON file
          structure_from_json_file = Structure.from_json_file("h2.structure.json") # Requires .structure.json suffix

          # Serialize to JSON object
          json_data = structure.to_json()

          # Deserialize from JSON object
          structure_from_json = Structure.from_json(json_data)

          # Serialize to JSON file
          structure.to_json_file("h2.new.structure.json") # Requires .structure.json suffix

Running an SCF Calculation
--------------------------

Once a ``Structure`` is created, an SCF calculation can be performed to produce an initial set of ``Orbitals`` as well as an SCF energy.
QDK/Chemistry performs SCF calculations via instantiations of the ``ScfSolver`` algorithm.
Instantiations of the ``ScfSolver`` algorithm (and all other ``Algorithm`` classes for that matter) are managed by a factory.
See the :doc:`comprehensive/advanced/factory_pattern` documentation for more information on how it is used in the QDK/Chemistry.
The basis for the SCF calculation can be set via a ``Settings`` instance. See below for language-specific examples.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;
      using namespace qdk::chemistry::algorithms;

      int main() {
          // Obtain a valid Structure object
          Structure structure;
          /* structure = ... */

          // Create the default ScfSolver instance
          auto scf_solver = ScfSolverFactory::create();

          // Print all settings available and their current values
          scf_solver->settings().to_json().dump(4);

          // Set the basis set
          scf_solver->settings().set("basis_set", "def2-tzvpp");

          // Run the SCF calculation
          auto [E_scf, scf_orbitals] = scf_solver->solve(structure);

          return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Structure
      from qdk.chemistry.algorithms import create_scf_solver

      if __name__ == '__main__':
          # Create a valid Structure object
          structure = Structure()
          # structure = ...

          # Create the default ScfSolver instance
          scf_solver = create_scf_solver()

          # Print all settings available and their current values
          print(scf_solver.settings())

          # Set the basis set
          scf_solver.settings().set("basis_set", "def2-tzvpp")

          # Run the SCF calculation
          E_scf, scf_orbitals = scf_solver.solve(structure)


Manipulate the ``Orbitals``
---------------------------

While SCF orbitals are useful for many applications, they are often not the optimal set for the selection of active spaces and other post-SCF calculations.
QDK/Chemistry generally characterizes the notion of orbital manipulation as "localization", even if the operation does not actually perform a spatial localization (e.g. natural orbitals).
As such, orbital manipulation is handled by instantiations of the ``Localizer`` algorithm.
See below for language specific examples of how to obtain MP2 natural orbitals.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;
      using namespace qdk::chemistry::algorithms;

      int main() {
        // Obtain a valid Orbitals instance
        Orbitals orbitals;
        /* orbitals = ... */

        // Create an instance of the MP2 natural orbital localizer
        auto localizer = LocalizerFactory::create("mp2_natural_orbitals");

        // Configure electron counts in settings
        localizer->settings().set("n_alpha_electrons", n_alpha);
        localizer->settings().set("n_beta_electrons", n_beta);

         // Create indices for orbitals to localize
         std::vector<size_t> loc_indices_a = {0, 1, 2, 3}; // Alpha orbital indices
         std::vector<size_t> loc_indices_b = {0, 1, 2, 3}; // Beta orbital indices


        // Localize the input orbitals
        // N.B. MP2 natural orbitals requires the input to be canonical
        auto localized_orbitals = localizer->run(orbitals, loc_indices_a, loc_indices_b);

        return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Orbitals
      from qdk.chemistry.algorithms import create_localizer

      if __name__ == "__main__":
         # Obtain a valid Orbitals instance
         orbitals = Orbitals()
         # orbitals = ...

         # Create an instance of the MP2 natural orbital localizer
         localizer = create("localizer", "mp2_natural_orbitals")

         # Configure electron counts in settings
         localizer.settings().set("n_alpha_electrons", n_alpha)
         localizer.settings().set("n_beta_electrons", n_beta)

         # Create indices for orbitals to localize
         loc_indices_a = [0, 1, 2, 3]  # Alpha orbital indices
         loc_indices_b = [0, 1, 2, 3]  # Beta orbital indices

         # Localize the input orbitals
         # N.B. MP2 natural orbitals requires the input to be canonical
         localized_orbitals = localizer.run(orbitals, loc_indices_a, loc_indices_b)


Select the Active Space
-----------------------

QDK/Chemistry offers many methods for the selection of active-spaces to accurately treat the quantum many-body problem while avoiding the prohibitive computational scaling of full configuration interaction.
See the :doc:`comprehensive/algorithms/active_space` documentation for a list of supported methods, along with their associated ``Settings``, which accompany the standard QDK/Chemistry distribution.
The following are language specific examples of how to select a so-called "valence" active space containing only those orbitals surrounding the Fermi-level.

.. todo::
   🔧 TODO: Expand this section to explain that active space selectors are also Orbital constructors (like localization, active space selectors),
   i.e., they take orbitals as input and return Orbitals (which carry an active space) as output.
   Add this relationship to the Localization section.

.. todo::
   TODO (NAB):  finish documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41390

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;
      using namespace qdk::chemistry::algorithms;

      int main() {
          // Obtain a valid Orbitals instance
          Orbitals orbitals;
          /* orbitals = ... */

          // Create an active space selector
          auto active_space_selector = ActiveSpaceSelectorFactory::create("valence");

          // Configure the active space selection parameters
          active_space_selector->settings().set("num_active_electrons", 10);
          active_space_selector->settings().set("num_active_orbitals", 10);

          // Select the active space indices
          auto active_space_indices = active_space_selector->run(orbitals);

          return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Orbitals
      from qdk.chemistry.algorithms import create_active_space_selector

      if __name__ == "__main__":
          # Obtain a valid Orbitals instance
          orbitals = Orbitals()
          # orbitals = ...

          # Create an active space selector
          active_space_selector = create_active_space_selector("valence")

          # Configure the active space selection parameters
          active_space_selector.settings().set("num_active_electrons", 10)
          active_space_selector.settings().set("num_active_orbitals", 10)

          # Select the active space indices
          active_space_indices = active_space_selector.run(orbitals)


Calculate the Hamiltonian
-------------------------

Once an active space has been selected, the electronic Hamiltonian can be computed within that active space.
QDK/Chemistry provides flexible Hamiltonian construction capabilities through the ``HamiltonianConstructor`` algorithm.
The Hamiltonian constructor can generate the one- and two-electron integrals needed for subsequent quantum many-body calculations.
See below for language specific examples.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;
      using namespace qdk::chemistry::algorithms;

      int main() {
          // Obtain a valid Orbitals instance
          Orbitals orbitals;
          /* orbitals = ... */

          // Create a Hamiltonian constructor
          auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

          // Set the active orbitals (if doing active space calculation)
          std::vector<int> active_orbitals = {4, 5, 6, 7}; // Example indices
          hamiltonian_constructor->settings().set("active_orbitals", active_orbitals);

          // Construct the Hamiltonian
          auto hamiltonian = hamiltonian_constructor->run(orbitals);

          // Access one- and two-electron integrals
          auto h1 = hamiltonian.get_one_body_integrals();
          auto h2 = hamiltonian.get_two_body_integrals();

          return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Orbitals
      from qdk.chemistry.algorithms import create_hamiltonian_constructor

      if __name__ == "__main__":
          # Obtain a valid Orbitals instance
          orbitals = Orbitals()
          # orbitals = ...

          # Create a Hamiltonian constructor
          hamiltonian_constructor = create_hamiltonian_constructor()

          # Set the active orbitals (if doing active space calculation)
          active_orbitals = [4, 5, 6, 7]  # Example indices
          hamiltonian_constructor.settings().set("active_orbitals", active_orbitals)

          # Construct the Hamiltonian
          hamiltonian = hamiltonian_constructor.run(orbitals)

          # Access one- and two-electron integrals
          h1 = hamiltonian.get_one_body_integrals()
          h2 = hamiltonian.get_two_body_integrals()

Compute a CAS Energy
--------------------

With the active space Hamiltonian constructed, quantum many-body calculations can be performed to obtain accurate electronic energies and wavefunctions.
QDK/Chemistry supports various Multi Configuration (MC) methods including full Configuration Interaction (CI) and selected CI approaches.
MC calculations are performed via instantiations of the ``MCCalculator`` algorithm.
See below for language specific examples of performing a full CI calculation.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;
      using namespace qdk::chemistry::algorithms;

      int main() {
          // Obtain a valid Hamiltonian
          Hamiltonian hamiltonian;
          /* hamiltonian = ... */

          // Create a multiconfigurational calculator (MACIS implementation)
          auto mc_calculator = MCCalculatorFactory::create();

          // Solve for the ground state
          auto [E_ci, ci_wavefunction] = mc_calculator->calculate(hamiltonian);

          // Print the CI energy
          std::cout << "CI Energy: " << E_ci << std::endl;

          return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Hamiltonian
      from qdk.chemistry.algorithms import create_mc_calculator

      if __name__ == "__main__":
          # Obtain a valid Hamiltonian
          hamiltonian = Hamiltonian()
          # hamiltonian = ...

          # Create a multiconfigurational calculator (MACIS implementation)
          mc_calculator = create_mc_calculator()

          # The MACIS implementation automatically determines electron numbers
          # from the orbital occupations in the Hamiltonian
          # No additional configuration needed for basic FCI

          # Solve for the ground state
          E_ci, ci_wavefunction = mc_calculator.calculate(hamiltonian)

          # Print the CI energy
          print(f"CI Energy: {E_ci}")

.. todo::
   TODO (NAB):  Need overview of sample scripts, sample problems, etc. -- including
   QPE and examples showing interoperability with other languages/tools.
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41390
