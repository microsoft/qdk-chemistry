Plugins
=======

Besides the core functionality provided by QDK/Chemistry, included and compiled into the library by default, such :term:`MACIS` :cite:`Williams-Young2023`, there is also a plugin system that allows for extending the functionality of QDK/Chemistry by adding custom algorithms, data structures, and interfaces.
Some plugins for popular quantum chemistry packages and quantum computing packages are provided with QDK/Chemistry, while others can be found in community repositories.

Core plugins
------------

The following lists included plugins that are available in QDK/Chemistry, developed and maintained by the QDK/Chemistry team.
These plugins are shipped and installed along with QDK/Chemistry and are enabled once the corresponding external packages are installed by the user.

- `Qiskit <https://www.ibm.com/quantum/qiskit>`_
- `PySCF <https://pyscf.org/>`_

Community plugins
-----------------

We welcome the addition of community-developed plugins to enhance the capabilities of QDK/Chemistry.

.. The following lists plugins that are available in addition to the above list, these plugins are developed and maintained by the community, for installation instructions please refer to the respective repositories and documentation.
.. (Note: This is likely an incomplete list.
.. If you are aware of other community plugins, please consider contributing to the documentation.)

.. - t.b.d.

Making custom plugins
---------------------

QDK/Chemistry's plugin system allows you to extend its functionality by creating custom algorithms, data structures, and interfaces.
The system is built around the factory pattern and a central registry that manages algorithm types and their implementations.

Creating a custom algorithm type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom plugin, you need to:

1. Define a new algorithm class that inherits from ``Algorithm``
2. Create a factory class that inherits from ``AlgorithmFactory``, if your algorithm is of a new type
3. Implement the algorithm's concrete implementation
4. Register both the factory and implementation with the registry system

Example: custom geometry optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example showing how to create a custom geometry optimizer and the corresponding algorithm type:

.. code-block:: python

    from qdk_chemistry.algorithms.base import AlgorithmFactory, Algorithm
    import qdk_chemistry.algorithms.registry as registry
    import qdk_chemistry.algorithms as algorithms
    from qdk_chemistry.data import Structure

    # Step 1: Define the custom algorithm type
    class GeometryOptimizer(Algorithm):
        def type_name(self) -> str:
            return "geometry_optimizer"

    # Step 2: Create a factory for this algorithm type
    class GeometryOptimizerFactory(AlgorithmFactory):
        def algorithm_type_name(self) -> str:
            return "geometry_optimizer"

        def default_algorithm_name(self) -> str:
            return "bfgs"  # Default algorithm

    # Step 3: Implement a concrete algorithm
    class BfgsOptimizer(GeometryOptimizer):
        def name(self) -> str:
            return "bfgs"

        def _run_impl(self, structure: Structure) -> Structure:
            # Implementation here
            ...
            return new_structure  # Return optimized structure

    # Step 4: Register the factory with the registry system.
    #         (Done in the initialization phase of your plugin
    #         when shipped as a package.)
    factory = GeometryOptimizerFactory()
    registry.register_factory(factory)

    # Step 5: Register algorithm implementation
    #         (Done in the initialization phase of your plugin
    #         when shipped as a package.)
    algorithms.register(lambda: BfgsOptimizer())


    # Now use via the top-level API
    optimizer = algorithms.create("geometry_optimizer", "bfgs")
    available_opts = algorithms.available("geometry_optimizer")
    print(available_opts)
    # Output: {'geometry_optimizer': ['bfgs']}

Key components
~~~~~~~~~~~~~~

**Algorithm Base Class**
    Your custom algorithm must inherit from ``Algorithm`` and implement the ``type_name()`` method to identify the algorithm type.

**Factory Class**
    The factory manages creation and registration of algorithm instances.
    It must implement ``algorithm_type_name()`` and ``default_algorithm_name()`` methods.

**Registry System**
    The registry (``qdk_chemistry.algorithms.registry``) maintains all available algorithm types and their implementations, enabling discovery and instantiation at runtime.

**Top-Level API**
    Once registered, your custom algorithms are accessible through the standard ``algorithms.create()`` and ``algorithms.available()`` functions, maintaining consistency with built-in algorithms.
