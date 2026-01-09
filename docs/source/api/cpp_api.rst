=======
C++ API
=======

This section provides complete documentation for the QDK/Chemistry C++ API.

QDK/Chemistry provides a comprehensive C++ library for quantum chemistry calculations.
The library is designed for high-performance computing with modern C++ features and efficient memory management.
The C++ implementation forms the core computational engine of QDK/Chemistry, offering maximum performance and fine-grained control for demanding quantum chemistry applications.

Data Classes
------------

The data classes in the C++ API provide efficient data structures for representing and manipulating quantum chemistry concepts such as molecular geometries, wavefunctions, and electronic states.
These classes are designed with performance in mind, employing optimized memory layouts and computational kernels for high-performance quantum chemistry calculations.

For detailed information, see :doc:`breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1data`.

.. toctree::
   :maxdepth: 2

   breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1data

Algorithms
----------

The algorithms namespace contains highly optimized implementations of various quantum chemistry methods.
This includes self-consistent field procedures, electron correlation methods, and specialized numerical techniques for solving quantum mechanical equations.
The C++ implementations take advantage of modern hardware features and parallelization to achieve maximum computational efficiency.

For detailed information, see :doc:`breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1algorithms`.

.. toctree::
   :maxdepth: 3

   breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1algorithms

Utilities
---------

The utils module provides various utility functions and helper routines that support the core functionality of QDK/Chemistry.
These utilities facilitate common tasks such as data manipulation, file I/O, and other ancillary operations.

Tensor Types
^^^^^^^^^^^^

QDK/Chemistry uses the `Kokkos mdspan <https://github.com/kokkos/mdspan>`_ reference implementation for multidimensional array support.
This provides efficient, type-safe access to tensor data with compile-time rank checking and flexible memory layouts.

The following tensor types are available in the ``qdk::chemistry`` namespace:

``tensor_span<T, Rank>``
   A non-owning multidimensional view over contiguous data, based on ``std::experimental::mdspan``.
   Uses column-major (Fortran) layout for compatibility with Eigen and standard quantum chemistry conventions.

``rank4_span<T>``
   A convenience alias for ``tensor_span<T, 4>``, commonly used for two-electron integrals.
   Access elements via ``span(i, j, k, l)`` syntax.

``tensor<T, Rank>``
   An owning multidimensional array based on ``std::experimental::mdarray`` with ``std::vector<T>`` storage.
   Provides implicit conversion to ``tensor_span`` for interoperability.

``rank4_tensor<T>``
   A convenience alias for ``tensor<T, 4>``, used for storing owned copies of two-electron integral tensors.

.. rubric:: Example Usage

.. code-block:: cpp

   #include <qdk/chemistry/utils/tensor.hpp>
   #include <qdk/chemistry/utils/tensor_span.hpp>

   // Create an owning 4D tensor with shape [n, n, n, n]
   auto eri = qdk::chemistry::make_rank4_tensor<double>(norb);

   // Access elements using multidimensional indexing
   eri(0, 1, 2, 3) = 1.5;

   // Get a non-owning view (implicit conversion)
   qdk::chemistry::rank4_span<double> view = eri;

   // Access via the Hamiltonian API
   auto [aaaa, aabb, bbbb] = hamiltonian->get_two_body_integrals();
   double integral = aaaa(i, j, k, l);  // Direct tensor indexing

For detailed information, see :doc:`breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1utils`.

.. toctree::
   :maxdepth: 3

   breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1utils

Constants
---------

The constants namespace provides access to fundamental physical constants, conversion factors, and reference values required for quantum chemistry calculations.
These constants are defined with high precision and are consistent with the latest CODATA recommendations, ensuring accurate and reproducible computational results across different hardware and software environments.

For detailed information, see :doc:`breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1constants`.

.. toctree::
   :maxdepth: 3

   breathe_api_autogen/namespace/namespaceqdk_1_1chemistry_1_1constants
