// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

namespace py = pybind11;

namespace qdk::chemistry::python {

/**
 * @brief Generic template to bind AlgorithmFactory instances to Python
 *
 * This template function automatically creates Python bindings for any
 * AlgorithmFactory-derived class without requiring boilerplate code.
 * Creates a Python class with static methods that mirror the C++ factory API.
 *
 * @tparam FactoryType The factory class (e.g., ScfSolverFactory)
 * @tparam AlgorithmType The algorithm base class (e.g., ScfSolver)
 * @tparam TrampolineType The Python trampoline class (e.g., ScfSolverBase)
 * @param m The pybind11 module to bind to
 * @param class_name Name for the Python class (e.g., "ScfSolverFactory")
 *
 * This creates a class with static methods:
 * - create(name="")
 * - available()
 * - register_instance(func)
 * - unregister_instance(key)
 * - has(key)
 *
 * Example usage:
 * @code
 * bind_algorithm_factory<ScfSolverFactory, ScfSolver, ScfSolverBase>(
 *     m, "ScfSolverFactory");
 * @endcode
 */
template <typename FactoryType, typename AlgorithmType,
          typename TrampolineType = void>
void bind_algorithm_factory(py::module& m, const std::string& class_name) {
  // Create a non-instantiable class for the factory
  py::class_<FactoryType> factory(m, class_name.c_str(),
                                  R"(
Algorithm factory for creating and managing algorithm implementations.

This class provides static methods for creating, listing, and managing algorithm implementations through a registry pattern.

All methods are static and the class cannot be instantiated.

See Also:
    :meth:`create` : Create an algorithm instance by name
    :meth:`available` : List all registered algorithm names
    :meth:`register_instance` : Register a new algorithm implementation
    :meth:`unregister_instance` : Remove an algorithm from the registry
    :meth:`has` : Check if an algorithm name is registered

)");

  // Bind create static method
  factory.def_static("create", &FactoryType::create, py::arg("name") = "",
                     R"(
Create an algorithm instance by name.

If no name is provided or the name is empty, returns the default implementation.

Args:
    name (Optional[str]): Name identifying which algorithm implementation to create.

        If empty string (default), returns the default implementation.

Returns:
    Algorithm: New instance of the requested algorithm implementation

Raises:
    RuntimeError: If the name is not found in the registry

Examples:
    >>> algo = Factory.create("implementation_name")
    >>> default_algo = Factory.create()

)");

  // Bind available static method
  factory.def_static("available", &FactoryType::available,
                     R"(
Get a list of all registered algorithm names.

Returns:
    list[str]: List of all registered algorithm implementation names

Examples:
    >>> names = Factory.available()
    >>> print(f"Available implementations: {names}")

)");

  // Bind register_instance static method with conditional compilation
  if constexpr (!std::is_void_v<TrampolineType>) {
    // Has trampoline - support Python subclasses
    factory.def_static(
        "register_instance",
        [](py::function creator_func) {
          FactoryType::register_instance(
              [creator_func]() -> std::unique_ptr<AlgorithmType> {
                py::object instance = creator_func();
                return instance.cast<std::unique_ptr<TrampolineType>>();
              });
        },
        py::arg("func"),
        R"(
Register a new algorithm implementation.

The algorithm is registered under its primary name and all aliases as determined by the instance's name() and aliases() methods.

Args:
    func (callable): Function that returns an algorithm instance.

        The instance must implement the required algorithm interface.

Raises:
    RuntimeError: If name conflicts exist or type validation fails

Examples:
    >>> def create_custom():
    ...     return MyCustomAlgorithm()
    >>> Factory.register_instance(create_custom)

)");
  } else {
    // No trampoline - C++ only
    factory.def_static(
        "register_instance",
        [](py::function creator_func) {
          FactoryType::register_instance(
              [creator_func]() -> std::unique_ptr<AlgorithmType> {
                py::object instance = creator_func();
                return instance.cast<std::unique_ptr<AlgorithmType>>();
              });
        },
        py::arg("func"),
        R"(
Register a new algorithm implementation.

Args:
    func (callable): Function that returns an algorithm instance

Raises:
    RuntimeError: If registration fails due to name conflicts or type validation

)");
  }

  // Bind unregister_instance static method
  factory.def_static("unregister_instance", &FactoryType::unregister_instance,
                     py::arg("key"),
                     R"(
Unregister an algorithm implementation.

Args:
    key (str): Name or alias identifying the algorithm to remove

Returns:
    bool: True if successfully removed, False if not found

Examples:
    >>> Factory.unregister_instance("my_custom")
    True

)");
  // Bind has static method
  factory.def_static("algorithm_type_name", &FactoryType::algorithm_type_name,
                     R"(
Return the type name of the created algorithms.

Returns:
    str: The type name of the created algorithms

)");
  factory.def_static("clear", &FactoryType::clear,
                     R"(
Clear all registered algorithm implementations.
)");
  factory.def_static("has", &FactoryType::has, py::arg("key"),
                     R"(
Check if an algorithm name exists in the registry.

Args:
    key (str): Name or alias to check

Returns:
    bool: True if the name is registered, False otherwise

Examples:
    >>> if Factory.has("implementation_name"):
    ...     algo = Factory.create("implementation_name")

)");

  // Add __repr__ for the class (though it can't be instantiated)
  factory.def("__repr__", [class_name](const FactoryType&) {
    return "<" + class_name + " (static factory class)>";
  });
}

}  // namespace qdk::chemistry::python
