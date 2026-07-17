// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <string>

namespace py = pybind11;

namespace qdk::chemistry::python {

template <typename AlgorithmType>
void warn_if_deprecated_algorithm(const AlgorithmType& algorithm,
                                  Py_ssize_t stack_level = 2) {
  if (const auto message =
          qdk::chemistry::algorithms::detail::DeprecationAccess::message(
              algorithm)) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning, message->c_str(), stack_level) <
        0) {
      throw py::error_already_set();
    }
  }
}

/**
 * @brief Add _create_nested() to any pybind11 algorithm class binding.
 *
 * This delegates to the Python-side ``create_from_ref`` helper so that
 * both pure-Python and C++-backed algorithm classes share the same
 * implementation.
 *
 * Call this after creating the py::class_ for every algorithm type.
 */
template <typename PyClassType>
void bind_create_nested(PyClassType& cls) {
  cls.def(
      "_create_nested",
      [](py::object self, const std::string& key) -> py::object {
        py::module_ base = py::module_::import("qdk_chemistry.algorithms.base");
        py::object create_from_ref = base.attr("create_from_ref");
        py::object settings = self.attr("settings")();
        return create_from_ref(settings, key);
      },
      py::arg("setting_key"),
      R"(
Instantiate a nested algorithm from an AlgorithmRef stored in settings.

Args:
    setting_key (str): Settings key that holds an AlgorithmRef value.

Returns:
    Algorithm: A fully configured algorithm instance.
)");
}

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
  factory.def_static(
      "create",
      [](const std::string& name) -> std::unique_ptr<AlgorithmType> {
        auto instance = FactoryType::create(name);
        if (!instance) {
          throw std::runtime_error("Factory returned nullptr");
        }
        warn_if_deprecated_algorithm(*instance);
        return instance;
      },
      py::arg("name") = "", R"(
Create an algorithm instance by name.

If no name is provided or the name is empty, returns the default implementation.

Args:
    name (str | None): Name identifying which algorithm implementation to create.

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
  factory.def_static("default_algorithm_name",
                     &FactoryType::default_algorithm_name,
                     R"(
Return the default algorithm name for this factory.

Returns:
    str: The name of the default algorithm implementation

Examples:
    >>> default_name = Factory.default_algorithm_name()
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

/**
 * @brief Add on_remote() method to algorithm class for remote execution
 *
 * This template function adds the on_remote() method to any algorithm class,
 * enabling fluent API for remote execution:
 * create("algo").on_remote("ssh")
 *
 * @tparam PyClass The pybind11 class type (e.g., py::class_<ScfSolver, ...>)
 * @param py_class Reference to the pybind11 class object
 *
 * Example usage:
 * @code
 * py::class_<ScfSolver, ...> scf_solver(m, "ScfSolver");
 * // ... bind other methods ...
 * add_on_remote_method(scf_solver);
 * @endcode
 */
template <typename PyClass>
void add_on_remote_method(PyClass& py_class) {
  py_class.def(
      "on_remote",
      [](py::object self, py::object remote, py::kwargs config) {
        py::module_ remote_module = py::module_::import("qdk_chemistry.remote");
        py::object RemoteAlgorithmProxy =
            remote_module.attr("RemoteAlgorithmProxy");
        // Call RemoteAlgorithmProxy(self, remote, **config)
        py::tuple args = py::make_tuple(self, remote);
        return RemoteAlgorithmProxy(*args, **config);
      },
      py::arg("remote"),
      R"(
Configure this algorithm to execute on a remote system.

This method returns a proxy that intercepts the run() call and executes
the algorithm on a remote system instead of locally.

Args:
    remote: Either a backend name (str) like "ssh" or "local",
        or a pre-configured RemoteBackend instance.
    **config: Backend-specific configuration options.

Returns:
    RemoteAlgorithmProxy: A proxy that executes on the remote system.

Examples:
    >>> algo = create("scf_solver").on_remote("ssh")
    >>> result = algo.run(structure, charge=0, spin_multiplicity=1)

    >>> # With configuration
    >>> algo = create("scf_solver").on_remote("ssh", host="compute-server.example.com")

    >>> # With pre-configured backend
    >>> from qdk_chemistry.remote import create_remote
    >>> ssh_remote = create_remote("ssh", host="compute-server.example.com")
    >>> algo = create("scf_solver").on_remote(ssh_remote)

)");
}

/**
 * @brief Add a hash() method to an algorithm class binding.
 *
 * Adds hash(*args, **kwargs) -> str that computes a deterministic content
 * hash for a run() call with the same inputs.  Delegates to
 * qdk_chemistry.algorithms.hashing.run_content_hash.
 *
 * @tparam PyClass The pybind11 class type
 * @param py_class Reference to the pybind11 class object
 */
template <typename PyClass>
void add_hash_method(PyClass& py_class) {
  py_class.def(
      "hash",
      [](py::object self, py::args args, py::kwargs kwargs) -> std::string {
        py::module_ hashing_module =
            py::module_::import("qdk_chemistry.algorithms.hashing");
        py::object run_content_hash = hashing_module.attr("run_content_hash");
        py::object type_name = self.attr("type_name")();
        py::object name = self.attr("name")();
        py::object settings = self.attr("settings")();
        // Build positional args: (type_name, name, settings, *args)
        py::tuple base_args = py::make_tuple(type_name, name, settings);
        py::tuple all_args = py::tuple(base_args.size() + args.size());
        for (size_t i = 0; i < base_args.size(); ++i)
          all_args[i] = base_args[i];
        for (size_t i = 0; i < args.size(); ++i)
          all_args[base_args.size() + i] = args[i];
        py::object result = run_content_hash(*all_args, **kwargs);
        return result.cast<std::string>();
      },
      R"(
Compute a deterministic content hash for a run() call with these inputs.

Same signature as run() but returns a hex hash string instead of
executing the algorithm.  Identical inputs produce identical hashes.

Returns:
    str: 16-character hex content hash.

Examples:
    >>> scf = create("scf_solver", "pyscf")
    >>> h = scf.hash(structure, 0, 1, "sto-3g")
    >>> print(h)  # e.g. "a1b2c3d4e5f67890"

)");
}

}  // namespace qdk::chemistry::python
