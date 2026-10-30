// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <string_view>

namespace qdk {
namespace chemistry {
namespace python {
namespace utils {

namespace nb = nanobind;

/**
 * @brief Helper function to bind a C++ getter method as both a method and a
 * property in Python.
 *
 * This function takes a C++ method starting with "get_" and binds it to Python
 * in two ways:
 * 1. As a regular method with the original name (e.g., "get_property")
 * 2. As a property without the "get_" prefix (e.g., "property")
 *
 * This makes the Python API more Pythonic, allowing both styles:
 * - obj.get_property()  # Method call
 * - obj.property        # Property access
 *
 * @tparam ClassBinding The nanobind class binding type
 * @tparam MethodPtr The type of the member function pointer
 * @tparam Options Additional nanobind binding options (rv_policy,
 * etc.)
 *
 * @param class_binding The nanobind class binding object
 * @param method_name The name of the method (should start with "get_")
 * @param method_ptr Pointer to the member function
 * @param docstring Documentation string for both method and property
 * @param options Additional nanobind options (e.g., rv_policy,
 * nb::arg())
 *
 * @example
 * ```cpp
 * nb::class_<MyClass> cls(m, "MyClass");
 * bind_getter_as_property(cls, "get_value", &MyClass::get_value,
 *                        "Get the value", nb::rv_policy::copy);
 * ```
 *
 * In Python:
 * ```python
 * obj = MyClass()
 * val1 = obj.get_value()  # Method call
 * val2 = obj.value        # Property access (same result)
 * ```
 */
template <typename ClassBinding, typename MethodPtr, typename... Options>
void bind_getter_as_property(ClassBinding& class_binding,
                             const char* method_name, MethodPtr method_ptr,
                             const char* docstring, const Options&... options) {
  // Bind the method with its original name
  class_binding.def(method_name, method_ptr, docstring, options...);

  // Extract property name by removing "get_" prefix
  std::string_view name_view(method_name);
  if (name_view.substr(0, 4) == "get_" && name_view.length() > 4) {
    std::string property_name(name_view.substr(4));

    // Bind as a property (read-only)
    class_binding.def_prop_ro(property_name.c_str(), method_ptr, docstring,
                              options...);
  }
}

/**
 * @brief Overload of bind_getter_as_property without extra arguments.
 *
 * @tparam ClassBinding The nanobind class binding type
 * @tparam MethodPtr The type of the member function pointer
 *
 * @param class_binding The nanobind class binding object
 * @param method_name The name of the method (should start with "get_")
 * @param method_ptr Pointer to the member function
 * @param docstring Documentation string for both method and property
 */
template <typename ClassBinding, typename MethodPtr>
void bind_getter_as_property(ClassBinding& class_binding,
                             const char* method_name, MethodPtr method_ptr,
                             const char* docstring) {
  // Bind the method with its original name
  class_binding.def(method_name, method_ptr, docstring);

  // Extract property name by removing "get_" prefix
  std::string_view name_view(method_name);
  if (name_view.substr(0, 4) == "get_" && name_view.length() > 4) {
    std::string property_name(name_view.substr(4));

    // Bind as a property (read-only)
    class_binding.def_prop_ro(property_name.c_str(), method_ptr, docstring);
  }
}

}  // namespace utils
}  // namespace python
}  // namespace chemistry
}  // namespace qdk
