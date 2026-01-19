// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/data_class.hpp>

#include "path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class PyDataClass : public DataClass, public py::trampoline_self_life_support {
 public:
  std::string get_data_type_name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, DataClass, get_data_type_name);
  }

  std::string get_summary() const override {
    PYBIND11_OVERRIDE_PURE(std::string, DataClass, get_summary);
  }

  void to_file(const std::string &filename,
               const std::string &type) const override {
    PYBIND11_OVERRIDE_PURE(void, DataClass, to_file, filename, type);
  }

  nlohmann::json to_json() const override {
    // Override to accept a Python dict and convert to nlohmann::json
    py::gil_scoped_acquire acquire;
    py::function override_func = py::get_override(this, "to_json");
    if (override_func) {
      py::object result = override_func();
      // Convert Python dict to JSON string, then parse to nlohmann::json
      py::object json_module = py::module_::import("json");
      py::str json_string = json_module.attr("dumps")(result);
      std::string json_str = json_string.cast<std::string>();
      return nlohmann::json::parse(json_str);
    }
    pybind11::pybind11_fail(
        "Tried to call pure virtual function \"DataClass::to_json\"");
  }

  void to_json_file(const std::string &filename) const override {
    PYBIND11_OVERRIDE_PURE(void, DataClass, to_json_file, filename);
  }

  void to_hdf5(H5::Group &group) const override {
    // Convert C++ H5::Group to h5py object and call Python's to_hdf5
    py::gil_scoped_acquire acquire;

    // Check if Python class has overridden to_hdf5
    py::function override_func = py::get_override(this, "to_hdf5");
    if (!override_func) {
      pybind11::pybind11_fail(
          "Python class must override to_hdf5(group: h5py.Group) method");
    }

    // Get h5py module
    py::module_ h5py = py::module_::import("h5py");

    // Get the file ID from the group
    hid_t group_id = group.getId();

    // Create an h5py Group object from the HDF5 ID
    // Note: We use h5py.h5g.open to open the group by ID
    py::object h5g = py::module_::import("h5py.h5g");
    py::object h5py_group_low = h5g.attr("open")(py::int_(group_id));

    // Wrap it in a high-level h5py.Group
    py::object h5py_group = h5py.attr("Group")(h5py_group_low);

    // Call the Python override with the h5py Group
    override_func(h5py_group);
  }

  void to_hdf5_file(const std::string &filename) const override {
    PYBIND11_OVERRIDE_PURE(void, DataClass, to_hdf5_file, filename);
  }
};

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

// Serialization (to_*) wrappers
void base_class_to_json_file_wrapper(DataClass &self,
                                     const py::object &filename) {
  std::string path = qdk::chemistry::python::utils::to_string_path(filename);
  // Release GIL during potentially long-running file I/O operation
  py::gil_scoped_release release;
  self.to_json_file(path);
}

void base_class_to_hdf5_wrapper(DataClass &self, const py::object &h5py_group) {
  // Call the Python implementation of to_hdf5 with the h5py Group object
  // This allows Python-derived classes to receive h5py.Group directly
  py::object py_self = py::cast(&self, py::return_value_policy::reference);

  // Check if the Python class has implemented to_hdf5
  if (!py::hasattr(py_self, "to_hdf5")) {
    throw std::runtime_error(
        "Python class must implement to_hdf5(group) method to accept "
        "h5py.Group objects");
  }

  py::object method = py_self.attr("to_hdf5");

  // Check if it's actually overridden (not just inherited from DataClass)
  py::object base_class =
      py::module_::import("qdk_chemistry.data").attr("DataClass");
  if (py::isinstance(py_self, base_class)) {
    // Check if the method is overridden
    py::object self_method = py_self.attr("__class__").attr("to_hdf5");
    py::object base_method = base_class.attr("to_hdf5");
    if (self_method.is(base_method)) {
      throw std::runtime_error(
          "Python class must override to_hdf5(group) method to work with "
          "h5py.Group objects. "
          "The base DataClass implementation cannot be called with h5py "
          "objects.");
    }
  }

  method(h5py_group);
}

void base_class_to_hdf5_file_wrapper(DataClass &self,
                                     const py::object &filename) {
  std::string path = qdk::chemistry::python::utils::to_string_path(filename);
  // Release GIL during potentially long-running file I/O operation
  py::gil_scoped_release release;
  self.to_hdf5_file(path);
}

void base_class_to_file_wrapper(DataClass &self, const py::object &filename,
                                const std::string &format_type) {
  std::string path = qdk::chemistry::python::utils::to_string_path(filename);
  // Release GIL during potentially long-running file I/O operation
  py::gil_scoped_release release;
  self.to_file(path, format_type);
}

// Deserialization (from_*) wrappers
// Note: These are not real implementations but placeholder stubs
// The actual implementations must come from Python-side derived classes
py::object base_class_from_json_wrapper(const py::object &json_data) {
  throw std::runtime_error(
      "from_json() must be implemented by derived Python classes. "
      "The base DataClass does not provide a default implementation.");
}

py::object base_class_from_json_file_wrapper(const py::object &filename) {
  throw std::runtime_error(
      "from_json_file() must be implemented by derived Python classes. "
      "The base DataClass does not provide a default implementation.");
}

py::object base_class_from_hdf5_wrapper(const py::object &h5py_group) {
  throw std::runtime_error(
      "from_hdf5() must be implemented by derived Python classes. "
      "The base DataClass does not provide a default implementation.");
}

py::object base_class_from_hdf5_file_wrapper(const py::object &filename) {
  throw std::runtime_error(
      "from_hdf5_file() must be implemented by derived Python classes. "
      "The base DataClass does not provide a default implementation.");
}

py::object base_class_from_file_wrapper(const py::object &filename,
                                        const std::string &format_type) {
  throw std::runtime_error(
      "from_file() must be implemented by derived Python classes. "
      "The base DataClass does not provide a default implementation.");
}

}  // namespace

void bind_base_class(py::module &m) {
  py::class_<DataClass, PyDataClass, py::smart_holder>(m, "DataClass", R"(
Base class providing common interface for all data classes.

This abstract base class defines a consistent interface for serialization
and basic operations that all data classes must implement. It ensures
uniform behavior across different data types for:

- Summary generation
- JSON serialization/deserialization
- HDF5 serialization
- Generic file I/O with format detection

The base class enforces that derived classes implement proper serialization
methods, ensuring data persistence and interoperability throughout the package.

Note:
    This is an abstract base class and cannot be instantiated directly.
    Use the concrete derived classes like Structure, BasisSet, Settings, etc.

    All derived classes are guaranteed to provide:
        - `get_summary()` - Human-readable summary string
        - `to_json()` - JSON serialization
        - `to_hdf5()` - HDF5 serialization
        - `to_file()` - Generic file export with format detection
        - Corresponding deserialization methods (implementation varies by class)

)")

      .def(py::init<>(), R"(
Initialize a DataClass object.

Note:
    This is an abstract base class and cannot be instantiated directly.
    Use the concrete derived classes like Structure, BasisSet, Settings, etc.

)")

      // Core interface methods
      .def("get_data_type_name", &DataClass::get_data_type_name, R"(
Get the data type name for this class.

This is used for file naming conventions and serialization.

Returns:
    str: The data type name (e.g., "structure", "wavefunction")

)")

      .def("get_summary", &DataClass::get_summary, R"(
Get a human-readable summary of the object.

Returns:
    str: Summary string describing the object's contents and properties

Examples:
    >>> obj = SomeDataClass(...)
    >>> print(obj.get_summary())
    "SomeDataClass with X properties..."

)")

      // JSON serialization
      .def("to_json", &DataClass::to_json, R"(
Serialize object to JSON string.

Returns:
    str: JSON string representation of the object

Examples:
    >>> json_str = obj.to_json()
    >>> print(json_str)
    {"version": "1.0", "data": {...}}

)")

      .def("to_json_file", base_class_to_json_file_wrapper, R"(
Save object to JSON file.

Args:
    filename (str | pathlib.Path): Path to the output JSON file

Raises:
    RuntimeError: If file cannot be written or I/O error occurs

Examples:
    >>> obj.to_json_file("data.json")
    >>> from pathlib import Path
    >>> obj.to_json_file(Path("data.json"))

)",
           py::arg("filename"))

      // HDF5 serialization
      .def("to_hdf5", base_class_to_hdf5_wrapper, R"(
Save object to HDF5 group.

Args:
    group (h5py.Group | h5py.File): HDF5 group or file object to save data to

Raises:
    RuntimeError: If I/O error occurs

Notes:
    This method is primarily for Python-derived classes that work with h5py.
    It allows writing to an existing HDF5 group within a larger file structure.

Examples:
    >>> import h5py
    >>> with h5py.File("data.h5", "w") as f:
    ...     group = f.create_group("my_data")
    ...     obj.to_hdf5(group)

)",
           py::arg("group"))

      .def("to_hdf5_file", base_class_to_hdf5_file_wrapper, R"(
Save object to HDF5 file.

Args:
    filename (str | pathlib.Path): Path to the output HDF5 file

Raises:
    RuntimeError: If file cannot be written or I/O error occurs

Examples:
    >>> obj.to_hdf5_file("data.h5")
    >>> from pathlib import Path
    >>> obj.to_hdf5_file(Path("data.h5"))

)",
           py::arg("filename"))

      // Generic file I/O
      .def("to_file", base_class_to_file_wrapper, R"(
Save object to file with specified format.

Args:
    filename (str | pathlib.Path): Path to the output file
    format_type (str): Format type (e.g., "json", "hdf5").

        Available formats depend on the specific derived class.

Raises:
    ValueError: If format_type is not supported by this class
    RuntimeError: If file cannot be written or I/O error occurs

Examples:
    >>> obj.to_file("data.json", "json")
    >>> obj.to_file("data.h5", "hdf5")
    >>> from pathlib import Path
    >>> obj.to_file(Path("data.json"), "json")

)",
           py::arg("filename"), py::arg("format_type"))

      // Deserialization methods (classmethods)
      .def_static("from_json", base_class_from_json_wrapper, R"(
Create object from JSON data.

Args:
    json_data (dict): Dictionary containing the serialized data

Returns:
    DataClass: New instance of the derived class

Raises:
    RuntimeError: If deserialization fails or data is invalid

Notes:
    This is a classmethod that must be overridden by derived classes.

Examples:
    >>> json_data = {"version": "1.0", "data": {...}}
    >>> obj = SomeDataClass.from_json(json_data)

)",
                  py::arg("json_data"))

      .def_static("from_json_file", base_class_from_json_file_wrapper, R"(
Load object from JSON file.

Args:
    filename (str | pathlib.Path): Path to the input JSON file

Returns:
    DataClass: New instance of the derived class

Raises:
    RuntimeError: If file cannot be read or I/O error occurs

Note:
    This is a classmethod that must be overridden by derived classes.

Examples:
    >>> obj = SomeDataClass.from_json_file("data.json")
    >>> from pathlib import Path
    >>> obj = SomeDataClass.from_json_file(Path("data.json"))

)",
                  py::arg("filename"))

      .def_static("from_hdf5", base_class_from_hdf5_wrapper, R"(
Load object from HDF5 group.

Args:
    group (h5py.Group | h5py.File): HDF5 group or file object to load data from

Returns:
    DataClass: New instance of the derived class

Raises:
    RuntimeError: If I/O error occurs

Notes:
    This is a classmethod that must be overridden by derived classes.
    It allows reading from an existing HDF5 group within a larger file structure.

Examples:
    >>> import h5py
    >>> with h5py.File("data.h5", "r") as f:
    ...     group = f["my_data"]
    ...     obj = SomeDataClass.from_hdf5(group)

)",
                  py::arg("group"))

      .def_static("from_hdf5_file", base_class_from_hdf5_file_wrapper, R"(
Load object from HDF5 file.

Args:
    filename (str | pathlib.Path): Path to the input HDF5 file

Returns:
    DataClass: New instance of the derived class

Raises:
    RuntimeError: If file cannot be read or I/O error occurs

Notes:
    This is a classmethod that must be overridden by derived classes.

Examples:
    >>> obj = SomeDataClass.from_hdf5_file("data.h5")
    >>> from pathlib import Path
    >>> obj = SomeDataClass.from_hdf5_file(Path("data.h5"))

)",
                  py::arg("filename"))

      .def_static("from_file", base_class_from_file_wrapper, R"(
Load object from file with specified format.

Args:
    filename (str | pathlib.Path): Path to the input file
    format_type (str): Format type (e.g., "json", "hdf5").

        Available formats depend on the specific derived class.

Returns:
    DataClass: New instance of the derived class

Raises:
    ValueError: If format_type is not supported by this class
    RuntimeError: If file cannot be read or I/O error occurs

Notes:
    This is a classmethod that must be overridden by derived classes.

Examples:
    >>> obj = SomeDataClass.from_file("data.json", "json")
    >>> obj = SomeDataClass.from_file("data.h5", "hdf5")
    >>> from pathlib import Path
    >>> obj = SomeDataClass.from_file(Path("data.json"), "json")

)",
                  py::arg("filename"), py::arg("format_type"));
}
