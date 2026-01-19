// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::bind_getter_as_property;

// Wrapper functions for file I/O methods that accept both strings and pathlib
// Path objects

// Wrapper for to_file() that converts Python path objects to strings
void stability_result_to_file_wrapper(StabilityResult &self,
                                      const py::object &filename,
                                      const std::string &format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

// Wrapper for from_file() that converts Python path objects to strings
std::shared_ptr<StabilityResult> stability_result_from_file_wrapper(
    const py::object &filename, const std::string &format_type) {
  return StabilityResult::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

// Wrapper for to_hdf5_file() that converts Python path objects to strings
void stability_result_to_hdf5_file_wrapper(StabilityResult &self,
                                           const py::object &filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

// Wrapper for from_hdf5_file() that converts Python path objects to strings
std::shared_ptr<StabilityResult> stability_result_from_hdf5_file_wrapper(
    const py::object &filename) {
  return StabilityResult::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

// Wrapper for to_json_file() that converts Python path objects to strings
void stability_result_to_json_file_wrapper(StabilityResult &self,
                                           const py::object &filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

// Wrapper for from_json_file() that converts Python path objects to strings
std::shared_ptr<StabilityResult> stability_result_from_json_file_wrapper(
    const py::object &filename) {
  return StabilityResult::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void bind_stability_result(py::module &m) {
  py::class_<StabilityResult, DataClass, py::smart_holder> stability_result(
      m, "StabilityResult", R"(
Result structure for wavefunction stability analysis.

The StabilityResult class encapsulates the results of a stability check performed on a wavefunction.
It contains information about whether the wavefunction is stable, along with the eigenvalues and eigenvectors of the stability matrix for both internal and external stability.

This class provides:

- Internal and external stability status
- Overall stability status (stable only if both internal and external are stable)
- Internal and external eigenvalues of the stability matrices
- Internal and external eigenvectors of the stability matrices
- Convenient access methods for stability analysis results

Examples:
    Create a stability result:

        >>> import qdk_chemistry.data as data
        >>> import numpy as np
        >>> internal_eigenvals = np.array([1.0, 2.0, 3.0])
        >>> external_eigenvals = np.array([0.5, 1.5])
        >>> internal_eigenvecs = np.eye(3)
        >>> external_eigenvecs = np.eye(2)
        >>> result = data.StabilityResult(True, True, internal_eigenvals, internal_eigenvecs,
        ...                               external_eigenvals, external_eigenvecs)

    Check if stable:

        >>> print(result.is_stable())
        True

    Get internal eigenvalues:

        >>> print(result.get_internal_eigenvalues())
        [1. 2. 3.]

    Get smallest eigenvalue overall:

        >>> print(result.get_smallest_eigenvalue())
        0.5
)");

  // Constructors
  stability_result.def(py::init<>(), "Default constructor");

  stability_result.def(
      py::init<bool, bool, const Eigen::VectorXd &, const Eigen::MatrixXd &,
               const Eigen::VectorXd &, const Eigen::MatrixXd &>(),
      R"(
Construct a StabilityResult with specific values.

Args:
    internal_stable (bool): True if internal stability is satisfied
    external_stable (bool): True if external stability is satisfied
    internal_eigenvalues (numpy.ndarray): Eigenvalues of the internal stability matrix
    internal_eigenvectors (numpy.ndarray): Eigenvectors of the internal stability matrix
    external_eigenvalues (numpy.ndarray): Eigenvalues of the external stability matrix
    external_eigenvectors (numpy.ndarray): Eigenvectors of the external stability matrix

Examples:
    >>> import numpy as np
    >>> from qdk_chemistry.data import StabilityResult
    >>> internal_eigenvals = np.array([1.0, 2.0, 3.0])
    >>> external_eigenvals = np.array([0.5, 1.5])
    >>> internal_eigenvecs = np.eye(3)
    >>> external_eigenvecs = np.eye(2)
    >>> result = StabilityResult(True, True, internal_eigenvals, internal_eigenvecs,
    ...                          external_eigenvals, external_eigenvecs)
)",
      py::arg("internal_stable"), py::arg("external_stable"),
      py::arg("internal_eigenvalues"), py::arg("internal_eigenvectors"),
      py::arg("external_eigenvalues"), py::arg("external_eigenvectors"));

  // Data access methods
  stability_result.def("is_stable", &StabilityResult::is_stable,
                       R"(
Check if the wavefunction is stable overall.

Returns:
    bool: True if both internal and external stability are satisfied

Examples:
    >>> if result.is_stable():
    ...     print("Wavefunction is stable")
)");

  stability_result.def("is_internal_stable",
                       &StabilityResult::is_internal_stable,
                       R"(
Check if internal stability is satisfied.

Returns:
    bool: True if internal stability is satisfied
)");

  stability_result.def("is_external_stable",
                       &StabilityResult::is_external_stable,
                       R"(
Check if external stability is satisfied.

Returns:
    bool: True if external stability is satisfied
)");

  bind_getter_as_property(stability_result, "get_internal_eigenvalues",
                          &StabilityResult::get_internal_eigenvalues,
                          R"(
Get the internal eigenvalues of the stability matrix.

Returns:
    numpy.ndarray: The internal eigenvalues of the stability matrix

Examples:
    >>> internal_eigenvals = result.get_internal_eigenvalues()
    >>> print(f"Internal eigenvalues: {internal_eigenvals}")
)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(stability_result, "get_internal_eigenvectors",
                          &StabilityResult::get_internal_eigenvectors,
                          R"(
Get the internal eigenvectors of the stability matrix.

Returns:
    numpy.ndarray: The internal eigenvectors of the stability matrix

Examples:
    >>> internal_eigenvecs = result.get_internal_eigenvectors()
    >>> print(f"Internal eigenvectors shape: {internal_eigenvecs.shape}")
)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(stability_result, "get_external_eigenvalues",
                          &StabilityResult::get_external_eigenvalues,
                          R"(
Get the external eigenvalues of the stability matrix.

Returns:
    numpy.ndarray: The external eigenvalues of the stability matrix

Examples:
    >>> external_eigenvals = result.get_external_eigenvalues()
    >>> print(f"External eigenvalues: {external_eigenvals}")
)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(stability_result, "get_external_eigenvectors",
                          &StabilityResult::get_external_eigenvectors,
                          R"(
Get the external eigenvectors of the stability matrix.

Returns:
    numpy.ndarray: The external eigenvectors of the stability matrix

Examples:
    >>> external_eigenvecs = result.get_external_eigenvectors()
    >>> print(f"External eigenvectors shape: {external_eigenvecs.shape}")
)",
                          py::return_value_policy::reference_internal);

  // Data modification methods
  stability_result.def("set_internal_stable",
                       &StabilityResult::set_internal_stable,
                       R"(
Set the internal stability status.

Args:
    internal_stable (bool): True if internal stability is satisfied

Examples:
    >>> result.set_internal_stable(True)
)",
                       py::arg("internal_stable"));

  stability_result.def("set_external_stable",
                       &StabilityResult::set_external_stable,
                       R"(
Set the external stability status.

Args:
    external_stable (bool): True if external stability is satisfied

Examples:
    >>> result.set_external_stable(False)
)",
                       py::arg("external_stable"));

  stability_result.def("set_internal_eigenvalues",
                       &StabilityResult::set_internal_eigenvalues,
                       R"(
Set the internal eigenvalues.

Args:
    internal_eigenvalues (numpy.ndarray): The internal eigenvalues of the stability matrix

Examples:
    >>> import numpy as np
    >>> new_internal_eigenvals = np.array([0.5, 1.5, 2.5])
    >>> result.set_internal_eigenvalues(new_internal_eigenvals)
)",
                       py::arg("internal_eigenvalues"));

  stability_result.def("set_internal_eigenvectors",
                       &StabilityResult::set_internal_eigenvectors,
                       R"(
Set the internal eigenvectors.

Args:
    internal_eigenvectors (numpy.ndarray): The internal eigenvectors of the stability matrix

Examples:
    >>> import numpy as np
    >>> new_internal_eigenvecs = np.random.rand(3, 3)
    >>> result.set_internal_eigenvectors(new_internal_eigenvecs)
)",
                       py::arg("internal_eigenvectors"));

  stability_result.def("set_external_eigenvalues",
                       &StabilityResult::set_external_eigenvalues,
                       R"(
Set the external eigenvalues.

Args:
    external_eigenvalues (numpy.ndarray): The external eigenvalues of the stability matrix

Examples:
    >>> import numpy as np
    >>> new_external_eigenvals = np.array([0.2, 1.8])
    >>> result.set_external_eigenvalues(new_external_eigenvals)
)",
                       py::arg("external_eigenvalues"));

  stability_result.def("set_external_eigenvectors",
                       &StabilityResult::set_external_eigenvectors,
                       R"(
Set the external eigenvectors.

Args:
    external_eigenvectors (numpy.ndarray): The external eigenvectors of the stability matrix

Examples:
    >>> import numpy as np
    >>> new_external_eigenvecs = np.random.rand(2, 2)
    >>> result.set_external_eigenvectors(new_external_eigenvecs)
)",
                       py::arg("external_eigenvectors"));

  // Utility methods
  stability_result.def("internal_size", &StabilityResult::internal_size,
                       R"(
Get the number of internal eigenvalues.

Returns:
    int: Number of internal eigenvalues in the stability matrix
)");

  stability_result.def("external_size", &StabilityResult::external_size,
                       R"(
Get the number of external eigenvalues.

Returns:
    int: Number of external eigenvalues in the stability matrix
)");

  stability_result.def("get_smallest_internal_eigenvalue",
                       &StabilityResult::get_smallest_internal_eigenvalue,
                       R"(
Get the smallest internal eigenvalue.

Returns:
    float: Smallest internal eigenvalue (most negative for unstable systems)

Raises:
    RuntimeError: If no internal eigenvalues are present
)");

  stability_result.def("get_smallest_external_eigenvalue",
                       &StabilityResult::get_smallest_external_eigenvalue,
                       R"(
Get the smallest external eigenvalue.

Returns:
    float: Smallest external eigenvalue (most negative for unstable systems)

Raises:
    RuntimeError: If no external eigenvalues are present
)");

  stability_result.def("get_smallest_eigenvalue",
                       &StabilityResult::get_smallest_eigenvalue,
                       R"(
Get the smallest eigenvalue overall.

Returns:
    float: Smallest eigenvalue across both internal and external (most negative for unstable systems)

Raises:
    RuntimeError: If no eigenvalues are present
)");

  stability_result.def(
      "get_smallest_internal_eigenvalue_and_vector",
      &StabilityResult::get_smallest_internal_eigenvalue_and_vector,
      R"(
Get the smallest internal eigenvalue and its corresponding eigenvector.

Returns:
    tuple[float, numpy.ndarray]: Tuple of (eigenvalue, eigenvector) for the smallest internal eigenvalue

Raises:
    RuntimeError: If no internal eigenvalues are present
)");

  stability_result.def(
      "get_smallest_external_eigenvalue_and_vector",
      &StabilityResult::get_smallest_external_eigenvalue_and_vector,
      R"(
Get the smallest external eigenvalue and its corresponding eigenvector.

Returns:
    tuple[float, numpy.ndarray]: Tuple of (eigenvalue, eigenvector) for the smallest external eigenvalue

Raises:
    RuntimeError: If no external eigenvalues are present
)");

  stability_result.def("get_smallest_eigenvalue_and_vector",
                       &StabilityResult::get_smallest_eigenvalue_and_vector,
                       R"(
Get the smallest eigenvalue and its corresponding eigenvector overall.

Returns:
    tuple[float, numpy.ndarray]: Tuple of (eigenvalue, eigenvector) for the smallest eigenvalue across both internal and external

Raises:
    RuntimeError: If no eigenvalues are present
)");

  bind_getter_as_property(stability_result, "get_summary",
                          &StabilityResult::get_summary,
                          R"(
Get summary string of stability result information.

Returns:
    str: Summary information about the stability result

Examples:
    >>> summary = result.get_summary()
    >>> print(summary)
)");

  // Data validation methods
  stability_result.def("empty", &StabilityResult::empty,
                       R"(
Check if stability result is empty (contains no eigenvalue/eigenvector data).

Returns:
    bool: True if stability result contains no eigenvalue/eigenvector data

Examples:
    >>> if not result.empty():
    ...     print("Result contains stability data")
)");

  stability_result.def("has_internal_result",
                       &StabilityResult::has_internal_result,
                       R"(
Check if stability result contains internal eigenvalue/eigenvector data.

Returns:
    bool: True if stability result contains internal eigenvalue/eigenvector data

Examples:
    >>> if result.has_internal_result():
    ...     print("Result contains internal stability data")
)");

  stability_result.def("has_external_result",
                       &StabilityResult::has_external_result,
                       R"(
Check if stability result contains external eigenvalue/eigenvector data.

Returns:
    bool: True if stability result contains external eigenvalue/eigenvector data

Examples:
    >>> if result.has_external_result():
    ...     print("Result contains external stability data")
)");

  // Serialization - following the pattern from Orbitals
  stability_result.def("to_file", stability_result_to_file_wrapper,
                       R"(
Save stability result data to file with specified format.

Generic method to save stability result data to a file.

Args:
    filename (str | pathlib.Path): Path to the file to write.
    type (str): File format type ('json' or 'hdf5')

Raises:
    RuntimeError: If the stability result data is invalid, unsupported type, or file cannot be opened/written

Examples:
    >>> result.to_file("water.stability_result.json", "json")
    >>> result.to_file("molecule.stability_result.h5", "hdf5")
    >>> from pathlib import Path
    >>> result.to_file(Path("water.stability_result.json"), "json")
)",
                       py::arg("filename"), py::arg("type"));

  stability_result.def_static("from_file", stability_result_from_file_wrapper,
                              R"(
Load stability result data from file with specified format (static method).

Generic method to load stability result data from a file.

Args:
    filename (str | pathlib.Path): Path to the file to read.
    type (str): File format type ('json' or 'hdf5')

Returns:
    StabilityResult: New ``StabilityResult`` object loaded from the file

Raises:
    ValueError: If format_type is not supported or filename doesn't follow naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid stability result data

Examples:
    >>> result = StabilityResult.from_file("water.stability_result.json", "json")
    >>> result = StabilityResult.from_file("molecule.stability_result.h5", "hdf5")
    >>> from pathlib import Path
    >>> result = StabilityResult.from_file(Path("water.stability_result.json"), "json")
)",
                              py::arg("filename"), py::arg("type"));

  stability_result.def("to_hdf5_file", stability_result_to_hdf5_file_wrapper,
                       R"(
Save stability result data to HDF5 file (with validation).

Writes all stability result data to an HDF5 file, preserving numerical precision.
HDF5 format is efficient for large datasets and supports hierarchical data structures, making it ideal for storing stability analysis results.

Args:
    filename (str | pathlib.Path): Path to the HDF5 file to write.
        Must have '.stability_result' before the file extension (e.g., ``water.stability_result.h5``, ``molecule.stability_result.hdf5``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the stability result data is invalid or the file cannot be opened/written

Examples:
    >>> result.to_hdf5_file("water.stability_result.h5")
    >>> result.to_hdf5_file("molecule.stability_result.hdf5")
    >>> from pathlib import Path
    >>> result.to_hdf5_file(Path("water.stability_result.h5"))
)",
                       py::arg("filename"));

  stability_result.def_static("from_hdf5_file",
                              stability_result_from_hdf5_file_wrapper,
                              R"(
Load stability result data from HDF5 file (static method with validation).

Reads stability result data from an HDF5 file.
The file should contain data in the format produced by ``to_hdf5_file()``.

Args:
    filename (str | pathlib.Path): Path to the HDF5 file to read.
        Must have '.stability_result' before the file extension (e.g., ``water.stability_result.h5``, ``molecule.stability_result.hdf5``)

Returns:
    StabilityResult: New ``StabilityResult`` object loaded from the file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid stability result data

Examples:
    >>> result = StabilityResult.from_hdf5_file("water.stability_result.h5")
    >>> result = StabilityResult.from_hdf5_file("molecule.stability_result.hdf5")
    >>> from pathlib import Path
    >>> result = StabilityResult.from_hdf5_file(Path("water.stability_result.h5"))
)",
                              py::arg("filename"));

  stability_result.def(
      "to_json",
      [](const StabilityResult &self) -> std::string {
        return self.to_json().dump(2);
      },
      R"(
Convert stability result data to JSON string.

Serializes all stability result information to a JSON string format.
JSON is human-readable and suitable for debugging or data exchange.

Returns:
    str: JSON string representation of the stability result data

Raises:
    RuntimeError: If the stability result data is invalid

Examples:
    >>> json_str = result.to_json()
    >>> print(json_str)  # Pretty-printed JSON
)");

  stability_result.def_static(
      "from_json",
      [](const std::string &json_str) {
        return StabilityResult::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load stability result data from JSON string (static method).

Parses stability result data from a JSON string and creates a new StabilityResult object.
The string should contain JSON data in the format produced by ``to_json()``.

Args:
    json_str (str): JSON string containing stability result data

Returns:
    StabilityResult: New StabilityResult object created from JSON data

Raises:
    RuntimeError: If the JSON string is malformed or contains invalid stability result data

Examples:
    >>> result = StabilityResult.from_json('{"internal_stable": true, "external_stable": false, ...}')
)",
      py::arg("json_str"));

  stability_result.def("to_json_file", stability_result_to_json_file_wrapper,
                       R"(
Save stability result data to JSON file.

Writes all stability result data to a JSON file with pretty formatting.
The file will be created or overwritten if it already exists.

Args:
    filename (str | pathlib.Path): Path to the JSON file to write.
        Must have '.stability_result' before the file extension (e.g., ``water.stability_result.json``, ``molecule.stability_result.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the stability result data is invalid or the file cannot be opened/written

Examples:
    >>> result.to_json_file("water.stability_result.json")
    >>> result.to_json_file("molecule.stability_result.json")
    >>> from pathlib import Path
    >>> result.to_json_file(Path("water.stability_result.json"))
)",
                       py::arg("filename"));

  stability_result.def_static("from_json_file",
                              stability_result_from_json_file_wrapper,
                              R"(
Load stability result data from JSON file (static method).

Reads stability result data from a JSON file.
The file should contain JSON data in the format produced by ``to_json_file()``.

Args:
    filename (str | pathlib.Path): Path to the JSON file to read.
        Must have '.stability_result' before the file extension (e.g., ``water.stability_result.json``, ``molecule.stability_result.json``)

Returns:
    StabilityResult: New ``StabilityResult`` object loaded from the file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid stability result data

Examples:
    >>> result = StabilityResult.from_json_file("water.stability_result.json")
    >>> result = StabilityResult.from_json_file("molecule.stability_result.json")
    >>> from pathlib import Path
    >>> result = StabilityResult.from_json_file(Path("water.stability_result.json"))
)",
                              py::arg("filename"));

  // String representation
  stability_result.def("__repr__", [](const StabilityResult &result) {
    return result.get_summary();
  });

  stability_result.def("__str__", [](const StabilityResult &result) {
    return result.get_summary();
  });

  // Pickle support - following the pattern from Orbitals
  stability_result.def(py::pickle(
      [](const StabilityResult &sr) -> std::string {
        return sr.to_json().dump();
      },
      [](const std::string &json_str) -> StabilityResult {
        auto result =
            StabilityResult::from_json(nlohmann::json::parse(json_str));
        return *result;
      }));

  // Data type name class attribute
  stability_result.attr("_data_type_name") =
      DATACLASS_TO_SNAKE_CASE(StabilityResult);
}
