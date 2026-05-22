"""Base class for immutable data classes with common serialization methods."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any

import h5py

from qdk_chemistry._core.data import (
    DataClass as _CoreDataClass,
)
from qdk_chemistry._core.data import (
    _validate_serialization_version,
)

__all__: list[str] = []


def _validate_filename_suffix(filename: str | Path, data_type: str, operation: str) -> str:
    """Validate that filename has the correct data type suffix.

    Args:
        filename (str | pathlib.Path): Filename to validate (e.g., "example.structure.json")
        data_type (str): Expected data structure type (e.g., "structure")
        operation (str): Operation type ("read" or "write") for error messages

    Returns:
        str: The original filename as string if valid

    Raises:
        ValueError: If filename doesn't have correct data type suffix

    Examples:
        Valid filenames:

        - ``example.structure.json`` when ``data_type="structure"``
        - ``data.wavefunction.h5`` when ``data_type="wavefunction"``
        - ``result.qpe_result.json`` when ``data_type="qpe_result"``

    """
    # Convert Path to string if needed
    filename_str = str(filename)

    # Find the last dot (extension)
    last_dot_idx = filename_str.rfind(".")
    if last_dot_idx == -1:
        # Check if filename ends with just the data type
        if filename_str.endswith(f".{data_type}"):
            return filename_str
        raise ValueError(f"Invalid filename for {operation}: Filename '{filename_str}' must have '.{data_type}' suffix")

    base = filename_str[:last_dot_idx]

    # Find the second-to-last dot (data type)
    second_last_dot_idx = base.rfind(".")
    if second_last_dot_idx == -1:
        raise ValueError(
            f"Invalid filename for {operation}: Filename '{filename_str}' "
            f"must have '.{data_type}.' before the file extension"
        )

    file_data_type = base[second_last_dot_idx + 1 :]
    if file_data_type != data_type:
        raise ValueError(
            f"Invalid filename for {operation}: Filename '{filename_str}' "
            f"has wrong data type '{file_data_type}', expected '{data_type}'"
        )

    return filename_str


class DataClass(_CoreDataClass):
    """Base class for immutable data classes with common serialization methods.

    This abstract base class provides:

    - Immutability after construction (__setattr__ and __delattr__ protection)
    - Common implementations of to_json_file, to_hdf5_file, and to_file
    - Filename validation to enforce naming convention (.<data_type>.<extension>)
    - Requires derived classes to implement abstract methods

    Derived classes MUST implement:

    1. _data_type_name class attribute (e.g., "structure", "wavefunction")
    2. _serialization_version class attribute (e.g., "0.1.0")
    3. get_summary() -> str
        Return a human-readable summary string of the object
    4. to_json() -> dict[str, Any]
        Return a dictionary representation suitable for JSON serialization
        Use _add_json_version() to include version information
    5. to_hdf5(group: h5py.Group) -> None
        Write the object's data to an HDF5 group
        Use _add_hdf5_version() to include version information
    6. from_json(json_data: dict[str, Any]) -> DataClass
        Create an instance from a JSON dictionary
        Use _validate_json_version() to check version compatibility
    7. from_hdf5(group: h5py.Group) -> DataClass
        Load an instance from an HDF5 group.
        Use _validate_hdf5_version() to check version compatibility

    Derived classes should:

    - Set all attributes before calling super().__init__()
    - Call super().__init__() at the end of __init__ to enable immutability

    Notes:
        These methods are pure virtual in the C++ base class and MUST be overridden.
        The C++ binding will enforce this at runtime.

        Filename validation enforces the convention that filenames must include
        the data type, e.g., "example.structure.json" or "data.wavefunction.h5"

        The _serialization_version attribute must be set for each derived class
        to track serialization format versions independently. Version validation
        ensures major and minor versions match exactly during deserialization,
        while allowing patch version differences for backward compatibility.

    """

    # Class attributes to be overridden by derived classes
    _data_type_name: str | None = None
    _serialization_version: str | None = None

    def __init__(self) -> None:
        """Initialize the base class and mark instance as initialized.

        This must be called by derived classes after setting all attributes.
        """
        super().__init__()
        # Mark instance as immutable after construction
        object.__setattr__(self, "_initialized", True)

    def __getattr__(self, name: str) -> Any:
        """Provide dynamic access to 'get_' prefixed methods as properties.

        Args:
            name (str): Attribute name

        Returns:
            Any: Value returned by the corresponding 'get_' method

        """
        if name.startswith("get_"):
            attr_name = name[4:]  # Remove "get_" prefix
            attr = self.__dict__.get(attr_name, None)
            if attr is not None:
                return attr
        # Forward to parent class for other attributes
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification after initialization.

        Args:
            name (str): Attribute name
            value (Any): Attribute value

        Raises:
            AttributeError: If attempting to modify after initialization

        """
        if hasattr(self, "_initialized"):
            raise AttributeError(f"Cannot modify immutable {self.__class__.__name__} attribute '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion.

        Args:
            name (str): Attribute name

        Raises:
            AttributeError: Always, as deletion is not allowed

        """
        raise AttributeError(f"Cannot delete immutable {self.__class__.__name__} attribute '{name}'")

    def get_summary(self) -> str:
        """Get a human-readable summary of the object.

        Returns:
            str: Summary string describing the object's contents and properties

        Note:
            This method must be implemented by derived classes.
            The C++ binding enforces this as a pure virtual function.

        """
        # This will be overridden by derived classes
        # The C++ trampoline will throw an error if not implemented
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_summary()")

    def _add_json_version(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add version field to JSON data.

        Args:
            data (dict[str, Any]): Dictionary to add version to

        Returns:
            dict[str, Any]: The same dictionary with version added

        """
        if self._serialization_version is None:
            raise NotImplementedError(f"{self.__class__.__name__} must implement _serialization_version")

        data["version"] = self._serialization_version
        return data

    def _add_hdf5_version(self, group: h5py.Group) -> None:
        """Add version attribute to HDF5 group if class defines a version.

        Args:
            group (h5py.Group): HDF5 group or file to add version to

        """
        if self._serialization_version is None:
            raise NotImplementedError(f"{self.__class__.__name__} must implement _serialization_version")

        group.attrs["version"] = self._serialization_version

    @classmethod
    def _validate_json_version(cls, expected_version: str, json_data: dict[str, Any]) -> None:
        """Validate serialization version in JSON data.

        Args:
            expected_version (str): The expected version string
            json_data (dict[str, Any]): Dictionary containing the serialized data

        Raises:
            RuntimeError: If version field is missing or incompatible

        """
        if "version" not in json_data:
            raise RuntimeError(f"JSON for {cls.__name__} missing required 'version' field")
        _validate_serialization_version(expected_version, json_data["version"])

    @classmethod
    def _validate_hdf5_version(cls, expected_version: str, group: h5py.Group) -> None:
        """Validate serialization version in HDF5 group.

        Args:
            expected_version (str): The expected version string
            group (h5py.Group): HDF5 group containing the serialized data

        Raises:
            RuntimeError: If version attribute is missing or incompatible

        """
        if "version" not in group.attrs:
            raise RuntimeError(f"HDF5 group for {cls.__name__} missing required 'version' attribute")
        _validate_serialization_version(expected_version, group.attrs["version"])

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the object

        """
        return self.to_json()

    def to_json(self) -> dict[str, Any]:
        """Convert the object to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the object

        Note:
            This method must be implemented by derived classes.
            The returned dictionary should contain all data needed to
            reconstruct the object. The C++ binding enforces this as
            a pure virtual function.

        """
        # This will be overridden by derived classes
        # The C++ trampoline will throw an error if not implemented
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_json()")

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the object to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write data to

        Note:
            This method must be implemented by derived classes.
            Use group.attrs for attributes and group.create_dataset()
            for array data. The C++ binding enforces this as a pure
            virtual function.

        """
        # This will be overridden by derived classes
        # The C++ trampoline will throw an error if not implemented
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_hdf5()")

    def to_json_file(self, filename: str | Path) -> None:
        """Save the object to a JSON file.

        Args:
            filename (str | pathlib.Path): Path to the output JSON file

                Must match pattern: <name>.<data_type>.json

        Raises:
            ValueError: If filename doesn't match required pattern

        """
        if self._data_type_name:
            _validate_filename_suffix(filename, self._data_type_name, "write")
        with Path(filename).open("w") as f:
            json.dump(self.to_json(), f, indent=2)

    def to_hdf5_file(self, filename: str | Path) -> None:
        """Save the object to an HDF5 file.

        Args:
            filename (str | pathlib.Path): Path to the output HDF5 file

                Must match pattern: <name>.<data_type>.h5 or <name>.<data_type>.hdf5

        Raises:
            ValueError: If filename doesn't match required pattern

        """
        if self._data_type_name:
            _validate_filename_suffix(filename, self._data_type_name, "write")
        with h5py.File(filename, "w") as f:
            self.to_hdf5(f)

    def to_file(self, filename: str | Path, format_type: str) -> None:
        """Save the object to a file with the specified format.

        Args:
            filename (str | pathlib.Path): Path to the output file

                Must match pattern: <name>.<data_type>.<extension>

            format_type (str): Format type ("json", "hdf5", or "h5")

        Raises:
            ValueError: If format_type is not supported or filename doesn't match required pattern

        """
        if self._data_type_name:
            _validate_filename_suffix(filename, self._data_type_name, "write")
        if format_type == "json":
            self.to_json_file(filename)
        elif format_type in {"hdf5", "h5"}:
            self.to_hdf5_file(filename)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    @classmethod
    def from_dict(cls, dict_data: dict[str, Any]) -> "DataClass":
        """Create an instance from a dictionary.

        Args:
            dict_data (dict[str, Any]): Dictionary containing the serialized data

        Returns:
            DataClass: New instance of the derived class

        """
        return cls.from_json(dict_data)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "DataClass":
        """Create an instance from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data

        Returns:
            DataClass: New instance of the derived class

        Note:
            This method must be implemented by derived classes.
            The C++ binding enforces this as a static method requirement.

        """
        raise NotImplementedError(f"{cls.__name__} must implement from_json() classmethod")

    @classmethod
    def from_json_file(cls, filename: str | Path) -> "DataClass":
        """Load an instance from a JSON file.

        Args:
            filename (str | pathlib.Path): Path to the input JSON file

                Must match pattern: <name>.<data_type>.json

        Returns:
            DataClass: New instance of the derived class

        Raises:
            ValueError: If filename doesn't match required pattern

        """
        if cls._data_type_name:
            _validate_filename_suffix(filename, cls._data_type_name, "read")
        with Path(filename).open("r") as f:
            json_data = json.load(f)
        return cls.from_json(json_data)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "DataClass":
        """Load an instance from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to read data from

        Returns:
            DataClass: New instance of the derived class

        Note:
            This method must be implemented by derived classes.
            Read data using group.attrs for attributes and group[key]
            for datasets. The C++ binding enforces this as a static
            method requirement.

        """
        raise NotImplementedError(f"{cls.__name__} must implement from_hdf5() classmethod")

    @classmethod
    def from_hdf5_file(cls, filename: str | Path) -> "DataClass":
        """Load an instance from an HDF5 file.

        Args:
            filename (str | pathlib.Path): Path to the input HDF5 file

                Must match pattern: <name>.<data_type>.h5 or <name>.<data_type>.hdf5

        Returns:
            DataClass: New instance of the derived class

        Raises:
            ValueError: If filename doesn't match required pattern

        """
        if cls._data_type_name:
            _validate_filename_suffix(filename, cls._data_type_name, "read")
        with h5py.File(filename, "r") as f:
            return cls.from_hdf5(f)

    @classmethod
    def from_file(cls, filename: str | Path, format_type: str) -> "DataClass":
        """Load an instance from a file with the specified format.

        Args:
            filename (str | pathlib.Path): Path to the input file

                Must match pattern: <name>.<data_type>.<extension>

            format_type (str): Format type ("json", "hdf5", or "h5")

        Returns:
            DataClass: New instance of the derived class

        Raises:
            ValueError: If format_type is not supported or filename doesn't match required pattern

        """
        if cls._data_type_name:
            _validate_filename_suffix(filename, cls._data_type_name, "read")
        if format_type == "json":
            return cls.from_json_file(filename)
        if format_type in {"hdf5", "h5"}:
            return cls.from_hdf5_file(filename)
        raise ValueError(f"Unsupported format type: {format_type}")
