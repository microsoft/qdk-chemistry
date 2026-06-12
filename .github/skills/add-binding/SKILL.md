---
name: add-binding
description: "Use when: adding pybind11 bindings for a C++ class, exposing C++ functionality to Python, updating module.cpp, creating .pyi stubs. Step-by-step guide for the qdk-chemistry binding layer."
---

# Add a pybind11 Binding for a C++ Class

## Overview

This skill walks through exposing an existing C++ class from the `qdk::chemistry` namespace to Python via pybind11. The binding layer lives in `python/src/pybind11/` with one `.cpp` file per class, a central module entry point (`module.cpp`), and `.pyi` type stubs for IDE support.

**Binding patterns used in this project:**

- **`py::smart_holder`** for automatic memory management across the language boundary
- **Trampoline classes** (`PyDataClass`, `PySettings`) enable Python subclassing of C++ abstract bases
- **Wrapper lambdas** for type conversion (e.g., `nlohmann::json` ↔ Python `dict`, `pathlib.Path` ↔ `std::string`)
- **GIL release** (`py::gil_scoped_release`) during I/O and heavy compute
- **Factory template** `bind_algorithm_factory<FactoryType, AlgorithmType, BaseType>()` for all factory classes
- **Property mapping**: C++ getters become Python properties via lambda wrappers

## Procedure

### Step 1: Create the binding file

Create `python/src/pybind11/bind_<classname>.cpp`. Follow the pattern of existing binding files:

```cpp
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Add other pybind11 headers as needed (numpy, eigen, etc.)
#include <qdk/chemistry/data/your_class.hpp>  // or algorithms/, utils/

namespace py = pybind11;

void bind_your_class(py::module_& m) {
    // For data classes, bind into the "data" submodule:
    auto data_m = m.def_submodule("data");

    py::class_<qdk::chemistry::YourClass>(data_m, "YourClass")
        .def(py::init<>())
        .def("method_name", &qdk::chemistry::YourClass::method_name)
        .def_property_readonly("prop_name", [](const qdk::chemistry::YourClass& self) {
            return self.get_prop();
        });
}
```

### Step 2: Register in module.cpp

Edit `python/src/pybind11/module.cpp`:

1. Add a forward declaration: `void bind_your_class(py::module_&);`
2. Call it inside the `PYBIND11_MODULE` block: `bind_your_class(m);`

**IMPORTANT**: Binding order matters — dependencies must be bound first (e.g., `Element` enums before `Structure`). Place your call after any types it depends on.

### Step 3: Add to CMake build

Edit `python/CMakeLists.txt` and add your new `.cpp` file to the pybind11 extension module's source list.

### Step 4: Create .pyi type stub

Create or update the type stub file at `python/src/qdk_chemistry/_core/<submodule>.pyi` (e.g., `data.pyi`, `_algorithms.pyi`, or `utils.pyi`). This is required for type checking and IDE support.

The stub should declare the class with its methods and properties using Python type hints:

```python
class YourClass:
    def __init__(self) -> None: ...
    def method_name(self) -> ReturnType: ...
    @property
    def prop_name(self) -> PropType: ...
```

### Step 5: Export from Python package

If the class should be part of the public API, add it to the appropriate `__init__.py`:

- Data classes: `python/src/qdk_chemistry/data/__init__.py`
- Algorithm classes: `python/src/qdk_chemistry/algorithms/__init__.py`

### Step 6: Build and test

1. Rebuild C++ (if headers changed): `cmake --preset release && cmake --build --preset release`
2. Install C++: `cmake --install .local/release/build`
3. Reinstall Python: `cd python && CMAKE_PREFIX_PATH=$(pwd)/../.local/release/install pip install .[all] -v`
4. Test import: `python -c "from qdk_chemistry._core.data import YourClass; print(YourClass)"`

## Key Paths

| Path | Purpose |
|------|---------|
| `python/src/pybind11/` | All binding source files (one per class) |
| `python/src/pybind11/module.cpp` | Module entry point — registers all bindings |
| `python/CMakeLists.txt` | Build config for pybind11 extension |
| `python/src/qdk_chemistry/_core/` | Type stub files (`.pyi`) |
| `python/src/qdk_chemistry/data/__init__.py` | Public API exports for data classes |
| `cpp/include/qdk/chemistry/` | C++ headers to bind |

## Common Pitfalls

- **Wrong binding order in module.cpp**: If class B uses class A, bind A first. Look at the existing order in `module.cpp` for guidance.
- **Missing `#include`**: The binding file must include both pybind11 headers and the C++ class header.
- **Forgetting .pyi stub**: The `check-pyi-stubs` pre-commit hook will catch this.
- **Not rebuilding C++ before Python**: If you changed C++ headers, you must rebuild and reinstall C++ before reinstalling Python.
- **GIL**: Release the GIL (`py::gil_scoped_release`) for any long-running computation or I/O to avoid blocking Python threads.
- **Memory management**: Use `py::smart_holder` for classes that need shared ownership. Use `py::return_value_policy::reference_internal` for getters returning references to internal data.
