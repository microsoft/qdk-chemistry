# C++ Examples

This directory contains C++ code snippets used in the QDK/Chemistry documentation.

## Prerequisites

To compile these examples, the QDK/Chemistry C++ library must be installed on your system.
See the [installation instructions](https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md) for details on building and installing the library.

## Compiling Examples

These examples can be compiled by linking against the installed `qdk::chemistry` CMake target.

### Using CMake

Create a `CMakeLists.txt` file:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_example LANGUAGES CXX)

find_package(qdk REQUIRED COMPONENTS chemistry)

add_executable(my_example example.cpp)
target_link_libraries(my_example PUBLIC qdk::chemistry)
```

Then build:

```bash
cmake -B build -DCMAKE_PREFIX_PATH="/path/to/qdk/install"
cmake --build build
```

### Using the Command Line

Alternatively, compile directly with a C++20 compiler:

```bash
g++ -std=c++20 -I/path/to/qdk/include example.cpp -o example -L/path/to/qdk/lib -lqdk-chemistry
```

## Further Reading

- [Installation Instructions](https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md)
- [Quickstart Guide](https://github.com/microsoft/qdk-chemistry/blob/main/docs/source/user/quickstart.rst)
