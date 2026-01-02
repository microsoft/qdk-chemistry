# Quantum Applications Toolkit (QDK/Chemistry)

QDK/Chemistry is an open-source C++ and Python package within the [Azure Quantum Development Kit (QDK)](https://github.com/microsoft/qdk).
It provides an end-to-end toolkit for quantum chemistry:  from molecular setup and Hamiltonian generation to quantum algorithm execution and results analysis.
Designed for seamless integration with existing Python and chemistry workflows, QDK Chemistry enables researchers to simulate and run problems on near-term quantum hardware, explore strongly correlated systems, and advance toward practical quantum chemistry applications.

## Overview

QDK/Chemistry provides a comprehensive suite of tools for:

- Molecular structure representation and manipulation
- Molecular orbital calculations and analysis
- Basis set management
- Configuration and settings management
- High-performance quantum algorithms

## Documentation

- **Website**: The full documentation is hosted [online](https://microsoft.github.io/qdk-chemistry/index.html)
- **C++ API**: Headers in `cpp/include/` contain comprehensive Doxygen documentation
- **Python API**: All methods include detailed docstrings with Parameters, Returns, Raises, and Examples sections
- **Examples**: See the `examples/` directory and [documentation](https://microsoft.github.io/qdk-chemistry/index.html) for usage examples

## Project Structure

```txt
qdk-chemistry/
├── cpp/                # C++ core library
│   ├── include/        # Header files
│   ├── src/            # Implementation files
│   └── tests/          # C++ unit tests
├── docs/               # Static documentation
├── examples/           # Example scripts showing usage and language interoperability
├── external/           # External libraries and scripts
└── python/             # Python bindings
    ├── src/            # pybind11 wrapper and python code
    └── tests/          # Python unit tests
```

## Installing

Detailed instructions for installing QDK/Chemistry can be found in [INSTALL.md](./INSTALL.md)

## Contributing

There are many ways in which you can participate in this project, for example:

- [Submit bugs and feature requests](https://github.com/microsoft/qdk-chemistry/issues), and help us verify as they are checked in
- Review [source code changes](https://github.com/microsoft/qdk-chemistry/pulls)
- Review the documentation and make pull requests for anything from typos to additional and new content

If you are interested in fixing issues and contributing directly to the code base,
please see the document [How to Contribute](https://github.com/microsoft/qdk-chemistry/blob/main/CONTRIBUTING.md).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.
