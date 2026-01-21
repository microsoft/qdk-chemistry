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

## Telemetry

By default, this library collects anonymous usage and performance data to help improve the user experience and product quality. The telemetry implementation can be found in [telemetry.py](./python/src/qdk_chemistry/utils/telemetry.py) and all telemetry events are defined in [telemetry_events.py](./python/src/qdk_chemistry/utils/telemetry_events.py).

To disable telemetry via bash, set the environment variable `QSHARP_PYTHON_TELEMETRY` to one of the following values: `none`, `disabled`, `false`, or `0`. For example:

```bash
export QSHARP_PYTHON_TELEMETRY='false'
```

Alternatively, telemetry can be disabled within a python script by including the following at the top of the `.py` file:

```python
import os
os.environ["QSHARP_PYTHON_TELEMETRY"] = "disabled"
```

If you have any questions about the library's use of Telemetry, please use the [Discussion forum](https://github.com/microsoft/qdk-chemistry/discussions).

## Contributing

There are many ways in which you can participate in this project, for example:

- [Submit bugs and feature requests](https://github.com/microsoft/qdk-chemistry/issues), and help us verify as they are checked in
- Review [source code changes](https://github.com/microsoft/qdk-chemistry/pulls)
- Review the documentation and make pull requests for anything from typos to additional and new content

If you are interested in fixing issues and contributing directly to the code base,
please see the document [How to Contribute](https://github.com/microsoft/qdk-chemistry/blob/main/CONTRIBUTING.md).

## Support

For help and questions about using this project, please see [SUPPORT](./SUPPORT.md).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-parties’ policies.
