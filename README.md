# Microsoft Quantum Development Kit for Chemistry (QDK/Chemistry)

QDK/Chemistry is an open-source C++ and Python package within the [Microsoft Quantum Development Kit (QDK)](https://github.com/microsoft/qdk).
It provides an end-to-end toolkit for quantum chemistry: from molecular setup and classical electronic structure through quantum algorithm execution and simulation.
Designed as both a **development platform** and a **composable framework**, QDK/Chemistry enables researchers to assemble modular quantum–classical pipelines, explore strongly correlated systems, and advance toward practical quantum chemistry applications on near-term and fault-tolerant quantum computers.

## Overview

QDK/Chemistry bridges classical computational chemistry with quantum computing by providing every stage of the quantum applications pipeline in a single, modular toolkit:

- **Quantum algorithms**: a growing collection of chemistry-aware quantum algorithms, with composable building blocks for constructing higher-level quantum workflows
- **Classical electronic structure**: production-quality classical methods that generate the high-quality inputs quantum algorithms require
- **Composable architecture**: a plugin system that lets users assemble custom pipelines from interchangeable components, mixing native high-performance C++ backends with established community packages
- **Multiple quantum backends**: execute circuits on a variety of simulators through a unified interface that decouples algorithm development from backend selection

## Documentation

- **Website**: The full documentation is hosted [online](https://microsoft.github.io/qdk-chemistry/index.html)
- **C++ API**: Headers in `cpp/include/` contain comprehensive Doxygen documentation
- **Python API**: All methods include detailed docstrings with Parameters, Returns, Raises, and Examples sections
- **Examples**: See the `examples/` directory and [documentation](https://microsoft.github.io/qdk-chemistry/index.html) for usage examples
- **Reference data and companion materials**: Curated simulation datasets, molecular benchmarks, and related assets are available at [microsoft/qdk-chemistry-data](https://github.com/microsoft/qdk-chemistry-data)

## Project Structure

```txt
qdk-chemistry/
├── .github/
│   └── skills/         # Development guidance for working in this repository
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

Files under `.github/skills/` provide development-time guidance for coding
agents working on the repository. They are not part of the installed package or
its runtime behavior.

## Installing

```bash
python3 -m venv venv && source venv/bin/activate
python3 -m pip install 'qdk-chemistry[all]'
```

The `[all]` extra pulls in all optional dependencies so that examples and tests work without chasing missing packages. For other installation methods (Dev Container, building from source) and platform-specific notes, see [INSTALL.md](./INSTALL.md).

## Agent Integration Files

QDK Chemistry publishes a Copilot agent plugin from this repository. Register
the repository marketplace, then run the plugin installer from the virtual
environment containing QDK Chemistry:

```bash
copilot plugin marketplace add https://github.com/microsoft/qdk-chemistry.git
qdkchem plugin install qdk-chemistry@qdk-chemistry
```

With no target directory, Copilot installs the plugin for the current user and
QDK Chemistry pins its MCP command to that virtual environment. A local plugin
directory is also accepted; QDK Chemistry registers its ancestor marketplace in
the same Copilot scope before installation. Copilot repository subdirectory
specs are accepted directly:

```bash
qdkchem plugin install ./copilot-plugins/qdk-chemistry
qdkchem plugin install OWNER/REPO:copilot-plugins/qdk-chemistry
```

To configure one workspace instead, pass its root. QDK Chemistry copies the
fetched agents and skills into `.github`, merges its MCP server into
`.vscode/mcp.json` and `.github/mcp.json`, and keeps fetch/update state beneath
the ignored `.qdk_chem` directory:

```bash
qdkchem plugin install ./copilot-plugins/qdk-chemistry \
    --target-dir /path/to/workspace
```

Update through the same CLI so the virtual-environment binding is restored
after Copilot refreshes the plugin files. Pass the same `--target-dir` for a
workspace installation:

```bash
qdkchem plugin update qdk-chemistry
qdkchem plugin update --all
qdkchem plugin rebind qdk-chemistry
```

VS Code discovers user plugins installed by Copilot CLI and workspace assets
written by `--target-dir`. The plugin supplies:

- the `quantum-agent`, `chemist`, `researcher`, `reviewer`, and `reporter` agents;
- QDK Chemistry overview, MCP, coding, and remote-execution skills; and
- the `qdk_chemistry` MCP server configuration.

Plugin MCP processes start in the installed plugin directory. Call
`bind_workspace` before any other QDK Chemistry tool. It uses a single
client-provided file root when available; otherwise pass the active workspace as
an absolute `workspace_root`. Plugin-launched servers reject other tool calls
until binding succeeds and cannot be rebound to another workspace.

**Skills** provide tested domain knowledge: tool references, workflow recipes, parameter guidance, and common pitfalls.

**Agents** coordinate multi-step quantum chemistry workflows (research → plan → critique → execute → visualize → report). Use them for complex tasks; use skills directly for simple questions.

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

## Citing QDK/Chemistry

If you use QDK/Chemistry in your work, please cite the following paper:

> N. A. Baker, B. Bilodeau, C. Chen, Y. Chen, M. Eckhoff, A. Efimovskaya, P. Gasparotto, P. van Gerwen, R. Gong, K. Hoang, Z. Hooshmand, A. J. Jenkins, C. S. N. Johnston, R. R. Li, J. Liang, H. Liu, A. Mills, M. Mörchen, G. Nishibuchi, C. Sun, B. Ticehurst, M. Troyer, J. P. Unsleber, S. Wernli, D. B. Williams-Young, and B. Zhang, "QDK/Chemistry: A Modular Toolkit for Quantum Chemistry Applications," [arXiv:2601.15253](https://arxiv.org/abs/2601.15253) (2026).

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
