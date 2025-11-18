# Quantum Applications Toolkit (QDK/Chemistry)

A high-performance quantum computing toolkit with a C++ core and Python bindings for quantum algorithms and molecular
simulations.

## Overview

QDK/Chemistry provides a comprehensive suite of tools for:

- Molecular structure representation and manipulation
- Molecular orbital calculations and analysis
- Basis set management
- Configuration and settings management
- High-performance quantum chemistry algorithms

## Documentation

- **Website**: The static documentation is hosted at [microsoft.github.io/qdk-cheistry](https://microsoft.github.io/qdk-cheistry/index.html)
- **C++ API**: Headers in `cpp/include/` contain comprehensive Doxygen documentation
- **Python API**: All methods include detailed docstrings with Parameters, Returns, Raises, and Examples sections
- **Examples**: See below, or the `python/examples/` and `docs/examples` directories for usage examples

## Project Structure

```txt
qdk-chemistry/
├── cpp/                # C++ core library
│   ├── include/        # Header files
│   ├── src/            # Implementation files
│   └── tests/          # C++ unit tests
├── docs/               # Static documentation
├── external/           # External libraries and scripts
└── python/             # Python bindings
    ├── src/            # pybind11 wrapper and python code
    └── tests/          # Python unit tests
```

## Installing

Detailed instructions for installing QDK/Chemistry can be found in [INSTALL.md](./INSTALL.md)

## Example Usage

```cpp
#include <Eigen/Dense>
#include <iostream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <vector>

// C++ API
std::vector<Eigen::Vector3d> coords = {
    {0.000000000, -0.0757918436, 0.000000000000},
    {0.866811829, 0.6014357793, -0.000000000000},
    {-0.866811829, 0.6014357793, -0.000000000000}
};

// Convert to Bohr
for (auto& coord : coords) {
    coord *= qdk::chemistry::constants::angstrom_to_bohr;
}

std::vector<qdk::chemistry::data::Element> elements = {
    qdk::chemistry::data::Element::O,
    qdk::chemistry::data::Element::H,
    qdk::chemistry::data::Element::H
};

qdk::chemistry::data::Structure water(coords, elements);

// JSON serialization - note the required naming convention
nlohmann::json json_data = water.to_json();
water.to_json_file("water.structure.json");  // Required: .structure before .json

// HDF5 serialization
qdk::chemistry::data::Orbitals orbitals;
orbitals.to_hdf5_file("molecule.orbitals.h5");  // Required: .orbitals before .h5

// Loading - using static deserialization methods (returns shared_ptr)
auto loaded = qdk::chemistry::data::Structure::from_json(json_data);
auto loaded_from_file = qdk::chemistry::data::Structure::from_json_file("water.structure.json");
auto loaded_orbitals = qdk::chemistry::data::Orbitals::from_hdf5_file("molecule.orbitals.h5");

// All other data types follow the same pattern
auto basis_set = qdk::chemistry::data::BasisSet::from_json_file("molecule.basis_set.json");
auto settings = qdk::chemistry::data::Settings::from_hdf5_file("config.settings.h5");
```

```python
# Python API
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, Settings, Orbitals

# Create water structure
coords = np.array([[0.0, 0.0, 0.0], [1.431, 1.107, 0.0], [-1.431, 1.107, 0.0]])  # Bohr
water = Structure(coords, ["O", "H", "H"])

# Serialization - note the required naming convention
json_str = water.to_json()
water.to_json_file("water.structure.json")
water.to_xyz_file("water.structure.xyz")

# Setup and use the SCF (Hartree-Fock) solver
scf_solver = create("scf_solver")
scf_settings = scf_solver.settings() # Access settings object with default parameters
scf_settings.set("basis_set", "def2-tzvp")  # change some settings parameters
charge = 0
multiplicity = 1
e_scf, wavefunction = scf_solver.run(water, charge, multiplicity) # Run SCF calculation
print(f"SCF Energy: {e_scf} Hartree")

# Store the settings and orbitals for later use
orbitals = wavefunction.get_orbitals()
orbitals.to_hdf5_file("molecule.orbitals.h5")
scf_settings.to_hdf5_file("config.settings.h5")

# Loading - using static deserialization methods (returns objects directly in Python)
loaded_structure = Structure.from_json_file("water.structure.json")
loaded_orbitals = Orbitals.from_hdf5_file("molecule.orbitals.h5")
loaded_settings = Settings.from_hdf5_file("config.settings.h5")
```

## Contributing

There are many ways in which you can participate in this project, for example:

- [Submit bugs and feature requests](https://github.com/microsoft/qdk-chemistry/issues), and help us verify as they are checked in
- Review [source code changes](https://github.com/microsoft/qdk-chemistry/pulls)
- Review the documentation and make pull requests for anything from typos to additional and new content

If you are interested in fixing issues and contributing directly to the code base,
please see the document [How to Contribute](https://github.com/microsoft/qdk-chemistryblob/main/CONTRIBUTING.md).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.
