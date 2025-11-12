# Building QDK/Chemistry documentation

This documentation assumes a UNIX-like environment (Linux, macOS, or Windows Subsystem for Linux).

## Install QDK/Chemistry

The main QDK/Chemistry python package must be installed following the instructions in the main [README](../README.md) file.
[Sphinx](https://www.sphinx-doc.org/en/master/), [breathe](https://breathe.readthedocs.io/en/latest/), and several related dependencies must be installed. This can be done when installing the main QDK/Chemistry package by using the `docs` extra:

```bash
pip install .[all]
```

## Install other dependencies

A few other dependencies are also required:

- [Graphviz](https://graphviz.org/download/) (for rendering diagrams)
- [Doxygen](https://www.doxygen.nl/download.html) (for C++ API documentation)

Either install the package through your OS distribution (e.g., `sudo apt install graphviz doxygen` on Ubuntu) or download and install from the links above.

## Build the documentation

Once all dependencies are installed, you can build the documentation by running the following command from the `docs/` directory:

```bash
make all
```

For a clean build, you can run:

```bash
make clean all
```

This will generate the HTML documentation in the `docs/build/html/` directory.
You can open the [`index.html`](docs/build/html/index.html) file in that directory with your web browser to view the documentation.
