# Building QDK/Chemistry documentation

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

Install them through your OS package manager, or download and install from the links above:

| Platform | Command |
|----------|---------|
| Ubuntu / Debian | `sudo apt install graphviz doxygen` |
| Fedora / RHEL | `sudo dnf install graphviz doxygen` |
| macOS | `brew install graphviz doxygen` |
| Windows | `winget install --id Graphviz.Graphviz -e -s winget; winget install --id DimitriVanHeesch.Doxygen -e -s winget` |

On Windows, open a new terminal after installing so that the updated `PATH` is picked up, then
confirm both tools resolve:

```powershell
doxygen --version; dot -V
```

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

### Building on Windows

The [`Makefile`](Makefile) requires GNU make and a POSIX shell (it uses `rm`, `grep`, `wc`, and
`tee`), so it does not run under `cmd.exe` or PowerShell directly. Windows users have three options:

1. **WSL** — run `make all` from a [WSL](https://learn.microsoft.com/windows/wsl/install) shell.
2. **MSYS2 or Git Bash with GNU make** — for example, `winget install --id MSYS2.MSYS2 -e -s winget`
   followed by `pacman -S make` inside the MSYS2 shell. Doxygen and Graphviz must be on the `PATH`
   of that shell.
3. **Invoke the underlying tools directly** — the pipeline is a fixed sequence of five commands,
   which can be run from PowerShell as shown below.

#### Running the pipeline from PowerShell

Run each step in order from the `docs/` directory. This mirrors what `make all` does, minus the
warning-count checks that fail the build in CI.

```powershell
# 1. Doxygen XML for the C++ API
doxygen Doxyfile

# 2. Breathe bridges the Doxygen XML into Sphinx
breathe-apidoc -f -m -g namespace -p "QDK/Chemistry" `
    -o source/api/breathe_api_autogen source/api/doxygen/xml

# 3. Sphinx API stubs for the Python package
sphinx-apidoc -f --separate --module-first --private --implicit-namespaces `
    --doc-project="QDK/Chemistry Python API" `
    -o source/api/api_autogen ../python/src/qdk_chemistry

# 4. First Sphinx pass, which populates the autosummary stubs
sphinx-build -M html source build -T -j 1 -n -w sphinx-autosummary-warnings.txt

# 5. Second Sphinx pass, which produces the final HTML
sphinx-build -M html source build -j 1 -n -w sphinx-docs-warnings.txt
```

Two Sphinx passes are required: the first generates the autosummary stub pages, and the second
renders the documentation that references them.

The equivalent of `make clean` is to remove the generated directories:

```powershell
Remove-Item -Recurse -Force build, source/api/doxygen, source/api/api_autogen, `
    source/api/breathe_api_autogen -ErrorAction SilentlyContinue
Remove-Item *-output.txt, *-warnings.txt -ErrorAction SilentlyContinue
```

#### Checking for warnings

`make all` fails the build if Sphinx emits any warnings. When running the commands manually, inspect
the warning files after the build:

```powershell
Get-Content sphinx-docs-warnings.txt, sphinx-autosummary-warnings.txt -ErrorAction SilentlyContinue
```

Any output means the equivalent `make` build would have failed.

## Regenerating tutorial figures

The [ground-state QPE figure maintenance guide](source/_static/diagrams/README.md)
documents source ownership, regeneration commands, and screenshot-derived asset
maintenance.
