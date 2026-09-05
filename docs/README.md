# Building QDK/Chemistry documentation

## Install QDK/Chemistry

The main QDK/Chemistry Python package must be installed following the instructions in [INSTALL.md](../INSTALL.md).
[Sphinx](https://www.sphinx-doc.org/en/master/), [breathe](https://breathe.readthedocs.io/en/latest/), and several related dependencies are also required.
Installing with the `all` extra covers both:

```bash
cd python
pip install '.[all]'
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
You can open the [`index.html`](build/html/index.html) file in that directory with your web browser to view the documentation.

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

Run each step in order from the `docs/` directory. This mirrors the five build stages used by
`make all`; the Makefile's validation checks are not included.

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

## Publishing the documentation

The published site is served by GitHub Pages from the `docs/` directory of the `gh-pages` branch, and holds one directory per documentation version:

```text
docs/
  index.html      redirect to stable/
  404.html        rewrites unversioned paths into stable/
  switcher.json   version list for the theme version switcher
  dev/            latest successful main build
  stable/         copy of the newest published release
  2.1/  ...       one directory per minor version
```

The [`Docs`](../.github/workflows/docs.yaml) workflow maintains it, and [`.github/scripts/docs_site.py`](../.github/scripts/docs_site.py) does the site assembly.

`dev/` is republished automatically after every successful `Build and Test` run on `main`. If publication fails, rerun the downstream `Docs` workflow while the artifact is available. To rebuild an expired or missing artifact, run `Build and Test` manually on `main`.

Final releases are rebuilt from their tag against the exact package version from PyPI. Publishing `2.1.3` replaces `2.1/` if it is newer than the version already there; its sidebar shows `Documentation 2.1.3`, while the version switcher and URL use `2.1`. Patch-version directories are not retained. The newest published release also replaces `stable/`, while a release on an older minor line updates only its own minor directory. Older releases cannot overwrite newer documentation, and prereleases are not published.

Historical tags can be selected in a manual run. Releases that predate the current theme retain their old theme and have no version switcher, so a reader who lands in one of those directories has no in-page way back to `stable/`. If that becomes a problem for a maintained line, update its documentation configuration and publish a patch release.

Python wheels are published by a separate, approval-gated pipeline. If the exact package version is not on PyPI when the GitHub release is published, the release-triggered `Docs` workflow fails with a direct error. Rerun that same workflow after the wheel is available; GitHub preserves the original release tag for the rerun.

Manual runs (`workflow_dispatch`) accept one immutable final-release tag for validating or backfilling a release. The exact PyPI package version and minor-version target are derived from that tag's root [`VERSION`](../VERSION) file, and the workflow verifies that the checkout resolves to the matching tag.

Published minor versions are retained. A build is currently about 50 MB and GitHub Pages limits a published site to 1 GB. The site assembler warns at 800 MB and rejects publication above 1 GB.
