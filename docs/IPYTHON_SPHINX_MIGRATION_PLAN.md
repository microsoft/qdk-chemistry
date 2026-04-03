# IPython Sphinx Directive Migration Plan

**Created:** December 13, 2025  
**Status:** Planning  
**Reference:** https://ipython.readthedocs.io/en/stable/sphinxext.html

## Overview

Migrate documentation code examples from static `literalinclude` directives to the IPython Sphinx Directive, enabling testable, executable Python code blocks that verify examples work during docs build.

## Current State

- **Documentation framework:** Sphinx with ReadTheDocs theme
- **Code examples location:** `docs/source/_static/examples/python/` (23 Python files)
- **RST pattern:** Uses `literalinclude` with cell markers (`# start-cell-*` / `# end-cell-*`)
- **Language support:** Both C++ and Python examples via tabbed content (`sphinx_inline_tabs`)
- **Key extensions:** `breathe`, `autodoc`, `autosummary`, `napoleon`, `sphinx_copybutton`

## Benefits of IPython Sphinx Directive

1. **Testable snippets:** Code blocks execute during docs build; failures break the build
2. **Output verification:** Use `:doctest:` decorator to validate expected output
3. **Session persistence:** Variables persist across blocks within a page
4. **Syntax highlighting:** Native IPython/Python highlighting with prompts
5. **Exception handling:** `:okexcept:` and `:okwarning:` for expected errors
6. **Plot integration:** `@savefig` decorator for matplotlib figures

## Migration Steps

### Step 1: Add Dependencies

**File:** `python/pyproject.toml`

Add `"ipython"` to the `docs` optional dependencies:

```toml
docs = [
  "sphinx",
  "sphinx-rtd-theme",
  "myst-parser",
  "breathe",
  "sphinx-autodoc-typehints",
  "sphinx-inline-tabs",
  "sphinxcontrib-napoleon",
  "sphinxcontrib-bibtex",
  "sphinx_copybutton",
  "ipython",  # ADD THIS
]
```

### Step 2: Enable Extensions

**File:** `docs/source/conf.py`

Add to `extensions` list:

```python
extensions = [
    # ... existing extensions ...
    # IPython directive for testable code blocks
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]
```

Add configuration options:

```python
# IPython directive configuration
ipython_mplbackend = "agg"  # Use non-interactive matplotlib backend
ipython_warning_is_error = True  # Fail build on warnings
ipython_execlines = [
    "import numpy as np",
    "import matplotlib.pyplot as plt",
]
```

### Step 3: Convert Python Examples

Transform existing Python files with cell markers into `.. ipython::` directive blocks.

**Before (literalinclude in RST):**
```rst
.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/scf_solver.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create
```

**After (ipython directive in RST):**
```rst
.. tab:: Python API

   .. ipython:: python

      from qdk_chemistry.algorithms import create
      from qdk_chemistry.data import Structure

      # Create the default ScfSolver instance
      scf_solver = create("scf_solver")
```

### Step 4: Update RST Files

**Files to update:** All `.rst` files under `docs/source/user/comprehensive/`

Pattern for conversion:
- Keep `literalinclude` for C++ examples (no change)
- Replace Python `literalinclude` with `.. ipython:: python` blocks
- Add `:doctest:` for output verification where applicable
- Add `:suppress:` for setup code that shouldn't appear in docs
- Add `:okexcept:` for examples demonstrating error handling

### Step 5: Add CI Validation

Ensure the docs build runs with `-W` (warnings as errors) to catch any IPython directive failures:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

## IPython Directive Reference

### Key Directives

```rst
.. ipython:: python
   :doctest:      # Verify output matches
   :okexcept:     # Allow exceptions
   :okwarning:    # Allow warnings
   :suppress:     # Execute but hide from output
   :verbatim:     # Show without executing
```

### Pseudo-Decorators (per-line)

```python
@doctest      # Verify this specific output
@suppress     # Hide this line/block
@savefig filename.png  # Save matplotlib figure
```

## Open Questions

1. **Session isolation:** Should each RST file have isolated sessions, or should imports persist across the entire docs build?

2. **Parallel structure:** Keep the standalone Python files in `_static/examples/python/` for users to download, or consolidate everything into RST?

3. **Output snapshots:** For non-deterministic outputs (e.g., memory addresses, timestamps), use `:okexcept:` or `@suppress`?

4. **C++ examples:** These will remain as `literalinclude` since IPython only supports Python. Is this acceptable asymmetry?

## Files Affected

### Configuration Files
- `python/pyproject.toml` - Add ipython dependency
- `docs/source/conf.py` - Add extensions and configuration

### RST Documentation (sample list)
- `docs/source/user/comprehensive/algorithms/scf_solver.rst`
- `docs/source/user/comprehensive/algorithms/state_preparation.rst`
- `docs/source/user/comprehensive/algorithms/qubit_mapper.rst`
- `docs/source/user/comprehensive/algorithms/localizer.rst`
- `docs/source/user/comprehensive/algorithms/active_space.rst`
- *(and others with `literalinclude` Python blocks)*

### Python Example Files (may deprecate or keep for downloads)
- `docs/source/_static/examples/python/scf_solver.py`
- `docs/source/_static/examples/python/state_preparation.py`
- `docs/source/_static/examples/python/qubit_mapper.py`
- *(23 total files)*

## Timeline Estimate

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Add dependencies and extensions | 30 min |
| 2 | Create proof-of-concept with one RST file | 2 hours |
| 3 | Convert remaining RST files | 4-8 hours |
| 4 | CI integration and testing | 1-2 hours |
| 5 | Documentation and cleanup | 1 hour |

**Total estimated effort:** 1-2 days

## Next Steps

1. Answer open questions above
2. Create proof-of-concept with `scf_solver.rst`
3. Validate docs build succeeds with testable snippets
4. Roll out to remaining documentation files
