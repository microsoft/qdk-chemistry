# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import shutil
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------

project = "qdk-chemistry"
copyright = "Microsoft Corporation. All rights reserved. Licensed under the MIT License"
author = "QDK/Chemistry Team"
release = "1.0.0-rc1"

# -----------------------------------------------------------------------------
# Perform initial setup and tests
# -----------------------------------------------------------------------------

# Add path to the Python package
sys.path.insert(
    0, str(Path(__file__).parent.parent.joinpath("python", "src").absolute())
)

# Check if Graphviz 'dot' executable is available
if shutil.which("dot") is None:
    sys.stderr.write("ERROR: Graphviz 'dot' executable not found.\n")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Extensions configuration
extensions = [
    # Sphinx core extensions
    "sphinx.ext.autodoc",  # Generate API documentation from docstrings
    "sphinx.ext.autosummary",  # Create summary tables for modules/classes
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.viewcode",  # Add links to view source code
    # Additional extensions
    "sphinx_autodoc_typehints",  # Better support for Python type annotations
    "sphinx_inline_tabs",  # Support for tabbed content in docs
    # C++ documentation
    "breathe",  # Bridge between Sphinx and Doxygen
    # Enable Google-style docstrings parsing
    "sphinx.ext.napoleon",  # Support for Google-style and NumPy-style docstrings
    "sphinx.ext.todo",  # Support for listing to-dos
    "sphinx.ext.graphviz",  # Support for Graphviz diagrams
    "sphinxcontrib.bibtex",  # Support for bibliographic references
    "sphinx_copybutton",  # Add "copy" buttons to code blocks
]

# Configure viewcode to only show source for project modules, not dependencies
viewcode_follow_imported_members = False
viewcode_enable_epub = False

# Bibtex configuration
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "label"  # alternatives: author_year

# Autodoc settings
autodoc_default_options = {
    "members": True,  # Include all class/module members in docs
    "undoc-members": True,  # Document members without docstrings
    "show-inheritance": True,  # Show base classes in documentation
    "special-members": "__init__",  # Include __init__ method documentation
    "imported-members": False,  # Don't include imported members in documentation
    "private-members": False,  # Exclude private members (those starting with underscore)
}
autodoc_typehints = (
    "description"  # Put type hints in description text instead of signatures
)
autodoc_typehints_format = (
    "short"  # Use shorter type names (e.g., List instead of typing.List)
)
autodoc_member_order = "bysource"  # Document members in the order they appear in source
autoclass_content = "class"  # "class", "both", or "init"
autodoc_docstring_signature = (
    True  # Extract signatures from docstrings; setting to True helps with C++ bindings
)
autodoc_preserve_defaults = True  # Preserve default values in function signatures
autodoc_mock_imports = [  # Configure autodoc to handle C++ extension modules and internal modules
    "qdk_chemistry.libblaspp",  # These are still mocked since they're only used for implementation
    "qdk_chemistry.liblapackpp",
    "qdk_chemistry.libmacis",
    "qdk_chemistry.libchemistry",
    "qdk_chemistry.libsparsexx",
]

# Autosummary settings
autosummary_generate = True  # Automatically generate stub pages for API
autosummary_imported_members = False  # Don't include imported members in autosummaries
autosummary_generate_overwrite = True  # Overwrite existing generated files
autosummary_ignore_module_all = True  # Don't respect __all__ when generating summaries
add_module_names = False  # Don't prepend module names to object names in output

# Breathe configuration for C++ documentation
breathe_projects = {
    "QDK/Chemistry": "api/doxygen/xml"  # Path to Doxygen XML output for C++ code
}
breathe_default_project = "QDK/Chemistry"  # Default project for Breathe to use
breathe_default_members = (
    "members",  # Include all class/module members in docs
    "undoc-members",  # Document members without docstrings
    "show-inheritance",  # Show base classes in documentation
)

# HTML output and theme settings
html_theme = "sphinx_rtd_theme"  # Use the ReadTheDocs theme for styling
templates_path = ["_templates"]  # Path to custom HTML templates
html_static_path: list[str] = ["_static"]  # Path to static files (CSS, JS, images)
html_css_files = [  # Include custom CSS file for additional styling
    "custom.css",
]
html_compact_lists = True  # Render lists more compactly
master_doc = "index"  # The name of the main documentation page
toc_object_entries_show_parents = "domain"  # Show parent modules in table of contents

# Language and cross-reference settings
primary_domain = "py"  # Set Python as the primary documentation domain

# Configure intersphinx mappings to link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
}

# Exclude patterns for documentation build
exclude_patterns = ["_build"]  # Directories to exclude from build

# Suppress specific warnings that don't affect the documentation quality
# TODO: update this list
show_warning_types = True  # Show types of warnings in output
suppress_warnings = [
    "duplicate_declaration.cpp",  # Suppress warnings about duplicate C++ declarations, this happens due to nested namespaces
    "ref.ref",  # sphinx does not like bools for some reason
    "ref.identifier:*breathe_api_autogen*",  # Suppress warnings about duplicate object descriptions from imported classes
    "toc.not_included",  # Suppress warnings about toctree entries not included in the documentation
]
nitpicky = True  # Enable nitpicky mode to catch all warnings/errors
nitpick_ignore = [
    ("cpp:identifier", "int64_t"),
    ("cpp:identifier", "size_t"),
    ("py:class", "callable"),
    ("py:class", "optional"),
    ("py:class", "_QdkMcSolverWrapper"),
    ("py:class", "HamiltonianType"),  # TODO: figure out how to resolve this reference
    ("py:exc", "SettingsAreLocked"),  # TODO: figure out how to resolve this reference
]
nitpick_ignore_regex = [
    (
        r"cpp:identifier",
        r"AlgorithmFactory.*",
    ),  # TODO: figure out how to resolve this reference
    (r"cpp:identifier", r"Args.*"),  # TODO: figure out how to resolve this reference
    (
        r"cpp:identifier",
        r"ConfigVector.*",
    ),  # TODO: figure out how to resolve this reference
    (r"cpp:identifier", r"Eigen.*"),
    (r"cpp:identifier", r"Element.*"),  # TODO: figure out how to resolve this reference
    (r"cpp:identifier", r"H5.*"),
    (
        r"cpp:identifier",
        r"ReturnType.*",
    ),  # TODO: figure out how to resolve this reference
    (
        r"cpp:identifier",
        r"SettingsAreLocked.*",
    ),  # TODO: figure out how to resolve this reference
    (r"cpp:identifier", r"macis.*"),
    (r"cpp:identifier", r"nlohmann.*"),
    (r"cpp:identifier", r"qcs.*"),
    (r"py:class", r"qsharp.*"),
    (r"py:class", r"h5py.*"),
    (r"py:class", r"numpy.*"),
    (r"py:class", r"pybind11_builtins.*"),
    (r"py:class", r"pyscf.*"),
    (r"py:class", r"qiskit.*"),
    (r"py:class", r"qdk_chemistry._core.*"),  # Ignore internal base class
]

# Configure output for to-dos
todo_include_todos = True  # Include todo directives in the output
todo_emit_warnings = True  # Emit warnings for todos found in the documentation
todo_link_only = True  # Link to the todo item only, not the full text

# -----------------------------------------------------------------------------
# Setup-related functions
# -----------------------------------------------------------------------------


def autodoc_skip_imports(app, what, name, obj, skip, options):
    """Skip documentation for imported standard library and third-party modules.

    Prevent Sphinx from documenting standard library and third-party modules.
    This stops pathlib, pydantic_settings, etc. from being included in docs.
    """
    # Get the module where the object is defined
    if hasattr(obj, "__module__"):
        module = obj.__module__
        # Skip standard library modules (pathlib, typing, etc.)
        if module and any(
            module.startswith(prefix)
            for prefix in [
                "pathlib",
                "typing",
                "collections",
                "abc",
                "enum",
                "pydantic_settings",
                "pyscf",
                "qiskit",
                "qiskit_aer",
                "qsharp",
                "ruamel",
                "dataclasses",
            ]
        ):
            return True
    return skip


def setup(app):
    """Setup function to connect autodoc-skip-member and viewcode filters."""
    import typing
    import sys

    sys.modules["typing"] = typing  # Ensure typing module is available to pybind
    app.connect("autodoc-skip-member", autodoc_skip_imports)
