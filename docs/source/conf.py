# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import shutil
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# Add path to the Python package
sys.path.insert(
    0, str(Path(__file__).parent.parent.joinpath("python", "src").absolute())
)

# -----------------------------------------------------------------------------
# Check if Graphviz 'dot' executable is available
# -----------------------------------------------------------------------------
if shutil.which("dot") is None:
    sys.stderr.write("ERROR: Graphviz 'dot' executable not found.\n")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "qdk-chemistry"
copyright = "Microsoft Corporation. All rights reserved. Licensed under the MIT License"
author = "QDK/Chemistry Team"
release = "1.0.0-rc1"

# -----------------------------------------------------------------------------
# Extensions configuration
# -----------------------------------------------------------------------------
extensions = [
    # Sphinx core extensions
    "sphinx.ext.autodoc",  # Generate API documentation from docstrings
    "sphinx.ext.autosummary",  # Create summary tables for modules/classes
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.viewcode",  # Add links to view source code
    # Additional extensions
    "sphinx_autodoc_typehints",  # Better support for Python type annotations
    # Enabling both numpydoc and napoleon results in duplicate documentation for
    # class members.  Napoleon is the cleaner option.
    # "numpydoc",  # Support NumPy-style docstrings
    # "myst_parser",  # Support Markdown as a source language
    "sphinx_inline_tabs",  # Support for tabbed content in docs
    # C++ documentation
    "breathe",  # Bridge between Sphinx and Doxygen
    # Enable Google and NumPy style docstrings parsing
    "sphinx.ext.napoleon",  # Support for Google-style and NumPy-style docstrings
    "sphinx.ext.todo",  # Support for listing to-dos
    "sphinx.ext.graphviz",  # Support for Graphviz diagrams
    "sphinxcontrib.bibtex",  # Support for bibliographic references
    "sphinx_copybutton",  # Add "copy" buttons to code blocks
]

# -----------------------------------------------------------------------------
# Viewcode configuration
# -----------------------------------------------------------------------------
# Configure viewcode to only show source for project modules, not dependencies
viewcode_follow_imported_members = False
viewcode_enable_epub = False


# -----------------------------------------------------------------------------
# Bibtex configuration
# -----------------------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "label"  # alternatives: author_year

# -----------------------------------------------------------------------------
# Autodoc settings
# -----------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,  # Include all class/module members in docs
    "undoc-members": True,  # Document members without docstrings
    "show-inheritance": True,  # Show base classes in documentation
    "special-members": "__init__",  # Include __init__ method documentation
    "imported-members": False,  # Don't include imported members in documentation
    "private-members": False,  # Exclude private members (those starting with underscore)
}
# Control documentation format
autodoc_typehints = (
    "description"  # Put type hints in description text instead of signatures
)
autodoc_typehints_format = (
    "short"  # Use shorter type names (e.g., List instead of typing.List)
)
autodoc_member_order = "bysource"  # Document members in the order they appear in source
autoclass_content = "both"  # Include both class and __init__ docstrings
# For pybind11 bindings - setting to True helps with C++ bindings
autodoc_docstring_signature = True  # Try to extract signatures from docstrings
autodoc_preserve_defaults = True  # Preserve default values in function signatures
# Configure autodoc to handle C++ extension modules and internal modules
autodoc_mock_imports = [
    "qdk_chemistry.libblaspp",  # These are still mocked since they're only used for implementation
    "qdk_chemistry.liblapackpp",
    "qdk_chemistry.libmacis",
    "qdk_chemistry.libchemistry",
    "qdk_chemistry.libsparsexx",
]

# -----------------------------------------------------------------------------
# Autosummary settings
# -----------------------------------------------------------------------------
# Enable automatic generation of API documentation
autosummary_generate = True  # Automatically generate stub pages for API
autosummary_imported_members = False  # Don't include imported members in autosummaries
autosummary_generate_overwrite = True  # Overwrite existing generated files
autosummary_ignore_module_all = True  # Don't respect __all__ when generating summaries
add_module_names = False  # Don't prepend module names to object names in output

# -----------------------------------------------------------------------------
# C++ documentation settings
# -----------------------------------------------------------------------------
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

# # -----------------------------------------------------------------------------
# # MyST parser settings (Markdown support)
# # -----------------------------------------------------------------------------
# myst_enable_extensions = [
#     "colon_fence",  # Enable ::: directive-style blocks
#     "deflist",  # Enable definition lists
#     "dollarmath",  # Enable $math$ notation
#     "fieldlist",  # Enable field lists (needed for some directives)
# ]
# myst_heading_anchors = 4  # Generate anchors for headings up to level 4
# myst_parse_relative_links = True  # Process relative links in Markdown files
# myst_all_links_external = False  # Treat some links as internal references
# # Allow Markdown files to be included directly
# source_suffix = {
#     ".rst": "restructuredtext",  # Process .rst files with reStructuredText parser
#     ".md": "markdown",  # Process .md files with Markdown parser
# }

# -----------------------------------------------------------------------------
# HTML output settings
# -----------------------------------------------------------------------------
# Theme settings - using ReadTheDocs theme
html_theme = "sphinx_rtd_theme"  # Use the ReadTheDocs theme for styling
templates_path = ["_templates"]  # Path to custom HTML templates
html_static_path: list[str] = ["_static"]  # Path to static files (CSS, JS, images)
html_css_files = [
    "custom.css",
]  # Include custom CSS file for additional styling
html_compact_lists = True  # Render lists more compactly
# Set master doc (the main page)
master_doc = "index"  # The name of the main documentation page
# Control how API documentation is structured
toc_object_entries_show_parents = "domain"  # Show parent modules in table of contents

# -----------------------------------------------------------------------------
# Language and cross-reference settings
# -----------------------------------------------------------------------------
# Tell sphinx what the primary language being documented is
primary_domain = "py"  # Set Python as the primary documentation domain

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
# Configure intersphinx mappings to link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
}


# -----------------------------------------------------------------------------
# Type hints configuration
# -----------------------------------------------------------------------------
# Prevent Sphinx from documenting standard library and third-party modules
# This stops pathlib, pydantic_settings, etc. from being included in docs
def autodoc_skip_imports(app, what, name, obj, skip, options):
    """Skip documentation for imported standard library and third-party modules."""
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


# TODO:  remove or fix this code once we confirm we don't need it.
# Right now, it is mainly useful for debugging.
#
# def viewcode_find_source(app, modname):
#     """Prevent viewcode from generating source pages for external modules."""
#     # Only allow source pages for QDK/Chemistry modules
#     if not modname.startswith("qdk_chemistry"):
#         # Return None to skip source generation for this module
#         msg = "Skipping source generation for external module: {}".format(modname)
#         warn(msg)
#     return None


def setup(app):
    """Setup function to connect autodoc-skip-member and viewcode filters."""
    app.connect("autodoc-skip-member", autodoc_skip_imports)
    # app.connect("viewcode-find-source", viewcode_find_source)


# -----------------------------------------------------------------------------
# Numpydoc configuration
# -----------------------------------------------------------------------------
# Suppress class member documentation duplication
numpydoc_show_class_members = False  # Don't list class members twice
numpydoc_show_inherited_class_members = False  # Don't show inherited members
# Enable cross-referencing parameter types
numpydoc_xref_param_type = True  # Cross-reference types in parameter docs
numpydoc_xref_aliases = {
    # Add any custom type mappings here
    "ndarray": "numpy.ndarray",  # Map 'ndarray' to 'numpy.ndarray' in links
}

# -----------------------------------------------------------------------------
# Exclusion patterns
# -----------------------------------------------------------------------------
# Exclude patterns for documentation build
exclude_patterns = ["_build"]  # Directories to exclude from build

# -----------------------------------------------------------------------------
# Warning suppression
# -----------------------------------------------------------------------------
# Suppress specific warnings that don't affect the documentation quality
suppress_warnings = [
    "duplicate_declaration.cpp",  # Suppress warnings about duplicate C++ declarations, this happens due to nested namespaces
    "ref.ref",  # sphinx does not like bools for some reason
]

# -----------------------------------------------------------------------------
# Configure output for to-dos
# -----------------------------------------------------------------------------
todo_include_todos = True  # Include todo directives in the output
todo_emit_warnings = True  # Emit warnings for todos found in the documentation
todo_link_only = True  # Link to the todo item only, not the full text
