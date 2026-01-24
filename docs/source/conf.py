# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib
import os
import re
import shutil
import sys
from contextlib import suppress
from pathlib import Path

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------

project = "qdk-chemistry"
copyright = "Microsoft Corporation. All rights reserved. Licensed under the MIT License"
author = "QDK/Chemistry Team"
release = "1.0.0"

# -----------------------------------------------------------------------------
# Perform initial setup and tests
# -----------------------------------------------------------------------------

# Signal to qdk_chemistry that we're in a docs build so runtime hooks stay idle
os.environ.setdefault("QDK_CHEMISTRY_DOCS", "1")

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
    "pybind11_builtins",
    "qiskit_nature",
    "qiskit_aer",
    "h5py",
]

# Autosummary settings
autosummary_generate = True  # Automatically generate stub pages for API
autosummary_imported_members = False  # Don't include imported members in autosummaries
autosummary_generate_overwrite = True  # Overwrite existing generated files
autosummary_ignore_module_all = False  # Respect __all__ when generating summaries
add_module_names = True  # Don't prepend module names to object names in output

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
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
}

# Exclude patterns for documentation build
exclude_patterns = ["_build"]  # Directories to exclude from build

# Suppress specific warnings that don't affect the documentation quality
show_warning_types = True  # Show types of warnings in output
suppress_warnings = [
    "duplicate_declaration.cpp",  # Suppress warnings about duplicate C++ declarations, this happens due to nested namespaces
    "ref.ref",  # sphinx does not like bools for some reason
    "ref.identifier:*breathe_api_autogen*",  # Suppress warnings about duplicate object descriptions from imported classes
    "toc.not_included",  # Suppress warnings about toctree entries not included in the documentation
    "*pybind11_detail*",  # Suppress warnings about pybind11 internal implementation details
]
nitpicky = True  # Enable nitpicky mode to catch all warnings/errors
nitpick_ignore_regex = [
    (r"cpp:identifier", r"Eigen.*"),
    (r"cpp:identifier", r"H5.*"),
    (r"cpp:identifier", r"macis.*"),
    (r"cpp:identifier", r"nlohmann.*"),
    (r"cpp:identifier", r"qcs.*"),
    (r"cpp:identifier", r".*::value"),
    (r"cpp:identifier", r"fmt::format_string.*"),
    (r"cpp:identifier", r"spdlog.*"),
    # C++20 concepts - Sphinx/Breathe doesn't fully support concept references yet
    (r"cpp:identifier", r"NonBoolIntegral<.*>"),
    (r"cpp:identifier", r"NonBoolIntegralVector<.*>"),
    (r"cpp:identifier", r"NonIntegralBool"),
    (r"cpp:identifier", r"VariantMember<.*>"),
    (r"cpp:identifier", r"Vector<.*>"),
    (r"cpp:identifier", r"SupportedSettingType<.*>"),
    (r"py:class", r"h5py.*"),
    (r"py:class", r"numpy.*"),
    (r"py:class", r"pathlib.*"),
    (r"py:class", r"pyscf.*"),
    (r"py:class", r"qiskit.*"),
    (r"py:class", r"qiskit_aer.*"),
    (r"py:class", r"qdk_chemistry._core.data.DataClass"),
    (r"py:class", r"qdk_chemistry._core\.data\.PauliOperatorExpression"),
    (r"py:class", r"qdk::chemistry::data::SumPauliOperatorExpression"),
    (r"py:class", r"qdk::chemistry::algorithms::HamiltonianConstructor"),
    (r"py:class", r"^SumPauliOperatorExpression$"),
    (r"py:class", r"qsharp._native.*"),
    (r"py:class", r"qsharp._qsharp.*"),
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
        target_module = name.rsplit(".", 1)[0] if what != "module" else name

        # Skip re-exported members (e.g., qdk_chemistry.algorithms re-exporting data types)
        if (
            module
            and target_module
            and module != target_module
            and module.startswith("qdk_chemistry")
            and target_module.startswith("qdk_chemistry")
        ):
            return True

        # Skip standard library modules (pathlib, typing, etc.)
        if module and any(
            module.startswith(prefix)
            for prefix in [
                "pathlib",
                "typing",
                "collections",
                "abc",
                "enum",
                "numpy",
                "pydantic_settings",
                "qiskit",
                "qiskit_aer",
                "ruamel",
                "dataclasses",
                "pybind11_builtins",
                "qiskit_nature",
                "h5py",
            ]
        ):
            return True
    return skip


_MODULE_ALIAS_RULES: tuple[tuple[str, str], ...] = (
    ("qdk_chemistry._core._algorithms", "qdk_chemistry.algorithms"),
    ("qdk_chemistry._core.data", "qdk_chemistry.data"),
    ("qdk_chemistry._core.", "qdk_chemistry."),
    ("qdk_chemistry._algorithms", "qdk_chemistry.algorithms"),
)


def _rewrite_internal_module_path(text: str) -> str:
    """Map internal module paths to their public equivalents."""

    rewritten = text
    for old, new in _MODULE_ALIAS_RULES:
        if old in rewritten:
            rewritten = rewritten.replace(old, new)
    return rewritten


def normalize_autodoc_signature(
    app, what, name, obj, options, signature, return_annotation
):
    """Rewrite signatures/annotations pointing to internal modules."""

    new_signature = signature
    new_return = return_annotation

    if isinstance(signature, str):
        rewritten = _rewrite_internal_module_path(signature)
        if rewritten != signature:
            new_signature = rewritten

    if isinstance(return_annotation, str):
        rewritten = _rewrite_internal_module_path(return_annotation)
        if rewritten != return_annotation:
            new_return = rewritten

    if new_signature is not signature or new_return is not return_annotation:
        return new_signature, new_return
    return None


def normalize_autodoc_docstring(app, what, name, obj, options, lines):
    """Rewrite internal module references that appear inside docstrings."""

    for idx, line in enumerate(lines):
        rewritten = _rewrite_internal_module_path(line)
        if rewritten != line:
            lines[idx] = rewritten
    if options is not None and "._core." in name:
        options["noindex"] = True


def on_builder_inited(app):
    for internal_mod, public_mod in _MODULE_ALIAS_RULES:
        if internal_mod.endswith("."):
            continue  # prefix-only rewrite, nothing to alias
        if internal_mod not in sys.modules:
            continue  # nothing imported yet
        pub = importlib.import_module(public_mod)
        exports = getattr(pub, "__all__", ())
        for name in exports:
            obj = getattr(pub, name, None)
            module_name = getattr(obj, "__module__", "")
            if isinstance(module_name, str) and module_name.startswith(internal_mod):
                with suppress(AttributeError):
                    obj.__module__ = public_mod  # docs-only shim


# Pattern to match :cite:`key` in text nodes (handles the raw text form)
_CITE_TEXT_PATTERN = re.compile(r":cite:`([^`]+)`")

# Pattern to match ":cite:" followed by inline literal containing the key
# This matches what Breathe produces: ":cite:" as text + <literal> node with key
_CITE_PREFIX = ":cite:"


def _create_pending_citation(app, docname, key):
    """Create a pending cross-reference node for a bibtex citation.

    This creates a node that sphinxcontrib-bibtex will resolve during
    the final reference resolution phase, ensuring proper bibliography
    inclusion and link generation.
    """
    from docutils import nodes  # type: ignore[import-untyped]
    from sphinx.addnodes import pending_xref

    key = key.strip()

    # Create a pending cross-reference that Sphinx/bibtex will resolve
    # The 'cite' role in sphinxcontrib-bibtex uses 'cite:p' or similar
    # We create a pending_xref that will be resolved by the bibtex extension
    pxref = pending_xref(
        "",
        nodes.Text(f"[{key}]"),
        refdomain="cite",
        reftype="p",  # parenthetical citation
        reftarget=key,
        refwarn=True,
    )
    pxref["ids"] = []
    pxref["classes"] = ["bibtex"]

    return pxref


def _create_fallback_citation(app, docname, key):
    """Create a fallback citation reference when bibtex pending_xref fails.

    Creates a simple reference node linking to the references page.
    """
    from docutils import nodes  # type: ignore[import-untyped]

    key = key.strip()
    cite_text = nodes.inline("", f"[{key}]", classes=["bibtex-fallback"])
    ref_node = nodes.reference(
        "",
        "",
        cite_text,
        internal=True,
        refuri=f"references.html#id-{key.lower()}",
        classes=["bibtex", "internal"],
    )
    return ref_node


def transform_doctree_citations(app, doctree):
    """Transform :cite:`key` markers in the doctree after reading.

    This runs after Breathe has processed all Doxygen content, finding
    patterns where `:cite:` appears as literal text followed by the key
    in a literal/code node, and replaces them with proper citation nodes
    that sphinxcontrib-bibtex will process.
    """
    from docutils import nodes  # type: ignore[import-untyped]

    # Get current document name for reference resolution
    docname = app.env.docname if hasattr(app.env, "docname") else ""

    # Collect citation keys for later registration with bibtex
    citation_keys = set()

    # First pass: find patterns where ":cite:" text is followed by a literal node
    # This is how Breathe renders the Doxygen :cite:`key` text
    parents_to_process = set()

    for node in doctree.traverse(nodes.literal):
        parent = node.parent
        if parent is None:
            continue

        try:
            idx = parent.index(node)
        except (ValueError, TypeError):
            continue

        # Check if previous sibling is a text node ending with ":cite:"
        if idx > 0:
            prev_node = parent[idx - 1]
            if isinstance(prev_node, nodes.Text):
                text = str(prev_node)
                if text.rstrip().endswith(_CITE_PREFIX):
                    parents_to_process.add(id(parent))

    # Second pass: process and replace nodes in affected parents
    for node in doctree.traverse(nodes.Element):
        if id(node) not in parents_to_process:
            continue

        # Work through children looking for ":cite:" + literal patterns
        i = 0
        while i < len(node.children) - 1:
            child = node.children[i]
            next_child = node.children[i + 1] if i + 1 < len(node.children) else None

            if (
                isinstance(child, nodes.Text)
                and next_child is not None
                and isinstance(next_child, nodes.literal)
            ):
                text = str(child)
                if text.rstrip().endswith(_CITE_PREFIX):
                    # Found the pattern - extract citation key from literal
                    cite_key = next_child.astext().strip()
                    citation_keys.add(cite_key)

                    # Create new text without ":cite:" suffix
                    prefix_pos = text.rstrip().rfind(_CITE_PREFIX)
                    new_text = text[:prefix_pos]

                    # Create citation reference
                    cite_ref = _create_pending_citation(app, docname, cite_key)

                    # Replace nodes
                    node.remove(child)
                    node.remove(next_child)

                    insert_pos = i
                    if new_text:
                        node.insert(insert_pos, nodes.Text(new_text))
                        insert_pos += 1
                    node.insert(insert_pos, cite_ref)

                    # Don't increment i, process same position again
                    continue

            i += 1

    # Third pass: handle any remaining :cite:`key` in plain text nodes
    text_nodes_to_process = []
    for node in doctree.traverse(nodes.Text):
        text = str(node)
        if _CITE_TEXT_PATTERN.search(text):
            text_nodes_to_process.append(node)

    for node in text_nodes_to_process:
        text = str(node)
        parent = node.parent
        if parent is None:
            continue

        try:
            idx = parent.index(node)
        except (ValueError, TypeError):
            continue

        # Build replacement nodes
        new_nodes = []
        last_end = 0

        for match in _CITE_TEXT_PATTERN.finditer(text):
            # Text before citation
            if match.start() > last_end:
                new_nodes.append(nodes.Text(text[last_end : match.start()]))

            # Citation reference
            cite_key = match.group(1).strip()
            citation_keys.add(cite_key)
            new_nodes.append(_create_pending_citation(app, docname, cite_key))
            last_end = match.end()

        # Remaining text
        if last_end < len(text):
            new_nodes.append(nodes.Text(text[last_end:]))

        # Replace original node
        if new_nodes and last_end > 0:  # Only if we found matches
            parent.remove(node)
            for i, new_node in enumerate(new_nodes):
                parent.insert(idx + i, new_node)


def process_breathe_docstring(app, what, name, obj, options, lines):
    """Process :cite:`key` markers that appear in docstrings.

    For Python docstrings processed by autodoc, the :cite: role should work
    natively with sphinxcontrib-bibtex. This hook is available for any
    additional processing needs.
    """
    # The :cite:`key` syntax works as-is for autodoc docstrings
    pass


def setup(app):
    """Setup function to connect autodoc-skip-member and viewcode filters."""
    import sys
    import typing

    sys.modules["typing"] = typing  # Ensure typing module is available to pybind
    app.connect("autodoc-skip-member", autodoc_skip_imports)
    app.connect("autodoc-process-signature", normalize_autodoc_signature)
    app.connect("autodoc-process-docstring", normalize_autodoc_docstring)
    app.connect("autodoc-process-docstring", process_breathe_docstring)
    app.connect("builder-inited", on_builder_inited)
    # Transform :cite:`key` markers in Doxygen/Breathe content before reference resolution
    # Using doctree-read so pending_xref nodes get resolved by bibtex extension
    app.connect("doctree-read", transform_doctree_citations)
