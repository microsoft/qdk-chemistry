"""QDK/Chemistry Library."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

__version__ = "1.0.1"

import contextlib
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

# Import some tools for convenience
import qdk_chemistry.constants
from qdk_chemistry._core import QDKChemistryConfig
from qdk_chemistry.utils import telemetry_events
from qdk_chemistry.utils.telemetry import TELEMETRY_ENABLED

if TELEMETRY_ENABLED:
    telemetry_events.on_qdk_chemistry_import()

_DOCS_MODE = os.getenv("QDK_CHEMISTRY_DOCS", "0") == "1"


def _setup_resources() -> None:
    """Set the QDKChemistryConfig resources directory using the runtime helper.

    This function attempts to locate the QDK/Chemistry resources directory in several
    locations with the following precedence:
    0. If a user specifies a path via the environment variable QDK_CHEMISTRY_RESOURCES_PATH,
       that path is used.
    1. The standard fallback location of Path(__file__).parent / "share/qdk/chemistry/scf/resources"
    """
    # Fallback standard location within the installed package
    package_dir = Path(__file__).parent

    # Check if the user has specified a resources path via environment variable
    env_resources_path = os.getenv("QDK_CHEMISTRY_RESOURCES_PATH")
    if env_resources_path:
        env_path = Path(env_resources_path)
        try:
            if env_path.exists() and env_path.is_dir() and any(env_path.iterdir()):
                QDKChemistryConfig.set_resources_dir(str(env_path))
                return
        except (OSError, PermissionError):
            pass

        warnings.warn(
            f"The specified QDK_CHEMISTRY_RESOURCES_PATH '{env_resources_path}' is not valid. "
            f"Falling back to other resource discovery methods.",
            UserWarning,
            stacklevel=2,
        )

    try:
        original_dir = QDKChemistryConfig.get_resources_dir()
        if original_dir and Path(original_dir).exists():
            return  # Resources directory already set and exists
    except RuntimeError:
        pass  # Not set yet, continue to set it
    resources_dir = package_dir / "share" / "qdk" / "chemistry" / "scf" / "resources"
    # Fallback to the installed version with a check for invalid installations (e.g. missing resources)
    try:
        if resources_dir.exists() and resources_dir.is_dir() and any(resources_dir.iterdir()):
            QDKChemistryConfig.set_resources_dir(str(resources_dir))
            return
        raise OSError(
            f"The QDK/Chemistry resources directory '{resources_dir}' is missing or empty. "
            "Please check your installation of the QDK/Chemistry library."
        )
    except (OSError, PermissionError) as err:
        raise OSError(
            f"The standard install location for the QDK/Chemistry library {resources_dir} is not accessible. "
            "Please check your installation of the QDK/Chemistry library."
        ) from err


_setup_resources()


# Defer plugin imports until after module initialization
def _import_plugins() -> None:
    """Import pre-packaged plugins after module initialization."""
    with contextlib.suppress(ImportError):
        import qdk_chemistry.plugins.pyscf as pyscf_plugin  # noqa: PLC0415

        pyscf_plugin.load()
    with contextlib.suppress(ImportError):
        import qdk_chemistry.plugins.qiskit as qiskit_plugin  # noqa: PLC0415

        qiskit_plugin.load()


def _is_placeholder_stub(stub_file: Path) -> bool:
    """Check if a stub file is a placeholder that needs to be regenerated."""
    if not stub_file.exists():
        return True
    try:
        content = stub_file.read_text()
        return "placeholder" in content.lower()
    except (OSError, PermissionError):
        return False


def _update_stub_references(stub_file: Path) -> None:
    """Update references in stub files from _core to public API paths.

    Replaces:
    - _core.data -> qdk_chemistry.data
    - _core._algorithms -> qdk_chemistry.algorithms

    Also adds necessary imports if they don't exist.
    """
    try:
        content = stub_file.read_text()
        original_content = content
        needs_data_import = False
        needs_algorithms_import = False

        # Replace _core.data references
        if "_core.data" in content:
            content = content.replace("_core.data", "qdk_chemistry.data")
            needs_data_import = True

        # Replace _core._algorithms references
        if "_core._algorithms" in content:
            content = content.replace("_core._algorithms", "qdk_chemistry.algorithms")
            needs_algorithms_import = True

        # Only update if changes were made
        if content != original_content:
            # Check if imports need to be added
            lines = content.split("\n")
            import_section_end = 0
            has_data_import = False
            has_algorithms_import = False

            # Find where imports end and check for existing imports
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    import_section_end = i + 1
                    if "qdk_chemistry.data" in line:
                        has_data_import = True
                    if "qdk_chemistry.algorithms" in line:
                        has_algorithms_import = True
                elif stripped and not stripped.startswith("#"):
                    # Found non-import, non-comment line
                    break

            # Add missing imports
            new_imports = []
            if needs_data_import and not has_data_import:
                new_imports.append("import qdk_chemistry.data")
            if needs_algorithms_import and not has_algorithms_import:
                new_imports.append("import qdk_chemistry.algorithms")

            if new_imports:
                # Insert imports at the end of the import section
                lines[import_section_end:import_section_end] = new_imports
                content = "\n".join(lines)

            stub_file.write_text(content)
    except (OSError, PermissionError):
        pass  # Skip files that can't be read/written


def _generate_stubs_on_first_import() -> None:
    """Generate type stubs on first import (inline to avoid circular imports)."""
    # Check if stub files need to be generated by scanning the entire _core directory
    chemistry_dir = Path(__file__).parent
    qdk_dir = chemistry_dir.parent

    # Scan for all .pyi files in _core directory and its subdirectories
    stub_files: list[Path] = []
    # Recursively find all .pyi files in _core directory
    core_dir = chemistry_dir / "_core"
    if core_dir.exists():
        stub_files.extend(core_dir.rglob("*.pyi"))

    # Check if any stub file is a placeholder
    needs_generation = any(_is_placeholder_stub(stub) for stub in stub_files)

    if needs_generation:
        try:
            # Track existing stub files before generation
            existing_stubs = set()
            if qdk_dir.exists():
                existing_stubs = set(qdk_dir.rglob("*.pyi"))

            # Remove the entire _core stub directory if it exists
            if core_dir.exists() and core_dir.is_dir():
                shutil.rmtree(core_dir)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pybind11_stubgen",
                    "_core",
                    "-o",
                    ".",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(chemistry_dir),
            )

            # Remove any newly created stub files that weren't there before
            new_stubs = set(qdk_dir.rglob("*.pyi")) - existing_stubs
            for new_stub in new_stubs:
                # Only remove stubs outside the _core directory
                # (keep the ones in _core that stubgen created)
                if "_core" not in str(new_stub.relative_to(qdk_dir)):
                    new_stub.unlink()

            # Update references in all generated stub files in _core
            if core_dir.exists():
                for stub_file in core_dir.rglob("*.pyi"):
                    _update_stub_references(stub_file)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # pybind11-stubgen not available, skip

    # Generate registry overloads (deferred until algorithms module is imported)
    # This is done in qdk_chemistry.algorithms.__init__.py to avoid circular imports


def _generate_registry_stubs() -> None:
    """Generate registry.pyi with typed overloads for all algorithms."""
    try:
        # Import registry module - at this point all imports should be complete
        from qdk_chemistry.algorithms import registry as reg_module  # noqa: PLC0415

        registry_file_path = Path(reg_module.__file__)
        stub_file = registry_file_path.parent / "registry.pyi"

        # Only generate if stub file is a placeholder or doesn't exist
        if not _is_placeholder_stub(stub_file):
            return

        # Remove placeholder file if it exists
        if stub_file.exists():
            stub_file.unlink()

        overloads = [
            '"""Type stubs for registry.create() with all algorithm overloads."""',
            "",
            "from typing import Literal, overload, Union",
            "from .base import Algorithm",
            "",
        ]

        # Get all available algorithms
        all_algorithms = reg_module.available()
        all_return_types = {"Algorithm"}
        imported_modules = set()  # Track which modules need to be imported

        for algorithm_type, algorithm_names in all_algorithms.items():
            for algorithm_name in algorithm_names:
                try:
                    settings = reg_module.inspect_settings(algorithm_type, algorithm_name)
                    instance = reg_module.create(algorithm_type, algorithm_name)
                    class_type = type(instance)
                    class_name = class_type.__name__
                    class_module = class_type.__module__

                    # Special case: replace internal _core._algorithms with public algorithms API
                    if class_module == "qdk_chemistry._core._algorithms":
                        class_module = f"qdk_chemistry.algorithms.{algorithm_type}"

                    # Use the full module path for the return type
                    full_class_path = f"{class_module}.{class_name}"

                    # Track the module for import statement
                    imported_modules.add(class_module)

                    overload_lines = ["@overload"]
                    overload_lines.append("def create(")
                    overload_lines.append(f"    algorithm_type: Literal['{algorithm_type}'],")
                    overload_lines.append(f"    algorithm_name: Literal['{algorithm_name}'] | None = None,")

                    for setting_name, setting_type, default, _, _ in settings:
                        if setting_type == "str":
                            overload_lines.append(f'    {setting_name}: {setting_type} = "{default}",')
                        elif "int" in setting_type:
                            overload_lines.append(f'    {setting_name}: int = "{default}",')
                        else:
                            overload_lines.append(f"    {setting_name}: {setting_type} = {default},")

                    overload_lines.append(f") -> {full_class_path}: ...")
                    overload_lines.append("")
                    all_return_types.add(full_class_path)

                    overloads.extend(overload_lines)
                except (ImportError, AttributeError, RuntimeError, TypeError, OSError) as e:
                    # Log the exception for debugging but continue
                    warnings.warn(
                        f"Failed to generate stub for {algorithm_type}/{algorithm_name}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )

        # Add import statements for all discovered modules
        import_lines = []
        for module in sorted(imported_modules):
            import_lines.append(f"import {module}")

        if import_lines:
            # Insert imports after the initial type imports
            overloads[5:5] = [*import_lines, ""]

        # Add the base create signature
        all_return_types_str = " | ".join(sorted(all_return_types))
        overloads.extend(
            [
                "def create(",
                "    algorithm_type: str,",
                "    algorithm_name: str | None = None,",
                "    **kwargs,",
                f") -> Union[{all_return_types_str}]: ...",
            ]
        )

        overload_code = "\n".join(overloads)
        stub_file.write_text(overload_code)

    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        # Log but don't fail - type stubs are optional
        warnings.warn(
            f"Failed to generate registry stubs: {e}",
            UserWarning,
            stacklevel=2,
        )


if not _DOCS_MODE:
    # Import plugins to have their content registered in the stubs
    try:
        _import_plugins()
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Failed to import plugins: {e}", UserWarning, stacklevel=2)

    try:
        _generate_stubs_on_first_import()
        del _generate_stubs_on_first_import  # Prevent re-execution
    except (ImportError, AttributeError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        warnings.warn(
            f"Failed to generate type stubs: {e}. Type hints may be incomplete.",
            UserWarning,
            stacklevel=2,
        )

    # Generate registry stubs after all imports are complete
    try:
        _generate_registry_stubs()
        del _generate_registry_stubs  # Prevent re-execution
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        warnings.warn(
            f"Failed to generate registry type stubs: {e}. Type hints may be incomplete.",
            UserWarning,
            stacklevel=2,
        )
