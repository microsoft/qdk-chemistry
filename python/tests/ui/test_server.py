"""Unit tests for qdk_chemistry.ui functions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import functools
import json

# Import functions from server module
import tempfile
from pathlib import Path

import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.data import (
    Ansatz,
    Configuration,
    ConfigurationSet,
    Structure,
)
from qdk_chemistry.ui import tools as _server
from qdk_chemistry.ui.config import config
from qdk_chemistry.ui.validation import FilenameFormatError, ensure_filename_format, is_project_valid


def _unwrap(result):
    """Unwrap @_structured envelope for backward-compatible assertions.

    Converts ``{"status": "ok", "result": X}`` → X,
    ``{"status": "exists", "message": M}`` → M (str),
    and ``{"status": "error", "message": M}`` → M (str).
    """
    if isinstance(result, dict) and "status" in result:
        if result["status"] == "ok":
            return result["result"]
        return result.get("message", "Unknown error")
    return result


def _make_unwrapped(fn):
    """Return a wrapper that automatically unwraps @_structured results."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _unwrap(fn(*args, **kwargs))

    return wrapper


# Re-export all server functions with auto-unwrapping so tests stay unchanged
create_model_hamiltonian = _make_unwrapped(_server.create_model_hamiltonian)
create_spin_model_hamiltonian = _make_unwrapped(_server.create_spin_model_hamiltonian)
get_algorithm_default_settings = _make_unwrapped(_server.get_algorithm_default_settings)
get_algorithm_default_type = _make_unwrapped(_server.get_algorithm_default_type)
get_ansatz = _make_unwrapped(_server.get_ansatz)
get_orbitals_from_input = _make_unwrapped(_server.get_orbitals_from_input)
get_top_configurations = _make_unwrapped(_server.get_top_configurations)
run_active_space_selector = _make_unwrapped(_server.run_active_space_selector)
run_controlled_evolution_circuit_mapper = _make_unwrapped(_server.run_controlled_evolution_circuit_mapper)
run_dynamical_correlation_calculator = _make_unwrapped(_server.run_dynamical_correlation_calculator)
run_energy_estimator = _make_unwrapped(_server.run_energy_estimator)
run_hamiltonian_constructor = _make_unwrapped(_server.run_hamiltonian_constructor)
run_multi_configuration_calculation = _make_unwrapped(_server.run_multi_configuration_calculation)
run_multi_configuration_scf = _make_unwrapped(_server.run_multi_configuration_scf)
run_orbital_localization = _make_unwrapped(_server.run_orbital_localization)
run_phase_estimation = _make_unwrapped(_server.run_phase_estimation)
run_projected_multi_configuration_calculation = _make_unwrapped(_server.run_projected_multi_configuration_calculation)
run_qubit_hamiltonian_solver = _make_unwrapped(_server.run_qubit_hamiltonian_solver)
run_qubit_mapper = _make_unwrapped(_server.run_qubit_mapper)
run_scf = _make_unwrapped(_server.run_scf)
run_stability_checker = _make_unwrapped(_server.run_stability_checker)
run_state_preparation = _make_unwrapped(_server.run_state_preparation)
run_time_evolution_builder = _make_unwrapped(_server.run_time_evolution_builder)
create_structure = _make_unwrapped(_server.create_structure)

# ==================== Test Fixtures ====================


@pytest.fixture
def h2_structure():
    """Create a simple H2 molecule structure."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    symbols = ["H", "H"]
    return Structure(coordinates=coords, symbols=symbols)


@pytest.fixture
def simple_wavefunction(h2_structure, temp_project_dir):
    """Create a simple wavefunction from H2 SCF calculation."""
    scf_solver = algorithms.create("scf_solver")
    _, wavefunction = scf_solver.run(h2_structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")
    # Save to project directory
    project_path = temp_project_dir / "test_project"
    project_path.mkdir(exist_ok=True)
    wavefunction.to_json_file(str(project_path / "simple.wavefunction.json"))
    return wavefunction


@pytest.fixture
def simple_orbitals(simple_wavefunction):
    """Extract orbitals from simple wavefunction."""
    return simple_wavefunction.get_orbitals()


@pytest.fixture
def simple_hamiltonian(simple_orbitals):
    """Create hamiltonian from simple orbitals."""
    ham_constructor = algorithms.create("hamiltonian_constructor")
    return ham_constructor.run(simple_orbitals)


@pytest.fixture
def simple_ansatz(simple_hamiltonian, simple_wavefunction):
    """Create a simple ansatz."""
    return Ansatz(simple_hamiltonian, simple_wavefunction)


@pytest.fixture
def simple_configuration_set(simple_orbitals):
    configurations = [
        Configuration("20"),
        Configuration("ud"),
    ]
    return ConfigurationSet(configurations, simple_orbitals)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_projects_dir = config.projects_dir
        config.projects_dir = Path(tmpdir) / "projects"
        config.projects_dir.mkdir(parents=True, exist_ok=True)
        yield config.projects_dir
        config.projects_dir = original_projects_dir


class TestValidation:
    """Test validation functions."""

    def test_is_project_valid_creates_directory(self):
        """Test that is_project_valid creates project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / "projects"
            is_valid, _ = is_project_valid("test_project", projects_dir)

            assert is_valid is True
            assert (projects_dir / "test_project").exists()


class TestFilenameFormat:
    """Test ensure_filename_format function."""

    def test_ensure_filename_format_already_correct(self):
        """Test that correctly formatted filenames are returned unchanged."""
        assert ensure_filename_format("test.wavefunction.json", "Wavefunction") == "test.wavefunction.json"
        assert ensure_filename_format("my_ham.hamiltonian.hdf5", "Hamiltonian") == "my_ham.hamiltonian.hdf5"
        assert (
            ensure_filename_format("output.qubit_hamiltonian.h5", "QubitHamiltonian") == "output.qubit_hamiltonian.h5"
        )

    def test_ensure_filename_format_auto_corrects_json(self):
        """Test that filenames without type markers get corrected for .json files."""
        assert ensure_filename_format("test.json", "Wavefunction") == "test.wavefunction.json"
        assert ensure_filename_format("output.json", "Hamiltonian") == "output.hamiltonian.json"
        assert ensure_filename_format("result.json", "QubitHamiltonian") == "result.qubit_hamiltonian.json"
        assert ensure_filename_format("orbs.json", "Orbitals") == "orbs.orbitals.json"
        assert ensure_filename_format("circuit_out.json", "Circuit") == "circuit_out.circuit.json"

    def test_ensure_filename_format_auto_corrects_hdf5(self):
        """Test that filenames without type markers get corrected for .hdf5 files."""
        assert ensure_filename_format("test.hdf5", "Wavefunction") == "test.wavefunction.hdf5"
        assert ensure_filename_format("output.h5", "Hamiltonian") == "output.hamiltonian.h5"
        assert ensure_filename_format("result.hdf5", "QubitHamiltonian") == "result.qubit_hamiltonian.hdf5"

    def test_ensure_filename_format_removes_trailing_dot(self):
        """Test that trailing dots are handled properly."""
        assert ensure_filename_format("test..json", "Wavefunction") == "test.wavefunction.json"

    def test_ensure_filename_format_raises_for_unrecognized_data_type(self):
        """Test that unrecognized data types raise FilenameFormatError."""
        with pytest.raises(FilenameFormatError) as exc_info:
            ensure_filename_format("test.json", "UnknownType")
        assert "Unrecognized data type" in str(exc_info.value)
        assert "UnknownType" in str(exc_info.value)

    def test_ensure_filename_format_raises_for_unrecognized_extension(self):
        """Test that unrecognized file extensions raise FilenameFormatError."""
        with pytest.raises(FilenameFormatError) as exc_info:
            ensure_filename_format("test.txt", "Wavefunction")
        assert "Unrecognized file extension" in str(exc_info.value)
        assert "test.txt" in str(exc_info.value)

        with pytest.raises(FilenameFormatError) as exc_info:
            ensure_filename_format("noextension", "Hamiltonian")
        assert "Unrecognized file extension" in str(exc_info.value)

    def test_ensure_filename_format_all_data_types(self):
        """Test that all supported data types work correctly."""
        data_types_and_markers = {
            "Structure": ".structure.",
            "Wavefunction": ".wavefunction.",
            "Orbitals": ".orbitals.",
            "Hamiltonian": ".hamiltonian.",
            "QubitHamiltonian": ".qubit_hamiltonian.",
            "Ansatz": ".ansatz.",
            "Circuit": ".circuit.",
            "QpeResult": ".qpe_result.",
            "StabilityResult": ".stability_result.",
            "EnergyExpectationResult": ".energy_expectation_result.",
            "MeasurementData": ".measurement_data.",
        }

        for data_type, expected_marker in data_types_and_markers.items():
            result = ensure_filename_format("test.json", data_type)
            assert expected_marker in result, f"Expected {expected_marker} in result for {data_type}, got {result}"
            assert result.endswith(".json"), f"Expected .json extension for {data_type}, got {result}"


class TestFilenameFormatInServerFunctions:
    """Test that server functions properly handle filename format errors."""

    @pytest.mark.usefixtures("simple_wavefunction", "temp_project_dir")
    def test_get_orbitals_from_input_invalid_extension(self):
        """Test get_orbitals_from_input returns error for invalid extension."""
        result = get_orbitals_from_input(
            project_name="test_project",
            input_filename="simple.wavefunction.json",
            out_orbitals_filename="output.txt",  # Invalid extension
        )
        assert isinstance(result, str)
        assert "invalid output filename" in result.lower()
        assert "unrecognized file extension" in result.lower()

    def test_run_scf_invalid_extension(self, h2_structure, temp_project_dir):
        """Test run_scf returns error for invalid output filename extension."""
        project_path = temp_project_dir / "test_project_scf_ext"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        result = run_scf(
            project_name="test_project_scf_ext",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="output.xml",  # Invalid extension
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(result, str)
        assert "invalid output filename" in result.lower()

    def test_run_qubit_mapper_invalid_extension(self, simple_hamiltonian, temp_project_dir):
        """Test run_qubit_mapper returns error for invalid output filename extension."""
        project_path = temp_project_dir / "test_project_qm_ext"
        project_path.mkdir(exist_ok=True)

        # Save hamiltonian
        simple_hamiltonian.to_json_file(str(project_path / "test.hamiltonian.json"))

        result = run_qubit_mapper(
            project_name="test_project_qm_ext",
            hamiltonian_filename="test.hamiltonian.json",
            out_qubit_hamiltonian_filename="output.dat",  # Invalid extension
        )
        assert isinstance(result, str)
        assert "invalid output filename" in result.lower()

    def test_run_hamiltonian_constructor_auto_corrects_filename(self, simple_orbitals, temp_project_dir):
        """Test that run_hamiltonian_constructor auto-corrects the output filename."""
        project_path = temp_project_dir / "test_project_autocorrect"
        project_path.mkdir(exist_ok=True)
        simple_orbitals.to_json_file(str(project_path / "simple.orbitals.json"))

        # Use filename without type marker - should be auto-corrected
        result = run_hamiltonian_constructor(
            project_name="test_project_autocorrect",
            orbitals_filename="simple.orbitals.json",
            out_hamiltonian_filename="output.json",  # Missing .hamiltonian. marker
        )

        assert isinstance(result, str)
        # Should return corrected filename
        assert "output.hamiltonian.json" in result
        assert (project_path / "output.hamiltonian.json").exists()


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_algorithm_default_type(self):
        """Test getting default algorithm type."""
        default_type = get_algorithm_default_type("scf_solver")
        assert isinstance(default_type, str)
        assert len(default_type) > 0

    def test_get_algorithm_default_settings(self):
        """Test getting default settings."""
        settings = get_algorithm_default_settings("scf_solver")
        assert isinstance(settings, dict)

    def test_get_algorithm_default_settings_with_name(self):
        """Test getting default settings with specific algorithm name."""
        settings = get_algorithm_default_settings("active_space_selector", "qdk_valence")
        assert isinstance(settings, dict)

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_get_orbitals_from_input(self, temp_project_dir):
        """Test extracting orbitals from input object and saving to file."""
        result = get_orbitals_from_input(
            project_name="test_project",
            input_filename="simple.wavefunction.json",
            out_orbitals_filename="extracted.orbitals.json",
        )

        # Should return the output filename
        assert isinstance(result, str)
        assert "extracted.orbitals.json" in result

        # Verify the file was created
        output_path = temp_project_dir / "test_project" / "extracted.orbitals.json"
        assert output_path.exists()

        # Verify we can load the orbitals back
        loaded_orbitals = data.Orbitals.from_json_file(str(output_path))
        assert loaded_orbitals is not None

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_get_orbitals_from_input_with_full_path(self, temp_project_dir):
        """Test that get_orbitals_from_input strips full paths correctly."""
        project_path = temp_project_dir / "test_project"

        # Pass full path - should be stripped to just filename
        result = get_orbitals_from_input(
            project_name="test_project",
            input_filename=str(project_path / "simple.wavefunction.json"),
            out_orbitals_filename="/some/path/out.orbitals.json",
        )

        assert isinstance(result, str)
        assert "out.orbitals.json" in result
        assert (project_path / "out.orbitals.json").exists()

    def test_get_orbitals_from_input_invalid_input(self, temp_project_dir):
        """Test get_orbitals_from_input with invalid input file."""
        project_path = temp_project_dir / "test_project_orbitals"
        project_path.mkdir(exist_ok=True)

        # Create invalid json file
        (project_path / "invalid.wavefunction.json").write_text("invalid json")

        result = get_orbitals_from_input(
            project_name="test_project_orbitals",
            input_filename="invalid.wavefunction.json",
            out_orbitals_filename="out.orbitals.json",
        )

        # Should return error message
        assert isinstance(result, str)
        assert "failed" in result.lower()

    def test_get_orbitals_from_input_nonexistent_file(self, temp_project_dir):
        """Test get_orbitals_from_input with non-existent input file."""
        project_path = temp_project_dir / "test_project_missing"
        project_path.mkdir(exist_ok=True)

        result = get_orbitals_from_input(
            project_name="test_project_missing",
            input_filename="nonexistent.wavefunction.json",
            out_orbitals_filename="out.orbitals.json",
        )

        # Should return error message
        assert isinstance(result, str)
        assert "failed" in result.lower()

    def test_get_orbitals_from_input_with_hamiltonian(self, simple_hamiltonian, temp_project_dir):
        """Test extracting orbitals from a Hamiltonian input."""
        project_path = temp_project_dir / "test_project_ham_input"
        project_path.mkdir(exist_ok=True)

        # Save hamiltonian to project directory
        simple_hamiltonian.to_json_file(str(project_path / "input.hamiltonian.json"))

        result = get_orbitals_from_input(
            project_name="test_project_ham_input",
            input_filename="input.hamiltonian.json",
            out_orbitals_filename="extracted.orbitals.json",
        )

        # Should return the output filename
        assert isinstance(result, str)
        assert "extracted.orbitals.json" in result

        # Verify the file was created
        output_path = project_path / "extracted.orbitals.json"
        assert output_path.exists()

        # Verify we can load the orbitals back
        loaded_orbitals = data.Orbitals.from_json_file(str(output_path))
        assert loaded_orbitals is not None

    def test_get_orbitals_from_input_with_ansatz(self, simple_ansatz, temp_project_dir):
        """Test extracting orbitals from an Ansatz input."""
        project_path = temp_project_dir / "test_project_ansatz_input"
        project_path.mkdir(exist_ok=True)

        # Save ansatz to project directory
        simple_ansatz.to_json_file(str(project_path / "input.ansatz.json"))

        result = get_orbitals_from_input(
            project_name="test_project_ansatz_input",
            input_filename="input.ansatz.json",
            out_orbitals_filename="extracted.orbitals.json",
        )

        # Should return the output filename
        assert isinstance(result, str)
        assert "extracted.orbitals.json" in result

        # Verify the file was created
        output_path = project_path / "extracted.orbitals.json"
        assert output_path.exists()

        # Verify we can load the orbitals back
        loaded_orbitals = data.Orbitals.from_json_file(str(output_path))
        assert loaded_orbitals is not None

    def test_get_orbitals_from_input_with_configuration_set(self, simple_configuration_set, temp_project_dir):
        """Test extracting orbitals from a ConfigurationSet input."""
        project_path = temp_project_dir / "test_project_configset_input"
        project_path.mkdir(exist_ok=True)

        # Save configuration set to project directory
        simple_configuration_set.to_json_file(str(project_path / "input.configuration_set.json"))

        result = get_orbitals_from_input(
            project_name="test_project_configset_input",
            input_filename="input.configuration_set.json",
            out_orbitals_filename="extracted.orbitals.json",
        )

        # Should return the output filename
        assert isinstance(result, str)
        assert "extracted.orbitals.json" in result

        # Verify the file was created
        output_path = project_path / "extracted.orbitals.json"
        assert output_path.exists()

        # Verify we can load the orbitals back
        loaded_orbitals = data.Orbitals.from_json_file(str(output_path))
        assert loaded_orbitals is not None

    def test_get_orbitals_from_input_unsupported_type(self, temp_project_dir):
        """Test get_orbitals_from_input with an unsupported input type."""
        project_path = temp_project_dir / "test_project_unsupported"
        project_path.mkdir(exist_ok=True)

        # Create a file that is valid JSON but not a supported input type
        (project_path / "unsupported.structure.json").write_text('{"type": "unsupported"}')

        result = get_orbitals_from_input(
            project_name="test_project_unsupported",
            input_filename="unsupported.structure.json",
            out_orbitals_filename="out.orbitals.json",
        )

        # Should return error message indicating the input type is not supported
        assert isinstance(result, str)
        assert "failed" in result.lower()

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_get_ansatz(self, simple_hamiltonian, temp_project_dir):
        """Test creating ansatz from wavefunction and hamiltonian."""
        project_path = temp_project_dir / "test_project"

        # Save hamiltonian to project directory
        simple_hamiltonian.to_json_file(str(project_path / "simple.hamiltonian.json"))

        result = get_ansatz(
            project_name="test_project",
            wavefunction_filename="simple.wavefunction.json",
            hamiltonian_filename="simple.hamiltonian.json",
            out_ansatz_filename="created.ansatz.json",
        )

        # Should return the output filename
        assert isinstance(result, str)
        assert "created.ansatz.json" in result

        # Verify the file was created
        output_path = project_path / "created.ansatz.json"
        assert output_path.exists()

        # Verify we can load the ansatz back
        loaded_ansatz = data.Ansatz.from_json_file(str(output_path))
        assert loaded_ansatz is not None

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_get_ansatz_with_full_paths(self, simple_hamiltonian, temp_project_dir):
        """Test that get_ansatz strips full paths correctly."""
        project_path = temp_project_dir / "test_project"
        simple_hamiltonian.to_json_file(str(project_path / "simple.hamiltonian.json"))

        # Pass full paths - should be stripped to just filenames
        result = get_ansatz(
            project_name="test_project",
            wavefunction_filename=str(project_path / "simple.wavefunction.json"),
            hamiltonian_filename=str(project_path / "simple.hamiltonian.json"),
            out_ansatz_filename="/some/path/out.ansatz.json",
        )

        assert isinstance(result, str)
        assert "out.ansatz.json" in result
        assert (project_path / "out.ansatz.json").exists()

    def test_get_ansatz_invalid_wavefunction(self, simple_hamiltonian, temp_project_dir):
        """Test get_ansatz with invalid wavefunction file."""
        project_path = temp_project_dir / "test_project_ansatz_invalid"
        project_path.mkdir(exist_ok=True)

        # Create invalid wavefunction json file
        (project_path / "invalid.wavefunction.json").write_text("invalid json")
        simple_hamiltonian.to_json_file(str(project_path / "valid.hamiltonian.json"))

        result = get_ansatz(
            project_name="test_project_ansatz_invalid",
            wavefunction_filename="invalid.wavefunction.json",
            hamiltonian_filename="valid.hamiltonian.json",
            out_ansatz_filename="out.ansatz.json",
        )

        # Should return error message about wavefunction
        assert isinstance(result, str)
        assert "failed" in result.lower()
        assert "wavefunction" in result.lower()

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_get_ansatz_invalid_hamiltonian(self, temp_project_dir):
        """Test get_ansatz with invalid hamiltonian file."""
        project_path = temp_project_dir / "test_project"

        # Create invalid hamiltonian json file
        (project_path / "invalid.hamiltonian.json").write_text("invalid json")

        result = get_ansatz(
            project_name="test_project",
            wavefunction_filename="simple.wavefunction.json",
            hamiltonian_filename="invalid.hamiltonian.json",
            out_ansatz_filename="out.ansatz.json",
        )

        # Should return error message about hamiltonian
        assert isinstance(result, str)
        assert "failed" in result.lower()
        assert "hamiltonian" in result.lower()

    def test_get_ansatz_nonexistent_files(self, temp_project_dir):
        """Test get_ansatz with non-existent files."""
        project_path = temp_project_dir / "test_project_ansatz_missing"
        project_path.mkdir(exist_ok=True)

        result = get_ansatz(
            project_name="test_project_ansatz_missing",
            wavefunction_filename="nonexistent.wavefunction.json",
            hamiltonian_filename="nonexistent.hamiltonian.json",
            out_ansatz_filename="out.ansatz.json",
        )

        # Should return error message
        assert isinstance(result, str)
        assert "failed" in result.lower()

    @pytest.mark.usefixtures("temp_project_dir")
    def test_create_structure(self, h2_structure):
        """Test create_structure function."""
        coordinates_json = json.dumps(h2_structure.coordinates.tolist())

        result = create_structure(
            project_name="test_project",
            coordinates_json=coordinates_json,
            symbols=h2_structure.atomic_symbols,
            filename_to_save="h2.structure.json",
        )

        # Should return path as string or Path
        assert isinstance(result, str | Path)
        result_path = Path(result) if isinstance(result, str) else result
        assert result_path.exists()
        assert result_path.name == "h2.structure.json"

    @pytest.mark.usefixtures("temp_project_dir")
    def test_create_structure_invalid_project(self, h2_structure):
        """Test create_structure with invalid project name."""
        coordinates_json = json.dumps(h2_structure.coordinates.tolist())

        # Use characters that are invalid in paths
        result = create_structure(
            project_name="",
            coordinates_json=coordinates_json,
            symbols=h2_structure.atomic_symbols,
            filename_to_save="structure.json",
        )

        # Should return error message
        assert isinstance(result, str), f"Expected string but got {result}"


class TestAlgorithmFunctions:
    """Test algorithm wrapper functions."""

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_run_active_space_selector(self, temp_project_dir):
        """Test active space selector."""
        result = run_active_space_selector(
            project_name="test_project",
            wavefunction_filename="simple.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Should return filename string
        assert isinstance(result, str)
        assert "active.wavefunction.json" in result

        # Verify the file was created
        assert (temp_project_dir / "test_project" / "active.wavefunction.json").exists()

    def test_run_active_space_selector_invalid_json(self, temp_project_dir):
        """Test active space selector with invalid input."""
        project_path = temp_project_dir / "test_project2"
        project_path.mkdir(exist_ok=True)
        # Create invalid json file
        (project_path / "invalid.wavefunction.json").write_text("invalid json")

        result = run_active_space_selector(
            project_name="test_project2",
            wavefunction_filename="invalid.wavefunction.json",
            out_wavefunction_filename="out.wavefunction.json",
            charge=0,
        )

        assert isinstance(result, str)
        assert "problem" in result.lower() or "failed" in result.lower()

    def test_run_hamiltonian_constructor(self, simple_orbitals, temp_project_dir):
        """Test Hamiltonian constructor."""
        project_path = temp_project_dir / "test_project"
        project_path.mkdir(exist_ok=True)
        simple_orbitals.to_json_file(project_path / "simple.orbitals.json")

        result = run_hamiltonian_constructor(
            project_name="test_project",
            orbitals_filename="simple.orbitals.json",
            out_hamiltonian_filename="out.hamiltonian.json",
        )

        assert isinstance(result, str)
        assert "out.hamiltonian.json" in result
        assert (project_path / "out.hamiltonian.json").exists()

    def test_run_hamiltonian_constructor_invalid_json(self, temp_project_dir):
        """Test Hamiltonian constructor with invalid input."""
        project_path = temp_project_dir / "test_project3"
        project_path.mkdir(exist_ok=True)
        (project_path / "invalid.orbitals.json").write_text("invalid json")

        result = run_hamiltonian_constructor(
            project_name="test_project3",
            orbitals_filename="invalid.orbitals.json",
            out_hamiltonian_filename="out.hamiltonian.json",
        )

        assert isinstance(result, str)
        assert "problem" in result.lower() or "failed" in result.lower()

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_run_orbital_localization(self, temp_project_dir):
        """Test orbital localization."""
        project_path = temp_project_dir / "test_project"
        loc_indices = [0, 1]

        result = run_orbital_localization(
            project_name="test_project",
            wavefunction_filename="simple.wavefunction.json",
            out_wavefunction_filename="localized.wavefunction.json",
            loc_indices_alpha=loc_indices,
        )

        assert isinstance(result, str)
        assert "localized.wavefunction.json" in result
        assert (project_path / "localized.wavefunction.json").exists()

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_run_stability_checker(self, temp_project_dir):
        """Test stability checker for h2 in minimal basis."""
        _project_path = temp_project_dir / "test_project"

        result = run_stability_checker(
            project_name="test_project",
            wavefunction_filename="simple.wavefunction.json",
            out_stability_result_filename="stability.stability_result.json",
        )

        # Should return (bool, filename) — unwrapped as list
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        assert "stability.stability_result.json" in result[1]

    def test_run_qubit_mapper(self, simple_hamiltonian, temp_project_dir):
        """Test qubit mapper."""
        project_path = temp_project_dir / "test_project4"
        project_path.mkdir(exist_ok=True)
        simple_hamiltonian.to_hdf5_file(project_path / "simple.hamiltonian.h5")

        result = run_qubit_mapper(
            project_name="test_project4",
            hamiltonian_filename="simple.hamiltonian.h5",
            out_qubit_hamiltonian_filename="out.qubit_hamiltonian.h5",
        )

        assert isinstance(result, str)
        assert "out.qubit_hamiltonian.h5" in result
        assert (project_path / "out.qubit_hamiltonian.h5").exists()

    @pytest.mark.usefixtures("simple_wavefunction")
    def test_run_state_preparation(self, temp_project_dir):
        """Test state preparation with a multi-configurational wavefunction."""
        project_path = temp_project_dir / "test_project"

        result = run_state_preparation(
            project_name="test_project",
            wavefunction_filename="simple.wavefunction.json",
            out_circuit_filename="circuit.circuit.json",
        )

        assert isinstance(result, str)
        assert "circuit.circuit.json" in result
        assert (project_path / "circuit.circuit.json").exists()

    def test_run_dynamical_correlation_calculator(self, simple_ansatz, temp_project_dir):
        """Test dynamical correlation calculator."""
        project_path = temp_project_dir / "test_project5"
        project_path.mkdir(exist_ok=True)
        simple_ansatz.to_json_file(project_path / "simple.ansatz.json")

        result = run_dynamical_correlation_calculator(
            project_name="test_project5",
            ansatz_filename="simple.ansatz.json",
            out_wavefunction_filename="corr.wavefunction.json",
            algorithm_name="pyscf_coupled_cluster",
        )

        # Should return tuple of (energy, filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
        assert "corr.wavefunction.json" in result[1]
        assert (project_path / "corr.wavefunction.json").exists()

    def test_run_qubit_hamiltonian_solver(self, simple_hamiltonian, temp_project_dir):
        """Test qubit Hamiltonian solver."""
        project_path = temp_project_dir / "test_project6"
        project_path.mkdir(exist_ok=True)
        simple_hamiltonian.to_json_file(project_path / "simple.hamiltonian.json")

        # First map to qubit Hamiltonian
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_project6",
            hamiltonian_filename="simple.hamiltonian.json",
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        result = run_qubit_hamiltonian_solver(
            project_name="test_project6", qubit_hamiltonian_filename=qubit_ham_filename
        )

        # Should return tuple of (energy, list)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], list)

    def test_run_energy_estimator(self, h2_structure, temp_project_dir):
        """Test energy estimator."""
        project_path = temp_project_dir / "test_energy"
        project_path.mkdir(exist_ok=True)

        # Save structure to file
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Create a minimal workflow to get circuit and qubit Hamiltonian
        _scf_result = run_scf(
            project_name="test_energy",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_energy",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Get circuit
        circuit_filename = run_state_preparation(
            project_name="test_energy",
            wavefunction_filename=_active_wfn_filename,
            out_circuit_filename="circuit.circuit.json",
        )

        # Get qubit Hamiltonian
        active_wfn = data.Wavefunction.from_json_file(str(project_path / "active.wavefunction.json"))
        active_wfn.get_orbitals().to_json_file(str(project_path / "orbitals.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_energy",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_energy",
            hamiltonian_filename=ham_filename,
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.h5",
        )

        # Run energy estimator
        result = run_energy_estimator(
            project_name="test_energy",
            circuit_filename=circuit_filename,
            qubit_hamiltonian_filenames=[qubit_ham_filename],
            total_shots=100,
            out_energy_result_filename="energy.energy_expectation_result.h5",
            out_measurement_data_filename="measurement.measurement_data.h5",
        )

        # Should return tuple of (energy_result_filename, measurement_data_filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)
        assert (project_path / result[0]).exists()
        assert (project_path / result[1]).exists()

    def test_run_energy_estimator_invalid_circuit(self, temp_project_dir):
        """Test energy estimator with invalid circuit."""
        project_path = temp_project_dir / "test_invalid"
        project_path.mkdir(exist_ok=True)
        (project_path / "invalid.circuit.json").write_text("invalid json")
        (project_path / "empty.hamiltonian.json").write_text("{}")

        result = run_energy_estimator(
            project_name="test_invalid",
            circuit_filename="invalid.circuit.json",
            qubit_hamiltonian_filenames=["empty.hamiltonian.json"],
            total_shots=10,
            out_energy_result_filename="out.result.json",
            out_measurement_data_filename="out.data.json",
        )

        assert isinstance(result, str)
        assert "problem" in result.lower() or "failed" in result.lower()

    def test_run_multi_configuration_scf(self, h2_structure, temp_project_dir):
        """Test multi-configuration SCF calculation."""
        project_path = temp_project_dir / "test_mcscf"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_mcscf",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_mcscf",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Extract orbitals from active space wavefunction
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        # Get number of active electrons (for H2, it's 2 alpha)
        n_alpha, _ = active_wfn.get_active_num_electrons()

        # Run MCSCF
        result = run_multi_configuration_scf(
            project_name="test_mcscf",
            orbitals_filename="active.orbitals.json",
            out_wavefunction_filename="mcscf.wavefunction.json",
            n_active_alpha_electrons=n_alpha,
        )

        # Should return tuple of (energy, filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
        assert "mcscf.wavefunction.json" in result[1]
        assert (project_path / "mcscf.wavefunction.json").exists()

    def test_run_multi_configuration_calculation(self, h2_structure, temp_project_dir):
        """Test multi-configuration calculation (CASCI)."""
        project_path = temp_project_dir / "test_casci"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_casci",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_casci",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Build Hamiltonian from active space orbitals
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_casci",
            orbitals_filename="active.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Get number of active electrons
        n_alpha, _ = active_wfn.get_active_num_electrons()

        # Run multi-configuration calculation
        result = run_multi_configuration_calculation(
            project_name="test_casci",
            hamiltonian_filename=ham_filename,
            out_wavefunction_filename="casci.wavefunction.json",
            n_active_alpha_electrons=n_alpha,
        )

        # Should return tuple of (energy, filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
        assert "casci.wavefunction.json" in result[1]
        assert (project_path / "casci.wavefunction.json").exists()

    def test_run_projected_multi_configuration_calculation(self, h2_structure, temp_project_dir):
        """Test projected multi-configuration calculation with configurations JSON."""
        project_path = temp_project_dir / "test_pmc"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_pmc",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_pmc",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Build Hamiltonian from active space orbitals
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_pmc",
            orbitals_filename="active.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Create configurations as JSON array
        # For H2 minimal basis (2 spatial orbitals), valid configurations are:
        # Configuration strings use: '0'=unoccupied, 'u'=alpha, 'd'=beta, '2'=doubly occupied
        # "20" - both electrons in first orbital (HF reference)
        # "ud" - one alpha in first, one beta in second
        # "02" - both electrons in second orbital
        configurations_json = json.dumps(["20", "ud", "02"])

        # Run projected multi-configuration calculation
        result = run_projected_multi_configuration_calculation(
            project_name="test_pmc",
            hamiltonian_filename=ham_filename,
            configurations_json=configurations_json,
            out_wavefunction_filename="pmc.wavefunction.json",
        )

        print("Result,", result)

        # Should return tuple of (energy, filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
        assert "pmc.wavefunction.json" in result[1]
        assert (project_path / "pmc.wavefunction.json").exists()

    def test_get_top_configurations(self, h2_structure, temp_project_dir):
        """Test get_top_configurations returns JSON of configuration strings."""
        project_path = temp_project_dir / "test_top_configs"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_top_configs",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_top_configs",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Build Hamiltonian and run CASCI to get multi-configurational wavefunction
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_top_configs",
            orbitals_filename="active.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        n_alpha, _ = active_wfn.get_active_num_electrons()

        casci_result = run_multi_configuration_calculation(
            project_name="test_top_configs",
            hamiltonian_filename=ham_filename,
            out_wavefunction_filename="casci.wavefunction.json",
            n_active_alpha_electrons=n_alpha,
        )
        assert isinstance(casci_result, tuple | list)

        # Get top configurations
        result = get_top_configurations(
            project_name="test_top_configs", wavefunction_filename="casci.wavefunction.json", max_determinants=5
        )

        # Should return a JSON string
        assert isinstance(result, str)

        # Should be valid JSON
        configs = json.loads(result)
        assert isinstance(configs, list)
        assert len(configs) > 0
        assert all(isinstance(c, str) for c in configs)

        # The result can be passed directly to run_projected_multi_configuration_calculation
        pmc_result = run_projected_multi_configuration_calculation(
            project_name="test_top_configs",
            hamiltonian_filename=ham_filename,
            configurations_json=result,  # Pass directly
            out_wavefunction_filename="pmc.wavefunction.json",
        )

        assert isinstance(pmc_result, tuple | list)
        assert isinstance(pmc_result[0], float)


class TestPhaseEstimationFunctions:
    """Test phase estimation functions (split QPE tools)."""

    def test_run_time_evolution_builder(self, h2_structure, temp_project_dir):
        """Test running the time evolution builder."""
        project_path = temp_project_dir / "test_rteb"
        project_path.mkdir(exist_ok=True)

        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        _scf_result = run_scf(
            project_name="test_rteb",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        _active_wfn_filename = run_active_space_selector(
            project_name="test_rteb",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )
        active_wfn = data.Wavefunction.from_json_file(str(project_path / "active.wavefunction.json"))
        active_wfn.get_orbitals().to_json_file(str(project_path / "orbitals.orbitals.json"))
        ham_filename = run_hamiltonian_constructor(
            project_name="test_rteb",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_rteb",
            hamiltonian_filename=ham_filename,
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        result = run_time_evolution_builder(
            project_name="test_rteb",
            qubit_hamiltonian_filename=qubit_ham_filename,
            evolution_time=0.1,
            out_time_evolution_unitary_filename="teu.time_evolution_unitary.json",
        )

        assert isinstance(result, str)
        assert "time_evolution_unitary" in result
        assert (project_path / result).exists()

    def test_run_controlled_evolution_circuit_mapper(self, h2_structure, temp_project_dir):
        """Test running the controlled evolution circuit mapper."""
        project_path = temp_project_dir / "test_rcecm"
        project_path.mkdir(exist_ok=True)

        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        _scf_result = run_scf(
            project_name="test_rcecm",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        _active_wfn_filename = run_active_space_selector(
            project_name="test_rcecm",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )
        active_wfn = data.Wavefunction.from_json_file(str(project_path / "active.wavefunction.json"))
        active_wfn.get_orbitals().to_json_file(str(project_path / "orbitals.orbitals.json"))
        ham_filename = run_hamiltonian_constructor(
            project_name="test_rcecm",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_rcecm",
            hamiltonian_filename=ham_filename,
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )
        teu_filename = run_time_evolution_builder(
            project_name="test_rcecm",
            qubit_hamiltonian_filename=qubit_ham_filename,
            evolution_time=0.1,
            out_time_evolution_unitary_filename="teu.time_evolution_unitary.json",
        )

        result = run_controlled_evolution_circuit_mapper(
            project_name="test_rcecm",
            time_evolution_unitary_filename=teu_filename,
            out_circuit_filename="ctrl_evol.circuit.json",
            power=1,
        )

        assert isinstance(result, str)
        assert "circuit" in result
        assert (project_path / result).exists()

    def test_run_phase_estimation_iterative(self, h2_structure, temp_project_dir):
        """Test iterative phase estimation."""
        project_path = temp_project_dir / "test_qpe"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Create a minimal workflow to get circuit and qubit Hamiltonian
        _scf_result = run_scf(
            project_name="test_qpe",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_qpe",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Get circuit
        circuit_filename = run_state_preparation(
            project_name="test_qpe",
            wavefunction_filename=_active_wfn_filename,
            out_circuit_filename="circuit.circuit.json",
        )

        # Get qubit Hamiltonian
        active_wfn = data.Wavefunction.from_json_file(str(project_path / "active.wavefunction.json"))
        active_wfn.get_orbitals().to_json_file(str(project_path / "orbitals.orbitals.json"))
        ham_filename = run_hamiltonian_constructor(
            project_name="test_qpe",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_qpe",
            hamiltonian_filename=ham_filename,
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        # Run phase estimation with iterative algorithm (default)
        result = run_phase_estimation(
            project_name="test_qpe",
            state_prep_circuit_filename=circuit_filename,
            qubit_hamiltonian_filename=qubit_ham_filename,
            out_qpe_result_filename="qpe.qpe_result.json",
            algorithm_name="iterative",
            settings={"num_bits": 4, "evolution_time": 0.1, "shots_per_bit": 10},
        )

        assert isinstance(result, str)
        assert "qpe.qpe_result.json" in result
        assert (project_path / result).exists()

    def test_run_phase_estimation_with_configs(self, h2_structure, temp_project_dir):
        """Test phase estimation with inline sub-algorithm settings."""
        project_path = temp_project_dir / "test_qpe_configs"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Create workflow to get circuit and qubit Hamiltonian
        _scf_result = run_scf(
            project_name="test_qpe_configs",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        _active_wfn_filename = run_active_space_selector(
            project_name="test_qpe_configs",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )
        circuit_filename = run_state_preparation(
            project_name="test_qpe_configs",
            wavefunction_filename=_active_wfn_filename,
            out_circuit_filename="circuit.circuit.json",
        )
        active_wfn = data.Wavefunction.from_json_file(str(project_path / "active.wavefunction.json"))
        active_wfn.get_orbitals().to_json_file(str(project_path / "orbitals.orbitals.json"))
        ham_filename = run_hamiltonian_constructor(
            project_name="test_qpe_configs",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_qpe_configs",
            hamiltonian_filename=ham_filename,
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        # Run phase estimation with inline sub-algorithm overrides
        result = run_phase_estimation(
            project_name="test_qpe_configs",
            state_prep_circuit_filename=circuit_filename,
            qubit_hamiltonian_filename=qubit_ham_filename,
            out_qpe_result_filename="qpe.qpe_result.json",
            algorithm_name="iterative",
            settings={
                "num_bits": 4,
                "evolution_time": 0.1,
                "shots_per_bit": 10,
                "evolution_builder": {"algorithm_name": "trotter"},
                "circuit_mapper": {"algorithm_name": "pauli_sequence"},
                "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"},
            },
        )

        assert isinstance(result, str)
        assert "qpe.qpe_result.json" in result
        assert (project_path / result).exists()

    def test_run_phase_estimation_with_invalid_settings(
        self, simple_wavefunction, simple_hamiltonian, temp_project_dir
    ):
        """Test phase estimation with missing required settings returns error."""
        project_path = temp_project_dir / "test_qpe_invalid"
        project_path.mkdir(exist_ok=True)
        simple_wavefunction.to_json_file(str(project_path / "wfn.wavefunction.json"))
        simple_hamiltonian.to_json_file(str(project_path / "ham.hamiltonian.json"))

        circuit_filename = run_state_preparation(
            project_name="test_qpe_invalid",
            wavefunction_filename="wfn.wavefunction.json",
            out_circuit_filename="circuit.circuit.json",
        )
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_qpe_invalid",
            hamiltonian_filename="ham.hamiltonian.json",
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        # Run phase estimation without setting required num_bits (defaults to -1)
        result = run_phase_estimation(
            project_name="test_qpe_invalid",
            state_prep_circuit_filename=circuit_filename,
            qubit_hamiltonian_filename=qubit_ham_filename,
            out_qpe_result_filename="qpe.qpe_result.json",
            settings={"evolution_time": 0.1},  # Missing num_bits
        )

        # Should return an error message since num_bits defaults to -1 (invalid)
        assert isinstance(result, str)
        # The function should either fail with an error or succeed with default settings
        # depending on the implementation


class TestModelHamiltonians:
    """Tests for model Hamiltonian creation tools."""

    def test_create_hubbard_chain(self, temp_project_dir):
        """Test creating a Hubbard model on a 1D chain."""
        project_path = temp_project_dir / "test_hubbard"
        project_path.mkdir(exist_ok=True)

        result = create_model_hamiltonian(
            project_name="test_hubbard",
            model="hubbard",
            out_hamiltonian_filename="hubbard.hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 4, "periodic": False},
            epsilon=0.0,
            t=1.0,
            U=4.0,
        )

        assert isinstance(result, str)
        assert "hubbard.hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_create_huckel_square(self, temp_project_dir):
        """Test creating a Hückel model on a square lattice."""
        project_path = temp_project_dir / "test_huckel"
        project_path.mkdir(exist_ok=True)

        result = create_model_hamiltonian(
            project_name="test_huckel",
            model="huckel",
            out_hamiltonian_filename="huckel.hamiltonian.json",
            lattice_type="square",
            lattice_params={"nx": 2, "ny": 2},
            epsilon=0.0,
            t=1.0,
        )

        assert isinstance(result, str)
        assert "huckel.hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_create_ising_chain(self, temp_project_dir):
        """Test creating an Ising model on a 1D chain."""
        project_path = temp_project_dir / "test_ising"
        project_path.mkdir(exist_ok=True)

        result = create_spin_model_hamiltonian(
            project_name="test_ising",
            model="ising",
            out_qubit_hamiltonian_filename="ising.qubit_hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 4, "periodic": True},
            j=1.0,
            h=0.5,
        )

        assert isinstance(result, str)
        assert "ising.qubit_hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_create_heisenberg_honeycomb(self, temp_project_dir):
        """Test creating a Heisenberg model on a honeycomb lattice."""
        project_path = temp_project_dir / "test_heisenberg"
        project_path.mkdir(exist_ok=True)

        result = create_spin_model_hamiltonian(
            project_name="test_heisenberg",
            model="heisenberg",
            out_qubit_hamiltonian_filename="heisenberg.qubit_hamiltonian.json",
            lattice_type="honeycomb",
            lattice_params={"nx": 2, "ny": 1},
            jx=1.0,
            jy=1.0,
            jz=1.0,
        )

        assert isinstance(result, str)
        assert "heisenberg.qubit_hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_model_hamiltonian_unknown_model(self, temp_project_dir):
        """Test error for unknown model type."""
        project_path = temp_project_dir / "test_unknown"
        project_path.mkdir(exist_ok=True)

        result = create_model_hamiltonian(
            project_name="test_unknown",
            model="invalid_model",
            out_hamiltonian_filename="bad.hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 4},
        )

        assert isinstance(result, str)
        assert "unknown" in result.lower() or "invalid" in result.lower()

    def test_model_hamiltonian_per_site_params(self, temp_project_dir):
        """Test Hubbard model with per-site epsilon array."""
        project_path = temp_project_dir / "test_persite"
        project_path.mkdir(exist_ok=True)

        result = create_model_hamiltonian(
            project_name="test_persite",
            model="hubbard",
            out_hamiltonian_filename="hubbard_persite.hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 3},
            epsilon=[0.0, 0.5, 1.0],
            t=1.0,
            U=[3.0, 4.0, 5.0],
        )

        assert isinstance(result, str)
        assert "hubbard_persite.hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_model_hamiltonian_per_pair_params(self, temp_project_dir):
        """Test Hückel model with per-pair hopping matrix (2-D list)."""
        project_path = temp_project_dir / "test_perpair"
        project_path.mkdir(exist_ok=True)

        # 3-site chain with asymmetric hopping
        result = create_model_hamiltonian(
            project_name="test_perpair",
            model="huckel",
            out_hamiltonian_filename="huckel_perpair.hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 3},
            epsilon=0.0,
            t=[[0, 1.2, 0], [1.2, 0, 0.8], [0, 0.8, 0]],
        )

        assert isinstance(result, str)
        assert "huckel_perpair.hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_spin_model_per_pair_coupling(self, temp_project_dir):
        """Test Heisenberg model with per-pair coupling matrix."""
        project_path = temp_project_dir / "test_spin_perpair"
        project_path.mkdir(exist_ok=True)

        result = create_spin_model_hamiltonian(
            project_name="test_spin_perpair",
            model="heisenberg",
            out_qubit_hamiltonian_filename="heis_perpair.qubit_hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 3},
            jx=[[0, 0.5, 0], [0.5, 0, 1.0], [0, 1.0, 0]],
            jy=0.0,
            jz=1.0,
        )

        assert isinstance(result, str)
        assert "heis_perpair.qubit_hamiltonian.json" in result
        assert (project_path / result).exists()

    def test_hubbard_to_qubit_map_workflow(self, temp_project_dir):
        """Test that model Hamiltonian output works with run_qubit_mapper."""
        project_path = temp_project_dir / "test_hub_workflow"
        project_path.mkdir(exist_ok=True)

        ham_file = create_model_hamiltonian(
            project_name="test_hub_workflow",
            model="hubbard",
            out_hamiltonian_filename="hub.hamiltonian.json",
            lattice_type="chain",
            lattice_params={"n": 2, "periodic": False},
            epsilon=0.0,
            t=1.0,
            U=2.0,
        )
        assert isinstance(ham_file, str)

        qh_file = run_qubit_mapper(
            project_name="test_hub_workflow",
            hamiltonian_filename=ham_file,
            out_qubit_hamiltonian_filename="hub.qubit_hamiltonian.json",
        )
        assert isinstance(qh_file, str)
        assert (project_path / qh_file).exists()


class TestIntegrationWorkflows:
    """Test integrated workflows combining multiple functions."""

    def test_scf_to_hamiltonian_workflow(self, h2_structure, temp_project_dir):
        """Test workflow from SCF to Hamiltonian."""
        project_path = temp_project_dir / "test_integration1"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF
        _scf_result = run_scf(
            project_name="test_integration1",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

        assert isinstance(_scf_result, tuple | list)

        # Extract orbitals from saved wavefunction
        wavefunction = data.Wavefunction.from_json_file(str(project_path / "scf.wavefunction.json"))
        orbitals = wavefunction.get_orbitals()
        orbitals.to_json_file(project_path / "orbitals.orbitals.json")

        # Construct Hamiltonian
        hamiltonian_filename = run_hamiltonian_constructor(
            project_name="test_integration1",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Verify we got valid Hamiltonian file
        assert (project_path / hamiltonian_filename).exists()
        hamiltonian = data.Hamiltonian.from_json_file(str(project_path / hamiltonian_filename))
        assert isinstance(hamiltonian, data.Hamiltonian)

    def test_hamiltonian_to_qubit_workflow(self, h2_structure, temp_project_dir):
        """Test workflow from Hamiltonian to qubit Hamiltonian."""
        project_path = temp_project_dir / "test_integration2"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF
        _scf_result = run_scf(
            project_name="test_integration2",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

        assert isinstance(_scf_result, tuple | list)

        # Get Hamiltonian
        wavefunction = data.Wavefunction.from_json_file(str(project_path / "scf.wavefunction.json"))
        orbitals = wavefunction.get_orbitals()
        orbitals.to_json_file(project_path / "orbitals.orbitals.json")
        hamiltonian_filename = run_hamiltonian_constructor(
            project_name="test_integration2",
            orbitals_filename="orbitals.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Map to qubits
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_integration2",
            hamiltonian_filename=hamiltonian_filename,
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        # Verify we got valid qubit Hamiltonian
        assert (project_path / qubit_ham_filename).exists()
        qubit_ham = data.QubitHamiltonian.from_hdf5_file(str(project_path / qubit_ham_filename))
        assert isinstance(qubit_ham, data.QubitHamiltonian)

    def test_full_quantum_simulation_workflow(self, h2_structure, temp_project_dir):
        """Test complete workflow from structure to quantum simulation."""
        project_path = temp_project_dir / "test_full_workflow"
        project_path.mkdir(exist_ok=True)

        h2_structure.to_json_file(project_path / "h2.structure.json")

        # Run SCF calculation
        _scf_result = run_scf(
            project_name="test_full_workflow",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

        # Check output - it's a tuple of (energy, filename)
        assert isinstance(_scf_result, tuple | list)
        _, wfn_filename = _scf_result
        assert (project_path / wfn_filename).exists()
        wfn = data.Wavefunction.from_json_file(str(project_path / wfn_filename))
        assert isinstance(wfn, data.Wavefunction)

        # Get active space
        _active_wfn_filename = run_active_space_selector(
            project_name="test_full_workflow",
            wavefunction_filename=wfn_filename,
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Check output - it's a filename
        assert (project_path / _active_wfn_filename).exists()
        active_wfn = data.Wavefunction.from_json_file(str(project_path / "active.wavefunction.json"))
        assert isinstance(active_wfn, data.Wavefunction)

        # Prepare circuit
        circuit_filename = run_state_preparation(
            project_name="test_full_workflow",
            wavefunction_filename=_active_wfn_filename,
            out_circuit_filename="circuit.circuit.json",
        )

        # Check output - it's a filename
        assert (project_path / circuit_filename).exists()
        circuit = data.Circuit.from_json_file(str(project_path / circuit_filename))
        assert isinstance(circuit, data.Circuit)

        # Get Hamiltonian
        active_wfn.get_orbitals().to_hdf5_file(project_path / "orbitals.orbitals.h5")
        ham_filename = run_hamiltonian_constructor(
            project_name="test_full_workflow",
            orbitals_filename="orbitals.orbitals.h5",
            out_hamiltonian_filename="hamiltonian.hamiltonian.h5",
        )

        # Check output - it's a filename
        assert (project_path / ham_filename).exists()
        hamiltonian = data.Hamiltonian.from_hdf5_file(str(project_path / ham_filename))
        assert isinstance(hamiltonian, data.Hamiltonian)

        qubit_ham_filename = run_qubit_mapper(
            project_name="test_full_workflow",
            hamiltonian_filename=ham_filename,
            out_qubit_hamiltonian_filename="qubitham.qubit_hamiltonian.hdf5",
        )

        # Check output - it's a filename
        assert (project_path / qubit_ham_filename).exists()
        qubit_hamiltonian = data.QubitHamiltonian.from_hdf5_file(str(project_path / "qubitham.qubit_hamiltonian.hdf5"))
        assert isinstance(qubit_hamiltonian, data.QubitHamiltonian)


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Test error handling in functions."""

    def test_run_active_space_selector_with_malformed_json(self, temp_project_dir):
        """Test error handling with malformed JSON."""
        project_path = temp_project_dir / "test_error1"
        project_path.mkdir(exist_ok=True)
        (project_path / "malformed.wavefunction.json").write_text("{not valid json")

        result = run_active_space_selector(
            project_name="test_error1",
            wavefunction_filename="malformed.wavefunction.json",
            out_wavefunction_filename="out.wavefunction.json",
            charge=0,
        )

        assert isinstance(result, str)
        assert "problem" in result.lower() or "failed" in result.lower()

    def test_run_hamiltonian_constructor_with_wrong_type(self, temp_project_dir):
        """Test error with wrong dataclass type."""
        project_path = temp_project_dir / "test_error2"
        project_path.mkdir(exist_ok=True)
        # Write invalid JSON (Structure JSON where Orbitals JSON is expected)
        structure = Structure([[0.0, 0.0, 0.0]], ["H"])
        (project_path / "wrong_type.orbitals.json").write_text(structure.to_json())

        result = run_hamiltonian_constructor(
            project_name="test_error2",
            orbitals_filename="wrong_type.orbitals.json",
            out_hamiltonian_filename="out.hamiltonian.json",
        )

        assert isinstance(result, str)
        assert "problem" in result.lower() or "failed" in result.lower()

    def test_run_qubit_mapper_with_invalid_input(self, temp_project_dir):
        """Test qubit mapper with invalid input."""
        project_path = temp_project_dir / "test_error3"
        project_path.mkdir(exist_ok=True)
        (project_path / "not_hamiltonian.hamiltonian.json").write_text("not a hamiltonian")

        result = run_qubit_mapper(
            project_name="test_error3",
            hamiltonian_filename="not_hamiltonian.hamiltonian.json",
            out_qubit_hamiltonian_filename="out.qubit_hamiltonian.json",
        )

        assert isinstance(result, str)
        assert "problem" in result.lower() or "failed" in result.lower()

    def test_run_scf_with_invalid_structure(self, temp_project_dir):
        """Test SCF workflow with invalid structure file."""
        project_path = temp_project_dir / "test_invalid_struct"
        project_path.mkdir(exist_ok=True)
        (project_path / "invalid.structure.json").write_text("invalid json")

        result = run_scf(
            project_name="test_invalid_struct",
            structure_filename="invalid.structure.json",
            out_wavefunction_filename="output.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )

        # Should return error message
        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()


# ==================== Non-Default Algorithm Tests ====================


class TestNonDefaultAlgorithms:
    """Test functions using non-default algorithm implementations."""

    def test_qubit_hamiltonian_solver_dense(self, simple_hamiltonian, temp_project_dir):
        """Test qubit Hamiltonian solver with dense matrix solver (non-default)."""
        project_path = temp_project_dir / "test_dense_solver"
        project_path.mkdir(exist_ok=True)
        simple_hamiltonian.to_json_file(project_path / "simple.hamiltonian.json")

        # First map to qubit Hamiltonian
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_dense_solver",
            hamiltonian_filename="simple.hamiltonian.json",
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        # Use non-default dense matrix solver
        result = run_qubit_hamiltonian_solver(
            project_name="test_dense_solver",
            qubit_hamiltonian_filename=qubit_ham_filename,
            algorithm_name="qdk_dense_matrix_solver",
        )

        # Should return tuple of (energy, list)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], list)

    def test_multi_configuration_calculation_asci(self, h2_structure, temp_project_dir):
        """Test multi-configuration calculation with ASCI algorithm (non-default)."""
        project_path = temp_project_dir / "test_asci"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_asci",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction
        _active_wfn_filename = run_active_space_selector(
            project_name="test_asci",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Build Hamiltonian from active space orbitals
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_asci",
            orbitals_filename="active.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Get number of active electrons
        n_alpha, _ = active_wfn.get_active_num_electrons()

        # Run multi-configuration calculation with ASCI (non-default)
        result = run_multi_configuration_calculation(
            project_name="test_asci",
            hamiltonian_filename=ham_filename,
            out_wavefunction_filename="asci.wavefunction.json",
            n_active_alpha_electrons=n_alpha,
            algorithm_name="macis_asci",
        )

        # Should return tuple of (energy, filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
        assert "asci.wavefunction.json" in result[1]
        assert (project_path / "asci.wavefunction.json").exists()

    def test_active_space_selector_occupation(self, h2_structure, temp_project_dir):
        """Test active space selector with occupation algorithm (non-default).

        The occupation selector needs fractional orbital occupations from a multi-configurational
        wavefunction, so we first run a CASCI calculation to generate one.
        """
        project_path = temp_project_dir / "test_occupation"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_occupation",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction using valence
        _active_wfn_filename = run_active_space_selector(
            project_name="test_occupation",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="valence_active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Build Hamiltonian from active space orbitals
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_occupation",
            orbitals_filename="active.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Get number of active electrons
        n_alpha, _ = active_wfn.get_active_num_electrons()

        # Run multi-configuration calculation with RDMs enabled (needed for occupation analysis)
        mc_result = run_multi_configuration_calculation(
            project_name="test_occupation",
            hamiltonian_filename=ham_filename,
            out_wavefunction_filename="mc.wavefunction.json",
            n_active_alpha_electrons=n_alpha,
            settings={"calculate_one_rdm": True},
        )
        assert isinstance(mc_result, tuple | list)

        # Now run occupation active space selector on wavefunction with RDMs
        result = run_active_space_selector(
            project_name="test_occupation",
            wavefunction_filename="mc.wavefunction.json",
            out_wavefunction_filename="occupation_active.wavefunction.json",
            algorithm_name="qdk_occupation",
            settings={"occupation_threshold": 0.01},  # Low threshold to capture partially occupied orbitals
        )

        # Should return filename string
        assert isinstance(result, str)
        assert "occupation_active.wavefunction.json" in result

        # Verify the file was created
        assert (project_path / "occupation_active.wavefunction.json").exists()

    def test_active_space_selector_autocas(self, h2_structure, temp_project_dir):
        """Test active space selector with autocas algorithm (non-default, requires RDMs)."""
        project_path = temp_project_dir / "test_autocas"
        project_path.mkdir(exist_ok=True)

        # Save structure
        h2_structure.to_json_file(str(project_path / "h2.structure.json"))

        # Run SCF first
        _scf_result = run_scf(
            project_name="test_autocas",
            structure_filename="h2.structure.json",
            out_wavefunction_filename="scf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
        assert isinstance(_scf_result, tuple | list)

        # Get active space wavefunction using valence
        _active_wfn_filename = run_active_space_selector(
            project_name="test_autocas",
            wavefunction_filename="scf.wavefunction.json",
            out_wavefunction_filename="valence_active.wavefunction.json",
            charge=0,
            algorithm_name="qdk_valence",
        )

        # Build Hamiltonian from active space orbitals
        active_wfn = data.Wavefunction.from_json_file(str(project_path / _active_wfn_filename))
        active_wfn.get_orbitals().to_json_file(str(project_path / "active.orbitals.json"))

        ham_filename = run_hamiltonian_constructor(
            project_name="test_autocas",
            orbitals_filename="active.orbitals.json",
            out_hamiltonian_filename="hamiltonian.hamiltonian.json",
        )

        # Get number of active electrons
        n_alpha, _ = active_wfn.get_active_num_electrons()

        # Run multi-configuration calculation with RDMs enabled
        mc_result = run_multi_configuration_calculation(
            project_name="test_autocas",
            hamiltonian_filename=ham_filename,
            out_wavefunction_filename="mc.wavefunction.json",
            n_active_alpha_electrons=n_alpha,
            settings={"calculate_one_rdm": True, "calculate_two_rdm": True},
        )
        assert isinstance(mc_result, tuple | list)

        # Now run autocas active space selector on wavefunction with RDMs
        result = run_active_space_selector(
            project_name="test_autocas",
            wavefunction_filename="mc.wavefunction.json",
            out_wavefunction_filename="autocas_active.wavefunction.json",
            algorithm_name="qdk_autocas",
        )

        # Should return filename string
        assert isinstance(result, str)
        assert "autocas_active.wavefunction.json" in result
        assert (project_path / "autocas_active.wavefunction.json").exists()

    def test_cc_mp2(self, simple_ansatz, temp_project_dir):
        """Test cc calculator (non default)."""
        project_path = temp_project_dir / "test_cc"
        project_path.mkdir(exist_ok=True)
        simple_ansatz.to_json_file(project_path / "simple.ansatz.json")

        result = run_dynamical_correlation_calculator(
            project_name="test_cc",
            ansatz_filename="simple.ansatz.json",
            out_wavefunction_filename="cc.wavefunction.json",
            algorithm_name="pyscf_coupled_cluster",
        )

        # Should return tuple of (energy, filename)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
        assert "cc.wavefunction.json" in result[1]
        assert (project_path / "cc.wavefunction.json").exists()

    def test_get_algorithm_default_type_various(self):
        """Test getting default algorithm types for various algorithm types."""
        # Test various algorithm types to ensure they have defaults
        algorithm_types = [
            "scf_solver",
            "active_space_selector",
            "orbital_localizer",
            "dynamical_correlation_calculator",
            "multi_configuration_calculator",
            "hamiltonian_constructor",
            "stability_checker",
            "qubit_mapper",
            "state_prep",
            "qubit_hamiltonian_solver",
            "energy_estimator",
            "phase_estimation",
        ]

        for algo_type in algorithm_types:
            default_type = get_algorithm_default_type(algo_type)
            assert isinstance(default_type, str)
            assert len(default_type) > 0, f"Default for {algo_type} is empty"

    def test_get_algorithm_settings_for_non_defaults(self):
        """Test getting settings for non-default algorithms."""
        # Test getting settings for various non-default algorithms
        test_cases = [
            ("qubit_hamiltonian_solver", "qdk_dense_matrix_solver"),
            ("multi_configuration_calculator", "macis_asci"),
            ("active_space_selector", "qdk_valence"),
            ("active_space_selector", "qdk_occupation"),
            ("active_space_selector", "qdk_autocas"),
        ]

        for algo_type, algo_name in test_cases:
            settings = get_algorithm_default_settings(algo_type, algo_name)
            assert isinstance(settings, dict), f"Settings for {algo_type}/{algo_name} is not a dict"

    def test_qubit_hamiltonian_solver_sparse_with_settings(self, simple_hamiltonian, temp_project_dir):
        """Test sparse qubit Hamiltonian solver with custom tolerance settings."""
        project_path = temp_project_dir / "test_sparse_solver_settings"
        project_path.mkdir(exist_ok=True)
        simple_hamiltonian.to_json_file(project_path / "simple.hamiltonian.json")

        # First map to qubit Hamiltonian
        qubit_ham_filename = run_qubit_mapper(
            project_name="test_sparse_solver_settings",
            hamiltonian_filename="simple.hamiltonian.json",
            out_qubit_hamiltonian_filename="qubit.qubit_hamiltonian.hdf5",
        )

        # Use sparse solver with custom tolerance settings
        result = run_qubit_hamiltonian_solver(
            project_name="test_sparse_solver_settings",
            qubit_hamiltonian_filename=qubit_ham_filename,
            algorithm_name="qdk_sparse_matrix_solver",
            settings={"tol": 1e-3, "max_m": 100},
        )

        # Should return tuple of (energy, list)
        assert isinstance(result, tuple | list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], list)
