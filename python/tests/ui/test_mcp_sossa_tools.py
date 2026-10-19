"""MCP tests for the double-factorization, SOSSA block-encoding, QPE-builder, and QRE tools."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry import data
from qdk_chemistry.ui import tools as srv
from qdk_chemistry.ui.config import config

_SOSSA_MAPPER = {
    "algorithm_name": "sossa",
    "outer_prepare_algorithm": {"algorithm_name": "alias_sampling"},
    "inner_prepare_algorithm": "controlled_alias_sampling",
    "select_algorithm": "qrom_phase_gradient",
    "rotation_bit_precision": 8,
    "coefficient_bit_precision": 6,
}


def _ok(result):
    assert result["status"] == "ok", result
    return result["result"]


@pytest.fixture
def project():
    """Create an isolated project containing an H2 structure."""
    with tempfile.TemporaryDirectory() as t:
        original = config.projects_dir, config.jobs_dir, config.cache_dir
        config.projects_dir = Path(t) / "projects"
        config.jobs_dir = Path(t) / "jobs"
        config.cache_dir = Path(t) / "cache"
        for directory in (config.projects_dir, config.jobs_dir, config.cache_dir):
            directory.mkdir()
        project_dir = config.projects_dir / "h2"
        project_dir.mkdir()
        structure = data.Structure(coordinates=np.array([[0, 0, 0], [0, 0, 1.4]]), symbols=["H", "H"])
        structure.to_json_file(str(project_dir / "h2.structure.json"))
        yield "h2"
        config.projects_dir, config.jobs_dir, config.cache_dir = original


@pytest.fixture
def sos_project(project):
    """Run H2 through SCF, Cholesky, double factorization, and the sum-of-squares mapper."""
    _ok(
        srv.run_scf(
            project_name=project,
            structure_filename="h2.structure.json",
            out_wavefunction_filename="hf.wavefunction.json",
            charge=0,
            spin_multiplicity=1,
            basis_set="sto-3g",
        )
    )
    _ok(
        srv.get_orbitals_from_input(
            project_name=project, input_filename="hf.wavefunction.json", out_orbitals_filename="hf.orbitals.json"
        )
    )
    _ok(
        srv.run_hamiltonian_constructor(
            project_name=project,
            orbitals_filename="hf.orbitals.json",
            out_hamiltonian_filename="chol.hamiltonian.json",
            algorithm_name="qdk_cholesky",
            settings={"cholesky_tolerance": 1e-10},
        )
    )
    _ok(
        srv.run_hamiltonian_factorization(
            project_name=project,
            hamiltonian_filename="chol.hamiltonian.json",
            out_hamiltonian_filename="df.hamiltonian.json",
            algorithm_name="double_factorization",
        )
    )
    energy, _ = _ok(
        srv.run_multi_configuration_calculation(
            project_name=project,
            hamiltonian_filename="df.hamiltonian.json",
            out_wavefunction_filename="cas.wavefunction.json",
            n_active_alpha_electrons=1,
            n_active_beta_electrons=1,
            algorithm_name="macis_cas",
        )
    )
    _ok(
        srv.create_majorana_mapping(
            project_name=project, out_mapping_filename="jw.majorana_mapping.json", hamiltonian_filename="df.hamiltonian.json"
        )
    )
    mapped = _ok(
        srv.run_qubit_mapper(
            project_name=project,
            hamiltonian_filename="df.hamiltonian.json",
            mapping_filename="jw.majorana_mapping.json",
            out_qubit_hamiltonian_filename="sos.qubit_operator.json",
            algorithm_name="sum_of_squares",
        )
    )
    _ok(
        srv.run_state_preparation(
            project_name=project,
            wavefunction_filename="hf.wavefunction.json",
            out_circuit_filename="hf.circuit.json",
            algorithm_name="sparse_isometry",
        )
    )
    return project, mapped["qubit_hamiltonian_filename"], energy


def test_hamiltonian_constructor_selects_cholesky_and_factorization_preserves_energy(sos_project):
    """The Cholesky and double-factorized Hamiltonians reproduce the dense-integral CASCI energy."""
    project, _, df_energy = sos_project
    _ok(
        srv.get_orbitals_from_input(
            project_name=project, input_filename="hf.wavefunction.json", out_orbitals_filename="dense.orbitals.json"
        )
    )
    _ok(
        srv.run_hamiltonian_constructor(
            project_name=project, orbitals_filename="dense.orbitals.json", out_hamiltonian_filename="dense.hamiltonian.json"
        )
    )
    dense_energy, _ = _ok(
        srv.run_multi_configuration_calculation(
            project_name=project,
            hamiltonian_filename="dense.hamiltonian.json",
            out_wavefunction_filename="dense.wavefunction.json",
            n_active_alpha_electrons=1,
            n_active_beta_electrons=1,
            algorithm_name="macis_cas",
        )
    )
    assert df_energy == pytest.approx(dense_energy, abs=1e-8)


def test_block_encoding_builder_reports_sossa_normalization_and_query_count(sos_project):
    """SOSSA reports its normalization, energy shift, lambda_eff, and the Heisenberg-limited query count."""
    project, sos_file, energy = sos_project

    plain = _ok(
        srv.run_block_encoding_builder(
            project_name=project,
            qubit_hamiltonian_filename=sos_file,
            out_unitary_filename="plain.unitary_representation.json",
            algorithm_name="sossa",
            target_energy_error=1e-3,
        )
    )
    assert plain["container_type"] == "sossa_block_encoding"
    assert plain["normalization"] > 0
    assert "energy_shift" in plain
    assert "lambda_eff" not in plain
    assert plain["suggested_num_queries"] == math.ceil(math.pi * plain["normalization"] / 2e-3)

    amplified = _ok(
        srv.run_block_encoding_builder(
            project_name=project,
            qubit_hamiltonian_filename=sos_file,
            out_unitary_filename="sossa.unitary_representation.json",
            algorithm_name="sossa",
            settings={"reference_ground_state_energy": energy},
            target_energy_error=1e-3,
        )
    )
    assert 0 < amplified["lambda_eff"] < amplified["normalization"]
    assert amplified["suggested_num_queries"] == math.ceil(math.pi * amplified["lambda_eff"] / 2e-3)
    saved = config.projects_dir / project / amplified["unitary_filename"]
    assert data.UnitaryRepresentation.from_json_file(str(saved)).get_container().type == "sossa_block_encoding"


def test_block_encoding_builder_rejects_nonpositive_target_error(sos_project):
    """A non-positive target energy error is rejected before building."""
    project, sos_file, _ = sos_project
    result = srv.run_block_encoding_builder(
        project_name=project,
        qubit_hamiltonian_filename=sos_file,
        out_unitary_filename="bad.unitary_representation.json",
        algorithm_name="sossa",
        target_energy_error=0.0,
    )
    assert result["status"] == "error"
    assert "target_energy_error" in result["message"]


def test_qpe_circuit_builder_saves_unary_sossa_circuit_and_qre_estimates_it(sos_project):
    """The unary SOSSA QPE circuit is built without execution, saved, and estimated with qdk.qre."""
    pytest.importorskip("qdk.qre")
    project, sos_file, _ = sos_project

    built = _ok(
        srv.run_qpe_circuit_builder(
            project_name=project,
            state_prep_circuit_filename="hf.circuit.json",
            qubit_hamiltonian_filename=sos_file,
            out_circuit_filename="qpe.circuit.json",
            algorithm_name="qdk_unary",
            settings={
                "num_queries": 3,
                "unitary_builder": {"algorithm_name": "sossa"},
                "circuit_mapper": _SOSSA_MAPPER,
            },
        )
    )
    assert built["circuit_filenames"] == ["qpe.circuit.json"]
    assert built["num_qubits"][0] > 4

    estimate = _ok(
        srv.estimate_circuit_qre(
            project_name=project, circuit_filename="qpe.circuit.json", error_rate=1e-6, max_error=0.01
        )
    )
    assert estimate["architecture"] == "majorana"
    assert estimate["num_estimates"] == len(estimate["estimates"]) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"architecture": "superconducting"}, "Unsupported architecture"),
        ({"error_rate": 0.0}, "error_rate"),
        ({"max_error": 1.0}, "max_error"),
    ],
)
def test_estimate_circuit_qre_validates_arguments(project, kwargs, message):
    """Invalid architecture and error arguments return an error envelope."""
    result = srv.estimate_circuit_qre(project_name=project, circuit_filename="missing.circuit.json", **kwargs)
    assert result["status"] == "error"
    assert message in result["message"]
