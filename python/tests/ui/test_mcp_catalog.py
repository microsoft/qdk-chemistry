"""MCP catalog transport tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import asyncio

from qdk_chemistry.ui.mcp import (
    _compact_tool_description,
    _install_tool_description_compactor,
    _parse_args,
)
from qdk_chemistry.ui.tools import app

_REVIEWED_TOOL_NAMES = {
    "bind_workspace",
    "cancel_remote_job",
    "check_remote_job",
    "convert_coordinates",
    "convert_energy",
    "create_majorana_mapping",
    "create_model_hamiltonian",
    "create_project",
    "create_spin_model_hamiltonian",
    "create_structure",
    "describe_algorithm",
    "describe_backend",
    "get_active_space_indices",
    "get_algorithm_default_settings",
    "get_algorithm_default_type",
    "get_ansatz",
    "get_circuit_stats",
    "get_orbitals_from_input",
    "get_summary",
    "get_top_configurations",
    "get_top_determinants",
    "list_algorithms",
    "list_cache_backends",
    "list_project_files",
    "list_projects",
    "list_remote_backends",
    "list_remote_jobs",
    "list_tools",
    "retrieve_remote_results",
    "run_active_space_selector",
    "run_amplitude_amplification",
    "run_circuit_executor",
    "run_controlled_evolution_circuit_mapper",
    "run_dynamical_correlation_calculator",
    "run_energy_estimator",
    "run_evolution_circuit_builder",
    "run_geometry_optimization",
    "run_hadamard_test",
    "run_hamiltonian_constructor",
    "run_hamiltonian_simulation",
    "run_multi_configuration_calculation",
    "run_multi_configuration_scf",
    "run_nuclear_derivative_calculator",
    "run_orbital_localization",
    "run_phase_estimation",
    "run_population_analysis",
    "run_projected_multi_configuration_calculation",
    "run_qubit_hamiltonian_solver",
    "run_qubit_mapper",
    "run_resource_estimation",
    "run_scf",
    "run_stability_checker",
    "run_state_preparation",
    "run_term_grouper",
    "run_time_evolution_builder",
    "visualize_circuit",
    "visualize_molecule",
    "visualize_orbital_entanglement",
    "visualize_orbitals",
    "visualize_scatter_plot",
    "visualize_test_square",
}
_OPTIONAL_VISUALIZATION_TOOLS = {name for name in _REVIEWED_TOOL_NAMES if name.startswith("visualize_")}


def _compact_catalog():
    list_tools = app.list_tools
    try:
        _install_tool_description_compactor(app)
        return asyncio.run(app.list_tools())
    finally:
        app.list_tools = list_tools


def test_mcp_tool_descriptions_are_compact_by_default(monkeypatch):
    monkeypatch.delenv("QDK_CHEM_MCP_COMPACT_TOOL_DESCRIPTIONS", raising=False)
    monkeypatch.delenv("QDK_CHEM_MCP_STRIP_OUTPUT_SCHEMA", raising=False)

    defaults = _parse_args([])
    assert defaults.compact_tool_descriptions is True
    assert defaults.strip_output_schema is False
    assert _parse_args(["--no-compact-tool-descriptions"]).compact_tool_descriptions is False
    assert _parse_args(["--strip-output-schema"]).strip_output_schema is True


def test_mcp_tool_description_compactor_keeps_summary_paragraph():
    class Tool:
        def __init__(self, description):
            self.description = description

        def model_copy(self, *, update):
            return Tool(update["description"])

    class Server:
        async def list_tools(self):
            return [Tool("Run the calculation.\n\nDetailed prerequisites and examples."), Tool(None)]

    server = Server()
    _install_tool_description_compactor(server)
    tools = asyncio.run(server.list_tools())

    assert [tool.description for tool in tools] == ["Run the calculation.", None]
    assert _compact_tool_description("  One\nsummary line.\n\nDetails.  ") == "One summary line."


def test_live_mcp_catalog_has_compact_descriptions():
    tools = _compact_catalog()

    tool_names = {tool.name for tool in tools}
    descriptions = [tool.description or "" for tool in tools]
    assert "qdk-chemistry-mcp" in (app.instructions or "")
    assert _REVIEWED_TOOL_NAMES - _OPTIONAL_VISUALIZATION_TOOLS <= tool_names <= _REVIEWED_TOOL_NAMES
    assert all("\n\n" not in description for description in descriptions)
    assert sum(len(description.encode()) for description in descriptions) < 16_000


def test_compact_descriptions_retain_calling_invariants():
    descriptions = {tool.name: (tool.description or "").lower() for tool in _compact_catalog()}
    required_terms = {
        "create_structure": ("bohr", "convert_coordinates"),
        "run_active_space_selector": ("charge", "autocas", "restricted", "rdm/mi"),
        "run_multi_configuration_calculation": ("casci/sci", "rdm/mi", "autocas"),
        "run_phase_estimation": ("full eigenvalue qpe", "phase bits", "evolution time", "invalid sentinels"),
        "run_resource_estimation": ("exact circuit", "inline", "physical-qubit", "no result file"),
        "get_circuit_stats": ("logical-qubit", "physical resources"),
        "run_state_preparation": ("wavefunction", "saved circuit", "symmetric active spaces"),
        "run_controlled_evolution_circuit_mapper": ("controlled circuit", "power=1", "upstream"),
        "run_time_evolution_builder": ("exp(-iht)", "does not execute qpe"),
        "run_scf": ("hf or dft", "wavefunction", "restricted hf"),
    }
    for tool_name, terms in required_terms.items():
        assert all(term in descriptions[tool_name] for term in terms), tool_name

    for tool_name in _OPTIONAL_VISUALIZATION_TOOLS & descriptions.keys():
        assert "vs code mcp apps" in descriptions[tool_name]
    if "visualize_orbital_entanglement" in descriptions:
        assert "rdm/mi" in descriptions["visualize_orbital_entanglement"]
        assert "absolute" in descriptions["visualize_orbital_entanglement"]
    if "visualize_scatter_plot" in descriptions:
        assert "svg" in descriptions["visualize_scatter_plot"]
        assert "plotly" not in descriptions["visualize_scatter_plot"]
