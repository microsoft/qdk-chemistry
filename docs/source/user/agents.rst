.. _agents:

Working with AI Agents
======================

QDK/Chemistry can be driven entirely by AI agents — no Python scripting
required. An MCP server exposes the full chemistry pipeline as structured
tools, and a CLI provides the same capabilities for shell-based workflows.

.. contents:: On This Page
   :local:
   :depth: 2


Getting started
---------------

1. **Install QDK/Chemistry** into a virtual environment (see :doc:`quickstart`).

2. **Deploy agent configs** into your project directory:

   .. code-block:: bash

      # For VS Code / GitHub Copilot Chat
      qdk_chem_cli setup-agents --target-dir . --flavor vscode

      # For Claude Code
      qdk_chem_cli setup-agents --target-dir . --flavor claude

   This creates skills, agent definitions, and an MCP server config
   (``mcp.json``) — everything an agent needs to start working.

3. **Open the project** in VS Code (or your agent platform). The MCP
   server starts automatically when the agent makes its first tool call.

That's it. Ask the agent to *"run an SCF calculation on water with
cc-pVDZ"* and it will handle structure upload, coordinate conversion,
SCF, stability check, and result inspection autonomously.


What gets deployed
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Purpose
   * - ``.vscode/mcp.json``
     - Tells VS Code where to find the MCP server
   * - ``.github/skills/``
     - Domain knowledge: tool reference, workflow patterns, pitfalls, worked examples
   * - ``.github/agents/``
     - Multi-agent definitions (orchestrator, researcher, reviewer, chemist, reporter)
   * - ``.github/copilot-instructions.md``
     - Top-level instructions for the agent platform
   * - ``CLAUDE.md`` (Claude flavor)
     - Equivalent instructions for Claude Code

Customizing agent behavior
--------------------------

The deployed files are plain Markdown — edit them freely to change how
the agent works, what it prioritizes, and how it interacts with tools.

Editing skills
~~~~~~~~~~~~~~

Skills live in ``.github/skills/<skill-name>/SKILL.md`` with optional
``references/`` subdirectories. Each skill is a self-contained knowledge
bundle that the agent loads on demand.

To customize a skill, edit the ``.md`` files directly. For example, to
add a new workflow pattern:

1. Open ``.github/skills/qdk-chemistry-mcp/SKILL.md``
2. Add your pattern under the appropriate section
3. The agent will pick it up on the next invocation

To add an entirely new skill, create a new directory under
``.github/skills/`` with a ``SKILL.md`` file.

Editing agent definitions
~~~~~~~~~~~~~~~~~~~~~~~~~

Agent definitions in ``.github/agents/`` control the multi-agent
orchestration pipeline (research → plan → critique → execute → report).
Each ``.agent.md`` file specifies:

- Which tools the agent can use
- Which sub-agents it can delegate to
- Behavioral instructions and constraints

For simple tasks, agents use skills directly — the full multi-agent
pipeline is only invoked for complex, multi-step workflows.

Editing instructions
~~~~~~~~~~~~~~~~~~~~

The top-level instructions (``.github/copilot-instructions.md`` or
``CLAUDE.md``) set global conventions: coordinate units, stability
checks, file naming, when to use which skill. Edit these to change
the agent's default behavior.


MCP server
----------

The MCP server is the interface between the agent and QDK/Chemistry. It
exposes ~40 tools organized into categories that the agent discovers via
``list_tools``.

Every tool returns a structured JSON envelope with ``status`` (``"ok"``,
``"error"``, ``"exists"``, or ``"submitted"``). All ``run_*`` tools
accept ``overwrite=True`` to bypass the ``"exists"`` check, and
``remote``/``cache`` parameters for remote execution.

Start the server manually if needed:

.. code-block:: bash

   qdk_chem_mcp                                    # stdio (default)
   qdk_chem_mcp --transport streamable-http --port 8081  # HTTP


CLI
---

The CLI (``qc``) provides the same capabilities as the MCP server for
shell-based workflows. It's organized around five concepts:

**Algorithms** (``qc run ...``)
   Execute any chemistry algorithm — SCF, active space selection, CASCI,
   MCSCF, qubit mapping, state preparation, QPE. Each command mirrors
   an MCP tool with the same parameters.

**Data inspection** (``qc data ...``)
   Read back results from project files — summaries, energies, orbital
   indices, circuit QASM, QPE results. Useful for verifying intermediate
   steps or recovering context.

**Project management** (``qc project ...``)
   Create projects, list files, manage the workspace.

**Utilities** (``qc util ...``)
   Coordinate conversion, energy unit conversion, valence parameter
   computation, QPE phase resolution.

**Configuration** (``qc config ...``)
   Query available algorithms, inspect default settings, generate
   config templates for compound algorithms (MCSCF, QPE).

All algorithm commands accept ``--cache``, ``--remote``, and
``--remote-config`` for remote execution. Use ``qc --dry-run`` to
preview parameters without executing, and ``qc --help`` for the full
command list.
