.. _agents:

Working with AI Assistants
==========================

QDK/Chemistry can be used through AI assistants without Python scripting.
The plugin provides skills with QDK/Chemistry guidance and an MCP server that
exposes the chemistry pipeline as structured tools. A CLI provides the same
capabilities for shell-based workflows.

.. contents:: On This Page
   :local:
   :depth: 2


Getting started
---------------

1. **Install QDK/Chemistry** into a virtual environment (see :doc:`quickstart`).

2. **Deploy the plugin** into your project directory:

   .. code-block:: bash

      # Install the QDK/Chemistry plugin and deploy its skills and MCP configuration.
      qc plugin install qdk-chemistry@qdk-chemistry --target-dir .

      # Or install from a local QDK/Chemistry checkout.
      qc plugin install ./copilot-plugins/qdk-chemistry --target-dir .

   This creates skills and MCP server configurations
   (``.vscode/mcp.json`` and ``.github/mcp.json``) for the AI assistant.

3. **Open the project** in VS Code (or another compatible client). The MCP
   server starts automatically when the assistant makes its first tool call.

That's it. Ask the assistant to *"run an SCF calculation on water with
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
   * - ``.github/mcp.json``
     - Tells GitHub Copilot where to find the MCP server
   * - ``.github/skills/``
     - Domain knowledge: tool reference, workflow patterns, pitfalls, worked examples

Customizing plugin guidance
---------------------------

The deployed skills are plain Markdown. Edit them to customize the guidance
available to the assistant.

Editing skills
~~~~~~~~~~~~~~

Skills live in ``.github/skills/<skill-name>/SKILL.md`` with optional
``references/`` subdirectories. Each skill is a self-contained knowledge
bundle that the assistant loads on demand.

To customize a skill, edit the ``.md`` files directly. For example, to
add a new workflow pattern:

1. Open ``.github/skills/qdk-chemistry-mcp/SKILL.md``
2. Add your pattern under the appropriate section
3. The assistant will pick it up on the next invocation

To add an entirely new skill, create a new directory under
``.github/skills/`` with a ``SKILL.md`` file.

MCP server
----------

The MCP server is the interface between the AI assistant and QDK/Chemistry. It
exposes ~50 tools organized into categories that the assistant discovers via
``list_tools``.

Every tool returns a structured JSON envelope with ``status`` (``"ok"``,
``"error"``, ``"exists"``, or ``"submitted"``). All ``run_*`` tools
accept ``overwrite=True`` to bypass the ``"exists"`` check, and
``remote``/``cache`` parameters for remote execution.

Start the server manually if needed:

.. code-block:: bash

   qcmcp                                    # stdio (default)
   qcmcp --transport streamable-http --port 8081  # HTTP


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
