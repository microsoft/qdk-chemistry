<!-- qdk-chemistry-agent-config {{QDK_CHEMISTRY_VERSION}} -->
# QDK Chemistry — General Instructions

You are working in a **QDK Chemistry** workspace. This project provides quantum chemistry tools through an MCP server, CLI, and Python SDK.

## How to Work in This Workspace

### Use Skills First

Skills are pre-built domain knowledge bundles in `.github/skills/`. **Always load and read the relevant skill before acting.** Each skill contains tested workflows, parameter guidance, and pitfalls that save time and prevent errors.

| Skill | When to Load |
|---|---|
| `qdk-chemistry-overview` | Starting a new task, orienting yourself, understanding what's available |
| `qdk-chemistry-mcp` | Using any MCP tool — contains the full tool reference, active space guide, QPE guide, compression strategies, and common failure modes |
| `qdk-chemistry-coding` | Writing or reviewing Python code that uses the `qdk_chemistry` SDK — contains the API reference, factory patterns, and complete worked examples |
| `remote-execution` | Submitting jobs to remote compute (HPC clusters, SSH backends) |

**Workflow:** Read the skill → follow its guidance → consult its reference docs for specifics. Don't guess parameters or invent procedures when the skill has the answer.

### When to Use Sub-agents

Use the specialized agents (in `.github/agents/`) only for **complex, multi-step quantum chemistry workflows** — not for simple questions or one-off tool calls.

| Agent | Use When |
|---|---|
| `quantum-agent` | Full multi-step workflows: research → plan → critique → execute → visualize → report. This is the top-level orchestrator — invoke it for tasks like "study the dissociation of N₂" or "run a CASSCF calculation on benzene and build QPE circuits" |
| `researcher` | You need to look up specific facts from the QDK Chemistry Playbook or GitHub repos (called by quantum-agent, not directly by users) |
| `reviewer` | You need a plan critically evaluated before execution (called by quantum-agent) |
| `chemist` | You need to validate parameters or execute MCP tool workflows (called by quantum-agent) |
| `reporter` | You need a final scientific report with tables and reproducible script (called by quantum-agent) |

**For simple tasks** — answering a question, running a single MCP tool, explaining a concept, writing a short script — **use skills directly. Don't invoke the full agent pipeline.**

### Quick Decision Tree

```
Is this a multi-step quantum chemistry workflow?
├── YES → Use @quantum-agent
└── NO
    ├── Is this about using MCP tools? → Load qdk-chemistry-mcp skill
    ├── Is this about writing Python code? → Load qdk-chemistry-coding skill
    ├── Is this about remote execution? → Load remote-execution skill
    ├── Need background on a concept/algorithm? → Load qdk-chemistry-overview skill
    └── General question → Load qdk-chemistry-overview skill
```

### Key Conventions

- **Cite your sources** — every quantum chemistry or quantum computing statement must reference where it comes from: a skill file, tool output, or source file. Don't present training-data knowledge as fact without verification
- **Visuals over text** — whenever a VS Code visualization tool can show something (molecule, orbitals, circuit, entanglement diagram, scatter plot), use it. A chart or diagram communicates more than a paragraph of description
- **Coordinates are in Bohr** (not Ångström) for all MCP tools and SDK functions
- **Always run `run_stability_checker` after `run_scf`** — unstable SCF solutions produce wrong results downstream
- **Full-space vs active-space is a choice** — for small systems (up to ~16 spatial orbitals), the full orbital space is tractable. Don't default to active space compression — ask the user or check the system size
- **Use automated tools over manual expert choices** — e.g., `qdk_autocas_eos` for active space selection instead of asking users to pick orbitals
- **Logical + physical resources** — `get_circuit_stats` gives circuit-level metrics; `run_resource_estimation` gives the full profile including physical qubits, runtime, and error budget
- **File naming convention:** `{name}.{datatype}.{ext}` (e.g., `h2_scf.wavefunction.json`)
