---
name: azure-quantum-backend
version: 1.0.0
description: "Use when running or submitting QDK Chemistry circuits through the Azure Quantum backend, configuring an Azure Quantum connection, or choosing between a local demo workspace and a remote Azure workspace. Check local availability first, ask the user which execution mode to use, then connect and submit only with approval."
---

# Azure Quantum Backend

Guide circuit execution through an explicit local-or-remote choice. A local
demo workspace is not an Azure service, and locally stored Azure configuration
does not make a remote workspace local.

## 1. Check availability without submitting

Before making a connection or executing a circuit:

- Identify the active Python environment and inspect the installed
  `qdk_chemistry.plugins.azure_quantum` API. Verify optional dependencies with
  imports; do not assume a checkout's modules are installed.
- Check whether
  `qdk_chemistry.plugins.azure_quantum.demo.DemoWorkspace` is importable. This
  demo is optional and may be absent in an installed release. Report its
  absence; never substitute a real cloud target.
- Look for existing Azure workspace coordinates in user-provided settings or
  known workspace configuration. Check whether Azure CLI is available and
  signed in using `az account show`. Do not dump environment variables,
  credentials, CLI token caches, or connection strings.
- Report separately: local demo availability, existing remote workspace
  configuration, and authentication status. Neither an installed SDK nor an
  Azure login proves that a workspace or target is accessible.

If a required import fails, explain the missing dependency and use the active
environment's package manager to install the relevant extra if approved:

```bash
python -m pip install 'qdk-chemistry[azure-quantum]'
```

The extra does not guarantee that the optional local demo module exists.
Do not modify dependencies merely to perform availability checks.

## 2. Ask local or remote

Use the question tool before executing, unless the user has already explicitly
chosen a mode for this run. Present the available choices:

1. **Local demo:** ideal QDK CPU full-state simulation, no Azure connection,
   cloud job, or cloud billing.
2. **Remote Azure Quantum:** connect to an existing Azure workspace and
   discover its accessible targets; execution may incur charges.
3. **Inspect only:** report configuration and compatibility without executing.

If a local option is unavailable, say so rather than offering it as working.
If the user means a real Azure workspace whose configuration is stored
locally, treat it as remote. Do not infer their choice from an earlier run.
Respect cancellation and do not silently switch modes after an error.

## 3A. Local demo execution

Inspect the installed demo signatures first. Where supported, the direct
workspace interface avoids process-wide patching:

```python
from qdk_chemistry.plugins.azure_quantum.demo import DemoWorkspace

workspace = DemoWorkspace.from_connection_string("demo://local")
target = workspace.get_targets("demo.target")
job = target.submit(
    input_data=circuit.get_qir(),
    name="chemistry-local-validation",
    shots=shots,
)
histogram = job.get_results_histogram()
```

To exercise the unchanged `AzureQuantumBackend` implementation, an installed
demo module may instead expose:

```python
from qdk_chemistry.plugins.azure_quantum.circuit_executor import AzureQuantumBackend
from qdk_chemistry.plugins.azure_quantum.demo import (
    DEMO_BACKEND_SETTINGS,
    demo_azure_quantum,
)

with demo_azure_quantum(seed=42):
    executor = AzureQuantumBackend(**DEMO_BACKEND_SETTINGS)
    result = executor.run(circuit, shots=shots)
```

Use this patching context only in a dedicated local process. It patches SDK
behavior process-wide; do not mix it with real submissions or concurrent
threads. Never send `demo://local` or demo coordinates to Azure.

The demo accepts QIR, runs synchronously, and keeps jobs in memory. Target
aliases do not represent different hardware. It does not emulate provider
noise, provider-specific options, or attachments. Full-state simulation memory
grows exponentially; assess both width and gate count before running. Request
approval before reducing precision, shots, or circuit size, and label such a
run as validation rather than the original scientific calculation.

## 3B. Connect to remote Azure Quantum

1. Reuse confirmed workspace coordinates when present. Otherwise ask for the
   workspace resource ID or subscription ID, resource group, workspace name,
   and region. A resource ID has this form:
   `/subscriptions/<id>/resourceGroups/<group>/providers/Microsoft.Quantum/workspaces/<name>`.
   Do not ask the user to paste keys, tokens, or connection strings into chat.
2. Confirm the intended subscription. For Azure CLI authentication, check
   `az account show`; if login is needed, ask the user to complete `az login`
   (or an appropriate device-code login) interactively. Do not change the
   active subscription without approval. For managed or environment-based
   identity, use the supported `default` authentication mode instead.
3. If the user wants discovery, query the selected subscription explicitly:

   ```bash
   az resource list --subscription <subscription-id> \
       --resource-type Microsoft.Quantum/workspaces \
       --query '[].{name:name,resourceGroup:resourceGroup,location:location}' \
       --output json --only-show-errors
   ```

   An empty result means no visible workspaces in that subscription, not that
   none exist elsewhere. Report authorization and network errors explicitly.
   Do not create Azure resources, enable providers, or change plans implicitly.
4. Inspect the installed `azure.quantum.Workspace` and target APIs. Connect
   using the selected identity and list targets:

   ```python
   from azure.identity import AzureCliCredential
   from azure.quantum import Workspace

   workspace = Workspace(
       subscription_id=subscription_id,
       resource_group=resource_group,
       name=workspace_name,
       location=location,
       credential=AzureCliCredential(),
   )
   targets = workspace.get_targets()
   ```

   For `auth_mode="default"`, use `DefaultAzureCredential` consistently.
   List target names and availability; ask the user to select a target.
   Do not hard-code provider names or assume a validator is free.
5. Check target input/output formats, QIR profile, qubit capacity, supported
   operations, shots, and provider limits. Compile `circuit.get_qir()` locally
   before submission. Distinguish a simulator, syntax validator, resource
   estimator, and hardware execution target. A fault-tolerant resource model
   is not a runnable hardware target.
6. Inspect provider cost-estimation support when available; not every target
   exposes the same API. If no reliable quote is available, report that cost
   is unknown. Present the exact circuit scope, target, shots, expected
   limitations, and any quote. Obtain explicit submission and cost approval.
   Do not submit the full workload when only a small validation was approved.

## 4. Submit and track the chosen remote job

For a blocking execution, inspect the installed executor settings and use:

```python
from qdk_chemistry.plugins.azure_quantum.circuit_executor import AzureQuantumBackend

executor = AzureQuantumBackend(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    location=location,
    target_name=target_name,
    auth_mode="azure-cli",
    job_name="chemistry-circuit",
)
result = executor.run(circuit, shots=shots)
```

This is a `circuit_executor`, not a remote algorithm backend selected with
`remote="azure"`. The executor compiles QIR, submits, and waits; inspect its
`timeout_secs` and JSON-string `input_params` settings as needed. Do not pass
a `QuantumErrorProfile`: configure supported noise through the target instead.

If asynchronous submission or immediate job-ID persistence is needed, inspect
the selected SDK target's `submit` signature and use it directly. Preserve the
QDK QIR object's submission metadata; do not guess data formats. Persist the
returned job ID and nonsecret workspace/target coordinates immediately in the
user-approved output location. Do not persist access tokens or signed URLs.

Check status using the submitted job rather than submitting again. A timeout
does not imply cancellation or failure. On interruption, retain the job ID
for later retrieval; cancel only when requested. Report failures, lost shots,
and invalid outcomes explicitly rather than returning a success-shaped result.

## 5. Report what actually ran

Return the execution mode, circuit scope, target, shots, job ID if available,
status, and measurements or error details. Persist requested results and
nonsecret metadata.

For phase estimation, one circuit's measurements are not a completed molecular
energy calculation. Account for all rounds, feedback, repetitions, and the
Hamiltonian's core-energy offset before claiming an energy or precision.
