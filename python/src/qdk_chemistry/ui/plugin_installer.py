"""Install QDK Chemistry Copilot plugins with venv-bound MCP commands."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from copy import deepcopy
from pathlib import Path
from typing import Any

from ._mcp import MCP_INSTALL_MESSAGE, require_mcp

_PLUGIN_SERVER_SCRIPTS = {"qdk-chemistry": {"qdk_chemistry": "qcmcp"}}
_MANIFEST_PATHS = (
    Path("plugin.json"),
    Path(".plugin/plugin.json"),
    Path(".github/plugin/plugin.json"),
    Path(".claude-plugin/plugin.json"),
)
_BINDINGS_FILE = "qdk-chemistry-plugin-bindings.json"
_WORKSPACE_STATE_DIR = Path(".qdk_chem")
_MARKETPLACE_MANIFEST = Path(".github") / "plugin" / "marketplace.json"


class PluginInstallError(RuntimeError):
    """Raised when the QDK Chemistry plugin cannot be installed or bound."""


def plugin_names() -> tuple[str, ...]:
    """Return the QDK Chemistry plugins supported by the installer."""
    return tuple(_PLUGIN_SERVER_SCRIPTS)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginInstallError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PluginInstallError(f"expected a JSON object in {path}")
    return value


def _manifest_path(plugin_dir: Path) -> Path:
    for relative_path in _MANIFEST_PATHS:
        candidate = plugin_dir / relative_path
        if candidate.is_file():
            return candidate
    raise PluginInstallError(f"no Copilot plugin manifest found in {plugin_dir}")


def _plugin_name_from_source(source: str, explicit_name: str | None) -> str:
    if explicit_name is not None:
        name = explicit_name
    else:
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            name = str(_load_object(_manifest_path(source_path)).get("name", ""))
        elif "@" in source and "/" not in source.split("@", 1)[0]:
            name = source.split("@", 1)[0]
        else:
            name = Path(source.removesuffix(".git")).name
    if name not in _PLUGIN_SERVER_SCRIPTS:
        supported = ", ".join(_PLUGIN_SERVER_SCRIPTS)
        raise PluginInstallError(
            f"cannot identify a supported QDK Chemistry plugin from {source!r}; use --name ({supported})"
        )
    return name


def _local_marketplace(source: str, plugin_name: str) -> tuple[Path, str, str] | None:
    source_path = Path(source).expanduser()
    if not source_path.is_dir():
        return None
    plugin_dir = source_path.resolve()
    for root in (plugin_dir, *plugin_dir.parents):
        manifest_path = root / _MARKETPLACE_MANIFEST
        if not manifest_path.is_file():
            continue
        manifest = _load_object(manifest_path)
        marketplace_name = manifest.get("name")
        plugins = manifest.get("plugins")
        if not isinstance(marketplace_name, str) or not marketplace_name or not isinstance(plugins, list):
            raise PluginInstallError(f"invalid Copilot marketplace manifest: {manifest_path}")
        for entry in plugins:
            if not isinstance(entry, dict) or entry.get("name") != plugin_name:
                continue
            configured_source = entry.get("source")
            if isinstance(configured_source, str) and (root / configured_source).resolve() == plugin_dir:
                return root, marketplace_name, f"{plugin_name}@{marketplace_name}"
    raise PluginInstallError(f"local plugin directory {plugin_dir} is not listed in an ancestor Copilot marketplace")


def _copilot_home(target_dir: str | Path | None) -> Path:
    if target_dir is not None:
        return Path(target_dir).expanduser().absolute() / _WORKSPACE_STATE_DIR / "copilot"
    configured = os.environ.get("COPILOT_HOME")
    if configured:
        return Path(configured).expanduser().absolute()
    return Path.home() / ".copilot"


def _state_root(target_dir: str | Path | None, home: Path) -> Path:
    if target_dir is None:
        return home
    return Path(target_dir).expanduser().absolute() / _WORKSPACE_STATE_DIR


def _run_copilot(arguments: list[str], *, home: Path) -> str:
    executable = shutil.which("copilot")
    if executable is None:
        raise PluginInstallError("copilot is not installed or is not on PATH")
    env = os.environ.copy()
    env["COPILOT_HOME"] = str(home)
    result = subprocess.run(
        [executable, *arguments],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise PluginInstallError(f"copilot {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _ensure_local_marketplace(root: Path, marketplace_name: str, *, home: Path) -> None:
    output = _run_copilot(["plugin", "marketplace", "list"], home=home)
    expected = f"{marketplace_name} (Local: {root})"
    name_marker = f"{marketplace_name} ("
    for line in output.splitlines():
        if expected in line:
            return
        if name_marker in line:
            raise PluginInstallError(
                f"Copilot marketplace {marketplace_name!r} is already registered from a different source beneath {home}"
            )
    _run_copilot(["plugin", "marketplace", "add", str(root)], home=home)


def _installed_plugin_dir(home: Path, plugin_name: str) -> Path:
    root = home / "installed-plugins"
    matches: list[Path] = []
    if root.is_dir():
        for marketplace_dir in root.iterdir():
            if not marketplace_dir.is_dir():
                continue
            for plugin_dir in marketplace_dir.iterdir():
                if not plugin_dir.is_dir():
                    continue
                try:
                    manifest = _load_object(_manifest_path(plugin_dir))
                except PluginInstallError:
                    continue
                if manifest.get("name") == plugin_name:
                    matches.append(plugin_dir.absolute())
    if not matches:
        raise PluginInstallError(f"Copilot did not install {plugin_name!r} beneath {root}")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise PluginInstallError(f"multiple installed copies of {plugin_name!r} found: {locations}")
    return matches[0]


def _script_path(scripts_dir: Path, script_name: str) -> Path:
    candidates = [scripts_dir / script_name]
    if os.name == "nt":
        candidates.extend((scripts_dir / f"{script_name}.exe", scripts_dir / f"{script_name}.cmd"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise PluginInstallError(
        f"required MCP command {script_name!r} is not installed in the active environment at {scripts_dir}"
    )


def _commands_for_current_environment(plugin_name: str) -> dict[str, str]:
    if sys.prefix == sys.base_prefix and not hasattr(sys, "real_prefix"):
        raise PluginInstallError(
            "QDK Chemistry plugin installation requires a virtual environment; "
            "activate the venv and rerun with the 'qc' command from that environment"
        )
    try:
        require_mcp()
    except ModuleNotFoundError as exc:
        raise PluginInstallError(MCP_INSTALL_MESSAGE) from exc
    scripts_dir = Path(sysconfig.get_path("scripts"))
    return {
        server_name: str(_script_path(scripts_dir, script_name))
        for server_name, script_name in _PLUGIN_SERVER_SCRIPTS[plugin_name].items()
    }


def _mcp_config_path(plugin_dir: Path) -> Path:
    manifest = _load_object(_manifest_path(plugin_dir))
    configured = manifest.get("mcpServers", ".mcp.json")
    if not isinstance(configured, str):
        raise PluginInstallError(f"{plugin_dir} uses inline MCP configuration, which cannot be bound")
    path = plugin_dir / configured
    if not path.is_file():
        raise PluginInstallError(f"MCP configuration not found: {path}")
    return path


def _bind_mcp_commands(plugin_dir: Path, plugin_name: str, commands: dict[str, str]) -> Path:
    path = _mcp_config_path(plugin_dir)
    config = _load_object(path)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        raise PluginInstallError(f"expected an mcpServers object in {path}")
    for server_name in _PLUGIN_SERVER_SCRIPTS[plugin_name]:
        server = servers.get(server_name)
        if not isinstance(server, dict):
            raise PluginInstallError(f"required MCP server {server_name!r} is missing from {path}")
        server["command"] = commands[server_name]
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _jsonc_to_json(text: str) -> str:
    """Remove comments and trailing commas from JSONC text."""
    without_comments: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        next_character = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            without_comments.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            without_comments.append(character)
            index += 1
            continue
        if character == "/" and next_character == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if character == "/" and next_character == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index = min(index + 2, len(text))
            continue
        without_comments.append(character)
        index += 1

    stripped = "".join(without_comments)
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(stripped):
        character = stripped[index]
        if in_string:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            result.append(character)
            index += 1
            continue
        if character == ",":
            next_index = index + 1
            while next_index < len(stripped) and stripped[next_index].isspace():
                next_index += 1
            if next_index < len(stripped) and stripped[next_index] in "}]":
                index += 1
                continue
        result.append(character)
        index += 1
    return "".join(result)


def _load_workspace_config(path: Path, servers_key: str) -> dict[str, Any]:
    if not path.exists():
        return {servers_key: {}}
    try:
        value = json.loads(_jsonc_to_json(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginInstallError(f"cannot read workspace MCP configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PluginInstallError(f"expected a JSON object in {path}")
    servers = value.setdefault(servers_key, {})
    if not isinstance(servers, dict):
        raise PluginInstallError(f"expected a {servers_key} object in {path}")
    return value


def _write_workspace_config(path: Path, servers_key: str, servers: dict[str, Any]) -> None:
    config = _load_workspace_config(path, servers_key)
    config[servers_key].update(servers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _component_directories(plugin_dir: Path, manifest: dict[str, Any], field: str) -> list[Path]:
    configured = manifest.get(field, f"{field}/")
    values = [configured] if isinstance(configured, str) else configured
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise PluginInstallError(f"plugin field {field!r} must be a path or list of paths")
    root = plugin_dir.resolve()
    directories: list[Path] = []
    for value in values:
        directory = (plugin_dir / value).resolve()
        if not directory.is_relative_to(root):
            raise PluginInstallError(f"plugin field {field!r} points outside {plugin_dir}: {value}")
        if not directory.is_dir():
            raise PluginInstallError(f"plugin {field} directory not found: {directory}")
        directories.append(directory)
    return directories


def _copy_component_directories(sources: list[Path], destination: Path) -> list[str]:
    copied: list[str] = []
    for source in sources:
        source_root = source.resolve()
        for source_file in sorted(path for path in source.rglob("*") if path.is_file() or path.is_symlink()):
            resolved_source_file = source_file.resolve()
            if not resolved_source_file.is_relative_to(source_root):
                raise PluginInstallError(
                    f"plugin component file {source_file} resolves outside {source_root} "
                    "(symlink traversal is not allowed)"
                )
            target = destination / source_file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_source_file, target)
            copied.append(str(target))
    return copied


def _workspace_server_configs(plugin_dir: Path, plugin_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = _mcp_config_path(plugin_dir)
    config = _load_object(config_path)
    plugin_servers = config.get("mcpServers")
    if not isinstance(plugin_servers, dict):
        raise PluginInstallError(f"expected an mcpServers object in {config_path}")
    vscode_servers: dict[str, Any] = {}
    github_servers: dict[str, Any] = {}
    for server_name in _PLUGIN_SERVER_SCRIPTS[plugin_name]:
        server = plugin_servers.get(server_name)
        if not isinstance(server, dict):
            raise PluginInstallError(f"required MCP server {server_name!r} is missing from the plugin")
        vscode_entry = deepcopy(server)
        vscode_entry["type"] = "stdio"
        vscode_entry.pop("tools", None)
        vscode_servers[server_name] = vscode_entry

        github_entry = deepcopy(server)
        github_entry["type"] = "local"
        github_entry["tools"] = ["*"]
        github_servers[server_name] = github_entry
    return vscode_servers, github_servers


def _ignore_workspace_state(workspace: Path) -> None:
    path = workspace / _WORKSPACE_STATE_DIR / ".gitignore"
    ignored = {"/copilot/", f"/{_BINDINGS_FILE}"}
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    missing = sorted(ignored.difference(existing))
    if not missing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*existing, *missing]) + "\n", encoding="utf-8")


def _deploy_workspace(plugin_dir: Path, plugin_name: str, workspace: Path) -> dict[str, Any]:
    manifest = _load_object(_manifest_path(plugin_dir))
    workspace.mkdir(parents=True, exist_ok=True)
    agents = []
    if "agents" in manifest:
        agents = _copy_component_directories(
            _component_directories(plugin_dir, manifest, "agents"), workspace / ".github" / "agents"
        )
    skills = _copy_component_directories(
        _component_directories(plugin_dir, manifest, "skills"), workspace / ".github" / "skills"
    )
    vscode_servers, github_servers = _workspace_server_configs(plugin_dir, plugin_name)
    vscode_mcp = workspace / ".vscode" / "mcp.json"
    github_mcp = workspace / ".github" / "mcp.json"
    _write_workspace_config(vscode_mcp, "servers", vscode_servers)
    _write_workspace_config(github_mcp, "mcpServers", github_servers)
    _ignore_workspace_state(workspace)
    return {"agents": agents, "skills": skills, "mcp_configs": [str(vscode_mcp), str(github_mcp)]}


def _binding_path(root: Path) -> Path:
    return root / _BINDINGS_FILE


def _load_bindings(root: Path) -> dict[str, Any]:
    path = _binding_path(root)
    if not path.exists():
        return {"version": 1, "plugins": {}}
    state = _load_object(path)
    if state.get("version") != 1 or not isinstance(state.get("plugins"), dict):
        raise PluginInstallError(f"unsupported QDK Chemistry plugin binding state in {path}")
    return state


def _write_binding(root: Path, plugin_name: str, record: dict[str, Any]) -> None:
    state = _load_bindings(root)
    state["plugins"][plugin_name] = record
    path = _binding_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _update_spec(source: str, plugin_name: str) -> str:
    return source if source.startswith(f"{plugin_name}@") else plugin_name


def install_plugin(source: str, *, name: str | None = None, target_dir: str | Path | None = None) -> dict[str, Any]:
    """Install a QDK Chemistry plugin and bind its MCP command to this environment.

    Args:
        source: Copilot plugin source, plugin name, or local plugin directory.
        name: Explicit supported plugin name when it cannot be inferred from
            ``source``.
        target_dir: Workspace directory for a workspace-scoped installation.
            When omitted, installs for the current user.

    Returns:
        Installation details including the plugin, scope, bound commands, and
        MCP configuration path.

    Raises:
        PluginInstallError: If the plugin source, Copilot installation, or MCP
            command binding is invalid.

    """
    plugin_name = _plugin_name_from_source(source, name)
    commands = _commands_for_current_environment(plugin_name)
    home = _copilot_home(target_dir)
    state_root = _state_root(target_dir, home)
    install_source = source
    local_marketplace = _local_marketplace(source, plugin_name)
    if local_marketplace is not None:
        marketplace_root, marketplace_name, install_source = local_marketplace
        _ensure_local_marketplace(marketplace_root, marketplace_name, home=home)
    _run_copilot(["plugin", "install", install_source], home=home)
    plugin_dir = _installed_plugin_dir(home, plugin_name)
    plugin_config = _bind_mcp_commands(plugin_dir, plugin_name, commands)
    workspace = Path(target_dir).expanduser().absolute() if target_dir is not None else None
    deployment = _deploy_workspace(plugin_dir, plugin_name, workspace) if workspace is not None else {}
    _write_binding(
        state_root,
        plugin_name,
        {
            "commands": commands,
            "plugin_dir": str(plugin_dir),
            "source": source,
            "update_spec": _update_spec(install_source, plugin_name),
            **deployment,
        },
    )
    result = {
        "status": "installed",
        "plugin": plugin_name,
        "scope": "workspace" if workspace is not None else "user",
        "copilot_home": str(home),
        "plugin_dir": str(plugin_dir),
        "mcp_config": str(plugin_config),
        "commands": commands,
    }
    if workspace is not None:
        result.update({"workspace": str(workspace), **deployment})
        result["mcp_config"] = deployment["mcp_configs"][0]
    return result


def update_plugin(plugin_name: str, *, target_dir: str | Path | None = None) -> dict[str, Any]:
    """Update a plugin and restore its MCP command binding.

    Args:
        plugin_name: Name of the supported QDK Chemistry plugin to update.
        target_dir: Workspace directory for a workspace-scoped installation.
            When omitted, updates the current user's installation.

    Returns:
        Update details including the plugin, scope, bound commands, and MCP
        configuration path.

    Raises:
        PluginInstallError: If the plugin is unsupported, unbound, or cannot be
            updated and rebound.

    """
    if plugin_name not in _PLUGIN_SERVER_SCRIPTS:
        raise PluginInstallError(f"unsupported QDK Chemistry plugin: {plugin_name}")
    home = _copilot_home(target_dir)
    state_root = _state_root(target_dir, home)
    state = _load_bindings(state_root)
    record = state["plugins"].get(plugin_name)
    if not isinstance(record, dict):
        raise PluginInstallError(f"{plugin_name!r} has no QDK Chemistry binding beneath {state_root}")
    commands = record.get("commands")
    if not isinstance(commands, dict) or any(not isinstance(value, str) for value in commands.values()):
        raise PluginInstallError(f"invalid command binding for {plugin_name!r} beneath {state_root}")
    for command in commands.values():
        if not Path(command).is_file():
            raise PluginInstallError(f"bound MCP command no longer exists: {command}; rebind from the intended venv")
    _run_copilot(["plugin", "update", str(record.get("update_spec") or plugin_name)], home=home)
    plugin_dir = _installed_plugin_dir(home, plugin_name)
    plugin_config = _bind_mcp_commands(plugin_dir, plugin_name, commands)
    workspace = Path(target_dir).expanduser().absolute() if target_dir is not None else None
    deployment = _deploy_workspace(plugin_dir, plugin_name, workspace) if workspace is not None else {}
    record.update({"plugin_dir": str(plugin_dir), **deployment})
    _write_binding(state_root, plugin_name, record)
    result = {
        "status": "updated",
        "plugin": plugin_name,
        "scope": "workspace" if workspace is not None else "user",
        "copilot_home": str(home),
        "plugin_dir": str(plugin_dir),
        "mcp_config": str(plugin_config),
        "commands": commands,
    }
    if workspace is not None:
        result.update({"workspace": str(workspace), **deployment})
        result["mcp_config"] = deployment["mcp_configs"][0]
    return result


def update_all_plugins(*, target_dir: str | Path | None = None) -> dict[str, Any]:
    """Update every installed QDK Chemistry plugin binding.

    Args:
        target_dir: Workspace directory for workspace-scoped installations.
            When omitted, updates the current user's installations.

    Returns:
        A status mapping with the update result for each bound plugin.

    Raises:
        PluginInstallError: If no QDK Chemistry plugin bindings are found or an
            update fails.

    """
    home = _copilot_home(target_dir)
    state = _load_bindings(_state_root(target_dir, home))
    names = sorted(set(state["plugins"]).intersection(_PLUGIN_SERVER_SCRIPTS))
    if not names:
        raise PluginInstallError(f"no QDK Chemistry plugin bindings found beneath {_state_root(target_dir, home)}")
    return {"status": "updated", "plugins": [update_plugin(name, target_dir=target_dir) for name in names]}


def rebind_plugin(plugin_name: str, *, target_dir: str | Path | None = None) -> dict[str, Any]:
    """Bind an installed plugin to the MCP command in the current environment.

    Args:
        plugin_name: Name of the supported QDK Chemistry plugin to rebind.
        target_dir: Workspace directory for a workspace-scoped installation.
            When omitted, rebinds the current user's installation.

    Returns:
        Rebinding details including the plugin, scope, bound commands, and MCP
        configuration path.

    Raises:
        PluginInstallError: If the plugin is unsupported, missing, or the
            current environment does not provide its MCP command.

    """
    if plugin_name not in _PLUGIN_SERVER_SCRIPTS:
        raise PluginInstallError(f"unsupported QDK Chemistry plugin: {plugin_name}")
    home = _copilot_home(target_dir)
    state_root = _state_root(target_dir, home)
    plugin_dir = _installed_plugin_dir(home, plugin_name)
    commands = _commands_for_current_environment(plugin_name)
    plugin_config = _bind_mcp_commands(plugin_dir, plugin_name, commands)
    workspace = Path(target_dir).expanduser().absolute() if target_dir is not None else None
    deployment = _deploy_workspace(plugin_dir, plugin_name, workspace) if workspace is not None else {}
    previous = _load_bindings(state_root)["plugins"].get(plugin_name)
    if isinstance(previous, dict):
        source = str(previous.get("source") or plugin_name)
        update_spec = str(previous.get("update_spec") or plugin_name)
    else:
        marketplace = plugin_dir.parent.name
        source = plugin_name
        update_spec = plugin_name if marketplace == "_direct" else f"{plugin_name}@{marketplace}"
    _write_binding(
        state_root,
        plugin_name,
        {
            "commands": commands,
            "plugin_dir": str(plugin_dir),
            "source": source,
            "update_spec": update_spec,
            **deployment,
        },
    )
    result = {
        "status": "rebound",
        "plugin": plugin_name,
        "scope": "workspace" if workspace is not None else "user",
        "copilot_home": str(home),
        "plugin_dir": str(plugin_dir),
        "mcp_config": str(plugin_config),
        "commands": commands,
    }
    if workspace is not None:
        result.update({"workspace": str(workspace), **deployment})
        result["mcp_config"] = deployment["mcp_configs"][0]
    return result
