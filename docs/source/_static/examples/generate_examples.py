# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
"""Generate example files from templates by substituting coordinate data from JSON files."""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
TEMPLATES_DIR = SCRIPT_DIR / "templates"


def load_data(filename: str) -> dict:
    with open(DATA_DIR / f"{filename}.coords.json") as f:
        return json.load(f)


def py_coords(coords: list) -> str:
    """Multi-line np.array format."""
    rows = [f"        [{c[0]:.8f}, {c[1]:.8f}, {c[2]:.8f}]," for c in coords]
    return "np.array(\n    [\n" + "\n".join(rows) + "\n    ]\n)"


def py_coords_inline(coords: list) -> str:
    """Compact np.array format."""
    inner = ", ".join(f"[{c[0]}, {c[1]}, {c[2]}]" for c in coords)
    return f"np.array([{inner}])"


def py_elements(elements: list) -> str:
    return "[" + ", ".join(f'"{e}"' for e in elements) + "]"


def cpp_coords(coords: list) -> str:
    """Eigen::MatrixXd with << operator, wrapped at 80 chars."""
    n = len(coords)
    values = [f"{v:.6f}" for c in coords for v in c]
    lines = [f"Eigen::MatrixXd coords({n}, 3);"]
    line = "  coords << "
    for i, val in enumerate(values):
        sep = ", " if i < len(values) - 1 else ";"
        if len(line) + len(val) + len(sep) > 80 and i > 0:
            lines.append(line.rstrip(", ") + ",")
            line = "      " + val + sep
        else:
            line += val + sep
    lines.append(line)
    return "\n".join(lines)


def cpp_coords_vector(coords: list) -> str:
    """std::vector<Eigen::Vector3d> initialization."""
    formatted = [f"{{{c[0]}, {c[1]}, {c[2]}}}" for c in coords]
    if len(coords) <= 2:
        # Two-line compact format with alignment
        return (
            f"std::vector<Eigen::Vector3d> coords = {{{formatted[0]},\n"
            f"                                         {formatted[1]}}};"
        )
    lines = ["std::vector<Eigen::Vector3d> coords = {"]
    for i, f in enumerate(formatted):
        lines.append(f"    {f}{',' if i < len(formatted) - 1 else ''}")
    lines.append("};")
    return "\n".join(lines)


def cpp_elements(elements: list, var: str = "elements") -> str:
    """std::vector<std::string> initialization, wrapped at 5 elements."""
    quoted = [f'"{e}"' for e in elements]
    if len(quoted) > 5:
        line1 = ", ".join(quoted[:5]) + ","
        line2 = ", ".join(quoted[5:])
        return f"std::vector<std::string> {var} = {{{line1}\n                                       {line2}}};"
    return f"std::vector<std::string> {var} = {{" + ", ".join(quoted) + "};"


# Matches: {{COORDS_H2}}, {{COORDS_INLINE_H2}}, {{COORDS_VECTOR_BENZENE_DIRADICAL}},
#          {{ELEMENTS_H2}}, {{SYMBOLS_H2}}, etc.
PLACEHOLDER_RE = re.compile(
    r"\{\{(?P<type>COORDS|ELEMENTS|SYMBOLS)(?:_(?P<fmt>INLINE|VECTOR))?_(?P<name>[A-Z0-9_]+)\}\}"
)


def process_template(content: str, lang: str) -> str:
    # Collect and load all referenced data files
    names = {m.group("name").lower() for m in PLACEHOLDER_RE.finditer(content)}
    cache = {name: load_data(name) for name in names}

    def replace(m):
        ptype, fmt, name = m.group("type"), m.group("fmt"), m.group("name").lower()
        if name not in cache:
            return m.group(0)
        data = cache[name]

        if ptype == "COORDS":
            if fmt == "INLINE":
                return (
                    cpp_coords_vector(data["coords"])
                    if lang == "cpp"
                    else py_coords_inline(data["coords"])
                )
            if fmt == "VECTOR":
                return cpp_coords_vector(data["coords"])
            return (
                cpp_coords(data["coords"])
                if lang == "cpp"
                else py_coords(data["coords"])
            )
        if ptype in ("ELEMENTS", "SYMBOLS"):
            var = "symbols" if ptype == "SYMBOLS" else "elements"
            return (
                cpp_elements(data["elements"], var)
                if lang == "cpp"
                else py_elements(data["elements"])
            )
        return m.group(0)

    return PLACEHOLDER_RE.sub(replace, content)


def main():
    for lang, ext in [("python", ".py"), ("cpp", ".cpp")]:
        tmpl_dir = TEMPLATES_DIR / lang
        out_dir = SCRIPT_DIR / lang
        if not tmpl_dir.exists():
            continue
        for tmpl in tmpl_dir.glob(f"*{ext}.tmpl"):
            out_file = out_dir / tmpl.stem
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(process_template(tmpl.read_text(), lang))
            print(f"Generated: {out_file}")


if __name__ == "__main__":
    main()
