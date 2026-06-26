"""Command-line entry point: ``python -m qdk_chemistry.migrate OLD NEW``."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys

from . import MigrationError, convert_file


def main(argv: list[str] | None = None) -> int:
    """Run the migration CLI; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m qdk_chemistry.migrate",
        description="Migrate a qdk-chemistry data file written by an older "
        "release (<= 1.1.0) to the current (2.0) schema.",
    )
    parser.add_argument("src", help="Input file written by an older release (.json or .h5)")
    parser.add_argument("dst", help="Output file to write in the current schema (.json or .h5)")
    args = parser.parse_args(argv)

    try:
        out = convert_file(args.src, args.dst)
    except MigrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Migrated '{args.src}' -> '{out}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
