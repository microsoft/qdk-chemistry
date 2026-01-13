#!/usr/bin/env python3
"""Sanity checks for memory_pools.cmake (Ninja + non-Ninja fallback)."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
import subprocess
import tempfile
from pathlib import Path
import shutil
import textwrap


def run(cmd, cwd=None):
    """Run a command, raising on non-zero exit."""
    subprocess.check_call(cmd, cwd=cwd)


def check_ninja_available():
    """Return True if ninja is on PATH."""
    return shutil.which("ninja") is not None


def main():
    """Run Ninja and Makefile configure/build checks for memory pools."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    memory_pools = repo_root / "cmake" / "memory_pools.cmake"
    if not memory_pools.exists():
        # fallback if repo_root is root of repo and cpp is below
        memory_pools = repo_root / "cpp" / "cmake" / "memory_pools.cmake"
    assert memory_pools.exists(), f"memory_pools.cmake not found at {memory_pools}"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "foo.cpp").write_text("int foo() { return 42; }\n", encoding="utf-8")
        (td_path / "CMakeLists.txt").write_text(
            textwrap.dedent(f"""
            cmake_minimum_required(VERSION 4.2)
            project(test_pool LANGUAGES CXX)
            include("{memory_pools.as_posix()}")
            add_library(mylib foo.cpp)
            set_source_files_properties(foo.cpp PROPERTIES JOB_POOL_COMPILE heavy_compile)
        """)
        )

        # Test Ninja pool emission
        if check_ninja_available():
            build_ninja = td_path / "build_ninja"
            run(["cmake", "-G", "Ninja", "-S", str(td), "-B", str(build_ninja)])
            bn = build_ninja / "build.ninja"
            assert bn.exists(), "build.ninja not generated"
            data = bn.read_text(encoding="utf-8")
            assert "pool = heavy_compile" in data, (
                "heavy_compile pool not emitted in Ninja build"
            )
        else:
            print("[SKIP] Ninja not available", file=sys.stderr)

        # Test non-Ninja fallback target
        build_make = td_path / "build_make"
        run(["cmake", "-G", "Unix Makefiles", "-S", str(td), "-B", str(build_make)])
        cache = build_make / "CMakeCache.txt"
        assert cache.exists(), "CMakeCache.txt missing for non-Ninja build"
        cache_data = cache.read_text(encoding="utf-8")
        assert "QDK_BUILD_PARALLEL_LEVEL_HINT" in cache_data, (
            "QDK_BUILD_PARALLEL_LEVEL_HINT not in cache"
        )
        qdk_target_dir = build_make / "CMakeFiles" / "qdk_build_safe.dir"
        assert qdk_target_dir.exists(), "qdk_build_safe target directory not found"

        # Optionally ensure qdk_build_safe target is runnable (dry build)
        run(["cmake", "--build", str(build_make), "--target", "qdk_build_safe", "-j1"])


if __name__ == "__main__":
    main()
