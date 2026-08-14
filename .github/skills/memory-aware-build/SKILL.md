---
name: memory-aware-build
description: Compile QDK Chemistry with memory-aware parallelism. Use this skill whenever building the C++ code, Python package, or dependencies, especially on machines with limited RAM or after an out-of-memory compiler failure.
---

# Memory-aware builds

QDK Chemistry compilation can require up to 8 GB of RAM per concurrent compile
job. Limit build parallelism by both the available memory and the CPU count.

1. Determine the number of logical CPUs and the available RAM.
2. Reserve 8 GB of available RAM for each compile job.
3. Set the job count to the smaller of:
   - the logical CPU count;
   - `floor(available RAM in GB / 8)`.
4. Always use at least one job. If available memory cannot be determined, use
   one job.
5. Export the result as `CMAKE_BUILD_PARALLEL_LEVEL` before invoking a build,
   including builds started indirectly by `pip`.

For example, a machine with 32 GB of available RAM and 16 logical CPUs should
compile with at most six jobs:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=6
cmake --build cpp/build
```

For a Python package build, apply the same limit:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=6 python -m pip install ./python
```

If a build is killed or reports an out-of-memory error, reduce
`CMAKE_BUILD_PARALLEL_LEVEL` further, preferably by half, and retry. Never
increase parallelism beyond the 8 GB-per-job memory limit merely to use all
available CPU cores.
