# shellcheck shell=bash
#
# This file is intended to be sourced.
#
# It provides:
# - parallel_jobs_for_memory()
# - a default CMAKE_BUILD_PARALLEL_LEVEL (unless already set)
#
# parallel_jobs_for_memory <memory_per_job_gb>
#
# Compute a safe parallel job count based on available RAM and CPU cores.
#
# Result:
#   min(cpu_cores, floor(total_ram_gb / memory_per_job_gb))
#
# with a minimum value of 1.
#
parallel_jobs_for_memory() {
    local memory_per_job_gb="$1"
    local cores mem_bytes jobs

    if [ -r /proc/meminfo ] && command -v nproc >/dev/null 2>&1; then
        # Linux
        cores=$(nproc)
        mem_bytes=$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
    else
        # macOS (or other non-/proc environments)
        cores=$(sysctl -n hw.logicalcpu)
        mem_bytes=$(sysctl -n hw.memsize)
    fi

    # If running inside a cgroup with a memory limit, respect it.
    local cgroup_limit_bytes=""
    if [ -r /sys/fs/cgroup/memory.max ]; then
        cgroup_limit_bytes="$(cat /sys/fs/cgroup/memory.max)"                    # cgroup v2
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        cgroup_limit_bytes="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"  # cgroup v1
    fi
    if [[ "$cgroup_limit_bytes" =~ ^[0-9]+$ ]] && (( cgroup_limit_bytes > 0 )) && (( cgroup_limit_bytes < mem_bytes )); then
        mem_bytes="$cgroup_limit_bytes"
    fi

    # tiny overhead to get common cases correct (e.g., 15.9GB RAM, 8GB per job -> 2 jobs)
    mem_bytes=$(( mem_bytes * 103 / 100 ))

    jobs=$(( mem_bytes / (memory_per_job_gb * 1024 * 1024 * 1024) ))

    (( jobs < 1 )) && jobs=1
    (( jobs > cores )) && jobs=$cores

    echo "$jobs"
}

if [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
    export CMAKE_BUILD_PARALLEL_LEVEL="$(parallel_jobs_for_memory 8)"
fi
