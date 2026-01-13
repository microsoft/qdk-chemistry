#!/bin/bash
# Profile memory usage per C++ file during compilation using /proc

BUILD_DIR="/anfhome/cojohnston/Repos/qdk-chemistry/cpp/build"
SRC_ROOT="/anfhome/cojohnston/Repos/qdk-chemistry/cpp/src"
RESULTS_FILE="$BUILD_DIR/memory_profile_results.txt"

# Read compiler flags from CMake generated file (handle Make syntax)
CXX_DEFINES=$(grep "^CXX_DEFINES" "$BUILD_DIR/CMakeFiles/chemistry.dir/flags.make" | cut -d'=' -f2-)
CXX_INCLUDES=$(grep "^CXX_INCLUDES" "$BUILD_DIR/CMakeFiles/chemistry.dir/flags.make" | cut -d'=' -f2-)
CXX_FLAGS=$(grep "^CXX_FLAGS" "$BUILD_DIR/CMakeFiles/chemistry.dir/flags.make" | cut -d'=' -f2-)

COMPILER="/usr/bin/g++"
COMPILE_FLAGS="$CXX_FLAGS $CXX_DEFINES $CXX_INCLUDES"

cd "$BUILD_DIR"
> "$RESULTS_FILE"

echo "Profiling memory usage for each .cpp file..."
echo "Compiler: $COMPILER"
echo ""

# Function to get max RSS of a process by polling /proc
get_max_rss() {
    local pid=$1
    local max_rss=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ -f /proc/$pid/status ]]; then
            local rss=$(grep VmHWM /proc/$pid/status 2>/dev/null | awk '{print $2}')
            if [[ -n "$rss" && "$rss" -gt "$max_rss" ]]; then
                max_rss=$rss
            fi
        fi
        sleep 0.1
    done
    echo "$max_rss"
}

find "$SRC_ROOT" -name "*.cpp" | sort | while read -r src_file; do
    rel_path="${src_file#/anfhome/cojohnston/Repos/qdk-chemistry/cpp/}"
    echo -n "Profiling: $rel_path ... "

    # Start compilation in background
    $COMPILER $COMPILE_FLAGS -c "$src_file" -o /dev/null 2>/dev/null &
    pid=$!

    # Monitor memory
    max_rss=$(get_max_rss $pid)
    wait $pid
    exit_code=$?

    if [[ "$exit_code" -eq 0 && "$max_rss" -gt 0 ]]; then
        max_mb=$((max_rss / 1024))
        printf "%d MB\n" "$max_mb"
        printf "%6d MB - %s\n" "$max_mb" "$rel_path" >> "$RESULTS_FILE"
    else
        echo "FAILED (exit=$exit_code)"
    fi
done

echo ""
echo "=============================================="
echo "=== TOP 20 MEMORY CONSUMERS (by peak RSS) ==="
echo "=============================================="
sort -rn "$RESULTS_FILE" | head -20
echo ""
echo "Full results saved to: $RESULTS_FILE"
