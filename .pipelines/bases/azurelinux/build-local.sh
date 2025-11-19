#!/bin/bash
# Local build script for QDK Azure Linux Base Images
# Usage: ./build-local.sh [cpu|sm90|sm80|sm75] [amd64|arm64] [3.10|3.11|3.12|3.13]

set -e

# Default values
COMPUTE_TYPE=${1:-cpu}
ARCHITECTURE=${2:-amd64}
PYTHON_VERSION=${3:-3.13}

# Validate inputs
if [[ "$COMPUTE_TYPE" != "cpu" && "$COMPUTE_TYPE" != "sm90" && "$COMPUTE_TYPE" != "sm80" && "$COMPUTE_TYPE" != "sm75" ]]; then
    echo "Error: COMPUTE_TYPE must be 'cpu', 'sm90', 'sm80', or 'sm75'"
    exit 1
fi

if [[ "$ARCHITECTURE" != "amd64" && "$ARCHITECTURE" != "arm64" ]]; then
    echo "Error: ARCHITECTURE must be 'amd64' or 'arm64'"
    exit 1
fi

if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12|13)$ ]]; then
    echo "Error: PYTHON_VERSION must be 3.10, 3.11, 3.12, or 3.13"
    exit 1
fi

# GPU not supported on ARM64
if [[ "$COMPUTE_TYPE" != "cpu" && "$ARCHITECTURE" == "arm64" ]]; then
    echo "Error: GPU images (sm90, sm80, sm75) are not supported on ARM64 architecture"
    exit 1
fi

# Set platform for Docker buildx
PLATFORM="linux/amd64"
if [[ "$ARCHITECTURE" == "arm64" ]]; then
    PLATFORM="linux/arm64"
fi

# Set Docker architecture name (same as ARCHITECTURE for these platforms)
DOCKER_ARCH="$ARCHITECTURE"

# Build image name
IMAGE_NAME="azurelinux-base-${COMPUTE_TYPE}-${ARCHITECTURE}-py${PYTHON_VERSION//.}"
TAG="local-$(date +%Y%m%d-%H%M%S)"

echo "=================================================="
echo "Building QDK Azure Linux Base Image"
echo "=================================================="
echo "Compute Type: $COMPUTE_TYPE"
echo "Architecture: $ARCHITECTURE ($DOCKER_ARCH)"
echo "Platform: $PLATFORM"
echo "Python Version: $PYTHON_VERSION"
echo "Image Name: $IMAGE_NAME:$TAG"
echo "=================================================="

# Ensure buildx is set up
docker buildx create --name qdk-local-builder --use --bootstrap 2>/dev/null || docker buildx use qdk-local-builder 2>/dev/null || true

# Build arguments
BUILD_ARGS=(
    "--platform=$PLATFORM"
    "--build-arg=PYTHON_VERSION=$PYTHON_VERSION"
    "--build-arg=AZURE_LINUX_VERSION=3.0"
    "--build-arg=ARCHITECTURE=$DOCKER_ARCH"
)

if [[ "$COMPUTE_TYPE" == "cpu" ]]; then
    BUILD_ARGS+=(
        "--build-arg=IMAGE_TYPE=cpu"
        "--build-arg=CUDA_ARCH=none"
        "--build-arg=GPU_TYPE=none"
    )
else
    # GPU compute types (sm90, sm80, sm75)
    CUDA_ARCH="${COMPUTE_TYPE#sm}"  # Extract number from sm90 -> 90

    # Map SM architecture to GPU type for Docker labels
    case "$COMPUTE_TYPE" in
        "sm90")
            GPU_TYPE="h100"
            ;;
        "sm80")
            GPU_TYPE="a100"
            ;;
        "sm75")
            GPU_TYPE="generic"
            ;;
    esac

    BUILD_ARGS+=(
        "--build-arg=IMAGE_TYPE=gpu"
        "--build-arg=CUDA_VERSION=12.1.0"
        "--build-arg=CUDA_ARCH=$CUDA_ARCH"
        "--build-arg=GPU_TYPE=$GPU_TYPE"
    )
fi

# Build the image
echo "Building image..."
docker buildx build \
    "${BUILD_ARGS[@]}" \
    -t "$IMAGE_NAME:$TAG" \
    -t "$IMAGE_NAME:latest" \
    --load \
    .

echo "=================================================="
echo "Build completed successfully!"
echo "=================================================="
echo "Image: $IMAGE_NAME:$TAG"
echo "Image: $IMAGE_NAME:latest"
echo ""
echo "To run the container:"
echo "  docker run -it --rm $IMAGE_NAME:latest"
echo ""

# Optional: Run basic tests
read -p "Run basic tests? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running basic tests..."

    echo "Testing Python version..."
    docker run --rm "$IMAGE_NAME:latest" python --version

    echo "Testing basic imports..."
    docker run --rm "$IMAGE_NAME:latest" python -c "
import sys
print(f'Python: {sys.version}')
print('✓ Basic Python functionality working')
"

    echo "Testing installed tools..."
    docker run --rm "$IMAGE_NAME:latest" bash -c "
echo 'Testing development tools...'
which gcc && echo '✓ GCC available' || echo '✗ GCC not available'
which g++ && echo '✓ G++ available' || echo '✗ G++ not available'
which make && echo '✓ Make available' || echo '✗ Make not available'
which cmake && echo '✓ CMake available' || echo '✗ CMake not available'
which git && echo '✓ Git available' || echo '✗ Git not available'
which curl && echo '✓ Curl available' || echo '✗ Curl not available'
which wget && echo '✓ Wget available' || echo '✗ Wget not available'
echo 'Testing Python tools...'
which python && echo '✓ Python available' || echo '✗ Python not available'
which pip && echo '✓ Pip available' || echo '✗ Pip not available'
pip --version && echo '✓ Pip working' || echo '✗ Pip not working'
"

    if [[ "$COMPUTE_TYPE" != "cpu" ]]; then
        echo "Testing GPU components for $COMPUTE_TYPE (CUDA arch $CUDA_ARCH)..."
        docker run --rm "$IMAGE_NAME:latest" bash -c "
echo 'Testing CUDA installation...'
which nvcc 2>/dev/null && echo '✓ NVCC available' || echo '✗ NVCC not available'
nvcc --version 2>/dev/null || echo 'NVCC version check failed'
ls -la /usr/local/cuda*/lib*/libcudart* 2>/dev/null && echo '✓ CUDA runtime libraries found' || echo '✗ CUDA runtime libraries not found'
echo \"Compute Type: $COMPUTE_TYPE, CUDA Architecture: SM$CUDA_ARCH\"
"
    fi

    echo "All tests completed!"
fi

echo "=================================================="
