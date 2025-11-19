# QDK/Chemistry Development Container Dockerfile
# This image builds on top of the QDK/Chemistry base image and adds development tools
# such as VS Code Server, Git Credential Manager, testing frameworks, etc.

ARG developmentRegistry
ARG baseImageName
ARG march
ARG cuda_arch
ARG gitHash

# Use a built QDK/Chemistry base image as base
FROM ${developmentRegistry}/${baseImageName}-${march}-${cuda_arch}:${gitHash}

# Redeclare march so it can be used in this stage
ARG march

# Set QDK_UARCH so cmake/pip knows which architecture to build for
ENV QDK_UARCH=${march}

# Install development-specific apt packages not included in base image
RUN apt-get update && apt-get install -y \
    # Code quality and debugging tools
    clang-tidy \
    clang-format \
    gdb \
    gcovr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional development tools that are not in the base image
RUN pip install --no-cache-dir \
    # Development tools
    azure-cli \
    coverage \
    pytest \
    pytest-cov \
    gcovr \
    mypy \
    ruff \
    # Documentation tools
    sphinx \
    sphinx-rtd-theme \
    myst-parser

# Install VS Code Server (remote tunnel client)
# We need to download the appropriate binary based on architecture
RUN set -x && \
    if [ "${QDK_UARCH}" = "armv8-a" ]; then \
    echo "Downloading VS Code CLI for ARM64..." && \
    curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64" -o vscode_cli.tar.gz; \
    else \
    echo "Downloading VS Code CLI for x64..." && \
    curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" -o vscode_cli.tar.gz; \
    fi && \
    tar -xf vscode_cli.tar.gz && \
    mv code /usr/local/bin/ && \
    rm vscode_cli.tar.gz && \
    chmod +x /usr/local/bin/code

# Install Git Credential Manager
RUN set -x && \
    if [ "${QDK_UARCH}" = "armv8-a" ]; then \
    echo "Installing Git Credential Manager via .NET for ARM64..." && \
    # Install .NET SDK for ARM64 \
    wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh && \
    chmod +x dotnet-install.sh && \
    ./dotnet-install.sh --channel 8.0 --install-dir /usr/share/dotnet && \
    ln -s /usr/share/dotnet/dotnet /usr/local/bin/dotnet && \
    rm dotnet-install.sh && \
    # Install GCM as a .NET tool \
    dotnet tool install -g git-credential-manager && \
    ln -s /root/.dotnet/tools/git-credential-manager /usr/local/bin/git-credential-manager; \
    else \
    echo "Installing Git Credential Manager standalone binary for x64..." && \
    GCM_VERSION=$(curl -s https://api.github.com/repos/git-ecosystem/git-credential-manager/releases/latest | grep -Po '"tag_name": "v\K[^"]*') && \
    curl -fsSL "https://github.com/git-ecosystem/git-credential-manager/releases/download/v${GCM_VERSION}/gcm-linux_amd64.${GCM_VERSION}.tar.gz" \
    -o gcm.tar.gz && \
    tar -xzf gcm.tar.gz -C /usr/local/bin && \
    rm gcm.tar.gz && \
    chmod +x /usr/local/bin/git-credential-manager; \
    fi

# Configure Git Credential Manager
RUN git config --global credential.helper manager && \
    git config --global credential.credentialStore cache && \
    git config --global credential.cacheOptions "--timeout 7200" && \
    git config --global credential.https://dev.azure.com.useHttpPath true && \
    git config --global credential.azureRepos.credential microsoft && \
    git config --global credential.github.com.provider microsoft

# Set environment variables for Git Credential Manager
ENV GCM_CREDENTIAL_STORE=cache \
    GCM_CREDENTIAL_CACHE_OPTIONS="--timeout 7200" \
    GCM_PLAINTEXT_STORE_PATH=/tmp/gcm-store \
    LC_ALL=C.utf8

# Set working directory to workspace root
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
