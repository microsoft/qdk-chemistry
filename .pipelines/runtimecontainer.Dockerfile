# QDK/Chemistry Runtime Container Dockerfile
# This image builds on top of the QDK/Chemistry base image and adds QDK/Chemistry Python packages.

ARG developmentRegistry
ARG baseImageName
ARG march
ARG cuda_arch
ARG gitHash

# Use a built QDK/Chemistry base image as base
FROM ${developmentRegistry}/${baseImageName}-${march}-${cuda_arch}:${gitHash}

# Redeclare march so it can be used in this stage
ARG march
ARG cuda_arch

# Set QDK_UARCH so cmake/pip knows which architecture to build for
ENV QDK_UARCH=${march}

# Copy QDK/Chemistry source to workspace
COPY qatk/ /workspace/qatk/

WORKDIR /workspace

# Build and install QDK/Chemistry Python package
RUN pip3 install qatk/python/ && \
    if [ "$cuda_arch" = "none" ]; then \
    pip3 install pytest && \
    python -m pytest -v qatk/python/tests/; \
    fi

# Clean up source code to reduce image size
RUN rm -rf qatk

CMD ["/bin/bash"]
