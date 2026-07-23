# Ourochronos Docker Image
# Multi-stage build for minimal runtime image

# =============================================================================
# Build Stage
# =============================================================================
# Cargo.lock uses the stabilized v4 format, which requires Cargo 1.83 or
# newer. Pin a current-enough toolchain instead of silently rewriting the
# repository lockfile inside the image build.
FROM rust:1.85-bookworm AS builder

WORKDIR /build

# z3-sys links the system solver and runs bindgen during compilation. Declare
# both native dependencies instead of relying on whatever happens to be in the
# Rust builder image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        clang \
        libclang-dev \
        libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --locked && \
    cargo clean --release --package ourochronos && \
    rm -rf src

# Copy actual source code
COPY src ./src

# Build the application
RUN cargo build --release --locked

# =============================================================================
# Runtime Stage
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libz3-4 \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ouro

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/ourochronos /usr/local/bin/ourochronos

# Create directories for runtime
RUN mkdir -p /app/programs /app/audit && \
    chown -R ouro:ouro /app

USER ouro

# Default entrypoint
ENTRYPOINT ["ourochronos"]

# Show help by default
CMD ["--help"]

# =============================================================================
# Labels
# =============================================================================
LABEL org.opencontainers.image.title="Ourochronos"
LABEL org.opencontainers.image.description="Closed Timelike Curve Programming Language"
LABEL org.opencontainers.image.vendor="OUROCHRONOS Project"
LABEL org.opencontainers.image.source="https://github.com/sunyaisblank/Ourochronos"
