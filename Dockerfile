# Ourochronos Docker Image
# Multi-stage build for minimal runtime image

# =============================================================================
# Build Stage
# =============================================================================
# Cargo.lock uses the stabilized v4 format, which requires Cargo 1.83 or
# newer. Pin a current-enough toolchain instead of silently rewriting the
# repository lockfile inside the image build.
FROM rust:1.85-bookworm@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4 AS builder

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
ARG OUROCHRONOS_FEATURES=""

# Create dummy src to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    echo "fn main() {}" > build.rs

# Build dependencies (cached layer)
RUN RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN' cargo build --release --locked ${OUROCHRONOS_FEATURES:+--features "$OUROCHRONOS_FEATURES"} && \
    cargo clean --release --package ourochronos && \
    rm -rf src

# Copy actual source code
COPY src ./src
COPY build.rs ./build.rs

# Build the application
RUN RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN' cargo build --release --locked ${OUROCHRONOS_FEATURES:+--features "$OUROCHRONOS_FEATURES"}

# =============================================================================
# Runtime Stage
# =============================================================================
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587 AS runtime

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
