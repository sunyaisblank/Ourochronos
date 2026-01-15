# Ourochronos Docker Image
# Multi-stage build for minimal runtime image

# =============================================================================
# Build Stage
# =============================================================================
FROM rust:1.75-bookworm AS builder

WORKDIR /build

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release && \
    rm -rf src target/release/deps/ourochronos*

# Copy actual source code
COPY src ./src

# Build the application
RUN cargo build --release

# =============================================================================
# Runtime Stage
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ouro

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/ourochronos /usr/local/bin/ourochronos

# Copy example config
COPY ourochronos.toml.example /app/ourochronos.toml.example

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
LABEL org.opencontainers.image.source="https://github.com/ourochronos/ourochronos"
