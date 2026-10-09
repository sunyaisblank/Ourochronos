# Reproducible native Windows GNU target builder; execution is qualified on Windows.
FROM rust:1.85-bookworm@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc-mingw-w64-x86-64 clang libclang-dev unzip && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-pc-windows-gnu
RUN curl --fail --location --retry 2 --output /tmp/z3.zip https://github.com/Z3Prover/z3/releases/download/z3-4.8.12/z3-4.8.12-x64-win.zip && \
    echo 'de12fb2160798a464244954236b28da597e79289f33955b170853b8bf0d1f078  /tmp/z3.zip' | sha256sum --check && \
    unzip -q /tmp/z3.zip -d /opt && rm /tmp/z3.zip && \
    cp /opt/z3-4.8.12-x64-win/bin/libz3.lib /opt/z3-4.8.12-x64-win/bin/libz3.a
ENV CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
ENV Z3_SYS_Z3_HEADER=/opt/z3-4.8.12-x64-win/include/z3.h
ENV LIBRARY_PATH=/opt/z3-4.8.12-x64-win/bin
ENV RUSTFLAGS="-Lnative=/opt/z3-4.8.12-x64-win/bin -C link-arg=-Wl,--no-insert-timestamp"
WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
RUN cargo build --release --all-features --locked --target x86_64-pc-windows-gnu
RUN mkdir /artifact && cp target/x86_64-pc-windows-gnu/release/ourochronos.exe /artifact/ && \
    cp /opt/z3-4.8.12-x64-win/bin/libz3.dll /artifact/ && cp /opt/z3-4.8.12-x64-win/LICENSE.txt /artifact/

FROM scratch
COPY --from=builder /artifact /artifact
CMD ["/cross-build-artifacts-only"]
