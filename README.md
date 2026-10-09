# Ourochronos

## Description

Ourochronos is an experimental programming language for programs that read a
proposed future state. `ORACLE` reads that state; `PROPHECY` writes the present
state. The runtime repeats execution to look for a self-consistent result.

It supports stack-based programs, fixed-point analysis, portable bytecode
packages, and stochastic and numerical quantum experiments. Analyses use finite
resource bounds and can return an inconclusive result.

## Installation

Supported runtimes are Linux x86-64 and Windows x86-64. Z3 is mandatory.
Native host-effect commits and process supervision require Linux.

### Linux

Install Rust through rustup first. The repository selects Rust 1.85.0
automatically. On Debian or Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y git build-essential clang libclang-dev libz3-dev
git clone https://github.com/sunyaisblank/Ourochronos.git
cd Ourochronos
cargo install --path . --locked
```

Ensure `~/.cargo/bin` is on your PATH. To enable the language server, add
`--features lsp` to the installation command. To reinstall or upgrade, add
`--force`; to uninstall, run `cargo uninstall ourochronos`.

### Windows

From the repository directory on Linux or WSL with Docker, build the Windows
runtime and its Z3 library:

```sh
docker build -f scripts/windows.Dockerfile -t ourochronos-windows .
ouro_container=$(docker create ourochronos-windows)
mkdir -p windows-runtime
docker cp "$ouro_container:/artifact/." windows-runtime/
docker rm "$ouro_container"
```

Copy `windows-runtime` and `examples` to Windows. Keep `libz3.dll` beside
`ourochronos.exe` and install the
[Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
From the directory containing both folders, use PowerShell:

```powershell
.\windows-runtime\ourochronos.exe --version
.\windows-runtime\ourochronos.exe examples\hello.ouro
```

Add the runtime directory to your PATH to use `ourochronos` directly.

## Usage

Run the included example, start the interactive prompt, or view available commands:

```sh
ourochronos examples/hello.ouro
ourochronos repl
ourochronos --help
```

Programs use `.ouro` files. For example, save this as `answer.ouro`:

```text
40 2 ADD OUTPUT
```

Run `ourochronos answer.ouro` to print `[42]`.

Check a program without executing it, or solve for a point fixed state:

```sh
ourochronos examples/case_studies/mutual_exclusion.ouro --check
ourochronos examples/case_studies/mutual_exclusion.ouro --global --memory-cells 1
```

Build and execute a portable bytecode package:

```sh
ourochronos examples/hello.ouro --build hello.ouropkg
ourochronos run-package hello.ouropkg
```

The language server uses `ourochronos --lsp` when built with LSP support.
Exit codes are `0` for success, `1` for errors or unsupported operations, `2` for
inconsistent, ambiguous or refuted results, and `3` for resource-bounded unknowns.

## Licence

Ourochronos is licensed under the [MIT licence](LICENSE).
