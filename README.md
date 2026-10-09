# Ourochronos

A programming language where programs read their own future.

Ourochronos implements classical closed-timelike-curve consistency as an executable language and verification system. A program defines a transformation F over configurable finite memory. Standard mode follows one orbit; `--global` solves `F(S)=S` symbolically and independently replays the witness; `--all-fixed` proves uniqueness or exhibits ambiguity; `--verify` checks source properties over every point fixed state; and `--recurrent` exhaustively classifies all cycles in a declared small domain. Deutsch mode also accepts a reached deterministic cycle as its uniform stationary distribution. The distinction matters: Aaronson and Watrous's PSPACE theorem uses stationary distributions, not point states alone. The exact assumptions and concrete limits are stated in the [theory guide](docs/theory.md).

## The two memories

Every program sees two memory spaces. The *anamnesis* is read-only and holds the state of the world as it will be at the end of the run; reading it with `ORACLE` is how the future speaks. The *present* is read-write and holds the world being built; writing it with `PROPHECY` is how this run answers. Execution repeats the program, feeding each run's present back as the next run's anamnesis, until the two agree.

```ourochronos
# Ask the future for a factor of 15
0 ORACLE
DUP 15 SWAP MOD 0 EQ
IF {
    DUP 0 PROPHECY      # correct: stabilise the timeline
    OUTPUT
} ELSE {
    1 ADD 0 PROPHECY    # wrong: perturb, forcing another epoch
}
```

Running from the zero seed prints `3`. The program verifies the future's claim and perturbs invalid candidates; the iterative simulator follows that orbit to the first valid fixed point. Other fixed points, such as `5`, may exist and can be reached from other seeds.

In standard point-consistency mode, `0 ORACLE NOT 0 PROPHECY` oscillates with period 2 and is diagnosed as a grandfather paradox. With `--deutsch`, the same orbit is a valid stationary ensemble assigning probability 1/2 to each state. A Deutsch computation has a valid decision readout only when every state in the cycle agrees; the CLI reports an ambiguous readout otherwise. Epoch exhaustion remains an unknown result, distinct from a detected point-cycle.

## Install and run

Building from a repository checkout requires Rust 1.85 or newer. The symbolic
backend also links the native Z3 library.

The platform contract covers Linux x86-64 and native Windows x86-64 portable
execution. Native effect commits require Linux. The Windows runtime needs
`libz3.dll`; the official Z3 Windows build also uses the
[Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
The GNU Linux archive is built on Debian 12: glibc 2.36 or newer, GNU C++
runtime with `GLIBCXX_3.4.30`, and the normal GNU loader/GCC runtime are required.
It carries Z3 beside the executable. Native Windows qualification uses Windows
11 x64 with the installed VC++ runtime; the archive carries `libz3.dll`.

```bash
git clone https://github.com/sunyaisblank/Ourochronos.git
cd Ourochronos
cargo install --path . --locked

ourochronos examples/hello.ouro
ourochronos examples/paradox.ouro     # exits 2: oscillation, period 2
ourochronos repl

# Mandatory analysis without execution
ourochronos examples/hello.ouro --check

# Deterministic per-source objects, linked bytecode, and a portable package
mkdir hello-objects
ourochronos examples/hello.ouro --emit-objects hello-objects
ourochronos link hello-linked.ourobc hello-objects/*.ouroobj
ourochronos examples/hello.ouro --emit-bytecode hello.ourobc
ourochronos examples/hello.ouro --build hello.ouropkg
ourochronos run-package hello.ouropkg
ourochronos examples/hello.ouro --build-executable hello-native
./hello-native

# Independently check a small, complete finite no-fixed-point certificate
ourochronos examples/finite_flip.ouro --emit-bytecode flip.ourobc --memory-cells 1
ourochronos prove-finite flip.ourobc flip.ourofp 1 64
ourochronos check-finite flip.ourobc flip.ourofp 1 64

# Save actual classical VM progress; a pause exits 3 (UNKNOWN)
ourochronos examples/resumable_countdown.ouro --emit-bytecode count.ourobc --memory-cells 1
ourochronos halt-slice count.ourobc count.ourocp 4 1 64
ourochronos halt-resume count.ourobc count.ourocp 64 1 64

# Check coupled affine feedback and every recurrent-class parity readout
ourochronos examples/affine_feedback.ouro --emit-bytecode affine.ourobc --memory-cells 4
ourochronos analyse-affine affine.ourobc 1 4 64

# Linux process supervision also covers an uninterruptible solver call
ourochronos isolate 2000 256 examples/finite_flip.ouro --global --memory-cells 1

# Synthesize, independently replay, and embed an explicit point state
ourochronos examples/case_studies/mutual_exclusion.ouro \
  --build mutual.ouropkg --embed-global-witness --memory-cells 1

# Prove uniform generation and exhaustively certify exact input "1"
ourochronos examples/pspace_contract.ouro --verify-family 1 \
  --memory-cells 1 --state-bits 1 --state-limit 2
```

The symbolic backend links Z3. On Debian/Ubuntu, install `libz3-dev` (and a
Clang/libclang development package if the Rust binding generator requires it)
before building.

| Flag | Purpose |
|------|---------|
| `--diagnostic` | Record the full trajectory; diagnose paradoxes and divergence |
| `--deutsch` | Accept deterministic cycles as uniform Deutsch stationary ensembles |
| `--stationary` | Solve a source-level exact rational `MARKOV` model |
| `--quantum-fixed` | Verify a source-level qubit `QCHANNEL` over every fixed density |
| `--action` | Explore several seeds and select the least-action fixed point |
| `--seeds <n>`, `--seed <n>` | Control the action-mode search |
| `--typecheck` | Print the mandatory temporal type/effect analysis |
| `--check` | Run all compiler, linker, bytecode-verifier, type, and region gates without executing |
| `--emit-object <file>` | Write the entry source's relocatable object; dependencies remain typed imports |
| `--emit-objects <directory>` | Write every source/prelude relocatable object; the directory must be empty |
| `--emit-bytecode <file>` | Write deterministic validated linked bytecode |
| `--build <file>` | Write a deterministic portable bytecode package |
| `--build-executable <file>` | Copy this platform's runtime and embed a validated package as a directly runnable launcher |
| `--embed-global-witness` | Build modifier: solve and embed an independently replayed initial point state |
| `--runtime-global-package` | Build modifier: require the exact versioned Z3 point-solver contract at runtime |
| `--global` | Globally solve a point fixed state and replay it in the VM |
| `--all-fixed` | Prove zero/unique/multiple point fixed states |
| `--verify` | Verify source `PROPERTY` declarations over every fixed state |
| `--verify-family <bitstring>` | Prove restricted uniform generation and exhaustively verify that exact finite deterministic `FAMILY` input |
| `--recurrent` | Classify every cycle and basin in a bounded closed domain |
| `--state-bits <n>`, `--state-limit <n>` | Size the explicit recurrent domain |
| `--solver-timeout <ms>`, `--loop-unroll <n>` | Bound symbolic analysis |
| `--artifact <file>`, `--dot <file>` | Export JSON evidence or a complete Graphviz graph |
| `--smt` | Export the typed temporal IR as SMT-LIB2 |
| `--fast` | Prevalidated bytecode dispatch for programs with no temporal operations |
| `--max-inst <n>` | Instruction budget per epoch |
| `--memory-cells <n>` | Temporal-memory width (default 65,536) |
| `--resources` | Print exact finite resource bounds and the instance/family caveat |
| `--halting-bound <n>` | Sound bounded observation of deterministic classical halting |
| `--effects decline\|unrestricted` | Whether supported live host inputs may be captured before deterministic replay (default: decline) |
| `--allow-file-read <path>`, `--allow-file-write <path>` | Freeze one exact read path or authorize one selected exact write path; repeatable |
| `--network-input <host:port>=<file>` | Freeze one exact socket receive stream without granting sends; repeatable |
| `--allow-network-send <host:port>` | Authorize one exact selected TCP send; repeatable |
| `--allow-process <descriptor>` | Freeze a command result and authorize that exact selected shell command; repeatable |
| `--allow-sleep-ms <n>` | Authorize selected sleeps no longer than the exact bound |
| `--strict`, `--permissive` | Error-handling policy |
| `--audit [file]`, `--audit-json` | Structured logging of the run and its outcome |
| `--provenance-limit <n>` | Saturation limit for causal-dependency tracking |
| `--lsp` | Language server (build with `--features lsp`) |

Exit codes: 0 consistent/proven/decided/halted, 1 error or unsupported analysis, 2 paradox/nonexistence/ambiguity/refutation/vacuity, 3 resource-bounded unknown.

Named cells are retained declarations: `TEMPORAL future @ 7 DEFAULT 99;`
creates a typed, import-visible schema and `future` reads anamnesis cell 7.
`DEFAULT 99` is artifact metadata today; present and the initial anamnesis
still start at zero. `PROPERTY` accepts numeric or named cells and bounded
`NOT`/`AND`/`OR` predicates. Verification artifacts report the exact sorted
touched-address/name slice without claiming incremental solver reuse.

The CLI always resolves typed HIR, checks structural stack semantics, lowers
and links bytecode, validates its CFG, and enforces type/effect and temporal
region rules before executing or writing an artifact. `ourochronos link`
decodes and validates one or more `OUROOBJ` objects, links them in deterministic
module-name order, verifies the resulting CFG, and emits a bytecode envelope. The bytecode
VM implements 97 of the 99 primitive spellings; the two dynamic legacy FFI
spellings are rejected because linked `FOREIGN` declarations instead lower to
a typed `CallForeign` instruction. The standard point-orbit path uses explicit
language frames and paged copy-on-write temporal memory. Global,
all-fixed, property, and SMT modes lower that linked bytecode directly to typed
temporal IR and replay SAT witnesses in the bytecode VM. Recurrent analysis
enumerates its closed finite domain through that same bytecode VM. Diagnostic
and deterministic Deutsch orbit policies also share the bytecode epoch
transition. Action-guided orbit selection also evaluates linked bytecode.
`MARKOV` and `QCHANNEL` are separate finite declarative models rather than
alternative executors for ordinary program bodies.
`FAMILY` is also no longer declaration-only at the concrete-instance boundary:
`--verify-family x` retains the exact nonempty Boolean input `x`, freezes it
identically for every state transition, and enumerates the exact configured
finite domain. It rejects
partial, non-closed, observation-dependent, or effectful transitions, measures
the declared cell/work/step polynomials at `n=|x|`, and requires exactly one common
Boolean readout across every recurrent class. Its versioned certificate retains
the input, full contract, and complete successor/instruction/per-transition-
workspace/readout proof table; a VM-independent checker reconstructs closure,
recurrent classes, resource maxima, and the decision before the bytecode
recheck reruns enumeration.

The CLI also constructs a proof-carrying restricted uniform generator. One
exact admitted bytecode template with a complete acyclic reachable control-flow
graph is reused for all nonempty inputs; frozen `INPUT` is its only permitted
lowering boundary. A retained
polynomial width rule is proved below `CTC_CELLS(n)` for every `n>=1`; and the
specializer's work and descriptor size are canonical linear polynomials. An
explicit `--memory-cells m` selects the constant width rule `m`; otherwise the
declared CTC bound is used as the rule. The aggregate JSON artifact contains
the exact template bytes, generator proof, and finite-instance proof without
promoting finite enumeration to family-wide runtime totality/readout invariance.
Nature's ideal selector remains an explicit external model assumption.

For the recognized sparse-routing templates, `ProjectionFamilyCertificate`
goes further. The exact straight-line body may copy any fixed list of in-domain
anamnesis cells to present cells and write fitting constants; later writes to a
cell win and unwritten cells are zero. Assignments may also apply cell-wise
bitwise `AND`, `OR`, or `XOR` to two anamnesis cells; these preserve every
declared cell width. The same operators may use a fitting constant operand,
including an all-ones `XOR` mask lowered to per-bit `NOT`; constant right shifts
lower to routing plus zero-fill. Its readout is a Boolean constant or the retained first
input bit, hence independent of temporal state. The checker
proves for every nonempty input that the transition is total and closed, the
derived chronology/step polynomials fit the contract, and all recurrent
classes have the same decision. It also derives a polynomial output-wire bound
and emits an explicit Boolean circuit containing the exact temporal, constant,
bitwise-gate, and decision nodes. The circuit checker regenerates every node from
the theorem. Whenever finite enumeration is available, the CLI compares every
circuit successor and per-state decision against the independent VM table;
disagreement is an internal error, not a user-visible refutation. The original
one-cell projection is the smallest member of this subclass.
Complete-UNSAT results are emitted only when the IR is complete, executable gas
is sufficient, and a bounded evidence envelope containing the exact Z3 query
and proof AST reproduces in a fresh solver context. JSON artifacts retain both;
the public verifier can replay them again. This is Z3-backed verification, not
an independent proof-kernel check.
Every public source-facing execution, proof, SMT, recurrence, halting, and
object-build facade now shares one canonical admission judgment: type/effect
checking, exact-width region validation, HIR resolution, structural semantics,
bytecode lowering/linking, runtime-capability checking, and independent CFG
verification must all succeed before an immutable executable is sealed. The
retired AST executor and source temporal lowerer compile only in test builds as
differential oracles. `build.rs` rejects production facade drift back to those
raw paths. The exact cross-component invariants are defined in the
[alignment contract](docs/alignment_contract.md).
The optimized pure-program route seals an immutable validated bytecode artifact
and skips redundant validation scans; it deliberately uses the same dispatcher
so stack behavior, errors, observations, and gas remain identical.
Linked foreign declarations retain a narrow, exact `u64`/`i64` scalar
signature. Library users may attach a safe process-local host table; the CLI
does not guess host bindings, temporal solvers decline the external effect,
and portable packages reject foreign dependencies until the manifest can
declare them.
`.ouropkg` is a platform-neutral bytecode container; `--build-executable`
produces a platform-specific copy of the current runtime with those validated
package bytes embedded. Packaged execution rejects foreign calls, live input,
and ungranted external observations/effects; the VM models system primitives
through frozen snapshots and selected-only effect intents.
The current `OUROPK` v3 manifest records one closed policy: zero-seed orbit,
embedded point witness, or runtime global-point solving with the exact
versioned Z3 contract. It never substitutes orbit search for a declared global
solve.

Compiler outputs now wrap `OUROBC` v2 or `OUROPK` v3 in `OUROPA` v1 to retain
linked source names, lengths, digest claims, and optional program-bound evidence.
The checksum detects corruption; it does not authenticate the producer or verify
the evidence. Source text is not embedded. Rust consumers of CLI output should
use `PortableArtifact::from_bytes`, which also accepts legacy formats with source
provenance marked unavailable. The old bytecode/package decoders retain their
formats and reject the envelope; `legacy_bytes` exports a payload when needed.
Public error enums and the property `Unknown` result have additional variants or
fields; downstream exhaustive matches must be updated when migrating this
candidate. `BytecodeTimeLoopConfig` gains optional `diagnostic_sources`; use
`..Default::default()` when no manifest is retained. Ordinary orbit and packaged
runtime failures now identify the fetched record, original file claim and byte
range even after source removal. `BytecodeVm::run_diagnostic` exposes the same
structured location to embeddings. Missing metadata retains the numeric source
identity; exact source text is required to derive line numbers. Names are escaped
and never opened as paths.

Classical checkpoints bind the exact program, frozen input, configuration and
cumulative ceiling. Reload validates reachability with bounded prefix replay;
subsequent slices continue from the saved state. The CLI uses an empty input
tape and default stack/call/output limits; Rust embedding can supply frozen
input and cancellation. Temporal operations, foreign calls, effects and heap
collections are outside this recovery profile. Budget exhaustion stays
`UNKNOWN`, and changing the ceiling requires a new query.

The affine API checks parity assumptions, relations between initial and final
Boolean states, protected bits, and composition of successive epoch components.
It returns an explicit counterexample or unsatisfiable assumptions instead of a
vacuous positive contract. A separate recurrence certificate covers affine maps
on at most 64 Boolean cells that stabilize by their dimension; it proves every
recurrent class is a point state and returns either a uniform parity readout or
two disagreeing fixed states. The model checkers are independent of bytecode
extraction. These profiles do not admit general word arithmetic, heap or effects.

For configurable all-class stochastic examples, run
`cargo run --release --locked --example stochastic_models -- vm 2 3 random`
(or `tag` for disagreeing classes). `absorbing N` and `cycle N B` exercise
sparse states and exact large denominators, with analytic/residual checks.
Linux callers can add `supervise 1000 128` before the model arguments to contain
exact arithmetic and allocation failures. Growth measurements and the bounded
solver reuse comparison are in the [case studies](docs/case_studies.md).
The isolated higher-dimensional quantum SDP prototype is reproducible from
`examples/quantum_fixed_space.py`; its results are numerical estimates with
explicit uncertainty, including a near-identity counterexample. It adds no
Python dependency to the language runtime.

Linux `isolate` runs supported source analysis in a child with an address-space
limit, CPU limit, bounded captured output and parent-enforced wall time. It
accepts analysis flags and withholds host capabilities and file-producing flags.
Timeout, cancellation, output exhaustion or signal termination supplies no proof;
partial result output is withheld. Rust callers can use
`runtime::isolation::run_bounded` with cancellation. This process resource profile
requires a cooperating process group and is distinct from filesystem/network
confinement; trusted native callbacks that need confinement require an external
worker with the relevant OS policy.

This candidate is `0.3.0-rc.1`: the minor-version change covers additive
public result/config variants and unsafe loader API migration. `--version`
reports the actual platform, runtime ABI and compiled optional features.
`DynamicLibraryManager::load` and `ExtendedFFIContext::load_library` now require
an unsafe call whose caller guarantees trusted native initializers/destructors.
The safe host table rejects incorrect argument counts before a callback.
For passive portable scalar dependencies and frozen INPUT/observations, use
`runtime::host_manifest::HostManifest` and the executable
`cargo run --locked --example host_snapshot`. Checking requires the expected
program plus an independently approved snapshot digest. Decoding grants no
host capabilities and attaches no callbacks; packages still reject foreign
and effect dependencies. The embedder must execute the checked program and
supply the returned INPUT explicitly. Snapshot Debug output omits scalar data.

For a published release, sign the final archive checksums with a maintainer-held
release key and distribute its fingerprint through a trusted channel. Verify that
signature before unpacking and verify every listed file hash afterwards. Sign
Windows launchers after their package is appended; subsequent appending invalidates
the signed identity. An unpublished qualification candidate is labelled unsigned.
Envelope checksums and a public key supplied beside an artifact alone do not
establish publisher authenticity.

`--allow-process` descriptors are binary-safe: `OUROPROCESS/1\n`, an `i32`
exit-code line, a decimal UTF-8 command-byte-length line, then exactly that
many command bytes followed immediately by the frozen output bytes. Candidate
epochs never open files, connect to hosts, spawn commands, or sleep. The
selected batch is capability-preflighted and idempotency-ledgered before the
native adapter runs. Application is at-most-once for a commit token and digest:
the same success or failure is returned on replay, while a host failure can
leave the already applied prefix in place. Missing file targets are created
only when their first file intent is reached. Native effect commits are supported
on Linux only; other platforms reject the batch before any host call. On Linux, existing targets remain
open across the batch and are checked by device/inode identity; missing targets
retain a verified parent directory and use `openat` with exclusive,
no-follow creation. Other platforms fail closed for exact native file commits
because the portable standard library exposes no equivalent stable identity.
Embedded package witnesses carry
a recomputable digest binding the manifest, linked bytecode, canonical state,
and replay count; the runtime also independently replays the state.

Admission is bounded before execution: one source file is at most 8 MiB; a
module graph retains at most 256 modules, 128 imports per module, depth 64,
4,096 edges, 64 MiB of source, 1,250,000 tokens, and 1,000,000 expanded
statements; parser nesting is at most 64 and source MARKOV declarations at most
256 states. The LSP retains at most 256 documents, 64 MiB aggregate source, and
2,000,000 aggregate analysis units. Default frozen file, endpoint, and process
observations have per-item/count limits and separate 64 MiB aggregate byte
budgets. Each VM epoch separately caps buffered output and dynamic
collections/buffers at 64 MiB. A temporal search retains at most 256 MiB of
orbit state; action mode accepts at most 1,024 seeds and its epoch cache retains
at most 256 MiB. Explicit recurrence and finite FAMILY verification accept at
most 262,144 states and cap aggregate work at 100,000,000 instructions,
1,000,000 output items, and 64 MiB of output. `OUROBC` is capped at 64 MiB and one million instructions; CLI object
linking accepts at most 4,096 objects within a 128 MiB aggregate input budget;
dense exact-result paths and `OUROPK` memory stop at 1,048,576 cells, while a
package execution stops at 10,000,000 instructions. Decoders check lengths
before allocation, and artifacts/packages are structurally validated and
independently CFG-verified before dispatch or witness replay.
The exact-query plus Z3-proof payload for one complete-UNSAT certificate is
capped at 64 MiB; excess evidence yields `UNKNOWN`.

## The Action Principle

Many programs have several fixed points, and the all-zero timeline is usually one of them; a naive search finds it first and reports that nothing happened. Action-guided mode assigns every discovered fixed point a cost that penalises trivial and temporally independent states and rewards causal depth and output, then selects the minimum. The weights are derived from the program's own temporal footprint. `examples/sat.ouro` and `examples/quantum_suicide.ouro` show the difference it makes.

The configured `--seed` timeline is one of the action search candidates, so a
success reached only from that seed still produces the ordinary selected-batch
ledger and cannot bypass effect accounting. Deterministic Deutsch mode also
has a unique selected batch for a point fixed state (period one); an authorized
adapter may commit it once. Effects on longer recurrent cycles remain withheld
because no single chronology has been selected.

## Documentation and examples

The [specification](docs/specification.md) defines the abstract machine, typed finite IR, solver proof obligations, instruction set, and runtime behavior. The [alignment contract](docs/alignment_contract.md) defines the formal admission judgment and cross-mode invariants. The [theory guide](docs/theory.md) gives the Turing-completeness construction, deterministic/stochastic/quantum fixed-point distinctions, exact PSPACE conditions, all implemented forms, and the finite barrier to `Delta^0_2` and halting. [Positioning](docs/positioning.md) compares the language with esolangs, constraint/data-flow tools, and the broader formal-methods family without overclaiming. The [case studies](docs/case_studies.md) cover the three strongest applications: bounded self-consistency model checking, circular data flow, and retrocausal simulation/game rules. The [completion audit](docs/completion_audit.md) maps claims to code and tests.

## Development and local release qualification

Choose a mode from the [executable case studies](docs/case_studies.md): point
solving checks one fixed-state constraint, recurrence explores every state in
a declared small domain, exact stochastic analysis checks every stationary
class, and quantum exploration provides numerical estimates with uncertainty.
The configurable `application_studies` example accepts exclusion eligibility,
modular dataflow gains/biases and rule-game successor tables, with `--save` for
a new result file. Its ordinary Linux invocation includes process supervision.

Use the locked Rust 1.85 toolchain with clang/libclang and native Z3 development
headers. On Debian/Ubuntu these come from `clang libclang-dev libz3-dev`.
Optional `lsp` builds the stdio language server; optional `dynamic-ffi` adds the
explicitly unsafe native library adapter. Neither restores retired execution
facades. For a focused change, run the affected targets, then these integration
gates on the candidate:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo test --release --test main --all-features --locked -- --ignored
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features --locked
cargo package --all-features --locked
cargo build --all-features --locked
python3 scripts/qualify_interfaces.py --runtime target/debug/ourochronos
lean formal/CounterMachine.lean  # exact toolchain in formal/lean-toolchain
python3 scripts/check_solver_usage.py
cargo audit --ignore RUSTSEC-2026-0295
```

For an intentionally dirty local candidate, `cargo package --allow-dirty`
preserves the supplied worktree; CI uses a clean checkout. The packaged `.crate`
contains the conformance evaluators, finite checker, Lean source, examples,
research prototypes, tests and documentation. Rebuild/install that archive in
a fresh temporary prefix, exercise its actual executable, then remove only
that prefix. Use a fresh Cargo target directory when rebuilding another archive
with the same package version: normalized archive timestamps can otherwise
reuse stale own-package artifacts. If retaining a target cache, run
`cargo clean --release --package ourochronos` before that rebuild.
Upgrade a user installation with the same locked path/features
and `cargo install --force`; rollback reinstalls the retained prior verified
archive or restores the prior runtime directory. `cargo uninstall ourochronos`
removes a Cargo-managed executable; it preserves programs and saved results.
An extracted runtime archive is removed by deleting its own directory and PATH
entry, with no registry or persistent-service changes.

`scripts/package_candidate.py --help` documents deterministic unsigned Linux
and Windows archive construction from the verified `.crate` and qualified
runtime/native Z3 bytes. It rejects stale source archives or changing inputs
and includes SHA256SUMS, provenance, dependency/license inventory and notices.
Build Linux release archives with literal `$ORIGIN` native-library search and
verify the extracted Z3 is selected with inherited loader overrides removed.
Linux archives require the qualified glibc/GNU C++ runtime baseline; Windows
archives require the system VC++ x64 runtime. Dependency checksums establish
registry/archive identity rather than publisher authenticity. Reproduce builds
from identical packaged bytes/toolchains/native libraries and compare runtime
and archive hashes before release. The release signing policy above applies
to publication; local candidates remain explicitly unsigned.
Record and retain the immutable builder image and native-library hashes for
that reproduction; package-manager setup against a later repository is not an
identical native environment. `scripts/windows.Dockerfile` cross-builds the
Windows GNU runtime; execute `scripts/qualify_windows.ps1` on native Windows
against the extracted runtime. CI also runs the actual Linux interface gate,
excluded stress checks, strict API documentation, Lean model and bounded
quantum research campaign. Local qualification does not establish a remote
CI result or publish a release.

The locked Z3 Rust binding remains affected by
[RUSTSEC-2026-0295](https://rustsec.org/advisories/RUSTSEC-2026-0295.html).
Its faulty `ApplyResult::clone` API has no call path in the reviewed product's
private Solver/Model/AST consumers. CI checks their exact source hashes and the
dependency version/checksum before applying this one advisory exception; any
new solver consumer or changed reviewed code requires renewed analysis. This
guard records an unused-API disposition, not a dependency fix or a formal
reachability proof. Windows retains affected dependency code. Embedders using
additional Z3 APIs must assess that application code separately.

## Licence

MIT. The theoretical foundations rest on Deutsch's CTC self-consistency model (1991) and the Aaronson and Watrous PSPACE characterisation (2008).
