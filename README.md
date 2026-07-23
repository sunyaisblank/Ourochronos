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

```bash
git clone https://github.com/sunyaisblank/Ourochronos.git
cd Ourochronos
cargo install --path .

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

# Synthesize, independently replay, and embed an explicit point state
ourochronos examples/case_studies/mutual_exclusion.ouro \
  --build mutual.ouropkg --embed-global-witness --memory-cells 1
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
module-name order, verifies the resulting CFG, and emits `OUROBC`. The bytecode
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
The public source-level `Executor` is likewise a compatibility facade: it
type-checks and resolves its `Program`, runs mandatory semantic analysis,
compiles bytecode, independently verifies the CFG, and only then dispatches the
bytecode VM. The retired direct AST walker is private and compiled only for
differential tests. Similarly, the library's raw source-AST `SmtEncoder` is a
backwards-compatible parity oracle; the CLI and proof APIs lower verified
linked bytecode instead.
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

`--allow-process` descriptors are binary-safe: `OUROPROCESS/1\n`, an `i32`
exit-code line, a decimal UTF-8 command-byte-length line, then exactly that
many command bytes followed immediately by the frozen output bytes. Candidate
epochs never open files, connect to hosts, spawn commands, or sleep. The
selected batch is capability-preflighted and idempotency-ledgered before the
native adapter runs. Application is at-most-once for a commit token and digest:
the same success or failure is returned on replay, while a host failure can
leave the already applied prefix in place. Missing file targets are created
only when their first file intent is reached. On Unix, existing targets remain
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
at most 256 MiB. Explicit recurrence accepts at most 262,144 states and caps
aggregate work at 100,000,000 instructions, 1,000,000 output items, and 64 MiB
of output. `OUROBC` is capped at 64 MiB and one million instructions; CLI object
linking accepts at most 4,096 objects within a 128 MiB aggregate input budget;
dense exact-result paths and `OUROPK` memory stop at 1,048,576 cells, while a
package execution stops at 10,000,000 instructions. Decoders check lengths
before allocation, and artifacts/packages are structurally validated and
independently CFG-verified before dispatch or witness replay.

## The Action Principle

Many programs have several fixed points, and the all-zero timeline is usually one of them; a naive search finds it first and reports that nothing happened. Action-guided mode assigns every discovered fixed point a cost that penalises trivial and temporally independent states and rewards causal depth and output, then selects the minimum. The weights are derived from the program's own temporal footprint. `examples/sat.ouro` and `examples/quantum_suicide.ouro` show the difference it makes.

The configured `--seed` timeline is one of the action search candidates, so a
success reached only from that seed still produces the ordinary selected-batch
ledger and cannot bypass effect accounting. Deterministic Deutsch mode also
has a unique selected batch for a point fixed state (period one); an authorized
adapter may commit it once. Effects on longer recurrent cycles remain withheld
because no single chronology has been selected.

## Documentation and examples

The [specification](docs/specification.md) defines the abstract machine, typed finite IR, solver proof obligations, instruction set, and runtime behavior. The [theory guide](docs/theory.md) gives the Turing-completeness construction, deterministic/stochastic/quantum fixed-point distinctions, exact PSPACE conditions, all implemented forms, and the finite barrier to `Delta^0_2` and halting. [Positioning](docs/positioning.md) compares the language with esolangs, constraint/data-flow tools, and the broader formal-methods family without overclaiming. The [case studies](docs/case_studies.md) cover the three strongest applications: bounded self-consistency model checking, circular data flow, and retrocausal simulation/game rules. The [completion audit](docs/completion_audit.md) maps claims to code and tests.

## Licence

MIT. The theoretical foundations rest on Deutsch's CTC self-consistency model (1991) and the Aaronson and Watrous PSPACE characterisation (2008).
