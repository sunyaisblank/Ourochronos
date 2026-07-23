# Temporal-language and verifier completion audit

This audit covers the development objective to turn Ourochronos into a serious
general-purpose language with a finite time-travel verification core. It uses
“complete” only for the implemented, explicitly bounded claims; it does not
claim to solve arbitrary Turing-complete fixed-point problems.

This document now audits both the characterized finite verifier and its
whole-language compiler migration. “Achieved” means achieved only under the
stated finite bounds and supported lowering. Linked, validated bytecode is the
high-level execution and verifier-replay authority. Source syntax and
compatibility entry points may retain AST data structures, but the CLI and
temporal policy APIs compile them to bytecode. The crate still publicly exports
the historical source-shaped single-epoch `Executor`, but that API now performs
type/HIR/semantic admission, bytecode lowering, independent CFG verification,
and bytecode dispatch. Its former AST walker is private `cfg(test)` parity
machinery, not a production execution authority. The implementation ledger is in
`whole_language_migration.md`.

## 1. One symbolic semantics

**Status: achieved for point-state queries over linked bytecode.**

- The canonical lowerer consumes validated linked bytecode and produces a
  solver-independent typed acyclic IR (`Bool`, `Word64`, and memory arrays).
- Arithmetic, logical NOT, shifts, division by zero, address policies,
  PARADOX, procedures, conditionals, bounded loops, observations, and typed
  regions match reference-VM semantics.
- `--smt` and `GlobalFixedPointSolver` use this IR. Source-facing solver
  wrappers compile and link bytecode before invoking it; the historical source
  compiler remains differential-test and compatibility machinery, not the CLI
  proof authority.
- Unsupported operations, recursion, stack underflow, live input, and effects
  are explicit errors.
- Global, all-fixed, property, and SMT paths lower validated linked bytecode
  directly. The native lowerer has expression-for-expression parity tests
  against the source oracle across all 42 finite primitives,
  calls, branches, scopes, and bounded loops.
- The raw source-AST `SmtEncoder` remains a backwards-compatible parity oracle
  for differential tests. It is not the CLI exporter, production proof
  lowerer, or an independently trusted proof authority.

## 2. Global `F(s)=s` solving

**Status: achieved for complete finite IR; soundly bounded otherwise.**

- `--global` uses in-process Z3 instead of following the zero-seed orbit.
- SAT memory is decoded and replayed through the linked-bytecode VM; a result is
  returned only if replay finishes and `present == anamnesis`.
- Sparse Z3 array-model decoding makes witness extraction proportional to the
  represented stores rather than issuing one solver query per configured cell.
  The recorded million-cell circular-dataflow run replayed its 25-instruction
  witness at 83,456 KiB peak RSS; `case_studies.md` records the command and
  environment-specific measurement.
- Complete UNSAT is `PROVEN NO POINT FIXED STATE`.
- Loop-bounded UNSAT and solver timeout are `UNKNOWN`/exit 3.
- Tests include a fixed state unreachable from the zero orbit, logical
  negation UNSAT, PARADOX model elimination, typed regions, and exhaustive
  IR-vs-VM cardinality comparison over several two-bit transition functions.

## 3. All-fixed ambiguity and properties

**Status: achieved for point fixed states.**

- `--all-fixed` blocks the first full finite memory model, then proves
  uniqueness or returns two independently replayed witnesses and differing
  cells.
- `TEMPORAL name @ address DEFAULT value;` is a located, typed HIR declaration
  visible through imports. Its DEFAULT is retained in object metadata; it does
  not seed present or alter the zero-initialized fixed-point function.
- `PROPERTY` supports named or numeric cells, unsigned comparisons, and a
  bounded Boolean AST with `NOT`, `AND`, `OR`, and parentheses.
- `--verify` searches for a violating fixed state. Outcomes are PROVEN,
  REFUTED with replayed counterexample, VACUOUS when no fixed state exists, or
  UNKNOWN.
- `--artifact` emits versioned JSON with query digests, completeness,
  witnesses, outputs, replay status, and a deterministic named-cell/touched-
  address slice. The slice is a relevance report, not an incremental-speed
  claim.

## 4. Complete recurrent analysis

**Status: achieved for explicit closed small-state domains.**

- `--recurrent --memory-cells m --state-bits b` evaluates all `2^(m*b)`
  states up to `--state-limit`.
- It refuses an undefined, effectful, non-total, non-closed, or oversized
  transition system.
- It classifies every directed cycle, point fixed state, state-to-class map,
  transient distance, basin size, and cycle-wide output agreement.
- `--dot` emits every state and edge as Graphviz DOT.

## 5. Finite temporal regions in the general language

**Status: achieved as an enforced finite core.**

- `TEMPORAL base size BITS n { body }` uses base-relative addresses, local
  bounds policy, n-bit writes, range validation, present-state rollback on
  failure, and identical VM/IR lowering. Legacy scopes default to 64 bits.
- Nested regions are rejected to keep translation unambiguous.
- `TemporalRegionReport` is enforced before every execution and verification
  mode; `--typecheck` additionally prints it. It rejects recursive calls and
  operations without total deterministic IR semantics inside a region while
  documenting ordinary host effects outside it.
- The surrounding language retains procedures, loops, quotations, dynamic
  structures, strings, files, networking, FFI, and an idealized
  Turing-completeness construction. Effects during fixed-point search remain
  gated unless explicitly modeled. Exact frozen observation transcripts,
  staged intents, rollback, selection, file preconditions, capability
  preflight, and truthful at-most-once success/failure replay form the native
  bytecode transaction boundary. Dynamic FFI remains intentionally declined.

## 6. Formal and machine-readable treatment

**Status: achieved at executable-semantics level.**

- `specification.md` defines the typed lowering judgment, fixed-point formula,
  region address/value rules, replay obligation, and complete-vs-bounded proof
  rule.
- `theory.md` separates point, recurrent/stationary, stochastic, quantum,
  PSPACE-model, and unbounded-computability claims.
- Constraint digests identify exact SMT artifacts. Positive evidence is
  independently replay-checkable; JSON and SMT are interchange formats.
- A machine-checked Coq/Lean metatheory and independently checkable Z3 proof
  objects are not implemented; an UNSAT certificate currently records backend,
  completeness, and exact-query digest.

## 7. Strongest three case studies

**Status: achieved and reproducibly benchmarked.**

1. `mutual_exclusion.ouro`: two fixed schedules, both universally proven safe.
2. `circular_dataflow.ouro`: simultaneous equations with a unique solution
   `x=3, y=2`.
3. `retrocausal_game.ouro`: complete four-state graph with one period-two
   history and two equilibria.

`cargo run --release --example bench_case_studies -- 100` validates expected
results on every sample and reports median/p95 timings locally. `case_studies.md`
contains commands and interpretations.

## 8. Competitive claim

**Status: narrowly defensible, broadly qualified.**

Ourochronos can plausibly claim unusually strong formal and executable support
among time-travel/esoteric languages: global solving, all-model ambiguity,
universal properties, recurrent classes, replay, and artifacts live in one
language. It does not surpass mature theorem provers/model checkers in general
verification breadth. `positioning.md` states that boundary and the roadmap
needed to compete with that broader family.

## Verification gate

Executed on 2026-07-13:

```text
cargo test --all-targets
  390 library tests passed
  200 integration/CLI/benchmark tests passed
  11 stress tests intentionally ignored
  0 failures

cargo clippy --all-targets -- -D warnings
  passed

git diff --check
  passed
```

Total active tests: **590 passed**.

The counts above are the historical gate for this verifier audit. They are not
the current whole-language migration gate; the final result is recorded below.

## Whole-language migration checkpoint

**Status: complete for the bounded implementation claims recorded here.**

The following was an earlier migration checkpoint on 2026-07-13. It is
historical and superseded by subsequent implementation changes:

```text
cargo test --all-targets
  544 library tests passed
  215 integration/CLI/benchmark tests passed
  11 stress tests intentionally ignored
  0 failures

cargo clippy --all-targets -- -D warnings
  passed
cargo fmt --all -- --check
  passed
git diff --check
  passed
```

Historical checkpoint total: **759 active tests passed**. The final
post-migration gate, also executed on 2026-07-13, supersedes it:

```text
cargo test --all-targets --all-features
  702 library tests passed
  230 integration/CLI/benchmark tests passed
  11 stress tests intentionally ignored in the ordinary profile
  0 failures

cargo test --test main --release --all-features -- --ignored --nocapture
  11 release stress tests passed
  0 failures

cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
  all passed

cargo +1.85.0 check --all-targets --all-features
cargo audit
  pinned MSRV check passed; 0 RustSec advisories

cargo build --release --all-features --example bench_case_studies --bin ourochronos
docker build -t ourochronos:test .
docker run ... circular_dataflow.ouro --global --memory-cells 2
  release build and container build passed
  container replayed [0]=3, [1]=2 in 25 instructions
```

Total ordinary-profile active tests: **932 passed**. Every one of the 11
ignored stress tests was separately exercised and passed in the release
profile. An independent final adversarial authority audit reported zero
blockers and zero high- or medium-severity production/library findings.

- File-backed compilation now always passes through the canonical module
  graph, typed HIR, mandatory structural semantics, deterministic bytecode
  lowering/linking, structural validation, independent CFG verification,
  type/effect analysis, and temporal-region enforcement.
- Standard, diagnostic, Deutsch, action, recurrent, solver replay, package,
  REPL, bounded-halting, and optimized paths execute linked bytecode with an
  iterative explicit-frame VM, bounded gas/resources, persistent paged COW
  temporal state, verified collision-safe orbit journals, and selected-epoch
  output commit.
- The public source-level `Executor` is a compile/check/verify/dispatch facade
  over that VM. Direct AST execution exists only as private test parity code.
- Deterministic `OUROOBJ`, `OUROBC`, and `OUROPK` artifacts can be emitted,
  decoded, validated, linked, and run.
- `OUROPK` v3 records an explicit orbit, embedded-point, or runtime-global
  policy. Embedded witnesses have recomputable package-bound evidence and are
  replayed; runtime-global packages declare the exact Z3 contract and never
  fall back to an orbit.
- Global, all-fixed, property, and SMT queries compile the supported finite
  subset directly from linked bytecode and replay SAT witnesses in the
  bytecode VM. Explicit recurrence evaluates every transition in that VM;
  MARKOV and QCHANNEL are separate finite declarative analyses.
- The bytecode VM implements the classical primitive core plus typed
  `CallForeign` through an explicit process host table. The linked ABI is
  restricted to bounded `u64`/`i64` scalars with at most one result; bytecode,
  verifier, object linker, and runtime all retain/check the exact descriptor.
  Temporal modes and portable packages continue to reject the external host
  dependency.
- Per-source object compilation/import generation is implemented by the public
  `compile_objects` API, including retained located MANIFEST symbols, typed
  constant/call relocations, private quotation relocation, exact foreign
  descriptors, named temporal schemas, bounded Boolean properties, and
  dependency-first initializers. REPL/LSP file analysis uses the same
  importer-relative graph; `--emit-objects` exposes the complete real object
  set and refuses nonempty output directories.
- Frozen observation transcripts participate in candidate identity. File,
  network, process, and sleep effects are selected-only intents with exact
  capabilities, file preconditions, and token-plus-digest at-most-once replay.
  Host failure can leave an applied prefix, so this is not cross-host atomicity.
- On Unix, native file application retains verified handles (or a verified
  parent directory for a missing leaf), compares device/inode identity, and
  creates through exclusive no-follow `openat`. Non-Unix exact native file
  commits fail closed because no portable stable file identity is available.
- The configured action seed participates in candidate selection and produces
  a normal selected-batch ledger. A deterministic Deutsch period-one result
  likewise has one selected effect batch; effects on a longer recurrent cycle
  are withheld without an explicit chronology-selection policy.
- Source size, parser recursion, module count/depth/edges/retained bytes,
  frozen file/network/process observations, staged effects, bytecode/object/
  package sizes and tables, link inputs, memory width, and execution/replay gas
  all have checked bounds. Lengths are rejected before allocation, and decoded
  bytecode is structurally validated plus independently CFG-verified before
  dispatch or package-witness replay.
- Admission also caps a graph at 1,250,000 tokens and 1,000,000 expanded
  statements, a source MARKOV declaration at 256 states, and the LSP at 256
  documents, 64 MiB aggregate source, and 2,000,000 aggregate analysis units.
  The default VM budgets output and dynamic epoch-local storage separately at
  64 MiB each. Dense exact-result paths stop at 1,048,576 cells; retained
  temporal orbits and the action epoch cache each stop at 256 MiB. Action mode
  accepts at most 1,024 seed orbits. Explicit recurrence stops at 262,144
  states, 100,000,000 aggregate instructions, 1,000,000 output items, and
  64 MiB of output.
- Legacy embedding APIs including `Memory::with_size`,
  `EpochCache::with_capacity`, `Buffer::with_capacity`, direct `IOContext`
  buffer/socket sizing, and `FFIContext::alloc_buffer` accept sizes from a
  trusted Rust caller. They are not untrusted-input boundaries. Hostile source,
  artifacts, packages, CLI input, LSP input, and every production bytecode
  execution policy reach checked limits before allocation or retained-cache
  growth. The legacy direct `IOContext::exec` API is separately capped at 64
  MiB combined stdout/stderr and, on Unix, uses parent-owned nonblocking pipes
  so even a session-escaping descendant cannot retain a reader or exceed the
  cap; other platforms fail closed for this compatibility API.

## Independent pre-publication review

On 2026-07-23, the complete 111-file worktree was reviewed again before its
first publication to `main`. The review treated successful compilation as a
starting point rather than an acceptance result: separate passes covered
repository/remote state, the full diff and public API, unsafe/FFI and host
effects, bounded artifact decoding and allocation, compiler/verifier/runtime
authority, packaging, documentation, and deployment.

The review found and corrected:

- 11 strict-Clippy failures, including ambiguous arithmetic precedence and
  dynamic-buffer accounting control flow;
- nine fatal rustdoc warnings in the public API;
- an ignored `Cargo.lock` required by locked builds and by the Dockerfile;
- CI and container builds that did not enforce the lockfile consistently;
- a stale container source label and an over-narrow trusted-embedding caveat
  in this audit.

The post-fix evidence was:

```text
Rust 1.85.0 and stable 1.97.1
  cargo check --all-targets --all-features --locked
  cargo clippy --all-targets --all-features --locked -- -D warnings
  cargo fmt --all -- --check
  all passed

cargo test --all-targets --all-features --locked
  702 library tests passed
  230 integration/CLI/benchmark tests passed
  11 stress tests intentionally ignored in the ordinary profile

cargo test --test main --release --all-features --locked -- --ignored
  11 release stress tests passed

adversarial property campaigns
  10,000 cases for each malformed-artifact/UTF-8 parser property passed
  1,000 cases for each of the nine general property tests passed

RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
cargo test --doc --all-features --locked
cargo audit
cargo package --allow-dirty --locked
  all passed; 0 RustSec advisories; packaged archive verified

docker build --pull -t ourochronos:review .
docker run ... circular_dataflow.ouro --global --memory-cells 2
  container build passed and replayed [0]=3, [1]=2 in 25 instructions
```

## Remaining research frontiers

- termination proofs or exact loop summaries instead of bounded unrolling;
- compositional temporal modalities and proof contracts beyond bounded
  Boolean point-state predicates;
- checkable UNSAT proof objects and portfolio/incremental solvers;
- sparse/symbolic complete recurrence beyond explicit enumeration;
- crash-atomic multi-host transactions beyond preflighted at-most-once intent
  application;
- mechanized metatheory and independent implementation review;
- higher-dimensional exact all-fixed quantum verification.
