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
- The public `SmtEncoder` now passes through canonical source admission and the
  bytecode lowerer. The source-AST compiler remains only as private `cfg(test)`
  differential machinery.

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
- Constraint digests identify the exact declarations and assertions passed to
  Z3. Positive evidence is independently VM-replay-checkable; JSON and SMT are
  interchange formats.
- Complete-UNSAT certificates retain a bounded copy of that exact query plus
  Z3's proof AST. A public verifier checks the digest, replays in a fresh Z3
  context, and requires the same proof term. Certificate construction runs that
  verifier before returning; corruption tests alter both query and proof
  evidence. This removes reliance on the originating solver context, but still
  trusts Z3. A machine-checked Coq/Lean metatheory and an independent proof-kernel
  checker are not implemented.

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

## Canonical-admission alignment extension

On 2026-08-27, a fresh adversarial audit found that the CLI enforced the full
source contract while several public source-shaped library facades rebuilt HIR
and bytecode with weaker subsets of those gates. Successful bytecode validation
did not prove that an unused procedure's declared effects or finite-region
contract had been checked. This contradicted mode-invariant admission even
though ordinary CLI tests remained green.

The correction introduced `AdmittedProgram`, a sealed exact-width compiler
typestate, and routed single-epoch, fast, time-loop, bounded-halting, global,
all-fixed, property, SMT, recurrence, and object-build source APIs through it.
The public SMT facade no longer exposes the raw source lowerer; that independent
oracle now exists only under `cfg(test)`. Object emission admits the complete
graph, links its emitted object set, and seals that linked program before any
objects are returned. A build script checks that production facades retain the
canonical calls and do not regain direct HIR/bytecode compilation.

New adversarial coverage proves that one invalid unused effect contract is
rejected at the same named admission phase by every source execution/proof
facade, that an invalid unused region is mandatory, that obsolete dynamic FFI
cannot cross the runtime-capability phase, and that object emission cannot
bypass whole-graph admission. The normative judgment and result-space
discipline are in `alignment_contract.md`.

The same audit found that the claimed exact-query digest was computed before
session directives were removed, so it did not identify the byte sequence
passed to `Solver::from_string`. That defect is corrected. A complete-UNSAT
result now retains the exact hashed query and Z3 proof AST, enforces a 64 MiB
combined ceiling, and reproduces both UNSAT and the proof term in a fresh Z3
context before returning. The public replay API rejects wrong formats,
backends, completeness, missing/NUL-bearing queries, missing or oversized
evidence, digest substitution, SAT/UNKNOWN replay, and proof substitution. The
adversarial test covers exact-byte hashing plus query and proof corruption.

The next completion audit found that `FAMILY` remained declaration-only: the
example declared `READOUT_INVARIANT` even though its identity transition emitted
the state itself, so its two recurrent classes decided differently. The new
`PspaceFamilyVerifier` and `--verify-family x` mode reject that false claim and
machine-check every decidable obligation for one finite specialization. They
enumerate the exact admitted bytecode domain, reject partial/non-closed,
environment-dependent, effectful, or resource-truncated transitions, measure
peak transition steps and conservative input/decision/control/stack/dynamic/
scope workspace, check all three polynomials, and require one numeric Boolean
across every recurrent class.

The versioned certificate retains the full source contract, evaluated bounds,
decision, and ordered successor/instruction/per-transition-workspace/readout
proof table. Its VM-independent structural checker reconstructs the exact
graph, recurrent classes, workspace maxima, inequalities, and decision; its
public VM recheck reruns complete enumeration. Adversarial mutations cover
contract substitution, invalid program-counter widths, out-of-domain edges,
workspace maxima, and recurrent readout corruption. Duplicate
FAMILY fields are now parse errors instead of last-write-wins claims. The
corrected `pspace_contract.ouro` has two recurrent classes with the same
decision and verifies as a two-state instance. The artifact and documentation
still identify Nature's cost-free ideal selector as an external assumption; no
finite sample is promoted to an asymptotic PSPACE-family proof.

A subsequent adversarial pass found that the finite certificate's
`input_bits` value did not identify an actual language input: `INPUT` was
rejected rather than specialized. The verifier now retains an exact Boolean
word `x`, requires `input_bits=|x|`, freezes that word identically across the
complete temporal domain, and includes it in transition-evidence identity.
Length mismatch, non-Boolean substitution, exhausted input, and input/evidence
rebinding all fail closed. The CLI argument is consequently a bitstring, not a
numeric length.

`PspaceUniformFamilyGenerator` now mechanizes the restricted constant-template
uniform-program portion of the previously external generation obligation. Its exact
linked bytecode template is constant across all nonempty inputs and must have
complete acyclic reachable control, with frozen `INPUT` as the sole accepted
finite-IR boundary; its retained
polynomial temporal-width rule is mechanically proved positive and below
`CTC_CELLS(n)` for every `n>=1`; and its generation-work and descriptor-size
bounds are canonical linear polynomials derived from the template size. The
certificate structurally validates exact bytecode, minimum-width temporal
regions, bounds, and identifiers, and reconstructs itself against an expected
contract/link result. Loop, dynamic-state, width-polynomial, contract, and
bytecode substitutions fail closed. Specialization produces the exact configuration consumed
by the finite verifier. CLI artifacts aggregate exact template bytes and the
finite proof while keeping family-wide totality/readout induction and the ideal
selector visibly outside what was proved. General Boolean-circuit artifact
lowering remains separate from this random-access program-family proof, while
the sparse projection/routing subclass now discharges it exactly.

The sparse projection/routing subclass now also has a genuine all-input
semantic theorem. `ProjectionFamilyCertificate` matches exact linked
instructions for an ordered straight-line list of in-domain temporal-cell
copies, fitting word constants, and width-preserving cell-wise `AND`/`OR`/`XOR`
assignments between cells or with fitting constant operands. Constant-bit
simplification produces identity, zero, one, or explicit `NOT` nodes; constant
right shift becomes exact routing plus zero-fill under the VM's modulo-64
count. Later
writes win and unwritten present cells are zero. Its readout is either a Boolean constant or the retained first input
bit, hence independent of temporal state and unanimous over every recurrent
class. The checker derives exact transition/stack/program-counter costs and
proves the resulting polynomials below the contract for every `n>=1`. The
one-cell self-projection is the minimal case, not the limit. The CLI nests this
theorem with finite evidence and treats any decision/refutation conflict as an
internal alignment error. Noncanonical instructions, out-of-domain routing,
non-fitting constants, corrupted theorem metrics, and insufficient chronology/
step bounds are rejected.

The projection theorem now derives the exact circuit-output polynomial
`temporal_width(n) * cell_bits + 1`. For any concrete nonempty input length,
`ProjectionCircuit` materializes every next-state and decision output as an
explicit Boolean wire: routed prior-state bits, fitting constant bits, zero on
unwritten cells, one-bit `AND`/`OR`/`XOR` nodes, and a constant or first-input
readout, with exact last-write-wins behavior. The checker regenerates the topology without
allocating a second circuit. A linear-size final-assignment index prevents a
state-bit-by-assignment product during generation/replay. Concrete generation
has a 1,048,576-state-bit
resource ceiling; exceeding it is a resource error, not a semantic
refutation. In the CLI, every generated circuit successor and decision is
differentially checked against the independently replayed complete VM table,
and the aggregate artifact retains theorem, exact circuit, and finite evidence.
Rewiring, malformed input/state values, polynomial substitution, and
cross-backend mismatch fail closed.

Non-rendering validation for this extension:

```text
cargo check --all-targets --all-features --locked
  passed, including the build-time architecture gate

cargo test --lib --all-features --locked
  724 library tests passed

cargo test --test main --all-features --locked
  231 integration/CLI/benchmark tests passed
  11 stress tests remained intentionally ignored in the ordinary profile

cargo clippy --all-targets --all-features --locked -- -D warnings
cargo fmt --all -- --check
git diff --check
  all passed
```

Current ordinary-profile active total: **955 passed**. No rendering test was
used as evidence.

## Remaining research frontiers

- generalize the all-input sparse-routing theorem beyond the implemented
  bitwise gates to modular arithmetic, broader compositional acyclic templates,
  and inductive bounded loops;
- termination proofs or exact loop summaries instead of bounded unrolling;
- compositional temporal modalities and proof contracts beyond bounded
  Boolean point-state predicates;
- an independent proof-kernel checker for retained Z3 proof terms, plus
  portfolio/incremental solvers;
- sparse/symbolic complete recurrence beyond explicit enumeration;
- crash-atomic multi-host transactions beyond preflighted at-most-once intent
  application;
- mechanized metatheory and independent implementation review;
- higher-dimensional exact all-fixed quantum verification.

## Long-term programme: review and acceptance boundary (2026-10-08)

The long-term language/platform programme closes against the finite acceptance
records below, rather than the historical completion labels above. Its supplied
guide was read from
`/mnt/c/Users/alarisadmin/Desktop/Remediation.md`. The goal's explicit local-only
publication rule and required durable evidence take precedence over the guide's
generic GitHub-issue/document-consolidation defaults. This existing audit holds
the compact local queue; the specification remains the language authority.

### Reconciled starting candidate

- Repository `/home/astra/.project/Ourochronos`, branch `main`, one worktree,
  HEAD `998d590c9b5dcdc61297a89abff1058c3c5bd6ec`, package 0.2.0.
  No commits have advanced the drafting baseline. The substantial existing
  uncommitted admission, UNSAT-envelope, finite FAMILY, uniform-generation and
  projection-circuit changes were present before this review and are preserved.
- Starting tracked/untracked file identity:
  `bddad02c7ed24b8cae2c532b991adde2e3b6c39234c1a131428a194834823415`.
  Computation: sort unique `git ls-files -co --exclude-standard -z` names;
  SHA-256 the concatenation of each name, NUL, and its raw 32-byte content
  SHA-256. This identifies observed bytes, not authenticated provenance.
- Execution host: x86-64 Linux, WSL2 kernel 6.6.87.2, glibc 2.39;
  rustc/cargo 1.85.0, linked Z3 4.8.12.0. Solver CLI, Lean and Coq were not
  found on PATH; the existing Lean 4.29.1 toolchain was subsequently located
  under `/home/astra/.elan/toolchains/leanprover--lean4---v4.29.1/bin`. Z3 is currently a mandatory dependency, not an optional build.
  The user selected Linux x86-64 plus native Windows x86-64 portable execution,
  with native effect commits supported on Linux only. Native Windows was a required qualification gate; actual native execution
  now supplies its separate evidence, rather than local WSL/Linux execution.
- Fresh starting-candidate `cargo test --all-targets --all-features --locked`:
  724 library and 231 integration tests passed; 11 stress tests were excluded.
  `cargo fmt --all -- --check` and all-target/all-feature locked clippy with
  `-D warnings` passed. These results predate the repairs below.
- Reviewed CI configuration runs Ubuntu ordinary tests, feature builds, MSRV,
  formatting, clippy, coverage and dependency audit. Remote run status has not
  been retrieved; ignored stress, container, recovery and release qualification
  are not established by that configuration. Pinned GitHub page fetches failed;
  local Git and current source were used as the baseline authority.

### Whole-product contract and requirements-to-evidence map

Each row specifies the result's domain, important preconditions/frame, failure
meaning, and the finite evidence required to close the package. Existing tests
are supporting evidence only where their assertions cover the stated contract.

| Goal/package | Contract and acceptance evidence | Current disposition / authority |
|---|---|---|
| 1–4: baseline, semantics, trust, plan | Exact candidate/tool/feature identities; supported operations and failures traced through every facade; findings with reproductions, dependencies and closure checks; user-approved platform choices | Initial bounded whole-product review recorded here. Public source admission and linked-bytecode authority are present; supported platforms are Linux x86-64 and native Windows x86-64 portable execution, with Linux-only native effect commits |
| 5a: ordinary semantics/oracle | Independent small-step stores, operand/call stacks, quotes, scopes, observations and failures; fresh epoch scratch; read-only anamnesis; duplicable ORACLE taint; collision-safe equality; wrapping words, zero division and configured address policy. Independently represented evaluator must compare complete outcomes across source, objects, optimized, packaged and solver-replay paths, including negative/gas cases; meaningful mutations must be detected | Independent numeric-core parser/evaluator and 12 focused conformance checks now cover full successful stack/store/typed output, input consumption, failures, boundary arithmetic, objects/linker, optimized execution, solver replay and packages; six observation mutations are detected. Exact scope is specification §2.4; provenance/heap/host operations remain outside this oracle |
| 5b: metatheory | Declared universal counter-machine model, machine-checked translation into idealized unbounded ordinary core, step simulation theorem, explicit trusted base and finite executable conformance cases | Lean 4.29.1 checked CounterMachine.lean step/finite-trace/initialized simulation and injective readout; no sorry/custom axioms/unsafe/native_decide. CounterMachine executable target 3/3 passed, including 186 normal/prepared/artifact runs. The trusted base and precise universal CM2 premise are recorded in theory §3.1; no Rust compiler/host proof is claimed |
| 6a: point solving | `F(s)=s` witness only after exact configured VM replay; complete UNSAT only for complete admitted lowering and sufficient gas; bounded UNSAT, timeout and nontermination remain UNKNOWN. Corrupt/partial/stale responses cannot become proof/effect eligibility | Typed bytecode IR, sparse decoding, completeness metadata, replay and two-context Z3 query/proof replay present. R03 below is a separate quantum result defect |
| 6b: summaries/contracts | Declare useful syntactic admission for exact loop/procedure summaries or termination arguments; conservative unsupported rejection; compositional assume/guarantee extension with framing, resources and counterexamples. Previously unrolling-limited examples must gain complete results; wrapping, aliasing, gas, effects and nontermination cases required | Exact literal countdown/affine-accumulator summaries and contextual acyclic procedure inlining integrated: independent summaries 7/7, lowering 17/17, solver 24/24 passed. Checked fetched-record bounds preserve gas UNKNOWN. Affine assume/guarantee contracts now admit up to 64 coupled Boolean cells, protected bits, nonvacuity, real/abstract handoff distinctions and checked composition. Independent contract campaign 11/11 passed, including all 4096 two-bit component pairs; source correspondence still trusts the bounded extraction |
| 6c: proof checking/reuse | Independent certificate checker for a declared finite fragment, bound to exact program/query/assumptions/domain; substitutions and corruption rejected. Outside fragment retain solver-trust label. Measure incremental/cache/portfolio workloads; reused and fresh results agree under exact semantic keys | Independent finite scoped no-fixed-point generator/checker/codec and explicit prove-finite/check-finite CLI are integrated: 13/13 focused checks and the exact-program/gas CLI workflow passed. Portable false evidence remains a claim until semantic recheck; independent read-only review found no concrete defect in this declared profile. Existing Z3 evidence retains solver trust. The bounded solver study compared 384 queries over 12 exact finite workloads: 3840 Z3 checks and 384 cvc5 checks agreed with the independent full-domain oracle and original VM replay. Five-repeat medians were 2690.647 ms fresh / 166.200 ms reused; exact-key cache caps and changed-key rejection are tested. No production reuse/cache was introduced |
| 6d: recurrence/readout | Finite symbolic/compositional class with sound all-class readout evidence and useful disagreements; compare with complete small graphs; universal claim only under checked certificate/explicit assumptions | Affine all-class recurrence/readout certificates now check F^(n+1)=F^n, stabilized-image rank/count, parity invariance or two disagreeing fixed states up to 64 Boolean cells. Independent recurrence target 12/12 passed, including 4164 small models and a 64-step transient; source/config binding is explicit. Real analyse-affine CLI accepted uniform/ambiguous cases and refused periodic/query/gas neighbors |
| 7: restricted FAMILY | Static/solver/certificate/assumed obligations remain distinct in diagnostics/artifacts; uniform generator, width/register/work bounds, totality, frozen state and all-class invariant readout. Accepted, invalid adjacent and unsupported families; finite-instance bridge without sample-to-asymptotic promotion | Finite exhaustive checker, uniform constant-template generator, projection/bitwise theorem and circuit checker qualified by independent FAMILY conformance 7/7: all 16 one-bit maps/readouts, periodic/no-point classes, routing/overwrite/frame/input instances, exact resource neighbors, explicit assumptions and artifact outcomes. Claims remain restricted to the declared fragment |
| 8: stochastic | Arbitrary-precision exact rationals, sparse transitions and enforced arithmetic/allocation/state limits; exact normalization, nonnegative probabilities, closed classes, extremals and residuals; bounded frozen VM/random adapter with totality and independent transition replay; all-class ambiguity and measured growth | Arbitrary-precision sparse backend and bounded frozen joint-tape VM adapter exported. Sparse analytic/adversarial target 10/10 and extraction target 10/10 passed; independent numeric oracle retains scenario observations and rejects any failed state/scenario. Declared resource caps do not guarantee allocator success. Configurable absorber, sparse-cycle/large-denominator and frozen-VM class/readout examples are integrated; twelve profiles × five processes publish state/edge/bit/work/time/RSS growth in case studies. Linux supervised execution converts timeout/allocator abort into UNKNOWN without partial result promotion |
| 9a: resources | Enforce input/nesting/stack/call/allocation/enumeration/solver/journal/cache/decoder/diagnostic limits; predictable exhaustion and cancellation; hostile and adjacent valid cases; hardware/tool/warm-cold/sample/variation baselines across dense/sparse/wide/repeated/application profiles | Source/module/parser/VM/result and cache ceilings plus paged memory present. Linux resource supervisor passed four focused checks and ordinary/solver/timeout/capability CLI checks. Fourteen phase profiles, dense/wide/repeated runs and twelve stochastic growth profiles now have measured latency/RSS budgets in case studies. Exact page occupancy removes full-width sparse projection scans; eleven checks and independent COW/invariant review passed. Process supervision is not filesystem/network confinement |
| 9b: checkpoints | Versioned resumable finite work bound to exact program, semantics, frozen input, config and consumed budget; corrupt/stale rejection; split-run semantic argument and independent uninterrupted/cancel/recovery campaign. Exhaustion remains UNKNOWN | Versioned classical checkpoints now resume the actual shared VM dispatcher and reject decoded images until bounded canonical prefix replay establishes reachability; exact code/input/config/cumulative ceiling bind. 13 focused checks cover every fetched cut, cancellation, faults and resealed corrupt state; the CLI slice/resume/budget-reset/corruption workflow passed. Heap/temporal/host operations remain outside this recovery profile |
| 10a: artifacts | Source/object/link/package/witness/runtime policies preserve required checks and deterministic identities; portable source/proof manifest, bounded reload and corruption/version/compatibility rejection; integrity distinguished from authenticity and release signing policy | Object source manifests, deterministic linking, bounded formats, CFG verification and package witness replay present. OUROPA v1 preserves linked source-manifest claims and optional final-code-bound evidence through CLI bytecode/package/launcher outputs; legacy BC v2/PK v3 APIs remain explicit payload paths. Envelope target 9/9 and CLI source-removal/kind/corruption/legacy test passed; launcher 5/5 passed, including independently discovered total-size cap repair. Structured VM diagnostics now retain fetched pc/span/charged gas; ordinary/package errors resolve escaped manifest file claims without opening paths, including a deleted imported-source CLI failure. Release signing policy distinguishes unsigned local qualification from authenticated final archives/launchers; final runtime/archive identities, checksum verification and cross-platform reload records close the declared unsigned local release profile |
| 10b: effects/recovery | Only one selected chronology may apply effects. Exact token/batch equality, preflight, commit, acknowledgment/audit/replay; success, failed prefix and unresolved application remain distinct; longer Deutsch cycles withhold effects. One durable transactional/idempotent adapter profile with controlled crash injection; finite multi-host design prototype with stated assumptions | Linux private-directory managed KV retains exact batch/receipt/policy/value authority in one bounded atomic snapshot. Eleven active checks, six crash boundaries, twelve transaction checks and independent durable-profile review passed. No irreversible native host operation is dispatched by this profile. Bounded two-participant/coordinator research passed five checks and exhaustively reached 1776 states without depth cutoffs; unavailable outcomes remain Unresolved. Device/power-loss guarantees and general multi-host exactly-once remain outside the contract |
| 10c: embedding/trust | Exact scalar ABI/signatures/lifetimes and host-value checks, capability/path/subprocess/log/secret boundaries, dependency provenance; supported portable host-dependency/snapshot schema; fail-closed unsupported platforms; OS isolation and uninterruptible trusted callbacks explicit | OUROHM v1 binds exact bytecode/descriptors, INPUT and finite PURE scalar observations to an independently approved digest; bounded checked snapshots produce observation-only tables. Eight manifest checks passed with/without dynamic FFI, ten FFI unit checks passed, release arity regressions passed, and the host snapshot example returned 6 with zero effects. Loader/initializer trust is explicitly unsafe and captured immutable arity guards the C signature. Independent trust/durable review passed. Packages still reject FFI/effects; native callbacks/processes require separate host isolation |
| 11: quantum | Qualify CPTP/density checks, degeneracy/convergence/fixed-space/readout extrema with conservative uncertainty near thresholds. Bounded higher-dimensional SDP/general-POVM specification, prototype, independent/adversarial references and reproducible report; certify/approximate/unsupported claims separated | Dimension-2–4/general-POVM numerical SDP investigation completed with 80 cases, 90 solves and independent mathematical review. The review exposed oversized-integer configuration conversion and misleading promise-gap prose; both were repaired. Exact conditioning counterexample, hash-pinned environment and numerical-only/unsupported claims are recorded in specification §14.5 and case studies. This completes research, not a certified general-dimensional production backend |
| 12a: useful language/distribution | Actual CLI/REPL/LSP/API/module/debugging/solver/artifact/embedding/install/upgrade/remove journeys, actionable diagnostics; locked reproducible build, packaged-byte verification, license/dependency inventory, platform/migration/release/rollback records and repeatable CI stress/doc/package/container gates | Nineteen final Linux actual CLI/REPL/LSP journeys, thirty native Windows portable/interchange operations and nine native REPL/LSP checks passed. Verified source clean-install, forced upgrade/removal and retained-runtime rollback preserve caller files; versioned 0.2.0→0.3.0→0.2.0 directory rollback also passed. Both final runtimes reproduced byte for byte in independent offline retained builders. Deterministic archives bind current Cargo-normalized source, qualified native bytes, checksums and notices for the actual host/target dependency closure. CI now contains the finite gates; no remote CI result or publication is claimed |
| 12b: applications/teaching | At least three configurable complete workflows extending mutual exclusion, circular dataflow and retrocausal game; negative cases, saved results, commands and conventional independent cost/result comparison; stochastic all-class portfolio; executable tutorial/normative/API/contributor guidance | Configurable exclusion (1–8 participants), affine ring (1–8 nodes/1–4 bits) and deterministic game (1–16 actions) compare complete answers with conventional formulations, replay witnesses and save bounded evidence. Five tests, thirteen actual workflows and 48 independent ring enumerations passed. Default Linux supervision withholds partial results on interruption; explicit raw mode supports portable embedding. Tutorials and stochastic portfolio are connected to executable examples |
| 13: integrated release | Freeze exact candidate after every required implementation/research package; finite risk-based supported/adversarial/recovery/install/interchange/application/stress portfolio; independent review of consequential semantics/proof/unsafe/effects/serialization; original theorem assumptions; all required findings closed with typed evidence index and compatibility/recovery guidance | The integrated 0.3.0-rc.1 finite qualification portfolio closes the declared implementation and bounded research packages. The final source archive, runtime archives, exact source snapshot/tree, retained build identities and typed evidence index are in target/qualification/0.3.0-rc.1. Required high-consequence changes received independent review. This is an unsigned local candidate; published signing and remote CI are separate authorized workflows |

Different domains retain different guarantees: ordinary epoch execution;
one-seed point orbit; one reached deterministic Deutsch cycle; action selection
among discovered witnesses; symbolic point solving; all-point properties;
exhaustive finite recurrence; all-class rational stationary families; numerical
fixed-density exploration; idealized asymptotic theory. No term such as
“fixed point” upgrades another mode's domain. DEFAULT stays metadata. Static
malformation/capability rejection, runtime failure, interruption, gas UNKNOWN,
proof, counterexample, vacuity, ambiguity and numerical uncertainty must remain
distinguishable through APIs, CLI, tooling and serialized artifacts.

The source trace covers disk/virtual/LSP/REPL text and module configuration →
located lexer/parser/module graph → typed HIR/source admission → per-source
objects/relocations/linker → structural and CFG verification/seal → bytecode
dispatch. Temporal branches use typed IR/Z3 → decoding → VM replay, or finite
transition tables → recurrent/certificate analysis. Declarative MARKOV and
QCHANNEL bypass ordinary-body execution by design. Artifact branches use
bounded decoding/policy checks → seal/replay; selected-effect branches use
frozen observations → candidate intents → explicit selection → commit ledger
→ adapter preflight/application → acknowledgment. Compiler lowering, shared
core value/provenance operations, Z3, floating point, libc/dynamic libraries,
the trusted embedding callbacks and the OS remain trust dependencies. Digests
are identity/integrity aids, not authenticity or independent proof.

The fresh cross-layer CLI test exercises all three existing studies through
objects, link, point/properties, JSON, package witness replay, recurrence and
native launcher boundaries; this supports those exact fixtures only. Review
probes also inspected intermediate receipt/status/range observations, avoiding
inference from an apparently correct final output. High-consequence boundaries
still require the independent review and adversarial portfolios listed above.

### Local findings and dependency order

| ID / kind / severity | Evidence and consequence | Repair, dependencies and finite closure |
|---|---|---|
| R01 reproduced defect / high | `transaction.rs::commit_selected_with_adapter`: synthetic failing adapter first returns EffectAdapterFailed, replay returns successful AlreadyCommitted; recorded-only batch followed by adapter never dispatches. Existing transaction test expects the erroneous replay success; NativeEffectAdapter itself retains failure correctly | Separate recorded/application states in the shared log. Failed result must replay identically without callback; recorded-only batch may first dispatch once; panic/unacknowledged dispatch remains unresolved and never blindly retries. Check same/cross-transaction success/failure/conflict/deferred/unwind and native prefix failures. Prerequisite to durable recovery |
| R02 reproduced evidence defect / medium | `property_tests.rs` skips parse Err for generated valid source; `tests/benchmark/mod.rs` ignores terminal status/success. Standalone exact-fixture probe: arithmetic/stack/bitwise fixtures reject LoopBodyDrift; “Fibonacci” outputs 100000, factorial 0, nested loops [101,0]. Green tests therefore do not qualify named workloads | Require every generated parse/outcome; repair benchmark workloads and derive full outputs conventionally; assert VM/optimized/temporal outcomes. Focused campaign plus integrated tests. Independent of R01/R03 |
| R03 reproduced result defect / high | `quantum.rs`: reset-to-zero with basis 1, positive accept threshold 1/2000000000 and default tolerance prints range [0,0] but ACCEPT/exit 0. Finite 1e308+i1e308 Kraus entries overflow to NaN; NaN residual comparison admits invalid channel | Never widen thresholds to pass estimates; reject nonfinite intermediates/mutated matrices/residuals. Exact reset/identity, adjacent thresholds and overflow regression checks; CLI result check and independent review. Full numerical qualification remains package 11 |
| R04 source-established distribution defect / medium | Dockerfile copies manifests/dummy source then real `src`, but never the existing untracked `build.rs`; final image omits the required architecture gate | Include build.rs in the final source stage, preserving dependency cache. Check container build when runner available and package inclusion; no new installation/publication implied |
| R06 reproduced metadata defect / medium | Infinite `WHILE { 1 } { NOP }`, unroll 1, produces UNKNOWN/exit 3 in global/all-fixed/property CLI artifacts but all three omit typed completeness; property API drops it too. Diagnostic prose is not a structured bound | Preserve `IrCompleteness` through property results, serialize it on all UNKNOWN artifacts and expose it in CLI diagnostics; assert exact loop count/bound with and without an exemplar, plus unavailable metadata. Additive JSON v1 field; no opcode/binary-format change |
| R05 missing capability / high | Independent evaluator/metatheory, summary/contracts/checker, broader recurrence, arbitrary precision/extraction, checkpoints, portable provenance, durable recovery, bounded quantum research, complete applications and integrated qualification were absent or incomplete | Implementation/research profiles and final integrated local qualification are closed against the bounded contracts above |
| R07 reproduced interface defects / medium | LSP initialization nested capabilities incorrectly, shutdown retained the connection sender and hung joining I/O threads, and full-sync changes selected the first snapshot | Return the full InitializeResult, drop the connection before joining, and apply the final full snapshot. Twenty-five LSP unit checks and eighteen actual CLI/REPL/LSP journeys passed |
| R08 source-established unsafe defects / high | Wrong host arity reached callbacks; direct native closure calls or mutated cloned signatures could select a different C signature; safe library loading hid native initialization/destruction | Reject wrong linked-host arity before dispatch, capture validated arity in the native closure, and make library loading explicitly unsafe in both feature configurations. Default/dynamic manifest, FFI and release arity checks passed; no unsafe native mismatch was invoked |
| R09 reproduced candidate-identity defects / medium | Initial archive tooling admitted stale/altered source packages, checked source identity only after collection, omitted a supplied-license change check, and selected cross-build helper notices with the target platform. GNU Windows linking also inserted wall-clock PE timestamps | Compare every source member with Cargo's authoritative no-build normalization, compare before/after source and all explicit input bytes, and use Cargo's actual normal/build tree for notice inclusion. Disable PE timestamp insertion. Independent mutation/valid/deterministic archive checks and three cross-host closure fixtures passed; final production archives include the Cargo-required notices and both runtimes reproduce identically |
| R10 supplied dependency advisory / high upstream severity, not actionable in reviewed product paths | Refreshed RustSec audit reports RUSTSEC-2026-0295: z3 0.12.1 ApplyResult shallow Clone can double-decrement native ownership. The private Solver/Model/AST consumers do not construct Tactic/ApplyResult/Goal, and no public Rust API exports these owners. Windows retains affected dependency code; absence of call paths is not absence of vulnerable code | Independent source/artifact review established the bounded unused-API disposition. The dependency remains affected, and the unignored failed audit is retained. CI requires an exact manifest/lock/consumer regression guard before this single explicit advisory exception; 35 independent mutation cases were rejected, current/normalized valid inputs accepted. New dependencies/consumers/source changes require renewed review. This is not a dependency fix, formal reachability proof, or vulnerability-free claim; other application Z3 usage is outside this disposition |

Dependency graph: baseline review → R01/R02/R03/R04 qualification repairs →
5a shared semantics/oracle → {5b metatheory, 6b summaries/contracts, 6c checked
proof, 6d symbolic recurrence}; relevant 6 foundations → 7 FAMILY qualification.
Frozen shared semantics permits independent 8 stochastic and 9 resource/resume
work. R01 + 10 effect/schema contracts → controlled durable recovery;
10 provenance/serialization → package interchange and distribution.
11 quantum numerical repair and bounded investigation has its own gate.
12 application workflows consume these capabilities. 13 waits for **every**
required implementation and research gate and the chosen supported matrix.
Incremental/portfolio choices follow measured workloads, not an assumed speedup.

Repair status, evidence reuse boundaries and next executable step are updated
in this section as packages close; the final frozen identities live outside
the source tree to avoid self-referential fingerprints.

### Integrated repair evidence and historical checkpoints

R01 now records separate NotAttempted/Applying/Applied/Failed adapter states in
one shared commit log. Deferred ledger-only batches can first dispatch later;
failed replay returns the retained error without repeating the prefix, and
unacknowledged/panicked dispatch remains unresolved. Transaction/native-adapter
focused checks passed 12/13 respectively. This qualifies process-lifetime
in-memory at-most-once behavior; durable restart recovery remains package 10b.

R02 repaired generated-case admission and benchmark outcome checks, including
conventionally derived arithmetic/stack/bitwise/Fibonacci/factorial/nested-loop
expectations (property 9/9, benchmarks 14/14). The 11 formerly excluded release
stress gates also passed after strengthening million-write memory and 50,000-item
stack assertions; these checks do not establish universal performance budgets.

R03 rejects nonfinite numerical intermediates and brackets threshold arithmetic
outwardly instead of widening acceptance. Quantum 14/14 and source CLI 3/3
passed; independent rational/adjacent-float probes covered 4,186/4,104 cases.
Qubit rank/fixed-space estimates remain tolerance-dependent, without certified
conditioning error bounds; the separate bounded package 11 investigation is complete.

R04's Docker build.rs omission is repaired; local container build/version and
wrapping-output execution passed. That image predates the current extensions.
Building the normalized source package additionally exposed its explicit
`build = "build.rs"` during dependency caching. The cache now supplies a dummy
build script, cleans the own-package artifacts and overwrites it with the real
architecture gate before final compilation. Both base image identities are pinned.
R06's property/global/all-fixed UNKNOWN boundaries now retain typed completeness
in APIs, CLI and JSON (11 focused checks passed); additive API/JSON migration is
explicit. No opcode or legacy payload-format version was changed.

Native Windows qualification is active; the portability pilot has passed. A controlled cross-build with Rust 1.85,
MinGW x86-64 and official Z3 4.8.12 Windows binaries exposed a cfg-only undefined
file handle in the unsupported creation branch; the branch now compiles and
continues to reject unsupported commits. Pilot source fingerprint
`ade382da98e40b6ed9e06bfccf9288eea0198ff6c2e2ef1d11d953282486b082`
built a PE x86-64 executable; native PowerShell execution passed version, wrapping/zero-division/shifts, procedures, quotations, countdowns, paradox detection, complete/bounded global solver cases, and real module loading. Linux-built orbit and embedded-point packages execute on Windows; Windows-created packages execute on both platforms; a Windows-created standalone launcher executes natively.
This pilot excludes the still-changing finite proof module and is not the frozen
release candidate. Z3 release archive SHA-256:
`de12fb2160798a464244954236b28da597e79289f33955b170853b8bf0d1f078`.

Sparse arbitrary-precision MARKOV is now exported for Rust embedding: 10/10
analytic/adversarial focused checks passed. Canonical sparse rows, closed SCCs,
positive extremals, exact normalization/residual identities and all-class
threshold witnesses operate under integer/storage/charged-work caps. The source
i128 backend remains unchanged. Frozen joint-tape VM extraction, actual shared-dispatch resumable checkpoints and affine symbolic recurrence are integrated under declared restricted profiles; measured growth and final qualification remain gates.

The following pilot/progress entries are historical checkpoints; the final
qualification closure below supersedes their then-open gates. Historical
ordinary-suite totals are starting evidence, not final-candidate qualification.

Windows pilot executable SHA-256: `ef0fc119c0998c8ba95c33fbbb7985cae0ee36ca005c02dbcc9ab113e4a05a14`. Its later native/package checks reuse those exact bytes.

Current shared-dispatch integration check: `cargo test --lib --test main
--all-features --locked` passed 737 library and 239 active main tests (11 ignored
stress gates). Subsequent checkpoint reachability and CLI changes passed their
13-case target and one real slice/resume workflow; no broad final-candidate claim
is made. The checkpoint admission retains structurally validated unused prelude
procedures, follows direct calls, and conservatively includes all quotations once
a dynamic combinator is reachable.

Windows pilot additionally ran from a directory containing only the executable
and libz3.dll, using the already installed system VC++ runtime; Microsoft runtime
DLLs are not intended as bundled release assets. The current native adapter now
rejects every nonempty host-effect batch on non-Linux platforms before host calls;
this new platform guard requires the next Windows build/negative check.


The current Windows progress build includes finite proof/checkpoint/affine,
structured source diagnostics and Linux-only native-effect denial. Its source
build-input fingerprint is
`08cf4cf2d5a901c85b0f8caaf02eee005e7763c5c52141f6b780736884b6766a`;
PE SHA-256 is `aa2bb900079e1562a425394cbefbdc6822a33212ba5a5b4fab7d792183214c53`.
Native Windows passed 15 bounded cases: version, finite proof production and
Linux-proof checking/stale-gas rejection, checkpoint slice/reset/resume, affine
recurrence, native-effect and Linux-supervision denial, deleted-source package
diagnostics, Windows package build/run and standalone launcher build/run.
Windows-created proof/checkpoint/package bytes passed Linux reload/execution.
This build predates the LSP/FFI/page-skipping repairs and is progress evidence,
not the frozen release. Qualification fixture expectations were corrected from
"Built portable" to the CLI's actual "Wrote portable package"; no product
behavior was weakened.

The durable KV campaign initially observed one policy-mismatch fixture failure
without enough retained diagnostic detail to establish its cause. Isolated and
full reruns passed after assertion diagnostics were strengthened. Fixtures now
serialize their process sessions to avoid overlapping fork/lock lifetimes,
while still explicitly checking a competing owner returns Busy; the asserted
PolicyMismatch contract is unchanged. The final 11 active checks passed in
2.92 seconds. Serialization is a containment decision, not proof of the
original failure's cause. Subsequent independent durable review and six-boundary crash qualification passed.

Phase measurements now cover parse, module/HIR admission, object compilation,
linking, bytecode verification, IR/SMT construction, dense/sparse/wide/repeated
VM runs, provenance, orbit journals, exact arithmetic and package roundtrip.
The wide sparse profile exposed full-width iteration in sparse projections.
A private exact page-occupancy count now permits empty-page skipping without
hash authority or loss of zero-valued provenance; 11 paged-memory tests passed,
including a dense-vector oracle across boundaries, overwrites and COW clones.
Independent read-only review found no concrete invariant/COW/count defect.
Changed-path timing reduced the million-cell sparse profile from 1496–1521 µs to 17–22 µs, while dense execution remained near 567–579 µs. These are finite reference-host measurements; final integrated qualification remains required.

Actual LSP stdio checks reproduced malformed nested initialization capabilities,
shutdown retaining its sender, and multi-change full snapshots applying only
the first change. The repairs passed the bounded eighteen-journey actual-process
gate and twenty-five focused LSP checks. Frozen-runtime qualification remains required.

The integrated 0.3.0-rc.1 Rust sources passed 1124 active all-target/all-feature
checks (739 library, 239 main, 146 dedicated integration checks); all eleven
excluded main stress checks passed separately in release mode. Strict all-target
Clippy and formatting passed. Lean 4.29.1 rechecked instruction, finite-trace,
initialized simulation and injective readout with only its reported standard
`propext`/`Quot.sound` dependencies. The updated actual interface gate passed
nineteen RC process journeys with explicit platform/features discovery. These
source-level checks are reusable while only release documentation and packaging
change; packaged native executables still require their own qualification.

Final native checks exposed a stale literal REPL version; the banner now reads
the compiled Cargo version, with an exact CLI/REPL identity assertion in the
existing process gate. The Windows gate's report-path guard also shadowed its
script-scope runtime identity with a peer identity; distinct variable names
restore the final hash comparison. Its thirty product operations passed before
that guard failure; the failure is retained separately. The corrected final
gate subsequently passed all thirty cases, and nine separate native REPL/LSP
checks passed against the same final PE and bundled DLL.

### Final integrated local qualification (2026-10-09)

The qualified package is 0.3.0-rc.1 with `lsp,dynamic-ffi`, on Linux x86-64
GNU and native Windows x86-64 GNU portable execution. Native host-effect
commits and process resource supervision remain Linux-only. Linux requires
glibc 2.36/GNU C++ GLIBCXX_3.4.30; native Windows checks used Windows 11 and
the installed VC++ x64 runtime. Both distributions carry exact qualified Z3
4.8.12 bytes and notices. Loader checks remove inherited overrides and select
the extracted Linux library; Windows contains only the executable and solver
DLL as runtime assets. Source build verification additionally exercises the
default feature configuration. No optional-solver build is advertised.

Final runtime SHA-256 values are
`f760b4b49e402315775b3b130e07396d91360df2344f2f45b5765c87315e5a3d`
for Linux and
`cb93b19c6b0d393e01cfbd0c56f3996f7683a5bb1252be2f12b0c3b1758c5faa`
for Windows. Separate offline builders cleaned only their own release package
and reproduced these bytes. The final source package differs from their
compiled input only in documentation, CI and Python packaging/guard scripts;
all Rust, Cargo manifests/lock, build script and executable fixtures are
compared byte for byte. Final archives include that source, dependency/license
inventory, notices, unsigned provenance and hashes for every payload member.
The archive verifier detects payload mutation and executes extracted Linux
bytes, including the Debian 12 runtime baseline. Native Windows archive checks
execute the matching extracted PE/DLL. Portable Windows proof/checkpoint/
package/affine/continuation outputs passed five Linux reload checks; three
legacy Windows progress artifacts passed current Linux reload checks.

Strict Clippy passes in default and all-feature configurations; strict Rustdoc,
formatting and source package verification pass. The last Rust change binds
the REPL banner to Cargo's version; eleven REPL checks, the final actual
interface gates and byte-identical runtime reproduction qualify that change.
The 1124 active checks and eleven separate release stress checks above cover
the integrated Rust state before that isolated banner repair and are reused
for unchanged paths. The six Lean statements retain their declared trusted
base. Finite exhaustive/certificate results, replayed solver witnesses,
numerical estimates, bounded research and benchmark measurements remain
distinct evidence categories in the final machine-readable index.

Source installation exposed stale same-version own-package artifacts when a
shared target directory was reused across normalized archives. The retained
failure and clean-own-package recovery are in the installation record; the
README now gives the fresh-target/clean procedure. Synthetic upgrades,
removal and rollback preserved caller programs and results. R10's affected
binding remains a visible reviewed dependency limitation, with the unignored
audit and exact-consumer guard retained. Native callbacks, OS confinement,
general multi-host atomicity, unrestricted halting and certified general
quantum fixed spaces remain outside the declared contracts.

The source history remains on the stated base HEAD with supplied worktree
changes preserved. A separate temporary Git index exports the exact candidate
tree and binary patch without changing the user's index, branch or commits.
`qualification.json` in the existing ignored qualification directory binds
that tree, source snapshot, final source/archive hashes, build identities and
raw evidence. Remote CI status is unverified; CI configuration now gives these
gates a repeatable home. No release, push, issue or artifact was published.
Maintenance consists of affected-path checks followed by the finite release
commands in README; changed solver consumers require renewed R10 review.
