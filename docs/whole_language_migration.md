# Whole-language migration programme

This is the implementation record for replacing Ourochronos's prototype
whole-language path with one compiler, executable representation, verifier
lowering, linker, and runtime. It is deliberately separate from
`completion_audit.md`, which reconciles the delivered bounded claims and owns
the final validation gate. The mechanisms below are implemented unless a row
states a remaining limit; the completed post-change gate and exact counts are
recorded there.

## Mandate and baseline

**Engagement class:** structural remediation programme. The existing finite
temporal semantics and verification results are retained, while the parser-time
linker, direct AST interpreters, optional checker, and disconnected FFI/module
facades are replaced by one explicit toolchain.

**Locked semantic decisions:**

- An epoch is a deterministic transformation `F : M -> M` over the declared
  finite temporal state, parameterized only by explicitly frozen or modeled
  inputs. ORACLE reads the anamnesis and PROPHECY writes the present.
- Present temporal memory starts at zero on every epoch. Operand stacks, call
  frames, epoch-local ordinary heap, observations, and staged effects also start
  empty. Anamnesis is read-only during the epoch.
- ORACLE values are temporally tainted but duplicable. Linearity is not part of
  the language: the runtimes, verifier, and flagship programs all rely on
  duplication.
- Compiler constants distinguish words, quotation references, symbol
  references, and resource handles. A validated runtime encoding may use
  machine words, but the compiler never guesses a value's kind from that
  encoding.
- Mutable external state is not part of `F` by accident. A deterministic
  temporal evaluation must reject, freeze, or explicitly model every external
  influence. Irreversible effects are staged, selected only after a timeline
  resolves, and applied under an at-most-once token/digest record.
- Multiple fixed states remain observable. Orbit, require-unique, all-fixed,
  canonical-selection, universal-property, and recurrent queries are distinct
  policies; no backend may silently substitute one for another.
- Complete finite UNSAT requires complete executable lowering under the actual
  machine bounds. Bounded loops, gas truncation, solver timeout, or unsupported
  effects produce UNKNOWN or UNSUPPORTED, never a global proof.

The baseline was re-derived on 2026-07-13 from
`c5fed129098f3c2019b329fb8decadada3a896a0` on `main`. The worktree already
contained owner changes and is preserved as-is. The explicit toolchain path is:
`/home/astra/.cargo/bin/cargo`.

```text
cargo test --all-targets
  390 library tests passed
  200 integration/CLI/benchmark tests passed
  11 stress tests ignored
  0 failures

cargo clippy --all-targets -- -D warnings
  passed
cargo fmt --all -- --check
  passed
git diff --check
  passed
```

## Empirical evidence ledger

The older audit and documentation are inherited claims. The rows below were
confirmed against the current code or executable.

| ID | Location or command | Observation | Classification |
|---|---|---|---|
| INV-001 | `src/bytecode.rs`, `src/main.rs` | Typed, validated, deterministic bytecode is the mandatory compiler artifact and executable authority for ordinary, action, diagnostic, Deutsch, recurrent, solver-replay, package, REPL, and fast paths. MARKOV/QCHANNEL are separate finite declarations, not AST execution fallbacks. | Resolved mechanism |
| INV-002 | `src/main.rs`, mandatory CLI regressions | HIR semantics, per-source object linking, bytecode/CFG verification, type/effect analysis, and region analysis run without `--typecheck`; no CLI execution mode inlines into an AST interpreter. | Resolved silent optionality and fallback |
| INV-003 | `src/semantics.rs`, `src/bytecode_verifier.rs` | Underflow and unequal joins are errors; dynamic rows and recursive summaries become named obligations rather than fabricated effects. | Resolved silent failure |
| INV-004 | `src/types/mod.rs` flagship regression | All three flagship programs now pass with duplicable temporal ORACLE values. | Resolved conformance defect |
| INV-005 | `src/source.rs`, `src/lexer.rs`, `src/parser`, `src/module_graph.rs`, `src/hir.rs`, `src/bytecode.rs` | Exact UTF-8 spans flow through a recursively checked located AST sidecar, module linking, HIR errors/statements, and bytecode source maps. Legacy in-memory APIs remain deliberately unlocated; artifact source IDs need a source manifest for cross-process path lookup. | Located pipeline resolved; portable source manifest remains |
| INV-006 | `src/module_graph.rs` | Canonical importer-relative DFS now deduplicates, reports cycles, preserves declarations/initializers, and relocates typed quotation IDs. | Resolved missing mechanism |
| INV-007 | `src/hir.rs` | Procedures, quotations, and foreign targets resolve to typed IDs before semantic analysis and bytecode lowering; strings remain only at the source boundary. | Resolved for executable symbols |
| INV-008 | `src/vm/fast_vm.rs`, `src/bytecode_vm.rs`, differential corpus | The former recursive FastVM opcode switch is removed. Optimized pure execution seals immutable `PreparedBytecode` and uses the same VM dispatcher, preserving exact gas/errors while skipping repeated validation scans. | Resolved duplicate runtime authority |
| INV-009 | `src/bytecode_vm.rs`, `src/bytecode_timeloop.rs`, `src/bytecode_action.rs`, `src/main.rs` | Standard, diagnostic, deterministic Deutsch, action-guided, recurrent, bounded-halting, REPL, package, and optimized policies use linked bytecode with explicit bounded frames. The fast facade returns typed bytecode failures and never falls back to Executor. | Resolved production runtime authority |
| INV-010 | `src/temporal/global_solver.rs` gas regressions | Complete proofs require a control-flow-aware executable path bound within configured gas; unavailable complete bounds return UNKNOWN. | Resolved proof defect |
| INV-011 | `src/ast.rs`, `src/bytecode_vm.rs` effect regressions | Buffers and collections are fresh bounded deterministic epoch state; files/sockets/processes are frozen virtual observations plus staged external intents; CLOCK/RANDOM are explicit frozen nondeterministic tapes. | Resolved model defect |
| INV-012 | capability/effect integration corpus | Epoch-local resources run deterministically. External observations fail closed without an exact frozen capability, and irreversible effects remain data until one selected batch is ledgered and applied. | Resolved proof/runtime defect |
| INV-013 | `src/temporal/region.rs`, `src/main.rs` | Region contracts are enforced on main code, procedures, and quotations before mode-specific execution or verifier lowering. | Resolved dead path |
| INV-014 | `src/bytecode.rs`, `src/linker.rs`, `src/runtime/ffi.rs`, `src/bytecode_vm.rs` | Linked artifacts retain exact library, symbol, scalar type, result, and effect descriptors. The supported process-host ABI accepts only bounded `u64`/`i64` arguments and at most one scalar result; every other signature is rejected during lowering. Calls require an exact explicit host binding and consume ordinary VM gas. Fixed-point analyses still decline the external effect. | Narrow scalar ABI resolved; portable host-dependency schema remains |
| INV-015 | `src/core/paged_memory.rs`, `src/bytecode_timeloop.rs`, `src/temporal/cache.rs` | Standard bytecode orbit state uses persistent paged COW snapshots with collision-checked exact identity. Dense compatibility/proof representations retain exact width, provenance, and collision checks but are not a separate production execution authority. The million-cell sparse-witness measurement is recorded in `case_studies.md`. | Resolved orbit scalability mechanism; symbolic and compatibility representations remain bounded |
| INV-016 | `src/linker.rs`, `src/package.rs`, `src/launcher.rs`, `Dockerfile`, CLI build regression | Typed per-source linking, deterministic `OUROOBJ`/`OUROBC`, explicit orbit/embedded/runtime-global `OUROPK` resolution, and a directly runnable launcher exist. Embedded evidence is recomputably bound and replayed; runtime-global packages declare the exact Z3 contract. The container declares build/runtime dependencies. Cryptographic signing remains distribution policy. | Build/runtime mechanism resolved; signing intentionally external |
| INV-017 | `src/bytecode_temporal.rs`, `src/temporal/global_solver.rs`, `src/temporal/transition_graph.rs` | Global, all-fixed, property, SMT, and exhaustive recurrent modes execute/lower linked bytecode and replay through the bytecode VM; bounded loops remain honestly incomplete and complete proof requires a bytecode gas bound. | Resolved for point-state verification and recurrent enumeration |
| INV-018 | `src/temporal/transaction.rs`, `src/temporal/effect_adapter.rs`, `src/bytecode_timeloop.rs` | Exact input/clock/random/file/endpoint/process transcripts participate in candidate equality and receipts. Selected intents are whole-batch capability/precondition checked; file identity/content is verified; the idempotency claim precedes mutation; exact replay returns the stored success/failure without reapplication. File/socket/process/sleep opcodes use frozen virtual inputs and selected-only intents. | Resolved transaction and system-opcode boundary; no cross-host atomicity claim |
| INV-019 | `src/vm/executor.rs`, `src/vm/mod.rs`, `src/lib.rs` | The public source-shaped single-epoch `Executor` is now a compatibility facade that performs type checking, HIR resolution, mandatory semantic analysis, bytecode compilation, independent CFG verification, and bytecode-VM dispatch. Unsupported legacy mutable host contexts fail closed. The retired direct AST walker is private `cfg(test)` differential machinery. | Resolved public execution authority |

## Approach-family registry

Consequential alternatives remain recorded so later evidence can reopen a
choice without repeating the investigation.

| Family | Live alternatives | Decision and invariant |
|---|---|---|
| Located syntax | spans on legacy AST; side table; new located source AST | Use a located source AST. Every node owns a `SourceId` and half-open UTF-8 byte range; compatibility AST structures are parser/test inputs and do not authorize production execution. |
| Module loading | textual include; parser-time merge; canonical module graph | Use a canonical graph outside the parser. Imports resolve relative to their importer, each canonical path loads once, cycles report the complete trace, and dependency order is deterministic. |
| Name representation | strings/raw words; runtime quotation typing; static typed IDs | Resolve to stable newtypes for modules, procedures, constants, temporal fields, quotations, properties, and foreign functions before checking. |
| Executable IR | register MIR; stack bytecode; direct native AOT | Use validated stack bytecode with explicit blocks, branches, frames, typed constants, source maps, effects, and resource budgets. It matches the source language and gives the verifier and portable artifacts one authority. An SSA form may be an internal optimizer view only. |
| Runtime implementation | recursive AST walk; threaded bytecode loop; direct AOT | The reference runtime becomes an iterative bytecode machine with explicit operand and frame stacks. Optimized execution is a transformation or dispatch strategy over the same validated program. |
| Solver integration | lower source AST; lower HIR separately; lower executable CFG | Lower the validated executable CFG. Solver summaries may reject or bound instructions, but may not reinterpret source independently. |
| Temporal memory | dense clone; sparse ordered map; persistent paged COW | Use 256-cell persistent copy-on-write pages with exact width-aware equality, collision verification, deterministic witnesses, and cost proportional to changed pages. Release peak-memory evidence is recorded; page size remains an implementation parameter that future benchmarks may retune. |
| Effect handling | unrestricted replay; reject-only; frozen inputs plus staged commit | Reject or freeze/model influences during candidate evaluation; stage external observations/effects and apply only a selected timeline under token-plus-digest at-most-once evidence. Preserve and replay terminal failure rather than claiming rollback of an applied prefix. |
| FFI | broad dynamic/libffi surface; generated platform shims; narrow scalar ABI | Start with a genuinely implemented scalar ABI and reject every unsupported signature during check/link. No placeholder registrations are permitted. |
| Module initialization | silently discard; run on each import; deterministic once-only initializer | Preserve imported declarations and run each dependency initializer once in deterministic dependency order. Temporal initializers participate in the linked epoch program; external initialization effects obey transaction rules. |
| Packaging | renamed source; portable linked bytecode plus launcher; native AOT | Produce deterministic object files and linked bytecode first, then a launcher containing the linked artifact and compatible runtime. Native AOT remains optional. |

## Single authorities in the target design

1. `SourceManager` owns source identity, retained UTF-8 text, canonical paths,
   overlays, and line/UTF-16 conversion.
2. The lexer and parser produce located syntax and diagnostics without file I/O
   or symbol resolution.
3. The module graph owns loading, import traces, ordering, and duplicate/cycle
   detection.
4. Resolution produces typed HIR; the semantic checker is mandatory for every
   execution, verification, build, REPL, and tooling entry point.
5. The linked, validated bytecode program is the sole executable semantics for
   production paths. The public source-shaped `Executor` is an admission and
   compilation facade over bytecode; the direct AST walker is private
   test-only parity machinery.
6. Reference execution, optimized execution, temporal lowering, replay, and
   packaged execution consume that program rather than the source AST.
7. The opcode/statement definition generates or mechanically checks parsing,
   effects, stack typing, validation, reference dispatch, optimized dispatch,
   verifier support/rejection, serialization, documentation, and tests.
8. Temporal transactions own frozen inputs, staged effects, resolution policy,
   rollback, selected-batch receipts, and truthful at-most-once host
   application with replayed success or failure.

## Milestones and gates

Each milestone is independently testable. Runtime behavior is characterized
before a working path is rerouted.

1. **Conformance and soundness gate (complete for the characterized AST
   runtime).** Add adversarial parity
   tests; repair FastVM underflow; make ORACLE duplicable; reject unmodeled
   resource state during search; enforce region contracts on every path; stop
   low-gas UNSAT overclaims.
2. **Located source and module graph (implemented).** Retained UTF-8 sources,
   the fallible located lexer, recursively shape-checked AST sidecars, exact
   parser errors, canonical module graph, linked HIR locations, and serialized
   bytecode source maps are implemented. Path-relative canonical imports,
   cycle/dedup/order, imported quotation relocation, and duplicate-declaration
   spans are covered. Cross-process artifacts still need a source-path manifest.
3. **Resolved HIR and mandatory semantics (implemented).** Typed
   procedure, quotation, and foreign IDs; forward and recursive resolution;
   exact fixed stack rows; explicit dynamic/recursive obligations; and
   mandatory CLI enforcement are implemented. Exact source locations flow
   through the module graph, HIR, and bytecode source maps; the linked narrow
   FFI path carries exact scalar ABI/effect metadata. All flagship examples
   pass the mandatory analysis.
4. **Executable bytecode and validator (implemented).** Typed
   constants and call targets, flat structured control, code ranges, source
   maps, deterministic bounded serialization, malformed-input tests, and an
   independent CFG verifier exist. Objects and linked artifacts retain exact
   callable/foreign signatures and declared/observed effect metadata.
5. **Iterative reference runtime (implemented for 97/99 primitive spellings).** The explicit-frame bytecode VM
   implements 97/99 primitives with exact gas and resource limits, safe
   procedure/quotation recursion, temporal scopes, output, strings,
   collections, bounded typed in-memory buffers, independently frozen
   CLOCK/RANDOM tapes, virtual file/socket stores, and frozen process results.
   Host writes, sends, spawns, and sleeps are staged for selected-only commit.
   Only the two obsolete dynamic FFI spellings remain unsupported.
6. **Optimized execution (implemented as prepared dispatch).** The duplicate
   recursive FastVM opcode switch is gone. `PreparedBytecode` seals an immutable
   structurally validated program and reuses the same iterative dispatcher
   without rescanning the artifact per epoch, preserving exact gas and errors.
7. **Verifier migration (implemented for the declared finite point-state and recurrent queries).**
   Direct bytecode CFG analysis and native symbolic lowering now
   generate the typed temporal IR from validated linked bytecode for all 42
   deterministic finite primitives, structured `IF`, acyclic calls, temporal
   scopes, terminal flow, and honestly bounded `WHILE`. Typed source-located
   rejection covers quotes, foreign/dynamic/runtime/external operations,
   recursion, and cross-call nested scopes. Global, all-fixed, property, and
   SMT CLI modes consume this API and replay through bytecode. Recurrent graphs,
   diagnostic orbits, and deterministic Deutsch cycles also execute linked
   bytecode. Specialized stochastic/quantum declarations remain separate exact
   declarative analyzers; action selection executes the linked bytecode and
   retains the selected transaction batch. The raw source-AST `SmtEncoder`
   survives only as a backwards-compatible parity oracle, not a proof path.
8. **Persistent temporal memory (implemented for the production orbit).** Paged COW state has 65,536- and
   million-cell, collision, exact-provenance, equality, diff, and limit tests,
   and backs the standard bytecode orbit. Dense compatibility/proof structures
   use exact collision-safe identity; they remain bounded specialized
   representations rather than runtime fallbacks. The million-cell
   sparse-witness peak-memory gate is recorded in `case_studies.md`.
9. **Temporal transactions (implemented).** Exact frozen observation
   transcripts, typed staged effects, candidate selection, rollback, bounded
   receipts, preflighted capabilities/file preconditions, and truthful
    at-most-once success/failure replay are implemented. Standard, diagnostic,
    and action bytecode policies retain only the selected batch. File, network,
    process, and sleep opcodes lower directly to these intents. The configured
    action seed is an explored candidate rather than a ledgerless fallback.
    A Deutsch point fixed state has one selected period-one batch; effects on a
    longer recurrent cycle remain withheld without a chronology-selection
    policy. Unix native file commit retains verified target/parent handles,
    compares device/inode identity, and uses exclusive no-follow `openat` for
    a missing leaf; non-Unix exact native file commit fails closed.
10. **Object format and linker (implemented).** The linker implements typed
    symbols/signatures, imports/exports, constant/procedure/quotation/foreign
    relocations, source relocation, deterministic module ordering, and final
    validation. Deterministic bounded `OUROOBJ` v4 retains target/runtime ABI,
    canonical constants, named temporal/default schemas, bounded Boolean
    property predicates and touched-address slices, declared and observed
    effects, source manifests/maps, exact foreign descriptors, and bounded
    optional verification payloads. Linking preserves or rebuilds metadata and
    invalidates proof bytes when relocation changes code. The CLI emit/link
    flow is implemented. The canonical graph also retains each source-local
    located AST and its visibility set. `compile_objects` emits independently
    serializable per-source objects with public MANIFEST/named-temporal/procedure/foreign
    symbols, private local quotations, exact constant/call relocations,
    dependency anchors, and dependency-first once-only module initializers.
    `--emit-object` emits the real entry-source object with typed imports;
    `--emit-objects` emits the complete source/prelude set into a required-empty
    directory so stale modules cannot contaminate later links.
11. **Real narrow FFI (implemented for explicit process hosts).** Versioned
    bytecode and objects retain exact foreign descriptors. A safe host table
    binds bounded `u64`/`i64` callbacks by typed ID and exact signature; the VM
    checks the whole table before dispatch and enforces stack, return-shape,
    and gas bounds. The optional dynamic loader accepts only up to six unsigned
    `u64` arguments/results through an explicitly unsafe registration API
    because shared-library signatures are not machine-checkable; that adapter
    also rejects signed types. The linked ABI rejects pointer, handle, string,
    32-bit, multi-result, unknown-symbol, and mismatched bindings. The
    portable package manifest has no host-dependency section and therefore
    rejects every artifact carrying a foreign descriptor.
12. **Build and packaging (implemented for the declared `OUROPK` v3 policies).** `--check`, `--emit-object`,
    `--emit-objects`, `--emit-bytecode`, `link`, `--build`, and `run-package` produce, validate,
    link, and execute deterministic portable bytecode containers.
    `--build-executable` embeds the package in a directly runnable copy of the
    current native runtime. `--embed-global-witness` stores a canonical solver
    model plus digest/count only after bytecode replay and revalidates it on
    package decode, allowing synthesized point states to run without a runtime
    solver. `--runtime-global-package` instead declares the exact versioned Z3
    dependency and never substitutes orbit resolution. Embedded evidence
    digests bind the manifest, linked program, canonical state, and replay
    count. Cryptographic signing policy and broader provenance records remain.
13. **Application acceptance (implemented and measured).** Compile, link,
    solve/recur, replay, package, and execute mutual exclusion, circular
    dataflow, and retrocausal game families. Release median/p95 and peak-memory
    records are in `case_studies.md`.
14. **Hardening (implemented and gated).** Source files,
    parser nesting, module count/import count/depth/edges/retained bytes,
    frozen file/network/process inputs, staged effects, bytecode/object/package
    bytes and tables, link inputs, memory cells, and execution/replay gas are
    explicitly bounded. Decoders check counts before allocation, and structural
    validation plus independent CFG verification precede dispatch or witness
    replay. Malformed/fuzz-style corpora, adversarial transaction tests,
    release benchmarks, and peak-memory records exist.
    Sanitizer/Miri coverage remains environment-dependent. The final full
    tests, release ignored tests, Clippy, formatting, diff, RustSec, MSRV,
    release build, benchmark, and container checks are recorded by the
    completion audit.
15. **Completion audit (complete).** `completion_audit.md` maps the bounded
    claims to implementation, clearly labels historical counts, and records
    the superseding final gate with its exact test counts.

The standard stage gate is the baseline test suite, Clippy with warnings denied,
format check, diff check, focused adversarial tests, and a full diff review.
Test-count changes must be explained; ignored tests remain findings.

## Risk register

1. **Semantic drift during rerouting.** Control: retain test-only AST parity
   corpora while bytecode tapes are compared. Hard stop: unexplained output,
   error, memory, or gas difference.
2. **Recursive behavior reaches a compatibility inliner.** Control: resolve
   recursive SCCs before enabling them and route them only to the explicit frame machine.
   Hard stop: any host recursion on user call depth.
3. **Verifier proves a different machine.** Control: lower validated bytecode and
   include all execution bounds in completeness. Hard stop: SAT replay failure
   or UNSAT under a bound absent from the formula.
4. **Effects occur during candidate search.** Control: default-decline until the
   transaction mechanism is proven. Hard stop: an irreversible operation is
   observed more than once or before resolution.
5. **Sparse memory weakens identity.** Control: exact equality after hashing and
   deterministic page serialization. Hard stop: collision changes an answer or
   provenance changes fixed-state equality.
6. **Dirty-tree overlap loses owner work.** Control: serialize core edits,
   inspect every patch and preserve unrelated changes. Hard stop: an edit cannot
   be separated from pre-existing work.
7. **Compatibility adapters become a second authority.** Control: adapters only
   translate into the current milestone's canonical representation and carry a
   removal gate. Hard stop: a mode executes semantics not reachable through the
   linked bytecode.
