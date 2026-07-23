# Ourochronos Specification

This document defines the Ourochronos abstract machine, its instruction set, the fixed-point execution model, and the behaviour of the reference implementation. It is normative for language and runtime behavior. `theory.md` is normative for Turing-completeness, complexity, computability, and physics claims.

Sections 2 to 8 define the machine and its semantics. Sections 9 to 12 describe the analysis and verification layers built on it. Section 13 covers practical programming patterns, and the appendices catalogue the full instruction set and error numbering.

---

## 1. The computational model

Ourochronos implements several related consistency forms. Standard mode follows one orbit in search of `F(S)=S`; `--global` instead solves the finite typed temporal IR symbolically and replays every SAT witness in the linked-bytecode VM. `--all-fixed` proves zero, one, or multiple point fixed states. `--recurrent` classifies every cycle and basin in an explicitly bounded closed domain. Deutsch mode lifts one reached deterministic cycle to its exact uniform stationary ensemble. `MARKOV` and qubit `QCHANNEL` declarations cover finite stochastic and quantum fixed states. ORACLE/PROPHECY programs read a candidate state through the *anamnesis* and construct the next *present* state.

Aaronson and Watrous showed `P_CTC = BQP_CTC = PSPACE` for polynomially uniform, polynomial-width total circuits supplied with a stationary distribution/fixed density operator, with correctness required for every fixed point. Point consistency alone is not their classical model, and the concrete orbit-searching executable is not an ideal PSPACE oracle. `theory.md` §5 gives the complete conditions.

---

## 2. Abstract machine

### 2.1 State components

Each *epoch* (one evaluation of F) operates on the following state:

| Component | Type | Description |
|-----------|------|-------------|
| Stack (σ) | List(V) | LIFO operand stack |
| Anamnesis (A) | Addr_m → V | Read-only memory: the candidate fixed point |
| Present (P) | Addr_m → V | Read-write memory being constructed |
| Output (O) | List(V ∪ Char) | Output buffer for this epoch |
| Status (ξ) | {Running, Finished, Paradox, Error} | Epoch status |
| Scratch | heaps, handles, I/O contexts | Epoch-local working state (§3) |

where V is the set of 64-bit unsigned values and `Addr_m = {0, ..., m-1}` for the configured positive finite width `m`. Addresses are carried as 64-bit values. The reference default is `m = 65,536`; `--memory-cells` selects another finite width.

### 2.2 Values

A value is a pair (n, π): the numeric value n and its *provenance* π ⊆ Addr, the set of oracle addresses that influenced it. Provenance is analysis metadata; the fixed-point test compares numeric values only. Section 8 gives the provenance algebra.

### 2.3 Initial epoch state

At the start of each epoch the stack and output are empty, the present is all zeros, the status is Running, and the anamnesis is the previous epoch's present (or the seed for the first epoch).

---

## 3. The state boundary

The S in F(S) = S is the configured-width memory array and nothing else. Its width is fixed throughout one run. This is a definition, not an implementation accident, and three consequences follow.

First, data structures (vectors, hash tables, sets), buffers, file handles, and FFI contexts are *epoch-local scratch*. They are created fresh each epoch and are not part of the convergence test. A program that wants a value to survive the loop must write it to memory with PROPHECY; a value held only in a vector cannot participate in the fixed point.

Second, the epoch function must be a fixed function of the memory state, so operations that break that property are declined inside a fixed-point search by default (§7): external effects would fire once per search epoch rather than once per consistent timeline, and non-deterministic sources would make F vary between the iterations comparing its output.

Third, external observations are frozen. The first epoch that consumes INPUT, CLOCK, or RANDOM values fixes independent typed tapes, and every later epoch replays those same values; an epoch that tries to read past a frozen stream is an error, not a fresh prompt. File contents, endpoint receive streams, and process output/exit status likewise come from exact bounded snapshots. File snapshots are not read authority: exact read capabilities separately gate contents and metadata, so a write-only virtual base image cannot be exfiltrated. An endpoint tape permits only virtual connect/receive; staging a send requires a separate exact endpoint capability. Live CLOCK/RANDOM capture additionally requires `--effects unrestricted`; library and proof modes accept only caller-supplied tapes. Information flows from the external world into the loop, never backwards; the search cannot re-prompt, reread, or resample itself into a different timeline.

A program with no ORACLE in code reachable from main never reads the anamnesis, so F is constant and its unique fixed point is the result of a single epoch. Reachability follows direct procedure calls and conservatively includes every quotation after a reachable dynamic quotation combinator; unused prelude/procedure code does not disable the shortcut. Such *trivially consistent* programs run once, and the state-boundary rules for searches do not apply to them.

---

## 4. Execution model

### 4.1 The search

```
S₀ = seed (all zeros, or diverse seeds in action-guided mode)
Sₙ₊₁ = F(Sₙ)
stop when Sₙ₊₁ = Sₙ            (consistent)
      or Sₙ revisits a state   (oscillation, period = distance to the earlier visit)
      or a finite prefix shows monotone growth (heuristic divergence diagnosis)
      or the epoch budget runs out (timeout)
```

The convergence test is exact equality of all `m` cells' numeric values, including equality of widths. The default epoch budget is 1,000 (`DEFAULT_MAX_EPOCHS`), shared by the CLI, the REPL, and the library, and overridable through the library.

### 4.2 Verified cycle detection

Visited states are journalled under a 64-bit state hash. The hash is a fold, so distinct states can collide; a revisit is therefore only declared after the journalled state (stored in sparse non-zero-cell form, which determines the full state) compares equal to the current one. The epoch result cache applies the same rule: a cache hit requires the stored anamnesis to match, not merely its hash. A hash collision costs a re-execution, never a wrong verdict.

### 4.3 Execution modes

| Mode | Flag | Behaviour |
|------|------|-----------|
| Standard | (default) | Search with cycle detection and epoch caching; first fixed point wins |
| Diagnostic | `--diagnostic` | Standard plus full trajectory recording, divergence detection, and paradox diagnosis |
| Deutsch | `--deutsch` | Accept a reached deterministic cycle as its uniform stationary ensemble; report whether its output is unanimous |
| Action-guided | `--action` | Explore several seeds and select the fixed point with least action (§9) |
| Pure | (library) | Uncached iteration without cycle detection; still stops at the implementation safety ceiling |

All temporal search modes share input freezing and the effect gate. Standard, Diagnostic, Deutsch, and Action-guided modes use verified cycle detection where applicable; Pure mode deliberately omits it.

### 4.4 Finite temporal IR and proof obligations

For a supported deterministic program `p`, memory width `m`, bounds policy
`b`, and loop-unroll limit `k`, lowering produces a typed acyclic term graph

```text
L(p,m,b,k) = (A, P, valid, observations, completeness)
```

where `A` and `P` have type `Array Word64 Word64`, conditions have type
`Bool`, and values have type `Word64`. The point query is exactly
`valid AND P=A`. The lowering rules use 64-bit modular arithmetic, logical
`NOT` (`0 -> 1`, nonzero `-> 0`), zero-valued division/remainder by zero,
modulo-64 shifts, and the configured address policy. A typed temporal scope
maps local address `a` to `base + bound_b(a,size)` and masks every stored value
to its declared `BITS` width.

The canonical lowering API consumes a structurally validated, linked
`BytecodeProgram` directly. It symbolically executes flat typed procedure IDs,
structured `IF` targets, temporal enter/exit pairs, terminal flow, and bounded
`WHILE` back edges without reconstructing or consulting the source AST.
Acyclic calls inherit the caller's active temporal scope. Recursive SCCs,
quotation values, foreign calls, dynamic stack/code operations, runtime heap
state, and external effects are source-located typed errors. The native
lowerer covers the same 42 deterministic primitives as the finite source IR;
the source compiler remains a differential-test oracle, not a verifier input.
Deterministic step, expression-arena, observation, CFG, and call-graph limits
fail closed.

The public raw source-AST `SmtEncoder` is retained solely as a
backwards-compatible parity oracle for differential testing. `--smt`, global
solving, all-fixed enumeration, and property proof use the linked-bytecode
lowerer above; output from the compatibility encoder is not a production proof
authority.

An IR with no reachable bytecode `WHILE` is `Complete`. An IR containing loops is
`BoundedLoops(k)`: paths whose next condition remains true after `k`
iterations are excluded. Consequently:

- SAT is accepted only after decoding `A`, running the linked-bytecode VM from that
  exact memory, finishing normally, and checking the replayed `P=A`;
- UNSAT is a global nonexistence/property proof only for `Complete` IR;
- bounded-loop UNSAT and solver timeout are `UNKNOWN`;
- an unsupported, effectful, recursive, or ill-typed construct is an explicit
  error, never silently omitted.

This is the executable refinement relation between the solver and VM. The
constraint digest in every artifact identifies the exact SMT query used.

---

## 5. Instruction set

Programs are sequences of literals, opcodes, and structured statements. Stack effects are written `( before -- after )` with the top of stack rightmost. Appendix A lists all 99 opcodes; the core groups are:

**Temporal.** The reason the language exists.

| Opcode | Effect | Description |
|--------|--------|-------------|
| `ORACLE` | `( addr -- A[addr] )` | Read from the anamnesis; the result's provenance gains addr |
| `PROPHECY` | `( value addr -- )` | Write to the present |
| `PRESENT` | `( addr -- P[addr] )` | Read from the present |
| `PARADOX` | `( -- )` | Abort the epoch as explicitly inconsistent |

**Stack, arithmetic, logic, comparison.** Conventional stack-machine operators: `POP DUP SWAP OVER ROT DEPTH PICK ROLL REVERSE`, wrapping 64-bit `ADD SUB MUL DIV MOD NEG ABS MIN MAX SIGN`, bitwise `NOT AND OR XOR SHL SHR`, unsigned comparisons `EQ NEQ LT GT LTE GTE` and signed `SLT SGT SLTE SGTE`. Division and modulo by zero yield 0, keeping every operator total.

**Quotations and combinators.** `[ ... ]` pushes a code block's identifier; `EXEC DIP KEEP BI REC` run it. Quotations are scanned for ORACLE like any other code, so a temporal program cannot hide its oracle inside a quote.

**Memory and structures.** `PACK UNPACK INDEX STORE` address memory directly; `VEC_* HASH_* SET_*` provide epoch-local heap structures; `BUFFER_*` byte buffers; `STR_*` and `CONCAT REV SPLIT` string operations over length-suffixed character sequences.

**I/O, system, FFI.** `INPUT OUTPUT EMIT`, `FILE_* TCP_CONNECT SOCKET_* PROC_EXEC CLOCK SLEEP RANDOM`, and `FFI_CALL FFI_CALL_NAMED` with `FOREIGN` declarations. Their behaviour inside a search is governed by the effect gate (§7).

### 5.1 Structured statements

```
cond IF { then } ELSE { else }      pop cond; nonzero runs then
WHILE { cond } { body }             run cond, pop; nonzero runs body, repeat
TEMPORAL base size [BITS n] { body } isolated temporal region, base-relative;
                                    writes truncate to n bits (default 64)
TEMPORAL name @ addr DEFAULT v;     named temporal variable; references read addr
                                    from the anamnesis. DEFAULT is retained in
                                    HIR/object schema metadata only: present and
                                    initial anamnesis still start at zero.
MANIFEST name = value;              compile-time constant
LET name = expr;                    named binding (compiled to PICK)
PROCEDURE name { body }             procedure, resolved to a typed callable ID
PROPERTY name {
  ALL_FIXED predicate;              universal point-fixed-state property;
}                                   predicate supports CELL addr-or-name OP value
                                    with NOT, AND, OR, and parentheses
FOREIGN { ... }                     FFI declarations
IMPORT "file"                       source inclusion
FAMILY name {                       asymptotic PSPACE-family contract
  CTC_CELLS POLY c d a;             bound c*n^d+a
  CHRONOLOGY_BITS POLY c d a;
  TRANSITION_STEPS POLY c d a;
  UNIFORM; TOTAL; READOUT_INVARIANT;
  IDEAL_DEUTSCH; EFFECTS_FROZEN;
}
MARKOV name {                       finite exact stochastic Deutsch model
  STATES n;
  ROW i { p/q ... p/q };            n probabilities per row
  ACCEPTING { state, ... };
  ACCEPT_AT_LEAST p/q;              default 2/3
  REJECT_AT_MOST p/q;               default 1/3
}
QCHANNEL name {                     qubit CPTP channel in Kraus form
  QUBIT;
  KRAUS { C re im C re im C re im C re im };
  ACCEPT_BASIS 0|1;
  ACCEPT_AT_LEAST p/q;              default 2/3
  REJECT_AT_MOST p/q;               default 1/3
  VALIDATION_TOLERANCE p/q;         default 1/10^9
  ANALYSIS_TOLERANCE p/q;           default 1/10^9
}
```

`IMPORT` is file-backed source inclusion, not process-CWD text substitution.
The path is resolved relative to the importing file and canonicalized. Each
canonical file is loaded once, import cycles are errors with a complete trace,
and dependencies link in deterministic dependency-first order. Imported
top-level statements are once-only module initializers; procedures,
quotations, FFI declarations, model declarations, and properties are retained.
Duplicate source symbols and conflicting singleton model declarations are link
errors. Parsing an in-memory string cannot import a relative file; callers must
use the file-backed module graph.

The command-line compiler uses the fallible source-aware lexer. Unknown
characters, malformed literals and escapes, invalid radix digits, and integer
overflow are compile errors with exact half-open UTF-8 byte spans; they are
never skipped to continue compiling a different token stream.

### 5.2 Grammar

```ebnf
program     = { declaration } { statement }
declaration = manifest | temporal_decl | procedure | property | foreign | import | family | markov | qchannel
statement   = literal | opcode | quotation
            | if_stmt | while_stmt | temporal_stmt | let_stmt | block
literal     = integer | "'" CHAR "'" | '"' { CHAR } '"'
quotation   = "[" { statement } "]"
if_stmt     = "IF" block [ "ELSE" block ]
while_stmt  = "WHILE" block block
temporal_stmt = "TEMPORAL" integer positive_integer
                [ "BITS" integer_1_to_64 ] block
let_stmt    = "LET" IDENT "=" { statement } ";"
manifest    = "MANIFEST" IDENT "=" integer ";"
temporal_decl = "TEMPORAL" IDENT "@" integer "DEFAULT" integer ";"
procedure   = "PROCEDURE" IDENT "{" { statement } "}"
property    = "PROPERTY" IDENT "{" "ALL_FIXED" property_or ";" "}"
property_or = property_and { "OR" property_and }
property_and = property_unary { "AND" property_unary }
property_unary = [ "NOT" ] ( property_atom | "(" property_or ")" )
property_atom = [ "CELL" ] (integer | IDENT)
                ("EQ"|"NEQ"|"LT"|"LTE"|"GT"|"GTE") integer
family      = "FAMILY" IDENT "{"
              "CTC_CELLS" polynomial ";"
              "CHRONOLOGY_BITS" polynomial ";"
              "TRANSITION_STEPS" polynomial ";"
              { family_obligation ";" } "}"
polynomial  = "POLY" integer integer integer
family_obligation = "UNIFORM" | "TOTAL" | "READOUT_INVARIANT"
                  | "IDEAL_DEUTSCH" | "EFFECTS_FROZEN"
markov      = "MARKOV" IDENT "{" "STATES" integer ";"
              { "ROW" integer "{" rational { rational } "}" ";" }
              "ACCEPTING" "{" [ integer { "," integer } ] "}" ";"
              [ "ACCEPT_AT_LEAST" rational ";" ]
              [ "REJECT_AT_MOST" rational ";" ] "}"
rational    = integer "/" positive_integer
signed_rational = [ "+" | "-" ] integer "/" positive_integer
complex     = "C" signed_rational signed_rational
qchannel    = "QCHANNEL" IDENT "{" "QUBIT" ";"
              { "KRAUS" "{" complex complex complex complex "}" ";" }
              "ACCEPT_BASIS" ( "0" | "1" ) ";"
              [ "ACCEPT_AT_LEAST" rational ";" ]
              [ "REJECT_AT_MOST" rational ";" ]
              [ "VALIDATION_TOLERANCE" rational ";" ]
              [ "ANALYSIS_TOLERANCE" rational ";" ] "}"
block       = "{" { statement } "}"
```

Property predicates are limited to 256 AST nodes and 32 nested Boolean or
parenthesized levels in source and object decoding. Operators use the usual
precedence `NOT` > `AND` > `OR`; comparisons are unsigned 64-bit words.

Opcodes are the keywords of Appendix A. A registration test guarantees every
opcode is reachable from source.

### 5.3 Compiler and executable artifacts

Every file-backed CLI entry passes through the same mandatory front end:
canonical module loading, typed HIR resolution, structural stack/effect
analysis, bytecode lowering, deterministic linking, bytecode validation and
CFG verification, followed by the source type/effect and temporal-region
checks. `--typecheck` prints those reports; it does not enable checks that are
otherwise skipped. Linked `FOREIGN` declarations lower to an exact bounded
scalar ABI and execute only when a library caller supplies a matching explicit
process-local host table. The CLI, temporal proof modes, and portable packages
do not invent or persist such host bindings.

The executable bytecode is a flat, versioned representation with typed word,
quotation, procedure, and foreign operands; explicit structured-control
targets; temporal-scope pairs; callable code ranges; and source-map entries.
Its little-endian `OUROBC` encoding is deterministic and allocation-bounded.
Decoding rejects unknown versions and flags, truncation, trailing bytes,
invalid identities, crossing control structures, out-of-range targets, and
oversized tables before execution. The independent CFG verifier proves fixed
stack-height and temporal-scope joins and records explicit obligations for
value-dependent primitives, foreign calls, recursive procedure components,
and terminal scope unwinding rather than inventing stack effects.

The linker supports multiple typed object modules, symbols, stack/effect
signatures, imports, exports, and relocations, and orders inputs by stable
module name. The canonical module graph retains each source-local located AST
and compiles one object per source. Because source syntax has no `EXPORT`
declaration, visibility is explicit and uniform: `MANIFEST` words, named
temporal schemas, procedures, and foreign declarations are public to
importers, anonymous quotations are module-private, properties are retained
as linked metadata, and each
top-level body is a once-only module initializer. Direct imports also produce
private typed dependency-anchor symbols; ordinal object names preserve the
graph's dependency-first initializer order even when object inputs are
permuted. Imported calls carry exact procedure/effect or foreign-ABI
signatures and typed relocation sites, while each private quotation identity
is relocated with its defining procedure. The version-4, bounded,
little-endian `OUROOBJ` format embeds
relocation-aware local bytecode plus an explicit portable target/runtime ABI,
a canonical literal pool, exact finite-region, named temporal (including
declarative DEFAULT), and bounded Boolean all-fixed-property metadata,
declared/observed whole-module effects, source-file names/lengths/digests, and
a bounded optional versioned verification payload. Constant, region, effect,
and source-map metadata is cross-checked against bytecode; decoding rejects
unknown targets/tags/flags, allocation-limit abuse, malformed names,
truncation, and trailing bytes. Linking preserves relocated source manifests,
properties, effects, and exact foreign descriptors, rebuilds code-derived
metadata, and invalidates pre-link verification payloads after rewriting code.
The public `compile_objects` API emits these independently serializable
per-source objects, and the `link` command emits validated `OUROBC` from one
or more objects. `--emit-object` writes only the real entry-source object and
therefore retains typed imports; `--emit-objects` writes the complete
source/prelude set into an empty directory. Requiring an empty directory keeps
removed modules from surviving a rebuild as stale link inputs.

Standard point-orbit execution uses the iterative bytecode VM. Procedure,
quotation, and loop
control use explicit language frames, so user recursion cannot consume the
host call stack; gas, frame, operand-stack, temporal-depth, output, and
collection limits are checked. This backend implements 97 of the 99 registered
primitive spellings. `FFI_CALL` and `FFI_CALL_NAMED` are deliberately rejected
dynamic legacy spellings; real linked `FOREIGN` declarations lower to the typed
`CallForeign` instruction with an exact scalar ABI and host table. Files,
buffers, clock/random tapes, virtual sockets, frozen command results, and
selected sleep intents are native bytecode semantics rather than AST fallbacks.

Global, all-fixed, property, and SMT modes lower the validated linked bytecode
directly into the typed temporal IR. The native lowerer covers the 42 finite
primitives, structured conditionals, acyclic typed calls with inherited
temporal scopes, terminal flow, and iteratively constructed bounded-loop
formulas. It independently proves symbolic stack joins and rejects every
reachable quotation, foreign, dynamic-stack, mutable-runtime, external-effect,
recursive-call, or nested-scope boundary with a bytecode/source site. SAT
witnesses replay through the same bytecode machine. Complete UNSAT additionally
requires a conservative reachable bytecode instruction bound within configured
gas; loops, recursion, or arithmetic overflow cannot become a proof.

Explicit recurrent-graph analysis enumerates every declared-domain transition
through the linked bytecode VM. Source-facing library entry points first
compile to bytecode and then call that same analyzer.

---

## 6. Convergence semantics

Let F: M → M be the program's memory transformation. The runtime classifies a run as:

| Status | Condition | Exit code | Interpretation |
|--------|-----------|-----------|----------------|
| Consistent | P = A after an epoch | 0 | A point fixed state exists on this orbit |
| Deutsch-consistent | a verified state recurs in Deutsch mode | 0 | Uniform stationary ensemble over the recurrent cycle; readout may be unanimous or ambiguous |
| Paradox | explicit PARADOX executed | 2 | The program rejected the timeline |
| Oscillation | a verified earlier state recurs (period k) | 2 | No fixed point on this orbit; the grandfather paradox is period 2 |
| Divergence | a finite trajectory prefix is monotone (diagnostic mode) | 2 | Heuristic diagnosis, not proof of unbounded growth under wrapping arithmetic |
| Timeout | epoch budget exhausted | 3 | Unknown; the search gave up |
| Error | runtime error inside an epoch | 1 | The epoch itself failed |

Diagnostic mode additionally reports the oscillating cells, the cycle states, and recognises the negation pattern (a cell mapped to its own logical NOT) as a grandfather paradox. Deutsch mode gives that same verified cycle a different mathematical interpretation: a period-k deterministic cycle represents the exact stationary distribution with mass 1/k on each state. The result exposes output from every cycle state and marks a public output unanimous only if their numeric/character sequences agree. Divergence detection runs only in diagnostic mode, so a monotonically growing program exits 3 (timeout) under the default mode and 2 (divergence) under `--diagnostic`; the codes differ because the evidence differs, not the program.

A Deutsch point fixed state is a period-one ensemble and therefore identifies
one selected observation/effect batch, which an authorized adapter can ledger
and commit once. A longer recurrent cycle does not select one chronology;
staged effects on such a cycle are rejected and remain uncommitted.

---

## 7. Effects and determinism

Every opcode carries an effect class, defined in one place (`OpCode::effect_class`):

| Class | Opcodes | Inside a search |
|-------|---------|-----------------|
| Pure | word/stack/control/temporal memory, epoch-local collections, frozen `INPUT`, and buffered `OUTPUT`/`EMIT` | Permitted |
| External | file, socket, process, FFI, and sleep operations | Exact frozen observations and staged selected effects where modeled; dynamic FFI is declined |
| Non-deterministic input | `RANDOM CLOCK` | Frozen tape required; first-candidate live capture only under explicit unrestricted policy |

The bytecode VM models buffers, files, and sockets as fresh bounded virtual stores for every epoch. File reads require an exact snapshot and read capability; socket receives consume only a bounded frozen endpoint tape; `PROC_EXEC` returns an exact frozen command result. File writes, network sends, process spawns, and sleeps become typed intents and cannot reach the host during candidate execution. Unsupported dynamic FFI remains an epoch error naming the opcode.

`OUTPUT` and `EMIT` need no gate because they are buffered per epoch; only the
resolved epoch's buffer is printed. The transaction layer records the exact
input/clock/random/file/endpoint/process observation transcript, exact
candidates, typed staged intents, explicit selection, rollback, and idempotent
commit evidence. A capability-scoped adapter may run only after exact
selection. It checks every capability and file precondition before mutation,
records the commit token before its first irreversible operation, returns the
same stored failure on replay, rejects conflicting tokens, and checks each file
intent against its original frozen contents. Existing files are held by
verified handles across the batch; symlinks and non-regular paths are rejected.
On Unix those checks compare device/inode identity. A missing target retains a
verified parent-directory handle and is created relative to it with exclusive,
no-follow `openat` only when its first file intent is reached. On non-Unix
platforms exact native file commits fail closed because portable Rust exposes
no equivalent stable file identity. Native process
application regards any exit status as an applied spawn when creation and
waiting succeeded, because the status is already language-visible data.
Independent operating-system effects cannot be made atomically rollbackable
after a host failure; this is truthful preflighted at-most-once application,
not a distributed atomicity claim. A missing target is atomically created only
when its first file intent is reached, so an earlier failed effect cannot create
a later target; once a file intent is reached, that file can legitimately
remain changed if a subsequent heterogeneous effect fails.

---

## 8. Provenance

Provenance tracks which oracle reads influenced each value, forming a join-semilattice: bottom is the empty set (a *pure* value), joins are set unions, and a configurable saturation limit (default 256, `--provenance-limit`) collapses oversized sets to a top element so tracking stays O(1) in degenerate cases. Merges are commutative, associative, and saturating; these laws are enforced by tests.

| Operation | Output provenance |
|-----------|-------------------|
| Literal | ∅ |
| `ORACLE` at a | {a} |
| Binary op | π₁ ∪ π₂ |
| Unary op | π |
| `INPUT` | ∅ |

Provenance does not participate in the convergence test. Its consumers are the Action Principle's cost function (§9), which rewards genuine temporal dependency, and diagnostic display. The saturation limit is a thread-wide setting applied when a TimeLoop is constructed; construct one TimeLoop per run.

---

## 9. The Action Principle

The trivial all-zero state is a fixed point of many programs, and a search seeded with zeros finds it first (the Genie Effect: the timeline that answers your question by having nothing happen). Action-guided mode explores several deterministically generated seeds (zeros, small primes, sequences, powers of two, and seed-derived values), collects every fixed point found, and selects the one minimising an action functional:

```
action(S) =   Σ  [ pure(v)·w_pure  -  w_causal·depth_scale(|π(v)|)  +  unchanged(v)·w_unchanged ]
            v∈S
            + trivial-state penalty  -  w_zero·ln(1 + nonzero-count)
            - w_entropy·H(values)    -  w_output·ln(1 + outputs)
```

Lower action wins. The functional penalises pure and unchanged cells, rewards causal depth (on a piecewise linear-then-logarithmic curve, so deep temporal computation is not undervalued), rewards value diversity through Shannon entropy, and strongly rewards programs that produce output. Causal depth uses one curve everywhere; ties between equal-action candidates break by canonical lexicographic memory order, keeping selection deterministic.

The CLI derives the weights from the program's own temporal footprint: `w_zero = −ln(1−ρ)` from expected sparsity ρ, `w_pure = −ln τ` from the temporal fraction τ, and `w_causal = ln β` from an assumed branching factor β = 2, using the program's ORACLE/PROPHECY count as the temporal-core estimate. Programs with no measurable temporal core fall back to fixed anti-trivial weights.

The configured standard-orbit seed is included in this candidate set (replacing
the last heuristic seed when necessary while preserving the configured seed
budget). It is not retried as a ledgerless fallback: if only that candidate is
consistent, its observations and effects form the ordinary selected batch.

---

## 10. Type system

`--typecheck` reports the mandatory static temporal-tainting analysis before execution: every stack slot is Pure, Temporal, or Unknown, ORACLE produces a duplicable Temporal value, and taint joins across operators and control flow. ORACLE values are not linear resources; `DUP`, `OVER`, and `PICK` have the same value semantics for temporal and ordinary words. Procedures may declare effects that the checker verifies. File-backed execution, verification, build, REPL, and tooling paths run the resolved-HIR analysis whether or not its report is printed.

Cell types are recorded only for PROPHECY writes whose address is statically known (the `value addr PROPHECY` idiom with a literal address). Writes through computed addresses are counted and reported as untyped rather than being fabricated into the cell map.

`TEMPORAL base size BITS n { ... }` is the finite verification boundary inside
the otherwise general-purpose language. Addresses used by ORACLE, PROPHECY,
PRESENT, PACK, UNPACK, INDEX, and STORE are relative to the region; bounds are
handled within `0..size`, then translated by `base`. PROPHECY, PACK, and STORE
writes retain only the low `n` bits. Region entry validates that the entire
range fits configured memory. Nested regions are rejected. `--typecheck`
prints a region contract and rejects operations lacking total deterministic IR
lowering, recursion, and external or nondeterministic effects inside a region;
the same operations remain available to ordinary code outside the finite core.

---

## 11. SMT encoding

`--smt` exports the same typed QF_ABV temporal IR used by the in-process Z3
backend. Arrays use 64-bit indices and values; non-power-of-two address spaces
are normalized with the actual bounds policy rather than padded/truncated.
The fragment covers the total deterministic opcodes reported by the region
checker, structured IF, bounded WHILE, nonrecursive procedures, buffered
OUTPUT/EMIT observations, PARADOX path rejection, and typed TEMPORAL scopes.
INPUT and unmodeled effects are hard errors.

`--global` solves one point state. `--all-fixed` blocks the first complete
memory model and resolves, proving uniqueness or returning two replayed
witnesses. `--verify` searches for a fixed state violating each source
PROPERTY; complete UNSAT proves the property, SAT refutes it with a replayed
counterexample, and absence of every fixed state is explicitly `VACUOUS`.
`--artifact` writes these results using schema
`ourochronos.verification/v1` (or the property-batch schema). Property
artifacts include the deterministic sorted touched-address slice and retained
cell names. This is a sound relevance report, explicitly not a claim that the
solver incrementally reuses prior queries.

`--recurrent` is a separate explicit-state backend. It evaluates all
`2^(memory-cells*state-bits)` states up to `--state-limit`, requires every
transition to finish and remain inside the declared word domain, then reports
every recurrent class, point state, transient distance, and basin size. A
non-closed or oversized domain is refused. `--dot` exports the complete graph.

---

## 12. Programming patterns

### 12.1 The canonical temporal pattern

Almost every useful temporal program has four parts:

```
0 ORACLE            # 1. Read the future's claim
<verify it>         # 2. Check the claim against your condition
IF {
    DUP 0 PROPHECY  # 3. Correct: stabilise the timeline
    OUTPUT
} ELSE {
    <perturb>       # 4. Wrong: write something else, forcing a new epoch
    0 PROPHECY
}
```

The perturbation determines the search trajectory. Incrementing walks the candidate space; writing a computed correction converges faster; writing the negation guarantees a paradox.

### 12.2 Convergence pitfalls

**Incrementing feedback has no point fixed state.** `0 ORACLE 1 ADD 0 PROPHECY` has no solution under wrapping `u64` arithmetic and follows an enormous cycle. A finite diagnostic prefix is reported as monotone growth; Deutsch mode would regard the eventual cycle's uniform distribution as stationary if the search could reach the repeat. Bound every oracle-derived update, or make the update conditional so some value maps to itself when point consistency is intended.

**Negation oscillates.** Any cell whose new value is the logical inverse of its oracle value is a grandfather paradox with period 2. This includes indirect inversions through arithmetic.

**Every ORACLE needs a corresponding PROPHECY.** A cell that is read but never written converges only to zero; a cell written from its own oracle value converges immediately. Reading cell 3 and writing only cell 5 leaves cell 3 pinned at the seed.

**Stratify where possible.** If cell dependencies form an acyclic chain (5 depends on 3, 3 depends on 0), the search converges in one epoch per stratum. Cycles between cells are what require genuine search.

---

## 13. Command-line interface

```
ourochronos <file.ouro> [options]
ourochronos repl
ourochronos link <output.ourobc> <input.ouroobj>...
ourochronos run-package <file.ouropkg>
```

| Flag | Meaning |
|------|---------|
| `--diagnostic` | Trajectory recording, divergence detection, paradox diagnosis |
| `--deutsch` | Stationary-cycle semantics and cycle-wide readout agreement |
| `--stationary` | Require and solve a source-level MARKOV declaration; MARKOV-only files select this automatically |
| `--quantum-fixed` | Require and analyze a source-level qubit QCHANNEL; QCHANNEL-only files select this automatically |
| `--action` | Action-guided search (weights derived per §9) |
| `--seeds <n>` | Seeds to explore in action mode (default 4) |
| `--seed <n>` | Initial anamnesis seed value |
| `--typecheck` | Print the mandatory static temporal type/effect reports |
| `--check` | Run every mandatory compiler and verifier gate without executing |
| `--emit-object <file>` | Emit the entry source's relocatable `OUROOBJ`; dependencies remain typed imports |
| `--emit-objects <directory>` | Emit every source/prelude object into a required-empty directory |
| `--emit-bytecode <file>` | Emit deterministic linked `OUROBC` bytecode |
| `--build <file>` | Emit a deterministic portable `OUROPK` package |
| `--build-executable <file>` | Emit a platform-native runtime launcher containing one validated `OUROPK` package |
| `--embed-global-witness` | With `--build`/`--build-executable`, globally solve and embed an independently replayed initial point state |
| `--runtime-global-package` | With a build action, declare the exact versioned Z3 global-point runtime dependency |
| `--global` | Solve `F(s)=s` over the typed finite IR and replay the witness |
| `--all-fixed` | Prove zero/unique/multiple point fixed states globally |
| `--verify` | Check every source PROPERTY over all point fixed states |
| `--recurrent` | Exhaustively classify a bounded closed transition graph |
| `--state-bits <n>` | Bits per cell in the recurrent domain (default 1) |
| `--state-limit <n>` | Maximum explicit states (default 65,536) |
| `--solver-timeout <ms>` | In-process solver timeout (default 30,000) |
| `--loop-unroll <n>` | Loop depth represented by symbolic lowering (default 10) |
| `--artifact <file>` | Versioned JSON proof/witness/counterexample artifact |
| `--dot <file>` | Graphviz export for `--recurrent` |
| `--smt` | Export the typed temporal IR as SMT-LIB2 |
| `--fast` | Immutable prevalidated bytecode dispatch for pure programs (differential-tested against ordinary bytecode dispatch) |
| `--max-inst <n>` | Instruction budget per epoch (default 10,000,000) |
| `--memory-cells <n>` | Positive finite temporal-memory width (default 65,536) |
| `--resources` | Print the concrete state width, address width, gas/epoch bounds, semantics, and the fact that one instance is not a PSPACE-family proof |
| `--halting-bound <n>` | Run the deterministic non-temporal core for at most n instructions; return HALTED or UNKNOWN-within-bound, never unrestricted nonhalting |
| `--effects decline\|unrestricted` | Effect gate policy (§7; default decline) |
| `--allow-file-read <path>` / `--allow-file-write <path>` | Exact repeatable frozen-read / selected-write capabilities |
| `--network-input <host:port>=<file>` | Exact repeatable frozen receive tape; does not authorize send |
| `--allow-network-send <host:port>` | Exact repeatable selected TCP-send capability |
| `--allow-process <descriptor>` | Frozen result plus exact selected shell-command capability |
| `--allow-sleep-ms <n>` | Maximum selected sleep duration |
| `--strict` / `--permissive` | Error-handling policy for bounds and underflow |
| `--provenance-limit <n>` | Provenance saturation limit (default 256) |
| `--audit [file]`, `--audit-json` | Structured logging of parse, startup, and run outcome |
| `--lsp` | Language server (build with `--features lsp`) |

Exit codes: 0 consistent/proven/decided (or bounded-halting positive), 1 usage/parse/runtime/unsupported-analysis error, 2 paradox, nonexistence, ambiguity, refuted property, or vacuous property, 3 resource-bounded unknown (including bounded-loop UNSAT and solver timeout).

The six source compiler actions are mutually exclusive and cannot be combined with
an execution or search mode. A portable package contains a UTF-8 identity,
runtime ABI version, finite memory width, and one validated linked bytecode
artifact. `run-package` validates the container, bytecode, and CFG before
running the standard iterative point-orbit machine. Package execution rejects
foreign calls, live input, and primitives outside that machine. The package is
portable bytecode, not a native executable. `--build-executable` copies the
currently running, platform-specific runtime and appends that package under a
bounded footer; invoking the result with no arguments validates and runs the
embedded package. Rebuilding from identical runtime and package bytes is
byte-identical. This envelope is not a cross-platform AOT format or a
cryptographic signature, and modifying a signed executable after signing can
invalidate the platform signature; sign the completed launcher where required.

`OUROPK` v3 records the exact memory-bounds policy, instruction budget, closed
resolution policy, and optional exact solver dependency. Orbit packages follow
their declared seed policy. Embedded-point packages carry a canonical sparse
state, exact replay instruction count, and a recomputable evidence digest that
binds the manifest, linked bytecode, state, and count. Decode recomputes that
digest and replays the state, rejecting the package unless the executable
transition reproduces the same point state and exact count. Runtime-global
packages declare the versioned Z3 contract and never silently fall back to an
orbit if the solve is unavailable or inconclusive. These policies do not
change `--all-fixed` or `--recurrent` semantics.

Package decoding checks all lengths and table counts before allocation,
structurally validates the embedded `OUROBC`, checks runtime compatibility, and
independently verifies its CFG before any embedded witness is replayed. Witness
replay must finish within the declared package gas and match its recorded exact
instruction count.

---

## 14. Implementation notes

Stack underflow is an epoch error under the default policy (`--permissive` substitutes zeros). Addresses are 64-bit and are checked against the configured memory width; the selected bounds policy errors, wraps modulo `m`, or clamps. Shift amounts mask to 6 bits. Division and modulo by zero return 0. Invalid run configurations (zero or oversized widths, zero budgets, or invalid action seeds) are rejected at construction, not discovered as empty searches. Standard bytecode temporal memory uses persistent 256-cell copy-on-write pages with exact equality after cached-hash filtering. Candidate output is buffered and only the selected fixed epoch is committed to the public result. The transaction API represents frozen input and typed effect intents. An explicitly authorized `EffectCommitAdapter` receives only the selected ledgered batch. `NativeEffectAdapter` is deny-by-default, grants exact file, send-endpoint, process-invocation, sleep-duration, and custom-namespace capabilities, and independently deduplicates commit token plus batch digest. The unified VM lowers file writes, socket sends, process spawns, and sleeps into those intents; it performs no corresponding host operation during candidate evaluation. Portable packages and proof lowering still reject these environment dependencies until their manifests/proof inputs can declare the required snapshots and capabilities.

Resource admission is finite and fail-closed. A source file is limited to
8 MiB. A module graph permits 256 modules, 128 imports per module, import depth
64, 4,096 edges, and 64 MiB retained source; parser nesting is limited to 64.
Default VM configuration permits 1,024 file snapshots and applies per-item/
identity limits plus 64 MiB aggregate budgets independently to frozen files,
endpoint receive tapes, process identities/output, and staged effect bytes.
`OUROBC` is limited to 64 MiB and one million instructions; `OUROOBJ` is
limited to 128 MiB, and CLI linking admits at most 4,096 objects within a
128 MiB aggregate input budget. `OUROPK` derives a hard byte cap from its
bounded artifact and witness tables, limits memory to 1,048,576 cells, and
limits execution/replay gas to 10,000,000 instructions. All of these are
admission limits, not claims that a permitted maximum will fit every host.

The public `Executor` keeps its historical source-shaped API but is a bytecode
facade: it type-checks, resolves HIR, runs mandatory semantic analysis,
compiles and independently verifies bytecode, then dispatches the same VM. Its
former direct AST walker is private and present only under `cfg(test)` for
differential parity.

`PreparedBytecode` is the optimized execution form. It owns an immutable
`BytecodeProgram`, performs complete structural validation when sealed, and
then uses the ordinary bytecode dispatcher without rescanning the artifact on
each epoch. It is not a divergent opcode implementation or a gas-changing
peephole optimizer. Reference and prepared dispatch therefore count the same
instructions and expose the same intermediate resource-limit failures.

Linked `FOREIGN` declarations use a deliberately narrow scalar ABI: zero to sixteen `u64`/`i64` arguments and zero or one `u64`/`i64` result. Signed words retain their exact two's-complement bits. Bytecode stores the typed ID, library, external symbol alias, parameter/result types, and effects; the verifier uses that table for exact stack effects. Execution requires a process host to bind every descriptor exactly before dispatch. One call consumes one bytecode instruction, so gas bounds call count; underflow and projected stack overflow are checked before the callback fires, and result-shape mismatches are typed errors. Host callbacks are trusted and are not preempted. The optional dynamic-library adapter is an explicitly unsafe API and supports only up to six unsigned `u64` arguments/results because a shared object cannot prove its C signature. Pointer, string, handle, signed, 32-bit, and multiple-result declarations never enter that dynamic adapter; the safe process-host table retains the full linked `u64`/`i64` subset. All temporal search/SMT modes continue to decline foreign effects, and portable packages reject any foreign descriptor because their current manifest has no host-dependency schema.

### 14.1 Complexity metadata API

`ConcreteResourceProfile` reports exact resources for one run: `m`, `64m` temporal state bits, required address bits, temporal-op count, epoch gas, epoch-search bound, and consistency semantics. It always labels itself a finite instance rather than a PSPACE-family proof.

The top-level `FAMILY` declaration records polynomial bounds for CTC cells, chronology-respecting bits, and transition steps, plus explicit declarations of polynomial-time uniformity, transition totality, all-fixed-point readout agreement, ideal Deutsch selection, and frozen/modeled effects. It is preserved in `Program.family_declaration` and converts to `PspaceFamilyContract`. `missing_obligations()` exposes absent assumptions. `declared_eligible()` means all obligations were declared; it is not a machine-checked proof of those semantic facts. At most one FAMILY declaration is permitted; all three polynomial fields are required. `POLY c d a` denotes `c*n^d+a`.

### 14.2 Exact stochastic backend

`RationalMarkovChain` implements finite classical Deutsch semantics beyond deterministic maps. It accepts an exact rational row-stochastic matrix, rejects negative or non-normalized rows, computes every closed recurrent class, and solves one extremal stationary distribution per class using exact rational Gaussian elimination. `StationaryFamily` represents their convex hull and computes exact acceptance probabilities for a caller-supplied decision predicate. The implementation uses checked `i128` rational arithmetic and reports overflow; it is intended for finite research instances, not arbitrary-precision production solving.

The top-level `MARKOV` declaration exposes this backend in Ourochronos source. Every row is required exactly once and must contain `STATES` probabilities. `ACCEPTING` defines the chronology-respecting decision predicate. Analysis classifies ACCEPT only if every extremal stationary distribution meets `ACCEPT_AT_LEAST`, REJECT only if every one meets `REJECT_AT_MOST`, and AMBIGUOUS otherwise. Since every stationary distribution is a convex combination of the extremal distributions, checking the extremal recurrent classes is sufficient. Ambiguity exits 2 and names every class and exact acceptance probability; the runtime never chooses a favorable fixed point.

### 14.3 Bounded halting analyzer

`BoundedHaltingAnalyzer` covers the deterministic, non-temporal, externally isolated language core. `Halted` is a positive witness. `NotHaltedWithinBound { T }` means only that the observed run executed T instructions without halting and is an unknown result for unbounded execution. Temporal, nondeterministic, and externally stateful opcodes are rejected as `Unsupported`; ordinary runtime failures remain distinct. The CLI maps bounded exhaustion to exit code 3.

### 14.4 Qubit quantum CPTP backend

`QuantumChannel` is a separate finite-dimensional research backend. A channel is constructed from dense complex Kraus operators. Kraus form guarantees complete positivity; construction numerically checks `sum_i K_i^dagger K_i = I` within a caller-supplied tolerance, establishing trace preservation to that tolerance. `apply` computes `Phi(rho) = sum_i K_i rho K_i^dagger`.

`fixed_point` begins with the maximally mixed density operator and forms Cesaro averages of successive channel iterates. It returns only when the Frobenius residual `||Phi(rho)-rho||_F` is within tolerance and unit trace is retained; otherwise it returns a nonconvergence error with the final residual. This remains a one-point numerical exploration API.

The top-level `QCHANNEL` form provides the stronger qubit decision semantics. Each Kraus operator has four row-major complex entries; `C re im` gives signed rational real and imaginary parts. The analyzer recovers the affine Bloch map `r -> Ar+c`, solves the entire fixed affine space `(I-A)r=c`, intersects it with the Bloch ball, and analytically minimizes/maximizes the selected computational-basis acceptance probability over that complete intersection. ACCEPT and REJECT require the respective threshold on every fixed density; otherwise the result is AMBIGUOUS and exits 2. `quantum_reset.ouro` has a singleton accepting fixed density, whereas `quantum_identity.ouro` fixes the whole Bloch ball and correctly reports acceptance range `[0,1]`.

This complete fixed-space verifier is specific to qubits and computational-basis readout. Floating-point tolerances remain explicit. Higher dimensions require semidefinite optimization or an equivalent exact characterization and are not implemented.

---

## Appendix A: Opcode reference

Stack effects: `( before -- after )`, top of stack rightmost. Effect classes per §7 are marked E (external) and N (non-deterministic); unmarked opcodes are pure.

| Opcode | Stack effect | Description |
|--------|--------------|-------------|
| `NOP` | `( -- )` | No operation. |
| `HALT` | `( -- )` | Halt execution of the current epoch. |
| `POP` | `( a -- )` | Pop and discard the top of stack. |
| `DUP` | `( a -- a a )` | Duplicate the top of stack. |
| `SWAP` | `( a b -- b a )` | Swap the top two elements. |
| `OVER` | `( a b -- a b a )` | Copy the second element to the top. |
| `ROT` | `( a b c -- b c a )` | Rotate the top three elements. |
| `DEPTH` | `( -- n )` | Push the current stack depth. |
| `PICK` | `( n -- v )` | Copy the nth element to the top. |
| `ROLL` | `( n -- )` | Roll the nth element to the top. |
| `REVERSE` | `( n -- )` | Reverse the top n elements. |
| `EXEC` | `( quote_id -- )` | Execute the quotation with the given id. |
| `DIP` | `( x quote_id -- x )` | Execute a quotation with the top element hidden. |
| `KEEP` | `( x quote -- ... x )` | Keep x, execute quote, restore x. |
| `BI` | `( x p q -- ... ... )` | Apply p to x, then q to x. |
| `REC` | `( quote -- ... )` | Execute quote with itself on the stack. |
| `STR_REV` | `( chars... len -- reversed... len )` | Reverse a string. |
| `STR_CAT` | `( c1.. len1 c2.. len2 -- c1..c2.. len1+len2 )` | Concatenate two strings. |
| `STR_SPLIT` | `( chars... len delim -- s1 .. sn count )` | Split a string by delimiter. |
| `ADD` | `( a b -- a+b )` | Addition (wrapping). |
| `SUB` | `( a b -- a-b )` | Subtraction (wrapping). |
| `MUL` | `( a b -- a*b )` | Multiplication (wrapping). |
| `DIV` | `( a b -- a/b )` | Division; 0 if divisor is 0. |
| `MOD` | `( a b -- a%b )` | Modulo; 0 if divisor is 0. |
| `NEG` | `( a -- -a )` | Two's-complement negation. |
| `ABS` | `( a -- \|a\| )` | Absolute value (signed). |
| `MIN` | `( a b -- min(a,b) )` | Minimum. |
| `MAX` | `( a b -- max(a,b) )` | Maximum. |
| `SIGN` | `( a -- sign(a) )` | Signum (signed): -1, 0, or 1. |
| `ASSERT` | `( cond -- )` with message | Assert; syntax `cond ASSERT "message"`. |
| `NOT` | `( a -- !a )` | Logical NOT. |
| `AND` | `( a b -- a&b )` | Bitwise AND. |
| `OR` | `( a b -- a\|b )` | Bitwise OR. |
| `XOR` | `( a b -- a^b )` | Bitwise XOR. |
| `SHL` | `( a n -- a<<n )` | Left shift. |
| `SHR` | `( a n -- a>>n )` | Logical right shift. |
| `EQ` | `( a b -- a==b )` | Equal. |
| `NEQ` | `( a b -- a!=b )` | Not equal. |
| `LT` | `( a b -- a<b )` | Less than (unsigned). |
| `GT` | `( a b -- a>b )` | Greater than (unsigned). |
| `LTE` | `( a b -- a<=b )` | Less or equal (unsigned). |
| `GTE` | `( a b -- a>=b )` | Greater or equal (unsigned). |
| `SLT` | `( a b -- a<b )` | Less than (signed). |
| `SGT` | `( a b -- a>b )` | Greater than (signed). |
| `SLTE` | `( a b -- a<=b )` | Less or equal (signed). |
| `SGTE` | `( a b -- a>=b )` | Greater or equal (signed). |
| `ORACLE` | `( addr -- A[addr] )` | Read from the anamnesis. |
| `PROPHECY` | `( value addr -- )` | Write to the present. |
| `PRESENT` | `( addr -- P[addr] )` | Read from the present. |
| `PARADOX` | `( -- )` | Abort the epoch as inconsistent. |
| `PACK` | `( v1 .. vn base n -- )` | Store n values at base. |
| `UNPACK` | `( base n -- v1 .. vn )` | Load n values from base. |
| `INDEX` | `( base index -- P[base+index] )` | Indexed read. |
| `STORE` | `( value base index -- )` | Indexed write. |
| `INPUT` | `( -- value )` | Read input (scripted, then interactive; frozen in searches). |
| `OUTPUT` | `( value -- )` | Buffer a value for output. |
| `EMIT` | `( char -- )` | Buffer a character for output. |
| `VEC_NEW` | `( -- vec )` | New vector (epoch-local). |
| `VEC_PUSH` | `( vec value -- vec )` | Append to a vector. |
| `VEC_POP` | `( vec -- vec value )` | Pop from a vector. |
| `VEC_GET` | `( vec index -- vec value )` | Read a vector element. |
| `VEC_SET` | `( vec value index -- vec )` | Write a vector element. |
| `VEC_LEN` | `( vec -- vec length )` | Vector length. |
| `HASH_NEW` | `( -- hash )` | New hash table (epoch-local). |
| `HASH_PUT` | `( hash key value -- hash )` | Insert or update. |
| `HASH_GET` | `( hash key -- hash value found )` | Lookup. |
| `HASH_DEL` | `( hash key -- hash )` | Delete a key. |
| `HASH_HAS` | `( hash key -- hash found )` | Key existence. |
| `HASH_LEN` | `( hash -- hash count )` | Entry count. |
| `SET_NEW` | `( -- set )` | New set (epoch-local). |
| `SET_ADD` | `( set value -- set )` | Add an element. |
| `SET_HAS` | `( set value -- set found )` | Membership. |
| `SET_DEL` | `( set value -- set )` | Remove an element. |
| `SET_LEN` | `( set -- set count )` | Element count. |
| `FFI_CALL` | `( args... ffi_id -- results... )` | Call an FFI function by id. E |
| `FFI_CALL_NAMED` | `( args... name -- results... )` | Call an FFI function by name. E |
| `FILE_OPEN` | `( path mode -- file )` | Open a file. E for write/append/create/truncate modes |
| `FILE_READ` | `( file max -- file buffer read )` | Read into a buffer. |
| `FILE_WRITE` | `( file buffer -- file written )` | Write a buffer. E |
| `FILE_SEEK` | `( file offset origin -- file pos )` | Seek. |
| `FILE_FLUSH` | `( file -- file )` | Flush to disk. |
| `FILE_CLOSE` | `( file -- )` | Close a file. |
| `FILE_EXISTS` | `( path -- exists )` | Existence check. |
| `FILE_SIZE` | `( path -- size )` | File size. |
| `BUFFER_NEW` | `( capacity -- buffer )` | New byte buffer (epoch-local). |
| `BUFFER_FROM_STACK` | `( b1 .. bn n -- buffer )` | Build a buffer from bytes. |
| `BUFFER_TO_STACK` | `( buffer -- b1 .. bn n )` | Spill a buffer to the stack. |
| `BUFFER_LEN` | `( buffer -- buffer length )` | Buffer length. |
| `BUFFER_READ_BYTE` | `( buffer -- buffer byte )` | Read at the cursor. |
| `BUFFER_WRITE_BYTE` | `( buffer byte -- buffer )` | Write at the cursor. |
| `BUFFER_FREE` | `( buffer -- )` | Free a buffer. |
| `TCP_CONNECT` | `( host port -- socket )` | Connect to a TCP server. E |
| `SOCKET_SEND` | `( socket buffer -- socket sent )` | Send. E |
| `SOCKET_RECV` | `( socket max -- socket buffer received )` | Receive. E |
| `SOCKET_CLOSE` | `( socket -- )` | Close a socket. |
| `PROC_EXEC` | `( command -- output exit_code )` | Run a shell command. E |
| `CLOCK` | `( -- timestamp )` | Next word from the frozen clock tape; unrestricted ordinary execution may capture the first candidate from the system clock. N |
| `SLEEP` | `( milliseconds -- )` | Delay. E |
| `RANDOM` | `( -- value )` | Next word from the frozen random tape; unrestricted ordinary execution may capture the first candidate from the process time-mixed sampler. N |

Keyword aliases: `DROP` for `POP`, `PEEK` for `PICK`, `REV` for `REVERSE`, `CALL` for `EXEC`, `CLEAVE` for `BI`, `CONCAT` for `STR_CAT`, `SPLIT` for `STR_SPLIT`, and `PROC`/`FN` for `PROCEDURE`.

---

## Appendix B: Error numbering

Errors carry structured context and a stable numeric code by category: runtime errors (stack, arithmetic, memory bounds, instruction limits) 1000 to 1999, temporal errors 2000 to 2999, parse errors 3000 to 3999, type errors 4000 to 4999, data-structure errors 5000 to 5999, FFI errors 6000 to 6999, I/O errors 7000 to 7999, and internal errors 9000 to 9999 (9001 internal, 9002 invalid run configuration). `src/core/error.rs` is the authoritative catalogue; the CLI maps every error to exit code 1.
