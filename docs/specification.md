# Ourochronos Specification

This document defines the Ourochronos abstract machine, its instruction set, the fixed-point execution model, and the behaviour of the reference implementation. It is the single reference for the language: the README introduces the ideas, and everything normative lives here.

Sections 2 to 8 define the machine and its semantics. Sections 9 to 12 describe the analysis and verification layers built on it. Section 13 covers practical programming patterns, and the appendices catalogue the full instruction set and error numbering.

---

## 1. The computational model

Ourochronos implements Deutsch's self-consistency model of closed timelike curves as an executable language. A program does not run once; it defines a transformation F over memory states, and execution searches for a state S with F(S) = S. The program reads the candidate state through ORACLE (the memory as it will be at the end of the run, called the *anamnesis*) and writes the next candidate through PROPHECY (the *present*). When the present equals the anamnesis, the timeline is consistent and the run reports that state. When no such state exists, the search detects the failure mode (oscillation, divergence, or an explicit PARADOX) and reports it.

Aaronson and Watrous showed that classical computation with access to such a fixed-point constraint decides exactly PSPACE. The language therefore expresses solutions to search problems as consistency conditions: the program verifies a witness read from the future and stabilises the timeline when the witness is correct.

---

## 2. Abstract machine

### 2.1 State components

Each *epoch* (one evaluation of F) operates on the following state:

| Component | Type | Description |
|-----------|------|-------------|
| Stack (σ) | List(V) | LIFO operand stack |
| Anamnesis (A) | Addr → V | Read-only memory: the candidate fixed point |
| Present (P) | Addr → V | Read-write memory being constructed |
| Output (O) | List(V ∪ Char) | Output buffer for this epoch |
| Status (ξ) | {Running, Finished, Paradox, Error} | Epoch status |
| Scratch | heaps, handles, I/O contexts | Epoch-local working state (§3) |

where V is the set of 64-bit unsigned values and Addr the set of 16-bit addresses (65,536 cells).

### 2.2 Values

A value is a pair (n, π): the numeric value n and its *provenance* π ⊆ Addr, the set of oracle addresses that influenced it. Provenance is analysis metadata; the fixed-point test compares numeric values only. Section 8 gives the provenance algebra.

### 2.3 Initial epoch state

At the start of each epoch the stack and output are empty, the present is all zeros, the status is Running, and the anamnesis is the previous epoch's present (or the seed for the first epoch).

---

## 3. The state boundary

The S in F(S) = S is the memory array and nothing else. This is a definition, not an implementation accident, and three consequences follow.

First, data structures (vectors, hash tables, sets), buffers, file handles, and FFI contexts are *epoch-local scratch*. They are created fresh each epoch and are not part of the convergence test. A program that wants a value to survive the loop must write it to memory with PROPHECY; a value held only in a vector cannot participate in the fixed point.

Second, the epoch function must be a fixed function of the memory state, so operations that break that property are declined inside a fixed-point search by default (§7): external effects would fire once per search epoch rather than once per consistent timeline, and non-deterministic sources would make F vary between the iterations comparing its output.

Third, external input is frozen. The first epoch that consumes INPUT values fixes them, and every later epoch replays the same values; an epoch that tries to read past the frozen stream is an error, not a fresh prompt. Information flows from the external world into the loop, never backwards; the search cannot re-prompt the user into a different timeline.

A program with no ORACLE anywhere (body, quotations, or procedure bodies) never reads the anamnesis, so F is constant and its unique fixed point is the result of a single epoch. Such *trivially consistent* programs run once, and the state-boundary rules for searches do not apply to them.

---

## 4. Execution model

### 4.1 The search

```
S₀ = seed (all zeros, or diverse seeds in action-guided mode)
Sₙ₊₁ = F(Sₙ)
stop when Sₙ₊₁ = Sₙ            (consistent)
      or Sₙ revisits a state   (oscillation, period = distance to the earlier visit)
      or values grow without bound (divergence, diagnostic mode)
      or the epoch budget runs out (timeout)
```

The convergence test is exact equality of all 65,536 cells' numeric values. The default epoch budget is 1,000 (`DEFAULT_MAX_EPOCHS`), shared by the CLI, the REPL, and the library, and overridable per run.

### 4.2 Verified cycle detection

Visited states are journalled under a 64-bit state hash. The hash is a fold, so distinct states can collide; a revisit is therefore only declared after the journalled state (stored in sparse non-zero-cell form, which determines the full state) compares equal to the current one. The epoch result cache applies the same rule: a cache hit requires the stored anamnesis to match, not merely its hash. A hash collision costs a re-execution, never a wrong verdict.

### 4.3 Execution modes

| Mode | Flag | Behaviour |
|------|------|-----------|
| Standard | (default) | Search with cycle detection and epoch caching; first fixed point wins |
| Diagnostic | `--diagnostic` | Standard plus full trajectory recording, divergence detection, and paradox diagnosis |
| Action-guided | `--action` | Explore several seeds and select the fixed point with least action (§9) |
| Pure | (library) | Unbounded iteration for theoretical exploration |

All modes share the same input freezing, effect gate, and verified cycle detection.

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
TEMPORAL base size { body }         isolated temporal region, base-relative
TEMPORAL name @ addr DEFAULT v;     named temporal variable at a fixed address
MANIFEST name = value;              compile-time constant
LET name = expr;                    named binding (compiled to PICK)
PROCEDURE name { body }             procedure, inlined at call sites
FOREIGN { ... }                     FFI declarations
IMPORT "file"                       source inclusion
```

### 5.2 Grammar

```ebnf
program     = { declaration } { statement }
declaration = manifest | procedure | foreign | import
statement   = literal | opcode | quotation
            | if_stmt | while_stmt | temporal_stmt | let_stmt | block
literal     = integer | "'" CHAR "'" | '"' { CHAR } '"'
quotation   = "[" { statement } "]"
if_stmt     = "IF" block [ "ELSE" block ]
while_stmt  = "WHILE" block block
temporal_stmt = "TEMPORAL" integer integer block
let_stmt    = "LET" IDENT "=" { statement } ";"
manifest    = "MANIFEST" IDENT "=" integer ";"
procedure   = "PROCEDURE" IDENT "{" { statement } "}"
block       = "{" { statement } "}"
```

Opcodes are the keywords of Appendix A. There is no bytecode; the reference implementation interprets the tree directly, and a registration test guarantees every opcode is reachable from source.

---

## 6. Convergence semantics

Let F: M → M be the program's memory transformation. The runtime classifies a run as:

| Status | Condition | Exit code | Interpretation |
|--------|-----------|-----------|----------------|
| Consistent | P = A after an epoch | 0 | A timeline exists; this is it |
| Paradox | explicit PARADOX executed | 2 | The program rejected the timeline |
| Oscillation | a verified earlier state recurs (period k) | 2 | No fixed point on this orbit; the grandfather paradox is period 2 |
| Divergence | monotone unbounded growth (diagnostic mode) | 2 | No fixed point; values escape |
| Timeout | epoch budget exhausted | 3 | Unknown; the search gave up |
| Error | runtime error inside an epoch | 1 | The epoch itself failed |

Diagnostic mode additionally reports the oscillating cells, the cycle states, and recognises the negation pattern (a cell mapped to its own logical NOT) as a grandfather paradox. Divergence detection runs only in diagnostic mode, so a monotonically growing program exits 3 (timeout) under the default mode and 2 (divergence) under `--diagnostic`; the codes differ because the evidence differs, not the program.

---

## 7. Effects and determinism

Every opcode carries an effect class, defined in one place (`OpCode::effect_class`):

| Class | Opcodes | Inside a search |
|-------|---------|-----------------|
| Pure | everything else | Permitted |
| External | `FILE_WRITE SOCKET_SEND SOCKET_RECV TCP_CONNECT PROC_EXEC FFI_CALL FFI_CALL_NAMED SLEEP` | Declined |
| Non-deterministic | `RANDOM CLOCK` | Declined |

Declining is an epoch error naming the opcode, raised at dispatch before operands are consumed. `FILE_OPEN` is mode-dependent and gated where its mode operand is known: write, append, create, and truncate opens decline inside a search (they would rewrite the file on every epoch), read opens are permitted. `SOCKET_RECV` and the FFI calls are External because a receive consumes stream state and a foreign call may do anything. `--effects unrestricted` restores permissive behaviour for experimentation, accepting that an external effect fires on every epoch of the search and that non-determinism may prevent convergence. Trivially consistent programs (§3) are exempt: their single epoch is the timeline itself.

`OUTPUT` and `EMIT` need no gate because they are buffered per epoch; only the converged epoch's buffer is printed. File *reads* are permitted in searches and assumed stable across epochs; a file that changes mid-search violates the model the same way an unfrozen input would. `INPUT` composes with the freeze of §3: once the stream is frozen, reading past its end inside a search is an error rather than a fresh interactive read.

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

---

## 10. Type system

`--typecheck` runs a static temporal-tainting analysis before execution: every stack slot is Pure, Temporal, or Unknown, ORACLE produces Temporal values, and taint joins across operators and control flow. Temporal values are additionally *linear* (consuming one twice is flagged), and procedures may declare effects that the checker verifies.

Cell types are recorded only for PROPHECY writes whose address is statically known (the `value addr PROPHECY` idiom with a literal address). Writes through computed addresses are counted and reported as untyped rather than being fabricated into the cell map.

---

## 11. SMT encoding

`--smt` emits an SMT-LIB2 encoding of the fixed-point constraint over QF_ABV: the anamnesis is an unconstrained array, the program is symbolically executed into store chains, and the final assertion is `present = anamnesis`. SAT models are consistent timelines; UNSAT proves no timeline exists within the encoded fragment.

The fragment covers stack manipulation, arithmetic, bitwise and comparison operators, ORACLE, PROPHECY, PRESENT, INDEX, STORE, IF, and WHILE unrolled to a bounded depth (default 10; deeper iteration is an explicit under-approximation stated in the emitted header). INPUT becomes a fresh symbolic constant. Everything else is a hard encoding error naming the construct: a verdict over a silently truncated program would be unsound. Feed the output to a solver directly, for example `z3 program.smt2`.

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

**Unbounded feedback diverges.** `0 ORACLE 1 ADD 0 PROPHECY` has no fixed point; each epoch increments the last. Bound every oracle-derived update, or make the update conditional so some value maps to itself.

**Negation oscillates.** Any cell whose new value is the logical inverse of its oracle value is a grandfather paradox with period 2. This includes indirect inversions through arithmetic.

**Every ORACLE needs a corresponding PROPHECY.** A cell that is read but never written converges only to zero; a cell written from its own oracle value converges immediately. Reading cell 3 and writing only cell 5 leaves cell 3 pinned at the seed.

**Stratify where possible.** If cell dependencies form an acyclic chain (5 depends on 3, 3 depends on 0), the search converges in one epoch per stratum. Cycles between cells are what require genuine search.

---

## 13. Command-line interface

```
ourochronos <file.ouro> [options]
ourochronos repl
```

| Flag | Meaning |
|------|---------|
| `--diagnostic` | Trajectory recording, divergence detection, paradox diagnosis |
| `--action` | Action-guided search (weights derived per §9) |
| `--seeds <n>` | Seeds to explore in action mode (default 4) |
| `--seed <n>` | Initial anamnesis seed value |
| `--typecheck` | Static temporal type analysis before running |
| `--smt` | Emit SMT-LIB2 instead of running |
| `--fast` | Fast VM for pure programs (differential-tested against the standard VM) |
| `--max-inst <n>` | Instruction budget per epoch (default 10,000,000) |
| `--effects decline\|unrestricted` | Effect gate policy (§7; default decline) |
| `--strict` / `--permissive` | Error-handling policy for bounds and underflow |
| `--provenance-limit <n>` | Provenance saturation limit (default 256) |
| `--audit [file]`, `--audit-json` | Structured logging of parse, startup, and run outcome |
| `--lsp` | Language server (build with `--features lsp`) |

Exit codes: 0 consistent (or pure-run success), 1 usage/parse/runtime error, 2 paradox (explicit, oscillation, or divergence), 3 epoch exhaustion.

---

## 14. Implementation notes

Stack underflow is an epoch error under the default policy (`--permissive` substitutes zeros). All addresses truncate to 16 bits; shift amounts mask to 6 bits. Division and modulo by zero return 0. Invalid run configurations (zero budgets, zero seeds) are rejected at construction, not discovered as empty searches. The `dynamic-ffi` feature enables loading shared libraries at runtime; without it, FFI declarations resolve only to built-ins.

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
| `CLOCK` | `( -- timestamp )` | Milliseconds since the Unix epoch. N |
| `SLEEP` | `( milliseconds -- )` | Delay. E |
| `RANDOM` | `( -- value )` | Time-derived pseudo-random value. N |

Keyword aliases: `DROP` for `POP`, `PEEK` for `PICK`, `REV` for `REVERSE`, `CALL` for `EXEC`, `CLEAVE` for `BI`, `CONCAT` for `STR_CAT`, `SPLIT` for `STR_SPLIT`, and `PROC`/`FN` for `PROCEDURE`.

---

## Appendix B: Error numbering

Errors carry structured context and a stable numeric code by category: runtime errors (stack, arithmetic, memory bounds, instruction limits) 1000 to 1999, temporal errors 2000 to 2999, parse errors 3000 to 3999, type errors 4000 to 4999, data-structure errors 5000 to 5999, FFI errors 6000 to 6999, I/O errors 7000 to 7999, and internal errors 9000 to 9999 (9001 internal, 9002 invalid run configuration). `src/core/error.rs` is the authoritative catalogue; the CLI maps every error to exit code 1.
