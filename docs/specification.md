# Ourochronos Specification

*Formal Definition of the Abstract Machine*

**Version:** 2.0  
**Status:** Stable

---

## 1. Introduction

This document provides the formal specification of the Ourochronos programming language. It defines the abstract machine, instruction set, execution semantics, and convergence behaviour.

The specification is intended to be:
- **Complete:** Every aspect of the language is defined.
- **Precise:** Definitions are unambiguous.
- **Implementable:** The spec can be directly translated to code.

---

## 2. Abstract Machine

### 2.1 State Components

The Ourochronos Abstract Machine (OAM) maintains the following state:

| Component | Type | Description |
|-----------|------|-------------|
| Stack ($\sigma$) | $\text{List}(\mathbb{V})$ | LIFO operand stack |
| Anamnesis ($A$) | $\mathbb{A} \to \mathbb{V}$ | Read-only memory (from future) |
| Present ($P$) | $\mathbb{A} \to \mathbb{V}$ | Read-write memory (being constructed) |
| Output ($O$) | $\text{List}(\mathbb{V})$ | Output buffer |
| Status ($\xi$) | $\{\text{Running}, \text{Halted}, \text{Paradox}\}$ | Execution status |

Where:
- $\mathbb{V} = \{0, 1, ..., 2^{64}-1\}$ is the set of 64-bit unsigned values
- $\mathbb{A} = \{0, 1, ..., 2^{16}-1\}$ is the set of 16-bit addresses

### 2.2 Initial State

At the start of each *epoch*:

| Component | Initial Value |
|-----------|---------------|
| $\sigma$ | $[]$ (empty list) |
| $A$ | Result of previous epoch (or seed) |
| $P$ | $\lambda a. 0$ (all zeros) |
| $O$ | $[]$ (empty list) |
| $\xi$ | Running |

### 2.3 Value Representation

Each value $v \in \mathbb{V}$ is represented as a pair $(n, \pi)$ where:
- $n \in \{0, ..., 2^{64}-1\}$ is the numeric value
- $\pi \subseteq \mathbb{A}$ is the *provenance set* (addresses of oracle reads that influenced this value)

Operations preserve and merge provenance according to §5.

---

## 3. Execution Model

### 3.1 Epochs

Execution proceeds in *epochs*. Each epoch:

1. **Initialisation:** Set $P$ to zeros, $\sigma$ to empty, load $A$ from previous epoch
2. **Execution:** Run the program sequentially
3. **Termination:** Execution ends when the program completes or HALT is reached
4. **Comparison:** Check if $P = A$ (ignoring provenance)

### 3.2 Convergence

After each epoch, the runtime checks for convergence:

**Consistent:** If $P = A$, the timeline is consistent. Return the result.

**Inconsistent:** If $P \neq A$, set $A_{next} = P$ and begin a new epoch.

### 3.3 Cycle Detection

To detect non-termination, the runtime maintains a history of visited states.

**Oscillation:** If a state $P$ has been seen before in epoch $k$, and we are now in epoch $k + n$, the program oscillates with period $n$.

**Divergence:** If values grow monotonically without repetition, the program diverges.

**Timeout:** If epoch count exceeds a limit, report timeout.

---

## 4. Instruction Set Architecture

### 4.1 Notation

Stack effects are written as:

$$(\text{before} \to \text{after})$$

where the rightmost element is the top of stack.

### 4.2 Literals

| Syntax | Effect | Description |
|--------|--------|-------------|
| $n$ | $(\to n)$ | Push integer literal |
| `'c'` | $(\to n)$ | Push character code |
| `"str"` | $(\to c_1 \cdots c_k \, k)$ | Push string chars and length |

### 4.3 Stack Manipulation

| Opcode | Effect | Description |
|--------|--------|-------------|
| `NOP` | $(\to)$ | No operation |
| `HALT` | $(\to)$ | Terminate epoch |
| `POP` | $(a \to)$ | Discard top |
| `DUP` | $(a \to a \, a)$ | Duplicate top |
| `SWAP` | $(a \, b \to b \, a)$ | Swap top two |
| `OVER` | $(a \, b \to a \, b \, a)$ | Copy second to top |
| `ROT` | $(a \, b \, c \to b \, c \, a)$ | Rotate top three |
| `DEPTH` | $(\to n)$ | Push stack depth |
| `PICK` | $(n \to v)$ | Copy $n$th element to top |

### 4.4 Arithmetic

All arithmetic is modulo $2^{64}$ (wrapping).

| Opcode | Effect | Description |
|--------|--------|-------------|
| `ADD` | $(a \, b \to a+b)$ | Addition |
| `SUB` | $(a \, b \to a-b)$ | Subtraction |
| `MUL` | $(a \, b \to a \times b)$ | Multiplication |
| `DIV` | $(a \, b \to a \div b)$ | Division (0 if $b=0$) |
| `MOD` | $(a \, b \to a \mod b)$ | Modulo (0 if $b=0$) |
| `NEG` | $(a \to -a)$ | Negation |

### 4.5 Bitwise Logic

| Opcode | Effect | Description |
|--------|--------|-------------|
| `NOT` | $(a \to \neg a)$ | Bitwise NOT |
| `AND` | $(a \, b \to a \land b)$ | Bitwise AND |
| `OR` | $(a \, b \to a \lor b)$ | Bitwise OR |
| `XOR` | $(a \, b \to a \oplus b)$ | Bitwise XOR |
| `SHL` | $(a \, n \to a \ll n)$ | Left shift |
| `SHR` | $(a \, n \to a \gg n)$ | Right shift (logical) |

### 4.6 Comparison

Results: 1 if true, 0 if false.

| Opcode | Effect | Description |
|--------|--------|-------------|
| `EQ` | $(a \, b \to a = b)$ | Equal |
| `NEQ` | $(a \, b \to a \neq b)$ | Not equal |
| `LT` | $(a \, b \to a < b)$ | Less than |
| `GT` | $(a \, b \to a > b)$ | Greater than |
| `LTE` | $(a \, b \to a \leq b)$ | Less or equal |
| `GTE` | $(a \, b \to a \geq b)$ | Greater or equal |

### 4.7 Temporal Operations

These are the core operators that enable retrocausal computation.

| Opcode | Effect | Description |
|--------|--------|-------------|
| `ORACLE` | $(addr \to A[addr])$ | Read from Anamnesis |
| `PROPHECY` | $(v \, addr \to)$ | Write to Present |
| `PRESENT` | $(addr \to P[addr])$ | Read from Present |
| `PARADOX` | $(\to)$ | Abort epoch as paradoxical |

**ORACLE Semantics:**

```
ORACLE : (addr → v)
  let a = pop()
  let v = A[a mod 2^16]
  let v' = (v.value, v.provenance ∪ {a mod 2^16})
  push(v')
```

The result's provenance includes the read address, marking it as temporally dependent.

**PROPHECY Semantics:**

```
PROPHECY : (v addr →)
  let a = pop()
  let v = pop()
  P[a mod 2^16] ← v
```

### 4.8 Control Flow

**IF Statement:**

```
condition IF { then-block } ELSE { else-block }
```

Semantics:
1. Pop condition from stack
2. If condition ≠ 0, execute then-block
3. Otherwise, execute else-block (if present)

**WHILE Loop:**

```
WHILE { condition-block } { body-block }
```

Semantics:
1. Execute condition-block
2. Pop result
3. If result ≠ 0, execute body-block and goto 1
4. Otherwise, continue

### 4.9 Input/Output

| Opcode | Effect | Description |
|--------|--------|-------------|
| `INPUT` | $(\to v)$ | Read from input |
| `OUTPUT` | $(v \to)$ | Write to output |

### 4.10 Memory Operations

| Opcode | Effect | Description |
|--------|--------|-------------|
| `PACK` | $(v_1 \cdots v_n \, base \, n \to)$ | Store $n$ values at $base$ |
| `UNPACK` | $(base \, n \to v_1 \cdots v_n)$ | Load $n$ values from $base$ |
| `INDEX` | $(base \, i \to P[base+i])$ | Indexed read |
| `STORE` | $(v \, base \, i \to)$ | Indexed write |

---

## 5. Provenance Semantics

### 5.1 Provenance Lattice

Provenance forms a join-semilattice:

- $\bot = \emptyset$ (no temporal dependency)
- $\top = \mathbb{A}$ (depends on all addresses)
- $\pi_1 \sqcup \pi_2 = \pi_1 \cup \pi_2$ (union of dependencies)

### 5.2 Provenance Rules

| Operation | Output Provenance |
|-----------|-------------------|
| Literal $n$ | $\emptyset$ |
| `ORACLE` at $a$ | $\{a\}$ |
| Binary op $(v_1, v_2)$ | $\pi_1 \cup \pi_2$ |
| Unary op $(v)$ | $\pi$ |
| `INPUT` | $\emptyset$ |
| `PRESENT` | $\emptyset$ (or tracked separately) |

### 5.3 Temporal Purity

A value is *pure* if its provenance is empty. A value is *temporal* if its provenance is non-empty.

Pure values do not depend on the future. Temporal values may differ across epochs until convergence.

---

## 6. Declarations

### 6.1 Manifest Constants

```
MANIFEST name = value;
```

Defines a compile-time constant. The name becomes a synonym for the value.

### 6.2 Procedures

```
PROCEDURE name {
    body
}
```

Defines a reusable procedure. Procedures are inlined at call sites.

### 6.3 Let Bindings

```
LET name = expression;
```

Binds a name to the result of an expression. Implemented via PICK operations.

---

## 7. Convergence Semantics

### 7.1 Formal Definition

Let $F: \mathcal{M} \to \mathcal{M}$ be the state transformation function defined by a program, where $\mathcal{M} = \mathbb{A} \to \mathbb{V}$ is the space of memory states.

A program is *consistent* if there exists $S \in \mathcal{M}$ such that $F(S) = S$.

### 7.2 Convergence Status

| Status | Condition | Interpretation |
|--------|-----------|----------------|
| Consistent | $\exists S: F(S) = S$ | Timeline exists |
| Oscillation | $\exists k > 1: F^k(S_0) = S_0$ | Grandfather paradox |
| Divergence | $\forall n: F^n(S_0) \neq F^m(S_0)$ for $m < n$ | Unbounded growth |
| Timeout | Epoch limit exceeded | Unknown |

### 7.3 Diagnosis

For oscillation, the runtime identifies:
- **Period:** The cycle length $k$
- **Oscillating cells:** Addresses that change within the cycle
- **Negative loops:** Cells where $v' = \neg v$ pattern occurs

---

## 8. Type System

### 8.1 Temporal Types

Ourochronos provides optional static analysis of temporal tainting.

| Type | Meaning |
|------|---------|
| `Pure` | No oracle dependency |
| `Temporal` | Depends on oracle |
| `Unknown` | Not yet determined |

### 8.2 Type Rules

| Operation | Type Rule |
|-----------|-----------|
| Literal | $\text{Pure}$ |
| `ORACLE` | $\text{Pure} \to \text{Temporal}$ |
| Binary | $\text{Temporal} \sqcup T \to \text{Temporal}$ |
| Binary | $\text{Pure} \sqcup \text{Pure} \to \text{Pure}$ |

### 8.3 Type Checking

Run with `--typecheck` to perform temporal type analysis before execution.

---

## 9. SMT Encoding

### 9.1 Logic

Programs are encoded in QF_ABV (Quantifier-Free Arrays and Bit-Vectors).

### 9.2 Variables

| Name | Type | Meaning |
|------|------|---------|
| `anamnesis` | Array[BV16, BV64] | Future memory |
| `present_*` | Array[BV16, BV64] | Constructed memory |

### 9.3 Constraint

The fixed-point constraint is:

```smt
(assert (= present_final anamnesis))
```

If SAT, the model gives a consistent timeline. If UNSAT, no timeline exists.

---

## 10. Operational Semantics

### 10.1 Small-Step Rules

We define the transition relation $\langle S, \sigma, P, A \rangle \to \langle S', \sigma', P', A \rangle$ where $S$ is the remaining program.

**Push:**
$$\frac{}{\langle n :: S, \sigma, P, A \rangle \to \langle S, n :: \sigma, P, A \rangle}$$

**Oracle:**
$$\frac{a = \text{head}(\sigma) \mod 2^{16} \quad v = A(a)}{\langle \text{ORACLE} :: S, a :: \sigma, P, A \rangle \to \langle S, v :: \sigma, P, A \rangle}$$

**Prophecy:**
$$\frac{a = \text{head}(\sigma) \mod 2^{16} \quad v = \text{head}(\text{tail}(\sigma))}{\langle \text{PROPHECY} :: S, a :: v :: \sigma, P, A \rangle \to \langle S, \sigma, P[a \mapsto v], A \rangle}$$

**Add:**
$$\frac{c = (a + b) \mod 2^{64}}{\langle \text{ADD} :: S, b :: a :: \sigma, P, A \rangle \to \langle S, c :: \sigma, P, A \rangle}$$

(Additional rules follow the same pattern.)

### 10.2 Epoch Semantics

$$\text{Epoch}(A) = \text{let } (P, O, \xi) = \text{Run}(\text{Program}, A) \text{ in } (P, O, \xi)$$

$$\text{Execute}(A_0) = \text{let } (P, O, \xi) = \text{Epoch}(A_0) \text{ in } \begin{cases} (P, O) & \text{if } P = A_0 \\ \text{Execute}(P) & \text{otherwise} \end{cases}$$

---

## 11. Syntax Grammar

```ebnf
program     = { declaration } { statement }

declaration = manifest | procedure

manifest    = "MANIFEST" IDENT "=" integer ";"

procedure   = "PROCEDURE" IDENT "{" { statement } "}"

statement   = literal
            | opcode
            | if_stmt
            | while_stmt
            | let_stmt
            | block

literal     = integer | char_lit | string_lit

integer     = DECIMAL | HEX | BINARY

char_lit    = "'" CHAR "'"

string_lit  = '"' { CHAR } '"'

opcode      = "NOP" | "HALT" | "POP" | "DUP" | "SWAP" | "OVER" | "ROT"
            | "DEPTH" | "PICK"
            | "ADD" | "SUB" | "MUL" | "DIV" | "MOD" | "NEG"
            | "NOT" | "AND" | "OR" | "XOR" | "SHL" | "SHR"
            | "EQ" | "NEQ" | "LT" | "GT" | "LTE" | "GTE"
            | "ORACLE" | "PROPHECY" | "PRESENT" | "PARADOX"
            | "INPUT" | "OUTPUT"
            | "PACK" | "UNPACK" | "INDEX" | "STORE"

if_stmt     = "IF" block [ "ELSE" block ]

while_stmt  = "WHILE" block block

let_stmt    = "LET" IDENT "=" { statement } ";"

block       = "{" { statement } "}"
```

---

## 12. Implementation Notes

### 12.1 Stack Underflow

Operations that pop from an empty stack produce an error. The epoch terminates with status Error.

### 12.2 Division by Zero

Division and modulo by zero return 0 (not an error). This ensures total functions.

### 12.3 Address Truncation

All addresses are truncated to 16 bits: $a' = a \mod 2^{16}$.

### 12.4 Shift Amounts

Shift amounts are masked: $n' = n \mod 64$.

---

## Appendix A: Opcode Summary

| Opcode | Hex | Stack Effect |
|--------|-----|--------------|
| NOP | 00 | $(\to)$ |
| HALT | 01 | $(\to)$ |
| POP | 02 | $(a \to)$ |
| DUP | 03 | $(a \to a \, a)$ |
| SWAP | 04 | $(a \, b \to b \, a)$ |
| OVER | 05 | $(a \, b \to a \, b \, a)$ |
| ROT | 06 | $(a \, b \, c \to b \, c \, a)$ |
| DEPTH | 07 | $(\to n)$ |
| PICK | 08 | $(n \to v)$ |
| ADD | 10 | $(a \, b \to a+b)$ |
| SUB | 11 | $(a \, b \to a-b)$ |
| MUL | 12 | $(a \, b \to a \times b)$ |
| DIV | 13 | $(a \, b \to a \div b)$ |
| MOD | 14 | $(a \, b \to a \mod b)$ |
| NEG | 15 | $(a \to -a)$ |
| NOT | 20 | $(a \to \neg a)$ |
| AND | 21 | $(a \, b \to a \land b)$ |
| OR | 22 | $(a \, b \to a \lor b)$ |
| XOR | 23 | $(a \, b \to a \oplus b)$ |
| SHL | 24 | $(a \, n \to a \ll n)$ |
| SHR | 25 | $(a \, n \to a \gg n)$ |
| EQ | 30 | $(a \, b \to a = b)$ |
| NEQ | 31 | $(a \, b \to a \neq b)$ |
| LT | 32 | $(a \, b \to a < b)$ |
| GT | 33 | $(a \, b \to a > b)$ |
| LTE | 34 | $(a \, b \to a \leq b)$ |
| GTE | 35 | $(a \, b \to a \geq b)$ |
| ORACLE | 40 | $(addr \to v)$ |
| PROPHECY | 41 | $(v \, addr \to)$ |
| PRESENT | 42 | $(addr \to v)$ |
| PARADOX | 43 | $(\to)$ |
| INPUT | 50 | $(\to v)$ |
| OUTPUT | 51 | $(v \to)$ |

---

## Appendix B: Error Codes

| Code | Name | Description |
|------|------|-------------|
| E001 | StackUnderflow | Pop from empty stack |
| E002 | InvalidOpcode | Unknown instruction |
| E003 | ParseError | Syntax error |
| E004 | EpochLimit | Max epochs exceeded |
| E005 | InstructionLimit | Max instructions exceeded |

---

*End of Specification*