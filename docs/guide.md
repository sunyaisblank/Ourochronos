# Programming Guide

*Patterns and Practices for Temporal Computation*

> "The best way to predict the future is to invent it."  
> — Alan Kay

## Introduction

This guide teaches you to write Ourochronos programs. We begin with the stack machine fundamentals, progress through temporal operations, and culminate in advanced patterns for solving problems via retrocausation.

The guide assumes no prior experience with stack-based or temporal programming. Each concept builds upon the previous, with working examples throughout.

---

## Part I: The Stack Machine

### 1.1 Thinking in Stacks

Ourochronos uses a *stack-based* execution model. Instead of named variables, values live on a stack. Operations pop their arguments and push their results.

**The Stack Visualised:**

```
Empty:      []
Push 10:    [10]
Push 20:    [10, 20]      ← 20 is on top
ADD:        [30]          ← 10 + 20
Push 5:     [30, 5]
MUL:        [150]         ← 30 × 5
```

**Basic Operations:**

```ourochronos
10 20 ADD     # Push 10, push 20, add → 30
5 MUL         # Push 5, multiply → 150
OUTPUT        # Print 150
```

### 1.2 Stack Manipulation

When values are in the wrong order, use stack manipulation:

| Opcode | Before | After | Purpose |
|--------|--------|-------|---------|
| `DUP` | [a] | [a, a] | Copy top value |
| `SWAP` | [a, b] | [b, a] | Exchange top two |
| `OVER` | [a, b] | [a, b, a] | Copy second to top |
| `ROT` | [a, b, c] | [b, c, a] | Rotate top three |
| `POP` | [a] | [] | Discard top |

**Example: Compute $a^2 + b$**

```ourochronos
# Given stack [a, b], compute a² + b
SWAP          # [b, a]
DUP           # [b, a, a]
MUL           # [b, a²]
ADD           # [a² + b]
```

### 1.3 The PICK Operation

For deeper stack access, use `PICK`. The index 0 is the top, 1 is second, and so on.

```ourochronos
10 20 30 40   # Stack: [10, 20, 30, 40]
2 PICK        # Stack: [10, 20, 30, 40, 20]  (copied index 2)
```

Note: `0 PICK` is equivalent to `DUP`. `1 PICK` is equivalent to `OVER`.

### 1.4 LET Bindings (Syntactic Sugar)

For readability, use LET bindings. They compile to PICK operations.

```ourochronos
LET x = 10;
LET y = 20;
x y ADD OUTPUT    # Outputs 30
```

This is equivalent to:
```ourochronos
10              # Push x
20              # Push y
1 PICK          # Get x (it's now at index 1)
1 PICK          # Get y (it's now at index 1)
ADD OUTPUT
```

---

## Part II: Control Flow

### 2.1 Conditionals

The IF statement pops a condition and branches:

```ourochronos
10 5 GT IF {
    # 10 > 5 is true
    1 OUTPUT
} ELSE {
    0 OUTPUT
}
```

**Truthiness:** Zero is false; any non-zero value is true.

### 2.2 Comparison Operators

| Operator | Meaning |
|----------|---------|
| `EQ` | Equal (=) |
| `NEQ` | Not equal (≠) |
| `LT` | Less than (<) |
| `GT` | Greater than (>) |
| `LTE` | Less or equal (≤) |
| `GTE` | Greater or equal (≥) |

All return 1 for true, 0 for false.

### 2.3 Logical Operators

Combine conditions with bitwise logic:

```ourochronos
# Check if x is between 1 and 10 (exclusive)
LET x = 5;
x 1 GT          # x > 1?
x 10 LT         # x < 10?
AND             # Both true?
IF { ... }
```

### 2.4 While Loops

WHILE takes two blocks: condition and body.

```ourochronos
# Count from 1 to 5
1                           # Counter
WHILE { DUP 5 LTE } {       # While counter ≤ 5
    DUP OUTPUT              # Print counter
    1 ADD                   # Increment
}
POP                         # Clean up
```

**Warning:** Loops that depend on ORACLE values may behave unexpectedly across epochs. See Part IV.

---

## Part III: Temporal Operations

### 3.1 The Core Insight

Ourochronos has two memory spaces:

- **Anamnesis (A):** Read-only. Contains the state from the "future" (the result of the previous epoch, or the final consistent state).
- **Present (P):** Read-write. The state you are constructing, which will become the next epoch's Anamnesis.

A program converges when Present equals Anamnesis: the future you expected matches the future you created.

### 3.2 ORACLE: Reading the Future

`ORACLE` reads from Anamnesis:

```ourochronos
0 ORACLE    # Push A[0] onto stack
```

The value you receive is what you (or a previous epoch) wrote to address 0. In the first epoch, all Anamnesis values are zero (or the seed value).

### 3.3 PROPHECY: Writing to the Past

`PROPHECY` writes to Present:

```ourochronos
42 0 PROPHECY    # Set P[0] = 42
```

After the epoch ends, P[0] becomes A[0] for the next epoch. If P equals A, you've created a consistent timeline.

### 3.4 The Simplest Temporal Program

```ourochronos
0 ORACLE 0 PROPHECY
```

**What happens:**

1. **Epoch 1:** A[0] = 0 (initial). Read 0. Write 0 to P[0]. P[0] = 0 = A[0]. Consistent!

The program converges immediately because it writes back what it reads.

### 3.5 The Grandfather Paradox

```ourochronos
0 ORACLE NOT 0 PROPHECY
```

**What happens:**

1. **Epoch 1:** A[0] = 0. Read 0. NOT gives all-ones (~0). Write ~0 to P[0]. P ≠ A.
2. **Epoch 2:** A[0] = ~0. Read ~0. NOT gives 0. Write 0 to P[0]. P ≠ A.
3. **Epoch 3:** A[0] = 0. (Same as Epoch 1.)

The program oscillates forever. This is the grandfather paradox: no consistent value exists.

### 3.6 PRESENT: Reading Current State

`PRESENT` reads from the Present memory (what you've written this epoch):

```ourochronos
42 0 PROPHECY    # P[0] = 42
0 PRESENT        # Push P[0] (which is 42)
OUTPUT           # Prints 42
```

Unlike ORACLE, PRESENT reads from the current epoch's state, not the future.

---

## Part IV: Temporal Patterns

### 4.1 Pattern: Self-Fulfilling Prophecy

Read a value, verify it, write it back if correct.

```ourochronos
0 ORACLE              # Read candidate
DUP 42 EQ IF {        # Is it 42?
    0 PROPHECY        # Yes: stabilise
    42 OUTPUT         # Celebrate
} ELSE {
    42 0 PROPHECY     # No: make it 42
}
```

**Convergence:** The first epoch reads 0 (wrong), writes 42. The second epoch reads 42 (correct), writes 42. Consistent after 2 epochs.

### 4.2 Pattern: Witness Verification

For search problems, ask the future for a solution, verify it, and stabilise if correct.

**Example: Find a factor of 15**

```ourochronos
0 ORACLE                    # Get candidate factor
DUP 1 GT IF {               # Factor > 1?
    DUP 15 LT IF {          # Factor < 15?
        DUP 15 SWAP MOD 0 EQ IF {   # 15 mod factor = 0?
            # Valid factor found!
            DUP 0 PROPHECY  # Stabilise
            OUTPUT          # Report
        } ELSE {
            1 ADD 0 PROPHECY    # Not a factor, try next
        }
    } ELSE {
        2 0 PROPHECY        # Too large, reset
    }
} ELSE {
    2 0 PROPHECY            # Too small, start at 2
}
```

**What happens:**

1. Epoch 1: Read 0. Fails (≤ 1). Write 2.
2. Epoch 2: Read 2. Valid range. 15 mod 2 = 1 ≠ 0. Write 3.
3. Epoch 3: Read 3. Valid range. 15 mod 3 = 0. Stabilise at 3. Output 3.

The program "finds" 3 without searching: the only consistent timeline is the one where the oracle provides a correct factor.

### 4.3 Pattern: Bootstrap (Information from Nowhere)

Create a self-consistent information loop with no external source.

```ourochronos
# Bootstrap the string "HI" (ASCII 72, 73)
0 ORACLE 1 ORACLE         # Read two characters

# Verify
1 PICK 72 EQ              # First char is H?
1 PICK 73 EQ              # Second char is I?
AND

IF {
    # Correct! Stabilise and output
    1 PROPHECY 0 PROPHECY
    0 PRESENT OUTPUT
    1 PRESENT OUTPUT
} ELSE {
    # Wrong. Seed the correct values.
    72 0 PROPHECY
    73 1 PROPHECY
}
```

**Result:** "HI" appears, but it was never entered as input. It exists because its existence is self-consistent.

### 4.4 Pattern: Conditional Paradox

Sometimes a program has consistent timelines for some inputs but not others.

```ourochronos
INPUT                     # External input
0 EQ IF {
    # Input is 0: simple identity
    0 ORACLE 0 PROPHECY
} ELSE {
    # Input is non-zero: grandfather paradox
    0 ORACLE NOT 0 PROPHECY
}
```

With input 0, the program converges. With any other input, it oscillates.

### 4.5 Pattern: Cascading Dependencies

Multiple memory cells can form dependency chains.

```ourochronos
# A[0] determines A[1], which determines A[2]
0 ORACLE 1 ADD 1 PROPHECY   # P[1] = A[0] + 1
1 ORACLE 1 ADD 2 PROPHECY   # P[2] = A[1] + 1
2 ORACLE 0 PROPHECY         # P[0] = A[2]
```

**Analysis:** This creates a cycle: P[0] depends on A[2], which depends on A[1], which depends on A[0]. What fixed point exists?

Let $a = A[0]$. Then:
- $P[1] = a + 1$
- $P[2] = (a+1) + 1 = a + 2$
- $P[0] = a + 2$

For consistency, $P[0] = A[0]$, so $a + 2 = a$. No solution exists (in finite arithmetic, this oscillates or diverges).

---

## Part V: Procedures and Modules

### 5.1 Defining Procedures

Procedures encapsulate reusable logic:

```ourochronos
PROCEDURE square {
    DUP MUL
}

PROCEDURE cube {
    DUP DUP MUL MUL
}

5 square OUTPUT     # 25
3 cube OUTPUT       # 27
```

Procedures are inlined at call sites. They have no separate stack frame.

### 5.2 Procedures with Multiple Arguments

Procedures consume arguments from the stack and leave results:

```ourochronos
PROCEDURE max {
    # Stack: [a, b]
    OVER OVER         # [a, b, a, b]
    LT IF {           # a < b?
        SWAP POP      # Keep b
    } ELSE {
        POP           # Keep a
    }
}

10 20 max OUTPUT    # 20
30 5 max OUTPUT     # 30
```

### 5.3 Manifest Constants

Define named constants for clarity:

```ourochronos
MANIFEST TARGET = 15;
MANIFEST MIN_FACTOR = 2;

0 ORACLE
DUP MIN_FACTOR GTE IF {
    DUP TARGET LT IF {
        TARGET OVER MOD 0 EQ IF {
            DUP 0 PROPHECY OUTPUT
        } ELSE {
            1 ADD 0 PROPHECY
        }
    } ELSE { MIN_FACTOR 0 PROPHECY }
} ELSE { MIN_FACTOR 0 PROPHECY }
```

---

## Part VI: Debugging Temporal Programs

### 6.1 Diagnostic Mode

Run with `--diagnostic` to see epoch-by-epoch execution:

```bash
ourochronos program.ouro --diagnostic
```

This shows:
- Each epoch's Anamnesis state
- Operations performed
- Present state after execution
- Convergence status

### 6.2 Common Problems

**Problem: Program oscillates unexpectedly**

*Cause:* A negation or increment in the causal loop.

*Solution:* Ensure the feedback path preserves values when verification succeeds.

**Problem: Program converges to all zeros**

*Cause:* The trivial fixed point is being selected.

*Solution:* Use `--action` mode to prefer non-trivial solutions, or add verification that rejects zero.

**Problem: Program times out**

*Cause:* Very slow convergence or divergence not detected.

*Solution:* Check for monotonically increasing values. Add bounds checks.

### 6.3 Type Checking

Run with `--typecheck` to analyse temporal tainting:

```bash
ourochronos program.ouro --typecheck
```

This reports:
- Which values depend on ORACLE reads (temporal)
- Which values are pure constants
- Warnings about temporal conditions in loops

### 6.4 SMT Verification

For complex programs, compile to SMT and check satisfiability:

```bash
ourochronos program.ouro --smt > constraints.smt2
z3 constraints.smt2
```

If Z3 reports SAT, a consistent timeline exists (and the model shows it). If UNSAT, no timeline exists—the program is a paradox.

---

## Part VII: Advanced Topics

### 7.1 The Action Principle

When multiple fixed points exist, use `--action` to select the most "interesting" one:

```bash
ourochronos program.ouro --action --seeds 8
```

The action principle penalises:
- Zero values (trivial)
- Pure values (no temporal dependency)
- Values unchanged from seed

And rewards:
- Non-zero values
- Deep causal chains
- Output production

### 7.2 Temporal Core Analysis

Not all memory participates in temporal feedback. The *temporal core* is the subset of addresses that form causal loops.

For optimisation, the runtime can:
1. Identify the temporal core via static analysis
2. Restrict fixed-point iteration to core addresses
3. Handle non-core addresses with single-pass execution

### 7.3 Stratification

If temporal dependencies are acyclic (each address depends only on lower-numbered addresses), the program can be solved in a single pass:

```ourochronos
# Stratified: each cell depends only on earlier cells
0 ORACLE 1 ADD 1 PROPHECY   # P[1] = A[0] + 1
1 PRESENT 2 MUL 2 PROPHECY  # P[2] = P[1] * 2
```

This converges in one epoch because there's no feedback cycle.

### 7.4 Parallel Exploration

In `--action` mode, multiple seeds can be explored in parallel. Each seed may lead to a different fixed point (or paradox). The best result is selected.

This is the computational analogue of the many-worlds interpretation: explore all timelines, keep the consistent ones.

---

## Part VIII: Worked Examples

### 8.1 Sorting via Time Travel

Classical sorting is $O(n \log n)$. With temporal verification, we can "guess" the sorted order and verify in $O(n)$.

```ourochronos
# Sort [4, 1, 3, 2] stored at addresses 10-13
MANIFEST ADDR = 10;
MANIFEST LEN = 4;

# Read candidate sorted values from future
ADDR ORACLE
ADDR 1 ADD ORACLE
ADDR 2 ADD ORACLE
ADDR 3 ADD ORACLE
# Stack: [v0, v1, v2, v3]

# Verify: is it sorted?
3 PICK 3 PICK LTE    # v0 ≤ v1?
3 PICK 2 PICK LTE    # v1 ≤ v2?
AND
2 PICK 1 PICK LTE    # v2 ≤ v3?
AND

# Verify: is it a permutation? (check sum = 1+2+3+4 = 10)
4 PICK 4 PICK ADD 4 PICK ADD 4 PICK ADD
10 EQ AND

IF {
    # Valid sorted permutation! Stabilise.
    ADDR 3 ADD PROPHECY
    ADDR 2 ADD PROPHECY
    ADDR 1 ADD PROPHECY
    ADDR PROPHECY
    
    # Output
    ADDR PRESENT OUTPUT
    ADDR 1 ADD PRESENT OUTPUT
    ADDR 2 ADD PRESENT OUTPUT
    ADDR 3 ADD PRESENT OUTPUT
} ELSE {
    # Invalid. Seed with [1, 2, 3, 4].
    1 ADDR PROPHECY
    2 ADDR 1 ADD PROPHECY
    3 ADDR 2 ADD PROPHECY
    4 ADDR 3 ADD PROPHECY
}
```

**Result:** Outputs `1 2 3 4`. The sorted order emerges from consistency, not comparison.

### 8.2 SAT Solving via Temporal Witness

A boolean formula can be solved by asking the future for a satisfying assignment:

```ourochronos
# Solve: (x OR y) AND (NOT x OR z) AND (NOT y OR NOT z)
# Variables: x at 0, y at 1, z at 2

0 ORACLE    # x
1 ORACLE    # y  
2 ORACLE    # z

# Clause 1: x OR y
2 PICK 2 PICK OR

# Clause 2: NOT x OR z
3 PICK NOT 2 PICK OR AND

# Clause 3: NOT y OR NOT z
2 PICK NOT 1 PICK NOT OR AND

IF {
    # SAT! Stabilise this assignment.
    2 PROPHECY 1 PROPHECY 0 PROPHECY
    
    # Output solution
    0 PRESENT OUTPUT
    1 PRESENT OUTPUT
    2 PRESENT OUTPUT
} ELSE {
    # UNSAT with this assignment. Perturb.
    # Simple strategy: increment x as a counter
    0 ORACLE 1 ADD 0 PROPHECY
}
```

**Note:** This example demonstrates the concept. A robust SAT solver would need systematic exploration or use `--action` mode.

### 8.3 Game-Playing Agent

A temporal agent can "know" the winning move by receiving it from the future:

```ourochronos
# Tic-tac-toe: find winning move for X
# Board at addresses 0-8, move at address 9

# Read board and candidate move from future
# ... (read all 10 values)

# Verify: is this move valid and winning?
# ... (check move is on empty square, results in three-in-a-row)

IF {
    # Winning move found! Stabilise.
    # ... (write back the position and move)
} ELSE {
    # Not winning. Try next square.
    9 ORACLE 1 ADD 9 MOD 9 PROPHECY
}
```

This finds a winning move (if one exists) without game-tree search.

---

## Summary

Ourochronos programs follow a consistent pattern:

1. **Read** candidate values from the future (ORACLE)
2. **Verify** that they satisfy the desired property
3. **Stabilise** by writing them back if valid (PROPHECY)
4. **Perturb** by writing different values if invalid

The runtime iterates until a consistent state is found—or diagnoses that none exists.

This inverts the traditional programming model. Instead of *constructing* solutions step by step, you *constrain* what solutions are acceptable and let temporal consistency do the rest.

The future computes the past. The solution exists because it must exist.

---

## Quick Reference

**Temporal Operations:**
```
n ORACLE      Read A[n] (from future)
v n PROPHECY  Write v to P[n] (to past)
n PRESENT     Read P[n] (current epoch)
PARADOX       Abort as paradoxical
```

**Common Patterns:**
```
# Identity (any value works)
0 ORACLE 0 PROPHECY

# Verification pattern
0 ORACLE DUP valid? IF { 0 PROPHECY } ELSE { perturb 0 PROPHECY }

# Bootstrap pattern
values ORACLE... verify IF { PROPHECY... } ELSE { seed PROPHECY... }
```

**Command Line:**
```bash
ourochronos file.ouro              # Run normally
ourochronos file.ouro --diagnostic # Show epochs
ourochronos file.ouro --action     # Prefer non-trivial solutions
ourochronos file.ouro --typecheck  # Static analysis
ourochronos file.ouro --smt        # Generate SMT constraints
```