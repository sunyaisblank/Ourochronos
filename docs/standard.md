# Ourochronos Coding Standard

*Governance for Causal Integrity*

**Version:** 2.0  
**Status:** Normative

> "With great power comes great responsibility."  
> — Uncle Ben

## Preface

Programming in Ourochronos is not ordinary software development. A bug in conventional code causes incorrect output. A bug in Ourochronos can destabilise a timeline, create undetectable paradoxes, or cause silent divergence.

This standard exists to prevent temporal catastrophes. It codifies hard-won lessons about causal integrity, temporal stability, and maintainable time-travel code.

The rules are organised by severity:

| Level | Meaning | Violation Consequence |
|-------|---------|----------------------|
| **MUST** | Mandatory | Program is incorrect |
| **SHOULD** | Strongly recommended | Program may misbehave |
| **MAY** | Optional best practice | Reduced maintainability |

---

## Part I: Causal Integrity

These rules prevent paradoxes and ensure temporal consistency.

### Rule 1: Bounded Causality

**Level:** MUST

Every causal loop MUST have a verifiable convergence condition.

**Rationale:** Unbounded loops cause infinite epoch iteration, consuming unbounded resources without producing results.

**Violation Example:**
```ourochronos
# VIOLATION: No fixed point exists
0 ORACLE 1 ADD 0 PROPHECY
```

This program reads $x$ and writes $x + 1$. The equation $x = x + 1$ has no solution.

**Compliant Example:**
```ourochronos
# COMPLIANT: Bounded increment with ceiling
0 ORACLE DUP 100 LT IF {
    1 ADD 0 PROPHECY
} ELSE {
    0 PROPHECY    # Stabilise at 100
}
```

### Rule 2: Grandfather Safety

**Level:** MUST

Programs MUST NOT negate a value in a direct feedback loop without an escape path.

**Rationale:** Negation loops oscillate forever (the grandfather paradox).

**Violation Example:**
```ourochronos
# VIOLATION: Direct negation loop
0 ORACLE NOT 0 PROPHECY
```

**Compliant Example:**
```ourochronos
# COMPLIANT: Negation with escape condition
0 ORACLE DUP 0 EQ IF {
    1 0 PROPHECY      # Escape: if 0, write 1
} ELSE {
    DUP 1 EQ IF {
        1 0 PROPHECY  # Stabilise at 1
    } ELSE {
        NOT 0 PROPHECY
    }
}
```

### Rule 3: No Naked Paradoxes

**Level:** MUST

The `PARADOX` instruction MUST only appear within a conditional block that detects an invalid state.

**Rationale:** Unconditional PARADOX always fails. Conditional PARADOX documents the invariant being protected.

**Violation Example:**
```ourochronos
# VIOLATION: Unconditional paradox
PARADOX
```

**Compliant Example:**
```ourochronos
# COMPLIANT: Paradox guards an invariant
0 ORACLE DUP 0 LT IF {
    # Negative values are invalid in this domain
    PARADOX
} ELSE {
    0 PROPHECY
}
```

### Rule 4: Divergence Guards

**Level:** SHOULD

Loops that increment values based on Anamnesis SHOULD include an upper bound check.

**Rationale:** Without bounds, values may grow until overflow, causing difficult-to-diagnose wraparound behaviour.

**Weak Example:**
```ourochronos
# WEAK: Unbounded growth until wraparound
0 ORACLE 1 ADD 0 PROPHECY
```

**Stronger Example:**
```ourochronos
# STRONGER: Explicit bound
MANIFEST MAX_VALUE = 1000;

0 ORACLE DUP MAX_VALUE LT IF {
    1 ADD 0 PROPHECY
} ELSE {
    PARADOX    # Or handle gracefully
}
```

---

## Part II: Temporal Hygiene

These rules ensure correct interaction between epochs.

### Rule 5: Oracle-Prophecy Correspondence

**Level:** MUST

Every ORACLE read that influences program behaviour MUST correspond to a PROPHECY write to the same address in at least one code path.

**Rationale:** Reading an address without writing to it creates implicit dependencies on uninitialised (zero) values.

**Violation Example:**
```ourochronos
# VIOLATION: Reads address 0, never writes to it
0 ORACLE OUTPUT
```

**Compliant Example:**
```ourochronos
# COMPLIANT: Every read has a corresponding write
0 ORACLE DUP OUTPUT 0 PROPHECY
```

### Rule 6: Causal Documentation

**Level:** SHOULD

Every PROPHECY instruction SHOULD be preceded by a comment explaining the causal relationship.

**Rationale:** Temporal code is inherently confusing. Documentation clarifies intent.

**Undocumented:**
```ourochronos
0 ORACLE 1 ADD 0 PROPHECY
```

**Documented:**
```ourochronos
# Causal Loop: A[0] → P[0]
# Seeking: x such that x = x + 1 (will diverge)
0 ORACLE 1 ADD 0 PROPHECY
```

### Rule 7: Epoch Purity

**Level:** SHOULD

Side effects (OUTPUT, INPUT) SHOULD occur only after the fixed-point condition has been verified.

**Rationale:** Output during non-convergent epochs produces confusing intermediate results.

**Impure:**
```ourochronos
# IMPURE: Outputs during iteration
0 ORACLE DUP OUTPUT     # May output wrong values
DUP 42 EQ IF {
    0 PROPHECY
} ELSE {
    1 ADD 0 PROPHECY
}
```

**Pure:**
```ourochronos
# PURE: Output only after stabilisation
0 ORACLE DUP 42 EQ IF {
    0 PROPHECY
    42 OUTPUT           # Only reached when consistent
} ELSE {
    1 ADD 0 PROPHECY
}
```

---

## Part III: Structural Clarity

These rules improve code readability and maintainability.

### Rule 8: Stack Discipline

**Level:** SHOULD

Every block `{ ... }` SHOULD have a documented stack effect, and the effect SHOULD be net-zero unless the block is explicitly a producer or consumer.

**Rationale:** Unbalanced stacks cause subtle bugs and make code hard to compose.

**Undisciplined:**
```ourochronos
# What's the stack effect? Who knows.
{
    10 20 ADD
    DUP 5 LT IF { 1 } ELSE { 2 3 }
}
```

**Disciplined:**
```ourochronos
# Stack effect: ( -- result )
# Produces exactly one value
{
    10 20 ADD               # ( -- 30 )
    DUP 5 LT IF {           # ( 30 -- 30 )
        POP 1               # ( 30 -- 1 )
    } ELSE {
        POP 2               # ( 30 -- 2 )
    }
}                           # ( -- result )
```

### Rule 9: Manifest Constants

**Level:** SHOULD

Magic numbers SHOULD be defined as MANIFEST constants.

**Rationale:** Named constants document intent and enable single-point changes.

**Magic Numbers:**
```ourochronos
0 ORACLE DUP 15 SWAP MOD 0 EQ IF { ... }
```

**Named Constants:**
```ourochronos
MANIFEST TARGET = 15;
MANIFEST ADDR_FACTOR = 0;

ADDR_FACTOR ORACLE DUP TARGET SWAP MOD 0 EQ IF { ... }
```

### Rule 10: Procedure Extraction

**Level:** MAY

Repeated code patterns MAY be extracted into procedures.

**Rationale:** Procedures improve readability and reduce duplication.

**Inline:**
```ourochronos
# Check if value at address 0 is valid factor of 15
0 ORACLE DUP 1 GT SWAP DUP 15 LT SWAP 15 SWAP MOD 0 EQ AND AND

# Check if value at address 1 is valid factor of 15
1 ORACLE DUP 1 GT SWAP DUP 15 LT SWAP 15 SWAP MOD 0 EQ AND AND
```

**Extracted:**
```ourochronos
MANIFEST TARGET = 15;

PROCEDURE is_valid_factor {
    # ( n -- bool )
    DUP 1 GT 
    OVER TARGET LT AND
    SWAP TARGET SWAP MOD 0 EQ AND
}

0 ORACLE is_valid_factor
1 ORACLE is_valid_factor
```

---

## Part IV: Memory Layout

These rules govern the use of the 65536-cell address space.

### Rule 11: Address Allocation

**Level:** SHOULD

Programs SHOULD document their memory layout using MANIFEST constants.

**Rationale:** Ad-hoc address usage leads to collisions and confusion.

**Undocumented:**
```ourochronos
0 ORACLE ...
5 ORACLE ...
10 ORACLE ...
```

**Documented:**
```ourochronos
# Memory Layout
# 0-3:   Working variables
# 10-19: Input array
# 20-29: Output array
# 100+:  Scratch space

MANIFEST WORK_BASE = 0;
MANIFEST INPUT_BASE = 10;
MANIFEST OUTPUT_BASE = 20;
MANIFEST SCRATCH_BASE = 100;
```

### Rule 12: Address Ranges

**Level:** SHOULD

Related data SHOULD be stored in contiguous address ranges.

**Rationale:** Contiguous storage enables use of PACK/UNPACK and indexed access.

**Scattered:**
```ourochronos
# Array elements at random addresses
0 PROPHECY    # Element 0
5 PROPHECY    # Element 1
23 PROPHECY   # Element 2
```

**Contiguous:**
```ourochronos
MANIFEST ARRAY_BASE = 10;
MANIFEST ARRAY_LEN = 3;

# Elements at 10, 11, 12
ARRAY_BASE PROPHECY
ARRAY_BASE 1 ADD PROPHECY
ARRAY_BASE 2 ADD PROPHECY
```

### Rule 13: Temporal Core Isolation

**Level:** MAY

Addresses involved in temporal feedback MAY be separated from pure storage.

**Rationale:** Isolating the temporal core simplifies analysis and debugging.

```ourochronos
# Temporal Core: addresses 0-9
# These participate in ORACLE/PROPHECY loops

# Pure Storage: addresses 1000+
# These are written once and read via PRESENT

MANIFEST TEMPORAL_BASE = 0;
MANIFEST TEMPORAL_SIZE = 10;
MANIFEST PURE_BASE = 1000;
```

---

## Part V: Error Handling

### Rule 14: Graceful Degradation

**Level:** SHOULD

Programs SHOULD handle unexpected oracle values gracefully rather than entering undefined states.

**Fragile:**
```ourochronos
# Assumes oracle value is always 0, 1, or 2
0 ORACLE
DUP 0 EQ IF { ... } ELSE {
    DUP 1 EQ IF { ... } ELSE {
        # Assumes this is 2, but what if it's 3?
        ...
    }
}
```

**Robust:**
```ourochronos
0 ORACLE
DUP 0 EQ IF { ... } ELSE {
    DUP 1 EQ IF { ... } ELSE {
        DUP 2 EQ IF { ... } ELSE {
            # Unexpected value: reset to known state
            POP 0 0 PROPHECY
        }
    }
}
```

### Rule 15: Paradox Documentation

**Level:** MUST

When PARADOX is used, the triggering condition MUST be documented.

**Rationale:** Future maintainers need to understand why certain states are impossible.

```ourochronos
# INVARIANT: Factor must be in range (1, TARGET)
# PARADOX: Triggered when oracle provides out-of-range value
#          and no valid correction is possible

0 ORACLE DUP 1 LTE IF {
    PARADOX    # Cannot correct: minimum valid value is 2
}
```

---

## Part VI: Testing

### Rule 16: Convergence Testing

**Level:** SHOULD

Programs SHOULD be tested for convergence with multiple initial seeds.

**Rationale:** Different seeds may reveal oscillation or divergence not apparent with seed 0.

```bash
# Test with default seed
ourochronos program.ouro

# Test with alternative seeds
ourochronos program.ouro --seed 1
ourochronos program.ouro --seed 42
ourochronos program.ouro --seed 12345
```

### Rule 17: Boundary Testing

**Level:** SHOULD

Temporal programs SHOULD be tested at boundary conditions.

**Boundary conditions to test:**
- Oracle returns 0 (initial state)
- Oracle returns maximum value (2^64 - 1)
- Oracle returns the exact target value (immediate convergence)
- Oracle returns one-off values (convergence after perturbation)

### Rule 18: SMT Verification

**Level:** MAY

Complex temporal logic MAY be verified using SMT solvers.

```bash
# Generate constraints
ourochronos program.ouro --smt > program.smt2

# Check satisfiability
z3 program.smt2

# If SAT: program has consistent timeline
# If UNSAT: program is a paradox
```

---

## Part VII: Performance

### Rule 19: Minimize Temporal State

**Level:** SHOULD

Programs SHOULD minimise the number of addresses involved in temporal feedback.

**Rationale:** Fixed-point search complexity grows with temporal core size.

**Wasteful:**
```ourochronos
# Uses 100 addresses for a single value
0 ORACLE 0 PROPHECY
1 ORACLE 1 PROPHECY
...
99 ORACLE 99 PROPHECY
```

**Efficient:**
```ourochronos
# Uses only necessary addresses
0 ORACLE DUP 0 PROPHECY    # Core computation
# Other data in non-temporal storage
```

### Rule 20: Stratify When Possible

**Level:** MAY

When addresses have acyclic dependencies, programs MAY be structured for single-pass convergence.

**Cyclic (requires iteration):**
```ourochronos
0 ORACLE 1 PROPHECY   # P[1] depends on A[0]
1 ORACLE 0 PROPHECY   # P[0] depends on A[1] — cycle!
```

**Acyclic (single-pass):**
```ourochronos
0 ORACLE 1 PROPHECY   # P[1] depends on A[0]
1 PRESENT 2 PROPHECY  # P[2] depends on P[1], not A[1]
```

---

## Appendix A: Checklist

Before committing temporal code, verify:

- [ ] Every causal loop has a bounded convergence condition
- [ ] No direct negation loops exist without escape paths
- [ ] PARADOX is used only within guarded conditions
- [ ] Every ORACLE has a corresponding PROPHECY
- [ ] Output occurs only after stabilisation
- [ ] Stack effects are balanced and documented
- [ ] Magic numbers are replaced with MANIFEST constants
- [ ] Memory layout is documented
- [ ] Program converges with multiple seeds
- [ ] Boundary conditions are tested

---

## Appendix B: Anti-Patterns

### Anti-Pattern 1: The Lazy Genie

```ourochronos
# Wants any factor of 15
0 ORACLE 0 PROPHECY    # Just accepts whatever the oracle says!
```

**Problem:** Converges immediately to 0, which is not a factor.

**Fix:** Add verification:
```ourochronos
0 ORACLE DUP valid_factor? IF { 0 PROPHECY } ELSE { ... }
```

### Anti-Pattern 2: The Infinite Optimist

```ourochronos
# Keeps trying forever
0 ORACLE 1 ADD 0 PROPHECY
```

**Problem:** Diverges (or wraps around after 2^64 epochs).

**Fix:** Add bounds and fallback:
```ourochronos
0 ORACLE DUP 1000 LT IF { 1 ADD 0 PROPHECY } ELSE { PARADOX }
```

### Anti-Pattern 3: The Forgetful Prophet

```ourochronos
# Reads from 0, writes to 1
0 ORACLE 1 PROPHECY
```

**Problem:** Address 0 is never written, so it stays at initial value. Address 1 is written but never read.

**Fix:** Match reads and writes:
```ourochronos
0 ORACLE DUP 0 PROPHECY    # Read from 0, write to 0
```

### Anti-Pattern 4: The Noisy Debugger

```ourochronos
# Outputs on every epoch
0 ORACLE DUP OUTPUT 0 PROPHECY
```

**Problem:** Produces output during non-convergent epochs, cluttering results.

**Fix:** Output only when stable:
```ourochronos
0 ORACLE DUP 0 PROPHECY
# ... verification ...
IF { 0 PRESENT OUTPUT }    # Only output when consistent
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Anamnesis** | Read-only memory containing the future state |
| **Present** | Read-write memory being constructed |
| **Epoch** | One execution of the program |
| **Convergence** | State where Present = Anamnesis |
| **Oscillation** | Cyclic non-convergence (grandfather paradox) |
| **Divergence** | Monotonic non-convergence |
| **Temporal Core** | Addresses involved in feedback loops |
| **Provenance** | Record of which oracle reads influenced a value |
| **Fixed Point** | State $S$ where $F(S) = S$ |

---

*End of Standard*