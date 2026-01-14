# The Type System of Ourochronos: A Deep Analysis

> "The limits of my language mean the limits of my world."
> – Ludwig Wittgenstein

---

## Introduction: Types as Ontology

A type system constitutes a statement about what *kinds of things* can exist in a computational universe; it extends beyond its role as a technical artefact. When we define the types of a language, we define its ontology: what entities are permitted, how they may interact, and what transformations are possible.

Ourochronos, in its current incarnation, embraces radical simplicity. Its type system admits essentially one kind of entity: the 64-bit unsigned integer. This document examines what this choice means, what it enables, and what it forecloses.

---

## Part I: The Atomic Type

### The Value Structure

At the foundation lies a deceptively simple structure:

```rust
pub struct Value {
    pub val: u64,
    pub prov: Provenance,
}
```

This structure represents an integer *with memory*: a number that knows where it came from. The separation of `val` (the numeric content) from `prov` (the causal history) represents a fundamental design insight: in temporal computation, *what* a value is matters less than *how* it came to be.

### The Number Space

The choice of `u64` defines a finite, discrete number space:

$$\mathbb{U}_{64} = \{0, 1, 2, \ldots, 2^{64}-1\}$$

This space has 18,446,744,073,709,551,616 elements, vast by human standards yet finite. Every computation maps from this space to itself:

$$f : \mathbb{U}_{64}^n \to \mathbb{U}_{64}^m$$

The finiteness has profound implications for fixed-point computation. In an infinite space, fixed-point iteration might never terminate. In $\mathbb{U}_{64}$, every function on a finite stack must eventually cycle, though the cycle might take longer than the age of the universe to manifest.

### Wrapping Semantics

Arithmetic in Ourochronos wraps on overflow:

```
18446744073709551615 + 1 = 0
0 - 1 = 18446744073709551615
```

This represents the natural topology of modular arithmetic rather than overflow "handling." The number space forms a ring:

$$(\mathbb{U}_{64}, +, \times)$$

Where addition and multiplication are performed modulo $2^{64}$.

The ring structure means:
- Addition and multiplication are closed (results stay in $\mathbb{U}_{64}$)
- Additive identity is 0
- Multiplicative identity is 1
- Every element has an additive inverse

This algebraic structure is well-behaved and predictable, yet it differs from the integers $\mathbb{Z}$ in crucial ways. Most notably, the ordering is not preserved under arithmetic: $a > b$ does not imply $a + c > b + c$ when wrapping occurs.

---

## Part II: Provenance

### The Five Sources

Every value carries knowledge of its origin:

```rust
pub enum ProvenanceSource {
    Literal,        // From source code
    Computation,    // From arithmetic/logic
    Oracle,         // From the future
    Memory,         // From memory read
    Unknown,        // Origin lost
}
```

This taxonomy reflects the different modes by which knowledge enters computation.

**Literal:** The most primitive source. A literal value exists because the programmer wrote it. It has no causal dependencies; it simply *is*.

```ouro
42  # Provenance: Literal
```

**Computation:** Values born from operations on other values. The provenance of a computed value depends on its inputs:

```ouro
10 20 ADD  # Provenance: Computation (derived from Literal + Literal)
```

**Oracle:** The distinctive marker of temporal computation. An Oracle value carries information from outside the present moment, specifically from a future or past iteration of a temporal loop.

```ouro
ORACLE 0  # Provenance: Oracle (epoch N+k for some k)
```

**Memory:** Values retrieved from storage. Memory provenance indicates interaction with state:

```ouro
100 FETCH  # Provenance: Memory
```

**Unknown:** The catch-all for values whose history has been lost. This can occur through certain operations or when provenance tracking fails.

### Provenance Propagation

How does provenance combine? The current system uses a lattice:

```
      Unknown
       /   \
   Oracle   Memory
      |    / |
  Computation
      |
   Literal
```

When values combine, their provenances join in this lattice. The key rule: **Oracle taints everything it touches.** Once a value depends on Oracle information, that dependency is preserved.

```ouro
42            # Literal
ORACLE 0      # Oracle
ADD           # Result: Oracle (tainted by temporal dependency)
```

### The Epoch Tag

```rust
pub struct Provenance {
    pub source: ProvenanceSource,
    pub epoch: Option<usize>,
}
```

The optional `epoch` field tags values with their temporal origin. A value from epoch 3 carries `epoch: Some(3)`. This information could support epoch-scoped type checking; it is currently recorded without enforcement.

---

## Part III: What the Type System Lacks

### Linear Types

**Definition:** In a linear type system, certain values must be used exactly once. They cannot be duplicated, and they cannot be discarded.

**Formal Statement:**
$$\frac{\Gamma, x : A \vdash e : B}{\Gamma \vdash \lambda x.e : A \multimap B}$$

The linear arrow $\multimap$ indicates that the argument must be consumed exactly once.

**Why Ourochronos Needs Them:**

Consider a message from the future:

```ouro
ORACLE 0  # Receive temporal message
DUP       # Duplicate it
```

If this message represents a quantum of temporal information (for instance, the result of a measurement that can only occur once), duplicating it violates the semantics of what the message represents.

Linear types would enforce:

```ouro
ORACLE 0       # Returns linear value
# DUP is a type error on linear values
CONSUME msg    # Explicitly use the value
```

**Implementation Sketch:**

```rust
pub enum Linearity {
    Unrestricted,  // Can be used any number of times
    Linear,        // Must be used exactly once
    Affine,        // Can be used at most once
}

pub struct Value {
    pub val: u64,
    pub prov: Provenance,
    pub linearity: Linearity,
}
```

With compiler enforcement:
- Linear values cannot be `DUP`licated
- Linear values must be used before scope exit
- Functions consuming linear values must be marked accordingly

### Refinement Types

**Definition:** Refinement types attach predicates to base types, restricting the set of valid values.

**Formal Statement:**
$$\{x : \tau \mid \phi(x)\}$$

The type $\{x : \text{u64} \mid x > 0\}$ contains only positive integers.

**Why Ourochronos Needs Them:**

Consider address validation:

```ouro
100000 FETCH  # Address out of bounds!
```

With refinement types:

```ouro
PROC fetch ( addr : { x : u16 | x < MEMORY_SIZE } -- val ) {
    # Address is guaranteed valid
    FETCH
}
```

The type system would reject programs that might produce invalid addresses, catching errors at compile time.

**Implementation Sketch:**

Refinement types require:
1. A predicate language for expressing constraints
2. SMT solver integration for checking satisfiability
3. Dependent type features for propagating refinements

```ouro
TYPE ValidAddr = { x : u16 | x < 65536 };
TYPE Positive = { x : u64 | x > 0 };
TYPE Bounded(lo, hi) = { x : u64 | lo <= x AND x < hi };
```

### Uniqueness Types

**Definition:** Uniqueness types guarantee that a reference is the sole reference to its target, enabling safe in-place mutation.

**Formal Statement:**
A unique reference $r : \text{unique}\ A$ guarantees that no other reference to the same memory location exists.

**Why Ourochronos Needs Them:**

Consider memory aliasing:

```ouro
100 DUP      # Duplicate address
FETCH        # Read from 100
SWAP 42 SWAP STORE  # Write to 100
FETCH        # Second read: value changed!
```

Without uniqueness, multiple references to the same location can exist, making reasoning about state difficult.

With uniqueness types:

```ouro
100 UNIQUE_REF  # Create unique reference
# DUP on unique ref is error
42 WRITE        # In-place write (safe)
RELEASE         # Release uniqueness
```

### Epoch-Scoped Types

**Definition:** Types that are valid only within a specific temporal epoch.

**Formal Statement:**
$$\tau^{[e]}$$

A value of type $\text{u64}^{[3]}$ is a 64-bit integer valid in epoch 3.

**Why Ourochronos Needs Them:**

In temporal computation, values from one epoch may have different meanings or validity in another epoch. Using an epoch-3 value in epoch-5 might be:
- Valid (the value is timeless)
- Invalid (the value's meaning was epoch-specific)
- Transformed (the value needs epoch-adjustment)

Currently, Ourochronos allows unrestricted cross-epoch value flow. Epoch-scoped types would enforce explicit handling:

```ouro
ORACLE[E] 0         # Value scoped to epoch E
# Using outside epoch E is type error
EPOCH_TRANSFER v    # Explicitly move to new epoch
```

---

## Part IV: Effect System Analysis

### Current State

Ourochronos provides effect annotations:

```ouro
PROC pure_fn ( x -- y ) PURE {
    1 ADD
}

PROC reading_fn ( x -- y ) READS {
    FETCH
}

PROC writing_fn ( x y -- ) WRITES {
    STORE
}
```

These annotations are **documentary only**. The compiler does not enforce them.

### What Enforcement Would Provide

**Composability Guarantees:**
```ouro
PROC combine ( -- ) PURE {
    pure_fn   # OK: calling pure from pure
    # reading_fn  # Would be error: calling READS from PURE
}
```

**Optimisation Opportunities:**
Pure functions can be:
- Memoised (same inputs → same outputs)
- Parallelised (no shared state)
- Reordered (no side effects to sequence)

**Temporal Isolation:**
```ouro
PROC temporal_fn ( -- ) TEMPORAL {
    ORACLE 0
}

PROC non_temporal ( -- ) PURE {
    # temporal_fn  # Error: temporal effects must be explicit
}
```

### Proposed Effect Hierarchy

```
           ┌─────────┐
           │   IO    │
           └────┬────┘
                │
     ┌──────────┼──────────┐
     │          │          │
┌────▼────┐ ┌───▼───┐ ┌────▼────┐
│ TEMPORAL│ │ READS │ │ WRITES │
└────┬────┘ └───┬───┘ └────┬────┘
     │          │          │
     └──────────┼──────────┘
                │
           ┌────▼────┐
           │  PURE   │
           └─────────┘
```

Effects form a lattice:
- `PURE` ⊂ `READS` ⊂ `IO`
- `PURE` ⊂ `WRITES` ⊂ `IO`
- `PURE` ⊂ `TEMPORAL` ⊂ `IO`

A function can only call functions with equal or lesser effects.

---

## Part V: Toward a Sound Type System

### Soundness Criteria

A type system is **sound** if well-typed programs do not exhibit undefined behaviour. For Ourochronos, soundness means:

1. **Memory Safety:** All address accesses are within bounds
2. **Temporal Safety:** ORACLE/PROPHECY operations maintain consistency
3. **Resource Safety:** Linear resources are used exactly once
4. **Effect Safety:** Side effects match declarations

### Proposed Type Grammar

```
τ ::= u64                        -- Unsigned 64-bit
    | i64                        -- Signed 64-bit
    | f64                        -- Floating-point
    | bool                       -- Boolean
    | char                       -- Character
    | String                     -- String
    | [τ]                        -- Vector
    | {τ → τ}                    -- Hash map
    | (τ, ...)                   -- Tuple
    | quote<τ... → τ... ! ε>     -- Typed quotation
    | ref<τ>                     -- Reference
    | unique<τ>                  -- Unique reference
    | linear<τ>                  -- Linear value
    | affine<τ>                  -- Affine value
    | temporal<τ, e>             -- Epoch-scoped
    | {x : τ | φ}                -- Refinement
```

### Typing Judgment

$$\Gamma; \Delta; \Sigma \vdash e : \tau\ !\ \varepsilon$$

Where:
- $\Gamma$ is the unrestricted context (copyable values)
- $\Delta$ is the linear context (use-once values)
- $\Sigma$ is the stack type (for stack-based operations)
- $\tau$ is the result type
- $\varepsilon$ is the effect

### Example Typing Rules

**Literal:**
$$\frac{}{\Gamma; \Delta; \Sigma \vdash n : \text{u64}\ !\ \text{pure}}$$

**Oracle:**
$$\frac{\Gamma; \Delta; \Sigma \vdash e : \text{u16}\ !\ \varepsilon}{\Gamma; \Delta; \Sigma \vdash \text{ORACLE}\ e : \text{linear}<\text{u64}>\ !\ \varepsilon \cup \text{temporal}}$$

Note that ORACLE returns a linear type; temporal values cannot be freely duplicated.

**Duplicate (for non-linear values):**
$$\frac{\Gamma; \Delta; \Sigma, \tau \vdash \cdot : \tau\ !\ \varepsilon \quad \tau \not\in \text{Linear}}{\Gamma; \Delta; \Sigma, \tau \vdash \text{DUP} : \Sigma, \tau, \tau\ !\ \varepsilon}$$

---

## Part VI: The Philosophy of Types

### Types as Constraints

Types restrict what programs can be written. This restriction serves as protection: every program excluded by the type system is a program that might fail at runtime.

The minimal type system of current Ourochronos excludes almost nothing. Any sequence of operations is well-typed (insofar as the concept applies). This freedom is also peril: the programmer bears full responsibility for ensuring semantic correctness.

### Types as Documentation

Even unenforced type annotations serve as documentation. The annotation `PURE` on a procedure communicates intent:

```ouro
PROC sqrt_approx ( n -- r ) PURE {
    # Implementation
}
```

A reader knows this function should not access memory or perform I/O. If the implementation violates this contract, the bug is in the implementation; the annotation states the design intent.

### Types as Proof

With a sufficiently rich type system, types become proofs. A function with signature:

$$\text{sorted\_insert} : (\text{SortedList}, \text{elem}) \to \text{SortedList}$$

*proves* that it maintains the sorted invariant, because the type system only permits outputs that satisfy the SortedList predicate.

Ourochronos cannot currently express such invariants. Every list is merely memory addresses; every structure is merely a convention. The type system provides no guarantee that conventions are followed.

---

## Conclusion: The Type Gap

Ourochronos has a type system in the sense that values have runtime representations. It lacks a type system in the deeper sense: it provides no static guarantees about program behaviour.

For a language dealing with temporal computation, where bugs might create causal paradoxes, this constitutes a significant gap. The recommendations of this document point toward a future where:

- **Temporal values** are linear, ensuring proper consumption
- **Memory accesses** are validated through refinement types
- **Effects** are tracked and enforced
- **Epochs** scope values to their valid temporal ranges

The path from here to there is substantial yet traversable. The foundations are sound; the architecture must be built.

---

*"Type systems are the most successful formal methods we have."*
*– Benjamin Pierce*