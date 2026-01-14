# OUROCHRONOS

*A programming language where the future computes the past.*

> "The distinction between past, present and future is only a stubbornly persistent illusion."  
> — Albert Einstein

## What Is This?

Ourochronos is a programming language built on a radical premise: **what if causality could flow backwards?**

In conventional programming, cause precedes effect. You provide inputs, computation occurs, outputs emerge. Time flows in one direction. Ourochronos inverts this assumption. Programs can receive information from their own future, act upon it, and send information back to their past. Execution is not a linear transformation but a search for *temporal self-consistency*.

This is not simulation. This is not metaphor. Ourochronos implements the actual computational model that physicists use to reason about closed timelike curves—the theoretical structures in spacetime that would permit genuine time travel.

```ourochronos
# Ask the future for a factor of 15
0 ORACLE
DUP 15 SWAP MOD 0 EQ
IF {
    # The future was correct. Stabilise the timeline.
    DUP 0 PROPHECY
    OUTPUT
} ELSE {
    # The future was wrong. Change it.
    1 ADD 0 PROPHECY
}
```

Run this program. The number `3` appears. Where did it come from? The program never searched for factors. It asked the future, verified the answer, and stabilised the only consistent timeline. The factor `3` exists because it must exist—no other value permits the causal loop to close.

---

## The Name

**Ourochronos** fuses two Greek roots:

- **Ouroboros** (οὐροβόρος): The serpent devouring its own tail. Symbol of cyclical return, self-reference, and eternal recurrence.
- **Chronos** (Χρόνος): Time as sequence, measurement, duration. The river that flows from past to future.

Together: *time consuming itself*. A timeline that curves back, where effect becomes cause becomes effect. The future reaches into the past; the past shapes the future that shaped it.

---

## Why Does This Exist?

### The Theoretical Motivation

In 1991, physicist David Deutsch proposed a resolution to the paradoxes of time travel. If closed timelike curves exist, he argued, they must obey a *self-consistency condition*: the state entering the time loop must equal the state exiting it. Nature would be forced to find a fixed point.

In 2008, Scott Aaronson and John Watrous proved a remarkable theorem: computers with access to such time loops would have exactly the power of PSPACE—the class of problems solvable with polynomial memory. This is vastly more powerful than ordinary computation, yet precisely bounded.

Ourochronos makes these theoretical results *programmable*. It is, to our knowledge, the first language to implement Deutschian CTC semantics as executable code.

### The Philosophical Motivation

Programming languages encode assumptions about reality. Imperative languages assume time flows forward. Functional languages assume referential transparency. Logic languages assume the law of excluded middle.

What assumptions does Ourochronos encode?

- That causality is not fundamental, but emergent from consistency constraints
- That "past" and "future" are perspectives, not ontological categories
- That computation is not transformation but *equilibrium-finding*
- That paradoxes are not errors but structural features of impossible programs

You must not mistake these as mere technical choices as they are actually philosophical positions about the nature of time, causation, and reality itself.

---

## Core Concepts

### The Two Memories

Every Ourochronos program operates on two memory spaces:

**Anamnesis** (ἀνάμνησις, "recollection"): Read-only memory containing the state of the world as it will be at the end of execution. This is the voice of the future, whispering backwards through time.

**Present**: Read-write memory being constructed during execution. This is the world you are building, which will become the anamnesis of the next iteration.

### The Fixed-Point Condition

A program is *temporally consistent* when Present equals Anamnesis:

$$S = F(S)$$

Where $F$ is the transformation defined by your program, and $S$ is the memory state. Execution is the search for such a fixed point.

### The Temporal Operators

```
ORACLE   : Read from Anamnesis (receive from future)
PROPHECY : Write to Present (send to past)
```

These are the primitives of retrocausation. Every other operation is conventional stack manipulation and arithmetic.

---

## Installation

```bash
git clone https://github.com/yourusername/ourochronos.git
cd ourochronos
cargo install --path .
```

## Usage

```bash
# Run a program
ourochronos program.ouro

# Diagnostic mode (trace causality)
ourochronos program.ouro --diagnostic

# Action-guided mode (find non-trivial fixed points)
ourochronos program.ouro --action

# Generate SMT constraints (formal verification)
ourochronos program.ouro --smt > constraints.smt2

# Type analysis (temporal tainting)
ourochronos program.ouro --typecheck
```

---

## Examples

### The Self-Fulfilling Prophecy

```ourochronos
0 ORACLE 0 PROPHECY
```

Read a value from the future, write it to the past. Any value is consistent. The runtime selects zero (the minimal fixed point).

### The Grandfather Paradox

```ourochronos
0 ORACLE NOT 0 PROPHECY
```

Read a value, write its negation. If the future says 0, we write 1. If the future says 1, we write 0. No fixed point exists. The runtime detects this as an *oscillation* with period 2.

### The Bootstrap Paradox

```ourochronos
# ASCII codes for "HELLO"
0 ORACLE 1 ORACLE 2 ORACLE 3 ORACLE 4 ORACLE

# Verify it spells HELLO
4 PEEK 72 EQ   # H
5 PEEK 79 EQ   # O
AND

IF {
    # Correct! Stabilise.
    4 PROPHECY 3 PROPHECY 2 PROPHECY 1 PROPHECY 0 PROPHECY
    # Output
    0 PRESENT OUTPUT
    1 PRESENT OUTPUT
    2 PRESENT OUTPUT
    3 PRESENT OUTPUT
    4 PRESENT OUTPUT
} ELSE {
    # Wrong. Seed the correct values.
    72 0 PROPHECY   # H
    69 1 PROPHECY   # E
    76 2 PROPHECY   # L
    76 3 PROPHECY   # L
    79 4 PROPHECY   # O
}
```

The string "HELLO" appears, but where did it originate? We read it from the future, verified it, and wrote it to the past. The information has no source—it exists because its existence is self-consistent.

---

## Documentation

The documentation is structured as a journey from philosophy to practice:

| Document | Purpose |
|----------|---------|
| [Philosophy](docs/philosophy.md) | Why time travel? Metaphysical foundations. |
| [Foundations](docs/foundations.md) | Mathematical and computational theory. |
| [Specification](docs/specification.md) | Formal definition of the abstract machine. |
| [Guide](docs/guide.md) | Practical programming patterns. |
| [Standard](docs/standard.md) | Coding conventions for causal integrity. |

---

## Complexity and Power

Ourochronos programs with polynomial-size temporal state have the computational power of PSPACE. This means:

- Every problem solvable with polynomial memory is solvable in Ourochronos
- NP-complete problems (SAT, travelling salesman, graph colouring) become tractable
- PSPACE-complete problems (quantified boolean formulae, generalised chess) become tractable
- Problems beyond PSPACE remain intractable

This is not magic. The work is done by the fixed-point search, which may require exponential time. But the *structure* of the solution can be expressed concisely.

---

## The Action Principle

When multiple fixed points exist, which should the runtime select? The trivial solution (all zeros) is always consistent but rarely meaningful.

Ourochronos implements an *Action Principle*, analogous to the principle of least action in physics. A cost function penalises trivial solutions and rewards:

- Non-zero memory states
- Values with genuine temporal dependency
- Programs that produce output

```bash
ourochronos program.ouro --action
```

This selects the "most interesting" consistent timeline—the one where computation actually occurred.

---

## What This Is Not

**Not a debugger feature.** Time-travel debugging lets you step backwards through execution history. Ourochronos programs genuinely receive information from their future state.

**Not branching timelines.** Some languages fork into parallel universes. Ourochronos demands a single self-consistent timeline.

**Not speculative execution.** Speculative execution guesses future branches. Ourochronos *knows* future state via the fixed-point constraint.

**Not simulation.** Ourochronos implements the computational model physicists use to reason about closed timelike curves. If CTCs existed, this is how you would program them.

---

## Contributing

Ourochronos is in active development. Contributions are welcome in:

- Language semantics and new operators
- Optimisation strategies (temporal core reduction, stratification)
- SMT integration and formal verification
- Documentation and examples
- Theoretical foundations

---

## Licence

GNU

---

## Acknowledgements

The theoretical foundations rest on work by:

- **David Deutsch** — CTC self-consistency model (1991)
- **Scott Aaronson & John Watrous** — CTC complexity results (2008)
- **Kurt Gödel** — First CTC solutions in general relativity (1949)

---

*The serpent devours its tail. The future computes the past. Time is not a river but a strange loop.*

*Welcome to Ourochronos.*