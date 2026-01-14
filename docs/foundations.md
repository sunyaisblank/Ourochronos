# Mathematical Foundations

*From Fixed Points to Temporal Computation*

> "The only way to rectify our reasonings is to make them as tangible as those of the Mathematicians."  
> — Gottfried Wilhelm Leibniz

## Overview

This document develops the mathematical theory underlying Ourochronos. We proceed from elementary concepts (fixed points, iteration) through the physics of closed timelike curves to the complexity-theoretic results that bound the power of temporal computation.

Prerequisites: basic familiarity with sets, functions, and computational complexity (P, NP, PSPACE). No physics background is assumed; we will develop the relevant concepts from scratch.

---

## Part I: Fixed-Point Theory

### 1.1 What Is a Fixed Point?

A *fixed point* of a function $f: X \to X$ is an element $x \in X$ such that:

$$f(x) = x$$

The element is "fixed" because applying the function leaves it unchanged.

**Examples:**

- For $f(x) = x^2$ on the reals, the fixed points are $0$ and $1$.
- For $f(x) = \cos(x)$ on the reals, there is one fixed point at approximately $0.739$ (the *Dottie number*).
- For the identity function $f(x) = x$, every element is a fixed point.
- For $f(x) = x + 1$ on the integers, there are no fixed points.

Fixed points are fundamental throughout mathematics. In Ourochronos, they are the foundation of execution: running a program means finding a memory state $S$ such that $F(S) = S$, where $F$ is the state transformation defined by the program.

---

### 1.2 Iteration and Convergence

One method for finding fixed points is *iteration*: start with an initial guess $x_0$, compute $x_1 = f(x_0)$, then $x_2 = f(x_1)$, and so on. If the sequence $(x_n)$ converges to some limit $x^*$, and $f$ is continuous, then $x^*$ is a fixed point.

**Example:** Finding $\sqrt{2}$ via iteration.

Let $f(x) = (x + 2/x) / 2$. Starting from $x_0 = 1$:

| $n$ | $x_n$ |
|-----|-------|
| 0 | 1.000000 |
| 1 | 1.500000 |
| 2 | 1.416667 |
| 3 | 1.414216 |
| 4 | 1.414214 |

The sequence converges rapidly to $\sqrt{2} \approx 1.414214$, which satisfies $f(x) = x$.

---

### 1.3 When Do Fixed Points Exist?

Not every function has a fixed point. Two classical theorems guarantee existence under certain conditions:

**Brouwer Fixed-Point Theorem:**

Every continuous function from a compact convex set to itself has at least one fixed point.

*Intuition:* If you stir a cup of coffee, at least one molecule ends up where it started.

**Banach Fixed-Point Theorem:**

If $(X, d)$ is a complete metric space and $f: X \to X$ is a *contraction* (there exists $k < 1$ such that $d(f(x), f(y)) \le k \cdot d(x, y)$ for all $x, y$), then $f$ has exactly one fixed point, and iteration from any starting point converges to it.

*Intuition:* If the function always brings points closer together, repeated application must converge to a single point.

---

### 1.4 Fixed Points in Computer Science

Fixed-point theory pervades computer science:

**Recursive Definitions:**

The factorial function is defined by:
$$\text{fact}(n) = \begin{cases} 1 & \text{if } n = 0 \\ n \cdot \text{fact}(n-1) & \text{otherwise} \end{cases}$$

This is a fixed-point equation: $\text{fact}$ is the unique function satisfying $\text{fact} = F(\text{fact})$, where $F$ is a higher-order functional.

**Denotational Semantics:**

The meaning of a while loop is the least fixed point of a functional:
$$\text{while } b \text{ do } S = \text{if } b \text{ then } (S; \text{while } b \text{ do } S) \text{ else skip}$$

**Datalog:**

A Datalog program computes the least fixed point of its rules. Starting from base facts, rules are applied until no new facts can be derived.

**Ourochronos:**

Execution finds a fixed point of the state transformation: a memory state that, when used as input (Anamnesis), produces itself as output (Present).

---

## Part II: Closed Timelike Curves

### 2.1 Spacetime and World Lines

In special and general relativity, space and time are unified into a four-dimensional *spacetime*. A point in spacetime is an *event*: a location and a time.

A *world line* is the path of an object through spacetime. For ordinary objects, world lines always move forward in time (in any reference frame). Such paths are called *timelike*.

A *closed* timelike curve is a world line that returns to its starting event. An object following such a path would meet its past self. This is the formal definition of time travel in physics.

---

### 2.2 Do CTCs Exist?

CTCs appear in several solutions to Einstein's field equations:

**Gödel Metric (1949):**

Kurt Gödel discovered a rotating universe solution containing CTCs through every point. While physically unrealistic (our universe is not rotating in this way), it demonstrated that CTCs are compatible with general relativity.

**Kerr Metric:**

The solution describing a rotating black hole contains CTCs in the interior region. Whether physical matter can traverse them is disputed.

**Traversable Wormholes:**

Theoretical wormholes, if stabilised with exotic matter, could permit CTCs. No such wormholes are known to exist.

**Chronology Protection:**

Stephen Hawking conjectured that physics conspires to prevent CTCs in physically realistic scenarios—the "chronology protection conjecture." This remains unproven.

*For Ourochronos, the physical existence of CTCs is irrelevant. We implement their computational model, which is mathematically well-defined regardless of physical possibility.*

---

### 2.3 The Grandfather Paradox

The classic objection to CTCs is the *grandfather paradox*: travel back in time, prevent your own birth, and you are never born to travel back.

Formally: let $x$ represent whether you exist. If $x = 1$, you travel back and cause $x = 0$. If $x = 0$, you do not travel back, so $x = 1$. There is no consistent value for $x$.

This is the fixed-point equation $x = \neg x$, which has no boolean solution.

---

### 2.4 Deutsch's Resolution

In 1991, David Deutsch proposed a quantum-mechanical resolution. Instead of classical bits, consider quantum states—density matrices. The fixed-point equation becomes:

$$\rho = \text{Tr}_A \left[ U (\rho_{\text{in}} \otimes \rho) U^\dagger \right]$$

where $U$ is the unitary evolution around the CTC.

**Key insight:** Even when no classical fixed point exists, a *probabilistic* (mixed-state) fixed point may exist. For the grandfather paradox, the solution is:

$$\rho = \frac{1}{2} |0\rangle\langle 0| + \frac{1}{2} |1\rangle\langle 1|$$

The time traveller exists with probability 1/2. This is consistent: if you exist with probability 1/2 and you prevent your existence when you exist, you end up existing with probability 1/2.

---

### 2.5 Deutschian CTCs in Ourochronos

Ourochronos implements a *classical, deterministic* version of Deutschian CTCs. Programs define a function $F$ on memory states, and execution seeks a fixed point $S = F(S)$.

When no classical fixed point exists, Ourochronos diagnoses the situation:

| Pattern | Fixed Point? | Ourochronos Status |
|---------|--------------|-------------------|
| $x' = x$ | All values | Consistent |
| $x' = 42$ | $x = 42$ | Consistent |
| $x' = \neg x$ | None | Oscillation (period 2) |
| $x' = x + 1$ | None | Divergence |

Oscillation corresponds to the grandfather paradox. Divergence corresponds to unbounded growth (no fixed point in finite memory).

---

## Part III: Complexity Theory

### 3.1 Complexity Classes

Complexity theory classifies problems by the resources required to solve them.

**P:** Problems solvable in polynomial time by a deterministic Turing machine. (Sorting, shortest path, primality testing.)

**NP:** Problems where solutions can be *verified* in polynomial time. (Satisfiability, graph colouring, travelling salesman.)

**PSPACE:** Problems solvable with polynomial *space* (memory), regardless of time. (Quantified boolean formulas, generalised chess, regular expression equivalence.)

The relationships: $\text{P} \subseteq \text{NP} \subseteq \text{PSPACE}$. It is believed (but unproven) that all inclusions are strict.

---

### 3.2 The Aaronson-Watrous Theorem

In 2008, Scott Aaronson and John Watrous proved a remarkable result:

**Theorem:** $\text{P}^{\text{CTC}} = \text{BQP}^{\text{CTC}} = \text{PSPACE}$

Where:
- $\text{P}^{\text{CTC}}$: Classical polynomial-time computation with access to polynomial-size CTCs
- $\text{BQP}^{\text{CTC}}$: Quantum polynomial-time computation with access to polynomial-size CTCs

**Meaning:** Computers with Deutschian CTCs—whether classical or quantum—have exactly the power of PSPACE. CTCs make quantum computers no more powerful than classical computers (both reach PSPACE), and both become much more powerful than they would be without CTCs.

---

### 3.3 Why PSPACE?

**Upper Bound (CTC ⊆ PSPACE):**

A PSPACE machine can simulate CTC computation by searching for fixed points. Given a polynomial-size state space, checking each candidate state requires polynomial time, and there are at most exponentially many states. PSPACE machines can iterate through exponentially many states using polynomial memory.

**Lower Bound (PSPACE ⊆ CTC):**

Any PSPACE problem can be encoded as a fixed-point search. The key technique: encode a quantified boolean formula (QBF) as a CTC computation.

For a QBF like $\forall x \exists y : \phi(x, y)$, the CTC reads candidate assignments from the future, verifies them against $\phi$, and writes corrections to the past. A consistent assignment exists if and only if the formula is true.

---

### 3.4 Implications for Ourochronos

Ourochronos programs with polynomial-size temporal state (the memory accessed via ORACLE and PROPHECY) can solve any problem in PSPACE.

**What becomes tractable:**

- NP-complete problems: SAT, 3-colouring, Hamiltonian path, subset sum
- PSPACE-complete problems: QBF, generalised chess, Go (on polynomial-size boards), regex equivalence

**What remains intractable:**

- EXPTIME-complete problems: generalised chess on exponential-size boards
- EXPSPACE-complete problems: certain formal language problems
- Undecidable problems: halting problem, Diophantine equations

**The catch:** While solutions can be *expressed* concisely, *finding* them may require exponential time (the fixed-point search). The power of CTCs is in expression, not in raw speed.

---

### 3.5 Unbounded CTCs

Without size restrictions, CTCs become even more powerful.

**Theorem (Aaronson et al., 2016):** Turing machines with unbounded Deutschian CTCs can solve exactly the problems in $\Delta_2$—the class of languages Turing-reducible to the halting problem.

This exceeds what any Turing machine can compute. In particular, CTC computers can solve the halting problem itself!

**How?** Encode a Turing machine's execution history in the CTC. If the machine halts, there is a consistent fixed point. If it runs forever, the fixed-point search diverges. The CTC "knows" whether halting occurs by whether a fixed point exists.

*Ourochronos, being a finite implementation, cannot achieve $\Delta_2$ power. But the theoretical result illuminates the potential of temporal computation.*

---

## Part IV: The Action Principle

### 4.1 The Problem of Multiple Fixed Points

A function may have many fixed points. For $f(x) = x$, every value is a fixed point. For a constant function $f(x) = c$, only $c$ is a fixed point.

For Ourochronos programs, multiple fixed points create ambiguity. Which timeline is "real"?

**Example:**

```ourochronos
0 ORACLE DUP IF { DUP 0 PROPHECY } ELSE { 42 0 PROPHECY }
```

Both $S = 0$ (read 0, condition false, write 42... wait, this is inconsistent) and $S = 42$ (read 42, condition true, write 42) are fixed points. Actually, let me reconsider: if $S = 0$, we read 0, the condition is false (0 is falsy), so we write 42, giving Present ≠ Anamnesis. If $S = 42$, we read 42, the condition is true, so we write 42, giving Present = Anamnesis. So only $S = 42$ is a fixed point.

But for simpler programs like `0 ORACLE 0 PROPHECY`, every value is consistent. Which should the runtime choose?

---

### 4.2 The Genie Effect

A naive runtime might always choose the simplest fixed point: all zeros. This is the "Genie Effect"—the system finds the path of least resistance rather than the intended solution.

For the primality example, the trivial fixed point is $S = 0$: read 0, check if 0 is a factor, fail the checks, write some value that eventually stabilises at 0. This is consistent but useless.

---

### 4.3 Variational Selection

Physics offers a solution: the *Principle of Least Action*. Among all possible paths, nature selects the one that minimises (or extremises) the action functional.

Ourochronos adapts this idea. We define an *action* for each fixed point:

$$\text{Action}(S) = \sum_i \left[ w_{\text{zero}} \cdot \delta(S_i = 0) + w_{\text{pure}} \cdot \delta(S_i \text{ is pure}) - w_{\text{causal}} \cdot \text{depth}(S_i) - w_{\text{output}} \cdot \text{outputs} \right]$$

Where:
- $w_{\text{zero}}$: Penalty for zero values (trivial)
- $w_{\text{pure}}$: Penalty for values not derived from ORACLE (non-temporal)
- $w_{\text{causal}}$: Bonus for values with deep causal chains
- $w_{\text{output}}$: Bonus for producing output

The runtime explores multiple seeds and selects the fixed point with minimal action. This prefers "interesting" solutions over trivial ones.

---

### 4.4 Provenance Tracking

To compute causal depth, Ourochronos tracks *provenance*: which ORACLE reads influenced each value.

Every value carries metadata indicating its temporal dependencies. When values are combined (added, compared, etc.), their provenances merge. Values derived purely from constants have no provenance (they are "pure"). Values derived from ORACLE have provenance indicating which addresses were read.

This enables:
1. Action computation (penalise pure values, reward causal depth)
2. Static analysis (identify temporal core, detect negative loops)
3. Paradox diagnosis (explain why oscillation occurs)

---

## Part V: Formal Properties

### 5.1 Termination

Ourochronos programs may not terminate. Three cases:

**Convergence:** The fixed-point iteration reaches $S = F(S)$. Execution halts with a consistent result.

**Oscillation:** The iteration enters a cycle: $S_1 \to S_2 \to ... \to S_k \to S_1$. No fixed point exists. This is the grandfather paradox.

**Divergence:** Values grow without bound. The iteration never revisits a state. This occurs when the program increments without bound.

The runtime detects all three cases and reports appropriate diagnostics.

---

### 5.2 Decidability

**Question:** Given an Ourochronos program, is it decidable whether a fixed point exists?

**Answer:** No, in general.

For programs with unbounded computation per epoch, the problem reduces to the halting problem. A program that runs a Turing machine and writes "halted" to memory if it halts has a fixed point if and only if the Turing machine halts.

For programs with bounded epoch computation (polynomial-size state, polynomial-time epochs), the problem is in PSPACE (exhaustively search the state space).

---

### 5.3 Uniqueness

**Question:** When a fixed point exists, is it unique?

**Answer:** Not necessarily.

Programs like `0 ORACLE 0 PROPHECY` have infinitely many fixed points (any value works). The Action Principle selects among them.

Programs with genuine verification (check a condition, stabilise if true, perturb if false) tend to have unique or few fixed points.

---

### 5.4 Compositionality

**Question:** If $P$ and $Q$ are programs with fixed points, does their composition $P; Q$ have a fixed point?

**Answer:** Not necessarily.

The composition interleaves their temporal constraints. Even if each program is individually consistent, their combination may create contradictions.

This is analogous to how two satisfiable SAT formulas may become unsatisfiable when conjoined.

---

## Conclusion

The mathematics of Ourochronos rests on three pillars:

1. **Fixed-point theory** provides the conceptual foundation: execution is the search for states satisfying $S = F(S)$.

2. **CTC physics** provides the model: Deutsch's self-consistency condition resolves paradoxes by demanding fixed points.

3. **Complexity theory** bounds the power: polynomial CTCs give PSPACE; unbounded CTCs give $\Delta_2$.

These are not arbitrary choices but principled derivations from the question: *what is computation in the presence of time travel?* The mathematics follows from the physics, and the language follows from the mathematics.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $S$ | Memory state |
| $F$ | State transformation function |
| $A$ | Anamnesis (future memory) |
| $P$ | Present (constructed memory) |
| $\rho$ | Density matrix (quantum state) |
| $U$ | Unitary operator |
| PSPACE | Polynomial space complexity class |
| $\Delta_2$ | Languages Turing-reducible to halting |
| CTC | Closed timelike curve |

---

## References

1. Aaronson, S., & Watrous, J. (2009). "Closed timelike curves make quantum and classical computing equivalent." *Proceedings of the Royal Society A*, 465(2102), 631-647.

2. Aaronson, S., Bavarian, M., & Gueltrini, G. (2016). "Computability theory of closed timelike curves." *arXiv:1609.05507*.

3. Deutsch, D. (1991). "Quantum mechanics near closed timelike lines." *Physical Review D*, 44(10), 3197.

4. Gödel, K. (1949). "An example of a new type of cosmological solutions of Einstein's field equations of gravitation." *Reviews of Modern Physics*, 21(3), 447.

5. Banach, S. (1922). "Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales." *Fundamenta Mathematicae*, 3, 133-181.