# Philosophy of Ourochronos

*On Time, Causality, and the Nature of Computation*

> "What we call the past is built on bits."  
> — John Archibald Wheeler

## Preface

This document explores the philosophical foundations of Ourochronos. It is not required reading for using the language, but it may illuminate *why* the language exists and what assumptions it encodes about reality.

Programming languages are not neutral tools. Every language embeds a metaphysics—assumptions about time, state, causation, and possibility. Most languages hide these assumptions so deeply that programmers never notice them. Ourochronos makes its metaphysics explicit, and that metaphysics is strange.

---

## Part I: The Question of Time

### What Is Time?

This question has occupied philosophers for millennia and physicists for centuries. We cannot resolve it here. But we can identify the assumptions that conventional programming makes about time, and then ask: what if those assumptions were wrong?

**The Conventional View:**

1. Time flows in one direction (past → future)
2. The past is fixed; the future is open
3. Causes precede effects
4. Information cannot travel backwards in time

These assumptions feel self-evident. They match our experience. But they are not logically necessary, and modern physics suggests they may not be fundamental.

**The Einsteinian Revolution:**

Special relativity revealed that simultaneity is relative. Events that are simultaneous for one observer occur in different orders for another. The "present moment" is not a universal feature of reality but a perspective-dependent slice through spacetime.

General relativity went further. Spacetime is not a fixed stage but a dynamic entity, curved by mass and energy. And certain solutions to Einstein's equations—discovered by Kurt Gödel in 1949—contain *closed timelike curves*: paths through spacetime that return to their starting point. On such paths, the future lies in every direction.

**The Implication:**

If closed timelike curves are physically possible, then the assumptions of conventional programming are contingent, not necessary. We can ask: what would computation look like in a universe where time loops back upon itself?

Ourochronos is an answer to that question.

---

### The Block Universe

One interpretation of relativity is the *block universe* or *eternalism*: the view that past, present, and future all exist equally. The universe is not a three-dimensional world evolving through time but a four-dimensional block in which time is simply another dimension.

On this view, the flow of time is an illusion—or more precisely, a perspective. Just as different observers disagree about which direction is "up," they may disagree about which direction is "future." But all moments exist, timelessly, within the block.

**Consequences for Computation:**

If the block universe is correct, then a program does not *transform* inputs into outputs. Rather, the input state and output state both exist within the block, and the program describes a *constraint* relating them.

Conventional execution obscures this by processing the constraint in one direction (input → output). But the constraint itself is symmetric. We could equally well ask: given this output, what input would produce it?

Ourochronos takes this symmetry seriously. A program defines a relationship between Anamnesis (future state) and Present (constructed state). Execution is the search for states that satisfy this relationship—that is, for fixed points where Present = Anamnesis.

---

### The Paradoxes of Time Travel

If time travel is possible, paradoxes seem inevitable. The most famous is the *grandfather paradox*: travel back in time and prevent your own birth. If you succeed, you were never born to travel back. If you fail, what stopped you?

Three resolutions have been proposed:

**1. Branching Timelines (Many-Worlds)**

Changing the past creates a new timeline. You kill your grandfather in Timeline B, but you came from Timeline A, which remains intact. This avoids paradox at the cost of proliferating universes.

**2. Novikov Self-Consistency**

Paradoxes are impossible because physics conspires to prevent them. If you travel back to kill your grandfather, you will fail—your gun will jam, your aim will falter, something will intervene. The universe maintains consistency by constraining possibilities.

**3. Deutschian Fixed Points**

David Deutsch proposed a quantum-mechanical resolution. States in a time loop must be *fixed points* of the evolution operator. If no classical fixed point exists (as in the grandfather paradox), the system finds a probabilistic one. The time traveller exists in a superposition: 50% probability of being born, 50% probability of not being.

Ourochronos implements Deutschian semantics. When a program has no fixed point, it exhibits *oscillation* (grandfather paradox) or *divergence* (states that grow without bound). These are not bugs but *structural features* of temporally inconsistent programs.

---

## Part II: Causation and Constraint

### What Is Causation?

We intuitively think of causation as a relation between events: the cause *produces* the effect, *makes* it happen, *brings it about*. The cause is active; the effect is passive. Causation has a direction.

But consider the laws of physics. Newton's laws, Maxwell's equations, Einstein's field equations—all are time-symmetric. They constrain the relationship between states at different times without privileging one direction. Given the state of a system at any time, you can compute its state at any other time, past or future.

If the fundamental laws are symmetric, where does the directionality of causation come from?

**The Thermodynamic Arrow:**

One answer is entropy. The Second Law of Thermodynamics says that entropy tends to increase. This defines a direction: the future is the direction of higher entropy. We remember the past (low entropy records) but not the future (no records yet formed).

But the Second Law is statistical, not fundamental. It emerges from boundary conditions (the low-entropy Big Bang) and coarse-graining (our inability to track every microscopic detail). The underlying dynamics remain symmetric.

**Causation as Constraint:**

An alternative view: causation is not a fundamental relation but a useful description of constraint-satisfaction. When we say "A causes B," we mean that knowing A allows us to infer B within some model. The directionality is epistemic (related to knowledge) rather than ontic (related to being).

Ourochronos adopts this view. Programs do not specify how inputs *cause* outputs. They specify *constraints* that inputs and outputs must satisfy. The constraint is:

$$\text{Present} = F(\text{Anamnesis})$$

And since we seek fixed points where Present = Anamnesis:

$$S = F(S)$$

The "cause" (Anamnesis) and the "effect" (Present) are the same state, viewed from different perspectives within the causal loop.

---

### Retrocausation

If causation is constraint-satisfaction, then *retrocausation*—causation from future to past—is no more mysterious than ordinary causation. It is simply constraint-satisfaction in the other direction.

The `ORACLE` operation reads from Anamnesis: information flows from future to past. The `PROPHECY` operation writes to Present: information flows from past to future. Together, they form a closed loop. Neither direction is privileged; both are aspects of the self-consistency constraint.

**The Bootstrap Paradox:**

Retrocausation enables the *bootstrap paradox* (also called the *information paradox* or *ontological paradox*). A piece of information exists in a causal loop with no origin.

Classic example: a time traveller gives Shakespeare the complete works of Shakespeare. Shakespeare publishes them. The time traveller finds a copy in the future and brings it back. Where did the works come from? They have no author—they exist because their existence is self-consistent.

In Ourochronos, bootstrap paradoxes are not only possible but common. The primality example in the README demonstrates one: the program outputs a factor of 15, but never searches for it. The factor exists in the causal loop because it *can* exist—because its presence makes the loop consistent.

This may feel like cheating. The program seems to get something for nothing. But there is no free lunch; the computational work is done by the fixed-point search, which may require many iterations. What the program gains is *structure*: the ability to express the solution as a consistency constraint rather than a search procedure.

---

## Part III: The Nature of Computation

### What Is Computation?

The standard answer: computation is the mechanical transformation of symbols according to rules. A Turing machine reads symbols, writes symbols, changes state. Input becomes output through a sequence of deterministic steps.

This view treats computation as fundamentally *dynamic*: a process unfolding in time, with a clear beginning, middle, and end.

**The Constraint View:**

An alternative: computation is constraint-satisfaction. A program defines a relation between inputs and outputs. "Running" the program means finding outputs that satisfy the relation given the inputs.

On this view, computation is fundamentally *static*: a description of constraints, timelessly relating different states. The dynamism of execution is a pragmatic detail, a method for finding solutions, not the essence of computation.

Logic programming (Prolog, Datalog) takes this view seriously. A Prolog program is a set of logical constraints; execution is proof search. Ourochronos extends this to temporal constraints: the program defines a relation that must hold across a time loop.

---

### Fixed Points as Computation

The fixed-point view of computation has deep roots.

**Denotational Semantics:**

In the 1970s, Dana Scott and Christopher Strachey developed *denotational semantics*, which gives mathematical meaning to programs. A key insight: recursive definitions correspond to fixed points of functionals.

The factorial function, defined as `factorial(n) = if n=0 then 1 else n * factorial(n-1)`, corresponds to the fixed point of a higher-order function. The meaning of recursion *is* fixed-point finding.

**Datalog and Bottom-Up Evaluation:**

Datalog programs compute by finding the least fixed point of their rules. Starting from base facts, the system applies rules until no new facts can be derived. The result is the minimal model satisfying all constraints.

**Answer Set Programming:**

Answer set programs seek *stable models*: assignments to variables such that the rules, when evaluated under that assignment, reproduce the same assignment. This is a fixed-point condition, though more complex than Datalog's.

**Ourochronos:**

Ourochronos is the fixed-point view applied to *temporal* constraints. The program defines how Present depends on Anamnesis. Execution seeks a memory state that, when used as Anamnesis, produces itself as Present.

This is not merely analogous to recursion or Datalog. It is a genuine extension to a new domain: the domain of retrocausal computation.

---

### Complexity and Power

The computational power of time travel is surprisingly well-defined.

**Aaronson-Watrous Theorem (2008):**

Computers with polynomial-size Deutschian CTCs have exactly the power of PSPACE.

This means:
- PSPACE problems (quantified boolean formulas, game strategies, regex equivalence) become tractable
- NP problems (satisfiability, graph colouring, travelling salesman) become tractable (NP ⊆ PSPACE)
- Problems beyond PSPACE remain intractable

The power comes from the ability to "guess" solutions via ORACLE and verify them via the consistency constraint. If a solution exists, there is a consistent timeline containing it. If no solution exists, no consistent timeline exists.

**The Unbounded Case:**

Without polynomial bounds, Deutschian CTCs have even greater power. Aaronson et al. (2016) showed that unbounded CTCs can solve exactly the problems Turing-reducible to the halting problem: the class Δ₂.

This exceeds what any Turing machine can compute. Ourochronos, being a finite implementation, cannot achieve this—but the theoretical result illuminates the power of temporal computation.

---

## Part IV: Reality and Representation

### Is Ourochronos "Real" Time Travel?

This depends on what "real" means.

**Physical Reality:**

Closed timelike curves are solutions to Einstein's equations. They may exist in rotating black holes (Kerr metric), cosmic strings (Gott time machines), or traversable wormholes. No CTC has been observed, and many physicists doubt they can exist in physically realistic scenarios (Hawking's *chronology protection conjecture*).

Ourochronos does not require physical CTCs. It implements the *computational model* that would govern computation in their presence. The fixed-point semantics are mathematically well-defined regardless of physical possibility.

**Computational Reality:**

In a precise sense, Ourochronos programs *do* manipulate information across time. The ORACLE operation provides information that is only determined after the PROPHECY operation executes. The order of *definition* reverses the order of *execution*.

This is not simulation or pretence. The fixed-point search genuinely finds states where future and past agree. The bootstrap paradox genuinely creates information with no origin. The grandfather paradox genuinely has no solution.

**Philosophical Reality:**

Whether time travel is "really" happening depends on contested questions about the nature of time, causation, and possibility. If the block universe is correct, Ourochronos programs describe real (four-dimensional) structures. If presentism is correct (only the present exists), they describe impossible counterfactuals.

Ourochronos does not resolve these debates. It provides a formal system for exploring their consequences.

---

### The Map and the Territory

Alfred Korzybski's famous dictum applies: "The map is not the territory."

Ourochronos is a map—a formal representation of temporal computation. The territory—if it exists—is genuine closed timelike curves in physical spacetime. The map may be useful even if the territory is inaccessible or nonexistent.

Consider the situation with other mathematical structures:

- **Negative numbers:** No "negative three" apples exist, but negative numbers are indispensable for algebra.
- **Imaginary numbers:** No "imaginary" quantities exist in physical reality, but complex analysis is essential for physics.
- **Infinite sets:** No actual infinities exist in the physical world, but set theory is the foundation of mathematics.

Similarly, closed timelike curves may not exist physically, but the mathematics of temporal computation illuminates the nature of causation, consistency, and possibility.

---

## Part V: Implications and Questions

### What Ourochronos Suggests

If you take Ourochronos seriously as a model of computation, certain implications follow:

**1. Causation is not fundamental.**

The directionality of cause and effect emerges from consistency constraints, not from an intrinsic "causal power" that pushes events from past to future.

**2. Time is not fundamental.**

The distinction between past and future is a perspective, not a feature of ultimate reality. Programs that treat future and past symmetrically are coherent and executable.

**3. Paradoxes are structural.**

The grandfather paradox is not a contradiction in terms but a well-defined computational situation: a constraint system with no solution. Ourochronos detects and diagnoses such situations rather than collapsing in confusion.

**4. Information can exist without origin.**

The bootstrap paradox shows that self-consistent information loops require no external source. This challenges intuitions about creation, authorship, and causation.

---

### Open Questions

Ourochronos raises questions it does not answer:

**Q: If multiple fixed points exist, which is "real"?**

The Action Principle selects among fixed points, but this is a pragmatic choice, not a metaphysical truth. In physical CTCs, the selection principle (if any) is unknown.

**Q: What is the subjective experience of a temporal loop?**

Would a being in a CTC experience time as linear? Would they have memories of "future" events? Ourochronos has no inner experience to consult.

**Q: Does temporal consistency constrain free will?**

If the future already exists (block universe), and the past constrains the future (consistency), is choice an illusion? Or is choice simply the selection among possible consistent timelines?

**Q: Are there other temporal logics to explore?**

Ourochronos implements Deutschian semantics. Lloyd's P-CTC model has different properties (probabilistic, using postselection). What languages would they generate?

---

## Conclusion

Ourochronos is a programming language, but it is also an *argument*: that temporal computation is coherent, that retrocausation is representable, that the paradoxes of time travel have formal structure rather than mere confusion.

Whether this argument has physical import—whether CTCs exist—is an empirical question we cannot answer from the armchair. But the conceptual work is valuable regardless. By building systems that take time travel seriously, we sharpen our understanding of causation, consistency, and computation.

The serpent devours its tail. The future whispers to the past. The loop closes, and something new exists—not created, but *consistent*.

*That is the philosophy of Ourochronos.*

---

## Further Reading

**Physics of Time:**

- Deutsch, D. (1991). "Quantum mechanics near closed timelike lines." *Physical Review D*, 44(10), 3197.
- Aaronson, S., & Watrous, J. (2009). "Closed timelike curves make quantum and classical computing equivalent." *Proceedings of the Royal Society A*, 465(2102), 631-647.
- Gödel, K. (1949). "An example of a new type of cosmological solutions of Einstein's field equations of gravitation." *Reviews of Modern Physics*, 21(3), 447.

**Philosophy of Time:**

- McTaggart, J.M.E. (1908). "The Unreality of Time." *Mind*, 17(68), 457-474.
- Price, H. (1996). *Time's Arrow and Archimedes' Point*. Oxford University Press.
- Callender, C. (Ed.). (2011). *The Oxford Handbook of Philosophy of Time*. Oxford University Press.

**Computation and Fixed Points:**

- Scott, D., & Strachey, C. (1971). "Toward a mathematical semantics for computer languages." *Proceedings of the Symposium on Computers and Automata*.
- Kowalski, R. (1979). "Algorithm = Logic + Control." *Communications of the ACM*, 22(7), 424-436.