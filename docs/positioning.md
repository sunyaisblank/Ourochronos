# Positioning Ourochronos

Ourochronos should make a narrow, defensible claim: it aims to be the strongest
tool for **executable finite self-consistent computation**, not the strongest
general formal-methods system.

## Compared with time-travel esolangs

Most fictional or esoteric time-travel languages make temporal order the joke
or puzzle. Ourochronos can still express those programs, but now adds a typed
finite IR, a precise `F(s)=s` equation, global SAT/UNSAT/UNKNOWN outcomes,
reference-VM witness replay, all-solution ambiguity checks, universal
properties, complete small-state recurrence, and machine-readable evidence.
Its advantage is that paradox and consistency are verification results rather
than folklore about an interpreter.

## Compared with logic and constraint languages

Constraint systems already solve circular equations extremely well.
Ourochronos's distinctive value is the combination of constraints with an
ordinary executable language and an explicit temporal state boundary: the
same source induces the transition, emits observations, is solved globally,
and is replayed operationally. It should interoperate with SMT rather than
pretend to replace mature solver front ends.

## Compared with synchronous/reactive and circular data-flow systems

Reactive languages usually reject instantaneous causality cycles, require a
delay, or select a least fixed point in a particular lattice. Ourochronos lets
the programmer state the consistency relation directly, distinguish
nonexistence from nonuniqueness, inspect all small recurrent histories, and use
PARADOX to eliminate invalid worlds. Its cost is a stricter finite/total core
and potentially exponential solving.

## Compared with formal verification and model checking

Mature theorem provers and model checkers remain much broader: richer temporal
logics, refinement, concurrency semantics, proof scripts, induction,
probabilistic verification, industrial front ends, and decades of validation.
Ourochronos does not currently surpass that family in generality. Its niche is
different: the verified object is itself a runnable time-loop program, and a
countermodel is replayed by the same linked-bytecode VM.

The present verifier is credible within that niche because it distinguishes:

- orbit evidence from global solving;
- SAT witnesses from solver trust through replay;
- complete UNSAT from loop-bounded UNKNOWN;
- existence from uniqueness and all-fixed safety;
- point fixed states from recurrent/stationary classes;
- a complete explicit domain from a non-closed or oversized request.

To compete directly with the broad formal-methods family, future work needs a
richer property logic, compositional region contracts, mechanized metatheory,
proof-producing/checkable UNSAT certificates, incremental/portfolio solvers,
concurrency, and larger independently reviewed case studies.

## The strongest three applications

1. **Bounded self-consistency synthesis and model checking.** Generate a state
   satisfying circular constraints, then prove safety over every fixed state.
   This is the clearest primary use.
2. **Circular data flow and bidirectional computation.** State mutually
   dependent equations without inventing an evaluation order, while detecting
   no solution or multiple solutions explicitly.
3. **Retrocausal simulation and game-rule engines.** Treat stable histories and
   recurrent cycles as semantic outcomes, enumerate all histories for small
   domains, and verify whether their observations agree.

The executable counterparts live in `examples/case_studies/`.
