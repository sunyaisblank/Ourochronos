# Theory, power, and limits

This document states what Ourochronos implements, what follows only under an
idealized model, and what is impossible for the concrete program. It is
normative for complexity and computability claims; `specification.md` remains
normative for syntax and operational behavior.

## 1. Claim ledger

| Claim | Precise status |
|---|---|
| Ourochronos is Turing complete | **Yes for the idealized non-temporal language**, where vector storage and execution time are unbounded. `examples/turing_complete.ouro` is an executable two-counter-machine witness. No concrete finite process is literally Turing complete as a physical state machine. |
| Ordinary point-consistency mode has PSPACE power | **No.** Requiring a pure state `F(s)=s` can reject deterministic cycles and is not the Aaronson--Watrous classical model. |
| Deutsch mode represents the missing classical fixed points | **Yes for deterministic transitions.** A period-`k` cycle is returned as the uniform stationary ensemble, with probability `1/k` per state. |
| The shipped executable is a PSPACE oracle | **No.** It follows one orbit with a configured finite width, finite epoch/gas limits, and can time out. Configurable width enables families but does not supply the ideal selector or prove uniformity/readout invariance. |
| A polynomially uniform Ourochronos family with an ideal Deutsch selector characterizes PSPACE | **Yes, under all conditions in §5.** This is a model-level equality, not a performance claim about iterative simulation. |
| Finite Ourochronos decides the halting problem | **No, and it cannot be made to do so while remaining an effective finite implementation.** It can decide bounded halting and semidecide halting. |
| Unbounded Deutschian CTC Turing machines characterize `Delta^0_2` | **Yes in the cited infinite-dimensional probabilistic/quantum model.** Infinite-support fixed points are essential; merely waiting for a point fixed state is not that construction. |

## 2. Implemented forms of Ourochronos

The project contains several related semantics. They must not be conflated.

1. **Pure/classical language.** Literals, stack operations, `WHILE`,
   quotations, procedures, and epoch-local dynamic structures run as an
   ordinary deterministic language. The standard VM tracks provenance; the
   fast VM is an optimized implementation for programs without temporal
   operations.
2. **Point-consistency CTC (`Standard`).** A program induces a deterministic
   map `F : M -> M`. The runtime iterates from one seed and accepts only when
   `F(s)=s`. A longer recurrent cycle is reported as `Oscillation`.
3. **Diagnostic point consistency (`Diagnostic`).** The same semantics with
   trajectory retention, cycle diagnosis, and a heuristic monotone-divergence
   detector.
4. **Deutsch deterministic CTC (`Deutsch`, CLI `--deutsch`).** A point fixed
   state is a period-one ensemble. A longer deterministic cycle is accepted as
   its uniform stationary distribution. The runtime returns every cycle state,
   the output associated with each state, and a `unanimous_output` only when
   all observable outputs agree.
5. **Action-guided search (`ActionGuided`).** Multiple deterministic seeds are
   tried and point fixed states are ranked by a project-specific action
   functional. This is a deterministic solution-selection heuristic, not
   Deutsch's maximum-entropy rule and not part of the PSPACE theorem.
6. **Pure iteration (`Pure`, library only).** Point search without revisit
   detection or memoization. The implementation still has a one-million-epoch
   safety stop, so the name does not mean physically unbounded execution.
7. **Typed symbolic point solver (`--global`, `--smt`).** A finite,
   solver-independent typed temporal IR is translated to QF_ABV and solved by
   in-process Z3. SAT models are decoded and independently replayed by the
   linked-bytecode VM. Complete-IR UNSAT proves point-state nonexistence; loop-bounded
   UNSAT is `UNKNOWN`. `--smt` exports this same IR rather than a second
   semantics.
8. **All-fixed and property verification.** `--all-fixed` blocks a first model
   and resolves to prove uniqueness or return two replayed witnesses.
   Source-level `PROPERTY ... ALL_FIXED CELL ...` declarations are verified by
   searching for a violating fixed state. Proof, refutation, vacuity, and
   unknown are distinct results with versioned JSON artifacts.
9. **Complete small-state recurrence (`--recurrent`).** For an explicitly
   bounded word domain, the analyzer executes every state, refuses partial or
   non-closed transitions, and classifies every cycle, point state, transient
   distance, and basin. This is a conventional exhaustive algorithm, not an
   ideal selector. Graphviz export contains the entire graph.
10. **Finite temporal regions.** `TEMPORAL base size BITS n` embeds a finite
    total core in the general stack language. Its addresses are relative, its
    writes are n-bit, and static region contracts reject recursion and
    operations without deterministic IR lowering. General I/O, structures,
    and FFI remain ordinary-language facilities outside that proof core.
11. **Static/observational layers.** Temporal taint analysis, provenance, the
   effect gate, frozen input, audit logging, REPL, and LSP support the above
   execution forms but do not add computational power.
12. **Parametric finite-width runtime.** `--memory-cells m` fixes a positive
   width for one run; 64-bit addresses can cross the legacy 16-bit boundary.
   The VM, convergence journal/cache, provenance top, and action seeds use that
   width. Typed SMT arrays retain 64-bit indices and normalize addresses against
   the exact configured width. The default remains 65,536.
13. **Complexity contracts.** `--resources` prints an exact concrete profile.
    A top-level `FAMILY` declaration and the Rust API's
    `PspaceFamilyContract` separately record polynomial bounds and every model
    assumption, while deliberately distinguishing a declared obligation from
    a verified proof. `examples/pspace_contract.ouro` is executable syntax.
14. **Exact rational stochastic backend.** `RationalMarkovChain` validates a
    finite row-stochastic matrix, finds every closed recurrent class, and
    solves its extremal stationary distribution exactly. `StationaryFamily`
    exposes the full convex hull and cycle/class-specific acceptance
    probabilities. A top-level `MARKOV` declaration exposes the backend in
    source and automatically classifies ACCEPT, REJECT, or AMBIGUOUS across
    every extremal fixed point.
15. **Bounded classical halting analysis.** `--halting-bound T` observes the
    deterministic, non-temporal core. A positive halt is conclusive;
    exhaustion returns `UNKNOWN`/exit 3. Temporal and externally stateful
    programs are explicitly outside this analyzer.
16. **Numerical quantum CPTP backend.** `QuantumChannel` accepts complex Kraus
    operators, validates trace preservation, applies the channel, and
    approximates one fixed density operator through Cesaro averaging with an
    explicit residual and iteration bound. A source-level qubit `QCHANNEL`
    additionally characterizes the complete affine fixed set in the Bloch ball
    and verifies basis-readout bounds over every fixed density. This is
    deliberately separate from exact classical execution.

Not implemented: unrestricted solving for arbitrary recursive/unbounded
programs, automatic termination proofs for general WHILE loops,
higher-dimensional all-fixed-density quantum solving,
general POVM source syntax, exact algebraic quantum amplitudes, a
maximum-entropy fixed-point selector, automatic extraction of a stochastic
matrix from arbitrary VM code, a machine-checked verifier for FAMILY
declarations, or an infinite-dimensional CTC.

## 3. Turing completeness

### 3.1 Idealized semantics

Let `Ouro_infinity` be the non-temporal language with the same operational
rules as Ourochronos except that:

- vector capacity and stack depth have no fixed bound;
- the number of executed instructions is not capped; and
- allocation succeeds whenever the mathematical run uses a finite amount of
  storage at that point.

`Ouro_infinity` can simulate a two-counter Minsky machine. Represent each
natural-number counter by a vector of identical tokens and keep a finite
program counter in a memory cell. Then:

```text
INC(c, next)       := vector[c].push(token); pc := next
DECJZ(c, z, nz)    := if len(vector[c]) = 0
                      then pc := z
                      else vector[c].pop(); pc := nz
HALT               := running := 0
```

A `WHILE` loop dispatches on `pc`. The translation from each finite
two-counter program to Ourochronos source is effective, and two-counter
machines are Turing universal. Therefore a universal Turing machine can be
simulated, establishing Turing completeness of `Ouro_infinity`.

`examples/turing_complete.ouro` implements this encoding and its integration
test executes a three-token transfer. The example is evidence that the
required instructions compose correctly; the universality statement rests on
the parametric translation above, not on that one finite run.

### 3.2 Concrete qualification

The Rust executable has a configured finite address space, 64-bit values,
host-limited vectors/stacks, and a default gas limit. Every actual run therefore explores
only finitely much state. “Turing complete” has the standard programming
language meaning: there is no language-level bound in the idealized storage
model, while any particular physical execution remains resource bounded.

## 4. Fixed-point theory and CTC physics

### 4.1 Pure-state fixed points

For temporal memory `M` and deterministic epoch function `F : M -> M`, a
point fixed state satisfies

```text
F(s) = s.
```

Point fixed states need not exist: Boolean negation has no solution. Iterating
`F` from a seed is an orbit-search algorithm, not a general fixed-point oracle.
It can find only a fixed state reachable from that seed.

### 4.2 Deterministic cycles as stationary distributions

Lift `F` to distributions by push-forward:

```text
(F_* mu)(y) = sum_{x : F(x)=y} mu(x).
```

A Deutsch-consistent classical state is a probability distribution `mu` with
`F_* mu = mu`. If `x_0 -> x_1 -> ... -> x_(k-1) -> x_0` is a deterministic
cycle, the uniform distribution on those `k` states is stationary. Thus every
deterministic map on a finite nonempty state space has at least one
distributional fixed point even when it has no point fixed state. The complete
set of stationary distributions is the convex hull of the uniform
distributions on all recurrent cycles.

`--deutsch` implements exactly this lift for the single recurrent cycle
reached from the configured seed. It stores the distribution exactly as a
cycle rather than using floating point. It does not enumerate all recurrent
cycles; consequently, one run does not certify global readout invariance.
For a declared small closed word domain, `--recurrent` removes that limitation
by enumerating the complete functional graph. Its completeness is relative to
the explicit finite domain and state limit, which are printed in the result.

### 4.3 General classical and quantum Deutsch models

For a stochastic transition matrix `P`, consistency is `pi P = pi`. A finite
Markov chain has at least one stationary distribution. Source-level `MARKOV`
models implement this equation exactly for rational `P`, enumerate every
closed recurrent class, and verify the bounded-error readout on all extremal
stationary distributions. For a quantum CTC, a
density operator `rho` is consistent under a completely positive,
trace-preserving map `Phi` when `Phi(rho)=rho`. Finite-dimensional CPTP maps
have fixed density operators. Ourochronos implements deterministic `F`, finite
rational stochastic `P`, and numerical Kraus/CPTP quantum `Phi`. For declared
qubit channels it characterizes every fixed density via the Bloch ball and
checks the basis readout globally; higher dimensions remain numerical
one-fixed-point exploration only.

### 4.4 Interaction form of Deutsch's prescription

The physics model distinguishes a chronology-respecting system `R` from the
system `C` traversing the CTC. Given an incoming chronology-respecting density
operator `sigma_R` and an interaction unitary `U` (or, more generally, a CPTP
interaction), the induced channel on the CTC system is

```text
Phi_sigma(rho_C) = Tr_R[ U (sigma_R tensor rho_C) U^dagger ].
```

Deutsch consistency requires

```text
rho_C = Phi_sigma(rho_C).
```

After a consistent `rho_C` is supplied, the observable
chronology-respecting output is

```text
sigma'_R = Tr_C[ U (sigma_R tensor rho_C) U^dagger ].
```

The classical stochastic model replaces density operators by probability
vectors, the interaction by a stochastic map, and partial trace by
marginalization. A deterministic map is the special case whose transition
matrix has one `1` per row. The finite probability simplex and the
finite-dimensional density-operator set are compact and convex, and their
stochastic/CPTP maps are continuous; fixed states therefore exist even when no
deterministic pure state satisfies `F(s)=s`.

Fixed states can be nonunique. Deutsch proposed a maximum-entropy selection
rule, but the Aaronson--Watrous complexity definitions do not rely on selecting
a favorable fixed point: a valid computation must return the promised answer
for every allowed fixed state. Ourochronos follows that robust rule in MARKOV
and qubit QCHANNEL analysis. Action mode is a separate project heuristic and
must not be identified with Deutsch's entropy proposal.

These equations are a hypothetical consistency prescription, not a claim that
physical CTCs exist. General relativity admits mathematical spacetimes with
CTCs under special global conditions, while their formation and compatibility
with quantum gravity remain unresolved. Ourochronos models the information
consistency equation; it does not simulate spacetime geometry, stress-energy,
chronology horizons, or quantum gravity. See Deutsch's original paper,
[*Quantum mechanics near closed timelike lines*](https://doi.org/10.1103/PhysRevD.44.3197).

## 5. The exact PSPACE statement

The theorem is

```text
P_CTC = BQP_CTC = PSPACE.
```

Here `P_CTC` and `BQP_CTC` are the Aaronson--Watrous classes, not ordinary
Ourochronos CLI runs. The equality requires a family indexed by input `x` of
length `n` satisfying all of the following:

1. A deterministic classical or quantum description of the circuit is
   generated uniformly in polynomial time.
2. The chronology-respecting and CTC registers contain at most `poly(n)` bits
   or qubits.
3. One application of the circuit has polynomial size/time and defines a
   total stochastic map (classical) or CPTP map (quantum).
4. Nature supplies a stationary distribution or fixed density operator; the
   algorithm is not charged the possibly exponential time needed by a
   conventional simulator to locate it.
5. Correctness holds for **every** fixed point allowed by the circuit: in a
   bounded-error formulation, every one accepts with probability at least
   `2/3`, or every one rejects with probability at least `2/3`. For a
   deterministic Ourochronos encoding, every recurrent cycle must therefore
   give the same decision readout.
6. External state and nondeterminism are part of the declared total map or are
   frozen; a transition that changes between evaluations is not one circuit.

Under those assumptions, classical and quantum CTC computation are both
contained in PSPACE, and a classical CTC construction simulates any
polynomial-space Turing machine. The equality and construction are proved in
Aaronson and Watrous, [*Closed Timelike Curves Make Quantum and Classical
Computing Equivalent*](https://arxiv.org/abs/0808.2669).

Ourochronos now has configurable finite CTC width and the correct deterministic stationary-cycle primitive, but
the concrete executable does not itself meet the asymptotic model:

- a sequence of widths can instantiate `m=p(n)`, but a uniform generator and
  proof of the polynomial bound remain external obligations;
- ordinary execution searches an orbit instead of receiving a fixed point;
- `max_epochs` and `max_instructions` can produce `Timeout` or `Error`;
- one seed witnesses one recurrent class, not correctness over all classes;
- action mode deliberately selects among point fixed states and is outside the
  theorem.

Accordingly, programs can encode finite PSPACE-style consistency
constructions, but claiming that running the binary grants PSPACE speed or
decides every PSPACE language would be false. A complete model-facing backend
would need parametric CTC width plus an ideal selector interface (or an
explicitly exponential conventional solver) and a readout-invariance proof.

## 6. Unbounded CTCs, `Delta^0_2`, and halting

The computability-theoretic equality is

```text
Computable_CTC = QComputable_CTC = Computable^Halt = Delta^0_2.
```

This concerns Deutschian CTC Turing machines with no bound on width or length,
equivalently computable Markov chains or quantum channels on countably
infinite-dimensional state spaces. The halting construction can require fixed
probability distributions with **infinite support**. Requiring finite support,
or even a computable bound on expected string length, collapses the power back
to computable languages. See Aaronson et al., [*Computability Theory of Closed
Timelike Curves*](https://arxiv.org/abs/1609.05507).

The tempting point-search recipe—“encode a history; if the simulated machine
halts a fixed point appears, otherwise search forever”—does not decide
halting. In the nonhalting case it never returns `NO`, so it is only a
semidecision procedure. Moreover, a finite deterministic state transition
always has a cycle, and therefore always has a Deutsch stationary
distribution. Absence of a point fixed state is not absence of a Deutsch fixed
point.

No total effective finite extension of Ourochronos can decide halting for all
Ourochronos programs. If a procedure `H(P,x)` always halted and answered
whether `P(x)` halts, the usual diagonal program that halts exactly when
`H` predicts it will not yields a contradiction. Larger integers, more epochs,
SMT, seed enumeration, and cycle detection change practical coverage but do
not evade this proof.

What can be implemented honestly is:

- **bounded halting:** simulate at most `T` steps and answer whether halting
  occurred within the bound. This is implemented by `BoundedHaltingAnalyzer`
  and `--halting-bound`;
- **halting semidecision:** simulate without a bound and return when the target
  halts, with no required result otherwise;
- **limit approximation:** increase `T` and revise a guess; this converges
  pointwise but supplies no computable time at which a negative answer is
  final;
- **relative computation:** query a real external halting oracle, making the
  new assumption explicit rather than presenting it as an implementation;
- **infinite-dimensional theory:** specify the countable-state stationary
  distribution model mathematically, without claiming a finite simulator
  realizes its noncomputable selector.

`examples/turing_complete.ouro` exercises the positive side: any particular
halting simulation can complete. An intentionally nonhalting program reaches
the gas limit or runs forever; that outcome is `unknown`, never proof that the
program cannot halt under an unbounded run.

## 7. Development frontier

The next sound extensions, in decreasing order of leverage, are:

1. a static/solver-backed verifier for in-language `FAMILY` declarations;
2. a sparse alternative for very wide, low-occupancy finite temporal memories;
3. add arbitrary-precision rationals and sparse matrices for larger MARKOV
   instances, plus extraction from a bounded VM transition;
4. static or solver-assisted verification that the decision bit agrees on all
   recurrent classes/fixed points;
5. extend bounded-halting analysis with resumable checkpoints and systematic
   cutoff experiments while retaining the `UNKNOWN` result;
6. a formal small-step semantics and machine-checked compiler from a universal
   counter machine;
7. quantum CPTP support, which is a distinct implementation project and should
   advance from complete qubit fixed-space verification to higher-dimensional
   semidefinite fixed-space/readout analysis and general POVMs.

None of these finite engineering steps can implement the noncomputable
infinite-support selector required for `Delta^0_2`; they can make the boundary
more explicit and test larger finite approximations.
