# Whole-system alignment contract

This document is the normative cross-component contract for source admission,
execution, proof, artifacts, and initialization. The language semantics remain
defined by `specification.md`; this contract defines when a component may claim
to implement or analyze those semantics.

## 1. Formal admission judgment

For source program `P` and configured temporal-memory width `m > 0`, write

```text
A_m(P) = seal(verify(lower(sem(resolve(region_m(types(P)))))))
```

`A_m(P)` is defined only if every phase succeeds:

1. temporal type, causal-flow, linearity, and declared-effect checking;
2. finite temporal-region validation against the exact width `m`;
3. typed HIR name and identity resolution;
4. mandatory structural stack/control semantics;
5. deterministic bytecode lowering or graph object linking;
6. rejection of primitives with no authoritative runtime semantics;
7. structural artifact validation and independent CFG/stack/temporal
   verification; and
8. construction of an immutable `PreparedBytecode` seal.

The judgment is fail-closed. A warning, obligation, or unsupported boundary may
be reported honestly, but no phase may reinterpret failure as success or fall
back to a weaker executor.

For a complete-UNSAT claim, write `U_m(P, q, pi)`. It is defined only when

```text
A_m(P) is defined
and IR(A_m(P)) is complete
and executable_path_bound(A_m(P)) <= configured_gas
and q is byte-for-byte the declarations/assertions passed to Z3
and |q| + |pi| <= 64 MiB
and Z3(q) = UNSAT with proof AST pi
and fresh_Z3(q) = UNSAT with proof AST pi.
```

The stored FNV-1a digest is a deterministic identifier for `q`, not a
cryptographic premise. The retained query and its fresh replay are authoritative.
`U_m` is a two-context backend check; it does not mean that an independent proof
kernel has validated Z3's proof rules.

For a specialized deterministic `FAMILY` instance, write
`D_(x,m,b)(P,C) = d`, where `x` is the exact immutable Boolean input,
`n=|x|`, `m` is the number of temporal cells, `b` is the finite cell width,
`C` is the exact source contract, and `d` is a Boolean decision. It is defined
only when

```text
A_m(P) is defined
and every INPUT observes only the corresponding prefix of retained x
and every one of the 2^(m*b) states is enumerated within the hard ceiling
and every transition terminates, remains in the m-by-b domain, and has no
    external observation or staged effect
and m <= C.CTC_CELLS(n)
and measured transition work <= C.TRANSITION_STEPS(n)
and conservative input/decision/control/stack/dynamic/scope workspace
    <= C.CHRONOLOGY_BITS(n)
and every state in every recurrent class emits exactly the same d in {0,1}.
```

This proves totality, closure, concrete polynomial bounds, effect isolation,
and readout invariance for that exact finite specialization. The retained full
contract plus ordered successor, instruction-count, per-transition workspace,
and readout tables form a VM-independent structural proof object; the checker
recomputes closure, recurrent classes, resource maxima, bounds, and decision.
Bytecode replay binds that table back to execution. Digests are identifiers,
not equality authority.

For the restricted uniform generator, write `G_w(P,C)`, where `w(n)` is a
retained nonnegative single-monomial polynomial. The certificate checker proves
for every integer `n>=1` that `0 < w(n) <= C.CTC_CELLS(n)`, that the same exact
admitted bytecode template has complete acyclic reachable control with only
frozen `INPUT` crossing the finite-IR support boundary, and that specialization
work and descriptor size are canonical linear polynomials in `n`. Specialization binds
the exact `x` and produces the configuration consumed by `D`. This is a real
uniformity proof for the constant-template random-access family, not a proof of
family-wide transition totality or recurrent readout invariance. Nature's
ideal selector remains an explicitly declared external model assumption.

For the sparse projection/routing subclass, write `H_w(P,C)=r`, where `r` is
either a Boolean constant or `x[0]`. Exact main-unit shape establishes a fixed
ordered list of assignments `present[t] := anamnesis[s]` or
`present[t] := k`, plus width-preserving cell-wise `AND`/`OR`/`XOR` and
constant right shifts, with last
write winning, in-domain addresses, fitting constants, and zero at every
unwritten cell. The structural checker derives the
step/stack/control and circuit-size polynomials and proves them for every
`n>=1`. Since `r` is independent of temporal state, every recurrent class reads
`r`; this is an all-input semantic theorem, not finite-sample induction. The
one-cell self-projection is the minimal case.

## 2. Required invariants

| ID | Invariant | Enforced evidence |
|---|---|---|
| ALIGN-001 | Mode invariance: the same `(P,m)` has one admission result in every source-facing execution and proof API. | `src/admission.rs` cross-facade rejection test |
| ALIGN-002 | Representation monotonicity: downstream stages consume only the output of a stronger preceding stage. | Private fields of `PreparedBytecode` and `AdmittedProgram` |
| ALIGN-003 | Memory coherence: region checking, execution, recurrence, and proof lowering use the same `m`. | `AdmissionConfig`; explicit propagation by every facade |
| ALIGN-004 | Semantic singularity: production execution and temporal lowering consume bytecode, never source AST. | Iterative bytecode VM; private test-only source oracle |
| ALIGN-005 | Proof/runtime agreement: SAT evidence is replayed by the VM; complete UNSAT requires complete lowering, sufficient executable gas, and a bounded envelope containing the exact solver query and Z3 proof term. | Global-solver replay, gas-bound, evidence-size, and fresh-Z3 verification gates |
| ALIGN-006 | Artifact closure: object emission proves that the full object set links and seals before returning bytes. | `compile_objects_with_memory` |
| ALIGN-007 | Effect chronology: candidate evaluation freezes observations and stages intents; only one selected chronology may commit. | Temporal transaction and adapter ledgers |
| ALIGN-008 | Build drift resistance: production source facades cannot regain direct HIR/AST compiler calls, and complete-UNSAT cannot silently drop proof production or fresh replay. | `build.rs` architecture gate plus locked all-target tests |
| ALIGN-009 | Family-claim discipline: a declaration is not a proof; a verified finite instance requires exhaustive total/closed/effect-free transition evidence, measured polynomial bounds, and one Boolean decision across all recurrent classes. | `PspaceFamilyVerifier`, retained transition proof table, VM-independent structural checker, VM recheck API, CLI and adversarial corruption/readout/bound/totality tests |
| ALIGN-010 | Uniform-family discipline: exact input is evidence, `CTC_CELLS` is an upper bound rather than an exact width, and uniformity may be claimed only from a retained constant template plus a proved polynomial width rule and canonical linear generator bounds. | `PspaceUniformFamilyGenerator`, exact bytecode certificate, structural checker/recheck, generated-instance bridge, aggregate CLI artifact, polynomial-substitution tests |
| ALIGN-011 | Family-theorem discipline: a family-wide semantic claim requires an exact recognized transition theorem and symbolic resource inequalities; whenever exhaustive evidence is available, symbolic and finite decisions must agree. | `ProjectionFamilyCertificate`, exact bytecode matcher, all-n polynomial checker, CLI disagreement hard failure, underbound/noncanonical/corruption tests |
| ALIGN-012 | Circuit-artifact discipline: a circuit claim requires an explicit topology under a proved polynomial size bound; its checker must regenerate every output wire, and available independent execution evidence must agree on every edge and readout. | `ProjectionCircuit`, canonical wire-bound derivation, allocation-bounded generator, topology checker, complete VM-table differential check, rewiring/input/state tests |

No hash, digest, test count, or successful narrow mode is proof of a broader
invariant by itself. Evidence must exercise the same boundary and configured
domain as the claim.

## 3. Public source-facade matrix

| Facade | Canonical entry | Downstream authority |
|---|---|---|
| Single epoch | `Executor` | `AdmittedProgram -> BytecodeVm` |
| Optimized compatibility | `FastExecutor` | `AdmittedProgram -> BytecodeVm::run_prepared` |
| Orbit/diagnostic/Deutsch/action | `TimeLoop` | admitted bytecode time-loop policies |
| Classical bounded halting | `BoundedHaltingAnalyzer` | admitted bytecode observation |
| Global/all-fixed/property | `GlobalFixedPointSolver` | admitted bytecode IR plus VM replay |
| SMT export | `SmtEncoder` | `GlobalFixedPointSolver::compile` |
| Explicit recurrence | `ProgramTransitionAnalyzer` | admitted bytecode enumeration |
| Finite FAMILY instance | `PspaceFamilyVerifier` | admitted bytecode, complete recurrence, measured workspace/readout certificate |
| Restricted uniform FAMILY | `PspaceUniformFamilyGenerator` | exact linked template, proved width polynomial, linear generator certificate, finite-instance bridge |
| Objects/build | `compile_objects_with_memory` | admitted graph, deterministic link, seal |

Raw bytecode APIs remain available for compilers, linkers, packages, and tests,
but validate and independently verify unsealed artifacts before dispatch.

## 4. Statistical and mathematical result discipline

Point fixed states, deterministic recurrent cycles, Markov stationary
distributions, and quantum fixed densities are distinct result spaces. A mode
may not promote one into another:

- point consistency proves `F(s)=s` for one word-memory state;
- recurrence classifies a declared finite deterministic state graph;
- Deutsch cycle lifting yields a uniform stationary distribution only for the
  reached deterministic cycle, with readout valid only under cycle-wide
  agreement;
- `MARKOV` solves exact rational stationary families; and
- `QCHANNEL` analyzes fixed density operators under its declared finite channel.

Unknown, bounded, vacuous, ambiguous, unsupported, and refuted outcomes remain
separate terminal results. In particular, bounded search exhaustion is never a
proof of nonexistence. A verified exact-input specialization is not promoted to
a family-wide semantic theorem, and the restricted uniformity certificate is
not promoted to an implementation of the ideal selector.

## 5. Change protocol

A new source-facing mode is incomplete until it:

1. calls canonical admission with its exact memory width;
2. consumes the sealed bytecode or a derivation from it;
3. defines effects, gas, state-domain, and result-completeness behavior;
4. adds an adversarial rejection case to the cross-facade matrix;
5. adds its architecture marker to the build gate; and
6. links the contract, relevant checks, and final CI revision in its GitHub issue
   and pull request.
