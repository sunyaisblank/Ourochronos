# Ourochronos case studies

These examples exercise three different reasons to use the temporal core. All
commands use small memory explicitly so the claimed finite model is visible.

## 1. Bounded self-consistency synthesis and model checking

`examples/case_studies/mutual_exclusion.ouro` treats a two-bit cell as a
resource-ownership mask. Invalid schedules execute `PARADOX`; both valid
schedules remain fixed states.

```sh
ourochronos examples/case_studies/mutual_exclusion.ouro --global --memory-cells 1
ourochronos examples/case_studies/mutual_exclusion.ouro --all-fixed --memory-cells 1
ourochronos examples/case_studies/mutual_exclusion.ouro --verify --memory-cells 1 \
  --artifact mutual-exclusion.json
ourochronos examples/case_studies/mutual_exclusion.ouro --build mutual-exclusion.ouropkg \
  --embed-global-witness --memory-cells 1
ourochronos run-package mutual-exclusion.ouropkg
```

The first command synthesizes a schedule. The second produces two replayed
witnesses proving state ambiguity. The third proves that every fixed schedule
has exactly one owner. This is the strongest immediate application: bounded
constraint solving plus universal model checking under self-consistency.

## 2. Circular data flow

`examples/case_studies/circular_dataflow.ouro` defines two simultaneous
equations over two-bit words. Ordinary evaluation would require choosing an
order or iterating heuristically; Ourochronos gives the cycle fixed-point
semantics directly.

```sh
ourochronos examples/case_studies/circular_dataflow.ouro --global --memory-cells 2
ourochronos examples/case_studies/circular_dataflow.ouro --all-fixed --memory-cells 2
ourochronos examples/case_studies/circular_dataflow.ouro --verify --memory-cells 2
ourochronos examples/case_studies/circular_dataflow.ouro \
  --build-executable circular-dataflow --memory-cells 2
./circular-dataflow
```

The unique replayed solution is `x=3, y=2`. This is useful for cyclic data-flow
graphs, bidirectional transformations, reactive equations, and compiler IRs
whose constraints are more natural than a hand-chosen schedule.

## 3. Retrocausal simulation and game rules

`examples/case_studies/retrocausal_game.ouro` defines a complete four-state
rule system with a period-two history and two stable equilibria.

```sh
ourochronos examples/case_studies/retrocausal_game.ouro --recurrent \
  --memory-cells 1 --state-bits 2 --dot retrocausal-game.dot
```

The analyzer evaluates all four states, then reports all recurrent classes,
periods, basin sizes, and output agreement. This supports small retrocausal
games and simulations where cycles are meaningful histories rather than mere
runtime failures.

## Soundness boundary

`--global`, `--all-fixed`, and `--verify` prove global facts only when lowering
is complete. A bounded loop can yield a replayed SAT witness, but bounded
UNSAT is `UNKNOWN`. `--recurrent` is exhaustive only for the explicitly sized,
closed domain and refuses a transition that escapes it. These restrictions are
part of the result, not informal caveats.

## Reproducible benchmark harness

The repository includes a dependency-free harness that loads each canonical
module graph, emits and links the real per-source objects, then validates each
expected bytecode-authoritative solver/recurrent result while timing it. Pass
the sample count as the final argument:

```sh
cargo run --release --example bench_case_studies -- 100
```

It reports median and p95 wall-clock time for the self-consistency/property
suite, the circular unique solve, and complete recurrent classification.

Measured on 2026-07-13 in the supplied Linux workspace with the release binary
(`n=100`):

| Case | Median | p95 |
|------|-------:|----:|
| Self-consistency plus two universal properties | 32,343 us | 47,063 us |
| Circular-dataflow solve plus uniqueness | 16,198 us | 17,413 us |
| Complete four-state recurrence | 12 us | 13 us |

`/usr/bin/time -v target/release/examples/bench_case_studies 100` reported
36,276 KiB maximum resident set size for the complete harness and 5.27 s wall
time. These are one reproducible environment record, not a cross-machine
performance promise.

The sparse-memory gate was measured separately with:

```sh
/usr/bin/time -v target/release/ourochronos \
  examples/case_studies/circular_dataflow.ouro \
  --global --memory-cells 1000000
```

It replayed the exact `[0]=3, [1]=2` witness in 25 instructions, took 0.07 s
wall time, and peaked at 83,456 KiB RSS. The million-cell width is therefore an
actually executed bound here, while the witness remains sparse.


## Configurable exact stochastic models

The Rust embedding examples keep arbitrary-precision sparse analysis separate
from the source `MARKOV` declaration's checked `i128` arithmetic:

```sh
cargo run --release --locked --example stochastic_models -- absorbing 4096
cargo run --release --locked --example stochastic_models -- cycle 128 8
cargo run --release --locked --example stochastic_models -- cycle 8 200
cargo run --release --locked --example stochastic_models -- vm 2 3 random
cargo run --release --locked --example stochastic_models -- vm 2 3 tag
# Linux containment for exact arithmetic and allocations:
target/release/examples/stochastic_models supervise 1000 128 vm 2 3 random
```

`absorbing N` has N closed singleton classes, half with accepting readout;
the all-class range is therefore `[0,1]`. `cycle N B` assigns forward rate
`1/(2^B+i)` to state i and retains the remaining self-loop probability.
Its unique stationary weight is independently checked against
`(2^B+i)/(N*2^B+N*(N-1)/2)` and against exact stationary residuals. The
singleton special case coalesces both edges. Changing B exercises denominator
growth without silently switching to floating point.

The explicit VM adapter freezes ordinary INPUT `[42]` and chooses a joint
random tape `[0]` or `[1]` with probabilities `1-p` and p before each epoch.
Two Boolean cells retain a class tag and overwrite the random bit. Every one
of four states and two scenarios must finish under gas 64; the adapter retains
full observations before coalescing equal edges. At p=2/3, the random-bit
readout is accepting in both classes, while the tag readout is ambiguous.
This adapter's admission/totality checks are distinct from normal source
admission, which rejects live INPUT/RANDOM in a temporal scope.

Measured 2026-10-08 on AMD Ryzen 9 PRO 8945HS, x86-64 WSL2 Linux 6.6.87.2,
glibc 2.39, Rust 1.85 release, linked Z3 4.8.12. Five fresh processes per row,
warmed filesystem, no forced disk-cache eviction. GNU time measured each
child's peak RSS; 2 GiB address-space and 15-second parent limits contained
each measurement. Other qualification workers were active. Time covers model
construction, exact analysis and analytic checks, excluding Cargo/startup;
RSS includes process startup and loaded dependencies. Integer bits are the
backend's peak conservative integer-size charge, including intermediates.

| Model | States / edges / classes | Peak bits | Charged operations | Median us (range) | Peak RSS KiB (range) |
|---|---:|---:|---:|---:|---:|
| absorbing 8 | 8 / 8 / 8 | 3 | 148 | 98 (95–106) | 5760–5888 |
| absorbing 64 | 64 / 64 / 64 | 3 | 1156 | 164 (145–189) | 5760–5888 |
| absorbing 512 | 512 / 512 / 512 | 3 | 9220 | 586 (517–695) | 6016–6144 |
| absorbing 4096 | 4096 / 4096 / 4096 | 3 | 73732 | 4002 (3824–4282) | 7936–8064 |
| cycle 8 8 | 8 / 16 / 1 | 24 | 473 | 157 (148–261) | 5760–6016 |
| cycle 32 8 | 32 / 64 / 1 | 29 | 2729 | 352 (324–424) | 5760–5888 |
| cycle 128 8 | 128 / 256 / 1 | 33 | 23273 | 1170 (1087–1211) | 5888–6144 |
| cycle 512 8 | 512 / 1024 / 1 | 37 | 289769 | 4933 (4826–5214) | 6400–6656 |
| cycle 1024 8 | 1024 / 2048 / 1 | 41 | 1103849 | 11538 (11129–11994) | 6932–7060 |
| cycle 8 200 | 8 / 16 / 1 | 408 | 473 | 281 (269–309) | 5760–5888 |
| VM random p=2/3 | 4 / 8 / 2 | 8 | 174 | 200 (182–251) | 6016–6272 |
| VM tag p=2/3 | 4 / 8 / 2 | 8 | 174 | 226 (170–282) | 6144 |

The reducible absorber workload scales linearly in graph work. The sparse
single-class elimination still performs quadratic charged scans; sparse edges
do not imply linear arithmetic. VM extraction enumerates state × scenario,
exponential in declared state bits. These measurements establish no PSPACE
speedup. Reference-host regression budgets for these exact profiles are
100 ms in-process and 32 MiB child peak RSS; runtime resource ceilings remain
explicit and distinct from these measured budgets. Supervision returns
UNKNOWN/exit 3 and withholds partial result output after timeout, signal or
allocator abort; logical arithmetic limits alone cannot guarantee allocator
success. Native Windows supports the portable raw model APIs, while the Linux
supervision profile rejects unsupported platforms before child execution.

## Solver reuse and backend comparison

```sh
cargo run --release --locked --example solver_reuse
# Optional separately installed cvc5 1.3.2 Python environment:
cargo run --release --locked --example solver_reuse -- --cvc5-python /path/to/venv/bin/python
```

Twelve frozen fixtures each submit 32 queries over a whole-main masked domain
of at most eight bits with zero outside the scope. A separate numeric
interpreter exhaustively evaluates all 412 transition rows and checks their
full result, stack, gas and frozen-input behavior against the original VM.
SAT answers replay that original executable. Both fresh and reused Z3 arms use
push/check/pop, avoiding a one-shot versus incremental tactic confound.
The immutable cache key binds exact original and specialized bytecode, IR,
domain, frozen tape, query, complete VM/solver settings and semantic revision.
Only replayed SAT or independently exhaustive finite UNSAT can enter the
bounded process-local study cache; no production solver/cache behavior changes.

On the reference host above, five repeats with proof generation enabled and a
3000 ms query timeout gave aggregate medians 2690.647 ms fresh, 166.200 ms
reused (16.189×), and 0.174 ms warm cache lookup. Cold cache validation/insertion
was 3.784 ms; repeated replay validation 61.994 ms; independent setup 7.914 ms.
The cache retained 384 entries with 119808 charged bytes under caps of 1000
entries/1 MiB. All 3840 Z3 checks agreed, with zero UNKNOWN results. Workload
SHA-256 was `41513f26a070d5a461e6f92f47e2b4eda205fc7cff360b4f8e7164b06ab2a71c`.

cvc5 1.3.2 checked the identical 384 finite queries with zero disagreements or
UNKNOWN results. Its original constant-zero arrays required the explicit
experimental `arrays-exp=true` option; no formula rewrite hid this requirement.
Fresh total time was 910.185 ms, with different solver settings and bindings,
so this is agreement evidence rather than a fair speed ranking. Session reuse
merits future integration only behind the same exact semantic key and resource
limits; the study does not establish benefit for unrelated workloads. See
[Z3 incrementality](https://z3prover.github.io/papers/programmingz3.html) and
[cvc5 1.3.2 array options](https://cvc5.github.io/docs/cvc5-1.3.2/options.html#arrays-theory-module)
for the interfaces used.

## Bounded multi-host recovery investigation

`cargo test --locked --test multi_host_protocol` explores a synthetic durable
coordinator and two managed single-key transactional participants. Prepared
reservations survive crashes. Commit is durable only after both votes; abort
is terminal. Each participant atomically records its exact token, value and
receipt, so duplicate delivery cannot apply twice. Recovery recognizes
KnownApplied only after both exact acknowledgments are recovered,
KnownNotApplied only after durable abort is reconciled at both participants,
and Unresolved for prepared, inaccessible or partially visible outcomes.

The finite prototype reached 1776 states and attempted 46176 transitions under
26 actions, depth cap 16 and state cap 100000, with zero depth cutoffs. This
closes the reachable graph of this fixed abstraction: four states were known
applied, four known not applied, 1768 unresolved; 288 had partial visibility.
Every reached state converged under an independently checked explicit healed
delivery schedule. Partition, lost acknowledgment, duplicates, crash/restore,
identity conflict and stale/forged cached acknowledgment cases are included.
This requires eventual recovery/delivery and cooperating durable participants;
prepared participants cannot decide abort from a timeout. It provides no
instantaneous cross-host read atomicity, Byzantine tolerance, coordinator
failover or general native exactly-once guarantee. The distinction follows
[Gray and Lamport's transaction-commit model](https://arxiv.org/pdf/cs/0408036).

## Higher-dimensional numerical quantum investigation

The separately installed Python prototype admits dimensions 2–4, one to sixteen
Kraus operators and a complete POVM of one to eight effects with a selected
outcome subset. Reproduce using the hash-pinned CPython 3.12/Linux x86-64 wheel
lock embedded in `examples/quantum_fixed_space.py --requirements`, as instructed
in that file, then run `tests/quantum_research.py` in the isolated environment.
It writes `target/quantum-research-report.json`; the Rust runtime has no new
Python dependency or higher-dimensional quantum source syntax.

The reference campaign used CPython 3.12.3, CVXPY 1.9.3, SCS 3.3.1, NumPy 2.5.3
and SciPy 1.18.1 with all sixteen wheel hashes pinned. Seven groups, 80 named
cases and 90 SDP solves passed in 2.321 seconds, under 120-second wall,
90-second CPU, 2 GiB address-space and single-thread BLAS limits. Each direction
has a 20000-iteration/three-second cap. Independent ideal-channel formulas
cover identity, reset, diagonal/block pinching and amplitude damping, including
nonunique fixed classes whose readouts disagree. Full-POVM validation, empty
subsets, adjacent thresholds, degeneracy, exhaustion, nonfinite data,
non-CPTP/bad-POVM inputs and model/config binding are exercised. The maximum
usable analytic discrepancy was 9.10e-14 in this campaign, not a uniform bound.

A consequential conditioning counterexample remains in the report. For ideal
`Phi=(1-epsilon)Id+epsilon Reset(|0><0|)` with epsilon=1e-12, the only exact
fixed density is the ground state and the chosen readout is 0.2. SCS reported
optimal with range approximately `[0.190726658116,0.809273341884]`, fixed
residual at most 1.414e-12, trace error 4.44e-16 and primal-dual gap 3.63e-14.
The conditioning guard labels it `numerical_uncertain` and withholds usable
extrema. Excellent residuals/gaps do not establish exact fixed-space geometry.
All prototype labels remain numerical estimates or uncertainty; no result is
an exact certificate or all-fixed threshold proof. IEEE realizations of ideal
analytic families are themselves only tolerance-admitted CPTP data. See the
[mathematical contract](specification.md#145-bounded-higher-dimensional-quantum-prototype)
for the optimization and trusted boundary.


## Configurable application workflows

```sh
cargo run --release --locked --example application_studies -- exclusion 8 255 --save exclusion-result.txt
cargo run --release --locked --example application_studies -- dataflow 2 2 1,2 1,0 --save dataflow-result.txt
cargo run --release --locked --example application_studies -- game 4 1,0,2,3 --save game-result.txt
```

The applications freeze configuration into source literals and use the normal
source-admission/object/bytecode path. Each reports point existence,
uniqueness and an all-point membership property, exact code/config identity,
selected witness, fetched gas, typed observations, package replay and timings.
Saved human-readable records are bounded and use create-new semantics; an
existing result file is preserved. Full large solver terms remain API evidence;
the record retains their query/proof SHA-256 and byte length for regeneration.

Exclusion admits one to eight contenders and an eligibility bitmask. Fixed
states are exactly the eligible one-hot owners, independently found by a
conventional scan of at most 256 masks. `exclusion 3 0` has no solution;
`exclusion 3 1` is unique; `exclusion 8 255` has all eight eligible owners.
The source uses literal eligibility equalities to avoid an unnecessarily large
bit-trick UNSAT term without changing the ownership predicate.

Dataflow admits one to eight nodes with one to four bits per cell,
`x_i=g_i*x_(i+1)+b_i` modulo the cell width. A scalar gain broadcasts; CSV gives
per-node gains. The baseline `2 2 1,2 1,0` derives `[3,2]`, replays 29 fetched
records and verifies its embedded package witness. The conventional method
tries at most sixteen x0 values, reconstructs every other cell backwards
without inverting gains, then checks closure at x0. This is complete even for
noninvertible gains. The all-point property tests first-cell membership;
the fixed equations then uniquely imply the complete reconstructed vector.
All conventional full vectors still receive original VM/package replay.
`dataflow 1 4 1 1` has no solution; the eight-node four-bit identity ring has
sixteen vectors, compared with a naive 16^8-state enumeration.

The game admits one to sixteen actions and an arbitrary successor CSV. Its
packed literal table fits one 64-bit word; padded encodings reset to zero.
A separate conventional orbit walker checks the complete transition graph,
recurrent classes, basins and readouts against VM recurrence. `game 2 1,0`
has a period-two recurrent class and no point state; that useful orbit result
remains distinct from point nonexistence. A sixteen-action cycle performs
240 fetched graph records. No artificial point witness is packaged for it.

Linux default commands run under 15-second, 512 MiB address-space and 1 MiB
output supervision. Explicit `raw` mode retains only logical caps and is the
portable choice on Windows; the Linux supervision profile refuses other
platforms before starting work. Add `--gas 1` to any study to retain an
independent conventional result beside solver UNKNOWN, with no witness/package
promotion. Stopped processes withhold partial output and do not create a
requested result file. Five focused tests cover thirteen actual workflows,
maximum supported configurations, negative/invalid cases, all 48 small
independently brute-forced rings, and save/replay/configuration boundaries.

## Runtime phase baselines and budgets

```sh
cargo run --release --locked --example bench_case_studies -- --phase vm-wide 20
/usr/bin/time -v target/release/examples/bench_case_studies --phase objects 20
```

Use `--phase NAME N` for one of the names below; N is bounded to 1–1000.
Preparation and one warmup precede timed samples. Each operation includes
contract checks, so these are verified-workflow costs rather than isolated
instruction nanoseconds. On the same reference host, three fresh processes
with twenty warm samples each, 512 MiB address-space and 15-second external
limits, gave these ranges. Peak RSS includes preparation and loaded libraries.
Module/HIR measurement includes warm filesystem reads; no cold-cache claim is
made. Solve/property and recurrence timings remain separately available from
the original three-case benchmark command and the solver study above.

| Phase | Median us across processes | All sample range us | Peak RSS KiB |
|---|---:|---:|---:|
| parse | 3–7 | 3–22 | 6912–7040 |
| modules-hir | 36 | 34–79 | 6912–7040 |
| objects | 94–98 | 86–171 | 7040–7168 |
| link | 8 | 8–265 | 7040–7168 |
| verify | 7–13 | 7–56 | 7040–7168 |
| ir | 3–5 | 3–9 | 7040 |
| smt-text | 4 | 4–31 | 6912–7040 |
| package encode/decode | 23–41 | 22–64 | 6912–7168 |
| vm-dense (4096 writes) | 567–579 | 562–650 | 7168–7296 |
| vm-wide (one write / million cells) | 17–22 | 17–62 | 7168–7296 |
| vm-repeated (prepared wide run) | 17 | 17–29 | 7168–7296 |
| provenance (512 joins, saturation) | 141–148 | 137–283 | 6912–7040 |
| orbit-journal (65 epochs) | 83–86 | 81–170 | 7424–7552 |
| exact (128-state sparse holding cycle) | 442–448 | 431–535 | 7040–7296 |

Before exact empty-page skipping, the wide and repeated verified workflows had
medians 1496–1521 us because sparse projections scanned the full virtual width.
The repair uses a private occupancy count based on complete value equality,
including zero-valued provenance, and never treats a zero hash as an empty
page. Dense, serialization, state identity and gas contracts are preserved.
Reference-host regression budgets are 10 ms per listed phase and 32 MiB process
peak RSS, with 1 second/128 MiB for each baseline application solver workload.
These provide headroom over the measured variation; they are qualification
budgets, not cross-machine latency promises. CI runs finite semantic/resource
checks and the phase workloads under a generous process deadline; a slower
runner requires a measured baseline before tightening performance limits.

The integrated 0.3.0-rc.1 application baseline on the same reference host used
one warmup and twenty samples while candidate builders were also running.
Self-consistency plus two properties had median 425763 us, p95 517767 us and
range 377024–688654 us; circular dataflow had median 129662 us, p95 142549 us
and range 118731–161938 us; complete four-state recurrence had median 21 us,
p95 29 us and range 19–32 us. Combined process peak RSS was 60720 KiB. These
measurements satisfy the stated application budgets and do not claim dedicated
host latency or a speedup over the historical baseline.
