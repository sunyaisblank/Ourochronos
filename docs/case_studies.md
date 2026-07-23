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
