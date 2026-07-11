# Ourochronos

A programming language where programs read their own future.

Ourochronos implements Deutsch's self-consistency model of closed timelike curves as an executable language. A program is not run once; it defines a transformation F over a 65,536-cell memory, and execution searches for a state S with F(S) = S. Aaronson and Watrous proved that classical computation with access to this fixed-point constraint decides exactly PSPACE, so the language expresses search problems as consistency conditions: verify a witness read from the future, and stabilise the timeline when it is correct.

## The two memories

Every program sees two memory spaces. The *anamnesis* is read-only and holds the state of the world as it will be at the end of the run; reading it with `ORACLE` is how the future speaks. The *present* is read-write and holds the world being built; writing it with `PROPHECY` is how this run answers. Execution repeats the program, feeding each run's present back as the next run's anamnesis, until the two agree.

```ourochronos
# Ask the future for a factor of 15
0 ORACLE
DUP 15 SWAP MOD 0 EQ
IF {
    DUP 0 PROPHECY      # correct: stabilise the timeline
    OUTPUT
} ELSE {
    1 ADD 0 PROPHECY    # wrong: perturb, forcing another epoch
}
```

Running this prints `3`. The program never searches for factors; it verifies the future's claim and rejects timelines where the claim is false. The value exists because no other value permits the loop to close.

Three outcomes are possible for any program. A *consistent* timeline is found and reported. A *paradox* is detected: the classic case is `0 ORACLE NOT 0 PROPHECY`, which negates whatever the future says and therefore oscillates with period 2 (the grandfather paradox). Or the search *times out* without a verdict. The process exit code distinguishes all three, so scripts can tell "no fixed point exists" from "gave up looking".

## Install and run

```bash
git clone https://github.com/sunyaisblank/Ourochronos.git
cd Ourochronos
cargo install --path .

ourochronos examples/hello.ouro
ourochronos examples/paradox.ouro     # exits 2: oscillation, period 2
ourochronos repl
```

| Flag | Purpose |
|------|---------|
| `--diagnostic` | Record the full trajectory; diagnose paradoxes and divergence |
| `--action` | Explore several seeds and select the least-action fixed point |
| `--seeds <n>`, `--seed <n>` | Control the action-mode search |
| `--typecheck` | Static temporal-taint analysis before running |
| `--smt` | Emit SMT-LIB2 constraints instead of running (`z3 program.smt2`) |
| `--fast` | Fast VM for programs with no temporal operations |
| `--max-inst <n>` | Instruction budget per epoch |
| `--effects decline\|unrestricted` | Whether file writes, sockets, RANDOM, and CLOCK are permitted inside the search (default: decline) |
| `--strict`, `--permissive` | Error-handling policy |
| `--audit [file]`, `--audit-json` | Structured logging of the run and its outcome |
| `--provenance-limit <n>` | Saturation limit for causal-dependency tracking |
| `--lsp` | Language server (build with `--features lsp`) |

Exit codes: 0 consistent, 1 error, 2 paradox, 3 epoch exhaustion.

## The Action Principle

Many programs have several fixed points, and the all-zero timeline is usually one of them; a naive search finds it first and reports that nothing happened. Action-guided mode assigns every discovered fixed point a cost that penalises trivial and temporally independent states and rewards causal depth and output, then selects the minimum. The weights are derived from the program's own temporal footprint. `examples/sat.ouro` and `examples/quantum_suicide.ouro` show the difference it makes.

## Documentation and examples

The [specification](docs/specification.md) is the single reference: the abstract machine, the state boundary (what is and is not part of the fixed point), all 99 opcodes, convergence semantics, the effect gate, provenance, the action functional, the SMT fragment, and the practical patterns for writing programs that converge. The `examples/` directory covers the canonical paradoxes, witness search (`sat.ouro`, `primality.ouro`), data structures, strings, procedures, and quotations.

## Licence

MIT. The theoretical foundations rest on Deutsch's CTC self-consistency model (1991) and the Aaronson and Watrous PSPACE characterisation (2008).
