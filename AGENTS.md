# Development instructions

Use the README for setup, supported capabilities and verification commands.
Keep the language and admission contracts in `docs/specification.md` and
`docs/alignment_contract.md` coherent with all source, bytecode, package and
editor consumers. Preserve the explicit boundaries between proof, bounded
search and numerical research.

Use GitHub issues as the durable repair queue. Before changing behavior, identify
the expected contract, concrete discrepancy, affected path and finite acceptance
checks. Group shared causes into coherent issues and check existing issues first.
Do not add repository task plans, audit logs, status reports or handover files.

Work on a branch and preserve unrelated user changes. Order repairs by their
shared dependencies. Parallel agents may handle independent work with explicit
file ownership; serialize changes to shared contracts. Independently review
consequential or uncertain changes without repeating the whole audit.

Use Rust 1.85.0 from `rust-toolchain.toml` for product checks and keep dependency
resolution locked. Run focused checks first, then the relevant integrated gates
in the README. Reuse valid evidence unless changed inputs or a failure justify
repeating it. Do not weaken assertions, required functionality or CI gates.
Changes to reviewed solver consumers require renewed advisory analysis.

Link each pull request to its issues, describe resulting behavior and attach
checks and limitations. Merge only with user authorization, review complete and
CI passing on the final revision. Existing session authorization applies. Update
and close issues only when their acceptance checks pass. Fast-forward local
`main` to `origin/main` after merge and confirm a clean worktree.

Keep scratch work outside tracked source and remove obsolete task-owned files
when finished. Preserve source data, legal notices, executable specifications,
active instructions and retained qualification identities. Never perform broad
cleanup of unrelated processes, images, directories or user work.
